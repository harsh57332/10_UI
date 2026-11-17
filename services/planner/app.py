"""cuRobo arm planner (14 arm joints, torso frame, waist fixed). Envelope on POST /infer.

actions
  fk          {q: [14 rad]}                       -> {joint_names, poses: {left_tcp: {position, quaternion}, ...}}
  plan_pose   {start: [14], goals: {left_tcp: {position, quaternion(wxyz)}, right_tcp: ...}, world: [cuboid]}
  plan_joint  {start: [14], goal: [14], world: [cuboid]}
  ik          {start, goals, world, ...} -> {joint_names, q: [14]}: only the collision-free arm configuration for
               the goals nearest the start (no path; e.g. the bottom of a descent planned bottom-up)
  (both take optional "attached": {"cuboid": cuboid, "q": [14]} = a held box moving with the hands,
   "map": {"pose": [7]} = use the obstacle map, whose frame has this pose in the planning torso frame, and
   "margin": m = every obstacle, map and cuboids, grows by m; a cuboid with its own "margin" grows by the
   smaller of the two)
     -> {joint_names, dt, positions: [[14], ...], duration}
  plan_line   {start, goals, world, attached, map, margin} -> trajectory: both palm frames move along straight
               lines (orientation slerped) to the goals; IK every line.step_m, each nearest the previous, so the
               arm never flips; a step that fails or jumps is retried in halves (line.max_halvings); refused if
               still blocked or a jump. Seeded noise: the same request always gives the same path
  map         {depth: base64 zlib uint16 [h, w], shape: [h, w], depth_scale, intrinsics: [fx, fy, cx, cy],
               camera_pose: [7] (optical frame in the torso frame), q: [14], exclude: [cuboid]}
     -> {voxels, ms}: obstacle map (distance field) from one depth frame; the robot's own arms are removed
        and the space inside the exclude cuboids (the object being handled) is cut out
  map_points  {max_points} -> {points: [[x, y, z], ...]}: occupied surface voxels of the map (for checks)
cuboid: {"name", "dims": [x, y, z], "pose": [x, y, z, qw, qx, qy, qz]}  (torso frame, metres)
Deterministic: the random state is reset for every request (same request, same answer); IK solutions are
ranked by how natural the arms look (planner.natural); free moves (plan_pose / plan_joint) are straight
joint-space paths, collision-checked, else straight up - across - straight down (planner.path), never
cuRobo's sampling planners.
Run: scripts/svc.sh start planner
"""
import logging
import math
import threading
import time

import torch
import uvicorn
from fastapi import FastAPI

from common import config, envelope
from common.fk import mat_to_quat, quat_to_mat
from common.envelope import Request

CFG = config.load()
P = CFG["services"]["planner"]
log = logging.getLogger("planner")
app = FastAPI(title="g1ui-planner")
_planner = None
_attached = None            # json of the currently attached held box, or None
_lock = threading.Lock()
_FAR = {"far": {"dims": [0.01, 0.01, 0.01], "pose": [10.0, 10.0, 10.0, 1, 0, 0, 0]}}
M = CFG["scene_map"]
_mapper = None              # cuRobo Mapper (depth -> TSDF -> distance field)
_map = None                 # distance-field VoxelGrid of the last map (torso frame at capture), or None
_checker = None             # cuRobo collision checker sharing the planner's world and kinematics (held box)
_batch_ik = None            # IK solver for many goal sets in one GPU solve (ik_batch), sharing the planner's world
NAT, PATH = P["natural"], P["path"]


_planners = {}             # hand profile -> (MotionPlanner, RobotCollisionChecker); the request names its profile


def _load():
    """One planner per hand profile whose model is built (config hands), then the depth mapper."""
    global _planner, _checker, _batch_ik
    active = config.hands()
    for name, h in CFG["hands"].items():
        if not config.path(h["planner_yml"]).exists():
            log.warning("hands '%s': no planner model (%s): run scripts.build_planner_robot --hands %s",
                        name, h["planner_yml"], name)
            continue
        config.use_hands(name)
        _planners[name] = _load_planner()
        log.info("hands '%s': planner model %s", name, h["planner_yml"])
    config.use_hands(active if active in _planners else next(iter(_planners)))
    _planner, _checker, _batch_ik = _planners[config.hands()]
    try:                                        # the batched IK's CUDA graphs now, not on the first grasp search (+2 s)
        q = [math.radians(v) for v in NAT["posture_deg"]]
        ik_batch(q, [fk(q)["poses"]], None)
    except Exception as e:
        log.warning("batched IK warm-up: %s", e)
    _load_mapper()


def _select(name):
    """Use the planner of this hand profile (the server sends the robot's detected hands with every request)."""
    global _planner, _checker, _batch_ik, _attached
    if not name or name == config.hands():
        return
    if name not in _planners:
        raise RuntimeError(f"no planner model for hands '{name}' (run scripts.build_planner_robot --hands {name})")
    config.use_hands(name)
    _planner, _checker, _batch_ik = _planners[name]
    _attached = None                            # that planner's attachment state is its own
    log.info("hands '%s' selected", name)


def _load_planner():
    import yaml
    from curobo.motion_planner import MotionPlanner, MotionPlannerCfg
    robot = yaml.safe_load(open(config.path(P["robot_yml"])))
    urdf = config.path(P["urdf"])          # the builder writes absolute paths; re-root them here
    robot["kinematics"]["urdf_path"], robot["kinematics"]["asset_root_path"] = str(urdf), str(urdf.parent)
    cfg = MotionPlannerCfg.create(
        robot=robot,
        scene_model={"cuboid": dict(_FAR)},
        collision_cache={"cuboid": P["max_cuboids"],
                         # the mapper's grid can round up by a voxel per axis: leave room for two
                         "voxel": {"layers": 1, "dims": [e + 2 * M["esdf_voxel_m"] for e in M["extent_m"]],
                                   "voxel_size": M["esdf_voxel_m"]}},
        interpolation_dt=P["interpolation_dt"],
    )
    planner = MotionPlanner(cfg)
    planner.warmup(enable_graph=True, num_warmup_iterations=3)
    from curobo.collision_checking import RobotCollisionChecker, RobotCollisionCheckerCfg
    checker = RobotCollisionChecker(RobotCollisionCheckerCfg.load_from_config(
        robot_config=robot, scene_collision_checker=planner.scene_collision_checker,
        collision_activation_distance=0.0, self_collision_activation_distance=0.0))
    checker.kinematics = planner.kinematics     # the held box attached to the planner is checked too
    # the same IK as the planner's (LBFGS, ik_solutions seeds, same tolerances), for up to ik_batch goal sets at once,
    # on the planner's own collision world: the grasp search asks ~50 candidates per waist turn in one solve instead
    # of one ~12 ms call each (2026-11-17: 707 calls, 36 s live)
    from curobo._src.solver.solver_ik import IKSolver
    from curobo._src.solver.solver_ik_cfg import IKSolverCfg
    # a pre-filter, so more lenient than the exact IK (tolerances x ik_batch_tolerance_x): it must never reject a pose
    # the exact IK solves (strict, it missed the one marginal grasp that planned on 2026-11-17); the exact IK decides
    tx = P["ik_batch_tolerance_x"]
    bik = IKSolver(IKSolverCfg.create(robot=robot, optimizer_configs=["ik/lbfgs_ik.yml"], num_seeds=P["ik_solutions"],
                                      max_batch_size=P["ik_batch"], self_collision_check=True,
                                      position_tolerance=0.005 * tx, orientation_tolerance=0.05 * tx),
                   planner.scene_collision_checker)
    return planner, checker, bik


def _load_mapper():
    global _mapper
    from curobo.perception import Mapper, MapperCfg
    cam = CFG["camera"]
    _mapper = Mapper(MapperCfg(
        voxel_size=M["voxel_m"], esdf_voxel_size=M["esdf_voxel_m"],
        extent_meters_xyz=tuple(M["extent_m"]), extent_esdf_meters_xyz=tuple(M["extent_m"]),
        grid_center=torch.tensor(M["center_m"], dtype=torch.float32), truncation_distance=4 * M["voxel_m"],
        depth_minimum_distance=0.1, depth_maximum_distance=M["depth_max_m"],
        num_cameras=1, image_height=cam["height"], image_width=cam["width"], device="cuda:0"))


def _t(x):
    return torch.as_tensor(x, device="cuda", dtype=torch.float32)


def _state(q):
    from curobo.types import JointState
    return JointState.from_position(_t(q).view(1, -1), joint_names=_planner.joint_names)


def _T(pose7):
    """[x, y, z, qw, qx, qy, qz] -> 4x4."""
    import numpy as np
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = quat_to_mat([float(v) for v in pose7[3:7]]), [float(v) for v in pose7[:3]]
    return T


def _world(cuboids, use_map=None, margin=0.0):
    """Load the cuboids (grown by margin) and, if asked, the obstacle map placed at use_map["pose"] (its
    distance field shifted by margin, which grows every mapped surface by that much)."""
    from curobo.scene import Cuboid, Scene, VoxelGrid
    m = float(margin or 0.0)
    cub = [Cuboid(name=k, dims=v["dims"], pose=v["pose"]) for k, v in _FAR.items()]
    for c in cuboids or []:
        mc = min(m, float(c["margin"])) if c.get("margin") is not None else m   # e.g. the support: hands come close
        cub.append(Cuboid(name=c["name"], dims=[float(d) + 2 * mc for d in c["dims"]], pose=list(map(float, c["pose"]))))
    vox = []
    if use_map and _map is not None:
        T = _T(use_map["pose"]) @ _T(_map.pose)   # grid pose in the planning torso frame
        vox.append(VoxelGrid(name="map", pose=[*map(float, T[:3, 3]), *map(float, mat_to_quat(T[:3, :3]))],
                             dims=list(_map.dims), voxel_size=_map.voxel_size,
                             feature_tensor=_map.feature_tensor - m, feature_dtype=_map.feature_tensor.dtype))
    _planner.update_world(Scene(cuboid=cub, voxel=vox))


def build_map(p):
    """Obstacle map from one depth frame or several ("frames": the task scene's views), fused into one grid. Each
    frame: its depth, its camera pose in its own torso frame (robot filter and cut-outs), "integrate_pose" = its
    camera pose in the map frame (the first view's torso), its joints and cut-out cuboids (own torso frame)."""
    global _map
    import base64
    import zlib
    import numpy as np
    from curobo.types import CameraObservation, Pose
    frames = p.get("frames") or [p]
    _mapper.reset()
    stats = {"depth": 0, "robot": 0, "used": 0}
    for fr in frames:
        h, w = fr["shape"]
        raw = np.frombuffer(zlib.decompress(base64.b64decode(fr["depth"])), dtype=np.uint16).reshape(h, w)
        depth = _t(raw.astype(np.float32) * float(fr["depth_scale"]))
        fx, fy, cx, cy = map(float, fr["intrinsics"])
        K = _t([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]])
        pose = Pose.from_list(list(map(float, fr["camera_pose"])))
        n_raw = int((depth > 0).sum())
        vs, us = torch.meshgrid(torch.arange(h, device="cuda"), torch.arange(w, device="cuda"), indexing="ij")
        cam = torch.stack([(us - cx) * depth / fx, (vs - cy) * depth / fy, depth], dim=-1).view(-1, 3)
        R = pose.get_rotation_matrix()[0]
        pts = cam @ R.T + pose.position[0]                                  # its own torso frame
        keep = depth.view(-1) > 0
        # the robot's own arms: points within robot_filter_m of its collision spheres (same model as planning)
        sph = _planner.kinematics.compute_kinematics(_state(fr["q"])).robot_spheres.view(-1, 4)
        sph = sph[sph[:, 3] > 0]
        for chunk in torch.split(torch.arange(h * w, device="cuda"), 65536):
            d = torch.cdist(pts[chunk], sph[:, :3]) - sph[:, 3]
            keep[chunk] &= d.min(dim=1).values > M["robot_filter_m"]
        n_robot = n_raw - int(keep.sum())
        # cut out the handled object(s): points inside a (padded) exclude cuboid get no depth
        for c in fr.get("exclude") or []:
            cp = Pose.from_list(list(map(float, c["pose"])))
            local = (pts - cp.position[0]) @ cp.get_rotation_matrix()[0]
            half = _t(c["dims"]) / 2 + M["exclude_pad_m"]
            keep &= ~(local.abs() <= half).all(dim=-1)
        depth = (depth.view(-1) * keep).view(h, w)
        stats["depth"] += n_raw
        stats["robot"] += n_robot
        stats["used"] += int((depth > 0).sum())
        ipose = Pose.from_list(list(map(float, fr.get("integrate_pose") or fr["camera_pose"])))
        _mapper.integrate(CameraObservation(
            depth_image=depth.unsqueeze(0), rgb_image=torch.zeros((1, h, w, 3), dtype=torch.uint8, device="cuda"),
            intrinsics=K.unsqueeze(0), pose=ipose))
    _map = _mapper.compute_esdf()
    f = _map.feature_tensor
    return {"pixels": stats, "frames": len(frames),
            "voxels_inside": int((f <= 0).sum()), "esdf_range": [float(f.min()), float(f.max())],
            "pose": [round(float(v), 3) for v in _map.pose], "dims": list(_map.dims), "voxel": _map.voxel_size}


def map_query(p):
    """Distance-field values (m, negative inside obstacles) of the map at points given in its frame."""
    pts = _t(p["points"]).view(-1, 3)
    f = _map.feature_tensor
    n = _t(list(f.shape))
    idx = torch.round((pts - _t(list(_map.pose[:3]))) / _map.voxel_size + (n - 1) / 2).long()
    idx = torch.minimum(torch.clamp(idx, min=0), (n - 1).long())
    return {"sdf": f[idx[:, 0], idx[:, 1], idx[:, 2]].float().cpu().tolist(), "shape": list(f.shape)}


def map_points(p):
    v = _mapper.extract_occupied_voxels(surface_only=True, max_points=int(p.get("max_points", 20000)))
    return {"points": v.centers.view(-1, 3).cpu().tolist()}


def _attach(attached):
    """Keep cuRobo's attached object in sync with the request: {"cuboid": {...torso frame}, "q": [14]} or None."""
    global _attached
    key = None if attached is None else repr(attached)
    if key == _attached:
        return
    am = _planner.attachment_manager
    if attached is None:
        am.detach("attached_object")
    else:
        from curobo.scene import Cuboid
        c = attached["cuboid"]
        am.attach(_state(attached["q"]), [Cuboid(name="held", dims=c["dims"], pose=c["pose"])],
                  link_name="attached_object", num_spheres=CFG["planner_model"]["held_object_spheres"])
    _attached = key


def _trajectory(result):
    if result is None or not bool(result.success.any()):
        raise RuntimeError("no collision-free plan found")
    plan = result.get_interpolated_plan()
    pos = plan.position.reshape(-1, plan.position.shape[-1]).cpu().tolist()
    dt = float(P["interpolation_dt"])
    return {"joint_names": _planner.joint_names, "dt": dt, "positions": pos, "duration": round(len(pos) * dt, 3)}


def fk(q):
    st = _planner.kinematics.compute_kinematics(_state(q))
    out = {}
    for name in _planner.tool_frames:
        p = st.tool_poses.get_link_pose(name)
        out[name] = {"position": p.position[0].cpu().tolist(), "quaternion": p.quaternion[0].cpu().tolist()}
    return {"joint_names": _planner.joint_names, "poses": out}


def _ik(start, goals, world, use_map=None, margin=0.0):
    """Collision-free IK solutions for the palm goals, nearest the start first."""
    from curobo.types import GoalToolPose
    frames = _planner.tool_frames
    missing = [f for f in frames if f not in goals]
    if missing:   # keep an unspecified hand where it is
        cur = fk(start)["poses"]
        goals = {**goals, **{f: cur[f] for f in missing}}
    pos = _t([[goals[f]["position"]] for f in frames]).view(1, 1, len(frames), 1, 3)
    quat = _t([[goals[f]["quaternion"]] for f in frames]).view(1, 1, len(frames), 1, 4)
    _world(world, use_map, margin)
    goal = GoalToolPose(tool_frames=frames, position=pos, quaternion=quat)
    n = P["ik_solutions"]
    ik = _planner.ik_solver.solve_pose(goal, current_state=_state(start), seed_config=_seeds(start, n),
                                       return_seeds=n)
    sol = ik.solution.reshape(-1, len(start))[ik.success.reshape(-1)]
    if len(sol) == 0:
        raise RuntimeError("no collision-free plan found (no IK solution)")
    wrist = [i for i, j in enumerate(_planner.joint_names) if "wrist" in j]
    sol = sol[(sol[:, wrist].abs() <= math.radians(NAT["wrist_max_deg"])).all(-1)]
    if len(sol) == 0:
        raise RuntimeError("no natural IK solution (every one twists a wrist past "
                           f"{NAT['wrist_max_deg']} deg)")
    return sol[_natural_cost(sol, _t(start)).argsort()]


def _seeds(start, n):
    """IK seeds, the same for the same request: the start, the natural posture, then fixed variations of both."""
    q0, nat = _t(start), _t([math.radians(v) for v in NAT["posture_deg"]])
    gen = torch.Generator(device="cuda").manual_seed(P["rng_seed"])
    base = torch.stack([q0, nat] * (n // 2 + 1))[:n]
    noise = torch.randn(base.shape, generator=gen, device="cuda") * math.radians(NAT["seed_spread_deg"])
    noise[:2] = 0.0
    return (base + noise).view(1, n, -1)


def _natural_cost(sol, start):
    """Lower = more natural: near the natural posture (elbows out to the sides, wrists straight), the shoulders
    not rolled in toward the body (use the free space sideways), and little change from the start."""
    nat = _t([math.radians(v) for v in NAT["posture_deg"]])
    c = (((sol - nat) ** 2) * _t(NAT["weights"])).sum(-1) + NAT["continuity"] * ((sol - start) ** 2).sum(-1)
    names = _planner.joint_names
    free = math.radians(NAT["adduction_free_deg"])
    for side, sign in (("left", 1.0), ("right", -1.0)):   # roll toward the body: left negative, right positive
        r = sol[:, names.index(f"{side}_shoulder_roll_joint")] * sign
        c = c + NAT["adduction_weight"] * torch.clamp(free - r, min=0.0) ** 2
    return c


def ik_batch(start, goal_sets, world, use_map=None, margin=0.0):
    """For each palm-goal set: is there a collision-free, natural IK solution (the test _ik makes, no held object)?
    Solved ik_batch sets at a time in one GPU call; the world and seeds are those of a single ik request."""
    from curobo.types import GoalToolPose, JointState
    frames = _planner.tool_frames
    _world(world, use_map, margin)
    cur = fk(start)["poses"]
    n, B = P["ik_solutions"], P["ik_batch"]
    wrist = [i for i, j in enumerate(_planner.joint_names) if "wrist" in j]
    seeds = _seeds(start, n)
    out = []
    for i in range(0, len(goal_sets), B):
        chunk = [{**{f: cur[f] for f in frames}, **g} for g in goal_sets[i:i + B]]
        real = len(chunk)
        chunk += [chunk[-1]] * (B - real)       # always the full batch (cuRobo's own padding missed the seeds)
        k = len(chunk)
        pos = _t([[[g[f]["position"]] for f in frames] for g in chunk]).view(k, 1, len(frames), 1, 3)
        quat = _t([[[g[f]["quaternion"]] for f in frames] for g in chunk]).view(k, 1, len(frames), 1, 4)
        state = JointState.from_position(_t(start).view(1, -1).repeat(k, 1), joint_names=_planner.joint_names)
        _batch_ik.reset_seed()
        ik = _batch_ik.solve_pose(GoalToolPose(tool_frames=frames, position=pos, quaternion=quat),
                                  current_state=state, seed_config=seeds.repeat(k, 1, 1), return_seeds=n)
        sol = ik.solution.reshape(k, -1, len(start))
        ok = ik.success.reshape(k, -1) & (sol[..., wrist].abs() <= math.radians(NAT["wrist_max_deg"])).all(-1)
        out += ok.any(-1).tolist()[:real]
    return {"ok": out}


def solve_ik(start, goals, world, use_map=None, margin=0.0, debug=False):
    """Only the arm configuration for the goals (the most natural one), no path to it."""
    sol = _ik(start, goals, world, use_map, margin)
    out = {"joint_names": _planner.joint_names, "q": sol[0].tolist()}
    if debug:                                   # every solution, most natural first, with its cost
        out["all"] = [[round(math.degrees(v), 1) for v in q] for q in sol.tolist()]
        out["cost"] = [round(float(c), 3) for c in _natural_cost(sol, _t(start))]
    return out


def plan_pose(start, goals, world, use_map=None, margin=0.0):
    # the IK solution nearest the start, then a joint-space plan to it: cuRobo's own pose planning takes
    # any collision-free solution, sometimes with the arms wound over the shoulders
    sol = _ik(start, goals, world, use_map, margin)
    err = None
    for i in range(min(len(sol), P["ik_tries"])):
        try:
            return _path(start, sol[i].tolist(), world)
        except RuntimeError as e:
            err = e
    raise RuntimeError(f"no collision-free plan found ({err})")


def _slerp(a, b, s):
    import numpy as np
    a, b = np.asarray(a, float), np.asarray(b, float)
    if np.dot(a, b) < 0:
        b = -b
    d = min(1.0, float(np.dot(a, b)))
    th = np.arccos(d)
    if th < 1e-6:
        return a
    return (np.sin((1 - s) * th) * a + np.sin(s * th) * b) / np.sin(th)


def plan_line(start, goals, world, use_map=None, margin=0.0):
    """Straight-line move of both palm frames (see the module docstring)."""
    _world(world, use_map, margin)
    return _timed_line(_line_path(start, goals))


def _line_path(start, goals):
    """The IK waypoints of a straight-line move (the world already loaded)."""
    import numpy as np
    from curobo.types import GoalToolPose
    L = P["line"]
    frames = _planner.tool_frames
    cur = fk(start)["poses"]
    goals = {**{f: cur[f] for f in frames if f not in goals}, **goals}
    p0 = {f: np.array(cur[f]["position"]) for f in frames}
    p1 = {f: np.array(goals[f]["position"]) for f in frames}
    dist = max(float(np.linalg.norm(p1[f] - p0[f])) for f in frames)
    ang = max(2 * math.acos(min(1.0, abs(float(np.dot(cur[f]["quaternion"], goals[f]["quaternion"]))))) for f in frames)
    n = max(1, math.ceil(max(dist / L["step_m"], math.degrees(ang) / L["step_deg"])))
    gen = torch.Generator(device="cuda").manual_seed(L["noise_seed"])   # same request -> same path

    def solve(q, s):
        """IK at fraction s of the line, seeded at the angles q: (solution nearest q, its largest joint change)
        or (None, None) if no solution."""
        pos = _t([[list(p0[f] + (p1[f] - p0[f]) * s)] for f in frames]).view(1, 1, len(frames), 1, 3)
        quat = _t([[list(_slerp(cur[f]["quaternion"], goals[f]["quaternion"], s))] for f in frames]).view(1, 1, len(frames), 1, 4)
        # seeds at the current angles (plus small variations): the solver then finds the nearby solution
        # instead of drifting to its default arm pose or another branch
        seeds = q.view(1, 1, -1).repeat(1, L["seeds"], 1)
        seeds[0, 1:] += torch.randn(seeds[0, 1:].shape, generator=gen, device="cuda") * math.radians(L["seed_noise_deg"])
        ik = _planner.ik_solver.solve_pose(GoalToolPose(tool_frames=frames, position=pos, quaternion=quat),
                                           current_state=_state(q.tolist()), seed_config=seeds,
                                           return_seeds=L["seeds"])
        sol = ik.solution.reshape(-1, len(start))[ik.success.reshape(-1)]
        if len(sol) == 0:
            return None, None
        jump = (sol - q).abs().max(dim=-1).values
        k = int(jump.argmin())
        return sol[k], math.degrees(float(jump[k]))

    q = _t(start)
    path = [q]
    s0, ds_min = 0.0, 1.0 / n / 2 ** L["max_halvings"]
    while s0 < 1.0 - 1e-9:
        # a step that fails or jumps is retried in halves: an arm that has to swivel along the way (e.g. the
        # elbow moving out as the hands close in) then moves continuously, while a real flip stays a jump
        ds = min(1.0 / n, 1.0 - s0)
        while True:
            sol, jump = solve(q, s0 + ds)
            if sol is not None and jump <= L["max_joint_step_deg"]:
                break
            if ds / 2 < ds_min - 1e-12:
                s = s0 + ds
                if sol is None:
                    raise RuntimeError(f"no straight path: blocked or out of reach at {100 * s:.0f}% of the line")
                d = torch.rad2deg(sol - q)
                worst = sorted(zip(d.abs().tolist(), _planner.joint_names, d.tolist()), reverse=True)[:3]
                raise RuntimeError(f"no straight path: the arm would flip at {100 * s:.0f}% of the line ("
                                   + ", ".join(f"{n.replace('_joint', '')} {v:+.0f}" for _, n, v in worst) + " deg)")
            ds /= 2
        s0 += ds
        q = sol
        path.append(q)
    return path


def _timed_line(path):
    """Constant joint speed along the waypoints (contact moves)."""
    L = P["line"]
    dt = float(P["interpolation_dt"])
    pts = [path[0]]
    for a, b in zip(path[:-1], path[1:]):
        k = max(1, math.ceil(float((b - a).abs().max()) / (L["joint_speed_rps"] * dt)))
        pts += [a + (b - a) * (j / k) for j in range(1, k + 1)]
    pos = torch.stack(pts).cpu().tolist()
    return {"joint_names": _planner.joint_names, "dt": dt, "positions": pos, "duration": round(len(pos) * dt, 3)}


def plan_joint(start, goal, world, use_map=None, margin=0.0):
    _world(world, use_map, margin)
    return _path(start, goal, world)


def _valid(qs):
    """Collision-free (world as loaded, held box, self) and within joint limits, per configuration [n, dof]."""
    return _checker.validate(qs.view(1, -1, qs.shape[-1]))[0]


def _segment(a, b):
    """The straight joint-space segment a -> b sampled every path.step_deg, or None if any sample collides."""
    n = max(2, math.ceil(math.degrees(float((b - a).abs().max())) / PATH["step_deg"]) + 1)
    qs = a + (b - a) * torch.linspace(0, 1, n, device="cuda").view(-1, 1)
    return qs if bool(_valid(qs).all()) else None


def _smooth(a, b):
    """a -> b timed with a minimum-jerk profile (starts and stops gently)."""
    dt = float(P["interpolation_dt"])
    big = math.degrees(float((b - a).abs().max()))
    T = max(PATH["min_duration_s"], 1.875 * big / PATH["joint_speed_dps"])
    k = max(2, math.ceil(T / dt))
    t = torch.linspace(0, 1, k + 1, device="cuda")[1:].view(-1, 1)
    return [a + (b - a) * x for x in (10 * t ** 3 - 15 * t ** 4 + 6 * t ** 5)]


def _traj(pts):
    dt = float(P["interpolation_dt"])
    pos = torch.stack(pts).cpu().tolist()
    return {"joint_names": _planner.joint_names, "dt": dt, "positions": pos, "duration": round(len(pos) * dt, 3)}


def _path(start, goal, world):
    """Deterministic free move: the straight joint-space path if it is clear, else straight up (both palms) to
    path.over_clear_m above the highest obstacle, across in joint space, and straight down to the goal."""
    a, b = _t(start), _t(goal)
    if not bool(_valid(a.view(1, -1))[0]):
        raise RuntimeError("the start pose is in collision (at this margin)")
    if not bool(_valid(b.view(1, -1))[0]):
        raise RuntimeError("the goal pose is in collision (at this margin)")
    if _segment(a, b) is not None:
        return _traj([a] + _smooth(a, b))
    tops = [float(c["pose"][2]) + float(c["dims"][2]) / 2 for c in world or []]
    top = max(tops) if tops else 0.0
    cur_a, cur_b = fk(start)["poses"], fk(goal)["poses"]
    errs = []
    rises = [[a]]                               # ways out of the start: as it is, or risen straight up
    for clear in PATH["over_clear_m"]:
        z = top + clear
        try:
            up_a = _line_path(start, {f: {"position": [*cur_a[f]["position"][:2], max(z, cur_a[f]["position"][2])],
                                          "quaternion": cur_a[f]["quaternion"]} for f in cur_a})
            up_b = _line_path(goal, {f: {"position": [*cur_b[f]["position"][:2], max(z, cur_b[f]["position"][2])],
                                         "quaternion": cur_b[f]["quaternion"]} for f in cur_b})
        except RuntimeError as e:
            errs.append(f"{100 * clear:.0f} cm over: {e}")
            continue
        rises.append(up_a)
        if _segment(up_a[-1], up_b[-1]) is None:
            errs.append(f"{100 * clear:.0f} cm over: blocked across")
            continue
        return _traj(_timed_pts(up_a) + _smooth(up_a[-1], up_b[-1]) + _timed_pts(up_b[::-1])[1:] + [b])
    for k, tr in enumerate(PATH["transit_deg"]):  # through a fixed transit posture (from the start or risen)
        via = _t([math.radians(v) for v in tr])
        if not bool(_valid(via.view(1, -1))[0]) or _segment(via, b) is None:
            errs.append(f"via transit {k}: blocked")
            continue
        for up in rises:
            if _segment(up[-1], via) is not None:
                return _traj((_timed_pts(up) if len(up) > 1 else [a]) + _smooth(up[-1], via) + _smooth(via, b))
        errs.append(f"via transit {k}: blocked from the start")
    try:                                        # last: cuRobo's planner, deterministic too (random state reset per
        return _trajectory(_planner.plan_cspace(_state(goal), _state(start)))   # request: same input, same path)
    except RuntimeError as e:
        errs.append(f"cuRobo: {e}")
    raise RuntimeError("no collision-free plan found (straight blocked; " + "; ".join(errs) + ")")


def _timed_pts(path):
    return [_t(p) for p in _timed_line(path)["positions"]]


@app.post("/infer")
def infer(req: Request):
    t0 = time.time()
    p = req.params
    try:
        with _lock:
            _select(p.get("hands"))
            if req.action == "fk":
                return envelope.done(req.id, fk(p["q"]), t0)
            if req.action in ("plan_pose", "plan_joint", "plan_line", "ik", "ik_batch"):
                torch.manual_seed(P["rng_seed"])        # the same request always gets the same plan
                _planner.reset_seed()
                _attach(p.get("attached"))
            if req.action == "plan_pose":
                return envelope.done(req.id, plan_pose(p["start"], p["goals"], p.get("world"), p.get("map"),
                                                       p.get("margin", 0.0)), t0)
            if req.action == "ik_batch":
                return envelope.done(req.id, ik_batch(p["start"], p["goal_sets"], p.get("world"), p.get("map"),
                                                      p.get("margin", 0.0)), t0)
            if req.action == "ik":
                return envelope.done(req.id, solve_ik(p["start"], p["goals"], p.get("world"), p.get("map"),
                                                      p.get("margin", 0.0), p.get("debug", False)), t0)
            if req.action == "plan_joint":
                return envelope.done(req.id, plan_joint(p["start"], p["goal"], p.get("world"), p.get("map"),
                                                        p.get("margin", 0.0)), t0)
            if req.action == "plan_line":
                return envelope.done(req.id, plan_line(p["start"], p["goals"], p.get("world"), p.get("map"),
                                                       p.get("margin", 0.0)), t0)
            if req.action == "map":
                return envelope.done(req.id, build_map(p), t0)
            if req.action == "map_query":
                return envelope.done(req.id, map_query(p), t0)
            if req.action == "map_points":
                return envelope.done(req.id, map_points(p), t0)
        return envelope.failed(req.id, f"unknown action '{req.action}'", t0, status="rejected")
    except (KeyError, ValueError) as e:
        return envelope.failed(req.id, f"bad params: {e}", t0, status="rejected")
    except RuntimeError as e:
        return envelope.failed(req.id, str(e), t0)


@app.get("/health")
def health():
    return {"service": "planner", "ready": _planner is not None,
            "joint_names": _planner.joint_names if _planner else None,
            "tool_frames": _planner.tool_frames if _planner else None}


def main():
    logging.basicConfig(level=CFG["logging"]["level"], format="%(asctime)s %(name)s %(levelname)s %(message)s")
    _load()
    log.info("cuRobo planner ready on :%s (joints %s)", P["port"], _planner.joint_names)
    uvicorn.run(app, host=P["host"], port=P["port"], log_level="warning")


if __name__ == "__main__":
    main()
