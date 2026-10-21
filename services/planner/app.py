"""cuRobo arm planner (14 arm joints, torso frame, waist fixed). Envelope on POST /infer.

actions
  fk          {q: [14 rad]}                       -> {joint_names, poses: {left_tcp: {position, quaternion}, ...}}
  plan_pose   {start: [14], goals: {left_tcp: {position, quaternion(wxyz)}, right_tcp: ...}, world: [cuboid]}
  plan_joint  {start: [14], goal: [14], world: [cuboid]}
  (both take optional "attached": {"cuboid": cuboid, "q": [14]} = a held box moving with the hands,
   "map": {"pose": [7]} = use the obstacle map, whose frame has this pose in the planning torso frame, and
   "margin": m = every obstacle, map and cuboids, grows by m)
     -> {joint_names, dt, positions: [[14], ...], duration}
  plan_line   {start, goals, world, attached, map, margin} -> trajectory: both palm frames move along straight
               lines (orientation slerped) to the goals; IK every line.step_m, each nearest the previous, so the
               arm never flips; refused if blocked or if a step needs a jump
  map         {depth: base64 zlib uint16 [h, w], shape: [h, w], depth_scale, intrinsics: [fx, fy, cx, cy],
               camera_pose: [7] (optical frame in the torso frame), q: [14], exclude: [cuboid]}
     -> {voxels, ms}: obstacle map (distance field) from one depth frame; the robot's own arms are removed
        and the space inside the exclude cuboids (the object being handled) is cut out
  map_points  {max_points} -> {points: [[x, y, z], ...]}: occupied surface voxels of the map (for checks)
cuboid: {"name", "dims": [x, y, z], "pose": [x, y, z, qw, qx, qy, qz]}  (torso frame, metres)
Run: scripts/svc.sh start planner
"""
import logging
import threading
import time

import torch
import uvicorn
from fastapi import FastAPI

from common import config, envelope
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


def _load():
    global _planner, _mapper
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
    _planner = MotionPlanner(cfg)
    _planner.warmup(enable_graph=True, num_warmup_iterations=3)
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


def _world(cuboids, use_map=None, margin=0.0):
    """Load the cuboids (grown by margin) and, if asked, the obstacle map placed at use_map["pose"] (its
    distance field shifted by margin, which grows every mapped surface by that much)."""
    from curobo.scene import Cuboid, Scene, VoxelGrid
    from curobo.types import Pose
    m = float(margin or 0.0)
    cub = [Cuboid(name=k, dims=v["dims"], pose=v["pose"]) for k, v in _FAR.items()]
    for c in cuboids or []:
        cub.append(Cuboid(name=c["name"], dims=[float(d) + 2 * m for d in c["dims"]], pose=list(map(float, c["pose"]))))
    vox = []
    if use_map and _map is not None:
        pose = Pose.from_list(list(map(float, use_map["pose"]))).multiply(Pose.from_list(list(_map.pose)))
        vox.append(VoxelGrid(name="map", pose=pose.tolist(), dims=list(_map.dims), voxel_size=_map.voxel_size,
                             feature_tensor=_map.feature_tensor - m, feature_dtype=_map.feature_tensor.dtype))
    _planner.update_world(Scene(cuboid=cub, voxel=vox))


def build_map(p):
    """Obstacle map from one depth frame (see the module docstring)."""
    global _map
    import base64
    import zlib
    import numpy as np
    from curobo.types import CameraObservation, Pose
    h, w = p["shape"]
    raw = np.frombuffer(zlib.decompress(base64.b64decode(p["depth"])), dtype=np.uint16).reshape(h, w)
    depth = _t(raw.astype(np.float32) * float(p["depth_scale"]))
    fx, fy, cx, cy = map(float, p["intrinsics"])
    K = _t([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]])
    pose = Pose.from_list(list(map(float, p["camera_pose"])))
    n_raw = int((depth > 0).sum())
    vs, us = torch.meshgrid(torch.arange(h, device="cuda"), torch.arange(w, device="cuda"), indexing="ij")
    cam = torch.stack([(us - cx) * depth / fx, (vs - cy) * depth / fy, depth], dim=-1).view(-1, 3)
    R = pose.get_rotation_matrix()[0]
    pts = cam @ R.T + pose.position[0]                                  # torso frame
    keep = depth.view(-1) > 0
    # the robot's own arms: points within robot_filter_m of its collision spheres (same model as planning)
    sph = _planner.kinematics.compute_kinematics(_state(p["q"])).robot_spheres.view(-1, 4)
    sph = sph[sph[:, 3] > 0]
    for chunk in torch.split(torch.arange(h * w, device="cuda"), 65536):
        d = torch.cdist(pts[chunk], sph[:, :3]) - sph[:, 3]
        keep[chunk] &= d.min(dim=1).values > M["robot_filter_m"]
    n_robot = n_raw - int(keep.sum())
    # cut out the handled object(s): points inside a (padded) exclude cuboid get no depth
    for c in p.get("exclude") or []:
        cp = Pose.from_list(list(map(float, c["pose"])))
        local = (pts - cp.position[0]) @ cp.get_rotation_matrix()[0]
        half = _t(c["dims"]) / 2 + M["exclude_pad_m"]
        keep &= ~(local.abs() <= half).all(dim=-1)
    depth = (depth.view(-1) * keep).view(h, w)
    n_used = int((depth > 0).sum())
    _mapper.reset()
    _mapper.integrate(CameraObservation(
        depth_image=depth.unsqueeze(0), rgb_image=torch.zeros((1, h, w, 3), dtype=torch.uint8, device="cuda"),
        intrinsics=K.unsqueeze(0), pose=pose))
    _map = _mapper.compute_esdf()
    f = _map.feature_tensor
    return {"pixels": {"depth": n_raw, "robot": n_robot, "used": n_used},
            "voxels_inside": int((f <= 0).sum()), "esdf_range": [float(f.min()), float(f.max())],
            "pose": [round(float(v), 3) for v in _map.pose], "dims": list(_map.dims), "voxel": _map.voxel_size}


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


def plan_pose(start, goals, world, use_map=None, margin=0.0):
    from curobo.types import GoalToolPose
    frames = _planner.tool_frames
    missing = [f for f in frames if f not in goals]
    if missing:   # keep an unspecified hand where it is
        cur = fk(start)["poses"]
        goals = {**goals, **{f: cur[f] for f in missing}}
    pos = _t([[goals[f]["position"]] for f in frames]).view(1, 1, len(frames), 1, 3)
    quat = _t([[goals[f]["quaternion"]] for f in frames]).view(1, 1, len(frames), 1, 4)
    _world(world, use_map, margin)
    # the IK solution nearest the start, then a joint-space plan to it: cuRobo's own pose planning takes
    # any collision-free solution, sometimes with the arms wound over the shoulders
    goal = GoalToolPose(tool_frames=frames, position=pos, quaternion=quat)
    ik = _planner.ik_solver.solve_pose(goal, current_state=_state(start), return_seeds=P["ik_solutions"])
    sol = ik.solution.reshape(-1, len(start))[ik.success.reshape(-1)]
    if len(sol) == 0:
        raise RuntimeError("no collision-free plan found (no IK solution)")
    for i in (sol - _t(start)).abs().sum(-1).argsort()[:P["ik_tries"]]:
        try:
            return _trajectory(_planner.plan_cspace(_state(sol[i].tolist()), _state(start)))
        except RuntimeError:
            continue
    raise RuntimeError("no collision-free plan found")


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
    import math
    import numpy as np
    from curobo.types import GoalToolPose
    L = P["line"]
    _world(world, use_map, margin)
    frames = _planner.tool_frames
    cur = fk(start)["poses"]
    goals = {**{f: cur[f] for f in frames if f not in goals}, **goals}
    p0 = {f: np.array(cur[f]["position"]) for f in frames}
    p1 = {f: np.array(goals[f]["position"]) for f in frames}
    dist = max(float(np.linalg.norm(p1[f] - p0[f])) for f in frames)
    ang = max(2 * math.acos(min(1.0, abs(float(np.dot(cur[f]["quaternion"], goals[f]["quaternion"]))))) for f in frames)
    n = max(1, math.ceil(max(dist / L["step_m"], math.degrees(ang) / L["step_deg"])))
    q = _t(start)
    path = [q]
    for i in range(1, n + 1):
        s = i / n
        pos = _t([[list(p0[f] + (p1[f] - p0[f]) * s)] for f in frames]).view(1, 1, len(frames), 1, 3)
        quat = _t([[list(_slerp(cur[f]["quaternion"], goals[f]["quaternion"], s))] for f in frames]).view(1, 1, len(frames), 1, 4)
        # seeds at the current angles (plus small variations): the solver then finds the nearby solution
        # instead of drifting to its default arm pose or another branch
        seeds = q.view(1, 1, -1).repeat(1, L["seeds"], 1)
        seeds[0, 1:] += torch.randn_like(seeds[0, 1:]) * math.radians(L["seed_noise_deg"])
        ik = _planner.ik_solver.solve_pose(GoalToolPose(tool_frames=frames, position=pos, quaternion=quat),
                                           current_state=_state(q.tolist()), seed_config=seeds,
                                           return_seeds=L["seeds"])
        sol = ik.solution.reshape(-1, len(start))[ik.success.reshape(-1)]
        if len(sol) == 0:
            raise RuntimeError(f"no straight path: blocked or out of reach at {100 * s:.0f}% of the line")
        jump = (sol - q).abs().max(dim=-1).values
        k = int(jump.argmin())
        if math.degrees(float(jump[k])) > L["max_joint_step_deg"]:
            d = torch.rad2deg(sol[k] - q)
            worst = sorted(zip(d.abs().tolist(), _planner.joint_names, d.tolist()), reverse=True)[:3]
            raise RuntimeError(f"no straight path: the arm would flip at {100 * s:.0f}% of the line ("
                               + ", ".join(f"{n.replace('_joint', '')} {v:+.0f}" for _, n, v in worst) + " deg)")
        q = sol[k]
        path.append(q)
    dt = float(P["interpolation_dt"])
    pts = [path[0]]
    for a, b in zip(path[:-1], path[1:]):
        k = max(1, math.ceil(float((b - a).abs().max()) / (L["joint_speed_rps"] * dt)))
        pts += [a + (b - a) * (j / k) for j in range(1, k + 1)]
    pos = torch.stack(pts).cpu().tolist()
    return {"joint_names": _planner.joint_names, "dt": dt, "positions": pos, "duration": round(len(pos) * dt, 3)}


def plan_joint(start, goal, world, use_map=None, margin=0.0):
    _world(world, use_map, margin)
    return _trajectory(_planner.plan_cspace(_state(goal), _state(start)))


@app.post("/infer")
def infer(req: Request):
    t0 = time.time()
    p = req.params
    try:
        with _lock:
            if req.action == "fk":
                return envelope.done(req.id, fk(p["q"]), t0)
            if req.action in ("plan_pose", "plan_joint", "plan_line"):
                _attach(p.get("attached"))
            if req.action == "plan_pose":
                return envelope.done(req.id, plan_pose(p["start"], p["goals"], p.get("world"), p.get("map"),
                                                       p.get("margin", 0.0)), t0)
            if req.action == "plan_joint":
                return envelope.done(req.id, plan_joint(p["start"], p["goal"], p.get("world"), p.get("map"),
                                                        p.get("margin", 0.0)), t0)
            if req.action == "plan_line":
                return envelope.done(req.id, plan_line(p["start"], p["goals"], p.get("world"), p.get("map"),
                                                       p.get("margin", 0.0)), t0)
            if req.action == "map":
                return envelope.done(req.id, build_map(p), t0)
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
