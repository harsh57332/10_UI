"""cuRobo arm planner (14 arm joints, torso frame, waist fixed). Envelope on POST /infer.

actions
  fk          {q: [14 rad]}                       -> {joint_names, poses: {left_tcp: {position, quaternion}, ...}}
  plan_pose   {start: [14], goals: {left_tcp: {position, quaternion(wxyz)}, right_tcp: ...}, world: [cuboid]}
  plan_joint  {start: [14], goal: [14], world: [cuboid]}
  (both take optional "attached": {"cuboid": cuboid, "q": [14]} = a held box moving with the hands)
     -> {joint_names, dt, positions: [[14], ...], duration}
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


def _load():
    global _planner
    import yaml
    from curobo.motion_planner import MotionPlanner, MotionPlannerCfg
    robot = yaml.safe_load(open(config.path(P["robot_yml"])))
    urdf = config.path(P["urdf"])          # the builder writes absolute paths; re-root them here
    robot["kinematics"]["urdf_path"], robot["kinematics"]["asset_root_path"] = str(urdf), str(urdf.parent)
    cfg = MotionPlannerCfg.create(
        robot=robot,
        scene_model={"cuboid": dict(_FAR)},
        collision_cache={"cuboid": P["max_cuboids"]},
        interpolation_dt=P["interpolation_dt"],
    )
    _planner = MotionPlanner(cfg)
    _planner.warmup(enable_graph=True, num_warmup_iterations=3)


def _t(x):
    return torch.as_tensor(x, device="cuda", dtype=torch.float32)


def _state(q):
    from curobo.types import JointState
    return JointState.from_position(_t(q).view(1, -1), joint_names=_planner.joint_names)


def _world(cuboids):
    from curobo.scene import Scene
    cub = dict(_FAR)
    for c in cuboids or []:
        cub[c["name"]] = {"dims": list(map(float, c["dims"])), "pose": list(map(float, c["pose"]))}
    _planner.update_world(Scene.create({"cuboid": cub}))


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


def plan_pose(start, goals, world):
    from curobo.types import GoalToolPose
    frames = _planner.tool_frames
    missing = [f for f in frames if f not in goals]
    if missing:   # keep an unspecified hand where it is
        cur = fk(start)["poses"]
        goals = {**goals, **{f: cur[f] for f in missing}}
    pos = _t([[goals[f]["position"]] for f in frames]).view(1, 1, len(frames), 1, 3)
    quat = _t([[goals[f]["quaternion"]] for f in frames]).view(1, 1, len(frames), 1, 4)
    _world(world)
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


def plan_joint(start, goal, world):
    _world(world)
    return _trajectory(_planner.plan_cspace(_state(goal), _state(start)))


@app.post("/infer")
def infer(req: Request):
    t0 = time.time()
    p = req.params
    try:
        with _lock:
            if req.action == "fk":
                return envelope.done(req.id, fk(p["q"]), t0)
            if req.action in ("plan_pose", "plan_joint"):
                _attach(p.get("attached"))
            if req.action == "plan_pose":
                return envelope.done(req.id, plan_pose(p["start"], p["goals"], p.get("world")), t0)
            if req.action == "plan_joint":
                return envelope.done(req.id, plan_joint(p["start"], p["goal"], p.get("world")), t0)
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
