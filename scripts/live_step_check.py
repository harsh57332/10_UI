"""Offline: plan the stack (as scene_plan_check) and run its live-only steps against a simulated robot, so the
code that only runs on the robot is exercised before it is: the touch-checked descent (free, and a glove stopped
by an edge part way), the test lift, and the lowering onto the target (landing part way, and never landing).
The simulated arms follow their commands; 'stuck' / 'landed' freeze the measured pose while the command moves on.
Run: ~/miniconda3/envs/g1ui_server/bin/python -m scripts.live_step_check [--x 0.328 --y -0.085 --gap 0.09]
     [--left X Y] [--map YAW ROLL PITCH]   (--map: with the robot's current depth map, after a Generate)"""
import argparse
import math
from contextlib import nullcontext

import numpy as np

from server.motion import ArmMotion, ArmPlan, SkillError
from server.services import Service
from server import skills
from server.skills import SKILLS, Ctx

ap = argparse.ArgumentParser()
ap.add_argument("--x", type=float, default=0.328)
ap.add_argument("--y", type=float, default=-0.085)
ap.add_argument("--gap", type=float, default=0.09)
ap.add_argument("--dims", type=float, nargs=3, default=[0.19, 0.14, 0.088], help="depth width height")
ap.add_argument("--support", type=float, default=0.0805)
ap.add_argument("--left", type=float, nargs=2, help="the left box's centre x y (default: beside the right one)")
ap.add_argument("--map", type=float, nargs=3, metavar=("YAW", "ROLL", "PITCH"),
                help="plan with the depth map the planner holds now, captured at these waist angles (deg)")
a = ap.parse_args()

m = ArmMotion(link=None, planner=Service("planner"), processing=lambda s: nullcontext())
ctx = Ctx(None, None, None, None, m, lambda s: nullcontext())
start = {i: 0.0 for i in range(29)}
start.update({15 + k: math.radians(v) for k, v in enumerate([16.6, 13.0, -1.2, 56.2, 11.4, 1.5, -2.0, 16.8, -12.6, 1.5,
                                                              56.2, -5.5, 1.3, -0.1])})
d, w, h = a.dims


def box(x, y):
    return {"center": [x, y, a.support + h / 2], "dims": [d, w, h], "top_z": a.support + h, "support_z": a.support,
            "front_x": x - d / 2, "distance_m": math.hypot(x, y), "yaw": 0.0}


ctx.bb["objects"] = {"right box": box(a.x, a.y), "left box": box(*(a.left or (a.x, a.y + w + a.gap)))}
if a.map:                                       # the live robot's last map (e.g. from a Generate just before)
    m.map_waist = [math.radians(v) for v in a.map]
plan = ArmPlan(m, start=dict(start))
plan.state.update(hand="EMPTY", grasp=None, grip=None)
for name, params in [("pre_grasp", {"object": "right box"}), ("grasp", {}), ("lift", {}), ("place", {"on": "left box"})]:
    SKILLS[name]["fn"](ctx, plan, **params)
    if name == "pre_grasp":
        g = plan.state["grasp"]
    if name == "grasp":
        grip = plan.state["grip"]               # filled in live by the squeeze; the test lift reads it
print("planned:", len(plan.steps), "steps")


def pose_at(step):
    """The robot's full pose (waist included) where a planned step starts, replaying the plan up to it."""
    q = dict(plan.q0)
    for st in plan.steps:
        if st is step:
            return q
        if st["kind"] == "traj":
            q = m.end_q(q, {**st["traj"], "positions": [st["traj"]["positions"][-1]]})
        elif st["kind"] == "waist":
            q.update({12 + i: v for i, v in enumerate(st["waist"])})
    return q


class Sim:
    """Arms that follow their commands, except that from piece `freeze_at` on the measured pose stays put."""

    def __init__(self, q, freeze_at=None):
        self.q, self.meas, self.freeze_at, self.n = dict(q), None, freeze_at, 0

    def q_now(self, commanded=False):
        return dict(self.q) if (commanded or self.meas is None) else dict(self.meas)

    def execute(self, t, holding=False, slowdown=None, stop=None):
        step = max(1, len(t["positions"]) // 8)
        for i in range(0, len(t["positions"]), step):
            self.q = m.end_q(self.q, {**t, "positions": [t["positions"][min(i + step, len(t["positions"]) - 1)]]})
            self.n += 1
            if self.freeze_at and self.n == self.freeze_at:
                self.meas = dict(self.q)
            if stop is not None and stop():
                return True
        return False


def run(label, step_label, freeze_at=None, before=None):
    k = [i for i, st in enumerate(plan.steps) if st["label"] == step_label][0]
    sim = Sim(pose_at(plan.steps[k]), freeze_at)
    m.q_now, m.execute = sim.q_now, sim.execute
    ctx.link = type("L", (), {"state": lambda self: {"joints": [{"index": i, "deg": math.degrees(v)}
                                                                for i, v in sim.q_now().items()],
                                                     "arms": {"engaged": True}}})()
    ctx.agent = lambda *args, **kw: {"ramp_s": 0.0}
    grip.update(G=None)                         # no live grip until a (simulated) squeeze measures one
    m.grip = None
    if before:
        try:
            before(sim)
        except SkillError as e:
            print(f"{label}: skipped (the simulated situation is unreachable here: {e})")
            return
    try:
        print(f"{label}: {plan.steps[k]['fn']()}")
    except SkillError as e:
        print(f"{label}: refused -> {e}")


def freeze(sim):
    """What the live squeeze does: the grip as the hands hold the object now."""
    m.grip = grip["G"] = m.grip_of(sim.q)
    grip["yaw"] = m.object_frame(*m.palms(sim.q))[1]


def gripped(sim):
    left, right = m.palms(sim.q)
    grip.update(gap=float(np.linalg.norm(left - right)), force=12.0)
    freeze(sim)


def resting_high(sim):
    """The gloves came to rest on the support 1.5 cm above the planned grasp height (as on the robot)."""
    lp, rp = m.palms(sim.q)
    mid = (lp + rp) / 2
    goals = {"left_tcp": {"position": [*lp[:2], float(lp[2]) + 0.015], "quaternion": None},
             "right_tcp": {"position": [*rp[:2], float(rp[2]) + 0.015], "quaternion": None}}
    sim.q = ArmPlan(m, start=dict(sim.q)).ik("resting high", goals, [], margin=0.0)
    print(f"   (hands raised from {mid[2]:.3f} to {float(m.palms(sim.q)[0][2] + m.palms(sim.q)[1][2]) / 2:.3f})")


def lined_up_unplannable(sim):
    """The line-up moved the arms 1.5 cm, and a lowering from there cannot be planned (as on the robot)."""
    lp, rp = m.palms(sim.q)
    goals = {k: {"position": [float(p[0]) + 0.015, float(p[1]), float(p[2])], "quaternion": None}
             for k, p in (("left_tcp", lp), ("right_tcp", rp))}
    sim.q = ArmPlan(m, start=dict(sim.q)).ik("lined up", goals, [], margin=0.0)
    real = skills._lower_live
    calls = {"n": 0}

    def once(*args, **kw):
        calls["n"] += 1
        if calls["n"] == 1:
            raise SkillError("lower until contact: planner: no collision-free plan found (simulated)")
        return real(*args, **kw)
    skills._lower_live = once


run("descent, free", "come down beside the object (touch-checked)")
run("descent, a glove stopped part way (must back up and refuse)", "come down beside the object (touch-checked)", 3)
run("descent, a glove rests on a rail near the bottom (grips from there)", "come down beside the object (touch-checked)", 7)
run("close in, as planned (touches the faces)", "close until contact", 4)
run("close in, gloves resting 1.5 cm high (touches the faces)", "close until contact", 4, before=resting_high)
run("test lift", "secure the grip (test lift)", before=gripped)
run("lowering, lands part way", "lower until it sits on the target", 4)
run("lowering, never lands (must refuse at the limit)", "lower until it sits on the target")
run("lowering after a line-up, live lowering impossible (back to the planned one, lands)",
    "lower until it sits on the target", 6, before=lined_up_unplannable)


def wide_grip(sim, extra=0.035):
    """The real grip holds the palms `extra` per side wider than the plan's squeeze gap (force squeeze; the
    command follows the hands): runs 10-12 had 16.2 cm real vs 9.2 cm planned."""
    lp, rp = m.palms(sim.q)
    u = (lp - rp) / np.linalg.norm(lp - rp)
    goals = {"left_tcp": {"position": list(lp + u * extra), "quaternion": None},
             "right_tcp": {"position": list(rp - u * extra), "quaternion": None}}
    sim.q = ArmPlan(m, start=dict(sim.q)).ik("wide grip", goals, [], margin=0.0)
    freeze(sim)


def gap_kept(label, step_label, extra=0.035):
    """A holding step re-planned from a wider (or narrower) real grip must keep the real gap (not go back to the
    planned one)."""
    k = [i for i, st in enumerate(plan.steps) if st["label"] == step_label][0]
    sim = Sim(pose_at(plan.steps[k]))
    m.q_now, m.execute = sim.q_now, sim.execute
    grip.update(G=None)
    m.grip = None
    wide_grip(sim, extra)
    real = float(np.linalg.norm(np.subtract(*m.palms(sim.q))))
    try:
        t = plan.replan_live(plan.steps[k])
        end = float(np.linalg.norm(np.subtract(*m.palms(m.end_q(sim.q, t)))))
        planned = float(np.linalg.norm(np.subtract(*m.palms(m.end_q(sim.q, plan.steps[k]["traj"])))))
        ok = abs(end - real) < 0.005
        print(f"{label}: {'OK' if ok else 'WRONG'} palm gap at the end {100 * end:.1f} cm (real grip {100 * real:.1f}, "
              f"plan {100 * planned:.1f})")
    except SkillError as e:
        print(f"{label}: refused -> {e}")


gap_kept("move above the target from a 7 cm wider real grip (keeps the real gap)", "move above the target")
gap_kept("move above the target from a 2.2 cm narrower real grip (run 13: 15.3 vs 17.5)", "move above the target", -0.011)
gap_kept("move above the target from a 3.1 cm narrower real grip (run 16: 13.7 vs 16.8)", "move above the target", -0.0155)
run("lowering from a 7 cm wider real grip, lands (keeps the real gap)", "lower until it sits on the target", 4,
    before=wide_grip)


def rigid_lift():
    """The squeeze left both wrists bent 6 deg (palms turned on the box): the lift must move the hands straight
    up exactly as they hold the box (same orientations, same offsets), not turn them back to the nominal pose."""
    from server.motion import ArmMotion
    k = [i for i, st in enumerate(plan.steps) if st["label"] == "lift"][0]
    sim = Sim(pose_at(plan.steps[k]))
    m.q_now, m.execute = sim.q_now, sim.execute
    f = m.palm_frames(sim.q)
    a = math.radians(6)
    tw = {"left_tcp": a, "right_tcp": -a}
    goals = {n: {"position": list(p), "quaternion": list(skills.mat_to_quat(
        np.array([[math.cos(tw[n]), -math.sin(tw[n]), 0], [math.sin(tw[n]), math.cos(tw[n]), 0], [0, 0, 1]]) @ R))}
        for n, (p, R) in f.items()}
    sim.q = ArmPlan(m, start=dict(sim.q)).ik("bent wrists", goals, [], margin=0.0)
    gripped(sim)
    before = m.palm_frames(sim.q)
    t = plan.replan_live(plan.steps[k])
    after = m.palm_frames(m.end_q(sim.q, t))
    dz = [float(after[n][0][2] - before[n][0][2]) for n in before]
    dxy = max(float(np.linalg.norm(after[n][0][:2] - before[n][0][:2])) for n in before)
    turn = max(math.degrees(math.acos(min(1.0, (np.trace(before[n][1].T @ after[n][1]) - 1) / 2))) for n in before)
    ok = dxy < 0.01 and turn < 1.0 and abs(dz[0] - dz[1]) < 0.01     # the 1 cm working precision
    print(f"lift with wrists bent by the squeeze (straight up, hands not turned): {'OK' if ok else 'WRONG'} "
          f"up {100 * dz[0]:.1f}/{100 * dz[1]:.1f} cm, sideways {100 * dxy:.2f} cm, hands turned {turn:.2f} deg")


rigid_lift()
