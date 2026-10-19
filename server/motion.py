"""Dual-arm motion: cuRobo plans the 14 arm joints in the torso frame, the Agent plays the result.

Objects, surfaces and hand goals are kept in the pelvis frame (it stays put while the robot stands);
an ArmPlan converts them to the torso frame with the waist angles it tracks, so the waist can turn
toward a target between steps. Section 6 builds a whole plan first and plays it later; AI control
builds and runs one skill at a time. While a box is squeezed the measured arm angles lag the
commanded ones, so planning starts from the Agent's commanded angles when the arms are engaged.
"""
import math

import numpy as np

from common import config, joints
from common.fk import FK, mat_to_quat, quat_to_mat


class SkillError(RuntimeError):
    pass


def pose(xyz, quat=(1.0, 0.0, 0.0, 0.0)):
    return {"position": [float(v) for v in xyz], "quaternion": [float(v) for v in quat]}


def pitch_quat(deg):
    """Palms keep facing each other (±y); fingers tilt down by `deg` (rotation about y), wxyz."""
    a = math.radians(deg) / 2
    return (math.cos(a), 0.0, math.sin(a), 0.0)


def cuboid(name, center, dims, yaw=0.0):
    q = (math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2))
    return {"name": name, "dims": [float(max(d, 0.005)) for d in dims], "pose": [*map(float, center), *q]}


class ArmMotion:
    def __init__(self, link, planner, processing):
        self.cfg = config.load()
        self.link, self.planner, self.processing = link, planner, processing
        self.fk = FK(config.path(self.cfg["robot"]["urdf"]))
        self.tcp = self.cfg["planner_model"]["tcp"]
        self.idx = {n + "_joint": i for i, n in enumerate(joints.NAMES)}
        self._names = None

    @property
    def names(self):
        if self._names is None:
            h = self.planner.health()
            if not h or not h.get("ready"):
                raise SkillError("planner service is not ready")
            self._names = h["joint_names"]
        return self._names

    # ---------------------------------------------------------------- robot state and frames
    def q_now(self, commanded=True):
        """{motor index: rad} for all 29 joints; arms and waist yaw from the commanded angles if engaged.
        Waist roll/pitch are measured (they track commands only approximately; the camera pose uses them)."""
        s = self.link.state()
        q = {j["index"]: math.radians(j["deg"]) for j in s["joints"]}
        cmd = s["arms"].get("q_cmd_deg")
        if commanded and cmd:
            q.update({12 + k: math.radians(v) for k, v in enumerate(cmd) if 12 + k not in (13, 14)})
        return q

    @staticmethod
    def waist_of(q):
        return [q[12], q[13], q[14]]

    def torso_in_pelvis(self, waist):
        return self.fk.pose("torso_link", {"waist_yaw_joint": waist[0], "waist_roll_joint": waist[1],
                                           "waist_pitch_joint": waist[2]}, base="pelvis")

    def to_torso(self, p, waist):
        """Pelvis-frame pose dict -> torso-frame pose dict."""
        Tinv = np.linalg.inv(self.torso_in_pelvis(waist))
        pos = Tinv[:3, :3] @ np.array(p["position"]) + Tinv[:3, 3]
        rot = Tinv[:3, :3] @ quat_to_mat(p["quaternion"])
        return pose(pos, mat_to_quat(rot))

    def cuboid_to_torso(self, c, waist):
        t = self.to_torso({"position": c["pose"][:3], "quaternion": c["pose"][3:]}, waist)
        return {"name": c["name"], "dims": c["dims"], "pose": t["position"] + t["quaternion"]}

    def tool_quat_torso(self, q, name):
        """Current orientation (wxyz) of a palm frame relative to the torso."""
        qn = {joints.NAMES[i] + "_joint": v for i, v in q.items()}
        return mat_to_quat(self.fk.pose(self.tcp[name]["parent"], qn)[:3, :3])

    def palms(self, q):
        """Pelvis-frame palm contact points (left, right) for {motor index: rad} (waist included)."""
        qn = {joints.NAMES[i] + "_joint": v for i, v in q.items()}
        out = []
        for name in ("left_tcp", "right_tcp"):
            t = self.tcp[name]
            T = self.fk.pose(t["parent"], qn, base="pelvis")
            out.append(T[:3, 3] + T[:3, :3] @ np.array(t["xyz"]))
        return out

    # ---------------------------------------------------------------- planning (torso frame)
    def plan_pose(self, start, goals, world, attached=None):
        with self.processing("curobo"):
            return self.planner.call("plan_pose", {"start": [start[self.idx[n]] for n in self.names],
                                                   "goals": goals, "world": world, "attached": attached})

    def plan_joint(self, start, goal, world, attached=None):
        with self.processing("curobo"):
            return self.planner.call("plan_joint", {"start": [start[self.idx[n]] for n in self.names],
                                                    "goal": [goal[self.idx[n]] for n in self.names], "world": world,
                                                    "attached": attached})

    def end_q(self, start, traj):
        q = dict(start)
        q.update({self.idx[n]: v for n, v in zip(traj["joint_names"], traj["positions"][-1])})
        return q

    def execute(self, traj, holding=False):
        a = self.cfg["arms"]
        dt = traj["dt"] * (a["playback_slowdown_holding"] if holding else a["playback_slowdown"])
        kf = [{"t": dt, "q": {str(self.idx[n]): round(math.degrees(v), 3)
                                      for n, v in zip(traj["joint_names"], row)}}
              for row in traj["positions"][1:]]
        if not kf:
            return
        r = self.link.send("play_upperbody", {"keyframes": kf, "interp": "linear"},
                           wait=True, wait_timeout_s=dt * len(kf) + 20)
        if not r.get("ok"):
            raise SkillError(f"arm motion failed: {r.get('error')}")

    def move_waist(self, waist, seconds):
        q = {str(12 + k): round(math.degrees(v), 3) for k, v in enumerate(waist)}
        r = self.link.send("play_upperbody", {"keyframes": [{"t": seconds, "q": q}]}, wait=True, wait_timeout_s=seconds + 15)
        if not r.get("ok"):
            raise SkillError(f"waist motion failed: {r.get('error')}")


class ArmPlan:
    """Steps built from a start configuration (all 29 joints incl. waist); `run()` executes them.
    Goals and world cuboids are given in the pelvis frame."""

    def __init__(self, motion, start=None):
        self.m = motion
        self.q = start if start is not None else motion.q_now()
        self.steps = []
        self.state = {"holding": None}
        self.attached = None                    # held box (torso frame) + arm joints when grabbed

    @property
    def waist(self):
        return ArmMotion.waist_of(self.q)

    def mark(self):
        return len(self.steps), dict(self.q), dict(self.state), self.attached

    def restore(self, mark):
        n, q, state, attached = mark
        del self.steps[n:]
        self.q, self.state, self.attached = q, state, attached

    def attach(self, cuboid_pelvis):
        """From now on plan with this box held in the hands (it moves with them)."""
        c = self.m.cuboid_to_torso(cuboid_pelvis, self.waist)
        self.attached = {"cuboid": c, "q": [self.q[self.m.idx[n]] for n in self.m.names]}

    def detach(self):
        self.attached = None

    def _torso(self, goals, world):
        """Pelvis -> torso frame. A goal with quaternion None keeps the hand's current torso-relative orientation."""
        w = self.waist
        out = {}
        for k, v in goals.items():
            if v.get("quaternion") is None:
                t = self.m.to_torso({"position": v["position"], "quaternion": (1, 0, 0, 0)}, w)
                out[k] = pose(t["position"], self.m.tool_quat_torso(self.q, k))
            else:
                out[k] = self.m.to_torso(v, w)
        return out, [self.m.cuboid_to_torso(c, w) for c in world]

    def to_pose(self, label, goals, world, goals_fn=None):
        """goals_fn (optional) recomputes the goals at run time from live state (e.g. a corrected grip)."""
        g, wld = self._torso(goals, world)
        try:
            traj = self.m.plan_pose(self.q, g, wld, self.attached)
        except RuntimeError as e:
            raise SkillError(f"{label}: {e}")
        self.steps.append({"kind": "traj", "label": label, "traj": traj, "goals": goals, "goals_fn": goals_fn,
                           "world": world, "attached": self.attached, "holding": bool(self.state.get("holding"))})
        self.q = self.m.end_q(self.q, traj)

    def to_joint(self, label, goal, world):
        _, wld = self._torso({}, world)
        try:
            traj = self.m.plan_joint(self.q, goal, wld, self.attached)
        except RuntimeError as e:
            raise SkillError(f"{label}: {e}")
        self.steps.append({"kind": "traj", "label": label, "traj": traj, "goal_q": goal, "world": world,
                           "attached": self.attached, "holding": bool(self.state.get("holding"))})
        self.q = self.m.end_q(self.q, traj)

    def waist_to(self, label, waist):
        """Turn/tilt the torso (arms keep their pose relative to it)."""
        delta = max(abs(math.degrees(a - b)) for a, b in zip(waist, self.waist))
        if delta < 1.0:
            return
        a = self.m.cfg["arms"]
        speed = a["waist_speed_holding_dps"] if self.state.get("holding") else a["waist_speed_dps"]
        seconds = max(1.0, delta / speed)
        self.steps.append({"kind": "waist", "label": label, "waist": list(waist), "seconds": seconds})
        self.q = dict(self.q)
        self.q.update({12 + k: v for k, v in enumerate(waist)})

    def call(self, label, fn):
        """A non-motion step evaluated at run time (contact check, blackboard update...)."""
        self.steps.append({"kind": "call", "label": label, "fn": fn})

    def palms(self):
        return self.m.palms(self.q)

    def summary(self):
        dur = {"traj": lambda s: s["traj"]["duration"], "waist": lambda s: round(s["seconds"], 2)}
        return [{"label": s["label"], "kind": s["kind"], "duration": dur[s["kind"]](s) if s["kind"] in dur else None}
                for s in self.steps]

    def replan_live(self, s):
        """Re-plan a step from the robot's live commanded pose (reality drifts from the plan: sag,
        a box that turned in the hands, grip corrections). The build-time plan proved it feasible."""
        live = ArmPlan(self.m)
        live.attached = s["attached"]
        live.state = {"holding": s["holding"]}
        try:
            if "goals" in s:
                live.to_pose(s["label"], s["goals_fn"]() if s.get("goals_fn") else s["goals"], s["world"])
            else:
                goal = dict(live.q)
                goal.update({k: v for k, v in s["goal_q"].items() if 15 <= k <= 28})
                live.to_joint(s["label"], goal, s["world"])
        except SkillError as e:
            raise SkillError(f"re-planning from the live pose failed: {e}")
        return live.steps[-1]["traj"]

    def run(self, on_step=None):
        for i, s in enumerate(self.steps):
            if on_step:
                on_step(i, s["label"])
            if s.get("dry"):
                continue
            if s["kind"] == "traj":
                self.m.execute(self.replan_live(s), holding=s["holding"])
            elif s["kind"] == "waist":
                self.m.move_waist(s["waist"], s["seconds"])
            else:
                s["fn"]()
