"""Dual-arm motion: cuRobo plans the 14 arm joints in the torso frame, the Agent plays the result.

An ArmPlan is a list of steps (trajectories and checks) built from a start configuration. Section 6
builds a whole plan first and plays it later; AI control builds and runs one skill at a time.
While a box is squeezed the measured arm angles lag the commanded ones, so planning always starts
from the Agent's commanded angles when the arms are engaged.
"""
import math

import numpy as np

from common import config, joints
from common.fk import FK


class SkillError(RuntimeError):
    pass


def pose(xyz, quat=(1.0, 0.0, 0.0, 0.0)):
    return {"position": [float(v) for v in xyz], "quaternion": [float(v) for v in quat]}


def pitch_quat(deg):
    """Palms keep facing each other (±y); fingers tilt down by `deg` (rotation about torso y), wxyz."""
    a = math.radians(deg) / 2
    return (math.cos(a), 0.0, math.sin(a), 0.0)


def cuboid(name, center, dims):
    return {"name": name, "dims": [float(max(d, 0.005)) for d in dims], "pose": [*map(float, center), 1, 0, 0, 0]}


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

    # ---------------------------------------------------------------- robot state
    def q_now(self, commanded=True):
        """{motor index: rad} for all 29 joints; upper body from the commanded angles if engaged."""
        s = self.link.state()
        q = {j["index"]: math.radians(j["deg"]) for j in s["joints"]}
        cmd = s["arms"].get("q_cmd_deg")
        if commanded and cmd:
            q.update({12 + k: math.radians(v) for k, v in enumerate(cmd)})
        return q

    def palms(self, q):
        """Torso-frame palm contact points (left, right) for {motor index: rad}."""
        qn = {joints.NAMES[i] + "_joint": v for i, v in q.items()}
        out = []
        for name in ("left_tcp", "right_tcp"):
            t = self.tcp[name]
            T = self.fk.pose(t["parent"], qn)
            out.append(T[:3, 3] + T[:3, :3] @ np.array(t["xyz"]))
        return out

    # ---------------------------------------------------------------- planning
    def plan_pose(self, start, goals, world):
        with self.processing("curobo"):
            return self.planner.call("plan_pose", {"start": [start[self.idx[n]] for n in self.names],
                                                   "goals": goals, "world": world})

    def plan_joint(self, start, goal, world):
        with self.processing("curobo"):
            return self.planner.call("plan_joint", {"start": [start[self.idx[n]] for n in self.names],
                                                    "goal": [goal[self.idx[n]] for n in self.names], "world": world})

    def end_q(self, start, traj):
        q = dict(start)
        q.update({self.idx[n]: v for n, v in zip(traj["joint_names"], traj["positions"][-1])})
        return q

    def execute(self, traj):
        kf = [{"t": traj["dt"], "q": {str(self.idx[n]): round(math.degrees(v), 3)
                                      for n, v in zip(traj["joint_names"], row)}}
              for row in traj["positions"][1:]]
        if not kf:
            return
        r = self.link.send("play_upperbody", {"keyframes": kf, "interp": "linear"},
                           wait=True, wait_timeout_s=traj["duration"] + 20)
        if not r.get("ok"):
            raise SkillError(f"arm motion failed: {r.get('error')}")


class ArmPlan:
    """Steps built from a start configuration; `run()` executes them on the robot."""

    def __init__(self, motion, start=None):
        self.m = motion
        self.q = start if start is not None else motion.q_now()
        self.steps = []

    def to_pose(self, label, goals, world):
        try:
            traj = self.m.plan_pose(self.q, goals, world)
        except RuntimeError as e:
            raise SkillError(f"{label}: {e}")
        self.steps.append({"kind": "traj", "label": label, "traj": traj})
        self.q = self.m.end_q(self.q, traj)

    def to_joint(self, label, goal, world):
        try:
            traj = self.m.plan_joint(self.q, goal, world)
        except RuntimeError as e:
            raise SkillError(f"{label}: {e}")
        self.steps.append({"kind": "traj", "label": label, "traj": traj})
        self.q = self.m.end_q(self.q, traj)

    def call(self, label, fn):
        """A non-motion step evaluated at run time (contact check, blackboard update...)."""
        self.steps.append({"kind": "call", "label": label, "fn": fn})

    def palms(self):
        return self.m.palms(self.q)

    def summary(self):
        return [{"label": s["label"], "kind": s["kind"],
                 "duration": s["traj"]["duration"] if s["kind"] == "traj" else None} for s in self.steps]

    def run(self, on_step=None):
        for i, s in enumerate(self.steps):
            if on_step:
                on_step(i, s["label"])
            if s["kind"] == "traj":
                self.m.execute(s["traj"])
            else:
                s["fn"]()
