"""Dual-arm motion: cuRobo plans the 14 arm joints in the torso frame, the Agent plays the result.

Objects, surfaces and hand goals are kept in the pelvis frame (it stays put while the robot stands);
an ArmPlan converts them to the torso frame with the waist angles it tracks, so the waist can turn
toward a target between steps. Section 6 builds a whole plan first and plays it later; AI control
builds and runs one skill at a time. While a box is squeezed the measured arm angles lag the
commanded ones, so planning starts from the Agent's commanded angles when the arms are engaged.

Obstacles: the fitted cuboids plus, once build_map() has run, the planner's depth obstacle map (everything
the camera saw, the robot's arms removed, the handled object cut out). Every move keeps transit_margin_m
from all of them unless a step asks for less (the last straight approach and the squeeze use 0).
"""
import base64
import logging
import math
import zlib

import numpy as np

from common import config, joints
from common.fk import FK, _rpy, mat_to_quat, quat_to_mat

log = logging.getLogger("motion")


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


class _HandsPlanner:
    """The planner service, told with every request which hands the robot has (it keeps a model per hand profile)."""

    def __init__(self, svc):
        self.svc = svc

    def call(self, action, params):
        return self.svc.call(action, {**params, "hands": config.hands()})

    def __getattr__(self, name):
        return getattr(self.svc, name)


class ArmMotion:
    def __init__(self, link, planner, processing):
        self.cfg = config.load()
        self.link, self.processing = link, processing
        self.planner = _HandsPlanner(planner) if planner is not None else None
        self.fk = FK(config.path(self.cfg["robot"]["urdf"]))
        self.tcp = config.tcp_frames()
        self.idx = {n + "_joint": i for i, n in enumerate(joints.NAMES)}
        self._names = None
        self.map_waist = None                   # waist angles when the obstacle map was taken, None = no map
        self.recovery = False                   # moves now are recovery moves (the agent runs them even after
                                                # a hold in place: they unload the arms)
        self.grip = None                        # the live grip (grip_of) while an object is squeezed, else None

    def hands_changed(self):
        """The robot reported other hands (config.use_hands): kinematics and palm frames of the new ones."""
        self.fk = FK(config.path(self.cfg["robot"]["urdf"]))
        self.tcp = config.tcp_frames()
        self.map_waist, self.grip = None, None

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
        out = {"name": c["name"], "dims": c["dims"], "pose": t["position"] + t["quaternion"]}
        if c.get("margin") is not None:         # its own (smaller) planning margin, e.g. the support
            out["margin"] = c["margin"]
        return out

    def tool_quat_torso(self, q, name):
        """Current orientation (wxyz) of a palm frame relative to the torso."""
        qn = {joints.NAMES[i] + "_joint": v for i, v in q.items()}
        return mat_to_quat(self.fk.pose(self.tcp[name]["parent"], qn)[:3, :3] @ _rpy(*self.tcp[name]["rpy"]))

    def palms(self, q):
        """Pelvis-frame palm contact points (left, right) for {motor index: rad} (waist included)."""
        qn = {joints.NAMES[i] + "_joint": v for i, v in q.items()}
        out = []
        for name in ("left_tcp", "right_tcp"):
            t = self.tcp[name]
            T = self.fk.pose(t["parent"], qn, base="pelvis")
            out.append(T[:3, 3] + T[:3, :3] @ np.array(t["xyz"]))
        return out

    def hand_points(self, q, local):
        """Pelvis-frame points given in each hand link's frame: {name: [points]} -> {name: [points]}."""
        qn = {joints.NAMES[i] + "_joint": v for i, v in q.items()}
        out = {}
        for name, pts in local.items():
            T = self.fk.pose(self.tcp[name]["parent"], qn, base="pelvis")
            out[name] = [T[:3, 3] + T[:3, :3] @ np.asarray(p, float) for p in pts]
        return out

    def palm_frames(self, q):
        """Pelvis-frame palm frames {name: (position, 3x3 rotation)} (the goal convention: twisted on the hand)."""
        qn = {joints.NAMES[i] + "_joint": v for i, v in q.items()}
        out = {}
        for name in ("left_tcp", "right_tcp"):
            t = self.tcp[name]
            T = self.fk.pose(t["parent"], qn, base="pelvis")     # the contact frame's full orientation on the hand
            out[name] = (T[:3, 3] + T[:3, :3] @ np.array(t["xyz"]), T[:3, :3] @ _rpy(*t["rpy"]))
        return out

    @staticmethod
    def object_frame(left, right):
        """The held object's frame from the two palm points: their midpoint and the palm axis' yaw."""
        d = np.asarray(left) - np.asarray(right)
        return (np.asarray(left) + np.asarray(right)) / 2, math.atan2(-d[0], d[1])

    def grip_of(self, q):
        """The grip at q: each palm's pose in the held object's frame. Frozen at the squeeze, it is how the
        hands hold the object from then on: every holding move only moves the object (held_goals)."""
        f = self.palm_frames(q)
        mid, yaw = self.object_frame(f["left_tcp"][0], f["right_tcp"][0])
        c, s_ = math.cos(-yaw), math.sin(-yaw)
        rinv = np.array([[c, -s_, 0.0], [s_, c, 0.0], [0.0, 0.0, 1.0]])
        return {k: (rinv @ (p - mid), rinv @ R) for k, (p, R) in f.items()}

    @staticmethod
    def held_goals(grip, x, y, z, yaw):
        """Palm goals (pelvis frame) that put the held object's frame at (x, y, z) turned `yaw`: the grip moved
        rigidly, the hands never turn on the object (re-orienting them to a nominal pose tilted the box)."""
        c, s_ = math.cos(yaw), math.sin(yaw)
        rz = np.array([[c, -s_, 0.0], [s_, c, 0.0], [0.0, 0.0, 1.0]])
        return {k: {"position": [float(v) for v in np.array([x, y, z]) + rz @ p],
                    "quaternion": [float(v) for v in mat_to_quat(rz @ R)]} for k, (p, R) in grip.items()}

    # ---------------------------------------------------------------- planning (torso frame)
    def plan_pose(self, start, goals, world, attached=None, use_map=None, margin=0.0):
        with self.processing("curobo"):
            return self.planner.call("plan_pose", {"start": [start[self.idx[n]] for n in self.names],
                                                   "goals": goals, "world": world, "attached": attached,
                                                   "map": use_map, "margin": margin})

    def ik(self, start, goals, world, attached=None, use_map=None, margin=0.0):
        with self.processing("curobo"):
            return self.planner.call("ik", {"start": [start[self.idx[n]] for n in self.names],
                                            "goals": goals, "world": world, "attached": attached,
                                            "map": use_map, "margin": margin})

    def plan_joint(self, start, goal, world, attached=None, use_map=None, margin=0.0):
        with self.processing("curobo"):
            return self.planner.call("plan_joint", {"start": [start[self.idx[n]] for n in self.names],
                                                    "goal": [goal[self.idx[n]] for n in self.names], "world": world,
                                                    "attached": attached, "map": use_map, "margin": margin})

    def plan_line(self, start, goals, world, attached=None, use_map=None, margin=0.0):
        with self.processing("curobo"):
            return self.planner.call("plan_line", {"start": [start[self.idx[n]] for n in self.names],
                                                   "goals": goals, "world": world, "attached": attached,
                                                   "map": use_map, "margin": margin})

    # ---------------------------------------------------------------- depth obstacle map
    def build_map(self, frame, geometry, exclude=()):
        """Obstacle map in the planner from one depth frame. exclude: pelvis-frame cuboids of the objects
        being handled (cut out of the map; they stay cuboids so a squeeze can leave them out)."""
        q = self.q_now(commanded=False)
        waist = self.waist_of(q)
        k = frame.intrinsics
        depth = base64.b64encode(zlib.compress(np.ascontiguousarray(frame.depth, dtype=np.uint16).tobytes(), 1))
        with self.processing("curobo"):
            r = self.planner.call("map", {
                "depth": depth.decode(), "shape": list(frame.depth.shape), "depth_scale": frame.depth_scale,
                "intrinsics": [k["fx"], k["fy"], k["ppx"], k["ppy"]],
                "camera_pose": [*map(float, geometry.t), *map(float, mat_to_quat(geometry.R))],
                "q": [q[self.idx[n]] for n in self.names],
                "exclude": [self.cuboid_to_torso(c, waist) for c in exclude]})
        self.map_waist = waist
        return r

    def build_map_views(self, views, geometry, exclude=()):
        """One obstacle map fused from the task scene's views (scene.View: depth, joints, torso pose at capture).
        The map frame is the first view's torso; exclude: pelvis-frame cuboids cut out of every view."""
        ref = self.waist_of(views[0].q)
        T_ref_inv = np.linalg.inv(self.torso_in_pelvis(ref))
        frames = []
        for v in views:
            waist = self.waist_of(v.q)
            T = T_ref_inv @ self.torso_in_pelvis(waist)
            Rr, tr = T[:3, :3] @ geometry.R, T[:3, :3] @ geometry.t + T[:3, 3]
            k = v.intrinsics
            depth = base64.b64encode(zlib.compress(np.ascontiguousarray(v.depth, dtype=np.uint16).tobytes(), 1))
            frames.append({"depth": depth.decode(), "shape": list(v.depth.shape), "depth_scale": v.depth_scale,
                           "intrinsics": [k["fx"], k["fy"], k["ppx"], k["ppy"]],
                           "camera_pose": [*map(float, geometry.t), *map(float, mat_to_quat(geometry.R))],
                           "integrate_pose": [*map(float, tr), *map(float, mat_to_quat(Rr))],
                           "q": [v.q[self.idx[n]] for n in self.names],
                           "exclude": [self.cuboid_to_torso(c, waist) for c in exclude]})
        with self.processing("curobo"):
            r = self.planner.call("map", {"frames": frames})
        self.map_waist = ref
        return r

    def map_pose(self, waist):
        """{"pose": map frame (torso at capture) in the torso frame for these waist angles}, or None."""
        if self.map_waist is None:
            return None
        T = np.linalg.inv(self.torso_in_pelvis(waist)) @ self.torso_in_pelvis(self.map_waist)
        return {"pose": [*map(float, T[:3, 3]), *map(float, mat_to_quat(T[:3, :3]))]}

    def end_q(self, start, traj):
        q = dict(start)
        q.update({self.idx[n]: v for n, v in zip(traj["joint_names"], traj["positions"][-1])})
        return q

    def check_swing(self, traj, label="move"):
        """Refuse a contorted solution: the joints of one arm turning more than arms.max_arm_turn_deg in
        total during one move (cuRobo returns any collision-free IK, e.g. arms wrapped over the shoulders)."""
        lim = self.cfg["arms"]["max_arm_turn_deg"]
        swing = np.degrees(np.abs(np.array(traj["positions"]) - traj["positions"][0]).max(axis=0))
        for side in ("left", "right"):
            turn = sum(s for n, s in zip(traj["joint_names"], swing) if n.startswith(side))
            if turn > lim:
                raise SkillError(f"{label}: refused, the {side} arm joints would turn {turn:.0f} deg in total "
                                 f"(limit {lim}): a contorted arm pose")

    def execute(self, traj, holding=False, slowdown=None, stop=None):
        """Play a trajectory. With `stop`, it plays in arms.chunk_s pieces and ends early (returning True) as
        soon as stop() says so, e.g. once a lowered object touches down."""
        a = self.cfg["arms"]
        dt = traj["dt"] * (slowdown or (a["playback_slowdown_holding"] if holding else a["playback_slowdown"]))
        kf = [{"t": dt, "q": {str(self.idx[n]): round(math.degrees(v), 3)
                                      for n, v in zip(traj["joint_names"], row)}}
              for row in traj["positions"][1:]]
        n = len(kf) if stop is None else max(1, int(a["chunk_s"] / dt))
        for i in range(0, len(kf), n):
            r = self.link.send("play_upperbody", {"keyframes": kf[i:i + n], "interp": "linear",
                                                  "recovery": self.recovery},
                               wait=True, wait_timeout_s=dt * n + 20)
            if not r.get("ok"):
                raise SkillError(f"arm motion failed: {r.get('error')}")
            if stop is not None and stop():
                return True
        return False

    def move_waist(self, waist, seconds):
        q = {str(12 + k): round(math.degrees(v), 3) for k, v in enumerate(waist)}
        r = self.link.send("play_upperbody", {"keyframes": [{"t": seconds, "q": q}], "recovery": self.recovery},
                           wait=True, wait_timeout_s=seconds + 15)
        if not r.get("ok"):
            raise SkillError(f"waist motion failed: {r.get('error')}")


def _free(world):
    """For free moves: every cuboid keeps the move's full margin. The support slab's own small margin lets the
    gloves come down beside a short object in the straight moves; in free moves it let the forearm pass 2 mm from
    the case edge, and arm sag hit the case (runs 19, 20)."""
    return [{k: v for k, v in c.items() if k != "margin"} for c in world]


class ArmPlan:
    """Steps built from a start configuration (all 29 joints incl. waist); `run()` executes them.
    Goals and world cuboids are given in the pelvis frame."""

    def __init__(self, motion, start=None):
        self.m = motion
        self.q = start if start is not None else motion.q_now()
        self.q0 = dict(self.q)                  # where it starts (the heat prediction replays the steps from it)
        self.pitch0 = self.q.get(14, 0.0)       # the waist pitch the balance policy rests at: leans are relative to it
        self.steps = []
        self.state = {"holding": None}
        self.attached = None                    # held box (torso frame) + arm joints when grabbed
        # the depth obstacle map guards the free moves; contact moves (closing on, lifting, lowering, letting go
        # of an object) are checked against the measured cuboids only: the hands touch things there by design,
        # and a glove resting on a rail the coarse map sees made every such move "start in collision"
        self.use_map = True

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

    def _map(self):
        return self.m.map_pose(self.waist) if self.use_map else None

    def _margin(self, margin):
        return self.m.cfg["manipulation"]["transit_margin_m"] if margin is None else margin

    def to_pose(self, label, goals, world, goals_fn=None, margin=None):
        """goals_fn (optional) recomputes the goals at run time from live state (e.g. a corrected grip).
        margin: distance kept from every obstacle (default transit_margin_m)."""
        g, wld = self._torso(goals, _free(world))
        margin = self._margin(margin)
        try:
            traj = self.m.plan_pose(self.q, g, wld, self.attached, self._map(), margin)
        except RuntimeError as e:
            raise SkillError(f"{label}: {e}")
        self.m.check_swing(traj, label)
        self.steps.append({"kind": "traj", "label": label, "traj": traj, "goals": goals, "goals_fn": goals_fn,
                           "world": world, "attached": self.attached, "holding": bool(self.state.get("holding")),
                           "margin": margin})
        self.q = self.m.end_q(self.q, traj)

    def ik(self, label, goals, world, margin=None):
        """The arm configuration (all joints) that puts the palms at the goals, nearest the current one; no
        step is added (no path is planned to it)."""
        g, wld = self._torso(goals, world)
        try:
            r = self.m.ik(self.q, g, wld, self.attached, self._map(), self._margin(margin))
        except RuntimeError as e:
            raise SkillError(f"{label}: {e}")
        q = dict(self.q)
        q.update({self.m.idx[n]: v for n, v in zip(r["joint_names"], r["q"])})
        return q

    def to_joint(self, label, goal, world, margin=None):
        _, wld = self._torso({}, _free(world))
        margin = self._margin(margin)
        try:
            traj = self.m.plan_joint(self.q, goal, wld, self.attached, self._map(), margin)
        except RuntimeError as e:
            raise SkillError(f"{label}: {e}")
        # no swing check: the goal angles are chosen by us (e.g. the ready pose), not by an IK solver
        self.steps.append({"kind": "traj", "label": label, "traj": traj, "goal_q": goal, "world": world,
                           "attached": self.attached, "holding": bool(self.state.get("holding")),
                           "margin": margin})
        self.q = self.m.end_q(self.q, traj)

    def line(self, label, goals, world, goals_fn=None, margin=0.0, traj=None, use_map=True):
        """Straight-line move of the palms (contact moves): no free re-planning, so no arm flips. `traj`: a line
        already proven from this same pose (its IK is seeded randomly, so near a limit a re-plan can fail).
        use_map=False: only the given cuboids (an escape that starts touching what the map holds)."""
        g, wld = self._torso(goals, world)
        try:
            if traj is None:
                traj = self.m.plan_line(self.q, g, wld, self.attached, self._map() if use_map else None, margin)
        except RuntimeError as e:
            raise SkillError(f"{label}: {e}")
        self.m.check_swing(traj, label)
        self.steps.append({"kind": "traj", "line": True, "label": label, "traj": traj, "goals": goals,
                           "goals_fn": goals_fn, "world": world, "attached": self.attached,
                           "holding": bool(self.state.get("holding")), "margin": margin})
        self.q = self.m.end_q(self.q, traj)

    def waist_to(self, label, waist):
        """Turn/tilt the torso (arms keep their pose relative to it). Joints in arms.waist_hold_joints keep
        their angle: the agent holds them where they were engaged, so the plan does too."""
        held = self.m.cfg["arms"]["waist_hold_joints"]
        waist = [waist[0], waist[1], self.pitch0 + waist[2]]   # pitch: a lean from the resting pitch (commanding an
        waist = [w if 12 + k not in held else c for k, (w, c) in enumerate(zip(waist, self.waist))]   # absolute 0 fought
                                                                                        # the policy: -11 Nm, 61 C)
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

    def at_live_grip(self, end, s):
        """A planned holding end pose (all joints) re-solved for the live grip: the object where the plan puts
        it (palm midpoint and axis at the planned end), held the way the hands really hold it (self.grip, frozen
        at the squeeze). The plan holds it with the modelled grip; going to the planned angles pulled the hands
        ~3.5 cm per side into the box (runs 10-12) and turned them on it. IK nearest the planned angles."""
        m = self.m
        if m.grip is None:
            return end
        mid, yaw = m.object_frame(*m.palms(end))
        probe = ArmPlan(m, start=end)
        probe.attached = s["attached"]
        probe.use_map = False
        return probe.ik(s["label"], m.held_goals(m.grip, *mid, yaw), s["world"], margin=0.0)

    def replan_live(self, s):
        """The trajectory to play for a step. Steps whose goals depend on live measurements (goals_fn: grip
        corrections, lining up) are re-solved from the live pose. Others play exactly what was planned and
        checked when the arms are where the plan expects; if not, they go to the planned end angles (a fresh
        IK solve could pick a different arm configuration from which the next straight move is impossible)."""
        live = ArmPlan(self.m)
        live.attached = s["attached"]
        live.state = {"holding": s["holding"]}
        live.use_map = not s.get("line")        # straight contact moves: the cuboids only
        traj = s["traj"]
        grip_end = None                         # a holding step's end, re-solved for the live grip
        if not s.get("goals_fn"):
            now = np.array([live.q[self.m.idx[n]] for n in traj["joint_names"]])
            if np.degrees(np.abs(now - np.array(traj["positions"][0]))).max() < 1.0:
                return traj
            end = dict(live.q)
            end.update({self.m.idx[n]: v for n, v in zip(traj["joint_names"], traj["positions"][-1])})
            try:
                if s["holding"]:
                    end = grip_end = self.at_live_grip(end, s)
                live.to_joint(s["label"], end, s["world"], margin=s.get("margin"))
                return live.steps[-1]["traj"]
            except SkillError as e:              # fall back to solving the step's goal from the live pose
                first = f"{'at the live grip' if s['holding'] else 'planned end angles'}: {e}"
            else:
                first = None
        # fallback chain: one marginal straight line must not end the task (the real pose is never exactly the
        # planned one). Straight -> free collision-checked path to the same goal -> the same with no extra margin
        # -> the planned end angles (not while holding: those assume the planned grip gap, not the real one)
        label, world, margin = s["label"], s["world"], s.get("margin")
        tries = []
        if "goals" in s:
            goals = s["goals_fn"]() if s.get("goals_fn") else s["goals"]
            if s["holding"] and not s.get("goals_fn") and self.m.grip is not None:   # the object where planned,
                mid, yaw = self.m.object_frame(goals["left_tcp"]["position"], goals["right_tcp"]["position"])
                goals = self.m.held_goals(self.m.grip, *mid, yaw)                     # held with the live grip
            if s.get("line"):
                tries.append(("straight", lambda: live.line(label, goals, world, margin=margin)))
            tries.append(("free path", lambda: live.to_pose(label, goals, world, margin=margin)))
            if margin:                          # a little tighter, never zero (zero grazed the boxes)
                tries.append(("free path, half margin", lambda: live.to_pose(label, goals, world, margin=margin / 2)))
            end_q = None if s["holding"] else {k: v for k, v in zip((self.m.idx[n] for n in traj["joint_names"]),
                                                                     traj["positions"][-1])}
        else:                                   # (not while holding either: planned at the modelled grip gap)
            end_q = None if s["holding"] else {k: v for k, v in s["goal_q"].items() if 15 <= k <= 28}
        if grip_end is not None:                # holding: the same chain toward the end re-solved for the live grip
            end_q = {k: v for k, v in grip_end.items() if 15 <= k <= 28}
        if end_q:
            goal = dict(live.q)
            goal.update(end_q)
            if grip_end is None:                # (the live-grip end was just tried at full margin)
                tries.append(("planned end angles", lambda: live.to_joint(label, goal, world, margin=margin)))
            if margin:                          # the move was proven at full margin when planned; from a slightly
                tries.append(("planned end angles, half margin",          # different live start, a little closer
                              lambda: live.to_joint(label, goal, world, margin=margin / 2)))

            def cuboids_only():                 # ...or against the measured cuboids alone
                live.use_map = False
                live.to_joint(label, goal, world, margin=min(margin or 0.0, 0.01))
            tries.append(("planned end angles, cuboids only", cuboids_only))
        errors = [first] if not s.get("goals_fn") and first else []
        for name, fn in tries:
            mk = live.mark()
            try:
                fn()
                if name != "straight" and s.get("line"):
                    log.warning("'%s': straight line impossible from the live pose, used the %s", label, name)
                return live.steps[-1]["traj"]
            except SkillError as e:
                live.restore(mk)
                errors.append(f"{name}: {e}")
        raise SkillError(f"re-planning from the live pose failed ({'; '.join(errors)})")

    def run(self, on_step=None):
        for i, s in enumerate(self.steps):
            if on_step:
                on_step(i, s["label"])
            if s.get("dry"):
                continue
            if s["kind"] == "traj":
                try:
                    traj = self.replan_live(s)
                except SkillError as e:
                    if not s.get("optional"):
                        raise
                    log.warning("skipped optional step '%s': %s", s["label"], e)
                    continue
                self.m.execute(traj, holding=s["holding"])
            elif s["kind"] == "waist":
                self.m.move_waist(s["waist"], s["seconds"])
            else:
                s["fn"]()
