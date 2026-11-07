"""Skill library shared by Motion planning (section 6) and AI control (section 7).

The LLM only chooses skills and their parameters; the poses come from OWLv2 + depth + cuRobo.
Arm skills append steps to an ArmPlan (so section 6 can plan everything first and play later);
the other skills run immediately. Positions are in the pelvis frame (x forward, y left, z up), which
does not move when the waist turns, so measurements stay valid until the robot walks or turns.
"""
import base64
import copy
import logging
import math
import threading
import time

import cv2
import numpy as np

from common import config, joints

from common.fk import mat_to_quat

from . import heat
from .motion import ArmPlan, SkillError, cuboid, pose
from .robot_view import publish

log = logging.getLogger("skills")
SURFACE_WORDS = ("table", "desk", "shelf", "counter", "bench", "stand", "cabinet")
TARGET_WORDS = SURFACE_WORDS + ("case", "bin", "platform", "stool", "cart")   # section 7 (no task scene): a big thing to place on, measured as a top surface
SKILLS = {}


# The hands are always in one of these states; every action says which ones it may start from.
#   EMPTY: arms at ready, nothing held      AT_OBJECT: gloves beside an object, not touching
#   GRIPPING: object squeezed, still on its support      HOLDING: object lifted (carry pose)
HANDS = ("EMPTY", "AT_OBJECT", "GRIPPING", "HOLDING")


def skill(name, params, desc, kind, hands=HANDS):
    """kind: 'perceive' (camera only), 'move' (walks/turns), 'arm' (adds to an ArmPlan), 'basic',
    'internal' (not offered to the LLM). hands: the hand states the action may start from."""
    def wrap(fn):
        SKILLS[name] = {"name": name, "params": params, "desc": desc, "kind": kind, "fn": fn, "hands": hands}
        return fn
    return wrap


def clean_params(name, params):
    """Keep only the parameters the skill declares (the LLM sometimes adds extras, e.g. grasp {"object": ...}).
    Returns (params, the required ones that are missing); a declaration starting with 'optional' is not required."""
    declared = SKILLS[name]["params"]
    params = {k: v for k, v in (params or {}).items() if k in declared and v not in (None, "")}
    return params, [k for k, d in declared.items() if k not in params and not d.startswith("optional")]


def catalog(kinds=None):
    """Skill list for the LLM prompt (internal skills are never offered)."""
    return [{"skill": s["name"], "params": s["params"], "what": s["desc"], "hands_before": list(s["hands"])}
            for s in SKILLS.values() if s["kind"] != "internal" and (kinds is None or s["kind"] in kinds)]


class Ctx:
    def __init__(self, link, camera, vision, geometry, motion, processing):
        self.cfg = config.load()
        self.link, self.camera, self.vision = link, camera, vision
        self.geometry, self.motion, self.processing = geometry, motion, processing
        self.bb = {"objects": {}, "surfaces": {}, "holding": None,   # blackboard
                   "hand": "EMPTY", "grasp": None, "grip": None}
        self.scene = None       # the task scene (server/scene.py) while a section 6 task is planned or played
        self.view = None        # the UI's robot's view panel (server/robot_view.py); None in scripts and replays

    def summary(self):
        """Short world state for LLM prompts."""
        r = lambda v: [round(x, 2) for x in v]
        return {
            "objects": {k: {"center_m": r(v["center"]), "size_m": r(v["dims"]), "distance_m": round(v["distance_m"], 2)}
                        for k, v in self.bb["objects"].items()},
            "surfaces": {k: {"front_edge_m": round(v["front_x"], 2), "height_m": round(v["top_z"], 2)}
                         for k, v in self.bb["surfaces"].items()},
            "holding": self.bb["holding"]["object"] if self.bb["holding"] else None,
        }

    def forget_positions(self):
        """After walking/turning every remembered position (and the obstacle map) is in an old frame."""
        self.bb["objects"], self.bb["surfaces"] = {}, {}
        self.motion.map_waist = None

    def agent(self, action, params=None, timeout=60.0):
        r = self.link.send(action, params or {}, wait=True, wait_timeout_s=timeout)
        if not r.get("ok"):
            raise SkillError(f"{action}: {r.get('error')}")
        return r.get("result")


def new_plan(ctx):
    """A plan that starts from the robot's live pose and the hands' current state."""
    plan = ArmPlan(ctx.motion)
    plan.state.update({k: ctx.bb[k] for k in ("holding", "hand", "grasp", "grip")})
    return plan


def check_hands(name, state):
    allowed = SKILLS[name]["hands"]
    if state not in allowed:
        raise SkillError(f"'{name}' needs the hands {' or '.join(allowed)}, but they are {state}")


def _arm_travel_deg(steps):
    """Arm motion of planned steps: for each move the path length of its most-travelled joint (deg), summed. The
    carry/place moves with the least of it are the simplest for the arms."""
    total = 0.0
    for st in steps:
        if st["kind"] == "traj" and not st.get("dry"):
            P = st["traj"]["positions"]
            total += math.degrees(max(sum(abs(P[i + 1][k] - P[i][k]) for i in range(len(P) - 1)) for k in range(len(P[0]))))
    return total


def _swing_deg(traj):
    """How far the most-turned arm joint goes between the start and the end of a planned move (deg)."""
    P = traj["positions"]
    return math.degrees(max(abs(b - a) for a, b in zip(P[0], P[-1])))


def _to_state(ctx, plan, hand, **info):
    """Change the hand state: in the plan now, on the robot when this step runs."""
    plan.state.update(hand=hand, **info)
    plan.call(f"hands: {hand}", lambda: ctx.bb.update(hand=hand, **info))


def _check_heat(ctx):
    arms = ctx.link.state()["arms"]
    if arms.get("warning") or arms.get("fault"):
        raise SkillError(arms.get("fault") or arms.get("warning"))


def execute(ctx, plan, on_step=None):
    """Run a plan, watching the arm motor temperatures before every step. Any failure brings the hands to
    a safe state for what they hold (retreat, or set down and retreat); the arms are never dropped."""
    def step(i, label):
        if on_step:
            on_step(i, label)
        _check_heat(ctx)
    check_waist_lock(ctx)
    heat.check(ctx, plan)                       # before anything moves: would it reach the heat warning?
    try:
        plan.run(on_step=step)
    except Exception as e:
        recovered = recover(ctx)
        raise SkillError(f"{e} -> {recovered}")


def recover(ctx):
    """Bring the hands to EMPTY the safe way for their current state. Returns what was done. Its moves are
    recovery moves (the agent runs them even after a hold in place). If nothing can be planned, the last
    resort opens and stows the arms with direct joint moves: a stop never leaves hot arms stuck high."""
    done = []
    ctx.motion.recovery = True
    try:
        hand = ctx.bb["hand"]
        if hand == "HOLDING" and not ctx.bb.get("holding"):   # already let go (the place released it): nothing
            hand = "EMPTY"                                       # to set down (run 37: set_down on None crashed)
        for name in {"HOLDING": ("set_down", "retreat"), "GRIPPING": ("retreat",),
                     "AT_OBJECT": ("retreat",)}.get(hand, ()):
            plan = new_plan(ctx)
            SKILLS[name]["fn"](ctx, plan)
            plan.run()
            done.append(name)
        return f"recovered: {', '.join(done) or 'nothing to do'}"
    except Exception as e:
        log.error("recovery failed after %s: %s", done, e)
        try:
            return f"recovery failed after {done} ({e}); last resort: {_last_resort(ctx)}"
        except Exception as e2:
            log.error("last resort failed: %s", e2)
            return f"RECOVERY FAILED after {done}: {e}; last resort failed: {e2} (arms still held)"
    finally:
        ctx.motion.recovery = False


def _escape_plan(ctx):
    """The way out when nothing else plans, typically because the hands touch the object (every plan then 'starts in
    collision', 2026-10-29: the arms stayed held beside a box and the waist pitch reached 98 C). Straight up until
    the hands' lowest point clears every object top, checked against everything but the object the hands touch
    (that one would block any plan from contact; the neighbours must not be swept), then down by the sides with
    everything known (normal margin and map, else the objects' exact boxes). A height that does not plan is tried
    higher. Returns the plan (not run). Tested on the stuck pose of 2026-10-29 16:47."""
    M = ctx.cfg["manipulation"]
    world = _scene_world(ctx)
    touched = (ctx.bb.get("grasp") or {}).get("object") or (ctx.bb.get("holding") or {}).get("object")
    others = [w for w in world if w["name"] != touched]
    plan = ArmPlan(ctx.motion)
    start = plan.mark()
    left, right = plan.palms()
    mid, yaw = plan.m.object_frame(left, right)
    half = float(np.linalg.norm(left - right)) / 2
    top = max([v["top_z"] for v in ctx.bb["objects"].values()] + [float(mid[2]) - _glove()["below"]])
    last, up = None, None
    for extra in M["escape_up_extra_m"]:
        plan.restore(start)
        z_up = top + _glove()["below"] + M["retreat_clear_m"] + extra
        try:
            if z_up > mid[2] + 0.005:
                plan.line("hands straight up", _hands(mid[0], mid[1], z_up, half, yaw, keep=True), others,
                          use_map=False)
        except SkillError as e:
            last = e
            log.warning("escape up %.0f cm above the tops: %s", 100 * (z_up - top), e)
            continue
        try:                                    # the hands are clear above everything: straighten the waist here. "Arms out to the
            plan.waist_to("waist straight", [math.radians(v) for v in M["waist_deg"]])   # sides" with the waist turned toward
        except SkillError as e:                 # the stack put the left arm inside it: goal in collision (3/3 run 2026-10-30
            last = e                            # 17:07, the arms stayed held and the shoulder reached 89 C)
            continue
        up = plan.mark()
        for margin, use_map in ((M["near_margin_m"], True), (0.0, False)):
            plan.restore(up)
            plan.use_map = use_map
            try:
                for name, angles in (("arms out to the sides", M["side_deg"]), ("arms down", M["rest_deg"])):
                    goal = dict(plan.q)
                    goal.update({15 + k: math.radians(v) for k, v in enumerate(angles)})
                    plan.to_joint(name, goal, world, margin=margin)
                plan.use_map = True
                return plan
            except SkillError as e:
                last = e
                log.warning("escape down (margin %.3f, map %s): %s", margin, use_map, e)
    for margin, use_map in ((M["near_margin_m"], True), (0.0, False)):   # no way down by the sides: the ready pose (cool, in
        plan.restore(up if up is not None else start)                     # front, away from the stack); the arms stay held there
        plan.use_map = use_map
        try:
            goal = dict(plan.q)
            goal.update({15 + k: math.radians(v) for k, v in enumerate(M["ready_deg"])})
            plan.to_joint("ready pose", goal, world, margin=margin)
            plan.use_map = True
            plan.ended_ready = True
            log.warning("escape: arms to the ready pose (no way down by the sides)")
            return plan
        except SkillError as e:
            last = e
            log.warning("escape to ready (margin %.3f, map %s): %s", margin, use_map, e)
    plan.use_map = True
    raise last


def _last_resort(ctx):
    """Nothing else could be planned: stop squeezing, then the escape (planned, see _escape_plan), waist straight,
    release. If even that cannot be planned the arms stay held: never blind joint moves (run 37: the old direct
    stow swept the just-made stack off)."""
    a = ctx.cfg["arms"]
    _grip_force(ctx, squeeze_n=0.0, payload_kg=0.0)
    try:
        plan = _escape_plan(ctx)
    except SkillError as e:
        log.error("last resort: no safe path (%s): the arms stay held", e)
        return f"arms HELD where they are (no collision-free way out: {e})"
    for st in plan.steps:
        if st["kind"] == "waist":
            ctx.motion.move_waist(st["waist"], st["seconds"])
        else:
            ctx.motion.execute(st["traj"])
    if getattr(plan, "ended_ready", False):     # held at the ready pose, waist straight: cool and clear, never dropped
        ctx.bb.update(hand="EMPTY", holding=None, grasp=None, grip=None)
        return "hands up, back to the ready pose (no way down by the sides); the arms stay held there, release them from the UI"
    j = {x["index"]: x["deg"] for x in ctx.link.state()["joints"]}
    if abs(j[12]) > 1.0:                        # the arms are down by the sides now: the waist can turn back
        ctx.motion.move_waist([0.0, math.radians(j[13]), math.radians(j[14])], max(1.0, abs(j[12]) / a["waist_speed_dps"]))
    ctx.agent("release_upperbody", {}, timeout=15)
    ctx.bb.update(hand="EMPTY", holding=None, grasp=None, grip=None)
    return "hands up, back, arms down by the sides (planned around everything), released"


def rest(ctx):
    """After a task: with empty hands the arms go back to their normal position, released to the walking
    policy (hanging at the sides), which keeps the motors cool. Arms holding or touching something stay held
    (never dropped near objects). Returns whether they were released."""
    if not ctx.cfg["manipulation"]["rest_after_task"] or ctx.bb["hand"] != "EMPTY":
        return False
    if not ctx.link.state()["arms"]["engaged"]:
        return False
    # first a planned, collision-checked move down to the hanging pose, then release: released from the ready
    # pose the arms swung down freely and the hands hit the case's front face (the user's photo, 2026-10-24)
    M = ctx.cfg["manipulation"]
    plan = ArmPlan(ctx.motion)
    try:                                        # down by the sides, around the support and the objects
        _via_sideways(ctx, plan, M["rest_deg"], "arms down")
        for st in plan.steps:
            ctx.motion.execute(st["traj"])
    except SkillError as e:                     # the arms touch something in the model (an elbow within the margin of a stack):
        log.warning("rest: %s", e)             # the same route with no margin and the exact boxes only, then (2026-11-02, told
        if not _rest_relaxed(ctx):              # twice: the arms stayed held) released from the ready pose if they are there
            if not _at_ready_pose(ctx):
                log.warning("rest: not at the ready pose and no planned way down: the arms stay held")
                return False
            log.warning("rest: no planned way down; released from the ready pose (the arms hang down by the sides)")
    ctx.agent("release_upperbody", {}, timeout=15)
    return True


@skill("arms_down", {}, "Arms back down by the sides: the planned, collision-checked way out to the sides first, then released "
       "(never straight down in front). Empty hands.", "internal", hands=("EMPTY",))
def arms_down(ctx):
    return {"released": rest(ctx)}


def _rest_relaxed(ctx):
    """The way down by the sides planned with no margin against the exact boxes only (no depth map): True if it ran."""
    M = ctx.cfg["manipulation"]
    plan = ArmPlan(ctx.motion)
    plan.use_map = False
    try:
        _via_sideways(ctx, plan, M["rest_deg"], "arms down", margin=0.0)
        for st in plan.steps:
            ctx.motion.execute(st["traj"])
        return True
    except SkillError as e:
        log.warning("rest (no margin, boxes only): %s", e)
        return False


def _at_ready_pose(ctx, tol_deg=15.0):
    """The arms are at the ready pose (commanded), the posture the task ends in."""
    m = ctx.motion
    q = m.q_now(commanded=True)
    want = [math.radians(v) for v in ctx.cfg["manipulation"]["ready_deg"]]
    return all(abs(q[15 + k] - w) <= math.radians(tol_deg) for k, w in enumerate(want))


def check_waist_lock(ctx):
    """SAFETY: no motion unless the robot's agent locks the same waist joints as this config. After the 2026-10-24
    fall (a freed waist pitch tilted the torso, the balance policy stepped, the robot fell) an agent left with an
    older unlocked config must not move before the locked config is deployed."""
    want = sorted(ctx.cfg["arms"]["waist_hold_joints"])
    have = ctx.link.state()["arms"].get("waist_held")
    if have is None or sorted(have) != want:
        raise SkillError(f"SAFETY: the robot locks waist joints {have}, this config {want}: deploy the config "
                         f"(scripts/robot.sh deploy && scripts/robot.sh restart agent) before any motion")


def check_cool(ctx):
    """A task starts only with cool arm motors: a stack heats the shoulder pitch by ~35 C, so a warmer start
    would reach the over-temperature warning half way through."""
    lim = ctx.cfg["manipulation"]["start_max_temp_c"]
    hot = {j["name"]: j["temp"] for j in ctx.link.state()["joints"] if 15 <= j["index"] <= 28 and j["temp"] > lim}
    if hot:
        raise SkillError(f"arm motors still warm from the last task ({', '.join(f'{k} {v} C' for k, v in hot.items())}): "
                         f"wait until all are at or below {lim} C (released arms cool fastest)")


def run_skill(ctx, name, params, plan=None):
    """Run one skill now (state-checked). Arm skills get a fresh plan (or the given one) and execute it."""
    s = SKILLS.get(name)
    if s is None:
        raise SkillError(f"unknown skill '{name}'")
    params, missing = clean_params(name, params)
    if missing:
        raise SkillError(f"{name} needs {', '.join(missing)}")
    if s["kind"] != "arm":
        check_hands(name, ctx.bb["hand"])
        return s["fn"](ctx, **params)
    if plan is None:
        plan = new_plan(ctx)
    check_hands(name, plan.state["hand"])
    result = s["fn"](ctx, plan, **params)
    execute(ctx, plan)
    return result or {"steps": plan.summary()}


# ==================================================================== perception
QUALIFIERS = {"left", "right", "nearest", "closest", "farthest", "middle", "center", "centre"}


def _parse(label):
    """'right black box' -> ('black box', ['right']): OWLv2 sees the noun, code applies the position."""
    words = label.lower().split()
    quals = [w for w in words if w in QUALIFIERS]
    base = " ".join(w for w in words if w not in QUALIFIERS) or label
    return base, quals


def _hand_rects(ctx, frame, q, T):
    """Image rectangles covered by the robot's own hands (so they are not taken for objects)."""
    rects = []
    k = frame.intrinsics
    Tinv = np.linalg.inv(T)
    qn = {joints.NAMES[i] + "_joint": v for i, v in q.items()}
    for side in ("left", "right"):
        pts = []
        t = config.load()["planner_model"]["tcp"][f"{side}_tcp"]
        Th = ctx.motion.fk.pose(t["parent"], qn, base="pelvis")
        axis = np.array(t.get("axis", [1.0, 0.0, 0.0]), float)
        for xyz in (0.0 * axis, 0.12 * axis, 0.24 * axis):     # wrist, palm, fingertips
            p = Tinv[:3, :3] @ (Th[:3, 3] + Th[:3, :3] @ np.array(xyz)) + Tinv[:3, 3]     # torso frame
            c = ctx.geometry.R.T @ (p - ctx.geometry.t)                                    # camera optical frame
            if c[2] > 0.05:
                pts.append((k["fx"] * c[0] / c[2] + k["ppx"], k["fy"] * c[1] / c[2] + k["ppy"]))
        if pts:
            us, vs = zip(*pts)
            m = 40
            rects.append((min(us) - m, min(vs) - m, max(us) - min(us) + 2 * m, max(vs) - min(vs) + 2 * m))
    return rects


def _detect_all(ctx, base, hands=(), frame=None):
    frame = frame if frame is not None else ctx.camera.latest()
    if frame is None:
        raise SkillError("camera is offline")
    ok, jpg = cv2.imencode(".jpg", frame.rgb, [cv2.IMWRITE_JPEG_QUALITY, 95])
    with ctx.processing("owl"):
        dets = ctx.vision.call("detect", {"labels": [base], "image_b64": base64.b64encode(jpg).decode()})["detections"]
    rects, kept = [], []
    for d in dets:                                   # drop near-duplicate boxes (keep the higher score)
        x1, y1, x2, y2 = d["box"]
        r = (x1, y1, x2 - x1, y2 - y1)
        cxy = (x1 + x2) / 2, (y1 + y2) / 2
        if any(hx <= cxy[0] <= hx + hw and hy <= cxy[1] <= hy + hh for hx, hy, hw, hh in hands):
            continue                                 # that's the robot's own hand
        if all(_iou(r, k) < 0.5 for k in rects):
            rects.append(r)
            kept.append((r, d["score"]))
    if not kept:
        raise SkillError(f"'{base}' not visible")
    return frame, kept


def _contains(r, o, t=10):
    """Image rect o (x, y, w, h) lies wholly within rect r (pixels, tolerance t): r is a group around o."""
    return r[0] <= o[0] + t and r[1] <= o[1] + t and r[0] + r[2] >= o[0] + o[2] - t and r[1] + r[3] >= o[1] + o[3] - t


def _waist_settled(ctx):
    """Wait (up to perception.waist_settle_s) until the waist yaw is at its command and has stopped moving."""
    P, t0, last = ctx.cfg["perception"], time.monotonic(), None
    while time.monotonic() - t0 < P["waist_settle_s"]:
        s = ctx.link.state()
        meas = next(j["deg"] for j in s["joints"] if j["index"] == 12)
        cmd = (s["arms"].get("q_cmd_deg") or [meas])[0]
        if abs(meas - cmd) < P["waist_settle_deg"] and last is not None and abs(meas - last) < 0.2:
            return
        last = meas
        time.sleep(0.1)
    log.warning("waist not settled after %.1f s (yaw %.1f, command %.1f)", P["waist_settle_s"], meas, cmd)


def _look(ctx, base, precision="fine", quals=()):
    """Detect `base` in a frame of the requested precision: 'fine' (manipulation: High Accuracy preset,
    median depth over several frames) or 'coarse' (walking: one default frame). If the wanted match (the
    left/rightmost for those words, else the best) is cut off at an image edge, turn the waist toward it
    (yaw; pitch only if the waist pitch is not held) and look again. Returns (frame, detections, torso-in-pelvis)."""
    P = ctx.cfg["perception"]
    held = ctx.cfg["arms"]["waist_hold_joints"]
    check_waist_lock(ctx)                       # a look turns the waist
    for attempt in range(P["look_max_tries"] + 1):
        _waist_settled(ctx)
        frame = ctx.camera.fine() if precision == "fine" else ctx.camera.coarse()
        if frame is None:
            raise SkillError("camera is offline")
        q = ctx.motion.q_now(commanded=False)   # the camera is where the waist really is (run 13: the commanded
        T = ctx.geometry.torso_in_pelvis(ctx.motion.waist_of(q))   # angle placed both boxes 10-14 deg off)
        frame, dets = _detect_all(ctx, base, _hand_rects(ctx, frame, q, T), frame)
        singles = [d for d in dets if not any(o is not d and _contains(d[0], o[0]) for o in dets)] or dets
        centre = lambda d: d[0][0] + d[0][2] / 2
        pick = (min if "left" in quals else max)(singles, key=centre) if {"left", "right"} & set(quals) else singles[0]
        (x, y, w, h), _ = pick
        (H, W), edge = frame.rgb.shape[:2], P["look_edge_px"]
        pstep = -P["look_step_deg"] if y < edge else (P["look_step_deg"] if y + h > H - edge else 0.0)
        ystep = P["look_step_deg"] if x < edge else (-P["look_step_deg"] if x + w > W - edge else 0.0)   # image left = robot's left
        pitch, yaw = math.degrees(q[14]) + pstep, math.degrees(q[12]) + ystep
        tilt = pstep and abs(pitch) <= P["look_max_pitch_deg"] and 14 not in held
        turn = ystep and abs(yaw) <= P["look_max_yaw_deg"] and 12 not in held and precision == "fine"   # walking looks never engage the arms
        log.info("look '%s': wanted match at px %d..%d of %d (%s); waist yaw %.1f pitch %.1f (measured)",
                 base, x, x + w, W,
                 f"turning the waist {ystep:+.0f} deg" if turn and attempt < P["look_max_tries"] else "in view",
                 math.degrees(q[12]), math.degrees(q[14]))
        if attempt == P["look_max_tries"] or not (tilt or turn):
            return frame, dets, T
        ctx.motion.move_waist([math.radians(yaw) if turn else q[12], q[13], math.radians(pitch) if tilt else q[14]], 1.0)
        time.sleep(0.3)
    return frame, dets, T


def _iou(a, b):
    ax2, ay2, bx2, by2 = a[0] + a[2], a[1] + a[3], b[0] + b[2], b[1] + b[3]
    iw, ih = max(0, min(ax2, bx2) - max(a[0], b[0])), max(0, min(ay2, by2) - max(a[1], b[1]))
    inter = iw * ih
    return inter / (a[2] * a[3] + b[2] * b[3] - inter + 1e-6)


def _choose(cands, quals):
    """cands: [(fit, score)], torso frame (y left). Pick by position words, else the best score."""
    if "left" in quals:
        return max(cands, key=lambda c: c[0]["center"][1])
    if "right" in quals:
        return min(cands, key=lambda c: c[0]["center"][1])
    if quals and quals[0] in ("nearest", "closest"):
        return min(cands, key=lambda c: c[0]["distance_m"])
    if "farthest" in quals:
        return max(cands, key=lambda c: c[0]["distance_m"])
    if quals and quals[0] in ("middle", "center", "centre"):
        return min(cands, key=lambda c: abs(c[0]["center"][1]))
    return max(cands, key=lambda c: c[1])


def _surface_ahead(ctx, label, view=None):
    """No detection (the thing fills the view): the dominant horizontal surface straight ahead, from depth alone."""
    A = ctx.cfg["ai"]
    frame = ctx.camera.fine(A["surface_ahead_frames"])
    q = ctx.motion.q_now(commanded=False)
    T = ctx.geometry.torso_in_pelvis(ctx.motion.waist_of(q))
    if view is not None:
        view.update(frame=frame, T=T, how="no detection: the surface straight ahead, from depth alone")
    pts = ctx.geometry.points(frame, step=3, T=T)
    low = ctx.cfg["perception"]["floor_below_m"] + A["surface_min_above_floor_m"]
    pts = pts[(pts[:, 2] > low) & (np.abs(pts[:, 1]) < A["walk_corridor_m"] / 2) & (pts[:, 0] < 1.0)]
    top = ctx.geometry.plane_height(pts[:, 2]) if len(pts) else None
    if top is None:
        raise SkillError(f"'{label}' not visible")
    s = pts[np.abs(pts[:, 2] - top) < 0.02]
    if len(s) < A["walk_min_points"]:
        raise SkillError(f"'{label}' not visible")
    x0, x1 = np.percentile(s[:, 0], [2, 98])
    y0, y1 = np.percentile(s[:, 1], [2, 98])
    ctx.bb["surfaces"][label] = {"top_z": float(top), "front_x": float(x0), "back_x": float(x1),
                                 "y_range": [float(y0), float(y1)], "t": time.time()}
    if view is not None:
        view["surface"] = ctx.bb["surfaces"][label]
    log.info("find '%s': no detection, surface ahead from depth: top %.3f, front %.3f", label, top, x0)
    return {"surface": label, "front_edge_m": round(float(x0), 3), "height_m": round(float(top), 3), "from": "depth"}


def _surface_or_face(ctx, frame, rect, T):
    """The top surface of what is in rect. Seen from far the top of a tall thing (a case, a desk) is out of the view and the
    plane found is only the floor: then its front face is measured instead (nearest depth above the floor), with the top
    left unknown (top_z = highest point seen, 'partial'), good for walking up to it; the top is measured when it is close."""
    A = ctx.cfg["ai"]
    f = ctx.geometry.fit_surface(frame, rect, T)
    low = ctx.cfg["perception"]["floor_below_m"] + A["surface_min_above_floor_m"]
    if f is not None and f["top_z"] >= low:
        return f
    pts = ctx.geometry.points(frame, rect, step=2, T=T)
    pts = pts[pts[:, 2] > low]
    if len(pts) < A["walk_min_points"]:
        return None
    return {"top_z": float(np.percentile(pts[:, 2], 97)), "front_x": float(np.percentile(pts[:, 0], 3)),
            "back_x": float(np.percentile(pts[:, 0], 97)), "y_range": [float(v) for v in np.percentile(pts[:, 1], [2, 98])],
            "partial": True}


@skill("find", {"object": "what to look for, e.g. 'box', 'left box', 'right black box', 'table'",
                "precision": "optional: 'fine' (default, for manipulation) or 'coarse' (for walking)"},
       "Look with the camera and measure the object's position, size and distance. Words left/right/nearest/"
       "farthest pick among several matches. Needed before approach, pre_grasp or place. Positions are forgotten "
       "after walking or turning, so find again afterwards.",
       "perceive")
def find(ctx, object, precision="fine"):
    if ctx.scene is not None and precision != "coarse":    # a task scene was scanned at Generate: look it up there
        base = _parse(object)[0]
        if any(w in base for w in TARGET_WORDS):     # a big thing to place on (table, case, ...): its top plane, as the
            return ctx.scene.surface(ctx, object)   # live find does (2026-11-07: SAM 3 gave 'black case' to the white box on it)
        return ctx.scene.find(ctx, object)
    view = {}                                   # what this look saw and made of it, for the UI's robot's view panel
    try:
        return _find_live(ctx, object, precision, view)
    except SkillError as e:
        view["note"] = str(e)
        raise
    finally:
        _show_find(ctx, f"find '{object}' ({precision})", object, view)


def _show_find(ctx, title, label, view):
    """The robot's view panel (UI server only): the frame this find used, the detector's boxes, every candidate fit
    (the chosen one with its points) or the surface, and the result or the reason it failed."""
    if getattr(ctx, "view", None) is None:
        return
    try:
        frame, T = view.get("frame"), view.get("T")
        if frame is None:                        # it failed before a frame was kept: show what the camera sees now
            frame = ctx.camera.latest()
            T = ctx.geometry.torso_in_pelvis(ctx.motion.waist_of(ctx.motion.q_now(commanded=False)))
    except Exception:
        log.exception("robot view: no frame")
        return
    publish(ctx, "look", frame, T, title)
    if view.get("dets"):
        publish(ctx, "dets", view["dets"], view.get("chosen"), _parse(label)[0])
    for b, chosen in view.get("boxes", []):
        publish(ctx, "box", b, label if chosen else "", chosen)
    if view.get("surface"):
        publish(ctx, "surface", view["surface"], label)
    for line in (view.get("how"), view.get("note")):
        if line:
            publish(ctx, "note", line)


def _find_live(ctx, object, precision, view):
    """find without a task scene: one look now (the live camera)."""
    check_waist_lock(ctx)                       # the side look turns the waist
    base, quals = _parse(object)
    P = ctx.cfg["perception"]
    side = 1 if "left" in quals else (-1 if "right" in quals else 0)
    if side and 12 not in ctx.cfg["arms"]["waist_hold_joints"]:
        # look toward the named side first: the camera sees only ~ +-17 cm sideways at the table, so a box at the
        # side was cut off at the image edge (or missed) with the waist straight
        q = ctx.motion.q_now()
        yaw = math.radians(side * P["side_look_deg"])
        if abs(q[12] - yaw) > math.radians(3):
            ctx.motion.move_waist([yaw, q[13], q[14]], max(1.0, abs(math.degrees(yaw - q[12])) / ctx.cfg["arms"]["waist_speed_dps"]))
            time.sleep(0.3)
    try:
        frame, dets, T = _look(ctx, base, "coarse" if precision == "coarse" else "fine", quals)
    except SkillError as e:
        if "not visible" in str(e) and any(w in base for w in TARGET_WORDS) and ctx.bb["hand"] != "HOLDING":
            return _surface_ahead(ctx, object, view)   # it fills the view (close to a case): the detector sees no 'object'
        raise
    view.update(frame=frame, T=T, dets=dets)
    if any(w in base for w in TARGET_WORDS):
        fits = [(_surface_or_face(ctx, frame, r, T), sc, r) for r, sc in dets]
        fits = [(f, sc, r) for f, sc, r in fits if f is not None]
        if not fits:
            raise SkillError(f"saw '{object}' but could not measure its top surface")
        s, _, rect = max(fits, key=lambda c: c[1])
        view.update(chosen=rect, surface=s)
        ctx.bb["surfaces"][object] = {**s, "t": time.time()}
        return {"surface": object, "front_edge_m": round(s["front_x"], 3), "height_m": round(s["top_z"], 3)}
    M = ctx.cfg["manipulation"]
    cands = []
    for r, sc in dets:
        b = ctx.geometry.fit_box(frame, r, T)
        if b is None or max(b["dims"]) > M["max_object_size_m"]:      # e.g. the table/case itself
            continue
        if b["support_z"] < ctx.cfg["perception"]["floor_below_m"]:     # lying on the floor (a 'left box' picked
            continue                                                    # a tray on the floor 2 m away)
        b["distance_m"] = math.hypot(b["center"][0], b["center"][1])
        b["rect"] = r
        cands.append((b, sc))
    if not cands:
        raise SkillError(f"saw '{base}' but nothing of graspable size with good depth")
    # a detection around another detected one is a group of them (two boxes side by side), not one object
    # a detection containing another is a group of objects side by side (one wide box over two) - but a STACK is
    # tall, not wide: it is kept when it is no wider than what it contains (the stack, not a 56 deg stray, 2026-10-26)
    wide = lambda b, o: b["dims"][1] > 1.4 * max(o["dims"][1], 0.05)
    small = ctx.cfg["manipulation"]["min_box_width_m"]          # narrower than any graspable object: a stray or a
    cands = [(b, sc) for b, sc in cands if min(b["dims"][0], b["dims"][1]) >= small] or cands   # sliver of an edge
    # only a detection about as confident as the one around it is a second object: a weak one inside is a part of it
    # (2026-11-07: a 0.17 'white box' patch inside the 0.67 box made the box a 'group'; the patch was picked)
    weak = ctx.cfg["perception"]["group_min_score_ratio"]
    single = [(b, sc) for b, sc in cands
              if not any(o is not b and so >= weak * sc and _contains(b["rect"], o["rect"]) and wide(b, o) for o, so in cands)]
    for b, sc in cands:
        log.info("find '%s': candidate at %s, %s cm, score %.2f%s", object, [round(v, 3) for v in b["center"]],
                 [round(100 * v, 1) for v in b["dims"]], sc, "" if any(b is c for c, _ in single) else " (a group: dropped)")
    cands = single or cands
    b, score = _choose(cands, quals)
    view.update(chosen=b["rect"], boxes=[(c, c is b) for c, _ in cands])
    for k, v in ctx.bb["objects"].items():      # 'left box' and 'right box' must not be the same box
        if k != object and math.hypot(v["center"][0] - b["center"][0], v["center"][1] - b["center"][1]) < 0.05:
            raise SkillError(f"'{object}' is the same object as '{k}' (only one of them is in view): "
                             f"move them into view or closer together")
    ctx.bb["objects"][object] = {**b, "t": time.time(), "imu_yaw": _pelvis_yaw(ctx)}
    return {"object": object, "center_m": [round(v, 3) for v in b["center"]], "size_m": [round(v, 3) for v in b["dims"]],
            "distance_m": round(b["distance_m"], 3), "score": score, "candidates": len(cands)}


# ==================================================================== locomotion (fixed AI speed)
def _walk_speed(ctx, direction, distance):
    """0.4 m/s only for a confident long forward/backward walk; turning, side steps and adjusting steps go at 0.2 m/s
    (user 2026-11-03: adjusting at 0.4 shook the robot and spoiled the setup)."""
    a = ctx.cfg["ai"]
    if ctx.bb.get("fast_walk"):
        return a["speed"]
    return a["speed"] if direction in ("forward", "backward") and abs(distance) >= a["confident_walk_min_m"] else a["speed_slow"]


def _walk_time(ctx, distance, v=None, direction="forward"):
    """Seconds of command that cover `distance` on the ground at commanded speed v (tape calibration, empty hands).
    0.4 m/s: whole gait steps, forward and backward measured separately. 0.2 m/s: the measured ground speed of each
    direction (forward/backward alike; side steps left and right differ) after the start-up lag."""
    a = ctx.cfg["ai"]
    v = a["speed"] if v is None else v
    cap = ctx.cfg["locomotion"]["duration_max_s"]
    if v == a["speed"] and a.get("walk_step_m"):     # 0.4 m/s: the gait moves in whole steps (tape 2026-11-04: 0.3 s 8 cm, 0.5 s 17, 0.7 s 20, 1 s 30, 2 s 65)
        step = a["walk_step_back_m"] if direction == "backward" else a["walk_step_m"]
        n = max(1, round(abs(distance) / step))
        return min(n * a["walk_step_s"] - a["walk_step_mid_s"], cap)
    if v == a["speed_slow"]:                         # 0.2 m/s (tape 2026-11-05)
        ground = {"left": a["walk_aside_left_mps"], "right": a["walk_aside_right_mps"]}.get(direction, a["walk_ground_slow_mps"])
    else:                                            # any other speed: the 0.4 m/s fit scaled linearly
        ground = a["walk_ground_mps"] * v / a["speed"]
    return min(abs(distance) / ground + a["walk_lag_s"], cap)


def _walk_bias(ctx, direction, v):
    """Steady sideways push (m/s, + = left) that cancels the gait's drift; 0 where the walk went straight
    (tape 2026-11-04/05, empty hands)."""
    a = ctx.cfg["ai"]
    fast = v == a["speed"]
    return {("forward", True): a["walk_vy_bias_mps"], ("forward", False): a["walk_vy_bias_slow_mps"],
            ("backward", True): a["walk_vy_bias_back_mps"]}.get((direction, fast), 0.0)


def clearance_ahead(ctx, width_m=None):
    """Distance from the pelvis origin to the nearest thing in the walking corridor ahead (depth, not a detector: it sees any
    object, table or wall). The corridor is the body width, from just above the floor up past the head; with a held box the
    sphere around it is ignored. Returns metres (None: nothing within the camera's range)."""
    A = ctx.cfg["ai"]
    frame = ctx.camera.coarse()
    if frame is None:
        raise SkillError("camera is offline")
    q = ctx.motion.q_now(commanded=False)
    T = ctx.geometry.torso_in_pelvis(ctx.motion.waist_of(q))
    pts = ctx.geometry.points(frame, step=4, T=T)
    half = (width_m or A["walk_corridor_m"]) / 2
    z0 = ctx.cfg["perception"]["floor_below_m"] + A["walk_floor_clear_m"]
    keep = (pts[:, 0] > 0.05) & (np.abs(pts[:, 1]) < half) & (pts[:, 2] > z0)
    if ctx.bb["holding"]:       # the held box and the arms around it are in front of the pelvis, not obstacles
        keep &= pts[:, 0] > A["walk_held_front_m"]
    ahead = pts[keep, 0]
    if len(ahead) < A["walk_min_points"]:
        return None
    n = pts[keep]
    log.info("clearance ahead: %d points, nearest 3%% at x %.2f; nearest 20 pts mean (x %.2f y %.2f z %.2f)", len(ahead), float(np.percentile(ahead, 3)), *n[np.argsort(n[:, 0])[:20]].mean(axis=0))
    return float(np.percentile(ahead, 3))


def _walk_raw(ctx, direction, distance_m):
    """Walk now (no forgetting): the commanded 0.4 m/s for the time that covers distance_m on the ground."""
    action = {"forward": "forward", "backward": "backward", "left": "left_aside", "right": "right_aside"}.get(direction)
    if action is None or float(distance_m) <= 0:
        raise SkillError("walk needs direction forward|backward|left|right and distance_m > 0")
    v = _walk_speed(ctx, direction, float(distance_m))
    t = _walk_time(ctx, float(distance_m), v, direction)
    _before_move(ctx)
    if direction == "forward":                  # hit nothing: what is ahead must stay clear of the body after this walk
        A = ctx.cfg["ai"]
        c = clearance_ahead(ctx)
        ahead_need = A["walk_stop_clear_m"] + (A["walk_held_front_m"] if ctx.bb["holding"] else 0.0)   # the held box leads the body
        if c is not None and c - float(distance_m) < ahead_need:
            raise SkillError(f"not walking {float(distance_m):.2f} m: the nearest thing ahead is {c:.2f} m from the pelvis "
                             f"and {ahead_need:.2f} m must stay free")
    params = {"velocity": v, "duration": round(t, 2)}
    ctx.bb["moves"] = ctx.bb.get("moves", 0) + 1       # the place knows whether the robot stood still since the pick
    bias = _walk_bias(ctx, direction, v)                # forward (both speeds) and fast backward slide to the right
    if bias:
        params["vy_bias_mps"] = bias
    r = ctx.agent(action, params, timeout=t + 10)
    if r and r.get("tilt_deg"):
        raise SkillError(f"walk stopped: the body tilted {r['tilt_deg']} deg (roll, pitch); holding still")
    return t


@skill("walk", {"direction": "forward | backward | left | right", "distance_m": "metres (> 0)"},
       "Walk in a straight line at the fixed AI speed (0.4 m/s commanded); the time is calibrated so it covers "
       "distance_m on the ground. Hands empty or holding.",
       "move", hands=("EMPTY", "HOLDING"))
def walk(ctx, direction, distance_m):
    t = _walk_raw(ctx, direction, distance_m)
    ctx.forget_positions()
    _after_move(ctx)
    return {"direction": direction, "distance_m": float(distance_m), "seconds": round(t, 2)}


def _before_move(ctx):
    """Carrying: hot arm motors -> do not walk or turn; stop, hold still and say so."""
    if ctx.bb["hand"] == "HOLDING":
        try:
            _check_heat(ctx)
        except SkillError as e:
            speak(ctx, "My arms are too hot to carry this further. I am stopping.")
            raise SkillError(f"not moving while holding: {e}")


def _after_move(ctx):
    if ctx.bb["hand"] == "HOLDING":
        check_holding(ctx)                      # raises if the object slipped while walking


@skill("turn", {"direction": "left | right", "angle_deg": "positive degrees (0..180)"},
       "Turn in place by an exact angle, corrected by the IMU. Hands empty or holding.", "move",
       hands=("EMPTY", "HOLDING"))
def turn(ctx, direction, angle_deg):
    if direction not in ("left", "right") or not 0 < float(angle_deg) <= 180:
        raise SkillError("turn needs direction left|right and angle_deg in (0, 180]")
    signed = float(angle_deg) if direction == "left" else -float(angle_deg)
    _before_move(ctx)
    back = ctx.cfg["ai"]["back_before_turn_m"]
    if ctx.bb["hand"] == "HOLDING" and back > 0:       # the held box and the arms sweep round: first clear the chair/table in front
        ctx.bb["fast_walk"] = True                    # a clear 0.4 m/s step (0.2 m/s backward was not seen to move)
        try:
            walk(ctx, "backward", back)
        finally:
            ctx.bb["fast_walk"] = False
        ctx.bb["backed"] = {"m": back, "turn": direction, "deg": float(angle_deg)}
    A = ctx.cfg["ai"]
    yaw = lambda: ctx.link.state()["imu"]["pelvis_rpy"][2]
    yaw0, total = yaw(), 0.0
    ctx.bb["moves"] = ctx.bb.get("moves", 0) + 1
    for k in range(1 + A["turn_corrections"]):         # the same timed command as section 3 (turn_left / turn_right), then one IMU check
        rest = signed - total
        if k and abs(rest) <= A["turn_tolerance_deg"]:
            break
        ctx.agent("turn_left" if rest > 0 else "turn_right",
                  {"velocity": A["turn_rad_s"], "duration": round(abs(rest) / A["turn_deg_per_s"], 2)}, timeout=60)
        time.sleep(A["turn_settle_s"])
        # the IMU yaw wraps at +-180: measure the error from the target (small) instead of the total turned
        # (2026-11-05: a 180 deg left turn ended at 186, read as -174, and the correction added 354 deg more)
        total = signed + (yaw() - yaw0 - signed + 180) % 360 - 180
    r = {"target_deg": signed, "turned_deg": round(total, 1)}
    ctx.bb["last_turn"] = direction
    log.info("turn %s %s deg: turned %.1f deg", direction, angle_deg, total)
    ctx.forget_positions()
    _after_move(ctx)
    return r


@skill("approach", {"target": "a found object or surface",
                    "distance_cm": "optional: distance from the toe tip to the target when the user gave it (cm); walked blind"},
       "Walk straight to a target at 0.4 m/s until it is within arm reach (tables: front edge at the "
       "working distance; objects: centre at reach distance), side-stepping to centre objects. Finds it again "
       "between legs. Turns toward the target first if it is off to one side. Does nothing if the target is "
       "already within reach. Hands empty or holding.", "move",
       hands=("EMPTY", "HOLDING"))
def approach(ctx, target, distance_cm=None):
    # no stand-off from the LLM (2026-11-07): it invented 0.5 m for a box already in reach and the robot walked away
    # from it; the distance is always the one the arms need (table_standoff_m / box_reach_x_m)
    standoff_m = None
    a = ctx.cfg["ai"]
    if distance_cm is not None:       # a distance the user gave: walk it blind (less the margin), then look once
        blind = float(distance_cm) / 100.0 - a["blind_keep_m"]
        b = ctx.bb.pop("backed", None)
        if b and abs(b["deg"] - 90) < 15:                  # the step back before the turn put the target to one side: side-step it away
            walk(ctx, "right" if b["turn"] == "left" else "left", b["m"])
        c = clearance_ahead(ctx)               # what is already in view (the target itself, a wall): walk only up to it
        if c is not None:
            blind = min(blind, c - a["walk_stop_clear_m"] - (a["walk_held_front_m"] if ctx.bb["holding"] else 0.0))
        if blind > 0.05:
            walk(ctx, "forward", blind)
            log.info("approach: walked %.2f m blind of the %s cm given (%s)", blind, distance_cm,
                     "up to what the camera sees" if c is not None else "nothing in view")
    surface = any(w in target.lower() for w in TARGET_WORDS)
    ref = float(standoff_m) if standoff_m is not None else (a["table_standoff_m"] if surface else a["box_reach_x_m"])
    walked = 0.0
    for _ in range(6):
        for k in range(a["search_turns"] + 1):         # not in view after a turn: look around, a few small turns toward the last turn
            try:
                find(ctx, target, precision="coarse")
                break
            except SkillError as e:
                if "not visible" not in str(e) or k == a["search_turns"]:
                    raise
                log.info("approach: '%s' not in view, search turn %d/%d", target, k + 1, a["search_turns"])
                turn(ctx, ctx.bb.get("last_turn", "left"), a["search_turn_deg"])
        if surface:
            dist, lateral = ctx.bb["surfaces"][target]["front_x"] - ref, 0.0
        else:
            # an object: the waist turns to face it (STANDARDS M3.3), so its reach is the distance from the waist axis,
            # and the legs move only for what the waist cannot do (user 2026-11-07: waist before legs)
            o = ctx.bb["objects"][target]
            c = o["center"]
            radial, facing = math.hypot(c[0], c[1]), math.degrees(math.atan2(c[1], c[0]))
            lo, hi = a["stand_x_m"]
            waist_ok = abs(facing) <= a["approach_waist_ok_deg"]
            if lo <= radial <= hi and waist_ok:
                # already where a pick starts: no step (steps of a few cm are not repeatable; 8 cm asked moved 10-14)
                log.info("approach: '%s' is already within reach (%.3f m away, %+.0f deg: the waist turns to it): no walking",
                         target, radial, facing)
                find(ctx, target, precision="fine")
                return {"target": target, "walked_m": round(walked, 2), "remaining_m": round(radial - ref, 3), "in_reach": True}
            # forward/backward distance that brings it to the reach distance with the waist turned (lateral handled by the waist)
            along = math.sqrt(max(ref * ref - c[1] * c[1], 0.0)) if waist_ok else ref
            dist, lateral = min(c[0] - along, o.get("front_x", c[0]) - a["object_front_min_m"]), (0.0 if waist_ok else c[1])
            if waist_ok and dist > 0:
                room = _room_ahead(ctx)
                if room is not None and room < dist:
                    # the guard keeps 22 cm free ahead (the table): walk only what it allows, then let the pick try from there
                    if room < a["approach_min_step_m"]:
                        log.info("approach: '%s' is %.2f m away (reach %.2f-%.2f); the table allows no closer step: "
                                 "the pick tries from here", target, radial, lo, hi)
                        find(ctx, target, precision="fine")
                        return {"target": target, "walked_m": round(walked, 2), "remaining_m": round(radial - ref, 3),
                                "in_reach": False, "blocked_ahead_m": round(room, 3)}
                    log.info("approach: the guard allows %.2f of the %.2f m toward '%s'", room, dist, target)
                    walk(ctx, "forward", room)
                    walked += room
                    continue
        bearing = math.degrees(math.atan2(lateral, max(dist + ref, 0.05)))
        if abs(bearing) > a["face_turn_deg"]:
            turn(ctx, "left" if bearing > 0 else "right", abs(bearing))
            continue
        if abs(lateral) > a["lateral_tolerance_m"]:
            walk(ctx, "left" if lateral > 0 else "right", abs(lateral))
            continue
        if abs(dist) <= (a["approach_tolerance_m"] if surface else a["object_tolerance_m"]):
            find(ctx, target, precision="fine")
            return {"target": target, "walked_m": round(walked, 2), "remaining_m": round(dist, 3)}
        # several legs: never the whole way at once, measure again in between (a wrong calibration cannot run into it)
        last = abs(dist) <= 2 * a["walk_min_leg_m"]
        leg = abs(dist) if last else min(max(a["walk_leg_fraction"] * abs(dist), a["walk_min_leg_m"]), a["approach_max_leg_m"])
        last = last or leg >= abs(dist) - a["approach_tolerance_m"]
        if last and dist > 0 and ctx.bb["hand"] == "HOLDING" and ctx.bb.get("place_next") == target:
            _walk_with_prefetch(ctx, target, leg)
        else:
            try:
                walk(ctx, "forward" if dist > 0 else "backward", leg)
            except SkillError as e:
                if surface and dist > 0 and ctx.bb["hand"] == "HOLDING" and "not walking" in str(e) and dist < a["arrived_max_m"]:
                    log.info("approach: the guard stops the last %.2f m (the held box is at the target): arrived", dist)
                    find(ctx, target, precision="fine")
                    return {"target": target, "walked_m": round(walked, 2), "remaining_m": round(dist, 3), "guarded": True}
                raise
        walked += leg
    raise SkillError(f"could not get within {a['approach_tolerance_m']} m of '{target}'")


def _room_ahead(ctx):
    """How far the robot may still walk forward before the walk guard refuses (m; None: nothing in view ahead)."""
    A = ctx.cfg["ai"]
    c = clearance_ahead(ctx)
    if c is None:
        return None
    return max(0.0, c - A["walk_stop_clear_m"] - (A["walk_held_front_m"] if ctx.bb["holding"] else 0.0) - 0.005)


def stand_for_pick(ctx, target):
    """Hands empty, before a pick: look (coarse), step a little toward where the object should stand (centred within a few cm,
    stand_x_m ahead of the torso), settle and look again, a few rounds (the slow 0.2 m/s steps are not repeatable: 8 cm
    asked moved 10 and 14 cm, so every step is checked by the next look, never trusted)."""
    a = ctx.cfg["ai"]
    lo, hi = a["stand_x_m"]
    mid = (lo + hi) / 2
    moves = []
    for k in range(a["stand_rounds"] + 1):
        try:
            find(ctx, target, precision="fine")       # coarse reads the box's near face only (x about 5 cm too far)
        except SkillError as e:
            if "far side" not in str(e) or k == a["stand_rounds"]:
                raise
            walk(ctx, "forward", 0.05)                # its far edge is out of the top of the view: too far away
            moves.append(("forward", 0.05))
            time.sleep(a["stand_settle_s"])
            continue
        c = ctx.bb["objects"][target]["center"]
        side_ok, x_ok = abs(c[1]) <= a["stand_side_tol_m"], lo <= c[0] <= hi
        if (side_ok and x_ok) or k == a["stand_rounds"]:
            break
        if not side_ok:
            d, m = ("left" if c[1] > 0 else "right"), abs(c[1])
        else:
            d, m = ("forward" if c[0] > mid else "backward"), abs(c[0] - mid)
        m = max(0.03, min(0.7 * m, a["stand_step_max_m"]))
        walk(ctx, d, m)
        moves.append((d, round(m, 2)))
        log.info("stand for pick: %s %.2f m (object was at x %.2f y %.2f)", d, m, c[0], c[1])
        time.sleep(a["stand_settle_s"])
    return moves


def _walk_with_prefetch(ctx, target, leg):
    """The last leg: while the legs walk, cuRobo plans the place on the target where it WILL be (what was measured,
    moved back by the leg). The depth map is not used (the camera is moving); the plan is played only if the target
    is found within place_prefetch_tol_m of the prediction after the walk (see place_with_prefetch)."""
    saved = {k: copy.deepcopy(ctx.bb[k]) for k in ("objects", "surfaces")}
    pred = {k: copy.deepcopy(v) for k, v in saved.items()}
    for kind in ("objects", "surfaces"):
        for v in pred[kind].values():
            v["imu_yaw"] = None                                      # not turned back by the IMU while it plans
            if "center" in v:
                v["center"] = [v["center"][0] - leg, *v["center"][1:]]
            for key in ("front_x", "support_front_x", "back_x"):
                if key in v:
                    v[key] -= leg
    pf = {"target": target, "plan": None, "error": None, "pred": pred}
    ctx.bb["prefetch"] = pf
    ctx.motion.map_waist = None                       # an old map (before the walk) is not valid either

    def work():
        ctx.bb["objects"], ctx.bb["surfaces"], ctx.bb["no_map"] = pred["objects"], pred["surfaces"], True
        try:
            plan = new_plan(ctx)
            check_hands("place", plan.state["hand"])
            SKILLS["place"]["fn"](ctx, plan, on=target)
            pf["plan"] = plan
        except Exception as e:                       # planned again after the walk
            pf["error"] = str(e)
    th = threading.Thread(target=work, daemon=True, name="prefetch")
    th.start()
    try:
        _walk_raw(ctx, "forward", leg)
    finally:
        th.join()
        ctx.bb["no_map"] = False
        ctx.bb["objects"], ctx.bb["surfaces"] = {}, {}      # the walk is over: positions are in an old frame
        ctx.motion.map_waist = None
    log.info("place prefetch for '%s': %s", target, "planned during the walk" if pf["plan"] else f"not planned ({pf['error']})")
    _after_move(ctx)


def place_with_prefetch(ctx, on):
    """Place on `on`: play the plan made during the last leg if the target is where it was predicted, else plan now."""
    pf = ctx.bb.pop("prefetch", None)
    plan = None
    if pf and pf["plan"] is not None and pf["target"] == on:
        tol = ctx.cfg["ai"]["place_prefetch_tol_m"]
        now = ctx.bb["objects"].get(on) or ctx.bb["surfaces"].get(on)
        was = pf["pred"]["objects"].get(on) or pf["pred"]["surfaces"].get(on)
        if now and was:
            if "center" in now and "center" in was:
                off = math.hypot(now["center"][0] - was["center"][0], now["center"][1] - was["center"][1])
            else:
                off = abs(now["front_x"] - was["front_x"])
            dz = abs(now["top_z"] - was["top_z"])
            if off <= tol and dz <= 0.01:
                log.info("place prefetch: target %.1f cm / %.1f cm from the prediction: playing the plan made while walking",
                         100 * off, 100 * dz)
                plan = pf["plan"]
            else:
                log.info("place prefetch dropped: target %.1f cm / %.1f cm from the prediction", 100 * off, 100 * dz)
    return run_skill(ctx, "place", {"on": on}, plan=plan)


# ==================================================================== arms (dual-arm squeeze, dummy hands, no waist)
def _rz(a):
    c, s_ = math.cos(a), math.sin(a)
    return np.array([[c, -s_, 0.0], [s_, c, 0.0], [0.0, 0.0, 1.0]])


def _ry(a):
    c, s_ = math.cos(a), math.sin(a)
    return np.array([[c, 0.0, s_], [0.0, 1.0, 0.0], [-s_, 0.0, c]])


def _scene_world(ctx):
    """The support (a solid block) and every found object, for the arms' big moves near the body."""
    sup = [v for v in ctx.bb["objects"].values() if "support_z" in v]
    world = [_table(ctx, sup[0]["support_z"], sup[0].get("support_front_x", sup[0]["front_x"] - 0.10))] if sup else []
    return world + [cuboid(k, v["center"], v["dims"], v.get("yaw", 0.0)) for k, v in ctx.bb["objects"].items()]


def _via_sideways(ctx, plan, goal_deg, label, margin=None):
    """Arms up or down by the sides: first out to the sideways posture (arms raised out to the sides, away from
    everything in front), then to the goal. Lifting from hanging and lowering to hanging straight in front hit
    the case's front face (the user's photo, 2026-10-24)."""
    M = ctx.cfg["manipulation"]
    target = [math.radians(v) for v in goal_deg]
    if max(abs(plan.q[15 + k] - r) for k, r in enumerate(target)) <= math.radians(10):
        return
    world = _scene_world(ctx)
    for name, angles in (("arms out to the sides", M["side_deg"]), (label, goal_deg)):
        goal = dict(plan.q)
        goal.update({15 + k: math.radians(v) for k, v in enumerate(angles)})
        plan.to_joint(name, goal, world, margin=M["near_margin_m"] if margin is None else margin)


def _ensure_ready(ctx, plan):
    """Plans start far more reliably from the ready pose than from the arms hanging down (lifted by the sides)."""
    _via_sideways(ctx, plan, ctx.cfg["manipulation"]["ready_deg"], "arms to ready pose")


def _table(ctx, support_z, front_x, y=0.0):
    """Table/case top as a slab at its measured height and front edge (STANDARDS M1.2: no assumed rim; rails
    and other raised parts are in the depth map). The hands may come support_margin_m close to it."""
    M = ctx.cfg["manipulation"]
    c = cuboid("table", (front_x + 0.6, y, support_z - M["table_block_m"] / 2), (1.2, 1.6, M["table_block_m"]))
    c["margin"] = M["support_margin_m"]
    return c


def _support_cut(ctx, b):
    """The support's surface around an object, cut out of the depth map: it is modelled by the table slab,
    whose smaller margin lets the gloves come down low beside a short object (the map's distance field is
    1.5 cm coarse). The cut ends support_cut_above_m over the measured support: at 5 mm, the real top (2-4 mm
    higher) and depth noise left voxels in the open gap between two boxes, and the glove fingers reaching
    past the object 'collided' with them (run 13: no grasp pose at all). Taller things stay in the map;
    a low rail inside the cut is met by the touch-checked descent."""
    pad = ctx.cfg["scene_map"]["exclude_pad_m"]
    (cx, cy, _), (dx, dy, _) = b["center"], b["dims"]
    zc = b["support_z"] + ctx.cfg["scene_map"]["support_cut_above_m"] - pad   # (+ the map's exclude pad)
    return cuboid("support", (cx, cy, zc - 0.025), (dx + 0.20, dy + 0.34, 0.05), b.get("yaw", 0.0))


def _face_fractions(ctx):
    """Waist turns to try toward a target: none while the agent holds the waist yaw."""
    return [0.0] if 12 in ctx.cfg["arms"]["waist_hold_joints"] else ctx.cfg["manipulation"]["face_fractions"]


def _face(ctx, plan, x, y, what, frac=1.0, lean=0.0, extra=0.0):
    """Turn the waist (a fraction of the way, plus `extra` rad: the arms' stance beside neighbours) so (x, y) comes
    in front of the arms, leaning forward `lean` rad (from the resting pitch) when it is beyond the arms' reach."""
    lim = math.radians(ctx.cfg["manipulation"]["max_face_deg"])
    plan.waist_to(f"turn waist toward {what}" + (" (leaning)" if lean else ""),
                  [max(-lim, min(lim, math.atan2(y, x) * frac + extra)), 0.0, lean])


def _held(m, h, yaw):
    """goals(x, y, z) for the held object's frame at (x, y, z) turned `yaw`: the grip moved rigidly."""
    if h.get("grip"):
        return lambda x, y, z: m.held_goals(h["grip"](), x, y, z, yaw)
    return lambda x, y, z: _hands(x, y, z, float(np.linalg.norm(np.subtract(*m.palms(m.q_now())))) / 2, yaw, keep=True)


def _hands(cx, cy, z, half, yaw, pitch_deg=0.0, keep=False):
    """Palm goals around (cx, cy, z): palms face each other along the axis turned by `yaw`."""
    ux, uy = -math.sin(yaw), math.cos(yaw)
    q = None if keep else mat_to_quat(_rz(yaw) @ _ry(math.radians(pitch_deg)))
    return {"left_tcp": {"position": [cx + ux * half, cy + uy * half, z], "quaternion": q},
            "right_tcp": {"position": [cx - ux * half, cy - uy * half, z], "quaternion": q}}


def _reach_note(ctx, name, x, y, place=False):
    """STANDARDS M3.1 / M3.3: where an object's centre is relative to the zone where the task is tested to plan
    (None inside it). Advisory: the planner still tries outside it; a failure there names the standard."""
    M = ctx.cfg["manipulation"]
    limit = M["max_place_bearing_deg"] if place else M["max_bearing_deg"]
    r, bearing = math.hypot(x, y), math.degrees(math.atan2(y, x))
    (lo, hi), chest = M["reach_m"], M["chest_front_m"]
    if abs(bearing) > limit:
        return (f"STANDARDS M3.3: '{name}' is {abs(bearing):.0f} deg to the {'left' if y > 0 else 'right'}, beyond the "
                f"tested {limit} deg waist turn to {'place on' if place else 'pick'} it (move it, or let the robot side-step)")
    if not lo <= r <= hi:
        return (f"STANDARDS M3.1: '{name}' is {(r - chest) * 100:.0f} cm from the chest, outside the tested "
                f"{(lo - chest) * 100:.0f}-{(hi - chest) * 100:.0f} cm (move it, or let the robot step "
                f"{'backward' if r < lo else 'forward'} with the approach skill, never a fixed walk)")
    return None


_GLOVE = {}


def _glove():
    """The hand's size from the planner's own collision model (so these rules and the planner agree): thickness
    from its palm contact plane to its back (along the palm normal), and how far it reaches below the contact point
    (toward the little finger: the hand is upright at a grasp). (Named for the gloves the rubber hands wore.)"""
    if not _GLOVE:
        import yaml
        from common.fk import _rpy
        robot = yaml.safe_load(open(config.path(config.load()["services"]["planner"]["robot_yml"])))
        spheres = robot["kinematics"]["collision_spheres"]
        thick, below = [], []
        for name, t in config.tcp_frames().items():
            R = _rpy(*t["rpy"])                 # contact frame in the hand link: x fingers, z thumb side
            n = R[:, 1] * (-1.0 if name.startswith("left") else 1.0)     # palm normal, out of the palm
            c = np.array([sp["center"] for sp in spheres[t["parent"]]])
            r = np.array([sp["radius"] for sp in spheres[t["parent"]]])
            d = (c - np.array(t["xyz"])) @ n
            keep = (config.load()["planner_model"].get("glove") or {}).get("keep_thumb_m")
            beside = (c - np.array(t["xyz"])) @ R[:, 2] < keep if keep is not None else np.ones(len(c), bool)
            # the thumb that stands out over a gripped box (Inspire) is above it, not between it and a neighbour
            thick.append(float((d[beside] + r[beside]).max() - (d - r).min()))
            below.append(float(((np.array(t["xyz"]) - c) @ R[:, 2] + r).max()))
        _GLOVE.update(thickness=max(thick), below=max(below))
    return _GLOVE


def _glove_thickness():
    return _glove()["thickness"]


def _hands2(cx, cy, z, half_l, half_r, yaw, pitch_deg=0.0):
    """Palm goals like _hands, with each palm its own distance from the centre."""
    return {"left_tcp": _hands(cx, cy, z, half_l, yaw, pitch_deg)["left_tcp"],
            "right_tcp": _hands(cx, cy, z, half_r, yaw, pitch_deg)["right_tcp"]}


def _pelvis_yaw(ctx):
    """The pelvis heading (rad, IMU) or None (no robot)."""
    try:
        return math.radians(ctx.link.state()["imu"]["pelvis_rpy"][2])
    except Exception:
        return None


def _rebase_objects(ctx):
    """Found objects are stored in the pelvis frame of the moment they were seen. The robot turns a little on
    the spot as the waist swings (the balance policy steps: +3..5 deg per run, 2-3 cm at the table), so each
    is turned by how far the pelvis has turned since (IMU yaw)."""
    now = _pelvis_yaw(ctx)
    if now is None:
        return
    for k, v in ctx.bb["objects"].items():
        if v.get("imu_yaw") is None:
            continue
        d = now - v["imu_yaw"]
        if abs(d) < math.radians(0.2):
            continue
        c, s_ = math.cos(-d), math.sin(-d)
        x, y = v["center"][0], v["center"][1]
        nx, ny = c * x - s_ * y, s_ * x + c * y
        for key in ("front_x", "support_front_x"):
            if key in v:
                v[key] = v[key] + nx - x
        v["center"] = [nx, ny, *v["center"][2:]]
        v["yaw"], v["distance_m"], v["imu_yaw"] = v.get("yaw", 0.0) - d, math.hypot(nx, ny), now
        log.info("'%s': the robot turned %+.1f deg since it was seen: now at (%.3f, %.3f)", k, math.degrees(d), nx, ny)


def _obj_xy(ctx, name):
    _rebase_objects(ctx)
    return ctx.bb["objects"][name]["center"][:2]


def _grasp_geometry(ctx, object, toward=0.0):
    """Squeeze grasp of a found object: its centre, the width across the robot's left-right (the palms
    close on those two faces), the grasp height and the cuboids around it. toward: the grip moved this far toward
    the robot from the object's centre (a glove in a narrow slot keeps its bulkier wrist in front of the slot)."""
    M = ctx.cfg["manipulation"]
    _rebase_objects(ctx)
    b = ctx.bb["objects"].get(object)
    if b is None:
        raise SkillError(f"'{object}' has not been found yet")
    (cx, cy, cz), (dx, dy, dz) = b["center"], b["dims"]
    if b.get("stack_n", 1) > 1:
        raise SkillError(f"'{object}' is a stack of {b['stack_n']} objects: name the one to pick (e.g. the top one)")
    if not M["min_box_width_m"] <= dy <= M["max_box_width_m"]:
        raise SkillError(f"STANDARDS M4.3: '{object}' is {dy * 100:.1f} cm wide between its squeeze faces; the gloves "
                         f"hold {M['min_box_width_m'] * 100:.0f}-{M['max_box_width_m'] * 100:.0f} cm")
    # centre height, but the gloves' lowest point stays clear of the support: a short object is gripped with
    # the lower part of the palm, of which enough must be on the object
    zg = max(cz, b["support_z"] + _glove()["below"] + M["support_clear_m"])
    zg = max(zg, min(zg + M.get("grasp_up_m", 0.0), b["top_z"] - 0.02))   # gripped a little higher: the place keeps the
    contact = b["top_z"] - max(b["support_z"], zg - M["palm_below_m"])
    if contact < M["min_palm_contact_m"]:
        need = _glove()["below"] + M["support_clear_m"] - M["palm_below_m"] + M["min_palm_contact_m"]
        raise SkillError(f"STANDARDS M4.2: '{object}' is {dz * 100:.1f} cm tall; the gloves can grip objects at "
                         f"least {need * 100:.1f} cm tall (stand it on a taller side)")
    note = _reach_note(ctx, object, cx, cy)
    # M5.1: each glove comes straight down beside its face: in the middle of the slot when an object stands
    # beside it (the same clearance on both sides of the glove), else free_side_off_m off the face
    thick = _glove_thickness()
    off, slot = {}, set()
    for side, sign in (("left", 1), ("right", -1)):
        gap, near = None, None
        for k, v in ctx.bb["objects"].items():
            if k == object or v["top_z"] < b["support_z"] + 0.02:
                continue
            along = abs(v["center"][0] - cx) - (v["dims"][0] + dx) / 2      # gap in front of / behind it
            beside = sign * (v["center"][1] - cy) - (v["dims"][1] + dy) / 2  # gap on this side of it
            if along < 0.05 and -0.5 * v["dims"][1] < beside and (gap is None or beside < gap):
                gap, near = beside, k
        # in a slot the palm side gets the room (the real glove reaches ~2.5 cm past the model's palm plane toward
        # the object, runs 14 17 19 20); its flat back keeps slot_back_clear_m from the neighbour
        if gap is not None and gap < M["slot_min_gap_m"]:      # STANDARDS M5.1 (measured: the real glove needs it)
            raise SkillError(f"STANDARDS M5.1: only {gap * 100:.1f} cm between '{object}' and '{near}' (its {side} side); "
                             f"a glove needs {M['slot_min_gap_m'] * 100:.0f} cm to come down between them (runs 14-33: "
                             f"at 12-13 cm it touched one box or the other): move them apart")
        o = M["free_side_off_m"] if gap is None else \
            max(M["slot_clear_min_m"], min(M["slot_palm_max_m"], (gap - thick) * M["slot_palm_share"]) - M["slot_back_extra_m"])
        if o < M["slot_clear_min_m"]:
            raise SkillError(f"STANDARDS M5.1: only {max(gap, 0) * 100:.1f} cm free between '{object}' and '{near}' "
                             f"(its {side} side); a glove is {thick * 100:.1f} cm thick and needs "
                             f"{(thick + 2 * M['slot_clear_min_m']) * 100:.1f} cm: move them apart")
        off[side] = o
        if gap is not None and gap < M["slot_wrist_room_m"]:
            slot.add(near)                      # a narrow slot: planned with the slot margin (touch-checked descent)
        off[side + "_free"] = gap is None
        off[side + "_mid"] = o if gap is None else \
            max(M["slot_clear_min_m"], min(M["free_side_off_m"], (gap - thick) / 2) - M["slot_back_extra_m"])   # centred fallback
    byaw = b.get("yaw", 0.0)
    if abs(byaw) > math.radians(M["max_trusted_box_yaw_deg"]):
        byaw = 0.0                              # a large estimated angle is more likely a mis-fit than real
    others = [_slot_margin(ctx, cuboid(k, v["center"], v["dims"], v.get("yaw", 0.0)), k in slot)
              for k, v in ctx.bb["objects"].items() if k != object]
    # the lift clears the support and the objects right next to it; the place raises the object over anything
    # further away only for the last part of the way (holding high costs ~11 Nm per shoulder pitch, low ~8.5)
    gap = lambda v: max(abs(v["center"][0] - cx) - (v["dims"][0] + dx) / 2, abs(v["center"][1] - cy) - (v["dims"][1] + dy) / 2)
    tallest = max([v["top_z"] for k, v in ctx.bb["objects"].items() if k != object and gap(v) < M["lift_near_m"]],
                  default=b["support_z"])
    near_tops = [v["top_z"] for k, v in ctx.bb["objects"].items() if k != object and gap(v) < 0.25]
    z_above = max([b["top_z"]] + near_tops) + _glove()["below"] + M["above_clear_m"]
    # along the object: the hand's two contacts (wrist curve and middle fingertip) centred on the object's centre,
    # so the squeeze holds it at its centre of mass ("whenever told to lift anything, find its centre"). Run 18:
    # fingertips at the far corner put the grip 2.6 cm ahead of the centre and the box swung down in the gloves
    shift = config.load()["planner_model"]["contact_ahead_m"] - toward    # contact frame vs the grip's centre
    if M.get("grip_base_behind_m") is not None:  # flat hands: the hand's base just behind the near face, the whole
        t = config.load()["planner_model"]["tcp"]["left_tcp"]     # palm along the side (the user, 2026-10-30)
        along = float(np.dot(t["xyz"], t["axis"]) / np.linalg.norm(t["axis"]))   # contact frame from the hand base
        shift = along - dx / 2 - M["grip_base_behind_m"] - toward
    if M.get("wrist_past_face_m") is not None:   # the wrist curve touches the side, not the link behind it (user photo
        near = b.get("front_x", cx - dx / 2) - cx    # 2026-11-07: a 27 cm deep box put the wrist 2.3 cm past the near face
        cap = near + M["wrist_past_face_m"] + M["wrist_curve_to_tcp_m"]   # and the wrist pitch link pressed the corner)
        if shift > cap:
            log.info("grip on '%s': %.1f cm toward the robot so the wrist curve is %.1f cm past the near face", object,
                     100 * (shift - cap), 100 * M["wrist_past_face_m"])
            shift = cap
    gx, gy = cx + shift * math.cos(byaw), cy + shift * math.sin(byaw)
    return {"object": object, "cx": gx, "cy": gy, "cz": cz, "zg": zg, "half": dy / 2, "width": dy, "grip_shift": -shift,
            "center": list(b["center"]),
            "off_l": off["left"], "off_r": off["right"], "free_l": off["left_free"], "free_r": off["right_free"],
            "mid_l": off["left_mid"], "mid_r": off["right_mid"], "z_above": max(z_above, zg), "clear": M["release_open_m"][-1],
            "note": note,
            "dims": list(b["dims"]), "byaw": byaw, "support_z": b["support_z"], "top_z": b["top_z"],
            "support_front_x": b.get("support_front_x", b["front_x"] - 0.10),
            "box": _slot_margin(ctx, cuboid(object, b["center"], b["dims"], byaw), bool(slot)), "others": others,
            "slot": sorted(slot), "toward": toward, "cut": _support_cut(ctx, b),
            "table": _table(ctx, b["support_z"], b.get("support_front_x", b["front_x"] - 0.10), cy),
            "squeeze": min(M["squeeze_m"], max(0.0, (dy - M["min_squeeze_gap_m"]) / 2)),
            "lift": max(M["lift_m"], tallest - b["support_z"] + M["carry_clear_m"])}


def _slot_margin(ctx, c, narrow):
    """A cuboid beside a glove in a narrow slot is planned with the slot margin: the 5.0 cm glove goes into a 6 cm
    gap (the user, 2026-10-26: 6 cm is the safe cutoff), which a 1 cm margin on each side never allows."""
    if narrow:
        c["margin"] = ctx.cfg["manipulation"]["slot_margin_m"]
    return c


def _check_lines(plan, moves):
    """Plan these straight moves from the plan's current pose only to prove they are possible; returns
    {label: trajectory} so the real steps can reuse exactly what was proven."""
    mk, out = plan.mark(), {}
    try:
        for label, goals, world in moves:
            plan.line(label, goals, world)
            out[label] = plan.steps[-1]["traj"]
    finally:
        plan.restore(mk)
    return out


@skill("pre_grasp", {"object": "a found object (box)"},
       "Bring both gloves straight down beside the object to grasp height (in the middle of the space between it "
       "and a neighbour, else 2 cm off its side), not touching. The object must be within reach (the waist turns "
       "toward it). Next: grasp.", "arm", hands=("EMPTY",))
def pre_grasp(ctx, plan, object):
    M = ctx.cfg["manipulation"]
    g = _grasp_geometry(ctx, object)
    publish(ctx, "grasp", g)                    # the robot's view panel: grip centre and the two palm contacts
    last = None
    # the grip nearest the object's centre that plans (a narrow slot: toward the robot, the wrist in front of it); the
    # grip the user set comes first, then forward fallbacks (2026-10-30: base 4 cm behind the near face has no pose
    # beside a stack that stands nearer, base 1 cm does): the log says which one was used
    for toward in (M["slot_grip_toward_m"] if g["slot"] else [0.0]) + M.get("grip_forward_fallback_m", []):
        try:
            r = _pre_grasp_at(ctx, plan, object, g if not toward else _grasp_geometry(ctx, object, toward))
            if toward:
                log.info("pre_grasp '%s': grip %.0f cm %s", object, abs(100 * toward), "toward the robot (wrist in front "
                         f"of the slot beside {g['slot']})" if toward > 0 else "further forward than the set grip")
            return r
        except SkillError as e:
            last = e
    raise last


def _pre_grasp_at(ctx, plan, object, g):
    M = ctx.cfg["manipulation"]
    cx, cy, zg, half, sq = g["cx"], g["cy"], g["zg"], g["half"], g["squeeze"]
    hl, hr, za = half + g["off_l"], half + g["off_r"], g["z_above"]
    _map_scene(ctx, [g["box"]] + g["others"] + [g["cut"]])   # depth obstacle map first: even the move to ready
                                                              # avoids the case
    plan.waist_to("straighten waist", [plan.waist[0], 0.0, 0.0])
    _ensure_ready(ctx, plan)
    near = [g["table"], g["box"]] + g["others"]
    free = [g["table"]] + g["others"]           # the object itself is touched from here on
    best, start = (-1, None), plan.mark()
    # one free move (cuRobo) to above the object, then straight down beside it; each candidate (waist turn,
    # palm angle, finger pitch) must also squeeze and lift in straight lines from there
    # a free side comes down free_side_off_m off the face, or closer if the longer close-in cannot be planned
    near_l = half + 0.02 if g["free_l"] and g["off_l"] > 0.02 else hl
    near_r = half + 0.02 if g["free_r"] and g["off_r"] > 0.02 else hr
    widths = [(hl, hr)] + ([(near_l, near_r)] if (near_l, near_r) != (hl, hr) else [])
    mid = (half + g["mid_l"], half + g["mid_r"])          # a slot glove centred, if the palm-side room has no pose
    widths += [mid] if mid not in widths else []
    leans = [math.radians(d) for d in M["pick_lean_deg"]] if 14 not in ctx.cfg["arms"]["waist_hold_joints"] else [0.0]
    faces = [(f, 0.0) for f in _face_fractions(ctx)]       # (fraction of the bearing, extra yaw rad): the extra turns
    faces += [(1.0, math.radians(d)) for d in M.get("pick_face_extra_deg", [])]
    for lean, (frac, extra), palm, pitch, (hl, hr) in [(ln, f, p, t, w) for ln in leans for f in faces
                                                       for p in M["palm_align"] for t in M["grasp_pitch_deg"] for w in widths]:
        if True:
            if True:
                plan.restore(start)
                _face(ctx, plan, cx, cy, object, frac, lean, extra)   # frac 0: straight (not where a look left it)
                yaw = palm * g["byaw"] + (1 - palm) * plan.waist[0]
                stage = 0
                down, up = _hands2(cx, cy, zg, hl, hr, yaw, pitch), _hands2(cx, cy, za, hl, hr, yaw, pitch)
                try:
                    stage = _come_down(ctx, plan, up, down, near)
                    for sq_try in (sq, -M["squeeze_short_m"]):   # the closing line to the faces, else ending short of
                        try:                                    # them (far reach): the live close-in goes on by feel
                            lines = _check_lines(plan, [
                                ("squeeze", _hands(cx, cy, zg, half - sq_try, yaw, pitch), free),
                                ("lift", _hands(cx, cy, zg + g["lift"] + M["lift_sag_comp_m"], half - sq_try, yaw, pitch), free)])
                            g["squeeze"] = sq_try
                            break
                        except SkillError:
                            if sq_try != sq:
                                raise
                except SkillError as e:
                    log.debug("pre_grasp candidate waist %.0f palm %s pitch %s gloves +%.1f/+%.1f cm: stage %d: %s",
                              math.degrees(plan.waist[0]), palm, pitch, 100 * (hl - half), 100 * (hr - half), stage, e)
                    if stage > best[0]:             # report the candidate that got furthest, not the last one
                        best = (stage, f"{e} (waist {math.degrees(plan.waist[0]):.0f} deg, finger pitch {pitch} deg)")
                    continue
                g.update(yaw=yaw, pitch=pitch, lines=lines)
                # the descent runs touch-checked: from here on the gloves are at the object (a failure retreats)
                descent = plan.steps.pop()
                descent["dry"] = True
                _to_state(ctx, plan, "AT_OBJECT", grasp=g)
                plan.call("come down beside the object (touch-checked)",
                          lambda t=descent["traj"], d=down, u=up: _descend(
                              ctx, t, d, u, near, za - _glove()["below"] - M["above_clear_m"]))
                plan.steps.append(descent)
                return {"object": object, "width_m": round(g["width"], 3), "grasp_height_m": round(zg, 3),
                        "waist_deg": round(math.degrees(plan.waist[0]), 1), "palm_yaw_deg": round(math.degrees(yaw), 1),
                        "glove_clearance_cm": [round((hl - half) * 100, 1), round((hr - half) * 100, 1)]}
    plan.restore(start)
    raise SkillError(f"cannot bring the gloves beside '{object}': {best[1]}" + (f". {g['note']}" if g["note"] else ""))


@skill("grasp", {}, "Squeeze the object the gloves are beside (after pre_grasp): close in until they touch it, "
       "then press 2 cm more. Next: lift (or retreat to let go).", "arm", hands=("AT_OBJECT",))
def grasp(ctx, plan):
    g = plan.state["grasp"]
    cx, cy, zg, half, sq, yaw, pitch = (g[k] for k in ("cx", "cy", "zg", "half", "squeeze", "yaw", "pitch"))
    free = [g["table"]] + g["others"]
    grip = {"extra": 0.0, "dx": 0.0, "dy": 0.0, "gap": None, "force": 0.0,  # filled in by _press
            "lines": g.get("lines"), "object": g["object"]}
    _to_state(ctx, plan, "GRIPPING", grip=grip)
    plan.call("close until contact", lambda: _close_until_contact(ctx, cx, cy, zg, half, sq, yaw, pitch, free, grip))
    # planned only so later steps (lift) start from a squeeze pose; the call above does the real closing
    plan.line("squeeze", _hands(cx, cy, zg, half - sq, yaw, pitch), free, traj=g["lines"]["squeeze"])
    plan.steps[-1]["dry"] = True
    # ...and the plan holds the object as wide as the real grip is: the force squeeze stops the palms where the
    # object (wrists first) stops them, grip_hold_extra_m wider than its width. Planning the carry and place at
    # the modelled squeeze gap made them look reachable when they were not (runs 10-12: the far place)
    g["hold"] = None
    mk = plan.mark()
    try:
        plan.line("grip width", _hands(cx, cy, zg, half + ctx.cfg["manipulation"]["grip_hold_extra_m"] / 2, yaw, pitch),
                  free, margin=0.0)
        plan.steps[-1]["dry"] = True
        g["hold"] = half + ctx.cfg["manipulation"]["grip_hold_extra_m"] / 2
    except SkillError as e:
        plan.restore(mk)
        log.warning("grip width: %s; planning the carry at the squeeze gap", e)
    g["grip_plan"] = plan.m.grip_of(plan.q)     # how the plan holds it, until the live squeeze measures the grip
    g["grip_yaw"] = plan.m.object_frame(*plan.palms())[1]
    return {"object": g["object"], "squeeze_m": round(sq, 3)}


@skill("lift", {}, "Lift the squeezed object (after grasp): a 2 cm test lift first, then straight up clear of "
       "the other objects, then toward the chest for carrying. Next: place, turn, walk or approach.", "arm",
       hands=("GRIPPING",))
def lift(ctx, plan):
    M = ctx.cfg["manipulation"]
    g, grip = plan.state["grasp"], plan.state["grip"]
    cx, cy, cz, zg, half, sq, yaw, pitch = (g[k] for k in ("cx", "cy", "cz", "zg", "half", "squeeze", "yaw", "pitch"))
    free = [g["table"]] + g["others"]
    z_lift = zg + g["lift"]
    z_cmd = z_lift + M["lift_sag_comp_m"]       # the arms sag ~3 cm under the object; aim that much higher
    the_grip = lambda: grip.get("G") or g["grip_plan"]
    grip_yaw = lambda: grip["yaw"] if grip.get("G") else g["grip_yaw"]
    hold = lambda x, y, z: (lambda: plan.m.held_goals(the_grip(), x + grip["dx"], y + grip["dy"], z, grip_yaw()))
    plan.call("secure the grip (test lift)", lambda: _secure_grip(ctx, cx, cy, zg, half, sq, yaw, pitch, free, grip))
    up = g["others"]                            # straight up: the support it leaves is not in the way
    mk_lift = plan.mark()

    def minimum_lift():
        try:                                    # planned from the grip-width pose; else the line proven by pre_grasp
            plan.line("lift", hold(cx, cy, z_cmd)(), up, goals_fn=hold(cx, cy, z_cmd),
                      traj=None if g.get("hold") else g["lines"]["lift"])
        except SkillError as e:
            log.warning("lift at the grip width: %s; using the proven lift line", e)
            plan.line("lift", hold(cx, cy, z_cmd)(), up, goals_fn=hold(cx, cy, z_cmd), traj=g["lines"]["lift"])
    minimum_lift()
    extra_up = 0.0
    extras = sorted(M.get("lift_extra_m", []), reverse=True)
    if extras:                                  # as high as it goes without turning the arm joints any further (the object
        base_swing = _swing_deg(plan.steps[-1]["traj"])      # stays level: the held goals keep its orientation)
        for extra in extras:
            plan.restore(mk_lift)
            try:
                plan.line("lift", hold(cx, cy, z_cmd + extra)(), up, goals_fn=hold(cx, cy, z_cmd + extra))
            except SkillError as e:
                log.debug("lift +%.0f cm: %s", 100 * extra, e)
                continue
            swing = _swing_deg(plan.steps[-1]["traj"])
            if swing <= base_swing + M["lift_extra_swing_deg"]:
                extra_up = extra
                z_lift, z_cmd = z_lift + extra, z_cmd + extra
                log.info("lift: %.0f cm higher than the minimum (arm joints turn %.0f deg, minimum lift %.0f deg)",
                         100 * extra, swing, base_swing)
                break
            log.debug("lift +%.0f cm turns an arm joint %.0f deg (minimum lift %.0f): not used", 100 * extra, swing, base_swing)
        else:
            plan.restore(mk_lift)               # none qualified: the minimum lift, exactly as planned before
            minimum_lift()
    held = {"object": g["object"], "width": g["width"], "dims": g["dims"], "grasp_above_bottom": zg - g["support_z"],
            "support_z": g["support_z"], "gap_of": lambda: grip["gap"], "pitch": pitch, "clear": g["clear"], "yaw": yaw,
            "shift": g["grip_shift"], "grip": the_grip,           # the object's centre is this far ahead of the palms (along them)
            "from_xy": (g["center"][0], g["center"][1]), "moves": ctx.bb.get("moves", 0),   # where it stood (a place back there)
            "support_front_x": g["support_front_x"], "others": g["others"]}   # what stood around it (pick frame)
    _to_state(ctx, plan, "HOLDING", holding=held)
    plan.attach(cuboid("held", (g["center"][0], g["center"][1], cz + g["lift"] + extra_up + M["lift_sag_comp_m"]), g["dims"], g["byaw"]))
    # draw the object toward the chest along the torso's forward axis: shorter lever, less load (carry pose)
    wy = plan.waist[0]
    pull = min(M["pull_in_m"], max(0.0, math.hypot(cx, cy) - M["min_carry_x_m"]))
    if pull > 0.005:                            # optional: skipped if it cannot be done in a straight line
        px, py = cx - math.cos(wy) * pull, cy - math.sin(wy) * pull
        mk = plan.mark()
        try:
            plan.line("pull in to the carry pose", hold(px, py, z_cmd)(), free, goals_fn=hold(px, py, z_cmd))
            plan.steps[-1]["optional"] = True   # also skipped at run time if the live line is impossible
        except SkillError:
            plan.restore(mk)
    plan.call("check lift height", lambda: _check_height(ctx, z_lift, "the object did not come up to the lift height"))
    plan.call("check grip after lift", lambda: _check_gap(ctx, held["gap_of"](), "the object slipped during the lift"))
    plan.call("holding", lambda: ctx.bb["objects"].pop(g["object"], None))
    return {"object": g["object"], "lift_m": round(g["lift"] + extra_up, 3)}


@skill("set_down", {}, "Recovery while holding: lower the object straight down onto the surface under it "
       "(the gloves stay closed). Next: retreat.", "arm", hands=("HOLDING",))
def set_down(ctx, plan):
    M = ctx.cfg["manipulation"]
    h = plan.state["holding"]
    mid, yaw = plan.m.object_frame(*plan.palms())
    z_hands = float(sum(plan.palms()[0][2:3]) + sum(plan.palms()[1][2:3])) / 2
    floor = min(h["support_z"] + h["grasp_above_bottom"] - M["land_overrun_m"],     # never below where it came from, unless the surface
                z_hands - M["set_down_range_m"])                                   # under it is lower (a case): lower until it touches
    plan.call("set down until it sits on something", lambda: _lower_until_contact(
        ctx, lambda: mid[:2], floor, _held(ctx.motion, h, yaw), [], fast=False))
    plan.attached = None
    _to_state(ctx, plan, "GRIPPING", holding=None)
    return {"set_down": h["object"]}


@skill("retreat", {}, "Let go and step the hands back: open the gloves straight out, move them up, then the "
       "ready pose (after pre_grasp, grasp or set_down). Never drops the arms.", "arm", hands=("AT_OBJECT", "GRIPPING"))
def retreat(ctx, plan):
    M = ctx.cfg["manipulation"]
    g = plan.state.get("grasp") or {}
    left, right = plan.palms()
    mid, half = (left + right) / 2, float(np.linalg.norm(left - right)) / 2
    yaw = math.atan2(-(left - right)[0], (left - right)[1])
    world = [g["table"]] + g["others"] if g else []
    clear = list(world)
    if g:                                       # the object between the gloves, where it is now (set down elsewhere
        bottom = mid[2] - (g["zg"] - g["support_z"])         # than it was found): the hands rise clear of it
        clear.append(cuboid(g["object"], (mid[0], mid[1], bottom + g["dims"][2] / 2), g["dims"], yaw))
    plan.attached = None
    plan.call("stop squeezing", lambda: _grip_force(ctx, squeeze_n=0.0, payload_kg=0.0))
    for extra in (M["retreat_open_m"], 0.02, 0.01):  # as far as the neighbours allow
        opened = half + extra
        mk = plan.mark()
        try:
            plan.line("open the gloves", _hands(mid[0], mid[1], mid[2], opened, yaw, keep=True), world)
            break
        except SkillError:
            plan.restore(mk)
    else:                                       # pressed against a neighbour in the model: lift straight up
        opened = half                           # instead, checked against the support only
        lines = [g["table"]] if g else []
        _back_to_ready(ctx, plan, lines, mid[0], mid[1], mid[2], opened, yaw, 0.0, True, ready_world=world, clear=clear)
        _to_state(ctx, plan, "EMPTY", grasp=None, grip=None)
        return {"retreated": True, "opened": False}
    _back_to_ready(ctx, plan, world, mid[0], mid[1], mid[2], opened, yaw, 0.0, True, clear=clear)
    _to_state(ctx, plan, "EMPTY", grasp=None, grip=None)
    return {"retreated": True}


def _back_to_ready(ctx, plan, world, x, y, z, half, yaw, pitch, keep, ready_world=None, clear=None, before_waist=None):
    """From open gloves around an object: straight up as far as reachable, straight back toward the chest
    out of the object's area, then the ready pose planned around everything (the straight moves cannot
    fail to be tried; only the last free move needs a plan)."""
    M = ctx.cfg["manipulation"]
    wy = plan.waist[0]
    # first up until the gloves' lowest point clears every object top around the hands (a fixed 8 cm left the
    # gloves beside a just-placed 9 cm box, and drawing back dragged it off the stack)
    tops = [c["pose"][2] + c["dims"][2] / 2 for c in (clear or world) if c["name"] != "table"
            and math.hypot(c["pose"][0] - x, c["pose"][1] - y) < 0.25]
    need = max(tops, default=-1.0) + M["retreat_clear_m"] - (z - _glove()["below"])
    for rise in ([need] if need > 0.08 else []) + [0.08, 0.04, 0.02, 0.0]:
        mk = plan.mark()
        try:
            if rise:
                plan.line("move the hands up", _hands(x, y, z + rise, half, yaw, pitch, keep=keep), world)
            break
        except SkillError:
            plan.restore(mk)
    else:
        rise = 0.0
    goal = dict(plan.q)
    goal.update({15 + k: math.radians(v) for k, v in enumerate(M["ready_deg"])})

    def to_ready():
        for wld, margin in ((ready_world or world, M["near_margin_m"]), (ready_world or world, 0.0), (world, 0.0)):
            mk = plan.mark()
            try:
                plan.to_joint("ready pose", goal, wld, margin=margin)
                return None
            except SkillError as e:
                plan.restore(mk)
                last = e
        return last
    # from above everything straight to ready (a collision-checked joint path, else up and over); drawing the
    # open hands back first, keeping their angle, bent the wrists to their limit (92 deg): only if needed
    last = to_ready()
    if last is not None:
        for back in (0.12, 0.08, 0.04):
            mk = plan.mark()
            try:
                plan.line("draw the hands back", _hands(x - math.cos(wy) * back, y - math.sin(wy) * back, z + rise,
                                                        half, yaw, pitch, keep=keep), world)
                break
            except SkillError:
                plan.restore(mk)
        last = to_ready()
    if last is not None:
        raise last
    plan.call("relax the wrists", lambda: _relax_wrists(ctx))   # before the waist turns: a wrist pressed off its
    if before_waist:                                             # (e.g. the placed-object check, still facing it)
        plan.call(*before_waist)
    plan.waist_to("waist straight", [math.radians(v) for v in M["waist_deg"]])   # command tripped the guard (run 24)


WRISTS = (19, 20, 21, 26, 27, 28)


def _relax_wrists(ctx):
    """Waiting at ready: command the wrists where they actually are, so they stop pushing (the gloved left
    wrist yaw rests a few degrees off any command and overheated holding it)."""
    time.sleep(0.5)
    s = ctx.link.state()
    if not s["arms"].get("engaged"):
        return
    q = {str(j["index"]): round(j["deg"], 2) for j in s["joints"] if j["index"] in WRISTS}
    ctx.agent("play_upperbody", {"keyframes": [{"t": 0.5, "q": q}], "interp": "linear"}, timeout=15)


def _map_scene(ctx, exclude):
    """Depth obstacle map for this skill from a fresh frame (live robot only; offline checks plan with the
    fitted cuboids alone). exclude: every found object (each is planned around as its fitted cuboid, which
    is tighter than the map's surfaces); the map covers everything else (the support, its rim, unknowns)."""
    if ctx.bb.get("no_map"):                    # planning during a walk: the camera moves, there is no valid frame
        return None
    if ctx.scene is not None:                   # the task's map: built once from every view of the scene
        return ctx.scene.ensure_map(ctx, [_support_cut(ctx, v) for v in ctx.bb["objects"].values()])
    if ctx.camera is None or ctx.geometry is None:
        return None
    # the support around every found object is cut out too (it is the table slab, with its own small margin): the
    # whole plan is built up front and uses the LAST map for its live re-plans, so a map built for one skill (place)
    # must still let the hands rest low beside the object picked earlier (it made every live close-in fail)
    cuts = [_support_cut(ctx, v) for v in ctx.bb["objects"].values()]
    return ctx.motion.build_map(ctx.camera.fine(), ctx.geometry, list(exclude) + cuts)


def _close_until_contact(ctx, cx, cy, zg, half, sq, yaw, pitch, world, grip):
    """Close the gloves in one straight move until a hand touches (it lags its command sideways by
    contact_lag_m more than before closing), then squeeze by force (_press), which also brings the other hand
    onto the object. Closing far past the faces without contact means the object is not there."""
    M = ctx.cfg["manipulation"]
    m = ctx.motion
    axis = np.array([-math.sin(yaw), math.cos(yaw)])
    c = np.array([cx, cy])

    def state():                                # each hand's distance from the centre and its sideways lag
        lc, rc = m.palms(m.q_now(commanded=True))
        lm, rm = m.palms(m.q_now(commanded=False))
        return (float(np.dot(lc[:2] - c, axis)), float(-np.dot(rc[:2] - c, axis)),
                float(np.dot(lm[:2] - lc[:2], axis)), float(np.dot(rc[:2] - rm[:2], axis)), lm, rm)

    def goals(hl, hr, centre, z=zg, zr=None):
        left, right = _hands(centre[0], centre[1], z, hl, yaw, pitch), _hands(centre[0], centre[1], z if zr is None else zr, hr, yaw, pitch)
        return {"left_tcp": left["left_tcp"], "right_tcp": right["right_tcp"]}

    def stalled(name, palm, side):              # a hand stopped well outside the face: not the object
        out = side * float(np.dot(palm[:2] - c, axis)) - half
        if out > M["contact_early_m"]:
            raise SkillError(f"the {name} hand stopped {out * 100:.0f} cm outside the object's face: "
                             f"blocked by something else")

    time.sleep(M["close_settle_s"])
    hl, hr, base_l, base_r, lm, rm = state()
    for name, lag in (("left", base_l), ("right", base_r)):
        if lag > M["blocked_lag_m"]:
            raise SkillError(f"the {name} hand is already held {lag * 100:.0f} cm back before closing: blocked")
    # one straight close toward the faces (to close_overrun_m past them), played in short pieces and stopped
    # as soon as either hand touches; the force squeeze then closes the other hand onto the object (stepping
    # each hand with its own plan took 5-8 s with the arms out: the heat budget of the whole task)
    p = _contact_plan(m)
    # aim at the planned squeeze (pre_grasp proved that straight line); further in, the gloves nearly meet
    # and the wrists flip ("close in: the arm would flip at 100%")
    proven = (grip.get("lines") or {}).get("squeeze")       # the squeeze line pre_grasp proved, if we are on it
    if proven and max(abs(p.q[m.idx[n]] - v) for n, v in zip(proven["joint_names"], proven["positions"][0])) < math.radians(2):
        p.steps.append({"kind": "traj", "line": True, "label": "close in", "traj": proven})
    else:
        # not where the plan expected (e.g. the gloves came to rest on the support a little above the planned
        # height): close in sideways at the height the hands are at, not back down to the planned one
        # Each hand first straight up to its closing height: the planned grasp height, and at least close_lift_m
        # above where it really is (run 10-30 11:53: one hand rested 1.4 cm high, the other hung past the case's
        # edge and sagged 1.2 cm low, below the lid; closing in from there pushed it into the case's edge). Up
        # only, so planned without the support (a descent ending 1 mm low is inside its 2 mm margin: 10-29 16:47,
        # 10-30 11:51); then sideways in with the support and the neighbours checked.
        lm_, rm_ = m.palms(m.q_now(commanded=False))
        lift = M["close_lift_m"]
        zl, zr = max(float(lm_[2]), zg + M["close_raise_m"]) + lift, max(float(rm_[2]), zg + M["close_raise_m"]) + lift
        # straight up from where each palm is, its angle kept (turning it on the way up left no line in); the
        # support then stays >= close_raise_m below the hands' lowest points by construction, so both moves are
        # checked against the neighbours only (the support's block also caught the elbows in front of the case)
        up = {"left_tcp": {"position": [float(lm_[0]), float(lm_[1]), zl], "quaternion": None},
              "right_tcp": {"position": [float(rm_[0]), float(rm_[1]), zr], "quaternion": None}}
        beside, last = [w for w in world if w["name"] != "table"], None
        try:
            _line_or_free(p, "hands to the closing height", up, beside)
            _line_or_free(p, "close in", goals(half - sq, half - sq, c, zl, zr), beside)
        except SkillError as e:
            last = e
            log.warning("close in at the closing height: %s; from the planned height", e)
            p = _contact_plan(m)
            _line_or_free(p, "close in", goals(half - sq, half - sq, c, zg, zg), beside)
        if last is None:                       # the up move first (not touch-watched), then the close-in below
            m.execute(p.steps[0]["traj"], holding=True)
            p.steps = p.steps[1:]
            time.sleep(M["close_settle_s"])     # the touch baselines from where the hands are now
            hl, hr, base_l, base_r, lm, rm = state()
        log.info("close in at %.1f / %.1f cm above the planned height (hands were %+.1f / %+.1f)", 100 * (zl - zg),
                 100 * (zr - zg), 100 * (float(lm_[2]) - zg), 100 * (float(rm_[2]) - zg))
    touched, count = {}, {"left": 0, "right": 0}
    last = {"left": (hl, float(np.dot(lm[:2] - c, axis))), "right": (hr, float(-np.dot(rm[:2] - c, axis)))}

    def touch():
        """A hand touches when it lags its command sideways AND has nearly stopped closing while the command went on,
        for two pieces in a row (starting late, or dragging on the table, only slows it: that stopped a close-in
        4 cm before the object)."""
        cl, cr, lag_l, lag_r, lm_, rm_ = state()
        now = {"left": (cl, float(np.dot(lm_[:2] - c, axis))), "right": (cr, float(-np.dot(rm_[:2] - c, axis)))}
        for name, lag, palm, side in (("left", lag_l - base_l, lm_, 1), ("right", lag_r - base_r, rm_, -1)):
            closed_cmd, closed = last[name][0] - now[name][0], last[name][1] - now[name][1]
            stopped = closed_cmd > 0.003 and closed < M["descent_follow_min"] * closed_cmd
            count[name] = count[name] + 1 if lag >= M["contact_lag_m"] and stopped else 0
            log.info("close %s: lag %.1f mm, last piece in %.1f of %.1f mm", name, 1000 * lag, 1000 * closed,
                     1000 * closed_cmd)
            if count[name] >= 2:
                stalled(name, palm, side)
                touched[name] = True
        last.update(now)
        return bool(touched)
    if not m.execute(p.steps[-1]["traj"], holding=True, slowdown=M["close_slowdown"], stop=touch) and not touch():
        time.sleep(M["close_settle_s"])        # the move ended: a hand still held back from its command is against it
        _, _, lag_l, lag_r, _, _ = state()
        # the close aims only sq past the measured faces, so a hand the box stops lags at most ~sq: contact_lag_m
        # (6 mm) against a 5 mm overrun failed a good grip (practice 10-30: "perfect position, it just touched and
        # retreated" - user); half the overrun counts, and the force squeeze then checks the gap against the width
        if max(lag_l - base_l, lag_r - base_r) < min(M["contact_lag_m"], max(M["contact_lag_min_m"], 0.5 * sq)):
            raise SkillError("closed past the measured faces without touching: the object is not where the camera put it")
    return _press(ctx, cx, cy, half, sq, yaw, grip, M["squeeze_n"])


def _grip_force(ctx, squeeze_n=None, payload_kg=None):
    """Set the agent's squeeze force (N per hand, the palms pushing toward each other) and/or the held mass
    it carries in the feedforward, and wait while they ramp."""
    p = {k: v for k, v in (("squeeze_n", squeeze_n), ("payload_kg", payload_kg)) if v is not None}
    if squeeze_n == 0.0:
        ctx.motion.grip = None                  # not squeezing: holding moves no longer follow a grip
    r = ctx.agent("arm_ff", p)
    time.sleep(r.get("ramp_s", 0.0) + 0.1)
    return r


def _hold_here(ctx):
    """Move the arm position command to where the arms are: the position spring stops pushing, so only the
    feedforward (gravity, payload, squeeze) acts."""
    s = ctx.link.state()
    q = {str(j["index"]): round(j["deg"], 2) for j in s["joints"] if 15 <= j["index"] <= 28}
    ctx.agent("play_upperbody", {"keyframes": [{"t": 0.3, "q": q}], "interp": "linear",
                                 "recovery": ctx.motion.recovery}, timeout=15)


def _close_wrists(ctx, yaw=None):
    """The second half of the squeeze: with the hard wrist housings on the object, the wrist yaws turn the fingers
    in (left -, right +: 5 deg = 1.5 cm at the fingertips) so the fingertips press the far corners (the user's pick-up
    pose, Media 53). Commanded past the contact by wrist_close_deg: the wrist spring keeps them pressing."""
    M = ctx.cfg["manipulation"]
    d, toe = M["wrist_close_deg"], M.get("palm_toe_in_deg")
    if not d and (toe is None or yaw is None):  # off: the arms alone squeeze (the user, 2026-10-24)
        return
    s = ctx.link.state()
    cmd = s["arms"].get("q_cmd_deg")
    q = {str(15 + k): round(v, 2) for k, v in enumerate(cmd[3:17])} if cmd else \
        {str(j["index"]): round(j["deg"], 2) for j in s["joints"] if 15 <= j["index"] <= 28}
    if toe is not None and yaw is not None:
        # flat palms (Inspire): each hand turned to lie along the box's side, its fingertip end toe_in degrees in, so
        # the whole palm presses (run 10-30 11:59: the right hand stood 6 deg open toward the front, only the back
        # corners were pressed and the front of the box swung down - the user)
        axis = np.array([-math.sin(yaw), math.cos(yaw)])
        line = M["palm_line"]                   # one line for both hands, or one per hand ("left"/"right")
        lines = {n: line.get(n[:-4], line) for n in ("left_tcp", "right_tcp")}
        pts = ctx.motion.hand_points(ctx.motion.q_now(commanded=False),
                                     {n: [ln["heel"], ln["tip"]] for n, ln in lines.items()})
        turn = {}
        for name, joint, sign, out in (("left_tcp", "21", -1.0, 1.0), ("right_tcp", "28", 1.0, -1.0)):
            heel, tip = pts[name]
            along = float(np.linalg.norm((tip - heel)[:2]))
            opened = math.degrees(math.atan2(out * float(np.dot((tip - heel)[:2], axis)), along))   # >0: tip further out
            delta = max(-M["palm_turn_max_deg"], min(M["palm_turn_max_deg"], opened + toe))
            q[joint] = round(q[joint] + sign * delta, 2)
            turn[name[:-4]] = (round(opened, 1), round(delta, 1))
        log.info("palms along the box: (open deg, wrist turn deg) %s", turn)
    else:
        q["21"], q["28"] = q["21"] - d, q["28"] + d
    ctx.agent("play_upperbody", {"keyframes": [{"t": 0.6, "q": q}], "interp": "linear",
                                 "recovery": ctx.motion.recovery}, timeout=15)
    time.sleep(ctx.cfg["manipulation"]["press_settle_s"])


def _press(ctx, cx, cy, half, sq, yaw, grip, force):
    """Squeeze by force: the agent pushes each palm toward the other with `force`, and the position command
    follows the hands to where the object stops them, so the whole force presses the object (a position
    preload against a mis-detected contact gave no grip at all). The palm gap must then match the object."""
    M = ctx.cfg["manipulation"]
    m = ctx.motion
    _grip_force(ctx, squeeze_n=force)
    for _ in range(M["press_iters"]):
        time.sleep(M["press_settle_s"])
        _hold_here(ctx)
    time.sleep(M["press_settle_s"])
    lm, rm = m.palms(m.q_now(commanded=False))
    axis = np.array([-math.sin(yaw), math.cos(yaw)])
    gap = float(np.dot(lm[:2] - rm[:2], axis))
    lo, hi = (2 * half + t for t in M["grip_gap_tol_m"])
    log.info("squeeze %.0f N: palms %.1f cm apart, object measured %.1f cm wide", force, 100 * gap, 200 * half)
    if not lo <= gap <= hi:
        raise SkillError(f"squeezing with {force:.0f} N the palms are {gap * 100:.1f} cm apart, the object is "
                         f"{2 * half * 100:.1f} cm wide: {'nothing between them' if gap < lo else 'held apart by something'}")
    mid = (lm[:2] + rm[:2]) / 2
    shift = mid - np.array([cx, cy])
    grip.update(dx=float(shift[0]), dy=float(shift[1]), extra=(half - sq) - gap / 2,
                gap=float(np.linalg.norm(lm - rm)), force=force)
    _close_wrists(ctx, yaw)                     # ...and the fingers: squeezing is both ways (user, Media 53)
    lw, rw = m.palms(m.q_now(commanded=False))  # the slip reference AFTER the wrists close: turning the hands moves
    grip["gap"] = float(np.linalg.norm(lw - rw))   # the palm points ~3 cm closer (run 31 read that as a slip)
    qc = m.q_now(commanded=True)                # = where the hands are (the command follows them), wrists closed
    m.grip = grip["G"] = m.grip_of(qc)          # the grip, frozen: holding moves only move the object
    grip["yaw"] = m.object_frame(*m.palms(qc))[1]
    return {"squeeze_n": force, "palm_gap_m": round(gap, 3), "centre_shift_m": [round(float(v), 3) for v in shift]}


def _secure_grip(ctx, cx, cy, zg, half, squeeze, yaw, pitch, world, grip):
    """Test lift: carry the object's weight in the feedforward, raise 2 cm, hold, and check the hands came up
    without the palms closing in (a slipping object lets the squeeze force push them together). If not, put
    it back down, squeeze grip_retry_n harder and try again. Runs live on the robot."""
    M = ctx.cfg["manipulation"]
    m = ctx.motion
    lm, rm = m.palms(m.q_now(commanded=False))
    z0 = float(lm[2] + rm[2]) / 2               # where the hands really are (resting gloves sit above the plan)
    noun = (ctx.bb["objects"].get(grip.get("object") or "", {}) or {}).get("noun")
    seen = lambda: ctx.scene.held_top(ctx, noun, (lm + rm) / 2) if ctx.scene is not None and noun else None
    while True:
        top0 = seen()                           # the object's top before the test lift (camera), if it is seen
        gx, gy = cx + grip["dx"], cy + grip["dy"]
        _grip_force(ctx, payload_kg=M["payload_kg"])
        p = _contact_plan(m)
        up = [w for w in world if w["name"] != "table"]     # straight up cannot hit the support it leaves
        _line_or_free(p, "test lift", m.held_goals(grip["G"], gx, gy, z0 + M["test_lift_m"], grip["yaw"]), up,
                      free=False)
        m.execute(p.steps[-1]["traj"], holding=True)
        time.sleep(M["test_hold_s"])
        (lc, rc), (left, right) = m.palms(m.q_now(commanded=True)), m.palms(m.q_now(commanded=False))
        gap, z = float(np.linalg.norm(left - right)), float(left[2] + right[2]) / 2
        ok_gap = grip["gap"] - gap < M["test_slip_m"]   # the palms closing in at all = it starts to slip
        ok_z = z >= z0 + M["test_lift_m"] / 2    # the hands (and the object between them) really came up
        top1 = seen() if top0 is not None else None
        rose = None if top1 is None else top1 - top0
        if rose is not None:                     # the camera saw the object itself: that decides (palm points move
            ok_gap = rose >= M["test_lift_m"] / 2   # with the hands' deformation; the object rising is the grip)
        log.info("test lift at %.0f N: hands up %.1f cm (of %.1f), palms %.1f cm apart (gripped at %.1f), object rose %s",
                 grip["force"], 100 * (z - z0), 100 * M["test_lift_m"], 100 * gap, 100 * grip["gap"],
                 "unseen" if rose is None else f"{100 * rose:.1f} cm (camera)")
        if ok_gap and ok_z:
            # self-calibration: the gripping palms are centred on the box, so their midpoint (from joint
            # angles) minus the camera's box centre is the camera-to-arm offset along the squeeze axis
            mid = (left + right) / 2
            axis = np.array([-math.sin(yaw), math.cos(yaw)])
            err = float(np.dot(mid[:2] - np.array([cx, cy]), axis))
            ctx.bb["calib"] = {"offset_xy": (axis * err).tolist(), "t": time.time()}
            if grip["force"] < M["carry_squeeze_n"]:   # held: a firmer squeeze for lifting and turning with it (the
                _grip_force(ctx, squeeze_n=M["carry_squeeze_n"])   # 2 cm test passed at 12 N, the turn slid it out)
                grip["force"] = M["carry_squeeze_n"]
                time.sleep(M["press_settle_s"])     # the slip reference with the firmer squeeze: it compresses the
                lw, rw = m.palms(m.q_now(commanded=False))   # gloves further (run 35: a held, level box read as slipped)
                grip["gap"] = float(np.linalg.norm(lw - rw))
            return {"squeeze_n": grip["force"], "palm_gap_m": round(gap, 3), "camera_offset_m": round(err, 3),
                    "hands_above_command_m": round(z - float(lc[2] + rc[2]) / 2, 4)}   # >0: weight over-assumed
        if grip["force"] + M["grip_retry_n"] > M["max_squeeze_n"]:
            break
        _grip_force(ctx, payload_kg=0.0)        # back down, squeeze harder, retry
        p = _contact_plan(m)
        _line_or_free(p, "set back down", m.held_goals(grip["G"], gx, gy, z0 + 0.003, grip["yaw"]), up, free=False)
        m.execute(p.steps[-1]["traj"], holding=True)
        _press(ctx, cx, cy, half, squeeze, yaw, grip, grip["force"] + M["grip_retry_n"])
    raise SkillError(f"could not get a secure grip on the box (it slips in the test lift at {grip['force']:.0f} N)")


def _calibrated(ctx, plan, x, y):
    """Correct a camera-measured target by the offset found when the palms gripped the held box. In a
    pre-built plan the offset is only known at run time, so the place steps re-read it live."""
    if not ctx.cfg["manipulation"]["use_grasp_calibration"] or ctx.scene is not None:
        return x, y                             # task scene: the line-up measures camera to camera instead
    plan.state["use_calib"] = True
    off = ctx.bb.get("calib", {}).get("offset_xy", [0.0, 0.0])
    return x + off[0], y + off[1]


def _palm_mids(ctx):
    m = ctx.motion
    lc, rc = m.palms(m.q_now(commanded=True))
    lm, rm = m.palms(m.q_now(commanded=False))
    return (lc + rc) / 2, (lm + rm) / 2, float(np.linalg.norm(lc - rc)) / 2


def _align_over(ctx, live_xy, z, yaw, h, world, corr):
    """Closed loop: move until the MEASURED palm midpoint (= box centre) is over the target centre."""
    M = ctx.cfg["manipulation"]
    for _ in range(M["align_iters"]):
        time.sleep(0.4)
        cmd, meas, half = _palm_mids(ctx)
        err = np.array(live_xy()) - meas[:2]
        corr["xy"] = cmd[:2] - meas[:2]
        if np.linalg.norm(err) < M["align_tol_m"]:
            break
        p = _contact_plan(ctx.motion)
        try:
            _line_or_free(p, "line up", _held(ctx.motion, h, yaw)(*(cmd[:2] + err), z), world, free=False)
        except SkillError as e:                 # a refinement only: lower from here rather than fail the task
            log.warning("line up skipped (%.1f cm off): %s", 100 * float(np.linalg.norm(err)), e)
            break
        ctx.motion.execute(p.steps[-1]["traj"], holding=True)
    _, meas, _ = _palm_mids(ctx)
    return {"error_m": round(float(np.linalg.norm(np.array(live_xy()) - meas[:2])), 4),
            "correction_m": [round(float(v), 4) for v in corr["xy"]]}


def _align_by_camera(ctx, target_xy, z, yaw, h, world, corr, above_z):
    """Closed loop, camera to camera: move until the held object's centre AS THE CAMERA SEES IT is over the target's
    centre as the camera measured it. One sensor for both, so the camera-arm calibration and where the object sits
    in the grip cancel out (run 40 and 2026-10-26 16:02: placed 3-4 cm off with the arm-model line-up). Returns
    None if the camera cannot see the held object (the caller lines up with the arm model)."""
    S = ctx.cfg["scene"]
    noun = ctx.bb["objects"].get(h["object"], {}).get("noun") or h["object"]
    err = None
    for it in range(S["align_iters"] + 1):
        cmd, meas, _ = _palm_mids(ctx)
        held = ctx.scene.measure_held(ctx, noun, meas, above_z)
        if held is None:
            log.warning("line up: the camera does not see '%s' clearly; lining up with the arm model", h["object"])
            return None
        tgt = np.array(target_xy())
        err = tgt - held
        log.info("line up (camera) %d: held '%s' at (%.3f, %.3f), target (%.3f, %.3f): %.1f cm off", it, h["object"],
                 held[0], held[1], tgt[0], tgt[1], 100 * float(np.linalg.norm(err)))
        if np.linalg.norm(err) > S["align_max_m"]:
            raise SkillError(f"line up: the camera sees the held object {100 * float(np.linalg.norm(err)):.1f} cm from the "
                             f"target (more than {100 * S['align_max_m']:.0f}): not placing on a wrong measurement")
        if np.linalg.norm(err) < S["align_tol_m"] or it == S["align_iters"]:
            break
        p = _contact_plan(ctx.motion)
        try:
            _line_or_free(p, "line up", _held(ctx.motion, h, yaw)(*(cmd[:2] + err), z), world, free=False)
        except SkillError as e:
            if np.linalg.norm(err) > S["align_accept_m"]:
                raise SkillError(f"line up: {100 * float(np.linalg.norm(err)):.1f} cm off and no correction plannable: {e}")
            log.warning("line up: %.1f cm off, correction not plannable (%s): lowering from here", 100 * float(np.linalg.norm(err)), e)
            break
        ctx.motion.execute(p.steps[-1]["traj"], holding=True)
    cmd, meas, _ = _palm_mids(ctx)
    corr["cmd_xy"] = cmd[:2].copy()             # the lowering keeps the hands where the line-up left them
    return {"error_m": round(float(np.linalg.norm(err)), 4), "camera": True}


def _contact_plan(m):
    """A live plan for a contact move (closing on, lifting, lowering, letting go of an object): from the arms'
    pose now, without a held object, checked against the measured cuboids only (see ArmPlan.use_map)."""
    p = ArmPlan(m)
    p.attached = None
    p.use_map = False
    return p


def _line_or_free(p, label, goals, world, free=True):
    """A straight move, or if that is impossible from the real pose, a free collision-checked path to the
    same goal (live contact steps must not fail on one marginal line). free=False while an object is held:
    a contact plan has no held object and its world leaves out what the object lands on, so a free path can
    sweep the object through it (runs 10 and 11: the held box pushed into the target box while lowering)."""
    mk = p.mark()
    try:
        p.line(label, goals, world)
    except SkillError as e:
        if not free:
            raise
        p.restore(mk)
        log.warning("'%s': %s; using a free path", label, e)
        p.to_pose(label, goals, world, margin=p.m.cfg["manipulation"]["contact_free_margin_m"])


def _photo(ctx, tag):
    """Save the head camera's colour image now (logs/runs/<time>_<tag>.png): evidence of an unexpected touch."""
    try:
        f = ctx.camera.fine() if ctx.camera else None
        if f is not None:
            path = config.path("logs/runs") / f"{time.strftime('%Y%m%d_%H%M%S')}_{tag}.png"
            cv2.imwrite(str(path), f.rgb)
            log.info("photo: %s", path)
    except Exception as e:                      # never let a diagnostic photo break a recovery
        log.warning("photo failed: %s", e)


def _zlag(ctx):
    """How far the palms are above their command (they stop following it down once the object touches)."""
    m = ctx.motion
    (lc, rc), (lm, rm) = m.palms(m.q_now(commanded=True)), m.palms(m.q_now(commanded=False))
    return float(lm[2] + rm[2] - lc[2] - rc[2]) / 2


def _lower_until_contact(ctx, xy_fn, z_floor, goals_at, world, fast=True, proven=None):
    """Lower the held object straight down, slowly and in short pieces, until it sits on whatever is under it:
    the hands stop following their command down (land_lag_m). Then the command moves to where the hands
    stopped, so they never press down after landing (pressing on after an early landing slid the gloves down
    the box onto the one below). Fails if nothing is touched before z_floor. `proven`: the planned lowering
    (continuous by construction): played when the arms are at its start; if a lowering from where the arms are
    (e.g. after a line-up) cannot be planned, the arms go back to its start and play it instead."""
    M, m = ctx.cfg["manipulation"], ctx.motion

    def at_start():
        now = np.array([m.q_now()[m.idx[n]] for n in proven["joint_names"]])
        return np.degrees(np.abs(now - np.array(proven["positions"][0]))).max() < 2.0

    def play_proven():
        time.sleep(M["close_settle_s"])
        base = _zlag(ctx)
        if m.execute(proven, holding=True, slowdown=M["lower_slowdown"],
                     stop=lambda: _zlag(ctx) - base >= M["land_lag_m"]):
            _hold_here(ctx)
            return {"landed_lag_m": round(_zlag(ctx) - base, 4), "path": "planned"}
        return None                             # at the planned height without touching: the rest slowly

    if proven is not None and at_start():
        r = play_proven()
        if r:
            return r
        return _lower_live(ctx, xy_fn, z_floor, goals_at, world, fast=False)
    try:
        return _lower_live(ctx, xy_fn, z_floor, goals_at, world, fast)
    except SkillError as e:
        if proven is None or "without touching" in str(e) or "arm motion failed" in str(e):
            raise                               # only a lowering that cannot be PLANNED falls back (a blocked or
        log.warning("lowering from here is impossible (%s): back to the start of the planned lowering", e)  # hot arm: recover)
    p = _contact_plan(m)
    goal = dict(p.q)
    goal.update({m.idx[n]: v for n, v in zip(proven["joint_names"], proven["positions"][0])})
    turn = max(abs(math.degrees(goal[i] - p.q[i])) for i in goal)
    if turn > M["back_to_lowering_max_deg"]:    # its path is not checked for the held object: short moves only
        raise SkillError(f"lowering impossible from here and the planned lowering is {turn:.0f} deg away")
    gap0, gap = (float(np.linalg.norm(np.subtract(*m.palms(q)))) for q in (goal, p.q))
    if abs(gap - gap0) > 0.01:                  # planned at the modelled squeeze gap: it would crush the object
        raise SkillError(f"lowering impossible from here and the planned lowering holds the object "
                         f"{100 * gap0:.1f} cm wide (gripped at {100 * gap:.1f})")
    p.to_joint("back to the planned lowering", goal, world, margin=0.0)
    m.execute(p.steps[-1]["traj"], holding=True)
    return play_proven() or _lower_live(ctx, xy_fn, z_floor, goals_at, world, fast=False)


def _lower_live(ctx, xy_fn, z_floor, goals_at, world, fast=True):
    """_lower_until_contact planned from where the arms are now."""
    M, m = ctx.cfg["manipulation"], ctx.motion
    z_slow = z_floor + M["land_overrun_m"] + M["lower_fast_above_m"]   # just above where it should touch
    lc, rc = m.palms(m.q_now())
    if fast and (lc[2] + rc[2]) / 2 > z_slow + 0.005:   # the way down to there at normal speed (only when
                                                        # the surface height is known: not for a set-down)
        p = _contact_plan(m)
        _line_or_free(p, "lower", goals_at(*xy_fn(), z_slow), world, free=False)
        m.execute(p.steps[-1]["traj"], holding=True)
    time.sleep(M["close_settle_s"])
    base = _zlag(ctx)
    lc, rc = m.palms(m.q_now())
    z_now = float(lc[2] + rc[2]) / 2
    for z_end in [z_floor + k * 0.03 for k in range(8) if z_floor + k * 0.03 < z_now - 0.02] or [z_floor]:
        p = _contact_plan(m)                    # as far down as can be planned: it stops where it touches anyway
        try:
            _line_or_free(p, "lower until contact", goals_at(*xy_fn(), z_end), world, free=False)
            break
        except SkillError as e:
            last = e
    else:
        raise last
    landed = m.execute(p.steps[-1]["traj"], holding=True, slowdown=M["lower_slowdown"],
                       stop=lambda: _zlag(ctx) - base >= M["land_lag_m"])
    if not landed:
        time.sleep(M["close_settle_s"])
        landed = _zlag(ctx) - base >= M["land_lag_m"]
    lc, rc = m.palms(m.q_now(commanded=False))
    z_hands, z_contact = float(lc[2] + rc[2]) / 2, z_floor + M["land_overrun_m"]    # where the hands are / where the surface should stop them
    log.info("lowering ended: hands at z %.3f, the surface expected at z %.3f (%+.1f cm), lag %.1f mm, felt: %s",
             z_hands, z_contact, 100 * (z_hands - z_contact), 1000 * (_zlag(ctx) - base), landed)
    if not landed:
        # The planned line can end exactly where the object rests (no push beyond it, so no lag to feel): hands stopped at the
        # expected surface height (not through the whole overrun) = landed (user 2026-11-03: the box sat on the target, the run
        # failed 'without touching it'). A release from there is at most a ~1 cm drop onto the target.
        if -M["land_accept_below_m"] <= z_hands - z_contact <= M["land_accept_above_m"]:
            log.warning("lowering: no touch felt, but the hands stopped at the expected surface height: taken as landed")
        else:
            raise SkillError("lowered past where the surface should be without touching it")
    _hold_here(ctx)
    return {"landed_lag_m": round(_zlag(ctx) - base, 4)}


def _come_down(ctx, plan, up, down, world, labels=("grasp pose", "palms above the object", "come down beside the object"),
               margin=None):
    """Plan the move to above the object and the straight descent beside it, bottom-up: an arm configuration
    at the bottom (IK), the straight line from it up to the top, played in reverse. The descent is then
    continuous to the very bottom (a line planned down from whatever configuration the top pose got ran into a
    joint limit part way), and the free move (transit margin) goes to its top. Returns 2 (stages done) or raises
    a SkillError; stage numbers let pre_grasp report the candidate that got furthest."""
    M = ctx.cfg["manipulation"]
    dm = M["descent_margin_m"] if margin is None else margin    # the place asks for more (place_descent_margin_m)
    bottom = plan.ik(labels[0], down, world, margin=dm)
    probe = ArmPlan(plan.m, start=bottom)
    probe.attached = plan.attached
    probe.line(labels[2], up, world, margin=dm)
    rise = probe.steps[-1]["traj"]
    top = plan.m.end_q(bottom, rise)
    margins = (M["transit_margin_m"], M["near_margin_m"], M["min_free_margin_m"])
    for k, margin in enumerate(margins):        # cramped: a little closer, never 0 (an object close to the chest
        try:                                    # puts an elbow ~1 cm over its neighbour: run 15)
            plan.to_joint(labels[1], top, world, margin=margin)
            if k:
                log.info("'%s' planned with only %.1f cm clearance", labels[1], 100 * margin)
            break
        except SkillError:
            if k == len(margins) - 1:
                raise
    plan.line(labels[2], down, world, margin=dm, traj={**rise, "positions": rise["positions"][::-1]})
    return 2


def _descend(ctx, traj, down, up, world, top=None):
    """Come straight down beside the object, touch-checked: played in short pieces, stopped as soon as a glove
    stops following down (it met an edge: the space is tighter than the camera measured). Then it goes back
    up the same line and refuses, rather than pushing anything."""
    M, m = ctx.cfg["manipulation"], ctx.motion
    p = ArmPlan(m)
    p.attached = None
    now = np.array([p.q[m.idx[n]] for n in traj["joint_names"]])
    if np.degrees(np.abs(now - np.array(traj["positions"][0]))).max() > 2.0:   # not where planned: same line, live
        p.line("come down beside the object", down, world, margin=M["descent_margin_m"])
        traj = p.steps[-1]["traj"]
    time.sleep(M["close_settle_s"])

    def heights():                              # each palm's commanded and measured height
        (lc, rc), (lm, rm) = m.palms(m.q_now(commanded=True)), m.palms(m.q_now(commanded=False))
        return {"left": (float(lc[2]), float(lm[2])), "right": (float(rc[2]), float(rm[2]))}
    start = heights()
    base = {k: zm - zc for k, (zc, zm) in start.items()}
    last = dict(start)
    hit, count = {}, {"left": 0, "right": 0}
    below = _glove()["below"]

    def touched():
        """Blocked = the hand stopped going down for two pieces in a row: it lags its command AND hardly moved
        down while the command did (rubbing along a face only slows it), or it lags far behind. Not while its
        lowest point is still above every object top around (there is nothing to meet up there: a soft arm
        starting to move lags its command, which stopped two descents at their very start)."""
        now = heights()
        for side, (zc, zm) in now.items():
            lag = zm - zc - base[side]
            drop_cmd, drop = last[side][0] - zc, last[side][1] - zm
            log.info("descent %s: lag %.1f mm, last piece down %.1f of %.1f mm", side, 1000 * lag, 1000 * drop,
                     1000 * drop_cmd)
            stopped = drop_cmd > 0.004 and drop < M["descent_follow_min"] * drop_cmd
            near = top is None or zm - below < top + 0.01
            blocked = (lag >= M["land_lag_m"] and stopped and near) or lag >= M["descent_block_lag_m"]
            count[side] = count[side] + 1 if blocked else 0
            if count[side] >= 2:
                hit[side] = lag
        last.update(now)
        return bool(hit)
    if not m.execute(traj, slowdown=M["descent_slowdown"], stop=touched):
        lagged = {k: zm - zc - base[k] for k, (zc, zm) in heights().items()}
        if max(lagged.values()) > M["land_lag_m"] / 2:   # resting on the support: stop pressing into it
            _hold_here(ctx)
        log.info("descent done, lags at the bottom %s", {k: round(1000 * v, 1) for k, v in lagged.items()})
        return {"descended": True}
    lm, rm = m.palms(m.q_now(commanded=False))
    above = (lm[2] + rm[2]) / 2 - down["left_tcp"]["position"][2]
    contact = top - ((lm[2] + rm[2]) / 2 - M["palm_below_m"]) if top is not None else 0.0
    if above < M["land_lag_m"] * 1.5 or (above <= M["rest_high_max_m"] and contact >= M["min_palm_contact_m"]):
        # down, or resting on something just below (a rail, the case edge) with enough palm still on the
        # object: grip from here (the close-in works from where the hands are)
        _hold_here(ctx)
        log.info("descent: %s glove rests %.1f cm above the grasp height, %.1f cm of palm on the object: gripping "
                 "from here", " and ".join(sorted(hit)), 100 * above, 100 * contact)
        return {"descended": True, "touched": sorted(hit), "above_m": round(float(above), 3)}
    _hold_here(ctx)
    _photo(ctx, "descent_touch")                # what the glove met, before backing off
    back = _contact_plan(m)                     # straight back up the way it came (it may rest on an edge)
    _line_or_free(back, "back up", up, world)
    m.execute(back.steps[-1]["traj"])
    raise SkillError(f"the {' and '.join(sorted(hit))} glove met something {above * 100:.0f} cm above the grasp "
                     f"height: the space beside the object is tighter than measured")


def _take_off(ctx, world):
    """Let go of an object that rests on something: no push anywhere (command = where the hands are), no
    carried weight, no squeeze, then each glove straight outward along the line between the palms; if a
    glove meets something (the agent holds in place), a smaller opening. The moves are recovery moves, so a
    blocked first try never leaves the arms stuck. Then _back_to_ready lifts and stows them."""
    M, m = ctx.cfg["manipulation"], ctx.motion
    m.recovery, was = True, m.recovery
    try:
        _hold_here(ctx)
        _grip_force(ctx, squeeze_n=0.0, payload_kg=0.0)
        _hold_here(ctx)
        lm, rm = m.palms(m.q_now(commanded=False))
        mid, half = (lm + rm) / 2, float(np.linalg.norm(lm - rm)) / 2
        yaw = math.atan2(-(lm - rm)[0], (lm - rm)[1])
        last = None
        for extra in M["release_open_m"]:
            try:
                p = _contact_plan(m)
                _line_or_free(p, "open the gloves", _hands(mid[0], mid[1], mid[2], half + extra, yaw, keep=True), world)
                m.execute(p.steps[-1]["traj"], holding=True)
                return {"opened_m": extra}
            except SkillError as e:
                last = e
                time.sleep(0.3)
        raise SkillError(f"could not open the gloves: {last}")
    finally:
        m.recovery = was


def _turn_clear_angle(ctx, plan, h, goal):
    """The waist yaw (between now and `goal`) up to which the held object, swept around the waist axis at its
    current height, stays turn_clear_m away from every object taller than its bottom."""
    M = ctx.cfg["manipulation"]
    left, right = plan.palms()
    mid = (left + right) / 2
    bottom = mid[2] - h["grasp_above_bottom"]
    yaw0 = math.atan2(-(left - right)[0], (left - right)[1])
    held = (h["dims"][0] / 2 + M["turn_clear_m"], h["dims"][1] / 2 + M["turn_clear_m"])
    others = [v for k, v in ctx.bb["objects"].items() if k != h["object"] and v["top_z"] > bottom - M["turn_clear_m"]]
    w0 = plan.waist[0]
    n = max(1, int(abs(math.degrees(goal - w0))))
    for k in range(1, n + 1):
        d = (goal - w0) * k / n
        c = (math.cos(d) * mid[0] - math.sin(d) * mid[1], math.sin(d) * mid[0] + math.cos(d) * mid[1])
        for v in others:
            if _rects_overlap(c, held, yaw0 + d, v["center"][:2], (v["dims"][0] / 2, v["dims"][1] / 2), v.get("yaw", 0.0)):
                return w0 + (goal - w0) * (k - 1) / n
    return goal


def _rects_overlap(c1, e1, t1, c2, e2, t2):
    """Oriented rectangles (centre, half sizes, yaw) overlap (separating axis test)."""
    axes = [(math.cos(t), math.sin(t)) for t in (t1, t1 + math.pi / 2, t2, t2 + math.pi / 2)]
    d = (c2[0] - c1[0], c2[1] - c1[1])
    for nx, ny in axes:
        r = lambda e, t: e[0] * abs(math.cos(t) * nx + math.sin(t) * ny) + e[1] * abs(-math.sin(t) * nx + math.cos(t) * ny)
        if abs(d[0] * nx + d[1] * ny) > r(e1, t1) + r(e2, t2):
            return False
    return True


def _turn_carrying(ctx, plan, h, x, y, what, frac, z_high, world, lean=0.0, extra=0.0):
    """Turn the waist toward (x, y) with the object held low as far as it clears everything on the way, then
    raise it to z_high (hands' height) and finish the turn: the arms are high, and heating fastest, only for
    the last part of the way."""
    M = ctx.cfg["manipulation"]
    lim = math.radians(M["max_face_deg"])
    goal = max(-lim, min(lim, math.atan2(y, x) * frac + extra))
    w0 = plan.waist[0]
    safe = _turn_clear_angle(ctx, plan, h, goal)
    if safe == goal:
        plan.waist_to(f"turn waist toward {what}", [goal, 0.0, lean])
        return
    part = safe - math.copysign(math.radians(M["turn_margin_deg"]), goal - w0)
    if (part - w0) * (goal - w0) > 0:
        plan.waist_to(f"turn waist toward {what} (held low)", [part, 0.0, lean])
    m = ctx.motion

    def up():                                   # straight up from where the object is (the grip moved rigidly)
        mid, yaw_now = m.object_frame(*m.palms(m.q_now()))
        return _held(m, h, yaw_now)(mid[0], mid[1], z_high)
    mid, yaw_p = plan.m.object_frame(*plan.palms())
    plan.line("raise it over the objects", _held(plan.m, h, yaw_p)(mid[0], mid[1], z_high), world, goals_fn=up)
    plan.waist_to(f"turn waist toward {what}", [goal, 0.0, lean])


def _verify_placed(ctx, x, y, top_expected, dims, on):
    """The place is DONE only if the camera sees the placed object's top where it should be: the depth points within
    the object's footprint (shrunk 3 cm) reach top_expected (run 40: every step reported done, but the box had
    fallen off the target)."""
    _waist_settled(ctx)
    frame = ctx.camera.fine()
    if frame is None:
        raise SkillError("cannot check the placed object: camera offline")
    q = ctx.motion.q_now(commanded=False)
    P = ctx.geometry.points(frame, step=2, T=ctx.geometry.torso_in_pelvis(ctx.motion.waist_of(q)))
    r = max(0.03, min(dims[0], dims[1]) / 2 - 0.03)
    near = P[np.hypot(P[:, 0] - x, P[:, 1] - y) < r]
    if len(near) <= 30:                         # nothing at all in the target's column: it is outside the camera's view (a tall
        log.warning("placed-object check on '%s': the target is outside the camera's view (expected top %.3f): cannot verify, "   # stack's
                    "not counted as a failure", on, top_expected)                                                # top is above the image)
        return
    top = float(np.percentile(near[:, 2], 90))
    log.info("placed-object check on '%s': top %s, expected %.3f", on, "unseen" if top is None else f"{top:.3f}", top_expected)
    if top is None or top < top_expected - ctx.cfg["manipulation"]["placed_check_tol_m"]:
        raise SkillError(f"the object is NOT on '{on}': the camera sees its top at "
                         f"{'nothing' if top is None else f'{top:.3f} m'} where it should be {top_expected:.3f} m")
    return {"top_m": round(top, 3)}


def _check_gap(ctx, gap_ref, msg):
    """Slipped: the squeeze force pushed the palms closer together than when they gripped."""
    time.sleep(0.3)
    left, right = ctx.motion.palms(ctx.motion.q_now(commanded=False))
    gap = float(np.linalg.norm(left - right))
    if gap_ref - gap >= ctx.cfg["manipulation"]["drop_gap_m"]:
        raise SkillError(f"{msg} (palm gap {gap:.3f} m, gripped at {gap_ref:.3f} m)")
    return gap


def _check_height(ctx, z_target, msg):
    time.sleep(0.3)
    left, right = ctx.motion.palms(ctx.motion.q_now(commanded=False))
    z = float(left[2] + right[2]) / 2
    if z < z_target - ctx.cfg["manipulation"]["lift_tolerance_m"]:
        raise SkillError(f"{msg} (hands at {z:.3f} m, target {z_target:.3f} m)")
    return z


@skill("place", {"on": "'table' (or another found surface) or a found object to stack on"},
       "Put the held object down on a surface or on top of another found object, open the hands and bring the "
       "arms back to the ready pose.", "arm", hands=("HOLDING",))
def place(ctx, plan, on):
    M = ctx.cfg["manipulation"]
    h = plan.state.get("holding")
    if not h:
        raise SkillError("not holding anything")
    left, right = plan.palms()
    target_yaw = h["yaw"]
    _rebase_objects(ctx)
    if on in ctx.bb["objects"]:                       # stack on another object
        t = ctx.bb["objects"][on]
        x, y, z_s = t["center"][0], t["center"][1], t["top_z"]
        x, y = _calibrated(ctx, plan, x, y)
        back_first = False
        ox = oy = 0.0
        if t["dims"][0] > h["dims"][0] + M["place_big_target_m"]:   # a target much bigger than the box (a case, a bin): the
            nx = min(max(float(left[0] + right[0]) / 2, t["front_x"] + h["dims"][0] / 2 + 0.03), x)   # box goes near its
            room = max(t["dims"][1] / 2 - h["dims"][1] / 2 - 0.02, 0.0)                              # front edge, like a table
            ny = min(max(float(left[1] + right[1]) / 2, y - room), y + room)
            ox, oy, x, y = nx - x, ny - y, nx, ny
        note = _reach_note(ctx, on, x, y, place=True)
        target_yaw = t.get("yaw", 0.0)                # square the box to the one below it
        if abs(target_yaw) > math.radians(M["max_trusted_box_yaw_deg"]):
            target_yaw = 0.0
        world = [cuboid(on, t["center"], t["dims"], target_yaw),
                 _table(ctx, t["support_z"], t.get("support_front_x", t["front_x"] - 0.10), y)]
    else:
        note = None
        if on not in ctx.bb["surfaces"]:
            find(ctx, on)
        sf = ctx.bb["surfaces"][on]
        if sf.get("partial"):
            raise SkillError(f"only the front of '{on}' is seen (its top is out of the camera's view): get closer first")
        z_s, y = sf["top_z"], float(left[1] + right[1]) / 2
        back_first = False
        now_x = float(left[0] + right[0]) / 2 + h.get("shift", 0.0) * math.cos(h["yaw"])   # the object's centre now
        # the spots tried, first that plans: its front face place_edge_margins_m inside the front edge (the arms reach less
        # far the further in: 2026-11-07 a box lifted flush with the case edge could not be put 3 cm in)
        spots = [(max(now_x, sf["front_x"] + h["dims"][0] / 2 + m), y, z_s) for m in M["place_edge_margins_m"]]
        if h.get("from_xy") and h.get("moves") == ctx.bb.get("moves", 0) and abs(z_s - h["support_z"]) < M["place_back_dz_m"]:
            # the robot has not walked or turned since the pick and this is the surface it came from: back on the very
            # spot it was lifted from first (the lift reversed; it stood there before)
            spots.insert(0, (h["from_xy"][0], h["from_xy"][1], h["support_z"]))
            back_first = True
        spots = list(dict.fromkeys((round(a, 4), round(b, 4), c) for a, b, c in spots))
        x, y, z_s = spots[0]
        world = [_table(ctx, z_s, sf["front_x"], y)]
    if on in ctx.bb["objects"]:
        spots = [(x, y, z_s)]
    z = z_s + h["grasp_above_bottom"] + M["place_clear_m"]
    above = max(z + M["place_above_m"], float(left[2] + right[2]) / 2)  # never below the carry height before arriving
    sh = h.get("shift", 0.0)                    # the object's centre is this far ahead of the palms' midpoint

    def palms_at(px, py, th):                   # the palms' midpoint that puts the object's centre at (px, py)
        return px - sh * math.cos(th), py - sh * math.sin(th)
    mid = (left + right) / 2                    # the held box (between the palms) is cut out of the obstacle map
    map_now = lambda: _map_scene(ctx, [cuboid(h["object"], (mid[0] + sh * math.cos(h["yaw"]), mid[1] + sh * math.sin(h["yaw"]),
                                       mid[2] - h["grasp_above_bottom"] + h["dims"][2] / 2), h["dims"], h["yaw"])]
                                 + [cuboid(k, v["center"], v["dims"], v.get("yaw", 0.0)) for k, v in ctx.bb["objects"].items()])
    mapped = False
    last, base = None, plan.mark()
    leans = [math.radians(d) for d in M["place_lean_deg"]] if 14 not in ctx.cfg["arms"]["waist_hold_joints"] else [0.0]
    fracs = _face_fractions(ctx)
    if fracs != [0.0]:                          # the waist turn: any that plans (the less it turns, the further the far arm
        fracs = M.get("place_face_fractions", fracs)       # reaches across the body: right shoulder roll +10 deg, 12 Nm)
    faces = [(f, 0.0) for f in fracs] + [(1.0, math.radians(d)) for d in M.get("place_face_extra_deg", [])]
    th = target_yaw                             # squared to the target only: an unsquared box overhangs and falls

    def attempt(frac, extra, lean):
        """Turn the waist, then above the target and straight down onto it (planned bottom-up, always continuous to the
        bottom); the grip moved rigidly: the box stays level, the hands never turn on it. Raises if it does not plan."""
        plan.restore(base)
        if frac or lean or extra:
            _turn_carrying(ctx, plan, h, x, y, on, frac, above, world, lean, extra)
        px, py = palms_at(x, y, th)
        held = _held(plan.m, h, th)
        pm = M.get("place_descent_margin_m", M["descent_margin_m"])
        margins = sorted({M["descent_margin_m"], pm, (M["descent_margin_m"] + pm) / 2}, reverse=True)
        for dm in margins:                      # the most clearance that plans, else the pick's
            mk2 = plan.mark()
            try:
                _come_down(ctx, plan, held(px, py, above), held(px, py, z), world,
                           labels=("place pose", "move above the target", "lower"), margin=dm)
                if dm != margins[0]:
                    log.info("place on '%s': only %.0f mm clearance plans (%.0f mm wanted)", on, 1000 * dm, 1000 * margins[0])
                return
            except SkillError as e:
                log.debug("place candidate waist frac %.2f extra %.0f lean %.0f margin %.0f mm: %s", frac, math.degrees(extra),
                          math.degrees(lean), 1000 * dm, e)
                plan.restore(mk2)
                if dm == margins[-1]:
                    raise

    # the waist turns the object and the stack toward each other: it takes the bulk of a sideways move so the arms move
    # little. Of the turns that plan (largest first) the one with the least arm joint travel is used (a few are tried)
    scored = []
    for k, (x, y, z_s) in enumerate(spots):
        z = z_s + h["grasp_above_bottom"] + M["place_clear_m"]
        above = max(z + M["place_above_m"], float(left[2] + right[2]) / 2)
        back = k == 0 and on not in ctx.bb["objects"] and back_first
        if on not in ctx.bb["objects"]:
            world = [_table(ctx, z_s, ctx.bb["surfaces"][on]["front_x"], y)] + (list(h.get("others") or []) if back else [])
        # back on the spot it was lifted from: the gloves go down the way they came up (planned with the pick's map), so
        # the measured slab and the objects known at the pick suffice. A fresh map lacks the support cut beside the
        # object (the objects were forgotten after the pick): 2026-11-07 that spot did not plan and the box went 3 cm in
        plan.use_map = not back
        if not back and not mapped:
            map_now()
            mapped = True
        for (frac, extra), lean in [(f, ln) for f in faces for ln in leans]:
            try:
                attempt(frac, extra, lean)
            except SkillError as e:
                last = e
                log.debug("place candidate frac %.2f extra %.0f lean %.0f failed: %s", frac, math.degrees(extra), math.degrees(lean), e)
                continue
            scored.append((_arm_travel_deg(plan.steps[base[0]:]) + M.get("place_waist_cost", 0.1) * abs(math.degrees(plan.waist[0])),
                           frac, extra, lean))
            if len(scored) >= M.get("place_candidates", 1):
                break
        if not scored and k + 1 < len(spots):
            log.info("place on '%s': spot %d (%.3f, %.3f) does not plan: %s", on, k + 1, x, y, last)
        if scored:
            if len(spots) > 1:
                log.info("place on '%s': object centre to (%.3f, %.3f) (spot %d of %d%s)", on, x, y, k + 1, len(spots),
                         ": where it was lifted from" if back else "")
            break
    if not scored:
        plan.restore(base)
        raise SkillError(f"cannot bring the box above '{on}': {last}" + (f". {note}" if note else ""))
    best = min(scored, key=lambda c: c[0])
    if len(scored) > 1:
        log.info("place on '%s': waist turns %s -> least arm motion at %.0f deg of the way", on,
                 [(round(c[1], 2), round(c[0])) for c in scored], 100 * best[1])
    attempt(*best[1:])
    aligned, yaw = True, th
    keep = not aligned
    log.info("place on '%s': %s, waist %.0f deg", on, "squared to it" if aligned else "NOT squared (no plan squared)",
             math.degrees(plan.waist[0]))
    hx, hy = palms_at(x, y, yaw)                # where the palms go (x, y: where the object's centre goes)
    if on in ctx.bb["objects"]:                 # where the target is now (the robot may have turned since)
        live_xy = lambda: palms_at(*np.add(_calibrated(ctx, plan, *_obj_xy(ctx, on)), (ox, oy)), yaw)
    else:
        live_xy = lambda: (hx, hy)
    lower = plan.steps.pop()                    # the proven straight lowering: played live below
    lower["dry"] = True
    plan.call("check grip before lowering", lambda: _check_gap(ctx, h["gap_of"](), f"'{h['object']}' slipped while moving"))
    corr = {"xy": np.zeros(2)}                  # commanded - measured palm midpoint (tracking error under load)
    def line_up():
        r = None
        if ctx.scene is not None and on in ctx.bb["objects"]:   # camera to camera: held object over the target
            r = _align_by_camera(ctx, lambda: _obj_xy(ctx, on), above, yaw, h, world, corr,
                                 ctx.bb["objects"][on]["top_z"] + 0.01)
        if r is None:
            r = _align_over(ctx, live_xy, above, yaw, h, world, corr)
        log.info("line up over '%s': %.1f cm off after aligning (tracking correction %s cm)", on,
                 100 * r["error_m"], [round(100 * v, 1) for v in r["correction_m"]])
        return r
    plan.call("line up over the target", line_up)
    z_touch = z_s + h["grasp_above_bottom"]     # hands' height once the object sits on the target
    below = [w for w in world if w["name"] != on]   # the object comes down onto the target
    # ...and onto the surface slab: the touch-checked lowering may go land_overrun_m below its measured top (it stops on
    # the touch). With the slab at its measured top a box not yet touching at the planned end could not be lowered
    # further (2026-11-07: 'blocked at 44% of the line', the recovery set it down)
    drop = M["land_overrun_m"] + 0.005
    below = [{**w, "pose": [w["pose"][0], w["pose"][1], w["pose"][2] - drop, *w["pose"][3:]]} if w["name"] == "table" else w
             for w in below]
    plan.call("lower until it sits on the target", lambda: _lower_until_contact(
        ctx, lambda: corr["cmd_xy"] if "cmd_xy" in corr else np.array(live_xy()) + corr["xy"],
        z_touch - M["land_overrun_m"], _held(ctx.motion, h, yaw),
        below, proven=lower["traj"]))
    plan.steps.append(lower)                    # planned only so the steps after it start from the set-down pose
    plan.call("let go", lambda: _take_off(ctx, world))
    plan.attached = None                        # released: it stays on the target (a cuboid there from now on)
    placed = {"center": [x, y, z_s + h["dims"][2] / 2], "dims": h["dims"], "top_z": z_s + h["dims"][2],
              "support_z": z_s, "front_x": x - h["dims"][0] / 2, "distance_m": math.hypot(x, y), "yaw": yaw}
    away = world + [cuboid(h["object"], placed["center"], placed["dims"], yaw)]
    # planned for feasibility/continuity only; the gentle release above does the real opening. The opened pose
    # must clear the placed object (the move there starts from the squeeze pose, planned inside it, so the path
    # is not checked against it): the widest opening the live release would also try first
    for opened in M["release_open_m"]:
        try:
            q_open = plan.ik("open hands", _hands(hx, hy, z, h["width"] / 2 + opened, yaw, h.get("pitch", 0), keep=keep),
                             away, margin=0.0)
            plan.to_joint("open hands (check only)", q_open, world, margin=0.0)
            back_world = away
            break
        except SkillError as e:
            last = e
    else:                                       # no opened pose clear of the placed object in the model (the wrist
        back_world = world                      # beside it): open against the rest, and still rise clear of it first
        log.warning("open hands: %s; planning the release without the placed object", last)
        for opened in M["release_open_m"] + [0.0]:
            if not opened:                      # none reachable here: the live release opens as far as it can
                log.warning("open hands: no opened pose reachable; leaving the opening to the live release")
                break
            try:
                plan.to_pose("open hands (check only)", _hands(hx, hy, z, h["width"] / 2 + opened, yaw, h.get("pitch", 0),
                                                               keep=keep), world, margin=0.0)
                break
            except SkillError:
                continue
    if opened:
        plan.steps[-1]["dry"] = True
    plan.state["holding"] = None

    def done():
        ctx.bb["holding"] = None
        ctx.bb["objects"][h["object"]] = {**placed, "t": time.time(), "imu_yaw": _pelvis_yaw(ctx)}
    plan.call("released", done)
    _to_state(ctx, plan, "EMPTY", grasp=None, grip=None, holding=None)   # empty from here (a later stop: no set_down)
    plan.call("relax the wrists", lambda: _relax_wrists(ctx))           # stop pushing at once (run 37: left wrist 93 C)
    _back_to_ready(ctx, plan, back_world, hx, hy, z, h["width"] / 2 + opened, yaw, h.get("pitch", 0), keep, clear=away,
                   before_waist=("check the object is on the target",
                                 lambda: _verify_placed(ctx, x, y, z_s + h["dims"][2], h["dims"], on)))
    plan.use_map = True
    return {"placed": h["object"], "on": on, "height_m": round(z_s, 3)}


BACK = "where it was lifted from"


@skill("put_back", {}, "Put the held object back down on the very spot it was lifted from ('put it back', 'same place'), "
       "open the hands and bring the arms back to the ready pose. Needs no find; only while the robot has not walked or "
       "turned since the pick.", "arm", hands=("HOLDING",))
def put_back(ctx, plan):
    h = plan.state.get("holding")
    if not h or not h.get("from_xy"):
        raise SkillError("not holding anything lifted in this task")
    if h.get("moves") != ctx.bb.get("moves", 0):
        raise SkillError("the robot walked or turned since the pick: the spot it came from is behind it now; place it on a "
                         "found surface instead")
    ctx.bb["surfaces"][BACK] = {"top_z": h["support_z"], "front_x": h["support_front_x"], "t": time.time()}
    try:
        return place(ctx, plan, BACK)          # the spot it came from is tried first (robot not moved, same surface)
    finally:
        ctx.bb["surfaces"].pop(BACK, None)


@skill("ready", {}, "Move both arms to the ready pose (hands at chest height, out of the camera view). Hands "
                    "must be empty.", "arm", hands=("EMPTY",))
def ready(ctx, plan):
    if plan.state.get("holding"):
        raise SkillError("holding a box: place it first")
    M = ctx.cfg["manipulation"]
    plan.call("no squeeze force", lambda: _grip_force(ctx, squeeze_n=0.0, payload_kg=0.0))
    goal = dict(plan.q)
    goal.update({15 + k: math.radians(v) for k, v in enumerate(M["ready_deg"])})
    plan.to_joint("ready pose", goal, [])
    plan.call("relax the wrists", lambda: _relax_wrists(ctx))   # before the waist turns: a wrist pressed off its
    plan.waist_to("waist straight", [math.radians(v) for v in M["waist_deg"]])   # command tripped the guard (run 24)
    return {"ready": True}


# ==================================================================== checks and basics
@skill("check_holding", {}, "Verify the held box has not slipped (fails if it dropped).", "internal")
def check_holding(ctx):
    h = ctx.bb["holding"]
    if not h:
        return {"holding": None}
    gap = _check_gap(ctx, h["gap_of"](), f"'{h['object']}' slipped out of the hands")
    return {"holding": h["object"], "palm_gap_m": round(gap, 3)}


@skill("speak", {"text": "what to say (max 200 characters)"}, "Say something through the robot speaker.", "basic")
def speak(ctx, text):
    return ctx.agent("speak", {"text": str(text)[:200]})


@skill("headlight", {"hex": "colour like #00FF00"}, "Set the head light colour.", "basic")
def headlight(ctx, hex):
    return ctx.agent("headlight", {"hex": hex})


@skill("wait", {"seconds": "0..30"}, "Pause.", "basic")
def wait(ctx, seconds):
    time.sleep(min(max(float(seconds), 0.0), 30.0))
    return {"waited_s": float(seconds)}


@skill("clearance", {}, "Distance to the nearest thing in the walking corridor ahead (depth).", "internal")
def clearance(ctx):
    c = clearance_ahead(ctx)
    return {"clearance_m": None if c is None else round(c, 3)}
