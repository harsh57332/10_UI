"""Skill library shared by Motion planning (section 6) and AI control (section 7).

The LLM only chooses skills and their parameters; the poses come from OWLv2 + depth + cuRobo.
Arm skills append steps to an ArmPlan (so section 6 can plan everything first and play later);
the other skills run immediately. Positions are in the pelvis frame (x forward, y left, z up), which
does not move when the waist turns, so measurements stay valid until the robot walks or turns.
"""
import base64
import logging
import math
import time

import cv2
import numpy as np

from common import config, joints

from common.fk import mat_to_quat

from .motion import ArmPlan, SkillError, cuboid, pose

log = logging.getLogger("skills")
SURFACE_WORDS = ("table", "desk", "shelf", "counter", "bench", "stand", "cabinet")
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
    try:
        plan.run(on_step=step)
    except Exception as e:
        recovered = recover(ctx)
        raise SkillError(f"{e} -> {recovered}")


def recover(ctx):
    """Bring the hands to EMPTY the safe way for their current state. Returns what was done."""
    done = []
    try:
        for name in {"HOLDING": ("set_down", "retreat"), "GRIPPING": ("retreat",),
                     "AT_OBJECT": ("retreat",)}.get(ctx.bb["hand"], ()):
            plan = new_plan(ctx)
            SKILLS[name]["fn"](ctx, plan)
            plan.run()
            done.append(name)
        return f"recovered: {', '.join(done) or 'nothing to do'}"
    except Exception as e:
        log.error("recovery failed after %s: %s", done, e)
        return f"RECOVERY FAILED after {done}: {e} (arms still held, not dropped)"


def run_skill(ctx, name, params, plan=None):
    """Run one skill now (state-checked). Arm skills get a fresh plan (or the given one) and execute it."""
    s = SKILLS.get(name)
    if s is None:
        raise SkillError(f"unknown skill '{name}'")
    params = dict(params or {})
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
        Th = ctx.motion.fk.pose(f"{side}_rubber_hand", qn, base="pelvis")
        for xyz in ([0.0, 0.0, 0.0], [0.07, 0.0, 0.0], [0.13, 0.0, 0.0]):     # wrist, palm, fingertips
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


def _look(ctx, base, precision="fine"):
    """Detect `base` in a frame of the requested precision: 'fine' (manipulation: High Accuracy preset,
    median depth over several frames) or 'coarse' (walking: one default frame). If the best match is cut
    off at the top/bottom of the image, tilt the waist (pitch) to bring it into view and look again.
    Returns (frame, detections, torso-in-pelvis)."""
    P = ctx.cfg["perception"]
    for attempt in range(P["look_max_tries"] + 1):
        frame = ctx.camera.fine() if precision == "fine" else ctx.camera.coarse()
        if frame is None:
            raise SkillError("camera is offline")
        q = ctx.motion.q_now()
        T = ctx.geometry.torso_in_pelvis(ctx.motion.waist_of(q))
        frame, dets = _detect_all(ctx, base, _hand_rects(ctx, frame, q, T), frame)
        (x, y, w, h), _ = dets[0]
        H, edge = frame.rgb.shape[0], P["look_edge_px"]
        step = -P["look_step_deg"] if y < edge else (P["look_step_deg"] if y + h > H - edge else 0.0)
        pitch = math.degrees(q[14]) + step
        if not step or abs(pitch) > P["look_max_pitch_deg"] or attempt == P["look_max_tries"]:
            return frame, dets, T
        ctx.motion.move_waist([q[12], q[13], math.radians(pitch)], 1.0)    # lean back / forward to see it
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


@skill("find", {"object": "what to look for, e.g. 'box', 'left box', 'right black box', 'table'",
                "precision": "optional: 'fine' (default, for manipulation) or 'coarse' (for walking)"},
       "Look with the camera and measure the object's position, size and distance. Words left/right/nearest/"
       "farthest pick among several matches. Needed before approach, pre_grasp or place. Positions are forgotten "
       "after walking or turning, so find again afterwards.",
       "perceive")
def find(ctx, object, precision="fine"):
    base, quals = _parse(object)
    frame, dets, T = _look(ctx, base, "coarse" if precision == "coarse" else "fine")
    if any(w in base for w in SURFACE_WORDS):
        fits = [(ctx.geometry.fit_surface(frame, r, T), sc) for r, sc in dets]
        fits = [(f, sc) for f, sc in fits if f is not None]
        if not fits:
            raise SkillError(f"saw '{object}' but could not measure its top surface")
        s, _ = max(fits, key=lambda c: c[1])
        ctx.bb["surfaces"][object] = {**s, "t": time.time()}
        return {"surface": object, "front_edge_m": round(s["front_x"], 3), "height_m": round(s["top_z"], 3)}
    M = ctx.cfg["manipulation"]
    cands = []
    for r, sc in dets:
        b = ctx.geometry.fit_box(frame, r, T)
        if b is None or max(b["dims"]) > M["max_object_size_m"]:      # e.g. the table/case itself
            continue
        b["distance_m"] = math.hypot(b["center"][0], b["center"][1])
        b["rect"] = r
        cands.append((b, sc))
    if not cands:
        raise SkillError(f"saw '{base}' but nothing of graspable size with good depth")
    b, score = _choose(cands, quals)
    ctx.bb["objects"][object] = {**b, "t": time.time()}
    return {"object": object, "center_m": [round(v, 3) for v in b["center"]], "size_m": [round(v, 3) for v in b["dims"]],
            "distance_m": round(b["distance_m"], 3), "score": score, "candidates": len(cands)}


# ==================================================================== locomotion (fixed AI speed)
def _walk_time(ctx, distance):
    return min(abs(distance) / ctx.cfg["ai"]["speed"], ctx.cfg["locomotion"]["duration_max_s"])


@skill("walk", {"direction": "forward | backward | left | right", "distance_m": "metres (> 0)"},
       "Walk in a straight line at the fixed AI speed (0.4 m/s); time = distance / 0.4. Hands empty or holding.",
       "move", hands=("EMPTY", "HOLDING"))
def walk(ctx, direction, distance_m):
    action = {"forward": "forward", "backward": "backward", "left": "left_aside", "right": "right_aside"}.get(direction)
    if action is None or float(distance_m) <= 0:
        raise SkillError("walk needs direction forward|backward|left|right and distance_m > 0")
    t = _walk_time(ctx, float(distance_m))
    _before_move(ctx)
    ctx.agent(action, {"velocity": ctx.cfg["ai"]["speed"], "duration": round(t, 2)}, timeout=t + 10)
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
    r = ctx.agent("turn_by", {"angle_deg": signed}, timeout=60)
    ctx.forget_positions()
    _after_move(ctx)
    return r


@skill("approach", {"target": "a found object or surface", "standoff_m": "optional: final distance (m)"},
       "Walk straight to a target at 0.4 m/s until it is within arm reach (tables: front edge at "
       "standoff; objects: centre at reach distance), side-stepping to centre objects. Finds it again "
       "between legs. Turns toward the target first if it is off to one side. Hands empty or holding.", "move",
       hands=("EMPTY", "HOLDING"))
def approach(ctx, target, standoff_m=None):
    a = ctx.cfg["ai"]
    surface = any(w in target.lower() for w in SURFACE_WORDS)
    ref = float(standoff_m) if standoff_m is not None else (a["table_standoff_m"] if surface else a["box_reach_x_m"])
    walked = 0.0
    for _ in range(6):
        find(ctx, target, precision="coarse")
        if surface:
            dist, lateral = ctx.bb["surfaces"][target]["front_x"] - ref, 0.0
        else:
            c = ctx.bb["objects"][target]["center"]
            dist, lateral = c[0] - ref, c[1]
        bearing = math.degrees(math.atan2(lateral, max(dist + ref, 0.05)))
        if abs(bearing) > a["face_turn_deg"]:
            turn(ctx, "left" if bearing > 0 else "right", abs(bearing))
            continue
        if abs(lateral) > a["lateral_tolerance_m"]:
            walk(ctx, "left" if lateral > 0 else "right", abs(lateral))
            continue
        if abs(dist) <= a["approach_tolerance_m"]:
            find(ctx, target, precision="fine")
            return {"target": target, "walked_m": round(walked, 2), "remaining_m": round(dist, 3)}
        leg = min(abs(dist), a["approach_max_leg_m"])
        walk(ctx, "forward" if dist > 0 else "backward", leg)
        walked += leg
    raise SkillError(f"could not get within {a['approach_tolerance_m']} m of '{target}'")


# ==================================================================== arms (dual-arm squeeze, dummy hands, no waist)
def _rz(a):
    c, s_ = math.cos(a), math.sin(a)
    return np.array([[c, -s_, 0.0], [s_, c, 0.0], [0.0, 0.0, 1.0]])


def _ry(a):
    c, s_ = math.cos(a), math.sin(a)
    return np.array([[c, 0.0, s_], [0.0, 1.0, 0.0], [-s_, 0.0, c]])


def _ensure_ready(ctx, plan):
    """Plans start far more reliably from the ready pose than from the arms hanging down."""
    ready = [math.radians(v) for v in ctx.cfg["manipulation"]["ready_deg"]]
    if max(abs(plan.q[15 + k] - r) for k, r in enumerate(ready)) > math.radians(10):
        goal = dict(plan.q)
        goal.update({15 + k: r for k, r in enumerate(ready)})
        plan.to_joint("arms to ready pose", goal, [])


def _table(ctx, support_z, front_x, y=0.0):
    """Table/case top as a slab: its measured front edge, raised by the rim height."""
    M, P = ctx.cfg["manipulation"], ctx.cfg["perception"]
    top = support_z + P["table_rim_m"]
    return cuboid("table", (front_x + 0.6, y, top - M["table_block_m"] / 2), (1.2, 1.6, M["table_block_m"]))


def _face(ctx, plan, x, y, what, frac=1.0):
    """Turn the waist (a fraction of the way) so (x, y) comes in front of the arms."""
    lim = math.radians(ctx.cfg["manipulation"]["max_face_deg"])
    plan.waist_to(f"turn waist toward {what}", [max(-lim, min(lim, math.atan2(y, x) * frac)), 0.0, 0.0])


def _hands(cx, cy, z, half, yaw, pitch_deg=0.0, keep=False):
    """Palm goals around (cx, cy, z): palms face each other along the axis turned by `yaw`."""
    ux, uy = -math.sin(yaw), math.cos(yaw)
    q = None if keep else mat_to_quat(_rz(yaw) @ _ry(math.radians(pitch_deg)))
    return {"left_tcp": {"position": [cx + ux * half, cy + uy * half, z], "quaternion": q},
            "right_tcp": {"position": [cx - ux * half, cy - uy * half, z], "quaternion": q}}


def _grasp_geometry(ctx, object):
    """Squeeze grasp of a found object: its centre, the width across the robot's left-right (the palms
    close on those two faces), the grasp height and the cuboids around it."""
    M = ctx.cfg["manipulation"]
    b = ctx.bb["objects"].get(object)
    if b is None:
        raise SkillError(f"'{object}' has not been found yet")
    (cx, cy, cz), (dx, dy, dz) = b["center"], b["dims"]
    if not M["min_box_width_m"] <= dy <= M["max_box_width_m"]:
        raise SkillError(f"'{object}' is {dy:.2f} m wide, outside {M['min_box_width_m']}..{M['max_box_width_m']} m")
    zg = max(cz, b["support_z"] + M["grasp_min_above_support_m"])   # centre, but the gloved hand clears the support
    if zg > b["top_z"] - 0.01:
        raise SkillError(f"'{object}' is too short to squeeze without the hands hitting what it stands on")
    byaw = b.get("yaw", 0.0)
    if abs(byaw) > math.radians(M["max_trusted_box_yaw_deg"]):
        byaw = 0.0                              # a large estimated angle is more likely a mis-fit than real
    others = [cuboid(k, v["center"], v["dims"], v.get("yaw", 0.0)) for k, v in ctx.bb["objects"].items() if k != object]
    tallest = max([v["top_z"] for k, v in ctx.bb["objects"].items() if k != object], default=b["support_z"])
    return {"object": object, "cx": cx, "cy": cy, "cz": cz, "zg": zg, "half": dy / 2, "width": dy,
            "dims": list(b["dims"]), "byaw": byaw, "support_z": b["support_z"], "top_z": b["top_z"],
            "box": cuboid(object, b["center"], b["dims"], byaw), "others": others,
            "table": _table(ctx, b["support_z"], b.get("support_front_x", b["front_x"] - 0.10), cy),
            "squeeze": min(M["squeeze_m"], max(0.0, (dy - M["min_squeeze_gap_m"]) / 2)),
            "lift": max(M["lift_m"], tallest - b["support_z"] + M["carry_clear_m"])}


def _check_lines(plan, moves):
    """Plan these straight moves from the plan's current pose only to prove they are possible."""
    mk = plan.mark()
    try:
        for label, goals, world in moves:
            plan.line(label, goals, world)
    finally:
        plan.restore(mk)


@skill("pre_grasp", {"object": "a found object (box)"},
       "Bring both gloves beside the object at grasp height, 2 cm off its left and right sides (not touching), "
       "with the waist turned toward it. The object must be within reach. Next: grasp.", "arm", hands=("EMPTY",))
def pre_grasp(ctx, plan, object):
    M = ctx.cfg["manipulation"]
    g = _grasp_geometry(ctx, object)
    cx, cy, zg, half, sq = g["cx"], g["cy"], g["zg"], g["half"], g["squeeze"]
    plan.waist_to("straighten waist", [plan.waist[0], 0.0, 0.0])
    _ensure_ready(ctx, plan)
    _map_scene(ctx, [g["box"]] + g["others"])   # depth obstacle map of everything else (found objects stay cuboids)
    near = [g["table"], g["box"]] + g["others"]
    free = [g["table"]] + g["others"]           # the object itself is touched from here on
    last, start = None, plan.mark()
    # each candidate (waist turn, palm angle, clearance, finger pitch) must also squeeze and lift in straight lines
    for frac in M["face_fractions"]:
        for palm in M["palm_align"]:
            for clear in M["approach_clear_m"]:
                for pitch in M["grasp_pitch_deg"]:
                    plan.restore(start)
                    if frac:
                        _face(ctx, plan, cx, cy, object, frac)
                    yaw = palm * g["byaw"] + (1 - palm) * plan.waist[0]
                    try:
                        _come_down_beside(ctx, plan, cx, cy, zg, half, yaw, pitch, near)
                        plan.line("move palms beside the object", _hands(cx, cy, zg, half + clear, yaw, pitch), near)
                        _check_lines(plan, [("squeeze", _hands(cx, cy, zg, half - sq, yaw, pitch), free),
                                            ("lift", _hands(cx, cy, zg + g["lift"] + M["lift_sag_comp_m"], half - sq, yaw, pitch), free)])
                    except SkillError as e:
                        last = e
                        continue
                    g.update(yaw=yaw, pitch=pitch, clear=clear)
                    _to_state(ctx, plan, "AT_OBJECT", grasp=g)
                    return {"object": object, "width_m": round(g["width"], 3), "grasp_height_m": round(zg, 3),
                            "waist_deg": round(math.degrees(plan.waist[0]), 1), "palm_yaw_deg": round(math.degrees(yaw), 1)}
    plan.restore(start)
    raise SkillError(f"cannot bring the gloves beside '{object}': {last}")


@skill("grasp", {}, "Squeeze the object the gloves are beside (after pre_grasp): close in until they touch it, "
       "then press 2 cm more. Next: lift (or retreat to let go).", "arm", hands=("AT_OBJECT",))
def grasp(ctx, plan):
    g = plan.state["grasp"]
    cx, cy, zg, half, sq, yaw, pitch = (g[k] for k in ("cx", "cy", "zg", "half", "squeeze", "yaw", "pitch"))
    free = [g["table"]] + g["others"]
    grip = {"extra": 0.0, "dx": 0.0, "dy": 0.0}  # squeeze correction and hand-measured centre shift
    plan.line("squeeze", _hands(cx, cy, zg, half - sq, yaw, pitch), free)
    _to_state(ctx, plan, "GRIPPING", grip=grip)
    plan.call("settle the grip", lambda: _settle_grip(ctx, cx, cy, zg, half - sq, yaw, pitch, free, grip))
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
    hold = lambda x, y, z: (lambda: _hands(x + grip["dx"], y + grip["dy"], z, half - sq - grip["extra"], yaw, pitch))
    plan.call("secure the grip (test lift)", lambda: _secure_grip(ctx, cx, cy, zg, half, sq, yaw, pitch, free, grip))
    plan.line("lift", hold(cx, cy, z_cmd)(), free, goals_fn=hold(cx, cy, z_cmd))
    held = {"object": g["object"], "width": g["width"], "dims": g["dims"], "grasp_above_bottom": zg - g["support_z"],
            "support_z": g["support_z"], "gap_cmd": g["width"] - 2 * sq,
            "gap_of": lambda: g["width"] - 2 * (sq + grip["extra"]), "pitch": pitch, "clear": g["clear"], "yaw": yaw}
    _to_state(ctx, plan, "HOLDING", holding=held)
    plan.attach(cuboid("held", (cx, cy, cz + g["lift"] + M["lift_sag_comp_m"]), g["dims"], g["byaw"]))
    # draw the object toward the chest along the torso's forward axis: shorter lever, less load (carry pose)
    wy = plan.waist[0]
    pull = min(M["pull_in_m"], max(0.0, math.hypot(cx, cy) - M["min_carry_x_m"]))
    if pull > 0.005:                            # optional: skipped if it cannot be done in a straight line
        px, py = cx - math.cos(wy) * pull, cy - math.sin(wy) * pull
        mk = plan.mark()
        try:
            plan.line("pull in to the carry pose", hold(px, py, z_cmd)(), free, goals_fn=hold(px, py, z_cmd))
        except SkillError:
            plan.restore(mk)
    plan.call("check lift height", lambda: _check_height(ctx, z_lift, "the object did not come up to the lift height"))
    plan.call("check grip after lift", lambda: _check_gap(ctx, held["gap_of"](), "the object slipped during the lift"))
    plan.call("holding", lambda: ctx.bb["objects"].pop(g["object"], None))
    return {"object": g["object"], "lift_m": round(g["lift"], 3)}


@skill("set_down", {}, "Recovery while holding: lower the object straight down onto the surface under it "
       "(the gloves stay closed). Next: retreat.", "arm", hands=("HOLDING",))
def set_down(ctx, plan):
    M = ctx.cfg["manipulation"]
    h = plan.state["holding"]
    left, right = plan.palms()
    mid, half = (left + right) / 2, float(np.linalg.norm(left - right)) / 2
    yaw = math.atan2(-(left - right)[0], (left - right)[1])
    z = h["support_z"] + h["grasp_above_bottom"] + M["place_clear_m"]
    plan.line("set down", _hands(mid[0], mid[1], z, half, yaw, keep=True), [])
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
    plan.attached = None
    opened = half + M["retreat_open_m"]
    plan.line("open the gloves", _hands(mid[0], mid[1], mid[2], opened, yaw, keep=True), world)
    _back_to_ready(ctx, plan, world, mid[0], mid[1], mid[2], opened, yaw, 0.0, True)
    _to_state(ctx, plan, "EMPTY", grasp=None, grip=None)
    return {"retreated": True}


def _back_to_ready(ctx, plan, world, x, y, z, half, yaw, pitch, keep):
    """From open gloves around an object: straight up as far as reachable, straight back toward the chest
    out of the object's area, then the ready pose planned around everything (the straight moves cannot
    fail to be tried; only the last free move needs a plan)."""
    M = ctx.cfg["manipulation"]
    wy = plan.waist[0]
    for rise in (0.08, 0.04, 0.02, 0.0):
        mk = plan.mark()
        try:
            if rise:
                plan.line("move the hands up", _hands(x, y, z + rise, half, yaw, pitch, keep=keep), world)
            break
        except SkillError:
            plan.restore(mk)
    else:
        rise = 0.0
    for back in (0.12, 0.08, 0.04):
        mk = plan.mark()
        try:
            plan.line("draw the hands back", _hands(x - math.cos(wy) * back, y - math.sin(wy) * back, z + rise, half,
                                                    yaw, pitch, keep=keep), world)
            break
        except SkillError:
            plan.restore(mk)
    goal = dict(plan.q)
    goal.update({15 + k: math.radians(v) for k, v in enumerate(M["ready_deg"])})
    try:
        plan.to_joint("ready pose", goal, world, margin=M["near_margin_m"])
    except SkillError:
        plan.to_joint("ready pose", goal, world, margin=0.0)
    plan.waist_to("waist straight", [math.radians(v) for v in M["waist_deg"]])


def _come_down_beside(ctx, plan, cx, cy, zg, half, yaw, pitch, world):
    """Bring the palms down beside the object, as far off its faces as fits between its neighbours
    (approach_far_m, first that plans), from the highest approach_rise_m above the grasp height that is
    reachable (that move keeps the transit margin; the short descent the near margin)."""
    M = ctx.cfg["manipulation"]
    last = None
    for far in M["approach_far_m"]:
        mk = plan.mark()
        try:
            for rise in M["approach_rise_m"]:
                try:
                    plan.to_pose("palms above, beside the box", _hands(cx, cy, zg + rise, half + far, yaw, pitch), world)
                    break
                except SkillError:
                    plan.restore(mk)
            plan.to_pose("lower beside the box", _hands(cx, cy, zg, half + far, yaw, pitch), world,
                         margin=M["near_margin_m"])
            return far
        except SkillError as e:
            plan.restore(mk)
            last = e
    raise last


def _map_scene(ctx, exclude):
    """Depth obstacle map for this skill from a fresh frame (live robot only; offline checks plan with the
    fitted cuboids alone). exclude: every found object (each is planned around as its fitted cuboid, which
    is tighter than the map's surfaces); the map covers everything else (the support, its rim, unknowns)."""
    if ctx.camera is None or ctx.geometry is None:
        return None
    return ctx.motion.build_map(ctx.camera.fine(), ctx.geometry, exclude)


def _settle_grip(ctx, cx, cy, zg, half_cmd, yaw, pitch, world, grip):
    """The squeeze aims past the faces and the object stops the hands somewhere before that (glove and
    object models are never exact). Re-aim each hand grip_preload_m past where it actually stopped, centred
    where the hands found the object: a light, repeatable squeeze force for any object, no overheating."""
    M = ctx.cfg["manipulation"]
    m = ctx.motion
    time.sleep(0.4)
    lm, rm = m.palms(m.q_now(commanded=False))
    axis = np.array([-math.sin(yaw), math.cos(yaw)])
    gap = float(np.dot(lm[:2] - rm[:2], axis))
    mid = (lm[:2] + rm[:2]) / 2
    shift = mid - np.array([cx, cy])
    grip["dx"], grip["dy"] = float(shift[0]), float(shift[1])
    hs = max(M["min_squeeze_gap_m"] / 2, gap / 2 - M["grip_preload_m"])
    grip["extra"] = half_cmd - hs
    p = ArmPlan(m)
    p.line("settle the grip", _hands(mid[0], mid[1], zg, hs, yaw, pitch), world)
    m.execute(p.steps[-1]["traj"], holding=True)
    return {"palm_gap_m": round(gap, 3), "hold_gap_m": round(2 * hs, 3), "centre_shift_m": [round(v, 3) for v in shift]}


def _secure_grip(ctx, cx, cy, zg, half, squeeze, yaw, pitch, world, grip):
    """Test lift: raise 2 cm, hold, and check the box comes with the hands (height) without slipping
    (palm gap). If not, put it back down, squeeze 1 cm deeper and try again. Runs live on the robot."""
    M = ctx.cfg["manipulation"]
    m = ctx.motion
    for attempt in range(M["grip_retries"] + 1):
        hs = half - squeeze - grip["extra"]
        gx, gy = cx + grip["dx"], cy + grip["dy"]
        p = ArmPlan(m)
        p.line("test lift", _hands(gx, gy, zg + M["test_lift_m"], hs, yaw, pitch), world)
        m.execute(p.steps[-1]["traj"], holding=True)
        time.sleep(M["test_hold_s"])
        left, right = m.palms(m.q_now(commanded=False))
        gap, z = float(np.linalg.norm(left - right)), float(left[2] + right[2]) / 2
        gap_cmd = 2 * hs
        ok_gap = gap - gap_cmd >= M["drop_gap_m"]
        ok_z = z >= zg + M["test_lift_m"] - M["lift_tolerance_m"]
        if ok_gap and ok_z:
            # self-calibration: the gripping palms are centred on the box, so their midpoint (from joint
            # angles) minus the camera's box centre is the camera-to-arm offset along the squeeze axis
            mid = (left + right) / 2
            axis = np.array([-math.sin(yaw), math.cos(yaw)])
            err = float(np.dot(mid[:2] - np.array([cx, cy]), axis))
            ctx.bb["calib"] = {"offset_xy": (axis * err).tolist(), "t": time.time()}
            return {"extra_squeeze_m": grip["extra"], "palm_gap_m": round(gap, 3), "camera_offset_m": round(err, 3)}
        if grip["extra"] + M["grip_retry_step_m"] > M["max_extra_squeeze_m"]:
            break
        p = ArmPlan(m)                          # back down, squeeze a little deeper, retry
        grip["extra"] += M["grip_retry_step_m"]
        p.line("set back down", _hands(gx, gy, zg, half - squeeze - grip["extra"], yaw, pitch), world)
        m.execute(p.steps[-1]["traj"], holding=True)
    raise SkillError("could not get a secure grip on the box (it slips in the test lift)")


def _calibrated(ctx, plan, x, y):
    """Correct a camera-measured target by the offset found when the palms gripped the held box. In a
    pre-built plan the offset is only known at run time, so the place steps re-read it live."""
    if not ctx.cfg["manipulation"]["use_grasp_calibration"]:
        return x, y
    plan.state["use_calib"] = True
    off = ctx.bb.get("calib", {}).get("offset_xy", [0.0, 0.0])
    return x + off[0], y + off[1]


def _palm_mids(ctx):
    m = ctx.motion
    lc, rc = m.palms(m.q_now(commanded=True))
    lm, rm = m.palms(m.q_now(commanded=False))
    return (lc + rc) / 2, (lm + rm) / 2, float(np.linalg.norm(lc - rc)) / 2


def _align_over(ctx, live_xy, z, yaw, h, keep, world, corr):
    """Closed loop: move until the MEASURED palm midpoint (= box centre) is over the target centre."""
    M = ctx.cfg["manipulation"]
    for _ in range(M["align_iters"]):
        time.sleep(0.4)
        cmd, meas, half = _palm_mids(ctx)
        err = np.array(live_xy()) - meas[:2]
        corr["xy"] = cmd[:2] - meas[:2]
        if np.linalg.norm(err) < M["align_tol_m"]:
            break
        p = ArmPlan(ctx.motion)
        p.attached = None
        p.line("line up", _hands(*(cmd[:2] + err), z, half, yaw, h.get("pitch", 0), keep=keep), world)
        ctx.motion.execute(p.steps[-1]["traj"], holding=True)
    _, meas, _ = _palm_mids(ctx)
    return {"error_m": round(float(np.linalg.norm(np.array(live_xy()) - meas[:2])), 4),
            "correction_m": [round(float(v), 4) for v in corr["xy"]]}


def _gentle_release(ctx, yaw, h, keep, world, corr):
    """Stop pressing down, open the palms a little and slowly, lift away, then open fully."""
    M = ctx.cfg["manipulation"]
    m = ctx.motion
    pitch = h.get("pitch", 0)
    time.sleep(0.3)
    cmd, meas, half = _palm_mids(ctx)
    xy, zc = cmd[:2], float(meas[2]) + 0.003          # command at the contact height: no downward press
    for label, hw, zz in (("ease off", half, zc),
                          ("open a little", h["width"] / 2 + M["release_open_m"], zc),
                          ("lift hands", h["width"] / 2 + M["release_open_m"], zc + M["release_lift_m"]),
                          ("open fully", h["width"] / 2 + h["clear"], zc + M["release_lift_m"])):
        p = ArmPlan(m)
        p.line(label, _hands(*xy, zz, hw, yaw, pitch, keep=keep), world)
        m.execute(p.steps[-1]["traj"], holding=True)    # slow playback
    return {"released_at_z": round(zc, 3)}


def _check_gap(ctx, gap_cmd, msg):
    time.sleep(0.3)
    left, right = ctx.motion.palms(ctx.motion.q_now(commanded=False))
    gap = float(np.linalg.norm(left - right))
    if gap - gap_cmd < ctx.cfg["manipulation"]["drop_gap_m"]:
        raise SkillError(f"{msg} (palm gap {gap:.3f} m, commanded {gap_cmd:.3f} m)")
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
    half = float(np.linalg.norm(left - right)) / 2
    target_yaw = h["yaw"]
    if on in ctx.bb["objects"]:                       # stack on another object
        t = ctx.bb["objects"][on]
        x, y, z_s = t["center"][0], t["center"][1], t["top_z"]
        x, y = _calibrated(ctx, plan, x, y)
        target_yaw = t.get("yaw", 0.0)                # square the box to the one below it
        if abs(target_yaw) > math.radians(M["max_trusted_box_yaw_deg"]):
            target_yaw = 0.0
        world = [cuboid(on, t["center"], t["dims"], target_yaw),
                 _table(ctx, t["support_z"], t.get("support_front_x", t["front_x"] - 0.10), y)]
    else:
        if on not in ctx.bb["surfaces"]:
            find(ctx, on)
        sf = ctx.bb["surfaces"][on]
        z_s, y = sf["top_z"], float(left[1] + right[1]) / 2
        x = max(float(left[0] + right[0]) / 2, sf["front_x"] + h["dims"][0] / 2 + 0.03)
        world = [_table(ctx, z_s, sf["front_x"], y)]
    z = z_s + h["grasp_above_bottom"] + M["place_clear_m"]
    above = max(z + M["place_above_m"], float(left[2] + right[2]) / 2)  # never below the carry height before arriving
    mid = (left + right) / 2                    # the held box (between the palms) is cut out of the obstacle map
    _map_scene(ctx, [cuboid(h["object"], (mid[0], mid[1], mid[2] - h["grasp_above_bottom"] + h["dims"][2] / 2),
                            h["dims"], h["yaw"])]
               + [cuboid(k, v["center"], v["dims"], v.get("yaw", 0.0)) for k, v in ctx.bb["objects"].items()])
    last, base, aligned = None, plan.mark(), False
    for frac in M["face_fractions"]:            # turn toward the target; the lift was verified before this
        for align in (True, False):             # square the box to the target first, else keep the grip angle
            plan.restore(base)
            if frac:
                _face(ctx, plan, x, y, on, frac)
            goal = (_hands(x, y, above, half, target_yaw, h.get("pitch", 0)) if align
                    else _hands(x, y, above, half, plan.waist[0], keep=True))
            try:
                plan.to_pose("move above the target", goal, world)
                aligned = align
                break
            except SkillError as e:
                last = e
        else:
            continue
        break
    else:
        plan.restore(base)
        raise SkillError(f"cannot bring the box above '{on}': {last}")
    yaw = target_yaw if aligned else plan.waist[0]
    keep = not aligned
    live_half = lambda: float(np.linalg.norm(np.subtract(*ctx.motion.palms(ctx.motion.q_now())))) / 2
    tx, ty = x, y
    if plan.state.get("use_calib") and on in ctx.bb["objects"]:
        raw = ctx.bb["objects"][on]["center"]
        live_xy = lambda: _calibrated(ctx, plan, raw[0], raw[1])
    else:
        live_xy = lambda: (tx, ty)
    plan.steps[-1]["goals_fn"] = lambda: _hands(*live_xy(), above, live_half(), yaw, h.get("pitch", 0), keep=keep)
    plan.call("check grip before lowering", lambda: _check_gap(ctx, h["gap_of"]() if h.get("gap_of") else h["gap_cmd"],
                                                               f"'{h['object']}' slipped while moving"))
    corr = {"xy": np.zeros(2)}                  # commanded - measured palm midpoint (tracking error under load)
    plan.call("line up over the target", lambda: _align_over(ctx, live_xy, above, yaw, h, keep, world, corr))
    lower_goal = lambda: _hands(*(np.array(live_xy()) + corr["xy"]), z, live_half(), yaw, h.get("pitch", 0), keep=keep)
    plan.line("lower", _hands(x, y, z, half, yaw, h.get("pitch", 0), keep=keep), world, goals_fn=lower_goal)
    plan.call("release gently", lambda: _gentle_release(ctx, yaw, h, keep, world, corr))
    # planned for feasibility/continuity only; the gentle release above does the real opening
    plan.to_pose("open hands (check only)", _hands(x, y, z, h["width"] / 2 + h["clear"], yaw, h.get("pitch", 0), keep=keep), world,
                 margin=0.0)
    plan.steps[-1]["dry"] = True
    plan.state["holding"] = None
    placed = {"center": [x, y, z_s + h["dims"][2] / 2], "dims": h["dims"], "top_z": z_s + h["dims"][2],
              "support_z": z_s, "front_x": x - h["dims"][0] / 2, "distance_m": math.hypot(x, y)}

    def done():
        ctx.bb["holding"] = None
        ctx.bb["objects"][h["object"]] = {**placed, "t": time.time()}
    plan.call("released", done)
    away = world + [cuboid(h["object"], placed["center"], placed["dims"])]
    _back_to_ready(ctx, plan, away, x, y, z, h["width"] / 2 + h["clear"], yaw, h.get("pitch", 0), keep)
    _to_state(ctx, plan, "EMPTY", grasp=None, grip=None, holding=None)
    return {"placed": h["object"], "on": on, "height_m": round(z_s, 3)}


@skill("ready", {}, "Move both arms to the ready pose (hands at chest height, out of the camera view). Hands "
                    "must be empty.", "arm", hands=("EMPTY",))
def ready(ctx, plan):
    if plan.state.get("holding"):
        raise SkillError("holding a box: place it first")
    M = ctx.cfg["manipulation"]
    goal = dict(plan.q)
    goal.update({15 + k: math.radians(v) for k, v in enumerate(M["ready_deg"])})
    plan.to_joint("ready pose", goal, [])
    plan.waist_to("waist straight", [math.radians(v) for v in M["waist_deg"]])
    return {"ready": True}


# ==================================================================== checks and basics
@skill("check_holding", {}, "Verify the held box has not slipped (fails if it dropped).", "internal")
def check_holding(ctx):
    h = ctx.bb["holding"]
    if not h:
        return {"holding": None}
    gap = _check_gap(ctx, h["gap_of"]() if h.get("gap_of") else h["gap_cmd"], f"'{h['object']}' slipped out of the hands")
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
