"""Skill library shared by Motion planning (section 6) and AI control (section 7).

The LLM only chooses skills and their parameters; the poses come from OWLv2 + depth + cuRobo.
Arm skills append steps to an ArmPlan (so section 6 can plan everything first and play later);
the other skills run immediately. Positions are in the pelvis frame (x forward, y left, z up), which
does not move when the waist turns, so measurements stay valid until the robot walks or turns.
"""
import base64
import math
import time

import cv2
import numpy as np

from common import config

from common.fk import mat_to_quat

from .motion import ArmPlan, SkillError, cuboid, pose

SURFACE_WORDS = ("table", "desk", "shelf", "counter", "bench", "stand", "cabinet")
SKILLS = {}


def skill(name, params, desc, kind):
    """kind: 'perceive' (camera only), 'move' (walks/turns), 'arm' (adds to an ArmPlan), 'basic'."""
    def wrap(fn):
        SKILLS[name] = {"name": name, "params": params, "desc": desc, "kind": kind, "fn": fn}
        return fn
    return wrap


def catalog(kinds=None):
    """Skill list for the LLM prompt."""
    return [{"skill": s["name"], "params": s["params"], "what": s["desc"]}
            for s in SKILLS.values() if kinds is None or s["kind"] in kinds]


class Ctx:
    def __init__(self, link, camera, vision, geometry, motion, processing):
        self.cfg = config.load()
        self.link, self.camera, self.vision = link, camera, vision
        self.geometry, self.motion, self.processing = geometry, motion, processing
        self.bb = {"objects": {}, "surfaces": {}, "holding": None}   # blackboard

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
        """After walking/turning every remembered position is in an old torso frame."""
        self.bb["objects"], self.bb["surfaces"] = {}, {}

    def agent(self, action, params=None, timeout=60.0):
        r = self.link.send(action, params or {}, wait=True, wait_timeout_s=timeout)
        if not r.get("ok"):
            raise SkillError(f"{action}: {r.get('error')}")
        return r.get("result")


def run_skill(ctx, name, params, plan=None):
    """Run one skill now. Arm skills get a fresh plan (or the given one) and execute it."""
    s = SKILLS.get(name)
    if s is None:
        raise SkillError(f"unknown skill '{name}'")
    params = dict(params or {})
    if s["kind"] != "arm":
        return s["fn"](ctx, **params)
    if plan is None:
        plan = ArmPlan(ctx.motion)
        plan.state["holding"] = ctx.bb["holding"]
    result = s["fn"](ctx, plan, **params)
    plan.run()
    return result or {"steps": plan.summary()}


# ==================================================================== perception
QUALIFIERS = {"left", "right", "nearest", "closest", "farthest", "middle", "center", "centre"}


def _parse(label):
    """'right black box' -> ('black box', ['right']): OWLv2 sees the noun, code applies the position."""
    words = label.lower().split()
    quals = [w for w in words if w in QUALIFIERS]
    base = " ".join(w for w in words if w not in QUALIFIERS) or label
    return base, quals


def _detect_all(ctx, base):
    frame = ctx.camera.latest()
    if frame is None:
        raise SkillError("camera is offline")
    ok, jpg = cv2.imencode(".jpg", frame.rgb, [cv2.IMWRITE_JPEG_QUALITY, 95])
    with ctx.processing("owl"):
        dets = ctx.vision.call("detect", {"labels": [base], "image_b64": base64.b64encode(jpg).decode()})["detections"]
    rects, kept = [], []
    for d in dets:                                   # drop near-duplicate boxes (keep the higher score)
        x1, y1, x2, y2 = d["box"]
        r = (x1, y1, x2 - x1, y2 - y1)
        if all(_iou(r, k) < 0.5 for k in rects):
            rects.append(r)
            kept.append((r, d["score"]))
    if not kept:
        raise SkillError(f"'{base}' not visible")
    return frame, kept


def _look(ctx, base):
    """Detect, and if the best match is cut off at the top/bottom of the image, tilt the waist
    (pitch) to bring it into view and detect again. Returns (frame, detections, torso-in-pelvis)."""
    P = ctx.cfg["perception"]
    for _ in range(P["look_max_tries"] + 1):
        q = ctx.motion.q_now()
        T = ctx.geometry.torso_in_pelvis(ctx.motion.waist_of(q))
        frame, dets = _detect_all(ctx, base)
        (x, y, w, h), _ = dets[0]
        H = frame.rgb.shape[0]
        step = 0.0
        if y < P["look_edge_px"] and y + h < H - P["look_edge_px"]:
            step = -P["look_step_deg"]           # cut off at the top: lean back (look up)
        elif y + h > H - P["look_edge_px"] and y > P["look_edge_px"]:
            step = P["look_step_deg"]            # cut off at the bottom: lean forward (look down)
        pitch = math.degrees(q[14]) + step
        if not step or abs(pitch) > P["look_max_pitch_deg"]:
            return frame, dets, T
        ctx.motion.move_waist([q[12], q[13], math.radians(pitch)], 1.0)
        time.sleep(0.4)                          # let the image settle
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


@skill("find", {"object": "what to look for, e.g. 'box', 'left box', 'right black box', 'table'"},
       "Look with the camera and measure the object's position, size and distance. Words left/right/nearest/"
       "farthest pick among several matches. Needed before approach, pick_box or place. Positions are forgotten "
       "after walking or turning, so find again afterwards.",
       "perceive")
def find(ctx, object):
    base, quals = _parse(object)
    frame, dets, T = _look(ctx, base)
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
       "Walk in a straight line at the fixed AI speed (0.4 m/s); time = distance / 0.4.", "move")
def walk(ctx, direction, distance_m):
    action = {"forward": "forward", "backward": "backward", "left": "left_aside", "right": "right_aside"}.get(direction)
    if action is None or float(distance_m) <= 0:
        raise SkillError("walk needs direction forward|backward|left|right and distance_m > 0")
    t = _walk_time(ctx, float(distance_m))
    ctx.agent(action, {"velocity": ctx.cfg["ai"]["speed"], "duration": round(t, 2)}, timeout=t + 10)
    ctx.forget_positions()
    return {"direction": direction, "distance_m": float(distance_m), "seconds": round(t, 2)}


@skill("turn", {"direction": "left | right", "angle_deg": "positive degrees (0..180)"},
       "Turn in place by an exact angle, corrected by the IMU.", "move")
def turn(ctx, direction, angle_deg):
    if direction not in ("left", "right") or not 0 < float(angle_deg) <= 180:
        raise SkillError("turn needs direction left|right and angle_deg in (0, 180]")
    signed = float(angle_deg) if direction == "left" else -float(angle_deg)
    r = ctx.agent("turn_by", {"angle_deg": signed}, timeout=60)
    ctx.forget_positions()
    return r


@skill("approach", {"target": "a found object or surface", "standoff_m": "optional: final distance (m)"},
       "Walk straight to a target at 0.4 m/s until it is within arm reach (tables: front edge at "
       "standoff; objects: centre at reach distance), side-stepping to centre objects. Finds it again "
       "between legs.", "move")
def approach(ctx, target, standoff_m=None):
    a = ctx.cfg["ai"]
    surface = any(w in target.lower() for w in SURFACE_WORDS)
    ref = float(standoff_m) if standoff_m is not None else (a["table_standoff_m"] if surface else a["box_reach_x_m"])
    walked = 0.0
    for _ in range(6):
        find(ctx, target)
        if surface:
            dist, lateral = ctx.bb["surfaces"][target]["front_x"] - ref, 0.0
        else:
            c = ctx.bb["objects"][target]["center"]
            dist, lateral = c[0] - ref, c[1]
        if abs(lateral) > a["lateral_tolerance_m"]:
            walk(ctx, "left" if lateral > 0 else "right", abs(lateral))
            continue
        if abs(dist) <= a["approach_tolerance_m"]:
            find(ctx, target)
            return {"target": target, "walked_m": round(walked, 2), "remaining_m": round(dist, 3)}
        leg = min(abs(dist), a["approach_max_leg_m"])
        walk(ctx, "forward" if dist > 0 else "backward", leg)
        walked += leg
    raise SkillError(f"could not get within {a['approach_tolerance_m']} m of '{target}'")


# ==================================================================== arms (dual-arm squeeze, dummy hands)
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


def _face(ctx, plan, x, y, what, frac=1.0):
    """Turn the waist (a fraction of the way) so (x, y) comes in front of the arms."""
    lim = math.radians(ctx.cfg["manipulation"]["max_face_deg"])
    yaw = max(-lim, min(lim, math.atan2(y, x) * frac))
    plan.waist_to(f"turn waist toward {what}", [yaw, 0.0, 0.0])


def _table(ctx, support_z, front_x, y=0.0):
    M = ctx.cfg["manipulation"]
    return cuboid("table", (front_x + 0.6, y, support_z - M["table_block_m"] / 2 - 0.005), (1.2, 1.6, M["table_block_m"]))


def _table_front(ctx, fallback):
    s = next(iter(ctx.bb["surfaces"].values()), None)
    return s["front_x"] if s else fallback


@skill("pick_box", {"object": "a found box"},
       "Lift a box with both hands: palms move to its left and right faces, squeeze, and lift it.", "arm")
def pick_box(ctx, plan, object):
    M = ctx.cfg["manipulation"]
    b = ctx.bb["objects"].get(object)
    if b is None:
        raise SkillError(f"'{object}' has not been found yet")
    if plan.state.get("holding"):
        raise SkillError(f"already holding '{plan.state['holding']['object']}'")
    (cx, cy, cz), (dx, dy, dz) = b["center"], b["dims"]
    if not M["min_box_width_m"] <= dy <= M["max_box_width_m"]:
        raise SkillError(f"box width {dy:.2f} m is outside {M['min_box_width_m']}..{M['max_box_width_m']} m")
    zg = max(cz, b["support_z"] + M["grasp_min_above_support_m"])
    if zg > b["top_z"] - 0.01:
        raise SkillError("box is too short to squeeze without the hands hitting the table")
    half = dy / 2
    _ensure_ready(ctx, plan)
    table = _table(ctx, b["support_z"], _table_front(ctx, b["front_x"] - 0.10), cy)
    box = cuboid(object, b["center"], b["dims"])
    others = [cuboid(k, v["center"], v["dims"]) for k, v in ctx.bb["objects"].items() if k != object]

    def hands(off, z=zg, pitch=0.0):
        """Palms squeeze along the (turned) torso y axis; a light box rotates to meet them."""
        yaw = plan.waist[0]
        q = mat_to_quat(_rz(yaw) @ _ry(math.radians(pitch)))
        ux, uy = -math.sin(yaw), math.cos(yaw)
        r = half + off
        return {"left_tcp": pose((cx + ux * r, cy + uy * r, z), q), "right_tcp": pose((cx - ux * r, cy - uy * r, z), q)}

    squeeze = min(M["squeeze_m"], max(0.0, (dy - M["min_squeeze_gap_m"]) / 2))
    last, chosen, start = None, None, plan.mark()
    for yaw_frac in (1.0, 0.5, 0.0):            # face the box: the hands can't cross far past the midline
        for clear in M["approach_clear_m"]:     # wide boxes: less room outside the faces
            for pitch in M["grasp_pitch_deg"]:  # low boxes: fingers tilted down
                plan.restore(start)
                if yaw_frac:
                    _face(ctx, plan, cx, cy, object, yaw_frac)
                try:
                    plan.to_pose("move palms beside the box", hands(clear, pitch=pitch), [table, box] + others)
                    chosen = (clear, pitch)
                    break
                except SkillError as e:
                    last = e
            if chosen:
                break
        if chosen:
            break
    if not chosen:
        plan.restore(start)
        raise SkillError(f"cannot reach the sides of '{object}': {last}")
    clear, pitch = chosen
    plan.to_pose("squeeze", hands(-squeeze, pitch=pitch), [table] + others)
    gap_cmd = dy - 2 * squeeze
    plan.call("check contact", lambda: _check_gap(ctx, gap_cmd, "the hands closed without touching the box"))
    # lift so the box bottom clears every other object: later waist turns swing it sideways unchecked
    tallest = max([v["top_z"] for k, v in ctx.bb["objects"].items() if k != object], default=b["support_z"])
    lift = max(M["lift_m"], tallest - b["support_z"] + M["carry_clear_m"])
    plan.to_pose("lift", hands(-squeeze, zg + lift, pitch=pitch), [table] + others)
    held = {"object": object, "width": dy, "dims": b["dims"], "grasp_above_bottom": zg - b["support_z"],
            "gap_cmd": gap_cmd, "pitch": pitch, "clear": clear}
    plan.state["holding"] = held

    def done():
        ctx.bb["holding"] = held
        ctx.bb["objects"].pop(object, None)
    plan.call("holding", done)
    return {"object": object, "width_m": round(dy, 3), "grasp_height_m": round(zg, 3), "pitch_deg": pitch}


def _check_gap(ctx, gap_cmd, msg):
    time.sleep(0.3)
    left, right = ctx.motion.palms(ctx.motion.q_now(commanded=False))
    gap = float(np.linalg.norm(left - right))
    if gap - gap_cmd < ctx.cfg["manipulation"]["drop_gap_m"]:
        raise SkillError(f"{msg} (palm gap {gap:.3f} m, commanded {gap_cmd:.3f} m)")
    return gap


@skill("place", {"on": "'table' (or another found surface) or a found object to stack on"},
       "Put the held box down on a surface or on top of another found object, then open the hands.", "arm")
def place(ctx, plan, on):
    M = ctx.cfg["manipulation"]
    h = plan.state.get("holding")
    if not h:
        raise SkillError("not holding anything")
    base = plan.mark()
    left, right = plan.palms()
    half = float(np.linalg.norm(left - right)) / 2
    world = []
    if on in ctx.bb["objects"]:                       # stack on another object
        t = ctx.bb["objects"][on]
        x, y, z_s = t["center"][0], t["center"][1], t["top_z"]
        world = [cuboid(on, t["center"], t["dims"]), _table(ctx, t["support_z"], _table_front(ctx, t["front_x"] - 0.10), y)]
    else:
        if on not in ctx.bb["surfaces"]:
            find(ctx, on)
        s = ctx.bb["surfaces"][on]
        z_s, y = s["top_z"], float(left[1] + right[1]) / 2
        x = max(float(left[0] + right[0]) / 2, s["front_x"] + h["dims"][0] / 2 + 0.03)
        world = [_table(ctx, z_s, s["front_x"], y)]
    z = z_s + h["grasp_above_bottom"] + M["place_clear_m"]
    plan.restore(base)

    def hands(hw, zz, xx=x):
        """Palms keep their grip orientation, so they sit apart along the (turned) torso y axis."""
        yaw = plan.waist[0]
        ux, uy = -math.sin(yaw), math.cos(yaw)
        return {"left_tcp": {"position": [xx + ux * hw, y + uy * hw, zz], "quaternion": None},
                "right_tcp": {"position": [xx - ux * hw, y - uy * hw, zz], "quaternion": None}}

    last = None
    for frac in (1.0, 0.5, 0.0):                # turn toward the target; hands keep their grip orientation
        plan.restore(base)
        if frac:
            _face(ctx, plan, x, y, on, frac)
        try:
            plan.to_pose("move above the target", hands(half, z + 0.05), world)
            break
        except SkillError as e:
            last = e
    else:
        plan.restore(base)
        raise SkillError(f"cannot bring the box above '{on}': {last}")
    plan.to_pose("lower", hands(half, z), world)
    plan.to_pose("open hands", hands(h["width"] / 2 + h["clear"], z), world)
    plan.state["holding"] = None
    placed = {"center": [x, y, z_s + h["dims"][2] / 2], "dims": h["dims"], "top_z": z_s + h["dims"][2],
              "support_z": z_s, "front_x": x - h["dims"][0] / 2, "distance_m": math.hypot(x, y)}

    def done():
        ctx.bb["holding"] = None
        ctx.bb["objects"][h["object"]] = {**placed, "t": time.time()}
    plan.call("released", done)
    plan.to_pose("move hands away", hands(h["width"] / 2 + h["clear"], z + 0.08),
                 world + [cuboid(h["object"], placed["center"], placed["dims"])])
    return {"placed": h["object"], "on": on, "height_m": round(z_s, 3)}


@skill("ready", {}, "Move both arms to the ready pose (forearms forward, palms facing). Do not use while "
                    "holding a box.", "arm")
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
@skill("check_holding", {}, "Verify the held box has not slipped (fails if it dropped).", "basic")
def check_holding(ctx):
    h = ctx.bb["holding"]
    if not h:
        return {"holding": None}
    gap = _check_gap(ctx, h["gap_cmd"], f"'{h['object']}' slipped out of the hands")
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
