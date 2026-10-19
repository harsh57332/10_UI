"""Skill library shared by Motion planning (section 6) and AI control (section 7).

The LLM only chooses skills and their parameters; the poses come from OWLv2 + depth + cuRobo.
Arm skills append steps to an ArmPlan (so section 6 can plan everything first and play later);
the other skills run immediately. All positions are in the torso frame (x forward, y left, z up).
"""
import base64
import math
import time

import cv2

from common import config

from .motion import ArmPlan, SkillError, cuboid, pitch_quat, pose

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
    plan = plan or ArmPlan(ctx.motion)
    plan.state = getattr(plan, "state", {"holding": ctx.bb["holding"]})
    result = s["fn"](ctx, plan, **params)
    plan.run()
    return result or {"steps": plan.summary()}


# ==================================================================== perception
def _detect(ctx, label):
    frame = ctx.camera.latest()
    if frame is None:
        raise SkillError("camera is offline")
    ok, jpg = cv2.imencode(".jpg", frame.rgb, [cv2.IMWRITE_JPEG_QUALITY, 95])
    with ctx.processing("owl"):
        dets = ctx.vision.call("detect", {"labels": [label], "image_b64": base64.b64encode(jpg).decode()})["detections"]
    if not dets:
        raise SkillError(f"'{label}' not visible")
    x1, y1, x2, y2 = dets[0]["box"]
    return frame, (x1, y1, x2 - x1, y2 - y1), dets[0]["score"]


@skill("find", {"object": "name of the object or surface, e.g. 'black box', 'table'"},
       "Look with the camera and measure the object's position, size and distance. Needed before approach, "
       "pick_box or place. Positions are forgotten after walking or turning, so find again afterwards.",
       "perceive")
def find(ctx, object):
    frame, rect, score = _detect(ctx, object)
    if any(w in object.lower() for w in SURFACE_WORDS):
        s = ctx.geometry.fit_surface(frame, rect)
        if s is None:
            raise SkillError(f"saw '{object}' but could not measure its top surface")
        ctx.bb["surfaces"][object] = {**s, "rect": rect, "t": time.time()}
        return {"surface": object, "front_edge_m": round(s["front_x"], 3), "height_m": round(s["top_z"], 3)}
    b = ctx.geometry.fit_box(frame, rect)
    if b is None:
        raise SkillError(f"saw '{object}' but depth was not good enough to measure it")
    b["distance_m"] = math.hypot(b["center"][0], b["center"][1])
    ctx.bb["objects"][object] = {**b, "rect": rect, "t": time.time()}
    return {"object": object, "center_m": [round(v, 3) for v in b["center"]],
            "size_m": [round(v, 3) for v in b["dims"]], "distance_m": round(b["distance_m"], 3), "score": score}


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
    table = _table(ctx, b["support_z"], _table_front(ctx, b["front_x"] - 0.10), cy)
    box = cuboid(object, b["center"], b["dims"])

    def hands(off, z=zg, q=(1, 0, 0, 0)):
        return {"left_tcp": pose((cx, cy + half + off, z), q), "right_tcp": pose((cx, cy - half - off, z), q)}

    squeeze = min(M["squeeze_m"], max(0.0, (dy - M["min_squeeze_gap_m"]) / 2))
    last, chosen = None, None
    for clear in M["approach_clear_m"]:         # wide boxes: less room outside the faces
        for pitch in M["grasp_pitch_deg"]:      # low boxes: fingers tilted down
            quat = pitch_quat(pitch)
            try:
                plan.to_pose("move palms beside the box", hands(clear, q=quat), [table, box])
                chosen = (clear, pitch)
                break
            except SkillError as e:
                last = e
        if chosen:
            break
    if not chosen:
        raise SkillError(f"cannot reach the sides of '{object}': {last}")
    clear, pitch = chosen
    plan.to_pose("squeeze", hands(-squeeze, q=quat), [table])
    gap_cmd = dy - 2 * squeeze
    plan.call("check contact", lambda: _check_gap(ctx, gap_cmd, "the hands closed without touching the box"))
    plan.to_pose("lift", hands(-squeeze, zg + M["lift_m"], q=quat), [table])
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
    gap = float(left[1] - right[1])
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
    left, right = plan.palms()
    half = float(left[1] - right[1]) / 2
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

    quat = pitch_quat(h.get("pitch", 0))

    def hands(hw, zz, xx=x):
        return {"left_tcp": pose((xx, y + hw, zz), quat), "right_tcp": pose((xx, y - hw, zz), quat)}

    plan.to_pose("move above the target", hands(half, z + 0.05), world)
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
    plan.call("waist straight", lambda: ctx.agent("play_upperbody", {"keyframes": [{"t": 1.0, "q": {
        str(12 + k): v for k, v in enumerate(M["waist_deg"])}}]}))
    plan.to_joint("ready pose", goal, [])
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
