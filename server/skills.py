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

from common import config, joints

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
       "farthest pick among several matches. Needed before approach, pick_box or place. Positions are forgotten "
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
        find(ctx, target, precision="coarse")
        if surface:
            dist, lateral = ctx.bb["surfaces"][target]["front_x"] - ref, 0.0
        else:
            c = ctx.bb["objects"][target]["center"]
            dist, lateral = c[0] - ref, c[1]
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
    plan.waist_to("straighten waist", [plan.waist[0], 0.0, 0.0])
    _ensure_ready(ctx, plan)
    table = _table(ctx, b["support_z"], b.get("support_front_x", b["front_x"] - 0.10), cy)
    byaw = b.get("yaw", 0.0)                    # palms close flat on the real faces, not on corners
    box = cuboid(object, b["center"], b["dims"], byaw)
    others = [cuboid(k, v["center"], v["dims"], v.get("yaw", 0.0)) for k, v in ctx.bb["objects"].items() if k != object]
    squeeze = min(M["squeeze_m"], max(0.0, (dy - M["min_squeeze_gap_m"]) / 2))

    grip = {"extra": 0.0}                       # extra squeeze found by the test lift
    gap_of = lambda: dy - 2 * (squeeze + grip["extra"])
    tallest = max([v["top_z"] for k, v in ctx.bb["objects"].items() if k != object], default=b["support_z"])
    lift = max(M["lift_m"], tallest - b["support_z"] + M["carry_clear_m"])
    z_lift = zg + lift
    z_cmd = z_lift + M["lift_sag_comp_m"]       # the arms sag ~3 cm under the box; aim that much higher
    hold = lambda x, y, z: (lambda: _hands(x, y, z, half - squeeze - grip["extra"], yaw, pitch))

    # each candidate (waist turn, clearance, finger pitch) must plan the whole grasp, not just the approach
    last, chosen, start = None, None, plan.mark()
    for frac in M["face_fractions"]:
      for palm in M["palm_align"]:              # 1 = flat on the box faces ... 0 = along the turned torso
        for clear in M["approach_clear_m"]:
            for pitch in M["grasp_pitch_deg"]:
                plan.restore(start)
                if frac:
                    _face(ctx, plan, cx, cy, object, frac)
                yaw = palm * byaw + (1 - palm) * plan.waist[0]
                try:
                    plan.to_pose("move palms beside the box", _hands(cx, cy, zg, half + clear, yaw, pitch),
                                 [table, box] + others)
                    plan.to_pose("squeeze", _hands(cx, cy, zg, half - squeeze, yaw, pitch), [table] + others)
                    plan.call("secure the grip (test lift)",
                              lambda: _secure_grip(ctx, cx, cy, zg, half, squeeze, yaw, pitch, [table] + others, grip))
                    plan.to_pose("lift", hold(cx, cy, z_cmd)(), [table] + others, goals_fn=hold(cx, cy, z_cmd))
                    chosen = (clear, pitch)
                    break
                except SkillError as e:
                    last = e
            if chosen:
                break
        if chosen:
            break
      if chosen:
          break
    if not chosen:
        plan.restore(start)
        raise SkillError(f"cannot grasp and lift '{object}' with the hands: {last}")
    clear, pitch = chosen
    lifted = cuboid("held", (cx, cy, cz + lift + M["lift_sag_comp_m"]), b["dims"], byaw)
    plan.attach(lifted)                         # the planner now avoids collisions with the held box
    # draw the box toward the chest along the torso's forward axis: shorter lever, less arm sag
    wy = plan.waist[0]
    hold = lambda x, y, z: (lambda: _hands(x, y, z, half - squeeze - grip["extra"], yaw, pitch))
    pull = min(M["pull_in_m"], max(0.0, math.hypot(cx, cy) - M["min_carry_x_m"]))
    if pull > 0.005:                            # optional: skip if it cannot be planned
        px, py = cx - math.cos(wy) * pull, cy - math.sin(wy) * pull
        mk = plan.mark()
        try:
            plan.to_pose("pull the box in", hold(px, py, z_cmd)(), [table] + others, goals_fn=hold(px, py, z_cmd))
        except SkillError:
            plan.restore(mk)
    plan.call("check lift height", lambda: _check_height(ctx, z_lift, "the box did not come up to the lift height"))
    plan.call("check grip after lift", lambda: _check_gap(ctx, gap_of(), "the box slipped during the lift"))
    gap_cmd = dy - 2 * squeeze
    held = {"object": object, "width": dy, "dims": b["dims"], "grasp_above_bottom": zg - b["support_z"],
            "gap_cmd": gap_cmd, "gap_of": gap_of, "pitch": pitch, "clear": clear, "yaw": yaw}
    plan.state["holding"] = held

    def done():
        ctx.bb["holding"] = held
        ctx.bb["objects"].pop(object, None)
    plan.call("holding", done)
    return {"object": object, "width_m": round(dy, 3), "grasp_height_m": round(zg, 3), "lift_m": round(lift, 3),
            "pitch_deg": pitch, "palm_yaw_deg": round(math.degrees(yaw), 1)}


def _secure_grip(ctx, cx, cy, zg, half, squeeze, yaw, pitch, world, grip):
    """Test lift: raise 2 cm, hold, and check the box comes with the hands (height) without slipping
    (palm gap). If not, put it back down, squeeze 1 cm deeper and try again. Runs live on the robot."""
    M = ctx.cfg["manipulation"]
    m = ctx.motion
    for attempt in range(M["grip_retries"] + 1):
        hs = half - squeeze - grip["extra"]
        p = ArmPlan(m)
        p.to_pose("test lift", _hands(cx, cy, zg + M["test_lift_m"], hs, yaw, pitch), world)
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
        if grip["extra"] + 0.01 > M["max_extra_squeeze_m"]:
            break
        p = ArmPlan(m)                          # back down, squeeze deeper, retry
        grip["extra"] += 0.01
        p.to_pose("set back down", _hands(cx, cy, zg, half - squeeze - grip["extra"], yaw, pitch), world)
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
        p.to_pose("line up", _hands(*(cmd[:2] + err), z, half, yaw, h.get("pitch", 0), keep=keep), world)
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
        p.to_pose(label, _hands(*xy, zz, hw, yaw, pitch, keep=keep), world)
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
       "Put the held box down on a surface or on top of another found object, then open the hands.", "arm")
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
    plan.to_pose("lower", _hands(x, y, z, half, yaw, h.get("pitch", 0), keep=keep), world, goals_fn=lower_goal)
    plan.call("release gently", lambda: _gentle_release(ctx, yaw, h, keep, world, corr))
    # planned for feasibility/continuity only; the gentle release above does the real opening
    plan.to_pose("open hands (check only)", _hands(x, y, z, h["width"] / 2 + h["clear"], yaw, h.get("pitch", 0), keep=keep), world)
    plan.steps[-1]["dry"] = True
    plan.state["holding"] = None
    placed = {"center": [x, y, z_s + h["dims"][2] / 2], "dims": h["dims"], "top_z": z_s + h["dims"][2],
              "support_z": z_s, "front_x": x - h["dims"][0] / 2, "distance_m": math.hypot(x, y)}

    def done():
        ctx.bb["holding"] = None
        ctx.bb["objects"][h["object"]] = {**placed, "t": time.time()}
    plan.call("released", done)
    plan.to_pose("move hands away", _hands(x, y, z + 0.08, h["width"] / 2 + h["clear"], yaw, h.get("pitch", 0), keep=keep),
                 world + [cuboid(h["object"], placed["center"], placed["dims"])])
    return {"placed": h["object"], "on": on, "height_m": round(z_s, 3)}


@skill("ready", {}, "Move both arms to the ready pose (hands at chest height, out of the camera view). Do not "
                    "use while holding a box.", "arm")
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
