"""Robot-side Agent: blindly executes complete commands from the server.

- One envelope endpoint (POST /cmd). A command runs only if its whole JSON body arrived and parsed.
- Command ids are remembered, so a re-transmitted command is never executed twice.
- One motion at a time (walk / mode / arms); others are rejected while busy. Switching to
  damping is always accepted and pre-empts the running motion.
- Audio (speak / play file) has its own slot and can run during motion.
- Keeps executing if WiFi drops; nothing depends on the connection after a command is accepted.

Run: python -m robot.agent   (from the app dir, see scripts/robot.sh)
"""
import base64
import logging
import math
import threading
import time
from collections import OrderedDict

import uvicorn
from fastapi import FastAPI, HTTPException

from common import config, envelope, joints
from common.config import mode_id, mode_name
from common.envelope import Request

from .g1 import G1, rebuild_gravity
from .inspire import HandError, Hands, probe

log = logging.getLogger("agent")
CFG = config.load()
app = FastAPI(title="g1ui-agent")
g1: G1 = None

_history: "OrderedDict[str, envelope.Response]" = OrderedDict()
_hist_lock = threading.Lock()
_slots = {"motion": None, "audio": None}   # {"id", "action", "started", "stop": Event}
_slot_lock = threading.Lock()
_audio_stop = threading.Event()


class Rejected(Exception):
    pass


# ---------------------------------------------------------------- helpers
def _num(params, key, lo, hi, default=None):
    v = params.get(key, default)
    if v is None:
        raise ValueError(f"'{key}' is required")
    v = float(v)
    if not (lo <= v <= hi):
        raise ValueError(f"'{key}'={v} out of range [{lo}, {hi}]")
    return v


def _current_mode():
    return mode_name(g1.fsm_id())


def _require_mode(allowed_key):
    allowed = CFG["modes"][allowed_key]
    cur = _current_mode()
    if cur not in allowed:
        raise Rejected(f"needs mode {allowed}, robot is in '{cur}'")


# ---------------------------------------------------------------- motion actions
def _walk(params, stop, vx=0.0, vy=0.0, wz=0.0, turn=False):
    loc = CFG["locomotion"]
    vmax = loc["turn_speed_max"] if turn else loc["speed_max"]
    vdef = loc["turn_speed_default"] if turn else loc["speed_default"]
    v = _num(params, "velocity", 0.0, vmax, vdef)
    d = _num(params, "duration", 0.0, loc["duration_max_s"])
    _require_mode("walk_modes")
    limit = loc.get("tilt_stop_deg")
    rpy0 = g1.imu().get("pelvis_rpy") if limit else None
    tilt = None
    bias = _num(params, "vy_bias_mps", -0.2, 0.2, 0.0)      # a steady sideways push to cancel the gait's drift (+ = left)
    bias_x = _num(params, "vx_bias_mps", -0.2, 0.2, 0.0)    # ...and forward/back (- = back): holding a box the robot creeps
    g1.velocity(vx * v + bias_x, vy * v + bias, wz * v, d)  # forward while turning (2026-11-18)
    t_end = time.monotonic() + d
    while time.monotonic() < t_end and not stop.is_set():
        stop.wait(0.05)
        rpy = g1.imu().get("pelvis_rpy") if rpy0 else None
        if rpy and max(abs(rpy[0] - rpy0[0]), abs(rpy[1] - rpy0[1])) > limit:    # tipping: stop walking, hold still
            tilt = [round(rpy[0] - rpy0[0], 1), round(rpy[1] - rpy0[1], 1)]
            break
    g1.stop_move()
    return {"velocity": v, "duration": d, "interrupted": stop.is_set() or tilt is not None, "tilt_deg": tilt}


def _turn_by(params, stop):
    """Turn in place by angle_deg (+ = left) using pelvis IMU yaw as feedback."""
    loc = CFG["locomotion"]
    angle = math.radians(_num(params, "angle_deg", -180.0, 180.0))
    v = _num(params, "velocity", 0.05, loc["turn_speed_max"], loc["turn_speed_default"])
    _require_mode("walk_modes")
    yaw0 = g1.yaw()
    lead = v * loc["turn_stop_lead_s"]          # stop early by what the robot turns while stopping
    t_end = time.monotonic() + abs(angle) / v * 2.0 + 2.0
    g1.velocity(0.0, 0.0, math.copysign(v, angle), abs(angle) / v * 2.0 + 2.0)
    turned = 0.0
    while time.monotonic() < t_end and not stop.is_set():
        turned = math.remainder(g1.yaw() - yaw0, math.tau)
        if abs(turned) >= abs(angle) - lead:
            break
        time.sleep(0.02)
    g1.stop_move()
    stop.wait(loc["turn_settle_s"])
    turned = math.remainder(g1.yaw() - yaw0, math.tau)
    return {"target_deg": round(math.degrees(angle), 1), "turned_deg": round(math.degrees(turned), 1),
            "interrupted": stop.is_set()}


def _set_mode(params, stop):
    target = params.get("mode")
    ids = CFG["modes"]["ids"]
    if target not in ids:
        raise ValueError(f"unknown mode '{target}', expected one of {list(ids)}")
    cur = _current_mode()
    if target == cur:
        return {"mode": cur, "fsm_id": ids[cur], "changed": False}
    if target in CFG["modes"]["disabled"]:
        raise Rejected(f"mode '{target}' is disabled (not in priority)")
    if target != "damping" and target not in CFG["modes"]["transitions"].get(cur, []):
        raise Rejected(f"'{cur}' -> '{target}' is not an allowed transition")

    if target == "damping":
        g1.arms.drop()
    elif g1.arms.engaged and target not in CFG["modes"]["arm_modes"]:
        g1.arms.release()
    g1.set_fsm(ids[target])

    # SetFsmId returns 0 even when the firmware ignores the request, so read the mode back.
    t_end = time.monotonic() + CFG["modes"]["switch_timeout_s"]
    while time.monotonic() < t_end:
        now = g1.fsm_id()
        if now == ids[target]:
            return {"mode": target, "fsm_id": now, "changed": True}
        time.sleep(0.2)
    now = g1.fsm_id()
    raise RuntimeError(f"firmware did not switch to '{target}' ({ids[target]}); robot is in "
                       f"'{mode_name(now)}' ({now})")


def _upper_deg_to_rad(index, deg):
    return math.radians(joints.clamp_deg(index, float(deg)))


def _wait_move(stop, duration):
    """Wait for an arm move; fail at once if the arms were released for over-temperature."""
    t_end = time.monotonic() + duration
    while not stop.is_set() and time.monotonic() < t_end:
        if g1.arms.fault:
            raise RuntimeError(g1.arms.fault)
        stop.wait(min(0.1, max(0.0, t_end - time.monotonic())))


def _set_upperbody(params, stop):
    q = params.get("q")
    if not isinstance(q, list) or len(q) != 17:
        raise ValueError("'q' must be a list of 17 angles in degrees (motor index 12..28)")
    _require_mode("arm_modes")
    target = {idx: _upper_deg_to_rad(idx, d) for idx, d in zip(joints.UPPER_BODY, q)}
    duration = params.get("duration")
    if duration is None:
        g1.arms.engage()
        delta = max(abs(math.degrees(target[j] - c)) for j, c in zip(joints.UPPER_BODY, g1.arms.q))
        duration = max(0.5, delta / CFG["arms"]["max_speed_dps"])
    _wait_move(stop, g1.arms.move([(float(duration), target)], smooth=True))
    return {"duration": round(float(duration), 2), "interrupted": stop.is_set()}


def _play_upperbody(params, stop):
    """keyframes: [{"t": seconds, "q": {"18": 70, ...}}]; unnamed joints hold, angles in degrees."""
    frames = params.get("keyframes")
    if not isinstance(frames, list) or not frames:
        raise ValueError("'keyframes' must be a non-empty list of {t, q}")
    parsed = []
    for k, f in enumerate(frames):
        t = float(f.get("t", 0))
        if t <= 0:
            raise ValueError(f"keyframe {k}: 't' must be > 0")
        targets = {}
        for idx, deg in (f.get("q") or {}).items():
            idx = int(idx)
            if idx not in joints.UPPER_BODY:
                raise ValueError(f"keyframe {k}: joint {idx} is not an upper-body joint (12..28)")
            targets[idx] = _upper_deg_to_rad(idx, deg)
        parsed.append((t, targets))
    _require_mode("arm_modes")
    _wait_move(stop, g1.arms.move(parsed, smooth=params.get("interp", "smooth") == "smooth",
                                  recovery=bool(params.get("recovery"))))
    return {"frames": len(parsed), "interrupted": stop.is_set()}


def _release_upperbody(params, stop):
    g1.arms.release()
    return {"released": True}


def _not_priority(params, stop):
    raise Rejected("squat is currently not in priority")


# ---------------------------------------------------------------- audio actions
def _speak(params, stop):
    text = str(params.get("text", "")).strip()
    if not text or len(text) > CFG["audio"]["tts_max_chars"]:
        raise ValueError(f"'text' must be 1..{CFG['audio']['tts_max_chars']} characters")
    g1.speak(text)
    return {"chars": len(text)}


def _play_audio(params, stop):
    """pcm_b64: 16 kHz mono int16 little-endian PCM (the server converts uploaded files)."""
    a = CFG["audio"]
    pcm = base64.b64decode(params.get("pcm_b64", ""))
    if not pcm:
        raise ValueError("'pcm_b64' is empty")
    _audio_stop.clear()
    stream_id = str(int(time.time() * 1000))
    for off in range(0, len(pcm), a["chunk_bytes"]):
        if _audio_stop.is_set():
            break
        g1.audio.PlayStream(a["app_name"], stream_id, pcm[off:off + a["chunk_bytes"]])
        time.sleep(a["chunk_gap_s"])
    # wait for the robot to finish the buffered tail before stopping the stream
    chunks = math.ceil(len(pcm) / a["chunk_bytes"])
    tail_s = len(pcm) / (2.0 * a["sample_rate"]) - chunks * a["chunk_gap_s"]
    _audio_stop.wait(max(0.0, tail_s))
    g1.audio.PlayStop(a["app_name"])
    return {"seconds": round(len(pcm) / (2.0 * a["sample_rate"]), 1), "stopped": _audio_stop.is_set()}


# ---------------------------------------------------------------- instant actions
def _stop_audio(params):
    _audio_stop.set()
    g1.audio.PlayStop(CFG["audio"]["app_name"])
    return {"stopped": True}


def _headlight(params):
    g1.headlight.set(str(params.get("hex", "")))
    return {"hex": g1.headlight.color}


def _processing(params):
    g1.headlight.processing(str(params.get("source", "server")), bool(params.get("on")))
    return {"sources": sorted(g1.headlight.sources)}


def _set_volume(params):
    g1.set_volume(_num(params, "volume", 0, 100))
    return {"volume": g1.get_volume()}


def _get_mode(params):
    i = g1.fsm_id()
    return {"mode": mode_name(i), "fsm_id": i}


hands = None                    # the Inspire hands' link (polled even when absent: they are detected when they answer)


def _detected():
    """The hand profile the wrists show: inspire when an Inspire hand answers (they come as a pair), else dummy."""
    return "inspire" if hands is not None and any(h["connected"] for h in hands.state().values()) else "dummy"


def _watch_hands():
    """Auto-detection, one way only: dummy -> inspire when Inspire hands answer (while the arms are released; until
    then reported as pending). Never back on its own: hands that stop answering are still on the wrists (2026-10-29:
    the left hand dropped off the network and the robot switched to the rubber-hand model while the Inspire hands
    were fitted). With the dummy hands refitted, the agent's restart detects them."""
    warned = False
    while True:
        time.sleep(CFG["end_effector"]["inspire"]["detect_s"])
        name = _detected()
        if CFG["end_effector"]["detect"] != "auto" or name == config.hands():
            warned = False
            continue
        if name == "inspire" and not g1.arms.engaged:
            config.use_hands(name)
            rebuild_gravity(g1.arms)
            log.info("hands detected: %s (gravity model %s)", name, CFG["robot"]["urdf"])
        elif name == "dummy" and not warned:
            warned = True
            log.warning("the Inspire hands do not answer: keeping the Inspire model (restart the agent if they were removed)")


def _get_end_effector_state(params):
    ee = CFG["end_effector"]
    name = config.hands()
    out = {"type": name, "detected": name != "dummy", "detect": ee["detect"], "pending": _detected() == "inspire" != name}
    if name == "inspire" and hands is not None:
        out.update(model=ee["inspire"]["model"], hands=hands.state())
    return out


def _set_end_effector_state(params):
    """{"side": "left" | "right", "state": "open" | "close"} (UI section 5 only; skills never call it)."""
    if hands is None or config.hands() != "inspire":
        raise Rejected("no end effector detected (dummy hands)")
    try:
        return hands.command(str(params.get("side")), str(params.get("state")))
    except HandError as e:
        raise Rejected(str(e))


def _arm_ff(params):
    """Feedforward: {"enabled": bool, "payload_kg": total mass held between the palms, "squeeze_n": force each
    palm pushes toward the other}; all optional."""
    ff = CFG["arms"]["gravity_ff"]
    payload, squeeze = params.get("payload_kg"), params.get("squeeze_n")
    if payload is not None:
        payload = _num(params, "payload_kg", 0.0, ff["max_payload_kg"])
    if squeeze is not None:
        squeeze = _num(params, "squeeze_n", 0.0, ff["max_squeeze_n"])
    enabled = params.get("enabled")
    return g1.arms.set_feedforward(None if enabled is None else bool(enabled), payload, squeeze)


# name -> (function, kind). kind: motion | audio | instant
ACTIONS = {
    "forward":      (lambda p, s: _walk(p, s, vx=+1), "motion"),
    "backward":     (lambda p, s: _walk(p, s, vx=-1), "motion"),
    "left_aside":   (lambda p, s: _walk(p, s, vy=+1), "motion"),
    "right_aside":  (lambda p, s: _walk(p, s, vy=-1), "motion"),
    "turn_left":    (lambda p, s: _walk(p, s, wz=+1, turn=True), "motion"),
    "turn_right":   (lambda p, s: _walk(p, s, wz=-1, turn=True), "motion"),
    "turn_by":      (_turn_by, "motion"),
    "stand2squat":  (_not_priority, "motion"),
    "squat2stand":  (_not_priority, "motion"),
    "set_mode":     (_set_mode, "motion"),
    "set_upperbody": (_set_upperbody, "motion"),
    "play_upperbody": (_play_upperbody, "motion"),
    "release_upperbody": (_release_upperbody, "motion"),
    "speak":        (_speak, "audio"),
    "play_audio":   (_play_audio, "audio"),
    "stop_audio":   (_stop_audio, "instant"),
    "set_volume":   (_set_volume, "instant"),
    "get_volume":   (lambda p: {"volume": g1.get_volume()}, "instant"),
    "headlight":    (_headlight, "instant"),
    "processing":   (_processing, "instant"),
    "get_mode":     (_get_mode, "instant"),
    "get_29_states": (lambda p: {"joints": g1.joint_states()}, "instant"),
    "get_battery_status": (lambda p: {"battery": g1.battery()}, "instant"),
    "get_imu":      (lambda p: g1.imu(), "instant"),
    "get_end_effector_state": (_get_end_effector_state, "instant"),
    "set_end_effector_state": (_set_end_effector_state, "instant"),
    "arm_ff":       (_arm_ff, "instant"),
}


# ---------------------------------------------------------------- command plumbing
def _remember(resp):
    with _hist_lock:
        _history[resp.id] = resp
        _history.move_to_end(resp.id)
        while len(_history) > CFG["agent"]["command_history"]:
            _history.popitem(last=False)
    return resp


def _run_slot(slot, req, fn, stop, t0):
    try:
        result = fn(req.params, stop)
        _remember(envelope.done(req.id, result, t0))
    except (Rejected, ValueError) as e:
        _remember(envelope.failed(req.id, str(e), t0, status="rejected"))
    except Exception as e:
        log.exception("action %s failed", req.action)
        _remember(envelope.failed(req.id, str(e), t0))
    finally:
        with _slot_lock:
            if _slots[slot] and _slots[slot]["id"] == req.id:
                _slots[slot] = None


@app.post("/cmd")
def cmd(req: Request):
    t0 = time.time()
    with _hist_lock:
        if req.id in _history:            # re-transmission: never execute twice
            return _history[req.id]
    if req.action not in ACTIONS:
        return _remember(envelope.failed(req.id, f"unknown action '{req.action}'", t0, status="rejected"))
    fn, kind = ACTIONS[req.action]

    if kind == "instant":
        try:
            return _remember(envelope.done(req.id, fn(req.params), t0))
        except (Rejected, ValueError) as e:
            return _remember(envelope.failed(req.id, str(e), t0, status="rejected"))
        except Exception as e:
            log.exception("action %s failed", req.action)
            return _remember(envelope.failed(req.id, str(e), t0))

    slot = kind
    preempt = req.action == "set_mode" and req.params.get("mode") == "damping"
    with _slot_lock:
        busy = _slots[slot]
        if busy and not preempt:
            return _remember(envelope.failed(
                req.id, f"busy with '{busy['action']}' ({busy['id']})", t0, status="rejected"))
        if busy and preempt:
            busy["stop"].set()
        stop = threading.Event()
        _slots[slot] = {"id": req.id, "action": req.action, "started": t0, "stop": stop}
        resp = _remember(envelope.running(req.id, None, t0))
    threading.Thread(target=_run_slot, args=(slot, req, fn, stop, t0), daemon=True,
                     name=f"{slot}:{req.action}").start()
    return resp


@app.get("/cmd/{cmd_id}")
def cmd_status(cmd_id: str):
    with _hist_lock:
        if cmd_id not in _history:
            raise HTTPException(404, "unknown command id")
        return _history[cmd_id]


@app.get("/health")
def health():
    return {"service": "g1ui-agent", "robot": "unitree_g1", "urdf": CFG["robot"]["urdf"],
            "mode_machine": CFG["robot"]["mode_machine"], "ready": g1 is not None}


@app.get("/state")
def state():
    low = g1.lowstate(wait_s=0.2)
    try:
        fsm = g1.fsm_id()
        mode = {"mode": mode_name(fsm), "fsm_id": fsm}
    except Exception as e:
        mode = {"mode": None, "fsm_id": None, "error": str(e)}
    with _slot_lock:
        busy = {k: ({"id": v["id"], "action": v["action"], "for_s": round(time.time() - v["started"], 1)}
                    if v else None) for k, v in _slots.items()}
    return {
        "t": time.time(),
        "connected_dds": low is not None,
        "lowstate_age_s": g1.low.age(),
        "mode_machine": low.mode_machine if low else None,
        **mode,
        "busy": busy,
        "arms": {"engaged": g1.arms.engaged, "moving": g1.arms.moving(), "weight": round(g1.arms.weight, 3),
                 "loop_hz": round(g1.arms.loop_hz, 1), "max_gap_ms": round(g1.arms.max_gap_ms, 1),
                 "q_cmd_deg": g1.arms.q_cmd_deg(), "fault": g1.arms.fault,
                 "warning": g1.arms.warning, "waist_held": CFG["arms"]["waist_hold_joints"],
                 "ff": {"enabled": g1.arms.ff_enabled, "payload_kg": round(g1.arms.gravity.payload_kg, 3),
                        "squeeze_n": round(g1.arms.gravity.squeeze_n, 2),
                        "tau": {str(j): round(t, 2) for j, t in g1.arms.tau_ff.items()}}},
        "joints": g1.joint_states(),
        "imu": g1.imu(),
        "battery": g1.battery(),
        "headlight": {"hex": g1.headlight.color, "processing": sorted(g1.headlight.sources)},
        "end_effector": _get_end_effector_state({}),
    }


def main():
    global g1, hands
    logging.basicConfig(level=CFG["logging"]["level"], format="%(asctime)s %(name)s %(levelname)s %(message)s")
    ee = CFG["end_effector"]
    if ee["detect"] == "auto":                  # before the arms' gravity model is built: it depends on the hands
        config.use_hands("inspire" if probe(ee["inspire"]) else "dummy")
    log.info("hands: %s (%s)", config.hands(), "detected" if ee["detect"] == "auto" else "configured")
    g1 = G1(CFG)
    hands = Hands(ee["inspire"])
    threading.Thread(target=_watch_hands, daemon=True, name="hands_watch").start()
    log.info("G1 ready, agent on :%s", CFG["agent"]["port"])
    uvicorn.run(app, host="0.0.0.0", port=CFG["agent"]["port"], log_level="warning")


if __name__ == "__main__":
    main()
