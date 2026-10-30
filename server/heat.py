"""Motor heat prediction. Per arm motor dT/dt = a*tau^2 - b*(T - amb), fitted on this robot's own logs, with
tau from the same gravity / payload / squeeze model the robot sends as feedforward. Before an arm plan runs it
is replayed from the motors' current temperatures: a plan that would reach the warning is refused before
anything moves, with the start temperature it needs (no stops half way, no guessing)."""
import logging
import math

log = logging.getLogger("heat")

import numpy as np

from common import config, joints
from common.dynamics import Gravity

from .motion import SkillError

_GRAVITY = None


def _gravity(cfg):
    global _GRAVITY
    if _GRAVITY is None:
        _GRAVITY = Gravity(config.path(cfg["robot"]["urdf"]), cfg["arms"]["gravity_ff"], config.tcp_frames())
    return _GRAVITY


def segments(cfg, plan):
    """[(seconds, upper-body q (17 rad), payload kg, squeeze N)] along a plan, the way it will run."""
    th, a, M = cfg["thermal"], cfg["arms"], cfg["manipulation"]
    idx = {n + "_joint": i for i, n in enumerate(joints.NAMES)}
    q = dict(plan.q0)
    payload = squeeze = 0.0
    out = []
    upper = lambda: np.array([q[i] for i in joints.UPPER_BODY])
    for s in plan.steps:
        if s["kind"] == "traj":
            if s.get("dry"):
                q.update({idx[n]: v for n, v in zip(s["traj"]["joint_names"], s["traj"]["positions"][-1])})
                continue
            rows = s["traj"]["positions"]
            seconds = s["traj"]["duration"] * (a["playback_slowdown_holding"] if s["holding"] else a["playback_slowdown"])
            k = max(1, int(seconds / th["step_s"]))
            for r in rows[::max(1, len(rows) // k)] + [rows[-1]]:
                q.update({idx[n]: v for n, v in zip(s["traj"]["joint_names"], r)})
                out.append((seconds / (k + 1), upper(), payload, squeeze))
        elif s["kind"] == "waist":
            q.update({12 + k: v for k, v in enumerate(s["waist"])})
            out.append((s["seconds"], upper(), payload, squeeze))
        else:
            label = s["label"]
            if label in th["squeeze_on"]:
                squeeze = M["squeeze_n"]
            if label in th["payload_on"]:
                payload = M["payload_kg"]
            out.append((th["call_s"].get(label, th["call_s"]["default"]), upper(), payload, squeeze))
            if label in th["force_off"]:
                payload = squeeze = 0.0
    return out


WAIST_PITCH = 14
WATCHED = joints.ARMS + [WAIST_PITCH]     # the waist pitch holds everything above it: with the 0.9 kg Inspire hands
                                          # reaching out it carried 19 Nm and went 47 -> 98 C (2026-10-29)


def simulate(cfg, segs, T0):
    """{motor index: peak C} from start temperatures T0 {motor index: C} (arm motors and the waist pitch)."""
    th = cfg["thermal"]
    g = _gravity(cfg)
    T = {j: float(T0[j]) for j in WATCHED}
    peak = dict(T)
    for seconds, q, payload, squeeze in segs:
        g.payload_kg, g.squeeze_n = payload, squeeze
        tau = g.torques(q)
        tau[WAIST_PITCH] = th["waist_torque_scale"] * g.waist_pitch(q)
        n = max(1, int(math.ceil(seconds / th["step_s"])))
        for _ in range(n):
            for j in WATCHED:
                p = th["motors"].get(str(j), th["default"])
                amb = p.get("amb", T0[j])
                T[j] += seconds / n * (p["a"] * (th["torque_scale"] * tau[j]) ** 2 - p["b"] * (T[j] - amb))
                peak[j] = max(peak[j], T[j])
    g.payload_kg = g.squeeze_n = 0.0
    return peak


def check(ctx, plan):
    """Refuse a plan whose predicted motor peak would reach the warning; returns the predicted peaks."""
    cfg = ctx.cfg
    limit = cfg["arms"]["warn_motor_temp_c"] - cfg["thermal"]["margin_c"]
    T0 = {j["index"]: j["temp"] for j in ctx.link.state()["joints"] if j["index"] in WATCHED}
    segs = segments(cfg, plan)
    peak = simulate(cfg, segs, T0)
    j = max(peak, key=peak.get)
    if peak[j] <= limit:
        return {joints.NAMES[k]: round(v) for k, v in peak.items() if v >= limit - 15}
    if peak[WAIST_PITCH] > limit:               # never run anyway: the balance joint (no start_ok_c exception)
        raise SkillError(f"predicted waist pitch peak {peak[WAIST_PITCH]:.0f} C (limit {limit:.0f} C, now "
                         f"{T0[WAIST_PITCH]} C): the arms stay out too long or too far for the waist motor that holds "
                         f"them; refused")
    if max(T0.values()) <= cfg["thermal"]["start_ok_c"]:     # user 2026-10-26: starting this cool, run it anyway (the
        log.warning("predicted %s peak %.0f C (limit %.0f C), but every arm motor starts <= %d C: allowed",   # runtime
                    joints.NAMES[j], peak[j], limit, cfg["thermal"]["start_ok_c"])   # guard still holds at 95 C)
        return {joints.NAMES[k]: round(v) for k, v in peak.items() if v >= limit - 15}
    lo, hi = 0.0, 60.0                          # how much cooler every motor must start (the model is linear in T)
    for _ in range(12):
        d = (lo + hi) / 2
        if max(simulate(cfg, segs, {k: v - d for k, v in T0.items()}).values()) <= limit:
            hi = d
        else:
            lo = d
    need = T0[j] - hi
    floor = cfg["thermal"]["motors"].get(str(j), {}).get("amb", 0.0)
    why = ("wait for it to cool" if need > floor + 2 else
           "this task holds the arms too long or too far out for the motors even from cold: it needs a closer or "
           "quicker motion")
    raise SkillError(f"predicted {joints.NAMES[j]} peak {peak[j]:.0f} C (limit {limit:.0f} C): it needs to start at "
                     f"<= {need:.0f} C (now {T0[j]} C); {why}")
