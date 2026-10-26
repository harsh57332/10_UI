"""Arm gain / gravity feedforward check: hold the ready pose (no objects near the arms), with the feedforward
on and then off, and compare per joint the tracking error, the motor's torque estimate and the feedforward.
With the feedforward honoured, the error is ~0 and tau_est ~= tau_ff; without it, error ~= tau / kp.
Also reports the waist (held where engaged) and the pelvis sway.

Without --go it only prints what it would do.
  python -m scripts.gain_test --go [--hold 10] [--move] [--release]
"""
import argparse
import math
import time

import numpy as np

from common import joints
from server.robot_link import AgentLink

ARMS = joints.ARMS


def send(link, action, params, timeout=60):
    r = link.send(action, params, wait=True, wait_timeout_s=timeout)
    if not r.get("ok"):
        raise SystemExit(f"{action} failed: {r.get('error')}")
    return r.get("result")


def sample(link, seconds, period=0.2):
    """Per-sample {err_deg, tau_est, tau_ff, temp} for the upper body, and the pelvis pitch/roll."""
    rows, sway = [], []
    t_end = time.monotonic() + seconds
    while time.monotonic() < t_end:
        s = link.state()
        cmd = s["arms"]["q_cmd_deg"]
        if not cmd:
            raise SystemExit("arms are not engaged")
        if s["arms"]["fault"]:
            raise SystemExit(f"agent fault: {s['arms']['fault']}")
        j = {x["index"]: x for x in s["joints"]}
        ff = s["arms"]["ff"]["tau"]
        rows.append({i: (j[i]["deg"] - cmd[i - 12], j[i]["tau"], ff.get(str(i), 0.0), j[i]["temp"])
                     for i in joints.UPPER_BODY})
        sway.append(s["imu"]["pelvis_rpy"][:2])
        time.sleep(period)
    return rows, np.array(sway)


def report(title, rows, sway, kp):
    print(f"\n== {title}: {len(rows)} samples")
    print(f"{'joint':22s} {'err deg':>8s} {'tau_est':>8s} {'tau_ff':>8s} {'kp*err':>8s} {'temp':>5s}")
    for i in joints.UPPER_BODY:
        e, te, tf, tc = (np.array([r[i][k] for r in rows], float) for k in range(4))
        print(f"{joints.NAMES[i]:22s} {e.mean():8.2f} {te.mean():8.2f} {tf.mean():8.2f} "
              f"{-kp[i] * math.radians(e.mean()):8.2f} {tc.max():5.0f}")
    print(f"pelvis roll/pitch mean {sway.mean(0).round(2)} deg, std {sway.std(0).round(2)} deg")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--go", action="store_true", help="really move the arms (to the ready pose)")
    ap.add_argument("--hold", type=float, default=10.0, help="seconds per phase")
    ap.add_argument("--move", action="store_true", help="also raise both arms 20 deg and back during each phase")
    ap.add_argument("--release", action="store_true", help="release the arms at the end (else they stay at ready)")
    a = ap.parse_args()

    link = AgentLink()
    link.connect(link.cfg["robot"]["host"])
    cfg = link.cfg
    s = link.state()
    if s["mode"] not in cfg["modes"]["arm_modes"]:
        raise SystemExit(f"needs one of {cfg['modes']['arm_modes']}, the robot is in '{s['mode']}'")
    kp = {j: g["kp"] for g in cfg["arms"]["gains"].values() for j in g["joints"]}
    ready = [0.0, 0.0, 0.0] + cfg["manipulation"]["ready_deg"]      # waist entries are ignored (held)
    print("plan: arms to ready ->", "feedforward on, hold", a.hold, "s -> off, hold", a.hold, "s -> on again",
          "(+ a 20 deg shoulder-pitch raise and back in each)" if a.move else "")
    if not a.go:
        print("dry run: add --go to move the arms")
        return

    send(link, "set_upperbody", {"q": ready})
    try:
        for on in (True, False):
            send(link, "arm_ff", {"enabled": on})
            time.sleep(2.0)
            if a.move:
                up = {str(i): ready[i - 12] - 20 for i in (15, 22)}
                link.send("play_upperbody", {"keyframes": [{"t": 3.0, "q": up},
                                                           {"t": 3.0, "q": {k: ready[int(k) - 12] for k in up}}]})
            rows, sway = sample(link, a.hold)
            if a.move:
                time.sleep(1.0)
            report(f"feedforward {'ON' if on else 'OFF'}", rows, sway, kp)
    finally:
        send(link, "arm_ff", {"enabled": cfg["arms"]["gravity_ff"]["enabled"]})
    if a.release:
        send(link, "release_upperbody", {})


if __name__ == "__main__":
    main()
