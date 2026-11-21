"""One walking calibration trial through the UI's own path (server /api/cmd -> agent -> the active walker): the IMU
heading before (settled) and after (settled), the command, the walker; appended to
logs/calibration/2026-11-20_custom_walk.jsonl. Tape numbers (left toe tip) are added with --tape to the last record.
  python scripts/walk_trial.py forward 0.4 3          (action, velocity m/s or rad/s, duration s)
  python scripts/walk_trial.py --tape 31.5 -2         (forward cm, side cm: + left, - right; turns: the toe's drift)"""
import argparse
import json
import time

import httpx

LOG = "logs/calibration/2026-11-20_custom_walk.jsonl"
API = "http://localhost:8000"
ACTIONS = ("forward", "backward", "left_aside", "right_aside", "turn_left", "turn_right")


def state():
    return httpx.get(f"{API}/api/status", timeout=5).json()["state"]


def yaw_settled(n=10):
    v = []
    for _ in range(n):
        v.append(state()["imu"]["pelvis_rpy"][2])
        time.sleep(0.05)
    return sum(v) / len(v)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("action", nargs="?", choices=ACTIONS)
    ap.add_argument("velocity", nargs="?", type=float)
    ap.add_argument("duration", nargs="?", type=float)
    ap.add_argument("--tape", nargs=2, type=float, metavar=("FORWARD_CM", "SIDE_CM"))
    a = ap.parse_args()
    if a.tape:
        L = open(LOG).read().splitlines()
        r = json.loads(L[-1])
        r.update(tape_forward_cm=a.tape[0], tape_side_cm=a.tape[1], tape_point="left toe tip", tape_note="side: + left, - right")
        L[-1] = json.dumps(r)
        open(LOG, "w").write("\n".join(L) + "\n")
        print(L[-1])
        return
    s = state()
    if s.get("mode") not in ("locomotion", "manipulation"):
        raise SystemExit(f"not walking: the robot is in '{s.get('mode')}'")
    y0 = yaw_settled()
    r = httpx.post(f"{API}/api/cmd", json={"action": a.action, "params": {"velocity": a.velocity, "duration": a.duration},
                                          "wait": True}, timeout=a.duration + 30).json()
    time.sleep(1.5)                             # the last step and settling
    y1 = yaw_settled()
    rec = {"t": time.strftime("%Y-%m-%d %H:%M:%S"), "walker": s.get("walker"), "action": a.action, "velocity": a.velocity,
           "duration_s": a.duration, "ok": r.get("ok"), "error": r.get("error"), "result": r.get("result"),
           "imu_heading_deg": round(((y1 - y0 + 180) % 360) - 180, 2)}
    open(LOG, "a").write(json.dumps(rec) + "\n")
    print(json.dumps(rec))


if __name__ == "__main__":
    main()
