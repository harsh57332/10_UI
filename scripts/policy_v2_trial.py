"""One trial of the policy comparison (colleague's V2 on the robot), from the runner's tick log: the IMU heading change
over the last walk command (settled before / settled after) and whether the robot stood still again; appended to
logs/calibration/2026-11-19_policy_compare.jsonl. The tape numbers are added with --tape FORWARD_CM SIDE_CM
(side: + left, - right; for turns FORWARD/SIDE are the toe's drift) to the last record.
  python scripts/policy_v2_trial.py --section forward --trial 1 --vx 0.2 --seconds 3 --expected 60
  python scripts/policy_v2_trial.py --tape 31.5 -2"""
import argparse
import json
import math
import subprocess
import time

LOG = "logs/calibration/2026-11-19_policy_compare.jsonl"
ROBOT = "unitree@192.168.0.120"
CSV = "/home/unitree/g1ui/app/logs/policy_v2"


def segment():
    """The last ~3000 ticks of the newest runner log (60 s)."""
    out = subprocess.run(["ssh", ROBOT, f"f=$(ls -t {CSV}/*.csv | head -1); tail -3000 $f"], capture_output=True, text=True,
                         timeout=30).stdout
    return [r.split(",") for r in out.splitlines() if r.count(",") >= 49 and not r.startswith("t,")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--section")
    ap.add_argument("--trial", type=int)
    ap.add_argument("--vx", type=float, default=0.0)
    ap.add_argument("--vy", type=float, default=0.0)
    ap.add_argument("--wz", type=float, default=0.0)
    ap.add_argument("--seconds", type=float)
    ap.add_argument("--expected", type=float, help="cm (walks) or deg (turns)")
    ap.add_argument("--policy", default="v2")
    ap.add_argument("--tape", nargs=2, type=float, metavar=("FORWARD_CM", "SIDE_CM"))
    a = ap.parse_args()
    if a.tape:
        L = open(LOG).read().splitlines()
        r = json.loads(L[-1])
        r.update(tape_forward_cm=a.tape[0], tape_side_cm=a.tape[1], tape_point="left toe tip", tape_note="side: + left, - right")
        if r["section"] in ("forward", "backward", "left aside", "right aside"):
            along = a.tape[0] if r["section"] in ("forward", "backward") else abs(a.tape[1])
            r["achieved_pct"] = round(100 * abs(along) / r["expected"], 1)
        L[-1] = json.dumps(r)
        open(LOG, "w").write("\n".join(L) + "\n")
        print(L[-1])
        return
    rows = segment()
    want = (a.vx, a.vy, a.wz)
    on = [i for i, r in enumerate(rows) if r[1] == "policy" and all(abs(float(r[2 + k]) - want[k]) < 1e-3 for k in range(3))]
    if not on:
        raise SystemExit("no ticks with that command in the last 60 s")
    i1 = on[-1]
    i0 = i1
    while i0 - 1 in on:
        i0 -= 1
    yaw = [float(r[6]) for r in rows]
    t = [float(r[0]) for r in rows]
    knee = [math.degrees(float(r[12])) for r in rows]
    before = yaw[max(0, i0 - 25):i0]
    after = yaw[-25:]
    d = ((sum(after) / len(after) - sum(before) / len(before) + 180) % 360) - 180
    rec = {"t": time.strftime("%Y-%m-%d %H:%M:%S"), "policy": a.policy, "section": a.section, "trial": a.trial,
           "cmd": {"vx": a.vx, "vy": a.vy, "wz": a.wz, "seconds": round(t[i1] - t[i0] + 0.02, 2)}, "expected": a.expected,
           "unit": "deg" if a.wz else "cm", "imu_heading_deg": round(d, 2),
           "settled": max(knee[-50:]) - min(knee[-50:]) < 0.5}
    if a.wz:
        rec["achieved_pct"] = round(100 * abs(d) / a.expected, 1)
    open(LOG, "a").write(json.dumps(rec) + "\n")
    print(json.dumps(rec))


if __name__ == "__main__":
    main()
