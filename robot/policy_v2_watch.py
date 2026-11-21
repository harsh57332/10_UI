"""Watch a policy V2 test on the robot itself (20 Hz, local): follows the runner's tick log and switches the runner to
damping the moment a leg joint is further than --limit deg from the stand pose (feet-in-the-air tests). Prints a line
every 0.5 s.  python -m robot.policy_v2_watch --seconds 8 --limit 15"""
import argparse
import glob
import json
import math
import os
import time
import urllib.request

import yaml

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
NAMES = ["L hip p", "L hip r", "L hip y", "L knee", "L ank p", "L ank r", "R hip p", "R hip r", "R hip y", "R knee", "R ank p", "R ank r"]


def damp(why):
    req = urllib.request.Request("http://127.0.0.1:8201/cmd", data=json.dumps({"do": "damp"}).encode(), method="POST")
    urllib.request.urlopen(req, timeout=2).read()
    print(f"!!! DAMPED: {why}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=float, default=8.0)
    ap.add_argument("--limit", type=float, default=15.0)
    ap.add_argument("--tilt", type=float, default=90.0, help="roll or pitch beyond this (deg, IMU) -> damp")
    a = ap.parse_args()
    q0 = yaml.safe_load(open(os.path.join(ROOT, "g1_v2", "params", "deploy.yaml")))["default_joint_pos"][:12]
    f = open(max(glob.glob(os.path.join(ROOT, "logs", "policy_v2", "*.csv")), key=os.path.getmtime))
    f.seek(0, 2)
    t_end, t_print, worst = time.time() + a.seconds, 0.0, (0.0, "")
    while time.time() < t_end:
        line = f.readline()
        if not line:
            time.sleep(0.01)
            continue
        r = line.strip().split(",")
        if len(r) < 38:
            continue
        q = [float(v) for v in r[9:21]]
        dev = [(abs(math.degrees(x - y)), n) for x, y, n in zip(q, q0, NAMES)]
        d, n = max(dev)
        worst = max(worst, (d, n))
        roll, pitch = float(r[7]), float(r[8])
        if r[1] == "policy" and d > a.limit:
            damp(f"{n} {d:.0f} deg from the stand pose")
            return
        if r[1] == "policy" and max(abs(roll), abs(pitch)) > a.tilt:
            damp(f"body tilted: roll {roll:.1f}, pitch {pitch:.1f} deg")
            return
        if time.time() - t_print >= 0.5:
            t_print = time.time()
            print(f"{r[1]:8s} roll {roll:+5.1f} pitch {pitch:+5.1f} | max leg deviation {d:5.1f} deg ({n}); worst so far "
                  f"{worst[0]:.1f} ({worst[1]})", flush=True)
    print(f"done: worst leg deviation {worst[0]:.1f} deg ({worst[1]})", flush=True)


if __name__ == "__main__":
    main()
