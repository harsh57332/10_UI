"""Leg odometry check against the tape: one walk through the UI's own path, the odometry running on the PC.
  python scripts/odom_trial.py still 5                 (stand still 5 s: the odometry must stay at ~0)
  python scripts/odom_trial.py forward 0.2 3           (action, velocity, duration: open-loop command, odometry measures)
  python scripts/odom_trial.py --tape 31.5 -2          (left toe tip: forward cm, side cm + left)
Appends to logs/calibration/2026-11-21_odometry.jsonl."""
import argparse
import json
import math
import subprocess
import time

import threading

import httpx
import numpy as np

from common.fk import _rpy
from server.image_client import ImageClient

R_TORSO_LIDAR = _rpy(math.pi, 0.05112069379091391, 0.0)   # URDF mid360_joint: the lidar upside down in the head


def lidar_pose(host):
    out = subprocess.run(["ssh", "-o", "ConnectTimeout=5", f"unitree@{host}",
                          "source /opt/ros/noetic/setup.bash; source ~/lio_ws/devel/setup.bash; python3 ~/lio_ws/odom_pose.py"],
                         capture_output=True, text=True, timeout=20).stdout.strip().splitlines()
    return json.loads(out[-1]) if out else None


def quat_R(q):
    x, y, z, w = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])

from server.odometry import LegOdometry

LOG = "logs/calibration/2026-11-21_odometry.jsonl"
API = "http://localhost:8000"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("action", nargs="?")
    ap.add_argument("velocity", nargs="?", type=float)
    ap.add_argument("duration", nargs="?", type=float)
    ap.add_argument("--tape", nargs=2, type=float)
    a = ap.parse_args()
    if a.tape:
        L = open(LOG).read().splitlines()
        r = json.loads(L[-1])
        r.update(tape_forward_cm=a.tape[0], tape_side_cm=a.tape[1])
        L[-1] = json.dumps(r)
        open(LOG, "w").write("\n".join(L) + "\n")
        print(L[-1])
        return
    st = httpx.get(f"{API}/api/status", timeout=5).json()
    l0 = lidar_pose(st["host"])
    torso_rp = st["state"]["imu"]["torso_rpy"][:2]
    odo = LegOdometry(st["host"])
    odo.record = True
    odo.start()
    # camera frames for the RGB-D odometry (every new frame: rgb as gray, depth as is), from the robot's image server
    cam, frames, stop = ImageClient(), [], threading.Event()
    cam.start(st["host"])

    def grab():
        last = None
        while not stop.is_set():
            f = cam.latest()
            if f is not None and f is not last:
                frames.append((f.t, f.rgb.mean(axis=2).astype(np.uint8), f.depth.copy(), f.depth_scale, f.intrinsics))
                last = f
            time.sleep(0.01)
    th = threading.Thread(target=grab, daemon=True)
    th.start()
    time.sleep(1.0)
    time.sleep(0.5)
    if a.action == "still":
        time.sleep(a.velocity or 5.0)
        r = None
    else:
        r = httpx.post(f"{API}/api/cmd", json={"action": a.action, "params": {"velocity": a.velocity, "duration": a.duration},
                                              "wait": True}, timeout=a.duration + 30).json()
        time.sleep(1.5)                          # the last step and settling
    o = odo.read()
    odo.stop()
    l1 = lidar_pose(st["host"])
    stop.set()
    th.join(timeout=2)
    cam.stop()
    rec = {"t": time.strftime("%Y-%m-%d %H:%M:%S"), "action": a.action, "velocity": a.velocity, "duration_s": a.duration,
           "ok": None if r is None else r.get("ok"), "odometry_forward_cm": round(100 * o["forward_m"], 1),
           "odometry_left_cm": round(100 * o["left_m"], 1), "steps": o["steps"], "samples": o["samples"], "gaps": o["gaps"]}
    if l0 and l1:
        d_body = quat_R(l0["q"]).T @ (np.array(l1["p"]) - np.array(l0["p"]))      # in the lidar's frame at the start
        r_, p_ = (math.radians(v) for v in torso_rp)
        d = _rpy(0, p_, 0) @ _rpy(r_, 0, 0) @ R_TORSO_LIDAR @ d_body              # -> torso -> level robot frame
        R1 = quat_R(l0["q"]).T @ quat_R(l1["q"])
        rec.update(lidar_pose0=l0, lidar_pose1=l1, torso_rp0=torso_rp)
        rec.update(lidar_forward_cm=round(100 * float(d[0]), 1), lidar_left_cm=round(100 * float(d[1]), 1),
                   lidar_up_cm=round(100 * float(d[2]), 1),
                   lidar_turn_deg=round(-math.degrees(math.atan2(R1[1, 0], R1[0, 0])), 1))   # upside down: yaw sign flips
    raw = f"logs/calibration/odom_raw_{time.strftime('%H%M%S')}.json"
    json.dump(odo.raw, open(raw, "w"))
    rec["raw"] = raw
    if frames:
        fz = f"/tmp/claude-1000/-home-kiwi-Documents-Harsh-10-UI/22d741a1-ed07-4ab7-b20c-227d792265fd/scratchpad/frames_{time.strftime('%H%M%S')}.npz"
        np.savez_compressed(fz, t=np.array([x[0] for x in frames]), gray=np.stack([x[1] for x in frames]),
                            depth=np.stack([x[2] for x in frames]), depth_scale=frames[0][3],
                            intr=json.dumps(frames[0][4]))
        rec["frames"], rec["n_frames"] = fz, len(frames)
    open(LOG, "a").write(json.dumps(rec) + "\n")
    print(json.dumps(rec))


if __name__ == "__main__":
    main()
