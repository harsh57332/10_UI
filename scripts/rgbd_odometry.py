"""Camera odometry from recorded head-camera frames (scripts/odom_trial.py saves them): OpenCV 5's RGB-D odometry
(cv2.Odometry, RGB + depth) between consecutive frames, chained; the camera's motion is reported in the robot's level
frame at the start (forward along the heading, left), using the camera mount and the torso IMU of the first sample.
Deterministic: the same frames give the same answer.
  python scripts/rgbd_odometry.py FRAMES.npz [--raw logs/calibration/odom_raw_X.json] [--type rgbd|depth|rgb]"""
import argparse
import json
import math

import cv2
import numpy as np

from common.fk import _rpy
from common.joints import NAMES
from server.geometry import Geometry


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("frames")
    ap.add_argument("--raw", help="joint/IMU samples of the same walk (the camera's pose at the start)")
    ap.add_argument("--type", default="rgbd", choices=("rgbd", "depth", "rgb"))
    ap.add_argument("--step", type=int, default=1, help="use every n-th frame")
    a = ap.parse_args()
    d = np.load(a.frames)
    gray, depth, scale = d["gray"], d["depth"], float(d["depth_scale"])
    k = json.loads(str(d["intr"]))
    K = np.array([[k["fx"], 0, k["ppx"]], [0, k["fy"], k["ppy"]], [0, 0, 1]], np.float32)
    st = cv2.OdometrySettings()
    st.setCameraMatrix(K)
    st.setMinDepth(0.25)
    st.setMaxDepth(3.0)
    st.setMaxTranslation(0.15)
    st.setMaxRotation(15.0)
    typ = {"rgbd": cv2.OdometryType_RGB_DEPTH, "depth": cv2.OdometryType_DEPTH, "rgb": cv2.OdometryType_RGB}[a.type]
    odo = cv2.Odometry(typ, st, cv2.OdometryAlgoType_COMMON)

    def frame(i):
        z = depth[i].astype(np.float32) * scale
        z[(z < 0.25) | (z > 3.0)] = np.nan
        return z, gray[i]

    pose = np.eye(4)                                   # camera k in camera 0 (optical frame)
    fails = 0
    idx = list(range(0, len(gray), a.step))
    for i0, i1 in zip(idx, idx[1:]):
        z0, g0 = frame(i0)
        z1, g1 = frame(i1)
        ok, Rt = odo.compute(z0, g0, z1, g1) if a.type != "depth" else odo.compute(z0, z1)
        if not ok or not np.all(np.isfinite(Rt)):
            fails += 1
            continue
        # Rt maps points of frame i0 into frame i1 (Rt * src = dst): the camera moved by its inverse
        pose = pose @ np.linalg.inv(Rt)
    t_opt = pose[:3, 3]
    # optical frame -> level robot frame at the start: camera mount (torso) and torso attitude (IMU) of the first sample
    geo = Geometry()
    R_level = np.eye(3)
    if a.raw:
        s0 = json.load(open(a.raw))[0]
        q = {NAMES[i] + "_joint": math.radians(v) for i, v in enumerate(s0["q"])}
        T_tp = geo.fk.pose("torso_link", q, base="pelvis")
        r, p, _ = (math.radians(v) for v in s0["rpy"])
        R_level = _rpy(0, p, 0) @ _rpy(r, 0, 0) @ T_tp[:3, :3]      # torso -> level (pelvis IMU attitude)
    v = R_level @ (geo.R @ t_opt)                                    # optical -> torso -> level
    print(json.dumps({"frames": len(idx), "fails": fails, "type": a.type, "forward_cm": round(100 * float(v[0]), 1),
                      "left_cm": round(100 * float(v[1]), 1), "up_cm": round(100 * float(v[2]), 1)}))


if __name__ == "__main__":
    main()
