"""The head camera's real angle in its mount, from its own accelerometer (D435i) and the robot's torso IMU.

2026-10-29 every flat surface (floor, case, box tops) read ~13 deg tilted: the camera sat ~13 deg further down than
the URDF's d435_joint (moved in the head between 26 Oct and the hands' fitting). Both sensors see gravity: the
accelerometer in the camera frame, the IMU in the torso frame; the rotation between what the camera measures and what
the URDF predicts is the mount fix (pitch and roll; yaw is not seen by gravity).
Robot standing still. The camera service is stopped for ~10 s (the accelerometer has its own stream).
  python -m scripts.camera_tilt            measure, print the fix and how far the current config is off
  python -m scripts.camera_tilt --write    also put the fix into config.yaml (camera_mount.optical_fix_rpy_deg)
Then restart the server (it reads the camera mount at start).
The D435i accelerometer is not factory calibrated (~1-2 deg). The fix is then refined on the level surfaces in view
(floor, table, box tops; the large flat ones that agree within 1 deg): their normal must be the IMU's up. The
same check afterwards tells how far the camera model is off (expect < 0.5 deg)."""
import math
import re
import subprocess
import sys

import httpx
import numpy as np

from common import config
from common.fk import _axis_angle, _rpy
from server.geometry import Geometry

REMOTE = r'''
import time, numpy as np, pyrealsense2 as rs
pipe, rc = rs.pipeline(), rs.config()
rc.enable_stream(rs.stream.accel)
for k in range(5):
    try:
        pipe.start(rc); break
    except Exception as e:
        time.sleep(2)
acc, t0 = [], time.time()
while time.time() - t0 < 4:
    for f in pipe.wait_for_frames():
        if f.is_motion_frame():
            d = f.as_motion_frame().get_motion_data(); acc.append([d.x, d.y, d.z])
pipe.stop()
a = np.array(acc[40:])
print("ACC", *a.mean(0), a.std(0).max(), len(a))
'''


def accel(cfg):
    """Mean accelerometer reading (m/s^2, camera optical frame: x right, y down, z forward) over 4 s."""
    root = config.path(".")
    py = f"{cfg['robot']['conda_dir']}/envs/{cfg['robot']['conda_env']}/bin/python"
    subprocess.run([str(root / "scripts/robot.sh"), "stop", "camera"], check=True, capture_output=True)
    try:
        r = subprocess.run(["ssh", f"{cfg['robot']['ssh_user']}@{cfg['robot']['host']}", f"timeout 40 {py} -"],
                           input=REMOTE, text=True, capture_output=True, timeout=60)
    finally:
        subprocess.run([str(root / "scripts/robot.sh"), "start", "camera"], check=True, capture_output=True)
    line = [l for l in r.stdout.splitlines() if l.startswith("ACC")]
    if not line:
        raise SystemExit(f"no accelerometer data: {r.stdout[-300:]} {r.stderr[-300:]}")
    v = [float(x) for x in line[0].split()[1:]]
    if v[3] > 0.5:
        raise SystemExit(f"the camera was moving (accelerometer std {v[3]:.2f} m/s^2): stand still and retry")
    return np.array(v[:3])


def level_normal(geo):
    """The torso-frame normal of the level surfaces in a fine depth frame (large near-horizontal planes that agree),
    with their sizes; None if fewer than two agree."""
    from server.image_client import ImageClient
    import time
    cam = ImageClient()
    cam.start(config.load()["robot"]["host"])
    t0 = time.time()
    while cam.latest() is None and time.time() - t0 < 20:
        time.sleep(0.2)
    for attempt in range(8):                     # the camera service restarts after the accelerometer read
        try:
            f = cam.fine(10)
            break
        except RuntimeError:
            if attempt == 7:
                raise
            time.sleep(4)
    cam.stop()
    k = f.intrinsics
    vs, us = np.mgrid[0:f.depth.shape[0]:2, 0:f.depth.shape[1]:2]
    z = f.depth[vs, us].astype(np.float32) * f.depth_scale
    keep = (z > 0.15) & (z < 3.0)
    us, vs, z = us[keep], vs[keep], z[keep]
    P = np.stack([(us - k["ppx"]) * z / k["fx"], (vs - k["ppy"]) * z / k["fy"], z], 1) @ geo.R.T + geo.t
    rng, rem, planes = np.random.default_rng(0), np.ones(len(P), bool), []
    for _ in range(6):
        idx = np.nonzero(rem)[0]
        if len(idx) < 1500:
            break
        Q, best = P[idx], None
        for _ in range(1500):
            a, b, c = Q[rng.choice(len(Q), 3, replace=False)]
            n = np.cross(b - a, c - a)
            if np.linalg.norm(n) < 1e-9:
                continue
            n = n / np.linalg.norm(n) * (1 if n[2] > 0 else -1)
            if n[2] < math.cos(math.radians(25)):
                continue
            inl = np.abs((Q - a) @ n) < 0.004
            if best is None or inl.sum() > best.sum():
                best = inl
        if best is None or best.sum() < 1500:
            break
        S = Q[best]
        n = np.linalg.svd(S - S.mean(0))[2][2]
        planes.append((len(S), n if n[2] > 0 else -n, float(S[:, 2].mean())))
        rem[idx[best]] = False
    if not planes:
        return None, []
    ref = planes[0][1]
    agree = [p for p in planes if math.degrees(math.acos(min(1.0, float(p[1] @ ref)))) < 1.0]
    if len(agree) < 2:
        return None, planes
    n = sum(c * p for c, p, _ in agree)
    return n / np.linalg.norm(n), planes


def depression(up_cam):
    """Degrees the optical axis points below the horizon, given 'up' in the optical frame."""
    return math.degrees(math.asin(-up_cam[2]))


def main():
    cfg = config.load()
    st = httpx.get(f"http://127.0.0.1:{cfg['server']['port']}/api/status", timeout=5).json()["state"]
    r_deg, p_deg, _ = st["imu"]["torso_rpy"]
    a = accel(cfg)
    st2 = httpx.get(f"http://127.0.0.1:{cfg['server']['port']}/api/status", timeout=5).json()["state"]
    r2, p2, _ = st2["imu"]["torso_rpy"]
    if max(abs(r2 - r_deg), abs(p2 - p_deg)) > 0.5:
        raise SystemExit("the torso moved during the measurement: stand still and retry")
    up_meas = a / np.linalg.norm(a)                    # specific force at rest points up
    up_torso = _rpy(math.radians(r_deg), math.radians(p_deg), 0.0).T @ np.array([0.0, 0.0, 1.0])
    geo, geo0 = Geometry(), Geometry(fix=False)
    up_urdf = geo0.R.T @ up_torso                       # what the URDF mount predicts the camera feels
    up_now = geo.R.T @ up_torso                         # ... the mount in use now (with the current fix)
    ax = np.cross(up_meas, up_urdf)
    ang = math.asin(min(1.0, np.linalg.norm(ax)))
    C = _axis_angle(ax, ang) if ang > 1e-6 else np.eye(3)     # C @ up_meas = up_urdf
    off = math.degrees(math.acos(min(1.0, float(up_meas @ up_now))))
    print(f"accelerometer {np.round(a, 3)} (|g| {np.linalg.norm(a):.2f}), torso IMU roll {r_deg:.2f} pitch {p_deg:.2f} deg")
    print(f"camera looks {depression(up_meas):.1f} deg down; the URDF mount says {depression(up_urdf):.1f}, "
          f"the mount in use says {depression(up_now):.1f} ({off:.1f} deg off)")
    # refine on the level surfaces seen with the accelerometer's fix
    geo_a = Geometry(fix=False)
    geo_a.R = geo0.R @ C
    n, planes = level_normal(geo_a)
    for cnt, pn, h in planes:
        print(f"  flat surface: {cnt} points at height {h:+.3f} m (torso), "
              f"{math.degrees(math.acos(min(1.0, float(pn @ up_torso)))):.1f} deg from level")
    if n is None:
        print("level surfaces: fewer than two large ones agree -> the accelerometer's fix only")
    else:
        ax = np.cross(n, up_torso)
        ang = math.asin(min(1.0, np.linalg.norm(ax)))
        print(f"level surfaces: {math.degrees(ang):.2f} deg from the IMU's level after the accelerometer's fix -> refined")
        if ang > 1e-6:
            D = _axis_angle(ax, ang)                     # torso frame: D @ n = up
            C = geo0.R.T @ D @ geo0.R @ C
    fix = [math.degrees(x) for x in config._mat_rpy(C)]
    print(f"fix (optical frame rpy, deg): [{fix[0]:.2f}, {fix[1]:.2f}, {fix[2]:.2f}]")
    if "--write" in sys.argv:
        p = config.path("config/config.yaml")
        s = open(p).read()
        line = f"  optical_fix_rpy_deg: [{fix[0]:.2f}, {fix[1]:.2f}, {fix[2]:.2f}]"
        s2, n = re.subn(r"^  optical_fix_rpy_deg: \[.*\]$", line, s, flags=re.M)
        if n != 1:
            raise SystemExit("config.yaml has no camera_mount.optical_fix_rpy_deg line to update")
        open(p, "w").write(s2)
        print("written to config.yaml; restart the server")


if __name__ == "__main__":
    main()
