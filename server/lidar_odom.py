"""Walked distance from the head LiDAR (Livox MID-360 + FAST-LIO on the robot, ~/lio_ws, served at :8210/pose).

The LiDAR is in the head, mounted upside down (URDF mid360_joint), 1.2 m above the floor: a lean of 3 deg moves it ~6 cm
while the feet stay. So a displacement is taken at a body point (the pelvis, or the left toe tip) through the robot
model: the point's offset from the LiDAR (leg/waist FK from the joint angles) is turned level with the pelvis IMU at both
ends and added. Tape 2026-11-21 (left toe, 3 walks): 47.6/48, 48.2/49, -27.1/-26 cm (leg odometry was 9-10 cm off).
Deterministic: the same poses and joint angles give the same answer."""
import logging
import math
import subprocess
import threading
import time

import httpx
import numpy as np

from common import config
from common.fk import FK, _rpy
from common.joints import NAMES

log = logging.getLogger("lidar")
R_TORSO_LIDAR = _rpy(math.pi, 0.05112069379091391, 0.0)
POINTS = {"pelvis": ("pelvis", np.zeros(3)), "left_toe": ("left_ankle_roll_link", np.array([0.14, 0.0, -0.035]))}


def quat_R(q):
    x, y, z, w = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


class LidarOdom:
    def __init__(self, host):
        cfg = config.load()
        self.url = f"http://{host}:{cfg.get('lidar', {}).get('pose_port', 8210)}/pose"
        self.max_age = cfg.get("lidar", {}).get("max_age_s", 0.5)
        self.fk = FK(config.path(cfg["robot"]["urdf"]))
        self.http = httpx.Client(timeout=1.0)

    def available(self):
        try:
            return self.pose() is not None
        except Exception:
            return False

    def pose(self):
        r = self.http.get(self.url)
        if r.status_code != 200:
            return None
        d = r.json()
        return d if d.get("age_s", 9.9) <= self.max_age else None

    def snapshot(self, state):
        """The LiDAR pose with the robot's joint angles and IMU at (nearly) the same moment."""
        p = self.pose()
        if p is None:
            raise RuntimeError("no fresh lidar pose (is ~/lio_ws/run_lio.sh running on the robot?)")
        return {"lidar": p, "q": [j["deg"] for j in sorted(state["joints"], key=lambda j: j["index"])],
                "rpy": state["imu"]["pelvis_rpy"], "torso_rp": state["imu"]["torso_rpy"][:2]}

    def _offset(self, snap, point):
        """The body point from the LiDAR, in the level frame of that moment (pelvis IMU roll/pitch)."""
        q = {NAMES[i] + "_joint": math.radians(v) for i, v in enumerate(snap["q"])}
        Tl = self.fk.pose("mid360_link", q, base="pelvis")
        link, off = POINTS[point]
        Tp = np.eye(4) if link == "pelvis" else self.fk.pose(link, q, base="pelvis")
        v = (Tp[:3, :3] @ off + Tp[:3, 3]) - Tl[:3, 3]
        r, p, _ = (math.radians(x) for x in snap["rpy"])
        return _rpy(0, p, 0) @ _rpy(r, 0, 0) @ v

    def displacement(self, s0, s1, point="pelvis"):
        """How far `point` moved from s0 to s1, in the robot's level frame at s0: forward, left (m), turned (deg, + left)."""
        R0, R1 = quat_R(s0["lidar"]["q"]), quat_R(s1["lidar"]["q"])
        r, p = (math.radians(x) for x in s0["torso_rp"])
        M = _rpy(0, p, 0) @ _rpy(r, 0, 0) @ R_TORSO_LIDAR                  # lidar frame at s0 -> level robot frame
        d_head = M @ (R0.T @ (np.array(s1["lidar"]["p"]) - np.array(s0["lidar"]["p"])))
        Rr = M @ (R0.T @ R1) @ M.T
        yaw = math.atan2(Rr[1, 0], Rr[0, 0])
        d = d_head + _rpy(0, 0, yaw) @ self._offset(s1, point) - self._offset(s0, point)
        return float(d[0]), float(d[1]), math.degrees(yaw)


class LidarAuto:
    """Starts the LiDAR odometry on the robot by itself (~/lio_ws/run_lio.sh, it does not survive a reboot): only when the
    robot stands in a walking mode, nothing is running and body and legs have been still for still_s (FAST-LIO takes its
    gravity and IMU bias from the first seconds: never on a robot swinging on the crane). Moves nothing on the robot."""

    def __init__(self, link, busy):
        c = config.load()
        self.link, self.busy, self.cfg, self.user = link, busy, c["lidar"], c["robot"]["ssh_user"]
        self.state, self.detail, self.next_try, self.hist = "off", "", 0.0, []
        self.http = httpx.Client(timeout=1.0)
        threading.Thread(target=self._loop, daemon=True, name="lidar_auto").start()

    def status(self):
        text = {"off": "off", "waiting": "waiting for the robot to stand still", "starting": "starting, keep still",
                "running": "running", "error": "error"}[self.state]
        level = {"running": "ok", "error": "bad", "starting": "warn", "waiting": "warn"}.get(self.state, "")
        return {"state": self.state, "text": text + (f" · {self.detail}" if self.detail else ""), "level": level}

    def _pose(self, host):
        try:
            r = self.http.get(f"http://{host}:{self.cfg['pose_port']}/pose")
            d = r.json() if r.status_code == 200 else {}
            return d if d.get("age_s", 9.9) <= self.cfg["max_age_s"] else None
        except (httpx.HTTPError, ValueError):
            return None

    def _ssh(self, host, cmd):
        subprocess.run(["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=5", f"{self.user}@{host}", cmd], check=True,
                       capture_output=True, timeout=20)

    def _stop(self, host):
        self._ssh(host, 'pkill -f "[r]un_lio.sh"; pkill -f "[r]oslaunch livox_ros_driver2"; pkill -f "[r]oslaunch fast_lio"; '
                        'pkill -x fastlio_mapping; pkill -x livox_ros_driv; pkill -f "[p]ose_server.py"; sleep 1; true')

    def _still(self, s):
        c, arms, now = s.get("custom") or {}, s.get("arms") or {}, time.monotonic()
        ready = s.get("mode") in ("locomotion", "manipulation") and not self.busy() and not arms.get("moving") \
            and not any((s.get("busy") or {}).values()) and not c.get("vel_left_s")
        if not ready:
            self.hist = []
            return False
        v = list(s["imu"]["pelvis_rpy"]) + [j["deg"] for j in s["joints"] if j["index"] < 12]
        self.hist.append((now, v))
        if float(np.ptp(np.array([h[1] for h in self.hist]), axis=0).max()) > self.cfg["still_deg"]:
            self.hist = [(now, v)]                  # it moved: the still time starts again
            return False
        return now - self.hist[0][0] >= self.cfg["still_s"]

    def _start(self, host):
        self.state, self.detail = "starting", ""
        self._stop(host)
        self._ssh(host, "(setsid nohup ~/lio_ws/run_lio.sh > /dev/null 2>&1 < /dev/null &) ; true")
        t_end, a = time.monotonic() + self.cfg["start_timeout_s"], None
        while a is None and time.monotonic() < t_end:
            time.sleep(1.0)
            a = self._pose(host)
        if a is None:
            raise RuntimeError("no position from the LiDAR")
        time.sleep(2.0)
        b = self._pose(host)
        if b is None or math.dist(a["p"], b["p"]) > self.cfg["max_drift_m"]:
            self._stop(host)
            raise RuntimeError("position not steady")
        self.state, self.detail, self.hist = "running", "", []
        log.info("LiDAR odometry started")

    def _tick(self):
        host = self.link.host
        if not host:
            self.state, self.detail, self.hist = "off", "", []
        elif self._pose(host) is not None:
            self.state, self.detail, self.hist = "running", "", []
        elif not self.cfg["auto_start"]:
            self.state, self.detail = "off", "auto start disabled"
        elif time.monotonic() >= self.next_try:
            if self._still(self.link.state()):
                self._start(host)
            else:
                self.state, self.detail = "waiting", ""

    def _loop(self):
        while True:
            time.sleep(0.5)
            try:
                self._tick()
            except httpx.HTTPError:
                self.state, self.hist = "waiting", []
            except Exception as e:
                log.warning("LiDAR auto start: %s", e)
                self.state, self.detail, self.hist = "error", f"{str(e)[:60]}; retrying", []
                self.next_try = time.monotonic() + self.cfg["retry_s"]
