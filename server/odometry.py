"""Leg odometry: how far the robot has really walked, from its joint encoders and IMU (no camera, no command timing).

The foot on the ground does not move. Each sample, both feet are placed relative to the pelvis by the robot model
(leg forward kinematics from the 12 leg joint angles) and turned into the gravity/heading frame by the pelvis IMU
(roll, pitch, yaw). The pelvis moves over the stance foot: body = stance foot - (stance foot relative to the pelvis).
The stance foot is the lower one; it changes only when the other foot is lower by switch_dz_m (both are on the floor
in double support). Displacements add up step by step; reported along the heading at the start (forward, left).

Runs on the PC by polling the agent's /state (166 Hz possible; sampled at rate_hz), so the robot program is untouched.
Deterministic: the same joint/IMU stream gives the same distance.
"""
import math
import threading
import time

import httpx
import numpy as np

from common import config
from common.fk import FK, _rpy
from common.joints import NAMES

SOLE = np.array([0.035, 0.0, -0.035])        # ankle_roll_link: under the middle of the foot (contact spheres x -0.05..0.12)
LEG = [n + "_joint" for n in NAMES[:12]]


def _yaw_R(y):
    return _rpy(0.0, 0.0, y)


class LegOdometry:
    def __init__(self, host, rate_hz=60.0, switch_dz_m=0.01):
        cfg = config.load()
        self.fk = FK(config.path(cfg["robot"]["urdf"]))
        self.url = f"http://{host}:{cfg['agent']['port']}/state"
        self.dt, self.switch_dz = 1.0 / rate_hz, switch_dz_m
        self.http = httpx.Client(timeout=1.0)
        self.lock = threading.Lock()
        self.thread, self.stop_ev = None, threading.Event()
        self.reset_state()

    def reset_state(self):
        self.B = np.zeros(3)                      # pelvis position (world, gravity/heading frame), from the start
        self.F = None                             # stance foot position (world)
        self.stance = None                        # "left" | "right"
        self.yaw0 = None
        self.steps, self.samples, self.gaps = 0, 0, 0
        self.last_t = None
        self.raw = [] if getattr(self, "record", False) else None

    def _feet(self, s):
        q = {LEG[j["index"]]: math.radians(j["deg"]) for j in s["joints"] if j["index"] < 12}
        r, p, y = (math.radians(v) for v in s["imu"]["pelvis_rpy"])
        R = _rpy(0.0, 0.0, y) @ _rpy(0.0, p, 0.0) @ _rpy(r, 0.0, 0.0)
        out = {}
        for side in ("left", "right"):
            T = self.fk.pose(f"{side}_ankle_roll_link", q, base="pelvis")
            out[side] = R @ (T[:3, :3] @ SOLE + T[:3, 3])
        return out, y

    def update(self, s):
        if self.raw is not None:
            self.raw.append({"t": s.get("t"), "q": [j["deg"] for j in s["joints"]], "rpy": s["imu"]["pelvis_rpy"]})
        feet, yaw = self._feet(s)
        with self.lock:
            if self.yaw0 is None:
                self.yaw0 = yaw
            if self.stance is None:
                self.stance = min(feet, key=lambda k: feet[k][2])
                self.F = self.B + feet[self.stance]
            other = "right" if self.stance == "left" else "left"
            if feet[other][2] < feet[self.stance][2] - self.switch_dz:
                self.stance, self.steps = other, self.steps + 1
                self.F = self.B + feet[other]     # the new stance foot is where it touched down
            self.B = self.F - feet[self.stance]
            self.samples += 1

    def _loop(self):
        nxt = time.perf_counter()
        while not self.stop_ev.is_set():
            try:
                s = self.http.get(self.url).json()
                if self.last_t is not None and s.get("t") is not None and s["t"] - self.last_t > 3 * self.dt:
                    self.gaps += 1
                self.last_t = s.get("t")
                self.update(s)
            except (httpx.HTTPError, ValueError, KeyError):
                self.gaps += 1
            nxt += self.dt
            d = nxt - time.perf_counter()
            if d > 0:
                time.sleep(d)
            else:
                nxt = time.perf_counter()

    def start(self):
        self.stop()
        self.reset_state()
        self.stop_ev.clear()
        self.thread = threading.Thread(target=self._loop, daemon=True, name="leg_odometry")
        self.thread.start()

    def stop(self):
        if self.thread is not None:
            self.stop_ev.set()
            self.thread.join(timeout=2)
            self.thread = None

    def read(self):
        """Walked since start: forward and left along the heading at the start (m), heading change (deg), steps."""
        with self.lock:
            if self.yaw0 is None:
                return {"forward_m": 0.0, "left_m": 0.0, "yaw_deg": 0.0, "steps": 0, "samples": 0, "gaps": self.gaps}
            d = _yaw_R(self.yaw0).T @ self.B
            return {"forward_m": round(float(d[0]), 4), "left_m": round(float(d[1]), 4),
                    "steps": self.steps, "samples": self.samples, "gaps": self.gaps}
