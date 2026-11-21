"""Run the walking policy V2 on the G1 (low-level control). A separate program from the agent: stop the agent first.

Nothing is sent to the motors until 'release' (Unitree's controller is switched off then). States:
  idle     Unitree in control, nothing published
  damp     all joints soft (kp 0, kd): the robot hangs on the crane
  default  the joints go to the policy's stand pose over default_s (from where they are), then hold it
  policy   the legs follow the policy (from 'default' only); waist and arms held at home; velocity command zero
           unless 'vel' (vx, vy, wz for a duration, then zero again)
Safety, every 20 ms tick: no fresh lowstate for stale_s, the body tilted past tilt_max_deg, or a non-finite policy
output -> damp. Leg targets clipped to the URDF limits; commands capped (caps below the trained ranges).
Control (HTTP on port 8201, JSON):  GET /state     POST /cmd {"do": "release" | "damp" | "default" | "policy" |
  "vel", "vx", "vy", "wz", "seconds" | "stop" (= damp) | "unitree" (give the robot back to Unitree's controller)}
  python -m robot.policy_v2_run          (on the robot, in the app dir)
Every tick is logged to logs/policy_v2/<time>.csv for the accuracy measurements."""
import json
import logging
import math
import os
import signal
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np
from cyclonedds.core import Policy, Qos
from cyclonedds.sub import DataReader
from cyclonedds.topic import Topic
from unitree_sdk2py.comm.motion_switcher.motion_switcher_client import MotionSwitcherClient
from unitree_sdk2py.core.channel import ChannelFactory, ChannelFactoryInitialize, ChannelPublisher
from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_, LowState_
from unitree_sdk2py.utils.crc import CRC

from common import config, joints
from robot.policy_v2 import PolicyV2, quat_to_gravity, quat_to_yaw

log = logging.getLogger("policy_v2")

PORT = 8201
DEFAULT_S = 3.0                 # to the stand pose this slowly
STALE_S = 0.1                   # lowstate older than this: damp
TILT_MAX_DEG = 40.0             # the body tilted more than this (falling): damp
DAMP_KD = np.array([3.0] * 12 + [3.0] * 3 + [1.0] * 14)   # soft: the robot hangs, joints slowed only
CAPS = {"vx": (-0.3, 0.4), "vy": (-0.2, 0.2), "wz": (-0.6, 0.6), "seconds": 10.0}   # first tests, under the trained ranges


class Fresh:
    """The newest rt/lowstate every tick (the agent's reader keeps a sample up to 0.1 s: 5 policy steps) and its age."""

    def __init__(self, participant):
        self.reader = DataReader(participant, Topic(participant, "rt/lowstate", LowState_), qos=Qos(Policy.History.KeepLast(1)))
        self.msg, self.t = None, 0.0

    def get(self):
        s = self.reader.take(N=1)
        if s:
            self.msg, self.t = s[0], time.monotonic()
        return self.msg if self.msg is not None and time.monotonic() - self.t <= STALE_S else None


class Runner:
    def __init__(self):
        cfg = config.load()
        ChannelFactoryInitialize(0, cfg["robot"]["dds_interface"])
        part = ChannelFactory()._ChannelFactory__participant
        self.low = Fresh(part)
        self.mode_machine = None
        self.pub = ChannelPublisher("rt/lowcmd", LowCmd_)
        self.pub.Init()
        self.cmd = unitree_hg_msg_dds__LowCmd_()
        self.crc = CRC()
        self.msc = MotionSwitcherClient()
        self.msc.SetTimeout(5.0)
        self.msc.Init()
        self.pol = PolicyV2()
        lim = np.radians(np.array(joints.limits_deg(), float))
        self.lo, self.hi = lim[:, 0] + 0.02, lim[:, 1] - 0.02
        self.lock = threading.Lock()
        self.state, self.error, self.released_from = "idle", None, None
        self.vel, self.vel_until = (0.0, 0.0, 0.0), 0.0
        self.ramp = None                         # (t0, q_start) while going to the stand pose
        self.info = {}
        os.makedirs(config.path("logs/policy_v2"), exist_ok=True)
        self.logf = open(config.path(f"logs/policy_v2/{time.strftime('%Y%m%d_%H%M%S')}.csv"), "w")
        self.logf.write("t,state,vx,vy,wz,cmd_wz,yaw_deg,roll_deg,pitch_deg," + ",".join(f"q{i}" for i in range(29)) + ","
                        + ",".join(f"a{i}" for i in range(12)) + "\n")

    # ------------------------------------------------------------ control
    def do(self, c):
        what = c.get("do")
        with self.lock:
            if what == "release":
                if self.state != "idle":
                    return {"ok": False, "error": f"already released ({self.state})"}
                code, res = self.msc.CheckMode()
                self.released_from = (res or {}).get("name")
                for _ in range(10):
                    code, res = self.msc.CheckMode()
                    if not (res or {}).get("name"):
                        break
                    self.msc.ReleaseMode()
                    time.sleep(1.0)
                else:
                    return {"ok": False, "error": f"Unitree's controller did not release ({res})"}
                self._to("damp", f"released from '{self.released_from}'")
            elif what in ("damp", "stop"):
                if self.state != "idle":
                    self._to("damp", "asked")
            elif what == "default":
                if self.state == "idle":
                    return {"ok": False, "error": "release first"}
                m = self.low.get()
                if m is None:
                    return {"ok": False, "error": "no fresh lowstate"}
                secs = float(np.clip(float(c.get("seconds", DEFAULT_S)), DEFAULT_S, 15.0))   # never faster than DEFAULT_S
                self.ramp = (time.monotonic(), np.array([m.motor_state[i].q for i in range(29)]), secs)
                self._to("default", "asked")
            elif what == "policy":
                if self.state != "default" or self.ramp is None or time.monotonic() - self.ramp[0] < self.ramp[2]:
                    return {"ok": False, "error": "only from the stand pose (do 'default' and wait until it is there)"}
                self.pol.reset()
                self.vel, self.vel_until = (0.0, 0.0, 0.0), 0.0
                self._to("policy", "asked")
            elif what == "vel":
                if self.state != "policy":
                    return {"ok": False, "error": "the policy is not running"}
                v = [float(np.clip(float(c.get(k, 0.0)), *CAPS[k])) for k in ("vx", "vy", "wz")]
                s = float(np.clip(float(c.get("seconds", 2.0)), 0.0, CAPS["seconds"]))
                self.vel, self.vel_until = tuple(v), time.monotonic() + s
                log.info("velocity %s for %.1f s", v, s)
            elif what == "unitree":
                if self.state not in ("idle", "damp"):
                    return {"ok": False, "error": "damp first"}
                name = c.get("mode") or self.released_from or "ai"
                code = self.msc.SelectMode(name)
                self._to("idle", f"Unitree's controller selected ('{name}', code {code})")
            else:
                return {"ok": False, "error": f"unknown '{what}'"}
            return {"ok": True, **self.status()}

    def _to(self, state, why):
        log.info("%s -> %s (%s)", self.state, state, why)
        self.state = state
        if state == "damp":
            self.ramp = None

    def status(self):
        return {"state": self.state, "error": self.error, "vel": self.vel,
                "vel_left_s": round(max(0.0, self.vel_until - time.monotonic()), 1), **self.info}

    # ------------------------------------------------------------ 50 Hz loop
    def loop(self):
        dt = self.pol.dt
        nxt = time.perf_counter()
        while True:
            nxt += dt
            try:
                self.tick()
            except Exception as e:
                log.exception("tick")
                with self.lock:
                    self.error = str(e)
                    if self.state != "idle":
                        self._to("damp", f"error: {e}")
            delay = nxt - time.perf_counter()
            if delay > 0:
                time.sleep(delay)
            else:
                nxt = time.perf_counter()

    def tick(self):
        with self.lock:
            state = self.state
        if state == "stopping":
            return
        m = self.low.get()
        if m is None:
            if state != "idle":
                with self.lock:
                    self._to("damp", "no fresh lowstate")
                self.publish(None, None, np.zeros(29), DAMP_KD, np.zeros(29))
            return
        q = np.array([m.motor_state[i].q for i in range(29)])
        dq = np.array([m.motor_state[i].dq for i in range(29)])
        quat = np.array(m.imu_state.quaternion, float)
        gyro = np.array(m.imu_state.gyroscope, float)
        g = quat_to_gravity(quat)
        tilt = math.degrees(math.acos(max(-1.0, min(1.0, -g[2]))))
        yaw = quat_to_yaw(quat)
        rpy = [math.degrees(v) for v in m.imu_state.rpy]
        self.info = {"tilt_deg": round(tilt, 1), "yaw_deg": round(math.degrees(yaw), 1), "mode_machine": m.mode_machine,
                     "roll_deg": round(rpy[0], 1), "pitch_deg": round(rpy[1], 1)}
        if state == "idle":
            return
        if state in ("default", "policy") and tilt > TILT_MAX_DEG:
            with self.lock:
                self._to("damp", f"tilted {tilt:.0f} deg")
            state = "damp"
        a = np.zeros(12)
        vel = (0.0, 0.0, 0.0)
        if state == "damp":
            self.publish(m, None, np.zeros(29), DAMP_KD, np.zeros(29))
        elif state == "default":
            t0, q_start, secs = self.ramp
            s = min(1.0, (time.monotonic() - t0) / secs)
            s = s * s * (3 - 2 * s)              # smooth start and stop
            target = q_start + (self.pol.q0 - q_start) * s
            self.publish(m, np.clip(target, self.lo, self.hi), self.pol.kp, self.pol.kd, self.pol.tau_ff)
        elif state == "policy":
            if time.monotonic() < self.vel_until:
                vel = self.vel
            self.pol.command(*vel, yaw)
            target, kp, kd, ff, a = self.pol.step(q, dq, quat, gyro)
            self.publish(m, np.clip(target, self.lo, self.hi), kp, kd, ff)
        self.logf.write(f"{time.monotonic():.3f},{state},{vel[0]:.3f},{vel[1]:.3f},{vel[2]:.3f},{self.pol.cmd[2]:.3f},"
                        f"{math.degrees(yaw):.2f},{rpy[0]:.2f},{rpy[1]:.2f}," + ",".join(f"{v:.4f}" for v in q) + ","
                        + ",".join(f"{v:.4f}" for v in a) + "\n")

    def publish(self, m, target, kp, kd, ff):
        c = self.cmd
        c.mode_pr = 0
        if m is not None:
            self.mode_machine = m.mode_machine
        if self.mode_machine is not None:
            c.mode_machine = self.mode_machine
        for i in range(29):
            mc = c.motor_cmd[i]
            mc.mode = 1
            mc.q = float(target[i]) if target is not None else 0.0
            mc.dq = 0.0
            mc.kp = float(kp[i]) if target is not None else 0.0
            mc.kd = float(kd[i])
            mc.tau = float(ff[i])
        c.crc = self.crc.Crc(c)
        self.pub.Write(c)

    def shutdown(self, *_):
        with self.lock:
            ours = self.state != "idle"          # idle: Unitree is in control, never publish
            if ours:
                self._to("damp", "program stopping")
                self.state = "stopping"          # the loop publishes nothing more; only the damping below
        for _ in range(25 if ours else 0):       # half a second of damping before the commands stop
            self.publish(self.low.get(), None, np.zeros(29), DAMP_KD, np.zeros(29))
            time.sleep(0.02)
        self.logf.close()
        os._exit(0)


def serve(r):
    class H(BaseHTTPRequestHandler):
        def _send(self, obj):
            b = json.dumps(obj).encode()
            self.send_response(200)
            self.send_header("content-type", "application/json")
            self.send_header("content-length", str(len(b)))
            self.end_headers()
            self.wfile.write(b)

        def do_GET(self):
            self._send(r.status())

        def do_POST(self):
            n = int(self.headers.get("content-length") or 0)
            try:
                self._send(r.do(json.loads(self.rfile.read(n) or b"{}")))
            except Exception as e:
                log.exception("command")
                self._send({"ok": False, "error": str(e)})

        def log_message(self, *a):
            pass
    ThreadingHTTPServer(("0.0.0.0", PORT), H).serve_forever()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    r = Runner()
    signal.signal(signal.SIGINT, r.shutdown)
    signal.signal(signal.SIGTERM, r.shutdown)
    threading.Thread(target=serve, args=(r,), daemon=True).start()
    log.info("policy V2 runner ready on port %d (idle: Unitree in control, nothing sent)", PORT)
    r.loop()
