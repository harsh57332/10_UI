"""The colleague's walking policy V2 as the robot's walking controller (agent walker 'custom'), chosen at connect.

Unitree's controller is switched off (MotionSwitcher release) and ONE 50 Hz loop here publishes rt/lowcmd for all 29
motors: the legs follow the policy (robot/policy_v2.PolicyV2), the waist and arms follow the agent's Arms (the same
moves as under Unitree, blended in by the Arms' weight like arm_sdk) or are held at the policy's home pose.
Unitree's modes are emulated so the UI and the skills work unchanged:
  zero_torque / damping  -> damp: every joint soft (the robot hangs on the crane)
  stand_up               -> the policy's stand pose, reached in stand_s from where the joints are (ON THE CRANE: the
                            policy cannot get up from the floor)
  locomotion / manipulation -> the policy balances and walks; velocity commands as SetVelocity (vx, vy, wz, duration)
Safety every tick: no fresh lowstate, the body tilted past tilt_max_deg, or a non-finite policy output -> damp.
Tested stand-alone first as robot/policy_v2_run.py (2026-11-19/20, crane, tape)."""
import logging
import math
import threading
import time

import numpy as np
from cyclonedds.core import Policy, Qos
from cyclonedds.sub import DataReader
from cyclonedds.topic import Topic
from unitree_sdk2py.comm.motion_switcher.motion_switcher_client import MotionSwitcherClient
from unitree_sdk2py.core.channel import ChannelPublisher
from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_, LowState_
from unitree_sdk2py.utils.crc import CRC

from common import joints

from .policy_v2 import PolicyV2, quat_to_gravity, quat_to_yaw

log = logging.getLogger("custom_walker")

FSM = {"damp": 1, "stand": 4, "locomotion": 801, "manipulation": 501}
UB = np.array(joints.UPPER_BODY)


class CustomWalker:
    def __init__(self, g1, participant, cfg):
        self.g1, self.cfg = g1, cfg               # cfg: config custom_walker
        self.reader = DataReader(participant, Topic(participant, "rt/lowstate", LowState_), qos=Qos(Policy.History.KeepLast(1)))
        self.msg, self.msg_t = None, 0.0
        self.pub = None
        self.cmd = unitree_hg_msg_dds__LowCmd_()
        self.crc = CRC()
        self.msc = MotionSwitcherClient()
        self.msc.SetTimeout(5.0)
        self.msc.Init()
        self.pol = PolicyV2()
        lim = np.radians(np.array(joints.limits_deg(), float))
        self.lo, self.hi = lim[:, 0] + 0.02, lim[:, 1] - 0.02
        self.damp_kd = np.array(cfg["damp_kd"], float)
        self.lock = threading.Lock()
        self.active = False                     # Unitree released, this loop drives the motors
        self.state = "damp"                     # damp | stand | locomotion | manipulation
        self.damp_id = 1                        # the mode id damp reports: 1 damping, 0 zero torque (both soft here)
        self.ramp = None                        # (t0, q_start, seconds)
        self.vel, self.vel_until = (0.0, 0.0, 0.0), 0.0
        self.released_from = None
        self.mode_machine = None
        self.ub_prev = None
        self.error = None
        self.thread = None

    # ------------------------------------------------------------ switching controllers
    def unitree_released(self):
        """True when Unitree's controller is off (MotionSwitcher reports no mode)."""
        code, res = self.msc.CheckMode()
        return code == 0 and not (res or {}).get("name")

    def activate(self):
        """Unitree off, this loop on (in damp). The robot must hang on the crane in Unitree's damping or zero torque."""
        code, res = self.msc.CheckMode()
        self.released_from = (res or {}).get("name") or self.released_from or "ai"
        for _ in range(10):
            code, res = self.msc.CheckMode()
            if not (res or {}).get("name"):
                break
            self.msc.ReleaseMode()
            time.sleep(1.0)
        else:
            raise RuntimeError(f"Unitree's controller did not release ({res})")
        if self.pub is None:
            self.pub = ChannelPublisher("rt/lowcmd", LowCmd_)
            self.pub.Init()
        with self.lock:
            self.state, self.ramp, self.error = "damp", None, None
            self.active = True
        if self.thread is None:
            self.thread = threading.Thread(target=self._loop, daemon=True, name="custom_walker")
            self.thread.start()
        log.info("custom walker on (Unitree released from '%s'): damping", self.released_from)

    def deactivate(self):
        """Back to Unitree's controller; only from damp (hanging on the crane)."""
        with self.lock:
            if self.active and self.state != "damp":
                raise RuntimeError("put the robot in Damping (on the crane) first")
            self.active = False                 # the loop stops publishing at once
        code = self.msc.SelectMode(self.released_from or "ai")
        log.info("custom walker off: Unitree's controller '%s' selected (code %s)", self.released_from or "ai", code)
        return code

    # ------------------------------------------------------------ Unitree-like interface (G1 delegates here)
    def fsm_id(self):
        return self.damp_id if self.state == "damp" else FSM[self.state]

    def set_fsm(self, fsm_id):
        with self.lock:
            if fsm_id in (0, 1):                # zero torque is soft damping here too (the robot hangs on the crane)
                self._to("damp", "asked")
                self.damp_id = fsm_id
                return
            if fsm_id == 4:
                if self.state != "damp":
                    raise RuntimeError("stand up only from Damping")
                m = self._fresh()
                if m is None:
                    raise RuntimeError("no fresh lowstate from the robot")
                self.ramp = (time.monotonic(), np.array([m.motor_state[i].q for i in range(29)]), float(self.cfg["stand_s"]))
                self._to("stand", "asked")
                return
            want = {801: "locomotion", 501: "manipulation"}.get(fsm_id)
            if want is None:
                raise RuntimeError(f"mode {fsm_id} is not available with the custom walker")
            if self.state in ("locomotion", "manipulation"):
                self._to(want, "asked")
                return
            if self.state != "stand":
                raise RuntimeError("walking modes only from Stand Up")
            left = self.ramp[0] + self.ramp[2] - time.monotonic()
        if left > 0:                            # still going to the stand pose: wait for it
            time.sleep(left + 0.1)
        with self.lock:
            if self.state != "stand":
                raise RuntimeError(f"left the stand pose while waiting ({self.state})")
            self.pol.reset()
            self.vel, self.vel_until = (0.0, 0.0, 0.0), 0.0
            self._to(want, "asked")

    def velocity(self, vx, vy, wz, duration):
        with self.lock:
            if self.state not in ("locomotion", "manipulation"):
                raise RuntimeError(f"not walking: the custom walker is in '{self.state}'")
            self.vel, self.vel_until = (float(vx), float(vy), float(wz)), time.monotonic() + float(duration)

    def stop_move(self):
        with self.lock:
            self.vel, self.vel_until = (0.0, 0.0, 0.0), 0.0

    def status(self):
        return {"active": self.active, "state": self.state, "error": self.error, "vel": list(self.vel),
                "vel_left_s": round(max(0.0, self.vel_until - time.monotonic()), 2)}

    # ------------------------------------------------------------ the loop
    def _to(self, state, why):
        if state != self.state:
            log.info("custom walker: %s -> %s (%s)", self.state, state, why)
        self.state = state
        if state == "damp":
            self.damp_id = 1                    # a safety stop reports damping (set_fsm(0) sets 0 after this)
            self.ramp = None
            self.vel, self.vel_until = (0.0, 0.0, 0.0), 0.0

    def _fresh(self):
        s = self.reader.take(N=1)
        if s:
            self.msg, self.msg_t = s[0], time.monotonic()
        return self.msg if self.msg is not None and time.monotonic() - self.msg_t <= self.cfg["stale_s"] else None

    def _loop(self):
        nxt = time.perf_counter()
        while True:
            nxt += self.pol.dt
            try:
                self._tick()
            except Exception as e:
                log.exception("custom walker tick")
                with self.lock:
                    self.error = str(e)
                    self._to("damp", f"error: {e}")
            d = nxt - time.perf_counter()
            if d > 0:
                time.sleep(d)
            else:
                nxt = time.perf_counter()

    def _upper(self):
        """The waist+arms command: the policy's home hold, the Arms' command blended in by its weight."""
        p = self.pol
        q, kp, kd, tau = p.q0[UB].copy(), p.kp[UB].copy(), p.kd[UB].copy(), p.tau_ff[UB].copy()
        out = self.g1.arms.custom_out
        if out is not None:
            w, qa, kpa, kda, taua = out          # taua is the Arms' gravity feedforward, already x their weight
            q = w * qa + (1 - w) * q
            kp = w * kpa + (1 - w) * kp
            kd = w * kda + (1 - w) * kd
            tau = taua + (1 - w) * tau
        return q, kp, kd, tau

    def _tick(self):
        with self.lock:
            if not self.active:
                return
            state = self.state
        m = self._fresh()
        if m is None:
            with self.lock:
                self._to("damp", "no fresh lowstate")
            self._publish(None, None, np.zeros(29), self.damp_kd, np.zeros(29))
            return
        q = np.array([m.motor_state[i].q for i in range(29)])
        dq = np.array([m.motor_state[i].dq for i in range(29)])
        quat = np.array(m.imu_state.quaternion, float)
        gyro = np.array(m.imu_state.gyroscope, float)
        g = quat_to_gravity(quat)
        tilt = math.degrees(math.acos(max(-1.0, min(1.0, -g[2]))))
        if state != "damp" and tilt > self.cfg["tilt_max_deg"]:
            with self.lock:
                self._to("damp", f"tilted {tilt:.0f} deg")
            state = "damp"
        if state == "damp":
            self.ub_prev = None
            self._publish(m, None, np.zeros(29), self.damp_kd, np.zeros(29))
            return
        p = self.pol
        if state == "stand":
            t0, q_start, secs = self.ramp
            s = min(1.0, (time.monotonic() - t0) / secs)
            s = s * s * (3 - 2 * s)
            target = q_start + (p.q0 - q_start) * s
            self._publish(m, np.clip(target, self.lo, self.hi), p.kp, p.kd, p.tau_ff)
            return
        uq, ukp, ukd, utau = self._upper()
        arm_vel = np.zeros(17) if self.ub_prev is None else (uq - self.ub_prev) / p.dt
        self.ub_prev = uq
        with self.lock:
            vel = self.vel if time.monotonic() < self.vel_until else (0.0, 0.0, 0.0)
        p.command(*vel, quat_to_yaw(quat))
        target, kp, kd, ff, _ = p.step(q, dq, quat, gyro, arm_vel)
        kp, kd, ff = kp.copy(), kd.copy(), ff.copy()
        target[UB], kp[UB], kd[UB], ff[UB] = uq, ukp, ukd, utau
        self._publish(m, np.clip(target, self.lo, self.hi), kp, kd, ff)

    def _publish(self, m, target, kp, kd, ff):
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
