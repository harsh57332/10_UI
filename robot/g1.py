"""Thin wrapper over unitree_sdk2py for the G1: state reads, loco, audio/LED and the arm_sdk publisher.

State topics are read on demand (no 500 Hz Python callbacks, which starved the old agent).
Only the Arms thread ever writes rt/arm_sdk (two publishers made the old robot vibrate).
"""
import json
import logging
import math
import threading
import time

import numpy as np
from cyclonedds.core import Policy, Qos
from cyclonedds.sub import DataReader
from cyclonedds.topic import Topic
from unitree_sdk2py.core.channel import ChannelFactory, ChannelFactoryInitialize, ChannelPublisher
from unitree_sdk2py.g1.audio.g1_audio_api import ROBOT_API_ID_AUDIO_TTS
from unitree_sdk2py.g1.audio.g1_audio_client import AudioClient
from unitree_sdk2py.g1.loco.g1_loco_client import LocoClient
from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import BmsState_, IMUState_, LowCmd_, LowState_
from unitree_sdk2py.utils.crc import CRC

from common import config, joints
from common.dynamics import Gravity

log = logging.getLogger("g1")


class Latest:
    """Newest sample of a topic, taken only when asked (KeepLast(1), non-blocking take)."""

    def __init__(self, participant, name, typ, max_age):
        topic = Topic(participant, name, typ)
        self.reader = DataReader(participant, topic, qos=Qos(Policy.History.KeepLast(1)))
        self.lock = threading.Lock()
        self.msg, self.t, self.max_age = None, 0.0, max_age

    def get(self):
        with self.lock:
            if self.msg is None or time.monotonic() - self.t > self.max_age:
                samples = self.reader.take(N=1)
                if samples:
                    self.msg, self.t = samples[0], time.monotonic()
            return self.msg

    def age(self):
        return time.monotonic() - self.t if self.t else None


class Headlight:
    """Head RGB LED. While any 'processing' source is active it blinks the busy colour (a long bright phase, then off),
    else it shows the set colour. The colour is re-sent every refresh_s: sent once, the red failure light was not seen
    (something on the robot sets the light back, the user 2026-11-19)."""

    def __init__(self, audio, cfg):
        self.audio, self.cfg = audio, cfg
        self.color = cfg["default"]
        self.sources = set()
        self.lock = threading.Lock()
        self._sent, self._sent_t, self._t0 = None, 0.0, 0.0
        threading.Thread(target=self._loop, daemon=True, name="headlight").start()

    def set(self, hex_color):
        hex_to_rgb(hex_color)                   # a bad colour is refused here, not in the loop
        with self.lock:
            self.color = hex_color.upper()
        self._show()

    def processing(self, source, on):
        with self.lock:
            was = bool(self.sources)
            (self.sources.add if on else self.sources.discard)(source)
            if bool(self.sources) != was:
                self._t0 = time.monotonic()      # a blink starts with its bright phase
        self._show()

    def _want(self):
        if self.sources:
            on, off = self.cfg["blink_on_s"], self.cfg["blink_off_s"]
            lit = (time.monotonic() - self._t0) % (on + off) < on
            return hex_to_rgb(self.cfg["busy_color"]) if lit else (0, 0, 0)
        return hex_to_rgb(self.color)

    def _show(self):
        with self.lock:
            rgb, now = self._want(), time.monotonic()
            if rgb == self._sent and now - self._sent_t < self.cfg["refresh_s"]:
                return
            try:
                code = self.audio.LedControl(*rgb)
            except Exception as e:
                code = str(e)
            if code != 0 and rgb != self._sent:
                log.warning("head light %s: LedControl returned %s", rgb, code)
            self._sent, self._sent_t = rgb, now

    def _loop(self):
        while True:
            time.sleep(0.05)
            self._show()


class Arms:
    """The single rt/arm_sdk publisher for the 17 upper-body joints (motor index 12..28).
    The waist joints in waist_hold_joints (roll, pitch) stay where they were at engage (as in Unitree's
    teleop): tilting them under arm_sdk fought the balance policy (the body swayed, waist pitch heated)."""

    def __init__(self, g1, cfg):
        self.g1, self.cfg = g1, cfg
        self.dt = 1.0 / cfg["rate_hz"]
        self.kp = np.zeros(29)
        self.kd = np.zeros(29)
        for g in cfg["gains"].values():
            for j in g["joints"]:
                self.kp[j], self.kd[j] = g["kp"], g["kd"]
        ff = cfg["gravity_ff"]
        self._ff = ff
        self.gravity = Gravity(config.path(g1.cfg["robot"]["urdf"]), ff, config.tcp_frames())
        self.ff_enabled = bool(ff["enabled"])
        self.squeeze_target = 0.0 # N per hand; the feedforward ramps toward these
        self.payload_target = 0.0 # kg between the palms
        self.tau_ff = {}          # last feedforward sent {motor index: Nm}, for /state
        self._ff_error = None
        self.pub = ChannelPublisher("rt/arm_sdk", LowCmd_)
        self.pub.Init()
        self.cmd = unitree_hg_msg_dds__LowCmd_()
        self.crc = CRC()
        self.lock = threading.Lock()
        self.weight = 0.0         # arm_sdk blend: 0 = policy owns the arms, 1 = we do
        self.weight_target = 0.0
        self.weight_rate = 1.0    # weight change per second while ramping
        self.q = None             # commanded positions (rad) for UPPER_BODY
        self.segments = []        # [(t_start, t_end, q_from, q_to, smooth)], time.monotonic()
        self.loop_hz = 0.0        # measured publish/loop rate, for /state
        self.max_gap_ms = 0.0     # worst tick gap in the last second
        self.fault = None         # over-temperature release reason, until it cools and is engaged again
        self.warning = None       # motors above warn_motor_temp_c while engaged (the server backs off)
        self._push_since = None   # when an arm joint started pushing hard against its command
        self.recover_until = 0.0  # a recovery move runs until then (no over-temperature hold meanwhile)
        self.custom_out = None    # custom walker: (weight, q17, kp17, kd17, tau17 already x weight) for its loop; None = home
        threading.Thread(target=self._loop, daemon=True, name="arm_sdk").start()

    @property
    def engaged(self):
        return self.weight_target > 0 or self.weight > 0

    def moving(self):
        with self.lock:
            return bool(self.segments)

    def q_cmd_deg(self):
        """Commanded upper-body angles (deg, motor index 12..28) while engaged, else None."""
        with self.lock:
            return [round(math.degrees(v), 3) for v in self.q] if self.engaged and self.q is not None else None

    def engage(self, recovery=False):
        """Take over the arms at their current measured pose (no jump). After a hold in place, moves are
        refused until every upper-body motor has cooled below resume_motor_temp_c, except recovery moves:
        those unload the arms (set down, open, lower), and refusing them left hot arms stuck high."""
        if self.fault:
            hot = self._hot(self.cfg["resume_motor_temp_c"])
            if hot and not recovery:
                raise RuntimeError(f"{self.fault}; still too hot to engage ({self._describe(hot)}, "
                                   f"wait until below {self.cfg['resume_motor_temp_c']} C)")
            log.warning("cleared for a %s move: %s", "recovery" if recovery else "new", self.fault)
            self.fault = None
        with self.lock:
            if self.weight_target == 1.0:
                return
            self.q = self.g1.upper_q()
            self.segments = []
            self.weight_target = 1.0
            self.weight_rate = 1.0 / self.cfg["engage_s"]
        self._wait_weight(1.0)

    def release(self, wait=True):
        self.fault = None           # released on purpose: nothing is held, the next engage starts clean (a stale hold blocked 3 runs, 2026-11-03)
        with self.lock:
            self._let_go()
            self.segments = []
            self.weight_target = 0.0
            self.weight_rate = 1.0 / self.cfg["release_s"]
        if wait:
            self._wait_weight(0.0)

    def drop(self):
        """Emergency: hand the arms back to the policy immediately."""
        with self.lock:
            self._let_go()
            self.segments, self.weight_target, self.weight = [], 0.0, 0.0

    def _let_go(self):
        self.squeeze_target = self.payload_target = 0.0
        self.gravity.squeeze_n = self.gravity.payload_kg = 0.0

    def move(self, keyframes, smooth=True, recovery=False):
        """keyframes: [(duration_s, {motor_index: rad})]. Unnamed joints hold their position, and so does the
        waist joints in waist_hold_joints. Returns the total duration in seconds. A recovery move runs even
        after a hold in place and is not interrupted by the over-temperature hold (it unloads the arms)."""
        self.engage(recovery)
        held = self.cfg["waist_hold_joints"]
        with self.lock:
            t = t_begin = time.monotonic()
            q_from = self.q.copy()
            for duration, targets in keyframes:
                q_to = q_from.copy()
                for idx, rad in targets.items():
                    if idx not in held:
                        q_to[joints.UPPER_BODY.index(idx)] = rad
                self.segments.append((t, t + duration, q_from, q_to, smooth))
                t, q_from = t + duration, q_to
            if recovery:
                self.recover_until = max(self.recover_until, t + 1.0)
        return t - t_begin

    def _hot(self, limit):
        """{motor index: temperature C} of the upper-body motors hotter than limit."""
        m = self.g1.low.get()
        if m is None:
            return {}
        temps = {j: max(m.motor_state[j].temperature) for j in joints.UPPER_BODY}
        return {j: t for j, t in temps.items() if t > limit}

    @staticmethod
    def _describe(hot):
        return ", ".join(f"{joints.NAMES[j]} {t} C" for j, t in hot.items())

    def _overheat_check(self):
        """Once a second while engaged: report motors above warn_motor_temp_c (the server opens the hands and
        backs away gracefully), and release the arms as a last resort above max_motor_temp_c."""
        if not self.engaged:
            self.warning = None
            return
        warm = self._hot(self.cfg["warn_motor_temp_c"])
        self.warning = f"arm motors hot: {self._describe(warm)}" if warm else None
        hot = {j: t for j, t in warm.items() if t > self.cfg["max_motor_temp_c"]}
        self._hot_seen = getattr(self, "_hot_seen", 0) + 1 if hot else 0      # a one-sample reading is not a trip (the status read 57 C when this read 97)
        if hot:
            m = self.g1.low.get()
            log.warning("motor temperature above max (%d in a row): %s raw %s", self._hot_seen, self._describe(hot),
                        {joints.NAMES[j]: list(m.motor_state[j].temperature) for j in hot} if m else None)
        if self._hot_seen >= 3 and not self.fault and time.monotonic() > self.recover_until:
            self._hold_in_place(f"over-temperature: {self._describe(hot)}")

    def _blocked_check(self, now):
        """An arm joint pushing hard against its command (kp x error above max_push_nm) for max_push_s, moving
        or not, is blocked by something: stop pushing and hold in place. (The left upper arm pressed into the
        torso pushed 16 Nm and heated to 90 C in 3 s; normal moves push ~3 Nm, a grip ~4 Nm.) The held waist
        is left out: its error is the torso's weight."""
        m = self.g1.low.get()
        if m is None or self.q is None or self.fault or self.weight_target < 1.0 or self.weight < 0.999:
            self._push_since = None
            return
        k0 = len(joints.UPPER_BODY) - len(joints.ARMS)
        err = np.array([m.motor_state[j].q for j in joints.ARMS]) - self.q[k0:]
        push = np.abs(err) * self.kp[joints.ARMS]
        k = int(push.argmax())
        if push[k] < self.cfg["max_push_nm"]:
            self._push_since = None
        elif self._push_since is None:
            self._push_since = now
        elif now - self._push_since >= self.cfg["max_push_s"]:
            self._hold_in_place(f"blocked: {joints.NAMES[joints.ARMS[k]]} pushing {push[k]:.0f} Nm "
                                f"({math.degrees(err[k]):+.0f} deg from its command)")

    def _hold_in_place(self, why):
        """Stop pushing without moving: command the arms where they are (never drop them near objects).
        Further moves are refused until engage() finds everything cool again."""
        self.fault = f"arms holding in place: {why}"
        log.error(self.fault)
        with self.lock:
            self.segments = []
            self.q = self.g1.upper_q()

    def set_feedforward(self, enabled=None, payload_kg=None, squeeze_n=None):
        """Feedforward on/off; the mass held between the palms (kg, both hands together) and the squeeze force
        (N per hand). Both ramp to the new value; returns how long that takes (ramp_s)."""
        ff = self.cfg["gravity_ff"]
        with self.lock:
            if enabled is not None:
                self.ff_enabled = bool(enabled)
            if payload_kg is not None:
                self.payload_target = float(payload_kg)
            if squeeze_n is not None:
                self.squeeze_target = float(squeeze_n)
            ramp = max(abs(self.squeeze_target - self.gravity.squeeze_n) / ff["squeeze_rate_nps"],
                       abs(self.payload_target - self.gravity.payload_kg) / ff["payload_rate_kgps"])
        return {"enabled": self.ff_enabled, "payload_kg": self.payload_target, "squeeze_n": self.squeeze_target,
                "ramp_s": round(ramp, 2)}

    def _feedforward(self):
        """Gravity torques at the commanded pose, scaled by the arm_sdk weight (no extra push while the
        policy still owns part of the blend). A failure only logs: the publish loop must never die."""
        if not self.ff_enabled:
            self.tau_ff = {}
            return self.tau_ff
        try:
            ff, g = self.cfg["gravity_ff"], self.gravity
            g.squeeze_n = _toward(g.squeeze_n, self.squeeze_target, ff["squeeze_rate_nps"] * self.dt)
            g.payload_kg = _toward(g.payload_kg, self.payload_target, ff["payload_rate_kgps"] * self.dt)
            k = ff["scale"] * self.weight
            self.tau_ff = {j: k * t for j, t in g.torques(self.q).items()}
            self._ff_error = None
        except Exception as e:
            if self._ff_error != str(e):
                log.exception("gravity feedforward failed")
                self._ff_error = str(e)
            self.tau_ff = {}
        return self.tau_ff

    def _wait_weight(self, target):
        while abs(self.weight - target) > 1e-6:
            if target > 0 and self.fault:           # released for over-temperature during the engage ramp
                raise RuntimeError(self.fault)
            time.sleep(self.dt)

    def _loop(self):
        # Monotonic clock only: an NTP jump of the wall clock must never speed up or freeze the arms.
        # A late tick resyncs instead of bursting to catch up; ramps scale with real elapsed time.
        nxt = last = stat_t = block_t = time.monotonic()
        ticks, max_gap = 0, 0.0
        while True:
            nxt += self.dt
            delay = nxt - time.monotonic()
            if delay > 0:
                time.sleep(delay)
            else:
                nxt = time.monotonic()
            now = time.monotonic()
            gap, last = now - last, now
            ticks, max_gap = ticks + 1, max(max_gap, gap)
            if now - stat_t >= 1.0:
                self.loop_hz, self.max_gap_ms = ticks / (now - stat_t), max_gap * 1000
                ticks, max_gap, stat_t = 0, 0.0, now
                self._overheat_check()
            if now - block_t >= self.cfg["block_check_s"]:
                block_t = now
                self._blocked_check(now)
            with self.lock:
                if self.weight == 0.0 and self.weight_target == 0.0:
                    self.custom_out = None
                    continue
                step = self.weight_rate * min(gap, 2 * self.dt)
                self.weight = float(np.clip(self.weight + np.clip(self.weight_target - self.weight, -step, step), 0, 1))
                while self.segments and now >= self.segments[0][1]:
                    self.q = self.segments.pop(0)[3].copy()
                if self.segments:
                    t0, t1, qa, qb, smooth = self.segments[0]
                    u = min(max((now - t0) / max(t1 - t0, 1e-6), 0.0), 1.0)
                    s = u * u * (3 - 2 * u) if smooth else u
                    self.q = qa + (qb - qa) * s
                tau = self._feedforward()
                if self.g1.custom_active():        # the custom walker's loop drives the motors: hand it the command
                    ub = joints.UPPER_BODY
                    self.custom_out = (self.weight, self.q.copy(), self.kp[ub].copy(), self.kd[ub].copy(),
                                       np.array([tau.get(j, 0.0) for j in ub]))
                    continue
                self.custom_out = None
                for k, j in enumerate(joints.UPPER_BODY):
                    m = self.cmd.motor_cmd[j]
                    m.q, m.dq, m.kp, m.kd = float(self.q[k]), 0.0, float(self.kp[j]), float(self.kd[j])
                    m.tau = tau.get(j, 0.0)
                self.cmd.motor_cmd[29].q = self.weight
                self.cmd.crc = self.crc.Crc(self.cmd)
                self.pub.Write(self.cmd)


def rebuild_gravity(arms):
    """The hands changed (other hand profile): the feedforward's arm model (masses, palm points) with them."""
    arms.gravity = Gravity(config.path(config.load()["robot"]["urdf"]), arms._ff, config.tcp_frames())


class G1:
    def __init__(self, cfg):
        self.cfg = cfg
        ChannelFactoryInitialize(0, cfg["robot"]["dds_interface"])
        participant = ChannelFactory()._ChannelFactory__participant  # reuse the SDK's participant
        age = cfg["agent"]["state_max_age_s"]
        self.low = Latest(participant, "rt/lowstate", LowState_, age)
        self.torso_imu = Latest(participant, "rt/secondary_imu", IMUState_, age)
        self.bms = Latest(participant, "rt/lf/bmsstate", BmsState_, 1.0)

        timeout = cfg["agent"]["rpc_timeout_s"]
        self.loco = LocoClient()
        self.loco.SetTimeout(timeout)
        self.loco.Init()
        self.audio = AudioClient()
        self.audio.SetTimeout(timeout)
        self.audio.Init()

        self.headlight = Headlight(self.audio, cfg["headlight"])
        self.walker, self.custom, self._participant = "unitree", None, participant
        self.arms = Arms(self, cfg["arms"])
        self._restore_unitree()

    # ---------- walking controller: Unitree's (default) or the custom policy (chosen at connect) ----------
    def custom_active(self):
        return self.walker == "custom" and self.custom is not None and self.custom.active

    def _restore_unitree(self):
        """At start: an agent that stopped while the custom walker drove the robot left Unitree's controller released;
        give it back so the robot is in its normal state again."""
        try:
            from unitree_sdk2py.comm.motion_switcher.motion_switcher_client import MotionSwitcherClient
            msc = MotionSwitcherClient()
            msc.SetTimeout(5.0)
            msc.Init()
            code, res = msc.CheckMode()
            if code == 0 and not (res or {}).get("name"):
                log.warning("Unitree's controller was released (a custom-walker session ended): selecting 'ai' again")
                msc.SelectMode("ai")
        except Exception as e:
            log.warning("could not check Unitree's controller: %s", e)

    def set_walker(self, name):
        """'unitree' or 'custom'. Only on the crane in damping (the controller changes under the robot)."""
        if name == self.walker:
            return {"walker": name, "changed": False}
        if self.arms.engaged:
            raise RuntimeError("release the arms first")
        if name == "custom":
            cur = self.fsm_id()
            if cur not in (0, 1):
                raise RuntimeError("put the robot in Damping (hanging on the crane) before switching to the custom walker")
            if self.custom is None:
                from .custom_walker import CustomWalker
                self.custom = CustomWalker(self, self._participant, self.cfg["custom_walker"])
            self.custom.activate()
            self.walker = "custom"
        elif name == "unitree":
            self.custom.deactivate()
            self.walker = "unitree"
        else:
            raise ValueError(f"unknown walker '{name}' (unitree | custom)")
        return {"walker": self.walker, "changed": True}

    # ---------- state ----------
    def lowstate(self, wait_s=1.0):
        t_end = time.monotonic() + wait_s
        while True:
            m = self.low.get()
            if m is not None or time.monotonic() > t_end:
                return m
            time.sleep(0.01)

    def upper_q(self):
        m = self.lowstate()
        if m is None:
            raise RuntimeError("no rt/lowstate from the robot")
        return np.array([m.motor_state[j].q for j in joints.UPPER_BODY], dtype=float)

    def yaw(self):
        m = self.lowstate()
        if m is None:
            raise RuntimeError("no rt/lowstate from the robot")
        return float(m.imu_state.rpy[2])

    def fsm_id(self):
        if self.custom_active():
            return self.custom.fsm_id()
        code, data = self.loco.GetFsmId()
        if code != 0:
            raise RuntimeError(f"GetFsmId failed (code {code})")
        if isinstance(data, str):
            data = json.loads(data)
        return int(data["data"] if isinstance(data, dict) else data)

    def joint_states(self):
        m = self.lowstate()
        lim = joints.limits_deg()
        out = []
        for i, name in enumerate(joints.NAMES):
            s = m.motor_state[i] if m else None
            out.append({
                "index": i, "name": name,
                "deg": round(math.degrees(s.q), 2) if s else None,
                "min": lim[i][0], "max": lim[i][1],
                "temp": max(s.temperature) if s else None,
                "tau": round(s.tau_est, 2) if s else None,
            })
        return out

    def imu(self):
        def rpy(msg):
            return [round(math.degrees(v), 2) for v in msg.rpy] if msg else None
        m = self.lowstate()
        return {"pelvis_rpy": rpy(m.imu_state if m else None), "torso_rpy": rpy(self.torso_imu.get())}

    def battery(self):
        b = self.bms.get()
        if b is None:
            return None
        return {"soc": b.soc, "soh": b.soh, "current_ma": b.current, "cycle": b.cycle,
                "voltage_v": round(b.bmsvoltage[0] / 1000.0, 2), "temps_c": list(b.temperature[:4])}

    # ---------- motion ----------
    def set_fsm(self, fsm_id):
        if self.custom_active():
            return self.custom.set_fsm(fsm_id)
        code = self.loco.SetFsmId(fsm_id)
        if code != 0:
            raise RuntimeError(f"SetFsmId({fsm_id}) failed (code {code})")

    def velocity(self, vx, vy, wz, duration):
        if self.custom_active():
            return self.custom.velocity(vx, vy, wz, duration)
        code = self.loco.SetVelocity(vx, vy, wz, duration)
        if code != 0:
            raise RuntimeError(f"SetVelocity failed (code {code})")

    def stop_move(self):
        if self.custom_active():
            return self.custom.stop_move()
        self.loco.SetVelocity(0.0, 0.0, 0.0, 1.0)

    # ---------- audio ----------
    def speak(self, text):
        # Not AudioClient.TtsMaker: its "tts_index += tts_index" keeps the index at 0 forever, so the
        # voice service can treat every request as a repeat. Send a unique, increasing index instead.
        self._tts_index = max(getattr(self, "_tts_index", 0) + 1, int(time.time() * 1000) % 2_000_000_000)
        p = {"index": self._tts_index, "text": text, "speaker_id": self.cfg["audio"]["tts_speaker_id"]}
        code, _ = self.audio._Call(ROBOT_API_ID_AUDIO_TTS, json.dumps(p))
        if code != 0:
            raise RuntimeError(f"TTS failed (code {code})")

    def set_volume(self, v):
        code = self.audio.SetVolume(int(v))
        if code != 0:
            raise RuntimeError(f"SetVolume failed (code {code})")

    def get_volume(self):
        code, data = self.audio.GetVolume()
        if code != 0:
            raise RuntimeError(f"GetVolume failed (code {code})")
        return data.get("volume", data) if isinstance(data, dict) else data


def _toward(v, target, step):
    return min(v + step, target) if v < target else max(v - step, target)


def hex_to_rgb(h):
    h = h.strip().lstrip("#")
    if len(h) != 6 or any(c not in "0123456789abcdefABCDEF" for c in h):
        raise ValueError(f"bad hex colour '{h}', expected like #FF8800")
    return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))
