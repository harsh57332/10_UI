"""The colleague's walking policy V2 (g1_v2/: model_12000, task Unitree-G1-Flat-Loco-Man-V2), robot-independent part.

Legs (SDK 0-11) follow the policy at 50 Hz; waist and arms (SDK 12-28) are held at the home pose with
tau = kp (q_home - q) - kd dq + tau_ff. Everything here is pure numpy + onnxruntime, so the MuJoCo check
(scripts/policy_v2_sim.py) and the robot runner (robot/policy_v2_run.py) run the very same code.

Observation (104, training order, g1_v2/info/g1_loco_man_obs_contract.json), all scales 1, no clip:
  0  base_ang_vel 3        pelvis IMU gyroscope (body frame)
  3  projected_gravity 3   gravity in the pelvis frame (standing: 0, 0, -1)
  6  command 3             vx, vy, wz (wz from the heading hold while walking straight)
  9  phase 2               sin, cos of the gait phase (period 0.6 s)
  11 joint_pos 29          q - default_joint_pos
  40 joint_vel 29          dq
  69 actions 12            the last raw action
  81 arm_cmd_vel 17        the upper body's commanded velocity: 0 (held at home)
  98 upper_body_com_pos 3  constant (the home pose's upper-body centre of mass, deploy.yaml)
  101 upper_body_com_vel 3 0 (held)
Action: leg target q = offset + scale * action.
"""
import math
import os

import numpy as np
import onnxruntime as ort
import yaml

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "g1_v2")


def quat_to_gravity(q):
    """Gravity direction in the body frame from the body orientation quaternion (w, x, y, z)."""
    w, x, y, z = q
    return np.array([2 * (-x * z + w * y), -2 * (y * z + w * x), -(w * w - x * x - y * y + z * z)])


def quat_to_yaw(q):
    w, x, y, z = q
    return math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


class PolicyV2:
    def __init__(self, root=ROOT, phase_when_standing="zero", onset_phase=0.27, onset_min_stand=0.2, walk_norm_min=0.1):
        """phase_when_standing: 'run' (the gait clock always runs) or 'zero' (phase input 0, 0 while standing; MuJoCo: 'run'
        steps in place and drifts 13 cm in 5 s, 'zero' stands still). The colleague's deploy fix (2026-11-20): the robot
        walks when |command| >= walk_norm_min; the first walking tick after standing >= onset_min_stand s restarts the gait
        at onset_phase (0.27: weight onto the right leg, then the left foot steps): every start is the same first step.
        onset_phase None = the clock's phase at that moment (yesterday's behaviour)."""
        with open(os.path.join(root, "params", "deploy.yaml")) as f:
            d = yaml.safe_load(f)
        self.d = d
        self.dt = float(d["step_dt"])
        self.kp = np.array(d["stiffness"], float)
        self.kd = np.array(d["damping"], float)
        self.q0 = np.array(d["default_joint_pos"], float)
        act = d["actions"]["JointPositionAction"]
        self.leg_ids = np.array(act["joint_ids"])
        self.scale = np.array(act["scale"], float)
        self.offset = np.array(act["offset"], float)
        ub = d["upper_body"]
        self.ub_ids = np.array(ub["joint_ids"])
        self.tau_ff = np.zeros(29)
        self.tau_ff[self.ub_ids] = ub["tau_ff"]
        obs = d["observations"]
        self.period = float(obs["gait_phase"]["params"]["period"])
        self.com_pos = np.array(obs["upper_body_com_pos"]["params"]["value"], float)
        r = d["commands"]["base_velocity"]["ranges"]
        self.ranges = {"vx": r["lin_vel_x"], "vy": r["lin_vel_y"], "wz": r["ang_vel_z"]}
        self.hh = d["commands"]["base_velocity"]["heading_hold"]
        with open(os.path.join(root, "info", "g1_straight_line_hold.json")) as f:
            import json
            self.hh_extra = json.load(f)
        self.sess = ort.InferenceSession(os.path.join(root, "exported", "policy.onnx"), providers=["CPUExecutionProvider"])
        self.inp = self.sess.get_inputs()[0].name
        self.phase_when_standing = phase_when_standing
        self.onset_phase, self.onset_min_stand, self.walk_norm_min = onset_phase, onset_min_stand, walk_norm_min
        self.reset()

    def reset(self):
        self.last_action = np.zeros(12)
        self.t = 0.0
        self.phase = 0.0                        # gait phase, fraction of the period
        self.stand_time = 1e9                   # entering counts as 'stood long enough' (as the colleague's reset)
        self.yaw_ref, self.integ = None, 0.0
        self.cmd = np.zeros(3)                  # what goes into the observation (after the heading hold)

    # ------------------------------------------------------------ command
    def command(self, vx, vy, wz, yaw):
        """The velocity command for this step, clipped to the trained ranges. Walking (|v| > walk_thresh) with no turn asked
        (|wz| < deadzone): the heading hold turns wz into a PI correction toward the heading held since the walk started."""
        vx = float(np.clip(vx, *self.ranges["vx"]))
        vy = float(np.clip(vy, *self.ranges["vy"]))
        wz = float(np.clip(wz, *self.ranges["wz"]))
        walking = math.hypot(vx, vy) > self.hh_extra.get("walk_thresh", 0.05)
        if self.hh.get("enable") and walking and abs(wz) < self.hh["deadzone"]:
            if self.yaw_ref is None:
                self.yaw_ref, self.integ = yaw, 0.0
            err = wrap(self.yaw_ref - yaw)
            self.integ = float(np.clip(self.integ + err * self.dt, -self.hh["i_max"], self.hh["i_max"]))
            wz = float(np.clip(self.hh["stiffness"] * err + self.hh["ki"] * self.integ, *self.ranges["wz"]))
        else:
            self.yaw_ref, self.integ = None, 0.0     # turning or standing: the held heading starts again at the next walk
        self.cmd = np.array([vx, vy, wz])
        return self.cmd

    # ------------------------------------------------------------ one step
    def _advance_phase(self):
        """The colleague's gait_phase: decide stand or walk first, then move the rhythm on (or restart it at the onset)."""
        walking = float(np.linalg.norm(self.cmd)) >= self.walk_norm_min
        if walking and self.onset_phase is not None and self.stand_time >= self.onset_min_stand:
            self.phase = float(self.onset_phase)
        else:
            self.phase = (self.phase + self.dt / self.period) % 1.0
        self.stand_time = 0.0 if walking else self.stand_time + self.dt
        return walking

    def obs(self, q, dq, quat, gyro, arm_cmd_vel=None):
        walking = self._advance_phase()
        if walking or self.phase_when_standing == "run":
            phase = [math.sin(2 * math.pi * self.phase), math.cos(2 * math.pi * self.phase)]
        else:
            phase = [0.0, 0.0]
        o = np.concatenate([np.asarray(gyro, float), quat_to_gravity(quat), self.cmd, phase,
                            np.asarray(q, float) - self.q0, np.asarray(dq, float), self.last_action,
                            np.zeros(17) if arm_cmd_vel is None else np.asarray(arm_cmd_vel, float),
                            self.com_pos, np.zeros(3)])
        assert o.shape == (104,), o.shape
        return o.astype(np.float32)

    def step(self, q, dq, quat, gyro, arm_cmd_vel=None):
        """One 50 Hz step: returns (target q (29), kp (29), kd (29), tau_ff (29), raw action (12)). arm_cmd_vel: the upper
        body's commanded joint velocities (17, rad/s) while our arms move it (trained with moving arms), else 0."""
        o = self.obs(q, dq, quat, gyro, arm_cmd_vel)
        a = self.sess.run(None, {self.inp: o[None]})[0][0].astype(float)
        if not np.all(np.isfinite(a)):
            raise FloatingPointError("policy output is not finite")
        self.last_action = a
        self.t += self.dt
        target = self.q0.copy()
        target[self.leg_ids] = self.offset + self.scale * a
        return target, self.kp, self.kd, self.tau_ff, a
