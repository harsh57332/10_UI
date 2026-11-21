"""MuJoCo check of the walking policy V2 before the robot (sim2sim): Unitree's G1 29-DoF rev 1.0 model
(third_party/unitree_ros), a floor, the motor law of the real motors (tau = kp (q_t - q) - kd dq + tau_ff, at 500 Hz),
rotor inertia (armature) as in training (the leg gains are armature x (2 pi 10 Hz)^2), and the SAME code that runs on
the robot (robot/policy_v2.PolicyV2). Prints how it stands, walks and turns.
  python scripts/policy_v2_sim.py [--phase run|zero] [--seconds 6]
Needs mujoco + onnxruntime + pyyaml (a separate venv; not the server's environment)."""
import argparse
import math
import os
import re
import sys

import mujoco
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from robot.policy_v2 import PolicyV2, quat_to_yaw  # noqa: E402

DESC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "third_party", "unitree_ros", "robots", "g1_description")
W = (2 * math.pi * 10) ** 2
ARMATURE = {"hip_pitch": 40.179 / W, "hip_roll": 99.098 / W, "hip_yaw": 40.179 / W, "knee": 99.098 / W,
            "ankle": 28.501 / W, "waist_yaw": 0.0102, "waist": 0.0072, "shoulder": 0.0036, "elbow": 0.0036, "wrist": 0.0043}


def model():
    xml = open(os.path.join(DESC, "g1_29dof_rev_1_0.xml")).read()
    xml = xml.replace('meshdir="meshes"', f'meshdir="{os.path.join(DESC, "meshes")}"')

    def arm(m):
        name = m.group(1)
        key = next(k for k in ARMATURE if k in name)
        return m.group(0).replace("<joint ", f'<joint armature="{ARMATURE[key]:.6f}" ', 1)
    xml = re.sub(r'<joint name="([a-z_]+_joint)"[^>]*/>', lambda m: arm(m) if m.group(1) != "floating_base_joint" else m.group(0), xml)
    if 'name="floor"' not in xml:               # the model has its own floor
        xml = xml.replace("<worldbody>", '<worldbody><geom name="floor" type="plane" size="20 20 0.1"/>', 1)
    m = mujoco.MjModel.from_xml_string(xml)
    m.opt.timestep = 0.002
    return m


def run(phase, seconds, trials):
    m = model()
    d = mujoco.MjData(m)
    pol = PolicyV2(phase_when_standing=phase)
    sub = int(round(pol.dt / m.opt.timestep))
    lim = np.array([m.jnt_range[m.actuator_trnid[i, 0]] for i in range(29)])
    frc = np.array([m.jnt_actfrcrange[m.actuator_trnid[i, 0]] for i in range(29)])
    for name, (vx, vy, wz) in trials:
        mujoco.mj_resetData(m, d)
        d.qpos[2] = 0.80
        d.qpos[7:36] = pol.q0
        pol.reset()
        target = pol.q0.copy()
        kp, kd, ff = pol.kp, pol.kd, pol.tau_ff
        steps = int(seconds / pol.dt)
        start_xy, start_yaw, fell = None, None, None
        z_min, tilt_max = 9.0, 0.0
        for k in range(steps):
            q, dq = d.qpos[7:36].copy(), d.qvel[6:35].copy()
            quat, gyro = d.qpos[3:7].copy(), d.qvel[3:6].copy()
            settle = k < int(1.0 / pol.dt)           # 1 s standing at zero command first, then the trial command
            yaw = quat_to_yaw(quat)
            pol.command(*((0, 0, 0) if settle else (vx, vy, wz)), yaw)
            target, kp, kd, ff, a = pol.step(q, dq, quat, gyro)
            target = np.clip(target, lim[:, 0], lim[:, 1])
            if not settle and start_xy is None:
                start_xy, start_yaw = d.qpos[0:2].copy(), yaw
            for _ in range(sub):
                q, dq = d.qpos[7:36], d.qvel[6:35]
                d.ctrl[:] = np.clip(kp * (target - q) - kd * dq + ff, frc[:, 0], frc[:, 1])
                mujoco.mj_step(m, d)
            z = d.qpos[2]
            z_min = min(z_min, z)
            g = d.qpos[3:7]
            tilt = math.degrees(math.acos(max(-1, min(1, 1 - 2 * (g[1] ** 2 + g[2] ** 2)))))
            tilt_max = max(tilt_max, tilt)
            if z < 0.5 and fell is None:
                fell = k * pol.dt
                break
        dur = seconds - 1.0
        dx, dy = d.qpos[0:2] - start_xy
        yaw0 = start_yaw
        fwd = dx * math.cos(yaw0) + dy * math.sin(yaw0)
        side = -dx * math.sin(yaw0) + dy * math.cos(yaw0)
        dyaw = math.degrees(math.atan2(math.sin(quat_to_yaw(d.qpos[3:7]) - yaw0), math.cos(quat_to_yaw(d.qpos[3:7]) - yaw0)))
        print(f"{name:22s} cmd ({vx:+.2f},{vy:+.2f},{wz:+.2f}) over {dur:.0f} s: "
              + (f"FELL at {fell:.1f} s" if fell is not None else
                 f"forward {fwd:+.2f} m (asked {vx * dur:+.2f}), side {side:+.2f} m (asked {vy * dur:+.2f}), "
                 f"turned {dyaw:+.1f} deg (asked {math.degrees(wz * dur):+.1f}), pelvis z min {z_min:.3f}, tilt max {tilt_max:.1f} deg"))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", default="run", choices=["run", "zero"])
    ap.add_argument("--seconds", type=float, default=6.0)
    a = ap.parse_args()
    trials = [("stand", (0, 0, 0)), ("walk 0.2", (0.2, 0, 0)), ("walk 0.3", (0.3, 0, 0)), ("back 0.2", (-0.2, 0, 0)),
              ("side left 0.15", (0, 0.15, 0)), ("turn left 0.5", (0, 0, 0.5)), ("turn right 0.5", (0, 0, -0.5))]
    run(a.phase, a.seconds, trials)
