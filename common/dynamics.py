"""Arm joint torques the G1 needs at a pose (shared: the robot sends them as feedforward, the server predicts
motor heat from them)."""
import xml.etree.ElementTree as ET

import numpy as np

from . import joints
from .fk import FK, _axis_angle


class Gravity:
    """Arm joint feedforward torques at a pose. Gravity: the feedforward Unitree's teleop sends with its soft
    arm gains (kp 80 then holds the arms without sagging): URDF masses plus the gloves, plus a held payload
    hanging on the two palms. Squeeze: each palm pushes toward the other with squeeze_n (tau = J^T F), so
    the grip force is set directly instead of by a position error. Gravity is along the pelvis -z."""

    def __init__(self, urdf_path, cfg, tcp):
        self.fk = FK(urdf_path)
        self.mass = {}
        for link in ET.parse(urdf_path).getroot().findall("link"):
            i = link.find("inertial")
            if i is not None:
                o = i.find("origin")
                com = [float(v) for v in (o.get("xyz", "0 0 0") if o is not None else "0 0 0").split()]
                self.mass[link.get("name")] = (float(i.find("mass").get("value")), np.array(com))
        kids = {}
        for child, j in self.fk.joints.items():
            kids.setdefault(j["parent"], []).append(child)
        self.order, stack = [], ["torso_link"]          # links carried by the torso, parents first
        while stack:
            link = stack.pop()
            self.order.append(link)
            stack += kids.get(link, [])
        by_name = {j["name"]: child for child, j in self.fk.joints.items()}
        self.link_of = {i: by_name[joints.NAMES[i] + "_joint"] for i in joints.ARMS}   # motor -> its child link
        self.hands = [t["parent"] for t in tcp.values()]
        self.palms = [(t["parent"], np.array(t["xyz"], float)) for t in tcp.values()]   # glove contact points
        self.arm_of = [joints.GROUPS["left_arm" if "left" in link else "right_arm"] for link, _ in self.palms]
        self.glove_kg = cfg["glove_kg"]
        self.payload_kg = 0.0
        self.squeeze_n = 0.0

    def waist_pitch(self, q):
        """Gravity torque (Nm) the waist pitch motor holds at upper-body angles q (rad, motor 12..28), with the payload
        on the palms: everything above it. For the heat forecast only (never sent to the robot: the waist pitch is
        the balance joint). Matched the measured 7-9 Nm (ready pose) and 18.6 Nm (Inspire hands over a box), 2026-10-30."""
        qn = {joints.NAMES[j] + "_joint": float(v) for j, v in zip(joints.UPPER_BODY, q)}
        g = self.fk.pose("torso_link", qn, base="pelvis")[:3, :3].T @ np.array([0.0, 0.0, -9.81])   # torso frame
        T = {"torso_link": np.eye(4)}
        for link in self.order[1:]:
            j = self.fk.joints[link]
            M = T[j["parent"]] @ j["T"]
            if j["type"] in ("revolute", "continuous"):
                R = np.eye(4)
                R[:3, :3] = _axis_angle(j["axis"], qn.get(j["name"], 0.0))
                M = M @ R
            T[link] = M
        s = np.zeros(3)
        for link in self.order:
            mk, c = self.mass.get(link, (0.0, np.zeros(3)))
            if link in self.hands:
                mk += self.glove_kg
            s = s + mk * (T[link][:3, :3] @ c + T[link][:3, 3])
        for link, p in self.palms:
            s = s + self.payload_kg / 2 * (T[link][:3, :3] @ p + T[link][:3, 3])
        axis = np.asarray(self.fk.joints["torso_link"]["axis"], float)   # the waist pitch joint, at the torso origin
        return -float(axis @ np.cross(s, g))

    def torques(self, q):
        """{motor index: Nm} for the 14 arm joints at upper-body angles q (rad, motor 12..28)."""
        qn = {joints.NAMES[j] + "_joint": float(v) for j, v in zip(joints.UPPER_BODY, q)}
        g = self.fk.pose("torso_link", qn, base="pelvis")[:3, :3].T @ np.array([0.0, 0.0, -9.81])   # torso frame
        T = {"torso_link": np.eye(4)}
        for link in self.order[1:]:
            j = self.fk.joints[link]
            M = T[j["parent"]] @ j["T"]
            if j["type"] in ("revolute", "continuous"):
                R = np.eye(4)
                R[:3, :3] = _axis_angle(j["axis"], qn.get(j["name"], 0.0))
                M = M @ R
            T[link] = M
        m, s = {}, {}                                   # mass and first moment (sum m*c) of each subtree
        for link in self.order:
            mk, c = self.mass.get(link, (0.0, np.zeros(3)))
            if link in self.hands:                      # the glove sits around the hand's own centre of mass
                mk += self.glove_kg
            m[link], s[link] = mk, mk * (T[link][:3, :3] @ c + T[link][:3, 3])
        P = [T[link][:3, :3] @ p + T[link][:3, 3] for link, p in self.palms]
        for (link, _), p in zip(self.palms, P):         # a held payload hangs on the two palms
            m[link] += self.payload_kg / 2
            s[link] = s[link] + self.payload_kg / 2 * p
        for link in reversed(self.order[1:]):
            parent = self.fk.joints[link]["parent"]
            m[parent] += m[link]
            s[parent] = s[parent] + s[link]
        tau, axis = {}, {}
        for i, link in self.link_of.items():
            o, z = T[link][:3, 3], T[link][:3, :3] @ np.asarray(self.fk.joints[link]["axis"], float)
            axis[i] = (o, z)
            tau[i] = -float(z @ np.cross(s[link] - m[link] * o, g))   # the motor holds against gravity's torque
        if self.squeeze_n:
            d = P[1] - P[0]
            d = d / max(float(np.linalg.norm(d)), 1e-6)
            for p, f, arm in zip(P, (d, -d), self.arm_of):          # each palm pushes toward the other
                for i in arm:
                    o, z = axis[i]
                    tau[i] += float(z @ np.cross(p - o, self.squeeze_n * f))
        return tau
