"""Minimal URDF forward kinematics (numpy) for quick frame checks; the planner uses cuRobo."""
import xml.etree.ElementTree as ET

import numpy as np


def _rpy(r, p, y):
    cr, sr, cp, sp, cy, sy = np.cos(r), np.sin(r), np.cos(p), np.sin(p), np.cos(y), np.sin(y)
    return np.array([[cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
                     [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
                     [-sp, cp * sr, cp * cr]])


def _axis_angle(axis, a):
    axis = np.asarray(axis, float) / np.linalg.norm(axis)
    K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    return np.eye(3) + np.sin(a) * K + (1 - np.cos(a)) * K @ K


class FK:
    def __init__(self, urdf_path):
        root = ET.parse(urdf_path).getroot()
        self.joints = {}
        for j in root.findall("joint"):
            o = j.find("origin")
            xyz = [float(v) for v in (o.get("xyz", "0 0 0") if o is not None else "0 0 0").split()]
            rpy = [float(v) for v in (o.get("rpy", "0 0 0") if o is not None else "0 0 0").split()]
            ax = j.find("axis")
            T = np.eye(4)
            T[:3, :3], T[:3, 3] = _rpy(*rpy), xyz
            self.joints[j.find("child").get("link")] = {
                "name": j.get("name"), "parent": j.find("parent").get("link"), "T": T, "type": j.get("type"),
                "axis": [float(v) for v in ax.get("xyz").split()] if ax is not None else [0, 0, 1]}

    def pose(self, link, q=None, base="torso_link"):
        """4x4 pose of `link` in `base`. q: {joint_name: rad}."""
        q = q or {}
        T = np.eye(4)
        while link != base:
            j = self.joints[link]
            M = j["T"].copy()
            if j["type"] in ("revolute", "continuous"):
                R = np.eye(4)
                R[:3, :3] = _axis_angle(j["axis"], q.get(j["name"], 0.0))
                M = M @ R
            T = M @ T
            link = j["parent"]
        return T
