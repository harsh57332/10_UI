"""G1 29-DoF joint table. Names follow SDK motor index order; limits come from the URDF."""
import math
import xml.etree.ElementTree as ET
from functools import lru_cache

from . import config

NAMES = [
    "left_hip_pitch", "left_hip_roll", "left_hip_yaw", "left_knee", "left_ankle_pitch", "left_ankle_roll",
    "right_hip_pitch", "right_hip_roll", "right_hip_yaw", "right_knee", "right_ankle_pitch", "right_ankle_roll",
    "waist_yaw", "waist_roll", "waist_pitch",
    "left_shoulder_pitch", "left_shoulder_roll", "left_shoulder_yaw", "left_elbow",
    "left_wrist_roll", "left_wrist_pitch", "left_wrist_yaw",
    "right_shoulder_pitch", "right_shoulder_roll", "right_shoulder_yaw", "right_elbow",
    "right_wrist_roll", "right_wrist_pitch", "right_wrist_yaw",
]

GROUPS = {
    "left_leg": list(range(0, 6)),
    "right_leg": list(range(6, 12)),
    "waist": [12, 13, 14],
    "left_arm": list(range(15, 22)),
    "right_arm": list(range(22, 29)),
}

UPPER_BODY = list(range(12, 29))  # 17 joints: waist 3 + left arm 7 + right arm 7
ARMS = list(range(15, 29))        # 14 joints planned by cuRobo


@lru_cache(maxsize=1)
def limits_deg() -> list:
    """[(min_deg, max_deg)] per motor index, read from the URDF in the config."""
    root = ET.parse(config.path(config.load()["robot"]["urdf"])).getroot()
    by_name = {}
    for j in root.findall("joint"):
        lim = j.find("limit")
        if j.get("type") == "revolute" and lim is not None:
            by_name[j.get("name")] = (math.degrees(float(lim.get("lower"))), math.degrees(float(lim.get("upper"))))
    return [tuple(round(v, 1) for v in by_name[n + "_joint"]) for n in NAMES]


def clamp_deg(index: int, value: float) -> float:
    lo, hi = limits_deg()[index]
    return min(max(value, lo), hi)
