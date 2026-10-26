"""Does the planner's robot model accept the reference pose? Each arm link's clearance (cm, <0 = overlap) to the
boxes and the case top as measured in docs/reference_grasp (pelvis frame). Usage: check_pose.py [robot_yml]"""
import json
import math
import sys
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from common import config                      # noqa: E402
from common.fk import FK                        # noqa: E402
from server.geometry import Geometry            # noqa: E402

here = Path(__file__).resolve().parent
cfg = config.load()
yml = sys.argv[1] if len(sys.argv) > 1 else config.path(cfg["services"]["planner"]["robot_yml"])
sph = yaml.safe_load(open(yml))["kinematics"]["collision_spheres"]
fk = FK(config.path(cfg["services"]["planner"]["urdf"]))
meta = json.load(open(here / "ref_meta.json"))
q = {n + "_joint": math.radians(v) for n, v in meta["joints_deg"].items()}
T = Geometry().torso_in_pelvis([q["waist_yaw_joint"], q["waist_roll_joint"], q["waist_pitch_joint"]])
# measured 2026-10-23 (depth + image edges): boxes 9 cm tall on the case top at z 0.083, 5.6 cm apart
sup, top = 0.083, 0.173
obstacles = {"target box": ([0.267, -0.143, sup], [0.454, 0.000, top]),
             "left box": ([0.294, 0.056, sup], [0.475, 0.195, top]),
             "case top": ([0.15, -0.40, sup - 0.10], [0.80, 0.40, sup])}
print(f"model: {yml}")
for side in ("left", "right"):
    for link in sorted(l for l in sph if l.startswith(side) and "shoulder" not in l):
        try:
            L = T @ fk.pose(link, q, base="torso_link")
        except Exception:
            continue
        row = []
        for name, (lo, hi) in obstacles.items():
            lo, hi = np.array(lo), np.array(hi)
            d = min(np.linalg.norm(np.maximum(0, np.maximum(lo - c, c - hi))) - s["radius"]
                    for s in sph[link] for c in [L[:3, :3] @ np.array(s["center"]) + L[:3, 3]])
            row.append(f"{name} {100 * d:+5.1f}")
        print(f"  {link:24s} " + "   ".join(row))
