"""Offline: plan pick(right box) + place(on left box) for an upright-box scene from the robot's standing
arm pose. Run: ~/miniconda3/envs/g1ui_server/bin/python -m scripts.scene_plan_check [gap_m]"""
import math
import sys
from contextlib import nullcontext

from server.motion import ArmMotion, ArmPlan
from server.services import Service
from server.skills import SKILLS, Ctx

gap = float(sys.argv[1]) if len(sys.argv) > 1 else 0.25
motion = ArmMotion(link=None, planner=Service("planner"), processing=lambda s: nullcontext())
ctx = Ctx(None, None, None, None, motion, lambda s: nullcontext())
start = {i: 0.0 for i in range(29)}
start.update({15 + k: math.radians(v) for k, v in enumerate([16.6, 13.0, -1.2, 56.2, 11.4, 1.5, -2.0, 16.8, -12.6, 1.5, 56.2, -5.5, 1.3, -0.1])})
sup, h, w, d = 0.10, 0.20, 0.09, 0.14
rc = (0.39, -0.12)
box = lambda x, y: {"center": [x, y, sup + h / 2], "dims": [d, w, h], "top_z": sup + h, "support_z": sup,
                    "front_x": x - d / 2, "distance_m": math.hypot(x, y)}
ctx.bb["objects"] = {"right box": box(*rc), "left box": box(0.38, rc[1] + w + gap)}
plan = ArmPlan(motion, start=dict(start))
try:
    print("pick ", SKILLS["pick_box"]["fn"](ctx, plan, object="right box"))
    print("place", SKILLS["place"]["fn"](ctx, plan, on="left box"))
    print("ready", SKILLS["ready"]["fn"](ctx, plan))
except Exception as e:
    print("FAIL", e)
print("steps:", [(s["label"], s["duration"]) for s in plan.summary()])
