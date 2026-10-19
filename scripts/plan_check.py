"""Offline check of the arm skills against the planner (no robot): builds pick + place plans for
virtual boxes from the ready pose and prints each step.
Run: ~/miniconda3/envs/g1ui_server/bin/python -m scripts.plan_check"""
import math
from contextlib import nullcontext

from common import config
from server.motion import ArmMotion, ArmPlan
from server.services import Service
from server.skills import Ctx, SKILLS

cfg = config.load()
motion = ArmMotion(link=None, planner=Service("planner"), processing=lambda s: nullcontext())
ctx = Ctx(link=None, camera=None, vision=None, geometry=None, motion=motion, processing=lambda s: nullcontext())
ready = {i: 0.0 for i in range(29)}
ready.update({15 + k: math.radians(v) for k, v in enumerate(cfg["manipulation"]["ready_deg"])})

for name, (cx, w, d, h) in {"medium 26x20x18": (0.36, 0.26, 0.20, 0.18), "narrow 6x12x15": (0.34, 0.06, 0.12, 0.15),
                            "wide 45x25x20": (0.38, 0.45, 0.25, 0.20), "low 30x20x8": (0.32, 0.30, 0.20, 0.08), "medium@0.32": (0.32, 0.26, 0.20, 0.18), "tall 20x20x30": (0.32, 0.20, 0.20, 0.30)}.items():
    sup = -0.12
    ctx.bb = {"objects": {"box": {"center": [cx, 0.0, sup + h / 2], "dims": [d, w, h], "top_z": sup + h,
                                  "support_z": sup, "front_x": cx - d / 2, "distance_m": cx}},
              "surfaces": {"table": {"top_z": sup, "front_x": 0.22, "back_x": 1.0, "y_range": [-0.8, 0.8]}},
              "holding": None}
    plan = ArmPlan(motion, start=dict(ready))
    plan.state = {"holding": None}
    try:
        r = SKILLS["pick_box"]["fn"](ctx, plan, object="box")
        SKILLS["place"]["fn"](ctx, plan, on="table")
        total = sum(s["duration"] or 0 for s in plan.summary())
        print(f"{name:16s} OK   pitch {r['pitch_deg']:>2} deg, {len(plan.steps)} steps, {total:.1f} s of motion")
    except Exception as e:
        print(f"{name:16s} FAIL {e}")
