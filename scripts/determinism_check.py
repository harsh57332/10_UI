"""Offline: plan the stack (pre_grasp, grasp, lift, place) several times for the same scene and compare every
trajectory: a deterministic flow gives identical plans. Also prints each arm move's end pose summary.
Run: ~/miniconda3/envs/g1ui_server/bin/python -m scripts.determinism_check [--x 0.37 --y -0.10 --left 0.36 0.16]
     [--runs 3]"""
import argparse
import math
from contextlib import nullcontext

import numpy as np

from server.motion import ArmMotion, ArmPlan
from server.services import Service
from server.skills import SKILLS, Ctx

ap = argparse.ArgumentParser()
ap.add_argument("--x", type=float, default=0.371)
ap.add_argument("--y", type=float, default=-0.103)
ap.add_argument("--left", type=float, nargs=2, default=[0.358, 0.162])
ap.add_argument("--dims", type=float, nargs=3, default=[0.184, 0.140, 0.090])
ap.add_argument("--support", type=float, default=0.081)
ap.add_argument("--runs", type=int, default=3)
a = ap.parse_args()
m = ArmMotion(link=None, planner=Service("planner"), processing=lambda s: nullcontext())
start = {i: 0.0 for i in range(29)}
start.update({15 + k: math.radians(v) for k, v in enumerate([16.6, 13.0, -1.2, 56.2, 11.4, 1.5, -2.0, 16.8, -12.6, 1.5,
                                                              56.2, -5.5, 1.3, -0.1])})
d, w, h = a.dims


def box(x, y):
    return {"center": [x, y, a.support + h / 2], "dims": [d, w, h], "top_z": a.support + h, "support_z": a.support,
            "front_x": x - d / 2, "distance_m": math.hypot(x, y), "yaw": 0.0}


def plan_once():
    ctx = Ctx(None, None, None, None, m, lambda s: nullcontext())
    ctx.bb["objects"] = {"right box": box(a.x, a.y), "left box": box(*a.left)}
    plan = ArmPlan(m, start=dict(start))
    plan.state.update(hand="EMPTY", grasp=None, grip=None)
    for name, params in [("pre_grasp", {"object": "right box"}), ("grasp", {}), ("lift", {}), ("place", {"on": "left box"})]:
        SKILLS[name]["fn"](ctx, plan, **params)
    return [(s["label"], np.array(s["traj"]["positions"])) for s in plan.steps if s["kind"] == "traj"]


plans = [plan_once() for _ in range(a.runs)]
ref = plans[0]
same = True
for k, p in enumerate(plans[1:], 2):
    if [l for l, _ in p] != [l for l, _ in ref]:
        print(f"run {k}: DIFFERENT step list"); same = False; continue
    for (l, t), (_, t0) in zip(p, ref):
        diff = float(np.degrees(np.abs(t - t0)).max()) if t.shape == t0.shape else float("inf")
        if diff > 0.01:
            print(f"run {k}: '{l}' differs by {diff:.2f} deg (shape {t.shape} vs {t0.shape})"); same = False
print("IDENTICAL plans" if same else "NOT deterministic", f"({a.runs} runs, {len(ref)} arm moves)")
for l, t in ref:
    e = np.degrees(t[-1])
    print(f"  {l[:30]:30s} {len(t):4d} pts  L sh p/r/y {e[0]:+5.0f} {e[1]:+5.0f} {e[2]:+5.0f} el {e[3]:+5.0f}  "
          f"R sh p/r/y {e[7]:+5.0f} {e[8]:+5.0f} {e[9]:+5.0f} el {e[10]:+5.0f}  wrists max {np.abs(e[[4,5,6,11,12,13]]).max():3.0f}")
