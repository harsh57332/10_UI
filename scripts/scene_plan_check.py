"""Offline: plan the stack (pre_grasp, grasp, lift the right box, place it on the left box) from the robot's
standing arm pose, with fitted cuboids only (the live run adds the depth obstacle map).
  one scene:  python -m scripts.scene_plan_check [--x 0.385 --y -0.088 --gap 0.14]
  robustness: python -m scripts.scene_plan_check --grid      (the box pair over a grid of positions and gaps)
Box size/support default to the flat 18 x 13.9 x 9.1 cm boxes measured on 2026-10-23."""
import argparse
import math
from contextlib import nullcontext

from server.motion import ArmMotion, ArmPlan
from server.services import Service
from server.skills import SKILLS, Ctx, check_hands

ap = argparse.ArgumentParser()
ap.add_argument("--x", type=float, default=0.385)
ap.add_argument("--y", type=float, default=-0.088)
ap.add_argument("--gap", type=float, default=0.14, help="free space between the boxes")
ap.add_argument("--dims", type=float, nargs=3, default=[0.18, 0.139, 0.091], help="depth width height")
ap.add_argument("--support", type=float, default=0.0885)
ap.add_argument("--grid", action="store_true")
ap.add_argument("--gaps", type=float, nargs="+", default=[0.14, 0.18])
ap.add_argument("--xs", type=float, nargs="+", default=[0.31, 0.34, 0.37, 0.40])
ap.add_argument("--ms", type=float, nargs="+", default=[-0.06, -0.03, 0.0, 0.03, 0.06])
a = ap.parse_args()

motion = ArmMotion(link=None, planner=Service("planner"), processing=lambda s: nullcontext())
ctx = Ctx(None, None, None, None, motion, lambda s: nullcontext())
start = {i: 0.0 for i in range(29)}
start.update({15 + k: math.radians(v) for k, v in enumerate([16.6, 13.0, -1.2, 56.2, 11.4, 1.5, -2.0, 16.8, -12.6, 1.5, 56.2, -5.5, 1.3, -0.1])})
d, w, h = a.dims


def box(x, y):
    return {"center": [x, y, a.support + h / 2], "dims": [d, w, h], "top_z": a.support + h, "support_z": a.support,
            "front_x": x - d / 2, "distance_m": math.hypot(x, y), "yaw": 0.0}


def stack(x, y, gap, verbose):
    ctx.bb["objects"] = {"right box": box(x, y), "left box": box(x, y + w + gap)}
    plan = ArmPlan(motion, start=dict(start))
    plan.state.update(hand="EMPTY", grasp=None, grip=None)
    try:
        for name, params in [("pre_grasp", {"object": "right box"}), ("grasp", {}), ("lift", {}), ("place", {"on": "left box"})]:
            check_hands(name, plan.state["hand"])
            r = SKILLS[name]["fn"](ctx, plan, **params)
            if verbose:
                print(f"{name:9s}", r, "-> hands", plan.state["hand"])
    except Exception as e:
        return str(e)
    if verbose:
        print("steps:", [(s["label"], s["duration"]) for s in plan.summary()])
    return None


if not a.grid:
    err = stack(a.x, a.y, a.gap, True)
    print("FAIL", err) if err else print("OK")
else:
    # the pair of boxes over a grid of positions: x = their distance ahead, m = the midpoint between them
    xs, ms = a.xs, a.ms
    fails = []
    for gap in a.gaps:
        print(f"gap {gap * 100:.0f} cm: pair midpoint m (rows) x ahead (columns)")
        print("         " + "".join(f"{x:>7.2f}" for x in xs))
        for m in ms:
            row = []
            for x in xs:
                err = stack(x, m - (w + gap) / 2, gap, False)
                row.append("ok" if err is None else ("std" if err.startswith("STANDARDS") else
                                                     ("fail*" if "STANDARDS M3" in err else "FAIL")))
                if err:
                    fails.append((gap, x, m, err))
            print(f"  m {m:+.2f}" + "".join(f"{c:>7}" for c in row))
    print("std = refused by a standard; fail* = could not plan, outside the tested position zone (M3)")
    for f in fails:
        print("FAIL" if not f[3].startswith("STANDARDS") else "std ", f[:3], f[3][:160])
