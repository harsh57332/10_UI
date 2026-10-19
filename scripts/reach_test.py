"""Step-by-step test of pick_box on the real robot: find the box, build the full pick plan, print every
step (end joint angles, total turn per arm), and execute only up to a chosen step, then hold there.

Run: ~/miniconda3/envs/g1ui_server/bin/python -m scripts.reach_test                        (plan only)
     ~/miniconda3/envs/g1ui_server/bin/python -m scripts.reach_test --go                   (up to the reach)
     ~/miniconda3/envs/g1ui_server/bin/python -m scripts.reach_test --ready                (back to ready)
Options: --object "right box"  --until "move palms beside the box"
         --grasp-above 0.11   (palm point height above the box support, m; overrides the config)
         --obstacles "left box"   (also found, so the planner keeps the hands off them; "" for none)
         --clear 0.01   (palms this far outside the box faces at the reach; overrides approach_clear_m)
"""
import argparse
import math
import time
from contextlib import nullcontext

import numpy as np

from common import config
from server.geometry import Geometry
from server.image_client import ImageClient
from server.motion import ArmMotion, ArmPlan
from server.robot_link import AgentLink
from server.services import Service
from server.skills import SKILLS, Ctx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--object", default="right box")
    ap.add_argument("--until", default="move palms beside the box")
    ap.add_argument("--go", action="store_true")
    ap.add_argument("--ready", action="store_true")
    ap.add_argument("--grasp-above", type=float)
    ap.add_argument("--obstacles", default="left box")
    ap.add_argument("--clear", type=float)
    a = ap.parse_args()
    cfg = config.load()
    if a.grasp_above is not None:
        cfg["manipulation"]["grasp_min_above_support_m"] = a.grasp_above
    if a.clear is not None:
        cfg["manipulation"]["approach_clear_m"] = [a.clear]
    link = AgentLink()
    link.connect(cfg["robot"]["host"])
    motion = ArmMotion(link, Service("planner"), lambda s: nullcontext())
    if a.ready:
        plan = ArmPlan(motion)
        SKILLS["ready"]["fn"](Ctx(link, None, None, None, motion, lambda s: nullcontext()), plan)
        plan.run(on_step=lambda i, label: print("run:", label))
        return
    cam = ImageClient()
    cam.start(cfg["robot"]["host"])
    try:
        t = time.monotonic()
        while cam.latest() is None and time.monotonic() - t < 10:
            time.sleep(0.1)
        ctx = Ctx(link, cam, Service("vision"), Geometry(), motion, lambda s: nullcontext())
        for o in filter(None, (x.strip() for x in a.obstacles.split(","))):
            print("find (obstacle):", SKILLS["find"]["fn"](ctx, o))
        print("find:", SKILLS["find"]["fn"](ctx, a.object))
        if "support_z" not in ctx.bb["objects"][a.object]:
            print("no support surface under the box")
        b = ctx.bb["objects"][a.object]
        print(f"box: centre {np.round(b['center'], 3).tolist()} dims {np.round(b['dims'], 3).tolist()} "
              f"yaw {math.degrees(b.get('yaw', 0.0)):.1f} deg  support z {b.get('support_z', float('nan')):.3f}")
        plan = ArmPlan(motion)
        print("pick plan:", SKILLS["pick_box"]["fn"](ctx, plan, object=a.object))
    finally:
        cam.stop()
    stop = None
    for i, s in enumerate(plan.steps):
        line = f"{i:2d} {s['kind']:5s} {s['label']}"
        if s["kind"] == "traj":
            p = np.degrees(np.array(s["traj"]["positions"]))
            sw = np.abs(p - p[0]).max(axis=0)
            line += (f"  | turn L {sw[:7].sum():.0f} R {sw[7:].sum():.0f} deg | end L {np.round(p[-1, :7]).astype(int)}"
                     f" R {np.round(p[-1, 7:]).astype(int)}")
        elif s["kind"] == "waist":
            line += f"  | waist {np.round(np.degrees(s['waist']), 1).tolist()}"
        print(line)
        if stop is None and s["label"] == a.until:
            stop = i
    if stop is None:
        raise SystemExit(f"no step '{a.until}' in the plan")
    print(f"\nexecutes steps 0..{stop} and holds there")
    if not a.go:
        return
    plan.steps = plan.steps[:stop + 1]
    plan.run(on_step=lambda i, label: print("run:", label))
    print("holding at:", a.until)


if __name__ == "__main__":
    main()
