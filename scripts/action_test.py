"""Run actions on the real robot exactly as the UI's AI step mode does (run_skill: hand-state checks, motor
heat watch, automatic recovery). After the last action it holds for --hold seconds (still heat-watched),
then, unless --keep, brings the hands back to EMPTY the safe way (retreat, or set down and retreat).

Run: ~/miniconda3/envs/g1ui_server/bin/python -m scripts.action_test --find "right box,left box" \
         --actions "pre_grasp:right box,grasp" --hold 20
Action syntax: name or name:argument (pre_grasp:<object>, place:<target>, turn:left:90, walk:forward:0.5,
approach:<target>); --plan-only prints each action's plan without moving.
"""
import argparse
import time
from contextlib import nullcontext

from common import config
from server.geometry import Geometry
from server.image_client import ImageClient
from server.motion import ArmMotion
from server.robot_link import AgentLink
from server.services import Service
from server.skills import SKILLS, Ctx, _check_heat, check_hands, new_plan, recover, run_skill

ARGS = {"pre_grasp": ["object"], "place": ["on"], "find": ["object"], "approach": ["target"],
        "turn": ["direction", "angle_deg"], "walk": ["direction", "distance_m"], "speak": ["text"]}


def parse(spec):
    name, *vals = [v.strip() for v in spec.split(":")]
    return name, dict(zip(ARGS.get(name, []), vals))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--find", default="")
    ap.add_argument("--actions", required=True)
    ap.add_argument("--hold", type=float, default=20.0)
    ap.add_argument("--keep", action="store_true")
    ap.add_argument("--plan-only", action="store_true")
    a = ap.parse_args()
    cfg = config.load()
    link = AgentLink()
    link.connect(cfg["robot"]["host"])
    cam = ImageClient()
    cam.start(cfg["robot"]["host"])
    t = time.monotonic()
    while cam.latest() is None and time.monotonic() - t < 10:
        time.sleep(0.1)
    ctx = Ctx(link, cam, Service("vision"), Geometry(), ArmMotion(link, Service("planner"), lambda s: nullcontext()),
              lambda s: nullcontext())
    try:
        for name in filter(None, (x.strip() for x in a.find.split(","))):
            print("find:", run_skill(ctx, "find", {"object": name}))
        if a.plan_only:
            plan = new_plan(ctx)
            for spec in a.actions.split(","):
                name, params = parse(spec)
                check_hands(name, plan.state["hand"])
                print(f"{name}: {SKILLS[name]['fn'](ctx, plan, **params)} -> hands {plan.state['hand']}")
            for i, s in enumerate(plan.summary()):
                print(f"  {i:2d} {s['kind']:5s} {s['label']}  {s['duration'] or ''}")
            return
        for spec in a.actions.split(","):
            name, params = parse(spec)
            print(f"run {name} {params} ...", flush=True)
            r = run_skill(ctx, name, params)
            print(f"   -> {r}  | hands {ctx.bb['hand']}", flush=True)
        print(f"holding for {a.hold:.0f} s (heat-watched)", flush=True)
        t_end = time.monotonic() + a.hold
        while time.monotonic() < t_end:
            _check_heat(ctx)
            time.sleep(1.0)
    except Exception as e:
        print("STOPPED:", e)
    finally:
        if not a.keep and not a.plan_only and ctx.bb["hand"] != "EMPTY":
            print("back to EMPTY:", recover(ctx))
        cam.stop()


if __name__ == "__main__":
    main()
