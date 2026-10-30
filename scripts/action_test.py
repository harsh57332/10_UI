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
from server.skills import SKILLS, Ctx, _check_heat, check_cool, check_hands, execute, new_plan, recover, rest, run_skill

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
    ap.add_argument("--scene", action="store_true", help="find with the task scene (scan, SAM 3 objects, map) as Generate does")
    ap.add_argument("--all-at-once", action="store_true",
                    help="plan every action first (arms resting), then run them as one continuous sequence")
    a = ap.parse_args()
    cfg = config.load()
    link = AgentLink()
    link.connect(cfg["robot"]["host"])
    cam = ImageClient()
    cam.start(cfg["robot"]["host"])
    t = time.monotonic()
    while cam.latest() is None and time.monotonic() - t < 10:
        time.sleep(0.1)
    hands = (link.state().get("end_effector") or {}).get("type")
    if hands:
        config.use_hands(hands)                 # the hands the robot reports (as the server does)
    ctx = Ctx(link, cam, Service("vision"), Geometry(), ArmMotion(link, Service("planner"), lambda s: nullcontext()),
              lambda s: nullcontext())
    try:
        if not a.plan_only:
            check_cool(ctx)
        if a.scene:
            from server.scene import TaskScene, parse as label_noun
            ctx.scene = TaskScene(ctx, "practice: " + a.actions)
            nouns = sorted({label_noun(n.strip())[0] for n in a.find.split(",") if n.strip()})
            print("scene views at waist", ctx.scene.scan(ctx, nouns), "deg (hands:", config.hands() + ")")
        for name in filter(None, (x.strip() for x in a.find.split(","))):
            print("find:", run_skill(ctx, "find", {"object": name}))
        if a.plan_only or a.all_at_once:
            plan = new_plan(ctx)
            for spec in a.actions.split(","):
                name, params = parse(spec)
                check_hands(name, plan.state["hand"])
                print(f"{name}: {SKILLS[name]['fn'](ctx, plan, **params)} -> hands {plan.state['hand']}")
            for i, s in enumerate(plan.summary()):
                print(f"  {i:2d} {s['kind']:5s} {s['label']}  {s['duration'] or ''}")
            if a.plan_only:
                return
            print("executing the whole sequence ...", flush=True)
            execute(ctx, plan, on_step=lambda i, label: print(f"  > {label}", flush=True))
            print(f"   -> done | hands {ctx.bb['hand']}", flush=True)
        else:
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
        if not a.keep and not a.plan_only:
            print("arms released to their normal position:", rest(ctx))
        cam.stop()


if __name__ == "__main__":
    main()
