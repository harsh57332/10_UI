"""Offline replay of a recorded section 6 task (no robot): the same views, LLM steps and start pose go through the
same code as Generate (server.ai.plan_task, margin_check) with the real vision and planner services.
  python -m scripts.replay_task logs/runs/<time>_task [--twice] [--hands inspire]
Prints the objects found and the plan, and whether they are IDENTICAL to what the robot planned live."""
import argparse
import json
import math
import os
from contextlib import nullcontext

from server.ai import margin_check, plan_task
from server.geometry import Geometry
from server.motion import ArmMotion, SkillError
from server.scene import TaskScene
from server.services import Service
from server.skills import Ctx


class RecordedLink:
    """The robot's state at the start of planning, as recorded; nothing can be sent."""
    def __init__(self, state):
        self.s = state

    def state(self):
        return self.s

    def send(self, action, *a, **k):
        raise SkillError(f"replay: '{action}' would move the robot")


def replay(run_dir, hands=None):
    rec = json.load(open(os.path.join(run_dir, "scene.json")))
    from common import config               # the hands the robot had then (recorded before 2026-10-29: dummy)
    config.use_hands(hands or (rec["start_state"].get("end_effector") or {}).get("type") or "dummy")
    link = RecordedLink(rec["start_state"])
    proc = lambda s: nullcontext()
    motion = ArmMotion(link=link, planner=Service("planner"), processing=proc)
    ctx = Ctx(link, None, Service("vision"), Geometry(), motion, proc)
    ctx.scene = TaskScene.load(ctx, run_dir)
    steps = [dict(n) for n in rec["llm_steps"]]
    out = {"found": {}, "plan": None, "error": None, "margin": None}
    try:
        plan = plan_task(ctx, steps)
        out["plan"] = plan.summary()
        out["margin"] = margin_check(ctx, steps)
    except SkillError as e:
        out["error"] = str(e)
    out["found"] = {k: {"center": v["center"], "dims": v["dims"], "top_z": v["top_z"]} for k, v in ctx.bb["objects"].items()}
    return rec, out


def same(a, b, tol=0.002):
    return all(abs(x - y) <= tol for x, y in zip(a, b))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--twice", action="store_true", help="replay twice and compare the two replays too")
    ap.add_argument("--hands", help="plan with these hands instead of the recorded ones (what-if; not identical then)")
    a = ap.parse_args()
    rec, out = replay(a.run_dir, a.hands)
    print(f"task: {rec['text']}\nviews at waist {[round(math.degrees(v['q']['12']), 1) for v in rec['views']]} deg")
    ok = True
    for k, v in out["found"].items():
        live = rec.get("found", {}).get(k)
        match = live is not None and same(v["center"], live["center"]) and same(v["dims"], live["dims"])
        ok &= match
        note = "= live" if match else "LIVE WAS " + str(live and [round(x, 3) for x in live["center"]])
        print(f"  {k:28s} centre {[round(x, 3) for x in v['center']]} size {[round(100 * x, 1) for x in v['dims']]} cm  {note}")
    if out["error"]:
        print("REPLAY FAILED:", out["error"], "| live:", rec.get("error") or "planned")
        ok &= bool(rec.get("error"))
    else:
        live = [(s["label"], s["duration"]) for s in rec.get("plan") or []]
        mine = [(s["label"], s["duration"]) for s in out["plan"]]
        print(f"  plan: {len(mine)} steps; live plan: {len(live)} steps; {'same steps and durations' if live == mine else 'DIFFERENT'}")
        ok &= live == mine
        m = out["margin"]
        print(f"  margin: {'ROBUST' if m['robust'] else 'MARGINAL'} ({m['seconds']} s) " + "; ".join(f"{k}: {v[:90]}" for k, v in m["cases"].items() if v != "ok"))
    if a.twice:
        _, out2 = replay(a.run_dir, a.hands)
        same2 = out2["plan"] == out["plan"] and out2["error"] == out["error"]
        print("  second replay:", "identical" if same2 else "DIFFERENT")
        ok &= same2
    print("IDENTICAL to the live planning" if ok else "NOT identical")


if __name__ == "__main__":
    main()
