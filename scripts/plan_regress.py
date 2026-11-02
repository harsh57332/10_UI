"""Offline regression over recorded task scenes (no robot): re-plan each one with the current code/config/planner and
print what matters for safety and arm motion, so a planner change can be compared with the state before it.
  python -m scripts.plan_regress --save base                 plan the default scenes, save logs/regress/base.json
  python -m scripts.plan_regress --against base [--margin]    plan again and mark what got worse (-) or better (+)
  python -m scripts.plan_regress logs/runs/<time>_task ...    explicit scenes (default: the scenes in config regress.scenes,
                                                              else every logs/runs/*_task that has a scene.json)
Per scene: planned or the refusal, steps, heat peak (model, every motor starting at 45 C), the right/left shoulder
roll peak, the waist turn at the place, the lift height, the largest joint travel of any single move and of the
carry/place moves ("arm motion": lower = simpler), and with --margin the margin check (ROBUST / MARGINAL).
Run the planner service first (scripts/svc.sh start planner)."""
import argparse
import json
import math
import os
import time
from contextlib import nullcontext
from pathlib import Path

from common import config, joints
from server import heat
from server.ai import margin_check, plan_task
from server.geometry import Geometry
from server.motion import ArmMotion, SkillError
from server.scene import TaskScene
from server.services import Service
from server.skills import Ctx

from scripts.replay_task import RecordedLink

IDX = {n + "_joint": i for i, n in enumerate(joints.NAMES)}


def metrics(ctx, plan):
    m = {"steps": len(plan.steps), "roll_r": -99.0, "roll_l": -99.0, "roll_r_place": -99.0, "wrist_pitch_min": 99.0, "waist_place": None, "lift_cm": None,
         "max_move_deg": 0.0, "carry_travel_deg": 0.0, "moves": {}}
    q = dict(plan.q0)
    for s in plan.steps:
        if s["kind"] == "waist":
            q.update({12 + k: v for k, v in enumerate(s["waist"])})
        if s["kind"] == "traj" and s["label"] == "move above the target":
            m["waist_place"] = round(math.degrees(q[12]), 1)
        if s["kind"] != "traj":
            continue
        t = s["traj"]
        names, P = t["joint_names"], t["positions"]
        if s["label"] == "lift":                # how far the hands rose (torso frame)
            z0 = max(float(p[2]) for p in ctx.motion.palms(q))
        q.update({IDX[n]: v for n, v in zip(names, P[-1])})
        if s["label"] == "lift":
            m["lift_cm"] = round(100 * (max(float(p[2]) for p in ctx.motion.palms(q)) - z0), 1)
        if s.get("dry"):
            continue
        for k, n in enumerate(names):
            if n == "right_shoulder_roll_joint":
                m["roll_r"] = max(m["roll_r"], max(math.degrees(p[k]) for p in P))
                if s["label"] in ("move above the target", "lower", "line up over the target", "raise it over the objects") \
                        or s["label"].startswith("place"):
                    m["roll_r_place"] = max(m["roll_r_place"], max(math.degrees(p[k]) for p in P))
            if n in ("left_wrist_pitch_joint", "right_wrist_pitch_joint") and s["label"] in (
                    "move above the target", "lower", "line up over the target"):     # wrist roll/yaw align at pitch 0
                m["wrist_pitch_min"] = round(min(m["wrist_pitch_min"], min(abs(math.degrees(p[k])) for p in P)), 1)
            if n == "left_shoulder_roll_joint":
                m["roll_l"] = max(m["roll_l"], max(math.degrees(p[k]) for p in P))
        travel = max(sum(abs(P[i + 1][k] - P[i][k]) for i in range(len(P) - 1)) for k in range(len(names)))
        deg = round(math.degrees(travel), 1)
        m["moves"][s["label"]] = max(m["moves"].get(s["label"], 0.0), deg)
        m["max_move_deg"] = max(m["max_move_deg"], deg)
        if s["label"] in ("lift", "pull in to the carry pose", "move above the target", "lower", "raise it over the objects",
                          "line up over the target") or s["label"].startswith("place"):
            m["carry_travel_deg"] += deg
    try:
        segs = heat.segments(ctx.cfg, plan)
        T0 = {j: 45.0 for j in heat.WATCHED}
        peak = heat.simulate(ctx.cfg, segs, T0)
        m["heat_c"] = {joints.NAMES[k]: round(v) for k, v in peak.items() if v >= 60}
        m["heat_max_c"] = round(max(peak.values()))
    except Exception as e:                      # the heat model needs the robot URDF only
        m["heat_c"], m["heat_max_c"] = str(e), None
    m["carry_travel_deg"] = round(m["carry_travel_deg"], 1)
    return m


def run_scene(run_dir, with_margin):
    rec = json.load(open(os.path.join(run_dir, "scene.json")))
    config.use_hands((rec["start_state"].get("end_effector") or {}).get("type") or "dummy")
    link = RecordedLink(rec["start_state"])
    proc = lambda s: nullcontext()
    ctx = Ctx(link, None, Service("vision"), Geometry(), ArmMotion(link=link, planner=Service("planner"), processing=proc),
              proc)
    ctx.scene = TaskScene.load(ctx, run_dir)
    steps = [dict(n) for n in rec["llm_steps"]]
    out = {"hands": config.load().get("end_effector", {}).get("type")}
    t0 = time.time()
    try:
        plan = plan_task(ctx, steps)
        out.update(ok=True, **metrics(ctx, plan))
        if with_margin:
            mg = margin_check(ctx, steps)
            out["margin"] = "ROBUST" if mg["robust"] else "MARGINAL: " + "; ".join(k for k, v in mg["cases"].items() if v != "ok")
    except SkillError as e:
        out.update(ok=False, error=str(e)[:160])
    out["plan_s"] = round(time.time() - t0, 1)
    return out


def scenes(args):
    if args.scenes:
        return args.scenes
    listed = (config.load().get("regress") or {}).get("scenes")
    if listed:
        return listed
    return sorted(str(p.parent) for p in Path(config.path("logs/runs")).glob("*_task/scene.json"))


def show(name, r, base):
    print(f"\n{name}")
    if not r["ok"]:
        print(f"  REFUSED: {r['error']}" + (f"   (before: {'planned' if base and base.get('ok') else 'refused'})" if base else ""))
        return
    def cmp(key, worse_if_up=True, tol=0.5):
        if not base or not base.get("ok") or base.get(key) is None or r.get(key) is None:
            return ""
        d = r[key] - base[key]
        if abs(d) <= tol:
            return ""
        return "  (+ better)" if (d < 0) == worse_if_up else "  (- WORSE)"
    print(f"  steps {r['steps']}  plan {r['plan_s']} s  waist at place {r['waist_place']}  lift {r['lift_cm']} cm")
    print(f"  heat max {r['heat_max_c']} C {r['heat_c']}{cmp('heat_max_c')}")
    print(f"  right shoulder roll at the place {r['roll_r_place']:.1f}{cmp('roll_r_place', tol=0.3)}   (the real arm meets the torso at ~+9)")
    print(f"  smallest wrist pitch at the place {r['wrist_pitch_min']} deg   (near 0 the wrist roll and yaw axes align: corrections flip)")
    print(f"  arm motion: largest single-joint travel {r['max_move_deg']} deg{cmp('max_move_deg', tol=2)}, "
          f"carry+place moves {r['carry_travel_deg']} deg{cmp('carry_travel_deg', tol=3)}")
    if "margin" in r:
        was = f"   (before: {base.get('margin')})" if base and base.get("margin") != r["margin"] else ""
        print(f"  margin: {r['margin']}{was}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("scenes", nargs="*")
    ap.add_argument("--save")
    ap.add_argument("--against")
    ap.add_argument("--margin", action="store_true", help="also run the margin check (slow)")
    a = ap.parse_args()
    base = {}
    if a.against:
        base = json.load(open(config.path(f"logs/regress/{a.against}.json")))
    res = {}
    for d in scenes(a):
        name = Path(d).name
        res[name] = run_scene(d, a.margin)
        show(name, res[name], base.get(name))
    worse = [n for n, r in res.items() if base.get(n, {}).get("ok") and not r["ok"]]
    print("\nSCENES THAT PLANNED BEFORE AND DO NOT NOW:", worse or "none")
    if a.save:
        os.makedirs(config.path("logs/regress"), exist_ok=True)
        json.dump(res, open(config.path(f"logs/regress/{a.save}.json"), "w"), indent=1)
        print(f"saved logs/regress/{a.save}.json")


if __name__ == "__main__":
    main()
