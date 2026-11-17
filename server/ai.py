"""Section 6 (motion planning) and section 7 (AI control).

The LLM (Cosmos) only picks skills from the library; perception, poses and trajectories come from code.
Section 6: text -> skills (arms only) -> every trajectory planned up front -> played on request.
Section 7: text -> skill plan (temperature 0, checked against the hand-state machine) -> py_trees sequence of
fixed behaviours. A stretch of arm skills that picks something up is one node and runs like section 6 (task
scene, margin check, planned up front, played). Walking is in measured legs; the last leg towards a place target
is walked while cuRobo already plans the place. Compulsory safety checks run before every node; step mode waits
for "Next". A failed node triggers a bounded re-plan. Abort stops after the current node; the arms keep holding
whatever they hold. After the task everything measured is forgotten.
"""
import re
import copy
import json
import logging
import math
import os
import threading
import time

import py_trees
from py_trees.common import Status

from common import config

from .motion import ArmPlan, SkillError
from . import heat
from . import skills as skills_mod
from .scene import TaskScene, parse
from .services import ServiceError
from .skills import (SKILLS, TARGET_WORDS, stand_for_pick, SURFACE_WORDS, catalog, check_cool, check_hands, clean_params, execute, new_plan, place_with_prefetch,
                     recover, rest, run_skill)

log = logging.getLogger("ai")

BRIEF = (
    "You control a Unitree G1 humanoid robot in a lab. It has two dummy hands without working fingers, so it "
    "can only pick up boxes by squeezing them between both palms (boxes 4-50 cm wide, up to 2 kg). It walks at a "
    "fixed 0.4 m/s and turns in place. It never changes its control mode. It can also set its head light colour "
    "(headlight) and speak a text (speak). Distances are metres in the robot's "
    "torso frame: x forward, y left, z up."
)
RULES = (
    "Rules: use only the listed skills with their exact parameter names. find an object or surface before any "
    "skill that uses it. After walk, turn or approach every position is forgotten, so find again before using "
    "it (approach finds its own target). To pick something up always use pre_grasp, then grasp, then lift. "
    "While holding the camera view is partly blocked, so find every object you will need near the robot (for "
    "example a box to stack on) before pre_grasp. pre_grasp needs the object within reach: approach it first "
    "unless it is already close (distance under about 0.45 m). place needs a held object and ends with the arms "
    "at ready; to put it on a table you are not at, approach the table first. retreat lets go of an object "
    "that is gripped but not lifted. Each skill only works in the hand state it lists. Keep the plan short. "
    "To stack object A on object B: find A, find B, pre_grasp A, grasp, lift, place on B (A is the object that "
    "moves; B stays where it is). A turn of N degrees to the left/right is one turn step with angle_deg N. To put a "
    "held object on something far away: turn if told to, approach it, then place on it. If the user states how far the target is (from the toe tip), pass that number as distance_cm to approach (the camera only sees 50-60 cm: that part is walked blind). Only one object can be held at a time. Do not add ready at the end: the arms "
    "go back to their normal position by themselves after the task."
)
AFTER = {"pre_grasp": "AT_OBJECT", "grasp": "GRIPPING", "lift": "HOLDING", "place": "EMPTY", "put_back": "EMPTY",
         "set_down": "GRIPPING", "retreat": "EMPTY"}       # hand state after each arm skill


def plan_schema(names):
    # skills and parameters only: a 'why' per step and closing notes were about half of what the LLM wrote (Super
    # writes ~16 tokens/s: 14-17 s per plan; the user wants the motion within 15 s, 2026-11-17)
    return {"type": "object", "required": ["plan"], "additionalProperties": False, "properties": {
        "plan": {"type": "array", "items": {"type": "object", "required": ["skill", "params"], "additionalProperties": False,
                                            "properties": {"skill": {"type": "string", "enum": names},
                                                           "params": {"type": "object"}}}}}}


class Busy:
    """Section 6 and 7 never drive the robot at the same time."""
    lock = threading.Lock()
    owner = None


def llm_plan(ctx, llm, text, kinds, extra=""):
    """The LLM's skill plan, checked against the hand-state machine and find-before-use before anything is
    planned or moved; an invalid plan goes back to the LLM with the exact problem (bounded). Temperature 0:
    the same task gives the same plan."""
    allowed = catalog(kinds)
    fix = ""
    for attempt in range(config.load()["ai"]["plan_fix_tries"] + 1):
        base = (f"Task: {text}\n\nSkills:\n{json.dumps(allowed, indent=1)}\n\nWhat the robot currently knows:\n"
                f"{json.dumps(ctx.summary())}\n\n{RULES}{extra}{fix}\n\n")
        # schema-constrained JSON: never 'thinks' first (without the schema Cosmos3-Super sometimes wrote a <think> block for
        # 24-180 s, 2026-11-17); skills and parameters only, ~7 s
        with ctx.processing("llm"):
            out = llm.call("chat", {"system": BRIEF, "temperature": 0.0, "schema": plan_schema([s["skill"] for s in allowed]),
                                    "user": base + 'Return JSON: {"plan": [{"skill": ..., "params": {...}}]} (no explanations)'})
        plan = (out["json"] or {}).get("plan")
        if not plan:
            raise SkillError(f"the LLM returned no usable plan: {out['text'][:300]}")
        for n in plan:
            if n.get("skill") not in SKILLS or SKILLS[n["skill"]]["kind"] not in kinds:
                raise SkillError(f"the LLM chose a skill that is not allowed here: {n.get('skill')}")
            n["params"] = clean_params(n["skill"], n.get("params"))[0]
        for n in plan:                                 # a distance only counts when the user wrote that number
            d = n["params"].get("distance_cm")
            if d is not None and not any(w in str(n["params"].get("target", "")).lower() for w in TARGET_WORDS):
                n["params"].pop("distance_cm")                # a distance belongs to a case/table-like target, never to the box
            elif d is not None and not re.search(r"(?<![\d.])%s(?:\.0+)?(?![\d.])" % re.escape(str(d).rstrip("0").rstrip(".") if "." in str(d) else str(d)), text):
                n["params"].pop("distance_cm")
        plan = drop_redundant_finds(hoist_finds(plan))
        problem = check_plan(ctx, plan)
        if not problem:
            return plan, (out["json"] or {}).get("notes", "")
        fix = (f"\n\nYour previous plan {[(n['skill'], n['params']) for n in plan]} is invalid: {problem}. "
               f"Return a corrected plan.")
    raise SkillError(f"the LLM could not make a valid plan: {problem}")


def check_plan(ctx, plan):
    """The first problem in a skill plan (None if valid): each step must start from a hand state it allows
    (simulated through the plan), and an object must be found before pre_grasp uses it."""
    hand = ctx.bb["hand"]
    known = set(ctx.bb["objects"])
    lifted = False
    for k, n in enumerate(plan, 1):
        s = SKILLS[n["skill"]]
        missing = clean_params(n["skill"], n["params"])[1]
        if missing:
            return f"step {k} ({n['skill']}) is missing its parameter {', '.join(missing)}"
        if hand not in s["hands"]:
            return (f"step {k} ({n['skill']}) needs the hands {' or '.join(s['hands'])}, but after the steps "
                    f"before it they are {hand}")
        if s["kind"] == "move":                     # walking/turning forgets positions; approach finds its target
            known = {n["params"].get("target")} if n["skill"] == "approach" else set()
            if n["skill"] in ("walk", "turn"):
                lifted = False
        if n["skill"] == "lift":
            lifted = True
        if n["skill"] == "put_back" and not lifted:
            return (f"step {k} (put_back) puts the object back where it was lifted from, which needs a lift in this plan "
                    f"and no walk or turn after it: use place on a found surface instead")
        if n["skill"] == "find":
            known.add(n["params"].get("object"))
        if n["skill"] == "pre_grasp" and n["params"].get("object") not in known:
            return f"step {k} (pre_grasp) uses '{n['params'].get('object')}' before it is found"
        if n["skill"] == "place" and n["params"].get("on") not in known:
            return (f"step {k} (place) puts the object on '{n['params'].get('on')}', which is not found first "
                    f"(find it before pre_grasp: a held object blocks the camera)")
        hand = AFTER.get(n["skill"], hand)
    return None


def hoist_finds(plan):
    """A held box blocks part of the camera view, so any find between a pre_grasp and the next walk/turn
    moves to just before that pre_grasp (the robot has not moved, so the measurement stays valid)."""
    out = list(plan)
    i = 0
    while i < len(out):
        if out[i]["skill"] == "pre_grasp":
            j, moved = i + 1, []
            while j < len(out) and SKILLS[out[j]["skill"]]["kind"] != "move":
                if out[j]["skill"] == "find":
                    moved.append(out.pop(j))
                else:
                    j += 1
            out[i:i] = moved
            i += len(moved)
        i += 1
    return out


def drop_redundant_finds(plan):
    """approach measures its own target (and holding a box blocks the camera): a find right before it is dropped."""
    return [n for k, n in enumerate(plan) if not (n["skill"] == "find" and k + 1 < len(plan) and plan[k + 1]["skill"] == "approach"
                                                  and plan[k + 1]["params"].get("target") == n["params"].get("object"))]


def sync_hands(ctx, name, plan6=None):
    """The robot reports which hands it has (end-effector detection): switch the server's hand profile while
    nothing runs; the next Generate plans with the new hands and a plan made for the old ones is dropped."""
    if not name or name == config.hands() or not Busy.lock.acquire(blocking=False):
        return
    try:
        config.use_hands(name)
        ctx.motion.hands_changed()
        skills_mod._GLOVE.clear()
        heat._GRAVITY = None
        if plan6 is not None and plan6.plan is not None:
            plan6.plan, plan6.state, plan6.error = None, "idle", f"the hands changed to '{name}': generate again"
        log.info("hands: %s (robot model %s)", name, config.load()["robot"]["urdf"])
    finally:
        Busy.lock.release()


def _try(fn):
    try:
        fn()
    except Exception as e:                      # handed to the caller's thread
        return e
    return None


def plan_task(ctx, steps):
    """Plan the skill steps from the robot's live pose. Generate and its offline replay (scripts/replay_task.py)
    run exactly this, so a recorded task re-plans identically."""
    plan = new_plan(ctx)
    for n in steps:
        s = SKILLS[n["skill"]]
        if s["kind"] == "perceive":
            n["result"] = s["fn"](ctx, **n["params"])
        else:
            check_hands(n["skill"], plan.state["hand"])
            n["result"] = s["fn"](ctx, plan, **n["params"])
    return plan


def margin_check(ctx, steps):
    """The plan must not sit at the edge of what is possible: re-plan the arm steps with every found object shifted
    scene.margin_shift_m (4 directions) and the whole scene turned scene.margin_turn_deg about the robot (2). Robust =
    all hold; marginal = it works only near the measured spot (it would fail on a small error)."""
    S = ctx.cfg["scene"]
    d, a = S["margin_shift_m"], math.radians(S["margin_turn_deg"])
    cm, dg = round(100 * d), S["margin_turn_deg"]
    cases = [(f"{cm} cm further", d, 0, 0), (f"{cm} cm nearer", -d, 0, 0), (f"{cm} cm left", 0, d, 0),
             (f"{cm} cm right", 0, -d, 0), (f"turned {dg} deg left", 0, 0, a), (f"turned {dg} deg right", 0, 0, -a)]
    arm = [n for n in steps if SKILLS[n["skill"]]["kind"] != "perceive"]
    saved = copy.deepcopy(ctx.bb["objects"])
    out, t0 = {}, time.time()
    try:
        for name, dx, dy, da in cases:
            objs = copy.deepcopy(saved)
            for v in objs.values():
                x, y = v["center"][0], v["center"][1]
                nx, ny = math.cos(da) * x - math.sin(da) * y + dx, math.sin(da) * x + math.cos(da) * y + dy
                for key in ("front_x", "support_front_x"):
                    if key in v:
                        v[key] += nx - x
                v["center"] = [nx, ny, *v["center"][2:]]
                v["yaw"] = v.get("yaw", 0.0) + da
            ctx.bb["objects"] = objs
            try:
                plan_task(ctx, [dict(n) for n in arm])
                out[name] = "ok"
            except SkillError as e:
                out[name] = str(e)[:200]
            log.info("margin check, scene %s: %s", name, out[name])
    finally:
        ctx.bb["objects"] = saved
    bad = [k for k, v in out.items() if v != "ok"]
    return {"robust": not bad, "failed": bad, "cases": out, "seconds": round(time.time() - t0, 1)}


# ==================================================================== section 6
class MotionPlanning:
    KINDS = ("perceive", "arm")

    def __init__(self, ctx, llm):
        self.ctx, self.llm = ctx, llm
        self.state, self.error, self.llm_steps, self.plan, self.current = "idle", None, [], None, None
        self.heat = None                        # predicted motor peaks of the generated plan (or why it is refused)
        self.margin, self.scene_dir = None, None    # margin check of the plan; the task scene's record folder

    def generate(self, text):
        if not Busy.lock.acquire(blocking=False):
            raise SkillError(f"robot is busy ({Busy.owner})")
        Busy.owner = "motion planning"
        ctx = self.ctx
        try:
            self.state, self.error, self.plan, self.margin = "generating", None, None, None
            if ctx.scene is not None:               # a new task: the last one's scene is over
                ctx.scene.forget(ctx)
            ctx.forget_positions()                 # people move things between tasks: always look again
            if ctx.bb["hand"] != "EMPTY" and not ctx.link.state()["arms"]["engaged"]:
                log.warning("hands recorded as %s but the arms are released (a stopped run): hands are EMPTY", ctx.bb["hand"])
                ctx.bb.update(hand="EMPTY", holding=None, grasp=None, grip=None)   # released arms hold nothing
            ctx.bb.pop("calib", None)
            scene = ctx.scene = TaskScene(ctx, text)
            self.scene_dir = scene.run_dir
            # the straight view (and the likely side views) are taken while the LLM chooses the skills
            box = {}
            th = threading.Thread(target=lambda: box.update(err=_try(lambda: scene.prescan(ctx, text))), daemon=True)
            th.start()
            try:
                self.llm_steps, _ = llm_plan(ctx, self.llm, text, self.KINDS,
                                             " The robot stands still at the table: do not walk or turn.")
            finally:
                th.join()
            if box.get("err"):
                raise box["err"]
            nouns = sorted({parse(n["params"]["object"])[0] for n in self.llm_steps if n["skill"] == "find"
                            and not any(w in n["params"]["object"].lower() for w in TARGET_WORDS)})
            views = scene.scan(ctx, nouns)
            scene.note(llm_steps=[{k: n.get(k) for k in ("skill", "params", "why")} for n in self.llm_steps],
                       nouns=nouns, start_state=ctx.link.state())
            plan = plan_task(ctx, self.llm_steps)
            # the margin check only reports how fragile the plan is (6 full re-plans: 150 s on 2026-11-17): off by default
            self.margin = margin_check(ctx, self.llm_steps) if ctx.cfg["ai"].get("margin_check") else None
            self.plan, self.state = plan, "ready"
            try:
                self.heat = {"peaks_c": heat.check(ctx, plan)}
            except SkillError as e:
                self.heat = {"refused": str(e)}
            scene.note(views_deg=views, plan=plan.summary(), margin=self.margin, heat=self.heat)
            return self.status()
        except (SkillError, TypeError, ServiceError) as e:
            self.state, self.error = "failed", str(e)
            if ctx.scene is not None:               # nothing was played: the hands are where they were
                ctx.scene.note(error=str(e))
                ctx.scene.forget(ctx)
                ctx.scene = None
            raise SkillError(str(e))
        finally:
            Busy.owner = None
            Busy.lock.release()

    def play(self):
        if self.plan is None:
            raise SkillError("generate a motion first")
        if not Busy.lock.acquire(blocking=False):
            raise SkillError(f"robot is busy ({Busy.owner})")
        try:
            check_cool(self.ctx)
        except SkillError as e:
            self.error = str(e)                 # shown in the UI (a 409 alone was invisible)
            Busy.lock.release()
            raise
        Busy.owner = "motion planning"
        plan, self.plan = self.plan, None          # a plan is played once (its start state is now stale)
        self.state, self.error = "playing", None
        ctx = self.ctx

        def worker():
            scene = ctx.scene
            try:
                if scene is not None:              # closed loop: the scene must still be the one planned for
                    self.current = "check the scene is unchanged"
                    scene.check_unchanged(ctx)
                def on_step(i, label):                 # logged: a stop can be placed exactly afterwards
                    self.current = f"{i + 1}/{len(plan.steps)} {label}"
                    log.info("play step %s", self.current)
                # eyes during the play too (the user, 2026-11-17 evening, after a blind try: 'give the eyes'): the camera
                # line-up over the target (partial views refused), the test-lift and placed-object checks
                ctx.bb["blind"] = bool(ctx.cfg["ai"].get("section6_blind", False))
                execute(ctx, plan, on_step=on_step)
                self.state = "done"
            except (SkillError, ServiceError) as e:
                self.state, self.error = "failed", str(e)
                log.warning("play failed at %s: %s", self.current, e)
            finally:
                ctx.bb.pop("blind", None)
                released = False
                try:
                    released = rest(ctx)           # arms back to their normal position after every task
                except Exception as e:
                    self.error = f"{self.error or ''} (arms not released: {e})"
                try:                               # the task is over; once the hands are down, forget its scene
                    down = released or not ctx.link.state()["arms"]["engaged"]
                    if scene is not None:
                        scene.note(result=self.state, error=self.error, hands_down=down)
                        if down and ctx.bb["hand"] == "EMPTY":
                            scene.forget(ctx)
                            ctx.scene = None
                except Exception as e:
                    log.warning("after the task: %s", e)
                self.current, Busy.owner = None, None
                Busy.lock.release()
        threading.Thread(target=worker, daemon=True, name="plan6").start()
        return self.status()

    def scene_info(self):
        sc = self.ctx.scene
        if sc is None:
            return {"dir": self.scene_dir and os.path.basename(self.scene_dir)}
        return {"dir": sc.run_dir and os.path.basename(sc.run_dir), "views_deg": [round(v.yaw_deg) for v in sc.views],
                "objects": {k: {"center_m": [round(x, 3) for x in v["center"]], "size_m": [round(x, 3) for x in v["dims"]],
                                "shape": v.get("shape"), "stack": v.get("stack_n", 1)} for k, v in sc.used.items()}}

    def status(self):
        return {"state": self.state, "error": self.error, "current": self.current, "margin": self.margin,
                "scene": self.scene_info(),
                "llm_plan": [{k: n.get(k) for k in ("skill", "params", "why")} for n in self.llm_steps],
                "steps": self.plan.summary() if self.plan else None, "heat": self.heat,
                "duration_s": round(sum(s["duration"] or 0 for s in self.plan.summary()), 1) if self.plan else None}


# ==================================================================== section 7
class SkillNode(py_trees.behaviour.Behaviour):
    """One behaviour = one segment of the plan (a single skill, or the arm skills of a pick)."""
    def __init__(self, ai, seg):
        super().__init__(name="+".join(ai.nodes[i]["skill"] for i in seg))
        self.ai, self.seg, self.thread = ai, seg, None

    def initialise(self):
        self.thread = threading.Thread(target=self.ai.execute, args=(self.seg,), daemon=True)
        self.thread.start()

    def update(self):
        if self.thread.is_alive():
            return Status.RUNNING
        ok = all(self.ai.nodes[i]["status"] in ("done", "skipped") for i in self.seg)
        return Status.SUCCESS if ok else Status.FAILURE


def segments(ctx, nodes, start):
    """Group the plan from `start`: consecutive perceive/arm skills that begin with empty hands and contain an arm skill
    (a pick: find, pre_grasp, grasp, lift) are one segment, run like section 6; every other skill is its own."""
    out, hand, i = [], ctx.bb["hand"], 0
    for k in range(start):
        hand = AFTER.get(nodes[k]["skill"], hand)
    i = start
    while i < len(nodes):
        j = i
        while j < len(nodes) and SKILLS[nodes[j]["skill"]]["kind"] in ("perceive", "arm"):
            j += 1
        group = list(range(i, j))
        if hand == "EMPTY" and any(SKILLS[nodes[k]["skill"]]["kind"] == "arm" for k in group):
            out.append(group)
        else:
            out.extend([k] for k in group)
        for k in group:
            hand = AFTER.get(nodes[k]["skill"], hand)
        if j < len(nodes) and j == i:
            out.append([j])
            hand = AFTER.get(nodes[j]["skill"], hand)
            j += 1
        i = j
    return out


class AIControl:
    KINDS = ("perceive", "move", "arm", "basic")

    def __init__(self, ctx, llm):
        self.ctx, self.llm, self.cfg = ctx, llm, config.load()
        self.state, self.task, self.mode, self.nodes, self.log, self.error = "idle", None, "step", [], [], None
        self._gate = threading.Event()
        self._abort = threading.Event()

    # ---------------------------------------------------------------- controls
    def start(self, text, mode):
        if not Busy.lock.acquire(blocking=False):
            raise SkillError(f"robot is busy ({Busy.owner})")
        Busy.owner = "AI control"
        self.state, self.task, self.mode, self.nodes, self.log, self.error = "planning", text, mode, [], [], None
        self._abort.clear()
        threading.Thread(target=self._run, daemon=True, name="ai").start()
        return self.status()

    def next(self):
        self._gate.set()

    def abort(self):
        self._abort.set()
        self._gate.set()

    def set_mode(self, mode):
        self.mode = mode
        if mode == "auto":
            self._gate.set()

    def status(self):
        return {"state": self.state, "task": self.task, "mode": self.mode, "error": self.error,
                "nodes": [{k: n.get(k) for k in ("skill", "params", "why", "status", "result", "error")}
                          for n in self.nodes],
                "log": self.log[-30:], "known": self.ctx.summary()}

    def _say(self, msg):
        self.log.append(f"{time.strftime('%H:%M:%S')} {msg}")

    # ---------------------------------------------------------------- run
    def _run(self):
        try:
            self._safety(None)
            check_cool(self.ctx)
            self._forget()                        # people move things between tasks: always find again
            plan, notes = llm_plan(self.ctx, self.llm, self.task, self.KINDS)
            self.nodes = [{**n, "status": "pending"} for n in plan]
            self._say(f"plan: {' -> '.join(n['skill'] for n in plan)}" + (f" ({notes})" if notes else ""))
            replans = 0
            start = 0
            while True:
                result = self._tick_tree(start)
                if result == Status.SUCCESS:
                    self.state = "done"
                    self._say("task done")
                    return
                if self._abort.is_set():
                    self.state = "aborted"
                    self._say("aborted by user; arms keep their current hold")
                    return
                failed = next(i for i, n in enumerate(self.nodes) if n["status"] == "failed")
                if SKILLS[self.nodes[failed]["skill"]]["kind"] == "move":     # turning/walking is not retried or re-planned
                    raise SkillError(f"step {failed + 1} ({self.nodes[failed]['skill']}) failed: {self.nodes[failed]['error']}")
                if replans >= self.cfg["ai"]["max_replans"]:
                    raise SkillError(f"step {failed + 1} failed and the re-plan limit was reached")
                replans += 1
                done = [f"{n['skill']} {n['params']} -> ok" for n in self.nodes[:failed]]
                extra = (f"\n\nAlready done: {done}\nThis step failed: {self.nodes[failed]['skill']} "
                         f"{self.nodes[failed]['params']}: {self.nodes[failed]['error']}\nPlan only the remaining "
                         f"steps to finish the task from the current situation.")
                plan, notes = llm_plan(self.ctx, self.llm, self.task, self.KINDS, extra)
                self.nodes = self.nodes[:failed + 1] + [{**n, "status": "pending"} for n in plan]
                start = failed + 1
                self._say(f"re-plan {replans}: {' -> '.join(n['skill'] for n in plan)}")
        except SkillError as e:
            self.state, self.error = "failed", str(e)
            self._say(f"stopped: {e}")
            self._recover()
        finally:
            try:
                if rest(self.ctx):                 # arms back to their normal position after every task
                    self._say("arms released to their normal position")
            except Exception as e:
                self._say(f"arms not released: {e}")
            self._forget()                         # memory clear for the next task
            Busy.owner = None
            Busy.lock.release()

    def _recover(self):
        """A task that stops while the hands hold or touch something: the recovery for that state (set down / retreat; it
        ignores the heat gate by design, a held box heats the shoulders). Never when the user aborted: that keeps the hold."""
        ctx = self.ctx
        if ctx.bb["hand"] == "EMPTY" or not ctx.link.state()["arms"]["engaged"] or self._abort.is_set():
            return
        try:
            self._say(recover(ctx))
        except Exception as e:                         # no surface to set it on (lost): drop the object, arms down, stand
            self._say(f"recovery failed: {e}; releasing the arms (the object drops)")
            try:
                ctx.agent("release_upperbody", {}, timeout=15)
                ctx.bb.update(hand="EMPTY", holding=None)
            except Exception as e2:
                self._say(f"release failed: {e2}")

    def _forget(self):
        """Forget everything measured, planned or remembered for the task (objects, surfaces, map, scene, prefetched plan)."""
        ctx = self.ctx
        if ctx.scene is not None:
            ctx.scene.forget(ctx)
            ctx.scene = None
        ctx.forget_positions()
        for k in ("prefetch", "place_next", "no_map", "calib", "landed"):
            ctx.bb.pop(k, None)

    def _tick_tree(self, start):
        segs = segments(self.ctx, self.nodes, start)
        root = py_trees.composites.Sequence(name="task", memory=True, children=[SkillNode(self, g) for g in segs])
        tree = py_trees.trees.BehaviourTree(root)
        self.state = "running"
        while True:
            tree.tick()
            if root.status != Status.RUNNING:
                return root.status
            time.sleep(0.1)

    def execute(self, seg):
        """One behaviour: safety checks -> step gate -> the skill (or the whole pick)."""
        first = self.nodes[seg[0]]
        try:
            if self._abort.is_set():
                raise SkillError("aborted")
            for i in seg:
                self.nodes[i]["status"] = "checking"
            self._safety(first)
            if self.mode == "step":
                for i in seg:
                    self.nodes[i]["status"] = "waiting"
                self.state = "waiting"
                self._gate.clear()
                hold = self.ctx.bb["hand"] != "EMPTY"       # holding heats the shoulders: do not wait long for "Next"
                if not self._gate.wait(self.cfg["ai"]["hold_gate_max_s"] if hold else None):
                    self._say(f"no 'Next' within {self.cfg['ai']['hold_gate_max_s']} s while holding: continuing (the arms heat)")
                if self._abort.is_set():
                    raise SkillError("aborted")
            self.state = "running"
            for i in seg:
                self.nodes[i]["status"] = "running"
            if len(seg) > 1:
                self._say(f"steps {seg[0] + 1}-{seg[-1] + 1}: {' -> '.join(self.nodes[i]['skill'] for i in seg)} (planned up front)")
                self._pick(seg)
            else:
                self._one(seg[0])
            for i in seg:
                self.nodes[i]["status"] = "done"
        except (SkillError, ServiceError, TypeError, ValueError) as e:
            bad = next((i for i in seg if self.nodes[i]["status"] != "done"), seg[0])
            for i in seg:
                if self.nodes[i]["status"] != "done":
                    self.nodes[i]["status"] = "failed" if i == bad else "skipped"
            n = self.nodes[bad]
            if n["skill"] in ("place", "put_back") and self.ctx.bb.get("landed") and self.ctx.bb["hand"] == "EMPTY":
                # the object is down on the target; only letting go needed the recovery (hands away, empty): done, the
                # LLM is not asked to place a box that already sits there (2026-11-17)
                n["status"], n["error"] = "done", None
                n["result"] = {"placed": self.ctx.bb.get("landed"), "note": f"let go by the recovery: {e}"}
                self._say(f"step {bad + 1} {n['skill']}: the object is down; the hands came away by the recovery ({e})")
                for i in seg:
                    if self.nodes[i]["status"] == "skipped":
                        self.nodes[i]["status"] = "done"
                return
            self.nodes[bad]["error"] = str(e)
            self._say(f"step {bad + 1} {self.nodes[bad]['skill']} failed: {e}")
            if len(seg) > 1 and self.ctx.scene is not None:
                self._end_scene(error=str(e))

    def _one(self, i):
        n, ctx = self.nodes[i], self.ctx
        n["params"] = clean_params(n["skill"], n["params"])[0]
        self._say(f"step {i + 1}: {n['skill']} {json.dumps(n['params'])}")
        if n["skill"] == "approach":              # a place on this target follows: plan it while walking the last leg
            nxt = self.nodes[i + 1] if i + 1 < len(self.nodes) else None
            ctx.bb["place_next"] = n["params"]["target"] if nxt and nxt["skill"] == "place" and \
                nxt["params"].get("on") == n["params"]["target"] else None
        try:
            if n["skill"] == "place":
                n["result"] = place_with_prefetch(ctx, n["params"]["on"])
            else:
                n["result"] = run_skill(ctx, n["skill"], n["params"])
        finally:
            ctx.bb.pop("place_next", None)

    def _pick(self, seg):
        """Like section 6: scan the scene (SAM 3, side views where needed), plan every step, check it holds when the objects
        are a little off, then play at once (no play check: the scene was scanned seconds ago and nothing else moves). The scene is forgotten afterwards (the robot walks next)."""
        ctx = self.ctx
        steps = [{**self.nodes[i]} for i in seg]
        for n in steps:
            n["params"] = clean_params(n["skill"], n["params"])[0]
        for n in list(steps):                          # a walk forgot every position: a pick not led by its find gets one
            o = n["params"].get("object")
            if n["skill"] == "pre_grasp" and o and not any(m["skill"] == "find" and m["params"].get("object") == o
                                                         for m in steps[:steps.index(n)]):
                steps.insert(steps.index(n), {"skill": "find", "params": {"object": o}, "why": "find it again after the walk"})
        self._forget()
        first = next((n["params"]["object"] for n in steps if n["skill"] == "pre_grasp"), None)
        if first and ctx.bb["hand"] == "EMPTY" and self.cfg["ai"]["stand_for_pick"]:
            with ctx.processing("standing for the pick"):
                if stand_for_pick(ctx, first):
                    time.sleep(1.0)
            self._forget()
            if self._abort.is_set():
                raise SkillError("aborted")
        scene = ctx.scene = TaskScene(ctx, self.task)
        nouns = sorted({parse(n["params"]["object"])[0] for n in steps if n["skill"] == "find"
                        and not any(w in n["params"]["object"].lower() for w in TARGET_WORDS)})
        scene.capture(ctx, 0.0)
        views = scene.scan(ctx, nouns)
        scene.note(llm_steps=[{k: n.get(k) for k in ("skill", "params", "why")} for n in steps], nouns=nouns,
                   start_state=ctx.link.state())
        plan = plan_task(ctx, steps)
        margin = margin_check(ctx, steps) if self.cfg["ai"].get("margin_check") else None
        scene.note(views_deg=views, plan=plan.summary(), margin=margin)
        if margin:
            self._say("plan " + ("robust" if margin["robust"] else f"MARGINAL ({', '.join(margin['failed'])})"))
        if self._abort.is_set():
            raise SkillError("aborted")
        execute(ctx, plan, on_step=lambda i, label: log.info("ai step %d/%d %s", i + 1, len(plan.steps), label))
        self._end_scene()

    def _end_scene(self, error=None):
        ctx = self.ctx
        if ctx.scene is not None:
            ctx.scene.note(result="failed" if error else "done", error=error)
            ctx.scene.forget(ctx)
            ctx.scene = None

    def _safety(self, node):
        """Compulsory before every node."""
        s = self.ctx.link.state()
        if s.get("mode") != "manipulation":
            raise SkillError(f"AI control runs only in Manipulation mode (robot is in {s.get('mode')})")
        b = s.get("battery") or {}
        if b.get("soc", 100) < self.cfg["ai"]["min_battery_pct"]:
            raise SkillError(f"battery low ({b.get('soc')} %)")
        if s["arms"]["loop_hz"] < 40:
            raise SkillError(f"arm control loop unhealthy ({s['arms']['loop_hz']} Hz)")
        if s["busy"]["motion"]:
            raise SkillError(f"robot still busy with {s['busy']['motion']['action']}")
        if node and SKILLS[node["skill"]]["kind"] in ("perceive", "move", "arm") and not self.ctx.camera.online():
            raise SkillError("camera is offline")
        if self.ctx.bb["holding"]:
            run_skill(self.ctx, "check_holding", {})     # raises if the box slipped
