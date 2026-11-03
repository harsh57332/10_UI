"""Section 6 (motion planning) and section 7 (AI control).

The LLM only picks skills from the library; perception, poses and trajectories come from code.
Section 6: text -> skills (arms only) -> every trajectory planned up front -> played on request.
Section 7: text -> skill plan -> py_trees sequence. Before each node an isolated LLM call reviews it
against the live blackboard; compulsory safety checks run before every node; step mode waits for
"Next" before each node. A failed node triggers a bounded re-plan. Abort stops after the current
node; the arms keep holding whatever they hold.
"""
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
from .skills import (SKILLS, SURFACE_WORDS, catalog, check_cool, check_hands, clean_params, execute, new_plan, rest,
                     run_skill)

log = logging.getLogger("ai")

BRIEF = (
    "You control a Unitree G1 humanoid robot in a lab. It has two dummy hands without working fingers, so it "
    "can only pick up boxes by squeezing them between both palms (boxes 4-50 cm wide, up to 2 kg). It walks at a "
    "fixed 0.4 m/s and turns in place. It never changes its control mode. Distances are metres in the robot's "
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
    "moves; B stays where it is). Only one object can be held at a time. Do not add ready at the end: the arms "
    "go back to their normal position by themselves after the task."
)
AFTER = {"pre_grasp": "AT_OBJECT", "grasp": "GRIPPING", "lift": "HOLDING", "place": "EMPTY",
         "set_down": "GRIPPING", "retreat": "EMPTY"}       # hand state after each arm skill


def plan_schema(names):
    return {"type": "object", "required": ["plan"], "properties": {
        "plan": {"type": "array", "items": {"type": "object", "required": ["skill", "params", "why"], "properties": {
            "skill": {"type": "string", "enum": names}, "params": {"type": "object"}, "why": {"type": "string"}}}},
        "notes": {"type": "string"}}}


REVIEW_SCHEMA = {"type": "object", "required": ["decision", "params", "reason"], "properties": {
    "decision": {"type": "string", "enum": ["run", "skip", "abort"]},
    "params": {"type": "object"}, "reason": {"type": "string"}}}


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
        user = (f"Task: {text}\n\nSkills:\n{json.dumps(allowed, indent=1)}\n\nWhat the robot currently knows:\n"
                f"{json.dumps(ctx.summary())}\n\n{RULES}{extra}{fix}\n\nReturn JSON: "
                '{"plan": [{"skill": ..., "params": {...}, "why": ...}], "notes": ...}')
        with ctx.processing("llm"):
            out = llm.call("chat", {"system": BRIEF, "user": user, "temperature": 0.0,
                                    "schema": plan_schema([s["skill"] for s in allowed])})
        plan = (out["json"] or {}).get("plan")
        if not plan:
            raise SkillError(f"the LLM returned no usable plan: {out['text'][:300]}")
        for n in plan:
            if n.get("skill") not in SKILLS or SKILLS[n["skill"]]["kind"] not in kinds:
                raise SkillError(f"the LLM chose a skill that is not allowed here: {n.get('skill')}")
            n["params"] = clean_params(n["skill"], n.get("params"))[0]
        plan = hoist_finds(plan)
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
            # the straight view is taken while the LLM chooses the skills (both take a few seconds)
            box = {}
            th = threading.Thread(target=lambda: box.update(err=_try(lambda: scene.capture(ctx, 0.0))), daemon=True)
            th.start()
            try:
                self.llm_steps, _ = llm_plan(ctx, self.llm, text, self.KINDS,
                                             " The robot stands still at the table: do not walk or turn.")
            finally:
                th.join()
            if box.get("err"):
                raise box["err"]
            nouns = sorted({parse(n["params"]["object"])[0] for n in self.llm_steps if n["skill"] == "find"
                            and not any(w in n["params"]["object"].lower() for w in SURFACE_WORDS)})
            views = scene.scan(ctx, nouns)
            scene.note(llm_steps=[{k: n.get(k) for k in ("skill", "params", "why")} for n in self.llm_steps],
                       nouns=nouns, start_state=ctx.link.state())
            plan = plan_task(ctx, self.llm_steps)
            self.margin = margin_check(ctx, self.llm_steps)
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
                execute(ctx, plan, on_step=on_step)
                self.state = "done"
            except (SkillError, ServiceError) as e:
                self.state, self.error = "failed", str(e)
                log.warning("play failed at %s: %s", self.current, e)
            finally:
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
    def __init__(self, ai, index):
        super().__init__(name=f"{index}:{ai.nodes[index]['skill']}")
        self.ai, self.index, self.thread = ai, index, None

    def initialise(self):
        self.thread = threading.Thread(target=self.ai.execute, args=(self.index,), daemon=True)
        self.thread.start()

    def update(self):
        if self.thread.is_alive():
            return Status.RUNNING
        return Status.SUCCESS if self.ai.nodes[self.index]["status"] in ("done", "skipped") else Status.FAILURE


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
                "nodes": [{k: n.get(k) for k in ("skill", "params", "why", "status", "result", "error", "review")}
                          for n in self.nodes],
                "log": self.log[-30:], "known": self.ctx.summary()}

    def _say(self, msg):
        self.log.append(f"{time.strftime('%H:%M:%S')} {msg}")

    # ---------------------------------------------------------------- run
    def _run(self):
        try:
            self._safety(None)
            check_cool(self.ctx)
            self.ctx.forget_positions()           # people move things between tasks: always find again
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
        finally:
            try:
                if rest(self.ctx):                 # arms back to their normal position after every task
                    self._say("arms released to their normal position")
            except Exception as e:
                self._say(f"arms not released: {e}")
            Busy.owner = None
            Busy.lock.release()

    def _tick_tree(self, start):
        root = py_trees.composites.Sequence(name="task", memory=True,
                                            children=[SkillNode(self, i) for i in range(start, len(self.nodes))])
        tree = py_trees.trees.BehaviourTree(root)
        self.state = "running"
        while True:
            tree.tick()
            if root.status != Status.RUNNING:
                return root.status
            time.sleep(0.1)

    def execute(self, i):
        """One node: safety checks -> LLM review -> step gate -> skill."""
        n = self.nodes[i]
        try:
            if self._abort.is_set():
                raise SkillError("aborted")
            n["status"] = "checking"
            self._safety(n)
            n["status"] = "reviewing"
            review = self._review(i)
            n["review"] = review
            if review["decision"] == "abort":
                raise SkillError(f"LLM stopped the task: {review['reason']}")
            if review["decision"] == "skip":
                n["status"] = "skipped"
                self._say(f"step {i + 1} {n['skill']} skipped: {review['reason']}")
                return
            params, missing = clean_params(n["skill"], review.get("params") or n["params"])
            n["params"] = n["params"] if missing else params     # a review that drops a needed input is ignored
            if self.mode == "step":
                n["status"] = "waiting"
                self.state = "waiting"
                self._gate.clear()
                self._gate.wait()
                if self._abort.is_set():
                    raise SkillError("aborted")
            self.state = "running"
            n["status"] = "running"
            self._say(f"step {i + 1}: {n['skill']} {json.dumps(n['params'])}")
            n["result"] = run_skill(self.ctx, n["skill"], n["params"])
            n["status"] = "done"
        except (SkillError, TypeError, ValueError) as e:
            n["status"], n["error"] = "failed", str(e)
            self._say(f"step {i + 1} {n['skill']} failed: {e}")

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

    def _review(self, i):
        n = self.nodes[i]
        prev = [{"skill": m["skill"], "status": m["status"], "result": m.get("result")} for m in self.nodes[:i]]
        user = (f"Task: {self.task}\nPlan: {[(m['skill'], m['params']) for m in self.nodes]}\nDone so far: "
                f"{json.dumps(prev, default=str)[:3000]}\nWhat the robot knows now: {json.dumps(self.ctx.summary())}\n"
                f"Next step {i + 1}: skill {n['skill']} with params {json.dumps(n['params'])} "
                f"(parameters: {json.dumps(SKILLS[n['skill']]['params'])}).\n{RULES}\nDecide: run it (you may "
                'correct its params), skip it if it is no longer needed, or abort if the task cannot succeed. '
                'Return JSON {"decision": "run|skip|abort", "params": {...}, "reason": ...}.')
        with self.ctx.processing("llm"):
            out = self.llm.call("chat", {"system": BRIEF, "user": user, "schema": REVIEW_SCHEMA, "max_tokens": 1024})
        r = out["json"] or {}
        if r.get("decision") not in ("run", "skip", "abort"):
            return {"decision": "run", "params": n["params"], "reason": "review unreadable; running as planned"}
        return r
