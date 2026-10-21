"""Section 6 (motion planning) and section 7 (AI control).

The LLM only picks skills from the library; perception, poses and trajectories come from code.
Section 6: text -> skills (arms only) -> every trajectory planned up front -> played on request.
Section 7: text -> skill plan -> py_trees sequence. Before each node an isolated LLM call reviews it
against the live blackboard; compulsory safety checks run before every node; step mode waits for
"Next" before each node. A failed node triggers a bounded re-plan. Abort stops after the current
node; the arms keep holding whatever they hold.
"""
import json
import threading
import time

import py_trees
from py_trees.common import Status

from common import config

from .motion import ArmPlan, SkillError
from .skills import SKILLS, catalog, check_hands, execute, new_plan, run_skill

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
    "that is gripped but not lifted. Each skill only works in the hand state it lists. Keep the plan short."
)


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
    allowed = catalog(kinds)
    user = (f"Task: {text}\n\nSkills:\n{json.dumps(allowed, indent=1)}\n\nWhat the robot currently knows:\n"
            f"{json.dumps(ctx.summary())}\n\n{RULES}{extra}\n\nReturn JSON: "
            '{"plan": [{"skill": ..., "params": {...}, "why": ...}], "notes": ...}')
    with ctx.processing("llm"):
        out = llm.call("chat", {"system": BRIEF, "user": user, "schema": plan_schema([s["skill"] for s in allowed])})
    plan = (out["json"] or {}).get("plan")
    if not plan:
        raise SkillError(f"the LLM returned no usable plan: {out['text'][:300]}")
    for n in plan:
        if n.get("skill") not in SKILLS or SKILLS[n["skill"]]["kind"] not in kinds:
            raise SkillError(f"the LLM chose a skill that is not allowed here: {n.get('skill')}")
        n["params"] = n.get("params") or {}
    return hoist_finds(plan), (out["json"] or {}).get("notes", "")


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


# ==================================================================== section 6
class MotionPlanning:
    KINDS = ("perceive", "arm")

    def __init__(self, ctx, llm):
        self.ctx, self.llm = ctx, llm
        self.state, self.error, self.llm_steps, self.plan, self.current = "idle", None, [], None, None

    def generate(self, text):
        if not Busy.lock.acquire(blocking=False):
            raise SkillError(f"robot is busy ({Busy.owner})")
        Busy.owner = "motion planning"
        try:
            self.state, self.error, self.plan = "generating", None, None
            self.llm_steps, _ = llm_plan(self.ctx, self.llm, text, self.KINDS,
                                         " The robot stands still at the table: do not walk or turn.")
            plan = new_plan(self.ctx)
            for n in self.llm_steps:
                s = SKILLS[n["skill"]]
                if s["kind"] == "perceive":
                    n["result"] = s["fn"](self.ctx, **n["params"])
                else:
                    check_hands(n["skill"], plan.state["hand"])
                    n["result"] = s["fn"](self.ctx, plan, **n["params"])
            self.plan, self.state = plan, "ready"
            return self.status()
        except (SkillError, TypeError) as e:
            self.state, self.error = "failed", str(e)
            raise SkillError(str(e))
        finally:
            Busy.owner = None
            Busy.lock.release()

    def play(self):
        if self.plan is None:
            raise SkillError("generate a motion first")
        if not Busy.lock.acquire(blocking=False):
            raise SkillError(f"robot is busy ({Busy.owner})")
        Busy.owner = "motion planning"
        plan, self.plan = self.plan, None          # a plan is played once (its start state is now stale)
        self.state, self.error = "playing", None

        def worker():
            try:
                execute(self.ctx, plan, on_step=lambda i, label: setattr(self, "current", f"{i + 1}/{len(plan.steps)} {label}"))
                self.state = "done"
            except SkillError as e:
                self.state, self.error = "failed", str(e)
            finally:
                self.current, Busy.owner = None, None
                Busy.lock.release()
        threading.Thread(target=worker, daemon=True, name="plan6").start()
        return self.status()

    def status(self):
        return {"state": self.state, "error": self.error, "current": self.current,
                "llm_plan": [{k: n.get(k) for k in ("skill", "params", "why")} for n in self.llm_steps],
                "steps": self.plan.summary() if self.plan else None,
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
            n["params"] = review.get("params") or n["params"]
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
