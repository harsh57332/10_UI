# Manipulation pipeline (section 6, waist + arms, no legs)

Agreed with the user on 2026-10-26 after 40+ runs of patching one failure at a time. The goal is a robust
pipeline, not a tuned one. The scene comes from geometry, there is a margin before any motion, and the robot
checks with the camera while it plays. Any recorded task can be replayed offline through the same code.

## Why the old pipeline could not converge

Its errors were several centimetres: a box's printed top face taken as the box (3–4 cm), the robot or case
moving (up to 5 cm), the glove model (1–2.5 cm) and arm lag (1–3 cm). The motion had about 0.5 cm of room.
Every fix added a setting or a fallback (88 settings, 27 fallbacks), so a small change in the scene changed
the path a run took.

## One task, from Generate to forget

| Step | What happens | Code |
|---|---|---|
| 1. Scan | The straight view is taken while the LLM picks the skills. A side view (waist yaw ±25°, pitch locked) is added only where an object of the task is cut off at that image side. Every view is recorded. | `scene.TaskScene.capture/scan`, `ai.MotionPlanning.generate` |
| 2. Objects | SAM 3 masks each noun of the task, and each mask becomes 3D points (pelvis frame). A mask inside exactly one other is its fragment and is merged. A mask holding 2+ others is a group and is dropped. The same object seen in two views is one. Each object gets an oriented box, a support height and a shape class. | `scene.objects`, `_fit` |
| 3. Resolve | Objects resting on each other form one stack entity, and things are put on its top. left/right/nearest/farthest/middle choose among entities only. Without a word and with 2+ candidates, Generate refuses and asks which one. | `scene.entities`, `scene.find` |
| 4. Map | One obstacle map fused from all views. Found objects and the support around them are cut out (they are planned as fitted boxes). Built once per task. | `scene.ensure_map`, `motion.build_map_views`, planner `map` with `frames` |
| 5. Plan + margin | The skills plan from the live pose (`ai.plan_task`). The arm steps are planned again with every object 2 cm off in 4 directions and turned ±3° about the robot. ROBUST means all of these plan; MARGINAL means it would fail on a small error, and the UI says which. | `ai.margin_check` |
| 6. Play check | Before any arm motion, the straight view is taken again. Every object seen whole there must be within 1.5 cm of where Generate measured it, else "press Generate again". | `scene.check_unchanged` |
| 7. Pick | Unchanged and touch-checked: a straight descent beside the object, closing by feel, a test lift and a frozen grip. Picking out of a stack is refused. | `skills.pre_grasp/grasp/lift` |
| 8. Place | Squared to the target only (no unsquared fallback). Line-up camera to camera: SAM 3 measures the held object's centre and moves it over the target's camera-measured centre until within 5 mm, so the arm calibration and the grip offset cancel. Then lower until contact, let go, and check with the camera that it sits on the target. | `skills._align_by_camera`, `scene.measure_held`, `_verify_placed` |
| 9. Forget | After the task, once the hands are EMPTY and down (released after the planned arms-down), the scene, its map and the grip calibration are forgotten. A failed Generate forgets at once (nothing moved but the waist). | `scene.forget`, `ai.MotionPlanning.play` |

## Offline test bench

Every Generate writes `logs/runs/<time>_task/`: the views (`view_k.npz`), `scene.json` (text, LLM steps,
start state, found objects, plan, margin, play check, result) and `scene.png` (shown in the UI).

    python -m scripts.replay_task logs/runs/<time>_task --twice

This re-plans with the same code, the real SAM 3 and the real cuRobo, and prints IDENTICAL or the difference.
Any change to perception or planning must replay every recorded task identically, or explain the change,
before a robot run.

## Settings

`config.yaml` → `scene:` (scan, merging, stacks, play check, line-up, margin) and `services.vision.sam3_*`.

## Not in this stage

- LiDAR (Livox MID-360): its precision (about ±3 cm at 0.2 m, 0.1 m blind zone) is too coarse for grip
  geometry. It comes with locomotion (localisation, drift while walking).
- OWLv2 stays only for the UI's detect & track and section 7's coarse find while walking.
- Measuring the gloves in the camera image at the pick: the pick stays touch-checked (descent and close by feel).

## Test protocol

Supervised 3/3 means three successes in a row, the same task with different arrangements, and a failure
restarts the count. Then the user runs 5/5. Then locomotion.
