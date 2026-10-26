# G1 work standards: manipulation (v1, 2026-10-23)

These standards are strict. A manipulation task runs only when every rule below holds. Otherwise the robot first
fixes its own position (section 7: legs; section 6: waist yaw only), or it refuses and names the rule number.
Nothing here is specific to the test boxes. Each value says where it comes from:
- [measured]: from the robot or its logs
- [swept]: from offline planning over a grid with the real planner and models
- [derived]: computed from the two above
- [estimate]: still to be verified on the robot

## 0. Reference points (robot standing in Manipulation 501)

| What | Value | Source |
|---|---|---|
| Torso origin above the floor | 0.80 m | [measured] |
| Chest front (front surface of the torso) | 9.5 cm ahead of the torso origin | [measured] |
| Toes | 7.5 cm ahead of the chest front | [measured] |
| Head camera | 1.23 m above the floor, looking 47.6° down; sees 25°–67° below horizontal, ±27° sideways | [measured] |
| Robot midline | vertical plane through the robot's centre, pointing forward | |

Distances "from the chest" are measured horizontally from the chest front.

## M1. Table

- M1.1 Table top 0.75 – 0.95 m above the floor [swept]. At 0.72 m only close, straight-ahead objects reach. At 0.68 m nothing reaches.
- M1.2 The top is flat and rigid, with a straight front edge. Raised rims or rails must be measured, never assumed
  (the depth map covers them; the table itself is modelled at its measured height).

## M2. Robot stance at the table

- M2.1 The body is square to the table edge, within ±10°.
- M2.2 Chest to table edge: **10 – 15 cm, standard 12 cm** [derived]. Below 10 cm the toes or hips touch a solid table front.

## M3. Object position (robot at its stance)

- M3.1 Object centre **24 – 32 cm from the chest** (measured from the robot's centre line through the waist, along
  the direction the waist turns to face it) [swept 2026-10-23 with the measured glove: the full stack plans at every
  tested spot in this zone; failures only at 21 – 23 cm]. At the 12 cm stance this means the **object centre is 12 –
  20 cm back from the table edge**. Beyond 32 cm the shoulders heat up (the heat gate decides).
- M3.2 The object's front face is at least 17 cm from the chest: the camera cannot see closer at 0.75 m tables [derived].
- M3.3 Sideways: the waist turns toward the object on its own, **tested up to 25° to pick it, up to 32° to place
  on it**. Beyond that, section 7 side-steps.
- M3 is the tested zone, not a wall: outside it the planner still tries, and if it cannot plan, the error names
  the M3 rule that was exceeded.
- M3.4 The whole object is inside the camera image: its sideways offset + half its width ≤ 22 cm [derived].

## M4. Object (two-palm squeeze with the gloved dummy hands)

- M4.1 Rigid and box-like, with two flat, parallel, vertical faces toward the robot's left and right (the squeeze faces).
- M4.2 **Height ≥ 6 cm** [model + robot, revised 2026-10-23]. The palm point is at the object's centre height, but
  never below 7.1 cm above the support: the gloved hand reaches 6.8 cm below it (rubber hand 5.3 cm + ~0.9 cm glove,
  measured; the earlier 2.5 cm padding guess squeezed 2.5 cm too high on the robot) and keeps 3 mm off the table.
  A short object is gripped with the lower part of the palm: the rubber palm reaches 5.3 cm below the palm point,
  and at least 4 cm of it must be on the object.
- M4.3 Width between the squeeze faces 5 – 40 cm [swept 5 – 50 cm; capped at 40 by M3.4].
- M4.4 Depth (front to back) 6 – 25 cm [estimate].
- M4.5 Object top no higher than 1.05 m above the floor; above that the camera cannot see the top face [derived].
- M4.6 The squeeze faces are within ±15° of square to the robot [measured: larger angle estimates were mis-fits].
- M4.7 **Mass ≤ 1.0 kg** [measured]. Test lifts showed 0.5 kg needs 12 – 18 N per palm, and the squeeze maximum is 30 N.
  2 kg is not supported with these gloves.

## M5. Free space around the object

- M5.1 Left and right: **≥ 5.6 cm free** beside each squeeze face [measured + swept 2026-10-23: the gloved hand is
  5.0 cm thick, palm surface to back, measured with the head camera in the user's demonstration (docs/reference_grasp:
  the glove inside a 5.6 cm gap); the stack plans down to 5.65 cm]. A glove with an object beside it comes straight
  down in the middle of that space, at least 0.3 cm from both, slowly and touch-checked (if it meets an edge it goes
  back up and the task stops, nothing is pushed). Below about 6 cm this is at the limit of the camera's and the arm
  model's accuracy (~1 cm): expect the touch check to stop some runs. A free side: the glove comes down 2 cm off
  the face. A long object is gripped up to 4 cm in front of its centre, so the wrist (bulkier than the palm) stays
  in front of it.
- M5.2 In front and behind: ≥ 8 cm free. The glove fingertips reach about 4 cm past the back face of a 12 cm-deep object [derived].
- M5.3 Above: nothing within 20 cm of the object's top. The gloves come down from 8 cm above, then lift [derived].

## M6. Placing

- M6.1 The place spot, or the object to stack on, obeys M3 and M5.
- M6.2 The target top is no higher than 1.00 m above the floor [estimate].
- M6.3 For stacking, the top face of the lower object covers at least 60 % of the held object's footprint [estimate].

## M7. Motor heat and timing

- M7.1 A task starts only with every arm motor at or below 62 °C, and its predicted peak at or below 90 °C (heat gate) [measured].
- M7.2 The arms are out holding an object for at most 45 s per pick-and-place [measured: a 43 s stack peaked at 76 °C].
- M7.3 Long holds, such as carrying while walking: object centre ≤ 22 cm from the chest and ≤ 1 kg.
  This keeps the shoulder pitch at or below 6 Nm, the long-term limit [measured heat model].

## M8. How the sections apply this

- Section 6 (no walking): checks M1 – M7 at the robot's current stance. It turns the waist on its own (M3.3), also
  to bring an object that is cut off at the image side into view; anything else refuses with the rule number.
- Section 7: first walks to the standard stance (M2: square to the edge, chest 12 cm from it, object centred), then
  checks M3 – M7, then acts. A refusal always names the rule and the measured value.

## L. Locomotion

To be defined next.

---
Basis: offline sweeps on 2026-10-23 (cuRobo with the gloved-hand model; the same goals replayed with broader IK
gave identical reach); live measurements of the joint state and camera; heat model and test logs from 2026-10-21.
