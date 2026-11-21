# G1 loco-manipulation V2 — deployment files

Policy `model_12000` from task `Unitree-G1-Flat-Loco-Man-V2`.
- **Legs (12, SDK 0–11):** driven by the policy.
- **Waist + arms (17, SDK 12–28):** held at home with `tau = kp(q_home − q) − kd·dq + tau_ff`.

| File | Content |
|---|---|
| `exported/policy.onnx` | obs[1,104] → actions[1,12], normalizer baked in. Matches the PyTorch actor to 9.5e-7. |
| `params/deploy.yaml` | joint map (identity 29), kp/kd, home pose, action scale/offset, obs terms in training order, heading PI hold (kp 1.0, ki 0.2), lateral hold off, `upper_body.tau_ff` (gravity feedforward at home) |
| `info/g1_loco_man_obs_contract.json` | training observation layout (104) |
| `info/g1_straight_line_hold.json` | heading-hold gains, calibrated in sim |
| `info/SELECTION.md` | why `model_12000` was chosen |
| `info/eval_heldout.json` / `info/eval_v1_baseline.json` | held-out sim evaluation of V2 and of v1 |