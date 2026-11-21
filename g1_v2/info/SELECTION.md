Frozen V2 checkpoint: model_12000.pt (from 2026-10-03_11-59-25_v2_finetune), selected on the
SELECTION seed set (10000+) only; the held-out set (0-999) was not used for selection.
Candidates: 11500, 12000, 12500, 13000, 13500, 13999 (diag_g1_yaw_bias, 16 eps/mode) and a
selection eval on 12000 / 12500 / 13999 (16 eps/scenario, 3 per turn command).
Why 12000: lowest open-loop yaw bias in all modes (-0.09/-0.09/+0.04 deg/s), only candidate
passing every symmetry + straightness gate, turn offsets 0.14/0.09 deg/s, 0 falls.
Known misses at selection: turn gain at vx0.3 0.76 (<0.8), stand_tour palm p95 3.8 cm (>3.5).
Note: the final checkpoint 13999 had regressed (+0.9..+1.3 deg/s, pair asym 1.0).
