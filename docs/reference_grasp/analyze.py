"""Measure the reference grasp: box faces and the gap (depth), where the joint angles put each palm, and the
gloves' width in the image. Run: ~/miniconda3/envs/g1ui_server/bin/python docs/reference_grasp/analyze.py [tag]"""
import json
import math
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from common import config, joints              # noqa: E402
from server.geometry import Geometry           # noqa: E402

here = Path(__file__).resolve().parent
tag = sys.argv[1] if len(sys.argv) > 1 else "ref"
meta = json.load(open(here / f"{tag}_meta.json"))


class F:                                        # a saved frame
    depth = cv2.imread(str(here / f"{tag}_depth_u16.png"), cv2.IMREAD_UNCHANGED)
    rgb = cv2.imread(str(here / f"{tag}_rgb.png"))
    depth_scale, intrinsics = meta["depth_scale"], meta["intrinsics"]


geo = Geometry()
qdeg = meta["joints_deg"]
q = {n + "_joint": math.radians(qdeg[n]) for n in qdeg}
waist = [q["waist_yaw_joint"], q["waist_roll_joint"], q["waist_pitch_joint"]]
T = geo.torso_in_pelvis(waist)
P = geo.points(F, step=1, T=T)                  # pelvis frame (x forward, y left, z up)

# ---- box tops: the dominant flat height above the case, split into the two boxes by y
z = P[:, 2]
sel = P[(z > 0.12) & (z < 0.30) & (P[:, 0] > 0.15) & (P[:, 0] < 0.70)]
top = geo.plane_height(sel[:, 2], 0.005)
tops = sel[np.abs(sel[:, 2] - top) < 0.008]
ys = np.sort(tops[:, 1])
cut = ys[np.argmax(np.diff(ys))] + 1e-6         # the largest empty band in y separates the boxes
boxes = {}
for name, part in (("right box", tops[tops[:, 1] <= cut]), ("left box", tops[tops[:, 1] > cut])):
    lo, hi = np.percentile(part, 1, axis=0), np.percentile(part, 99, axis=0)
    boxes[name] = (lo, hi)
    print(f"{name}: top z {top:.3f}, x {lo[0]:.3f}..{hi[0]:.3f}, y {lo[1]:.3f}..{hi[1]:.3f} "
          f"(width {100 * (hi[1] - lo[1]):.1f} cm, length {100 * (hi[0] - lo[0]):.1f} cm, {len(part)} pts)")
bias = config.load()["perception"]["box_edge_bias_m"] / 2
gap = boxes["left box"][0][1] - boxes["right box"][1][1]
print(f"gap between the boxes (top-face edges): {100 * gap:.1f} cm; minus edge bias: {100 * (gap - 2 * bias):.1f} cm")
support = geo.plane_height(P[(P[:, 2] > -0.05) & (P[:, 2] < top - 0.04) & (P[:, 0] > 0.2) & (P[:, 0] < 0.7), 2], 0.005)
print(f"case top z {support:.3f}  -> box height {100 * (top - support):.1f} cm")

# ---- where the joint angles put the gloves (model), relative to the box faces
tcp = config.tcp_frames()
rb_lo, rb_hi = boxes["right box"]
for side in ("left", "right"):
    t = tcp[f"{side}_tcp"]
    H = geo.fk.pose(t["parent"], q, base="pelvis")
    p = H[:3, 3] + H[:3, :3] @ np.array(t["xyz"])
    face = rb_hi[1] if side == "left" else rb_lo[1]
    out = (p[1] - face) * (1 if side == "left" else -1)
    wrist = geo.fk.pose(f"{side}_wrist_yaw_link", q, base="pelvis")[:3, 3]
    hand_x = H[:3, 0]
    print(f"{side} palm point (model): {np.round(p, 3)}; {100 * out:+.1f} cm outside the right box's {side} face; "
          f"hand points {math.degrees(math.atan2(hand_x[1], hand_x[0])):+.0f} deg yaw, "
          f"{math.degrees(math.asin(-hand_x[2])):+.0f} deg down; wrist at {np.round(wrist, 3)}")

# ---- gloves in the image: dark pixels with no depth, between/beside the box tops, measured across rows
k = F.intrinsics
gray = cv2.cvtColor(F.rgb, cv2.COLOR_BGR2GRAY)
dark = (gray < 60) & (F.depth == 0)
H_img, W_img = gray.shape
cam_t, cam_R = geo.t, geo.R


def project(p_pelvis):
    pt = np.linalg.inv(T)[:3, :3] @ p_pelvis + np.linalg.inv(T)[:3, 3]     # torso
    c = cam_R.T @ (pt - cam_t)
    return k["fx"] * c[0] / c[2] + k["ppx"], k["fy"] * c[1] / c[2] + k["ppy"], c[2]


ann = F.rgb.copy()
for side in ("left", "right"):
    t = tcp[f"{side}_tcp"]
    Hh = geo.fk.pose(t["parent"], q, base="pelvis")
    pts = [Hh[:3, 3] + Hh[:3, :3] @ np.array(v) for v in ([0, 0, 0], t["xyz"], [0.13, t["xyz"][1], t["xyz"][2]])]
    uv = [project(p) for p in pts]
    for (u, v, _), col in zip(uv, ((0, 255, 255), (0, 0, 255), (255, 0, 255))):
        cv2.circle(ann, (int(u), int(v)), 5, col, -1)
    # glove width across image rows around its palm point
    u0, v0, zc = uv[1]
    widths = []
    for v in range(max(0, int(v0) - 60), min(H_img, int(v0) + 40), 4):
        row = dark[v, max(0, int(u0) - 70):min(W_img, int(u0) + 70)]
        runs, best, cur = [], 0, 0
        for b in row:
            cur = cur + 1 if b else 0
            best = max(best, cur)
        widths.append(best)
    w_px = np.median(sorted(widths)[len(widths) // 2:])     # the wider half of the rows (the glove body)
    print(f"{side} glove in the image: ~{w_px:.0f} px wide at {zc:.2f} m depth -> ~{100 * w_px * zc / k['fx']:.1f} cm "
          f"(seen from above: its thickness, palm to back, slightly foreshortened)")
cv2.imwrite(str(here / f"{tag}_annotated.png"), ann)
