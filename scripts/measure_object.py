"""Measure the nearest object in front of the robot from its head camera (robot standing still), by plain geometry:
  - the level frame: gravity from the torso IMU (roll, pitch), the floor from the feet (leg FK, sole) and checked
    against the floor seen in depth;
  - the object: depth points above the floor, the connected blob nearest the robot;
  - its top height above the floor (and whether the top is really in view: a flat top surface seen, not cut by the
    image edge), the forward distance from the left toe tip to its bottom front, and the camera's slant distance to
    its top front edge.
Prints one line per measurement and appends it to logs/calibration/2026-11-21_measure.jsonl (--tape adds the truth).
  python scripts/measure_object.py                      (5 frames, nearest object)
  python scripts/measure_object.py --tape 85 65 84      (true height, toe->bottom, camera->edge, cm: added to the last record)
"""
import argparse
import json
import math
import warnings
import time

import cv2
import httpx
import numpy as np

from common import config
from common.fk import _rpy
from common.joints import NAMES
from server.geometry import Geometry
from server.image_client import ImageClient

LOG = "logs/calibration/2026-11-21_measure.jsonl"
API = "http://localhost:8000"
TOE_TIP = np.array([0.14, 0.0, -0.035])   # left_ankle_roll_link: the toe contact sphere (x 0.12, r 5 mm) + the shoe's toe
SOLE = np.array([0.035, 0.0, -0.035])     # ...the sole under the middle of the foot (spheres at x -0.05 and 0.12, z -0.03)
TOP_FRACTION = 0.3   # 0.2 -> 0.3 (2026-11-21, the 4 taped objects): largest height error 2 -> 1 cm


def frames(n, host):
    cam = ImageClient()
    cam.start(host)
    out, last, t_end = [], None, time.time() + 10
    while len(out) < n and time.time() < t_end:
        f = cam.latest()
        if f is not None and f is not last:
            out.append(f)
            last = f
        time.sleep(0.03)
    cam.stop()
    if not out:
        raise SystemExit("no camera frames")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=int, default=5)
    ap.add_argument("--label", help="the object's name: measured inside the detector's box (else the nearest ahead)")
    ap.add_argument("--corridor", type=float, default=0.5, help="the object's centre within this of straight ahead (m)")
    ap.add_argument("--dump", help="save the object's points (.npz) for checking")
    ap.add_argument("--tape", nargs="+", type=float, help="true height, toe->bottom, camera->edge (cm)")
    a = ap.parse_args()
    if a.tape:
        L = open(LOG).read().splitlines()
        r = json.loads(L[-1])
        r["tape_cm"] = dict(zip(("height", "toe_to_bottom", "camera_to_edge"), a.tape))
        L[-1] = json.dumps(r)
        open(LOG, "w").write("\n".join(L) + "\n")
        print(L[-1])
        return

    status = httpx.get(f"{API}/api/status", timeout=5).json()
    st = status["state"]
    q = {NAMES[j["index"]] + "_joint": math.radians(j["deg"]) for j in st["joints"]}
    geo = Geometry()
    T_tp = geo.fk.pose("torso_link", q, base="pelvis")                 # torso in pelvis
    roll, pitch = (math.radians(v) for v in st["imu"]["torso_rpy"][:2])
    R_wt = _rpy(0.0, pitch, 0.0) @ _rpy(roll, 0.0, 0.0)              # gravity-level frame from the torso IMU
    R_wp = R_wt @ T_tp[:3, :3].T                                      # pelvis -> level (yaw: the pelvis heading)

    def lvl(p_pelvis):
        return p_pelvis @ R_wp.T

    feet = []
    for side in ("left", "right"):
        T = geo.fk.pose(f"{side}_ankle_roll_link", q, base="pelvis")
        feet.append((lvl(T[:3, :3] @ SOLE + T[:3, 3]), lvl(T[:3, :3] @ TOE_TIP + T[:3, 3])))
    floor_feet = min(f[0][2] for f in feet)
    toe = feet[0][1]                                                  # left toe tip (level frame, pelvis origin)
    cam_p = lvl((T_tp @ np.append(geo.t, 1.0))[:3])                   # camera centre

    fs = frames(a.frames, status["host"])
    depth = np.stack([f.depth.astype(np.float32) for f in fs])
    depth[depth == 0] = np.nan
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        med = np.nanmedian(depth, axis=0)
    f0 = fs[-1]
    f0.depth = np.nan_to_num(med, nan=0.0).astype(fs[-1].depth.dtype)
    # pixel rows kept with the points: the top cut by the image edge shows as object points in the first rows
    h_img, w_img = f0.depth.shape
    vs, us = np.mgrid[0:h_img:2, 0:w_img:2]
    z = f0.depth[vs, us].astype(np.float32) * f0.depth_scale
    ok = z > 0.1
    vs, us, z = vs[ok], us[ok], z[ok]
    k = f0.intrinsics
    camp = np.stack([(us - k["ppx"]) * z / k["fx"], (vs - k["ppy"]) * z / k["fy"], z], axis=1)
    P = lvl((camp @ geo.R.T + geo.t) @ T_tp[:3, :3].T + T_tp[:3, 3])

    # the floor seen: the dominant height among low points near the feet' level
    low = P[(P[:, 2] < floor_feet + 0.10) & (P[:, 2] > floor_feet - 0.10), 2]
    floor_seen = geo.plane_height(low) if low.size else None
    floor = floor_seen if floor_seen is not None and abs(floor_seen - floor_feet) < 0.05 else floor_feet
    h = P[:, 2] - floor

    # the object: points 4 cm+ above the floor ahead, 2 cm grid, the connected blob nearest the robot
    sel = (h > 0.04) & (P[:, 0] > 0.15) & (np.abs(P[:, 1]) < 1.5)
    det = None
    if a.label:                                                       # the named object: only the detector's box
        import base64
        from server.services import Service
        ok_, jpg = cv2.imencode(".jpg", f0.rgb, [cv2.IMWRITE_JPEG_QUALITY, 95])
        dets = Service("vision").call("detect", {"labels": [a.label], "image_b64": base64.b64encode(jpg).decode()})["detections"]
        if not dets:
            raise SystemExit(f"'{a.label}' not detected")
        det = max(dets, key=lambda d: d["score"])
        x1, y1, x2, y2 = det["box"]
        sel &= (us >= x1) & (us <= x2) & (vs >= y1) & (vs <= y2)
    Q, hq, vq, uq = P[sel], h[sel], vs[sel], us[sel]
    cell = 0.02
    ij = np.floor(Q[:, :2] / cell).astype(int)
    occ = {}
    for n, key in enumerate(map(tuple, ij)):
        occ.setdefault(key, []).append(n)
    seen, blobs = set(), []
    for start in sorted(occ, key=lambda c: c[0]):
        if start in seen or len(occ[start]) < 2:
            continue
        stack, idx = [start], []
        seen.add(start)
        while stack:
            c = stack.pop()
            idx += occ[c]
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    nb = (c[0] + dx, c[1] + dy)
                    if nb in occ and nb not in seen and len(occ[nb]) >= 2:
                        seen.add(nb)
                        stack.append(nb)
        if len(idx) >= 300:
            blobs.append(np.array(idx))
    if not blobs:
        raise SystemExit("no object found ahead")
    # the object straight ahead: blobs centred within --corridor of the robot's heading, the nearest of them (2026-11-21:
    # a canister at the image's right edge, 94 cm to the side, was nearer than the crate ahead and got measured)
    ahead = blobs if det else [ix for ix in blobs if abs(float(np.median(Q[ix, 1]))) <= a.corridor]
    if not ahead:
        raise SystemExit(f"no object within {a.corridor:.2f} m of straight ahead")
    b = max(ahead, key=len) if det else min(ahead, key=lambda ix: np.percentile(Q[ix, 0], 2))   # named: its biggest blob
    O, ho, vo, uo = Q[b], hq[b], vq[b], uq[b]

    # the top: the dominant height (1 cm histogram mode) among the blob's highest points; a high percentile alone took
    # the depth noise of a far top (+3.4 cm at 1.5 m on 2026-11-21: 31.4 for 28)
    # the top: 1 cm height bins over the blob's top 6 cm; the highest bin still holding TOP_FRACTION of the busiest one.
    # One rule for a flat lid seen from above, a lid seen at a grazing angle (far: only the face's end), a lid that bulges
    # (near: the lid read 26-28 cm and the busiest bin was 26 for a 28 cm box) and stray edge points above (far: 15 pts/cm
    # up to 32 cm). Tape 2026-11-21: black case 85, brown box 28 (near and far).
    top_raw = float(np.percentile(ho, 99.5))
    hb = np.round(ho * 100).astype(int)
    lo_c = int(hb.min())
    cnt = np.bincount(hb - lo_c)
    band = np.arange(len(cnt)) + lo_c >= int(round((top_raw - 0.06) * 100))
    peak = int(cnt[band].max())
    dense = np.nonzero(band & (cnt >= TOP_FRACTION * peak))[0]
    top = (int(dense.max()) + lo_c) / 100.0
    top_how = f"highest bin with >= {TOP_FRACTION:.0%} of {peak}"
    flat = O[ho > top - 0.02]                                         # the top surface seen: its depth along x
    top_depth = float(np.percentile(flat[:, 0], 95) - np.percentile(flat[:, 0], 5)) if len(flat) > 20 else 0.0
    cut = int((vo[ho > top - 0.03] <= 4).sum())                      # top points on the image's first rows
    # (a face taller than the view ends in the first image rows with no depth along x: its top is not seen)
    # the bottom front: the lowest 10 cm of what is seen of it (a crate's rim sticks out ~4 cm past its bottom; up close
    # the bottom may be below the image and the lowest visible part stands in for it)
    bottom = O[ho < float(ho.min()) + 0.10]
    front = float(np.percentile(bottom[:, 0], 2)) if len(bottom) > 30 else float(np.percentile(O[:, 0], 2))
    y_mid = float(np.median(O[:, 1]))
    # the front face's middle across: points within 6 cm behind the nearest part (blobs can run on into things behind)
    face = O[O[:, 0] < float(np.percentile(O[:, 0], 2)) + 0.06]
    y_c = float(np.percentile(face[:, 1], 2) + np.percentile(face[:, 1], 98)) / 2
    edge_pts = O[(ho > top - 0.03)]
    edge_pts = edge_pts[edge_pts[:, 0] < np.percentile(edge_pts[:, 0], 3) + 0.02] if len(edge_pts) else edge_pts
    slant = float(np.min(np.linalg.norm(edge_pts - cam_p, axis=1))) if len(edge_pts) else None
    # straight-line distances to the MIDDLE of the side facing the robot, for a box at any angle. Seen from the camera, a
    # box shows one or two sides; their top edges are the nearest top points in each image column (the lid's inside may
    # have no depth at all: the crate's glossy lid, 2026-11-21). Up to two straight lines are fitted to them (RANSAC,
    # fixed seed: same input, same answer); the side facing the robot is the one whose middle is nearest the left toe.
    # The top edge centre is that line's middle at the top height; the bottom one is the same line moved to where the
    # object's lowest 10 cm are (a crate's rim sticks out ~4 cm past its bottom). Tape 2026-11-21, crate turned ~55 deg:
    # toe 140.1 for 140, camera 176-179 for 178.
    def near_edge(mask):
        idx = np.nonzero(mask)[0]
        best = {}
        for i_, c_, r_ in zip(idx, uo[idx] // 8, np.linalg.norm(O[idx] - cam_p, axis=1)):
            if c_ not in best or r_ < best[c_][0]:
                best[c_] = (r_, i_)
        return O[[i_ for _, i_ in best.values()], :2] if best else np.zeros((0, 2))

    def fit_lines(pts, n_lines=2):
        rng, out, rest = np.random.default_rng(0), [], pts
        for _ in range(n_lines):
            if len(rest) < 8:
                break
            best = None
            for _ in range(2000):
                p_, q_ = rest[rng.choice(len(rest), 2, replace=False)]
                L = float(np.hypot(*(q_ - p_)))
                if L < 0.05:
                    continue
                u_ = (q_ - p_) / L
                inl = np.abs((rest - p_) @ np.array([-u_[1], u_[0]])) < 0.015
                if best is None or inl.sum() > best[0]:
                    best = (int(inl.sum()), p_, u_, inl)
            if best is None or best[0] < max(8, 0.15 * len(pts)):
                break
            _, p_, u_, inl = best
            t_ = (rest[inl] - p_) @ u_
            a_, b_ = p_ + u_ * np.percentile(t_, 2), p_ + u_ * np.percentile(t_, 98)
            if np.hypot(*(b_ - a_)) >= 0.10:
                out.append((a_, b_))
            rest = rest[~inl]
        return out

    window = O[:, 0] < front + 0.8                                    # not what lies far behind the object
    rim = near_edge((ho > top - 0.04) & window)
    sides = fit_lines(rim)
    if sides:
        a_, b_ = min(sides, key=lambda e: np.hypot(*((e[0] + e[1]) / 2 - toe[:2])))
        mid = (a_ + b_) / 2
        u_ = (b_ - a_) / np.hypot(*(b_ - a_))
        nrm = np.array([-u_[1], u_[0]])
        if np.dot(nrm, toe[:2] - mid) < 0:
            nrm = -nrm                                                 # toward the robot
        low = near_edge((ho < float(ho.min()) + 0.10) & window)
        t_l, o_l = (low - a_) @ u_, (low - mid) @ nrm
        on = (t_l > 0) & (t_l < np.hypot(*(b_ - a_))) & (np.abs(o_l) < 0.08)
        d_bot = float(np.clip(np.median(o_l[on]), -0.08, 0.03)) if on.sum() >= 5 else 0.0
        bot_mid = mid + nrm * d_bot
        side_len = float(np.hypot(*(b_ - a_)))
        box_yaw = float(math.degrees(math.atan2(-nrm[1], -nrm[0])))   # that side's facing: 0 = square to the robot
    else:                                                              # no straight edge found: the nearest points
        mid = bot_mid = np.array([front, y_c])
        side_len, box_yaw, d_bot = 0.0, 0.0, 0.0
    toe_c = float(np.hypot(*(bot_mid - toe[:2])))
    slant_c = float(np.linalg.norm(np.array([mid[0], mid[1], floor + top]) - cam_p))
    y_c = float(mid[1])
    # the highest a surface can be seen at this distance: the camera's upper view edge (row 0) at the object's front
    ray = np.array([(w_img / 2 - k["ppx"]) / k["fx"], (0 - k["ppy"]) / k["fy"], 1.0])
    up = lvl(((ray @ geo.R.T) @ T_tp[:3, :3].T))
    view_top = cam_p[2] + up[2] / max(1e-6, math.hypot(up[0], up[1])) * (front - cam_p[0]) - floor
    top_seen = bool(view_top > top + 0.03)   # geometry: the top edge is inside the camera's view (else 'top' = the view's limit)

    rec = {"t": time.strftime("%Y-%m-%d %H:%M:%S"), "frames": len(fs),
           "top_cut_rows_pts": cut, "camera_height_cm": round(100 * (cam_p[2] - floor), 1), "floor": {"feet": round(floor_feet, 3),
           "seen": None if floor_seen is None else round(floor_seen, 3)},
           "height_cm": round(100 * top, 1), "height_p995_cm": round(100 * top_raw, 1), "height_how": top_how, "top_seen": top_seen, "top_depth_cm": round(100 * top_depth, 1),
           "view_top_at_front_cm": round(100 * view_top, 1),
           "toe_to_bottom_cm": round(100 * (front - toe[0]), 1), "object_y_cm": round(100 * y_mid, 1),
           "camera_to_edge_cm": None if slant is None else round(100 * slant, 1),
           "toe_to_bottom_centre_straight_cm": round(100 * toe_c, 1), "camera_to_top_edge_centre_cm": round(100 * slant_c, 1),
           "object_centre_y_cm": round(100 * y_c, 1), "toe_y_cm": round(100 * toe[1], 1),
           "front_side_len_cm": round(100 * side_len, 1), "front_side_turned_deg": round(box_yaw, 1),
           "sides_seen": len(sides), "bottom_in_from_top_edge_cm": round(-100 * d_bot, 1),
           "camera_ground_to_bottom_cm": round(100 * (front - cam_p[0]), 1), "points": int(len(O)),
           "imu_torso_rpy": st["imu"]["torso_rpy"][:2],
           "label": a.label, "detection": det}
    if a.dump:
        np.savez(a.dump, O=O, h=ho, v=vo, cam=cam_p, floor=floor)
    open(LOG, "a").write(json.dumps(rec) + "\n")
    print(json.dumps(rec, indent=1))
    if not top_seen:
        print(f"NOTE: the top is not in view (the highest the camera sees at {100 * (front - toe[0]):.0f} cm is "
              f"~{100 * view_top:.0f} cm): the height above is only a lower bound")


if __name__ == "__main__":
    main()
