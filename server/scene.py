"""Task scene: what the robot knows about the table for ONE task (docs/PIPELINE.md).

Built when Generate is pressed, used by every step of that task, forgotten once the task is over and the hands
are down (TaskScene.forget). Nothing about a scene outlives its task.

  views      the head camera's fine depth at fixed waist yaws (pitch stays locked): the straight view, plus a
             side view only where an object of the task is cut off at that image side
  instances  SAM 3 masks of each noun of the task, as 3D points (pelvis frame)
  objects    whole objects: a mask inside exactly one other is a fragment of it (merged), a mask holding two or
             more others is a group (dropped), the same object in two views is one
  entities   what words refer to: objects resting on each other form one stack (things are put on its top);
             'left' / 'right' / 'nearest' ... choose among entities only, never among fragments
  map        the obstacle map fused from all views (found objects cut out: they are planned as fitted boxes)
Every view is recorded (logs/runs/<id>_task/) so scripts/replay_task.py re-plans the task offline with this code.
"""
import base64
import json
import logging
import math
import os
import time

import cv2
import numpy as np

from common import config

from .motion import SkillError, cuboid

log = logging.getLogger("scene")
QUALS = {"left", "right", "nearest", "closest", "farthest", "middle", "center", "centre"}


FILLER = {"side", "hand", "the", "a", "an"}   # 'right side', 'left-hand side': position, not the object's name


def parse(label):
    """'right side boxes' -> ('box', ['right']): SAM 3 sees the noun (singular: 'boxes' and 'box' are one noun,
    2026-10-29: the plural got only a whole-stack mask; 'side box' found no stack at all in one view), code applies
    the position words."""
    words = label.lower().replace("-", " ").split()
    noun = [w for w in words if w not in QUALS and w not in FILLER] or words
    return " ".join(noun[:-1] + [_singular(noun[-1])]), [w for w in words if w in QUALS]


def _singular(w):
    if len(w) > 3 and w.endswith("es") and w[-3] in "xsz" or w.endswith(("ches", "shes")):
        return w[:-2]
    return w[:-1] if len(w) > 3 and w.endswith("s") and not w.endswith("ss") else w


def clean_depth(depth, scale, S, fx):
    """Flying pixels: at a depth jump (a box edge, the case's edge 80 cm above the floor) the camera returns
    in-between depths, thin lines in mid air that the Default preset keeps (2026-10-29: a 18 x 14 cm box fitted
    29 x 20 cm, and the lines were obstacles in the map). A pixel whose 5 x 5 neighbourhood spans more than
    max(jump_m, jump_rel * its depth) is dropped; surfaces seen up to ~75 deg off-normal keep their pixels."""
    d = depth.astype(np.float32) * scale
    valid = d > 0
    k = np.ones((5, 5), np.uint8)
    lo = cv2.erode(np.where(valid, d, 1e3).astype(np.float32), k)
    hi = cv2.dilate(np.where(valid, d, 0.0).astype(np.float32), k)
    bad = valid & ((hi - lo) > np.maximum(S["jump_m"], S["jump_rel"] * d))
    # speckles: with the jumps cut, a small isolated patch is a false stereo match (2026-10-29: blobs on the bare
    # floor read 0.2 m instead of 1.2 m and blocked the arm in the map), not an object
    n, lab, stats, _ = cv2.connectedComponentsWithStats((valid & ~bad).astype(np.uint8), connectivity=8)
    area = stats[:, cv2.CC_STAT_AREA].astype(np.float64)
    mean_d = np.bincount(lab.ravel(), weights=d.ravel(), minlength=n) / np.maximum(area, 1)
    small = area * (mean_d / fx) ** 2 < S["min_blob_m2"]     # its size in the world, not in pixels
    small[0] = False                                    # label 0 = no depth
    out = depth.copy()
    out[bad | small[lab]] = 0
    return out


class View:
    def __init__(self, rgb, depth, depth_scale, intrinsics, q, T, imu_yaw, clean=True):
        """depth: as the camera gave it (kept as raw, recorded); the scene and the map use it without flying pixels
        (clean=False: recordings made before 2026-10-29 16:00, replayed as they ran)."""
        self.rgb, self.raw, self.depth_scale, self.intrinsics = rgb, depth, depth_scale, intrinsics
        self.depth = clean_depth(depth, depth_scale, config.load()["scene"], intrinsics["fx"]) if clean else depth
        self.q, self.T, self.imu_yaw = q, T, imu_yaw       # joints (rad), torso in pelvis at capture, IMU yaw (rad)

    @property
    def yaw_deg(self):
        return math.degrees(self.q[12])


class TaskScene:
    def __init__(self, ctx, text, run_dir=None, record=True):
        self.cfg = config.load()
        self.S = self.cfg["scene"]
        self.text, self.views, self.inst, self.used = text, [], {}, {}
        self.map_built = False
        self.geo = ctx.geometry
        self.run_dir = run_dir
        if run_dir is None and record and self.S["record"]:
            self.run_dir = os.path.join(config.path(self.cfg["logging"]["dir"]), "runs",
                                        time.strftime("%Y%m%d_%H%M%S") + "_task")
        if self.run_dir:
            os.makedirs(self.run_dir, exist_ok=True)
        self.record = {"text": text, "views": [], "found": {}, "t": time.time(),
                       "camera": ctx.geometry.mount() if ctx.geometry is not None else None, "depth_clean": True}

    # ---------------------------------------------------------------- views
    def capture(self, ctx, yaw_deg):
        """Turn the waist yaw (only) to yaw_deg, wait until it has settled, take a fine depth view."""
        from .skills import _pelvis_yaw, _waist_settled, check_waist_lock
        check_waist_lock(ctx)
        q = ctx.motion.q_now()
        goal = math.radians(yaw_deg)
        if abs(q[12] - goal) > math.radians(1.0):
            secs = max(1.0, abs(math.degrees(goal - q[12])) / ctx.cfg["arms"]["waist_speed_dps"])
            ctx.motion.move_waist([goal, q[13], q[14]], secs)
        _waist_settled(ctx)
        f = ctx.camera.fine(self.S["scan_frames"])
        q = ctx.motion.q_now(commanded=False)            # the camera is where the waist really is
        v = View(f.rgb, f.depth, f.depth_scale, f.intrinsics, q,
                 ctx.geometry.torso_in_pelvis(ctx.motion.waist_of(q)), _pelvis_yaw(ctx))
        self._add_view(v)
        return v

    def _add_view(self, v):
        k = len(self.views)
        self.views.append(v)
        if self.run_dir:
            np.savez_compressed(os.path.join(self.run_dir, f"view_{k}.npz"), rgb=v.rgb, depth=v.raw)
        self.record["views"].append({"file": f"view_{k}.npz", "depth_scale": v.depth_scale, "intrinsics": v.intrinsics,
                                     "q": {str(i): float(a) for i, a in v.q.items()}, "T": v.T.tolist(),
                                     "imu_yaw": v.imu_yaw})
        log.info("view %d: waist yaw %.1f deg", k, v.yaw_deg)

    def scan(self, ctx, nouns):
        """The straight view (taken already, or now), then a side view only where an object of the task is cut off
        at that image side (or not seen at all), then the waist straight again."""
        if not self.views:
            self.capture(ctx, 0.0)
        need = set()
        for noun in nouns:
            found = [i for i in self._segment(ctx, noun, 0) if i["ok"]]
            if not found:
                need |= {1, -1}
            for i in found:
                need |= {s for s in (1, -1) if i["cut"] == s}
        for s in sorted(need, reverse=True):                 # left first, then right
            self.capture(ctx, s * self.S["scan_side_deg"])
        if need:
            self.capture_straight(ctx)
        return [v.yaw_deg for v in self.views]

    def capture_straight(self, ctx):
        q = ctx.motion.q_now()
        if abs(q[12]) > math.radians(1.0):
            secs = max(1.0, abs(math.degrees(q[12])) / ctx.cfg["arms"]["waist_speed_dps"])
            ctx.motion.move_waist([0.0, q[13], q[14]], secs)

    # ---------------------------------------------------------------- 3D instances
    def _pixels_to_pelvis(self, ctx, v, us, vs):
        z = v.depth[vs, us].astype(np.float32) * v.depth_scale
        k = v.intrinsics
        keep = (z > 0.05) & (z < self.S["depth_max_m"])
        us, vs, z = us[keep], vs[keep], z[keep]
        cam = np.stack([(us - k["ppx"]) * z / k["fx"], (vs - k["ppy"]) * z / k["fy"], z], axis=1)
        p = cam @ ctx.geometry.R.T + ctx.geometry.t
        return p @ v.T[:3, :3].T + v.T[:3, 3]

    def _segment(self, ctx, noun, k):
        """SAM 3 instances of `noun` in view k, as 3D points (cached per view and noun)."""
        key = (noun, k)
        if key in self.inst:
            return self.inst[key]
        v, S = self.views[k], self.S
        ok, png = cv2.imencode(".png", v.rgb)
        with ctx.processing("owl"):
            res = ctx.vision.call("segment", {"texts": [noun], "image_b64": base64.b64encode(png.tobytes()).decode()})
        H, W = v.depth.shape
        e = S["mask_erode_px"]
        out = []
        for j, r in enumerate(res["instances"]):
            mask = cv2.imdecode(np.frombuffer(base64.b64decode(r["mask_png"]), np.uint8), cv2.IMREAD_GRAYSCALE) > 0
            inner = cv2.erode(mask.astype(np.uint8), np.ones((2 * e + 1, 2 * e + 1), np.uint8)) > 0
            vs, us = np.nonzero(inner[::S["point_step_px"], ::S["point_step_px"]])
            pts = self._pixels_to_pelvis(ctx, v, us * S["point_step_px"], vs * S["point_step_px"])
            x1, y1, x2, y2 = r["box"]
            # image left = robot's left: an object touching the left edge continues to the left (+yaw)
            cut = 1 if x1 <= S["edge_px"] else (-1 if x2 >= W - S["edge_px"] else 0)
            i = {"noun": noun, "view": k, "score": r["score"], "box": r["box"], "mask": mask, "pts": pts, "cut": cut,
                 "far_cut": y1 <= S["top_edge_px"], "id": f"{k}.{j}"}   # touches the image top = its far side
            i["ok"], i["why"] = self._usable(i)
            if i["ok"]:
                i["vox"] = self._vox(pts)
            out.append(i)
            log.info("view %d '%s' #%d: score %.2f, %d points%s", k, noun, j, r["score"], len(pts),
                     "" if i["ok"] else f" (not an object: {i['why']})")
        self.inst[key] = out
        return out

    def _usable(self, i):
        p, M = i["pts"], self.cfg["manipulation"]
        if len(p) < self.S["min_points"]:
            return False, "too few depth points"
        lo, hi = np.percentile(p, 2, axis=0), np.percentile(p, 98, axis=0)
        if float(np.median(p[:, 2])) < self.cfg["perception"]["floor_below_m"]:
            return False, "on the floor"
        if max(hi - lo) > M["max_object_size_m"]:
            return False, f"{100 * max(hi - lo):.0f} cm across: a surface, not an object"
        return True, ""

    def _vox(self, pts):
        return set(map(tuple, np.floor(pts / self.S["voxel_m"]).astype(np.int64).tolist()))

    @staticmethod
    def _dilate(vox):
        d = set()
        for (a, b, c) in vox:
            for i in (-1, 0, 1):
                for j in (-1, 0, 1):
                    for k in (-1, 0, 1):
                        d.add((a + i, b + j, c + k))
        return d

    def _inside(self, a, b):
        """Share of a's voxels within b (b grown by one voxel)."""
        if "voxd" not in b:
            b["voxd"] = self._dilate(b["vox"])
        return len(a["vox"] & b["voxd"]) / max(1, len(a["vox"]))

    # ---------------------------------------------------------------- objects and entities
    def objects(self, ctx, nouns):
        """Whole physical objects from the masks of every noun in every view (the words only choose among them, so
        an object measures the same whichever word names it). A mask holding two or more others that are apart from
        each other is a group of objects: dropped. The rest are joined in 3D: a mask mostly inside another is a
        fragment of it, or the same object seen from another view or under another word."""
        S = self.S
        noun = "/".join(nouns)
        inst = [i for nn in nouns for k in range(len(self.views)) for i in self._segment(ctx, nn, k) if i["ok"]]
        apart = lambda a, b: max(self._inside(a, b), self._inside(b, a)) < S["same_object_overlap"]
        groups = set()
        for c in inst:
            inner = [i for i in inst if i is not c and len(i["vox"]) < len(c["vox"]) and self._inside(i, c) > S["inside_ratio"]]
            if any(apart(a, b) for n, a in enumerate(inner) for b in inner[n + 1:]):
                groups.add((c["noun"], c["view"], c["id"]))
                for i in inner:                 # what the group was called applies to what it holds
                    i.setdefault("also", set()).add(c["noun"])
                log.info("'%s' %s holds %d separate others: a group, dropped", noun, c["id"], len(inner))
        inst = [i for i in inst if (i["noun"], i["view"], i["id"]) not in groups]
        self._groups = groups
        parent = list(range(len(inst)))

        def root(i):
            while parent[i] != i:
                i = parent[i]
            return i
        for x in range(len(inst)):
            for y in range(x + 1, len(inst)):
                a, b = inst[x], inst[y]
                if max(self._inside(a, b), self._inside(b, a)) > S["inside_ratio"] or \
                        min(self._inside(a, b), self._inside(b, a)) > S["same_object_overlap"]:
                    parent[root(x)] = root(y)
        comps = {}
        for x in range(len(inst)):
            comps.setdefault(root(x), []).append(inst[x])
        objs, bg = [], self._background(ctx)
        for members in comps.values():
            best = {}
            for i in members:                   # per view the largest mask (the same pixels under two words once)
                if i["view"] not in best or len(i["pts"]) > len(best[i["view"]]["pts"]):
                    best[i["view"]] = i
            o = {"pts": np.vstack([i["pts"] for i in best.values()]), "score": max(i["score"] for i in members),
                 "ids": [f"{i['noun']}:{i['id']}" for i in members], "views": {}, "cut": {},
                 "nouns": {i["noun"] for i in members} | set().union(*(i.get("also", set()) for i in members))}
            for i in sorted(members, key=lambda i: len(i["pts"])):     # per view: its largest mask
                o["views"][i["view"]], o["cut"][i["view"]] = i, i["cut"]
            if len(members) > 1:
                log.info("'%s': %s are one object", noun, o["ids"])
            o["members"] = members
            o.update(self._fit(o["pts"], bg, self._best_view(o)))
            extra = self._grow(ctx, o)
            if len(extra):
                o["pts"], o["extra"] = np.vstack([o["pts"], extra]), extra
                o.update(self._fit(o["pts"], bg, self._best_view(o)))
            objs.append(o)
        return self._join_slivers(objs, bg)

    def _best_inst(self, o):
        """The mask of the view that sees the object whole and nearest the image centre (None: never whole)."""
        W = self.views[0].depth.shape[1]
        whole = [i for k, i in o["views"].items() if o["cut"][k] == 0 and not i["far_cut"]] or \
                [i for k, i in o["views"].items() if o["cut"][k] == 0]
        if not whole:
            return None
        return min(whole, key=lambda i: abs((i["box"][0] + i["box"][2]) / 2 - W / 2))

    def _best_view(self, o):
        """The points of the best view (with what _grow added there), or None: all views."""
        i = self._best_inst(o)
        if i is None:
            return None
        return i["pts"] if o.get("extra") is None else np.vstack([i["pts"], o["extra"]])

    def _grow(self, ctx, o):
        """A mask edge can stop short of the object (2026-10-29: SAM left out the 2 cm dark rim of a box top, the box
        read 12.1 cm wide instead of 14 and the rim, outside every mask, became an obstacle where the glove goes).
        Depth points in a band of grow_m around the best view's mask are the object's too if they stand above its
        support (not the surface it is on), not above its top, within grow_m of its points and in no other object's
        mask."""
        S = self.S
        i = self._best_inst(o)
        if i is None:
            return np.zeros((0, 3))
        k, v = i["view"], self.views[i["view"]]
        mask = np.zeros(v.depth.shape, bool)
        for m in o["members"]:
            if m["view"] == k:
                mask |= m["mask"]
        others = np.zeros(v.depth.shape, bool)
        for (noun, kk), insts in self.inst.items():
            for j in insts if kk == k else []:
                if j["ok"] and not any(j is m for m in o["members"]) and (j["noun"], j["view"], j["id"]) not in self._groups:
                    others |= j["mask"]
        d = v.depth[mask & (v.depth > 0)]
        if d.size == 0:
            return np.zeros((0, 3))
        r = int(math.ceil(v.intrinsics["fx"] * S["grow_m"] / (float(np.median(d)) * v.depth_scale)))
        band = (cv2.dilate(mask.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))) > 0) \
            & ~mask & ~others & (v.depth > 0)
        vs, us = np.nonzero(band)
        P = self._pixels_to_pelvis(ctx, v, us, vs)
        sup, top = o["support_z"], o["top_z"]
        lo = sup + max(S["body_above_support_m"], S["body_above_frac"] * (top - sup))
        P = P[(P[:, 2] > lo) & (P[:, 2] <= top + S["above_top_m"])]
        body = o["pts"][(o["pts"][:, 2] > lo) & (o["pts"][:, 2] <= top + S["above_top_m"])]
        if not len(P) or not len(body):
            return np.zeros((0, 3))
        # connected to the object: within grow_m of its points seen from above (5 mm grid)
        g = 0.005
        x0, y0 = body[:, :2].min(axis=0) - S["grow_m"] - g
        cell = lambda q: np.floor((q[:, :2] - [x0, y0]) / g).astype(int)
        cb = cell(body)
        n = int(math.ceil(S["grow_m"] / g))
        H, W = cb.max(axis=0) + n + 2
        grid = np.zeros((H, W), np.uint8)
        grid[cb[:, 0], cb[:, 1]] = 1
        grid = cv2.dilate(grid, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * n + 1, 2 * n + 1)))
        cp = cell(P)
        inside = (cp[:, 0] >= 0) & (cp[:, 1] >= 0) & (cp[:, 0] < H) & (cp[:, 1] < W)
        keep = np.zeros(len(P), bool)
        keep[inside] = grid[cp[inside, 0], cp[inside, 1]] > 0
        if keep.sum() >= 10:
            log.info("%s: %d depth points just outside its mask are its edge (grown)", o["ids"], int(keep.sum()))
            return P[keep]
        return np.zeros((0, 3))

    def _join_slivers(self, objs, bg):
        """An object seen only cut off at an image side (a sliver of a face) belongs to the object whose footprint
        (grown by stack_pad_m) holds most of its points, if one does."""
        S, out = self.S, []
        whole = [o for o in objs if any(c == 0 for c in o["cut"].values())]
        for o in objs:
            if any(o is w for w in whole):
                out.append(o)
                continue
            best, share = None, 0.0
            for w in whole:
                c, s_ = math.cos(-w["yaw"]), math.sin(-w["yaw"])
                rel = (o["pts"][:, :2] - w["center"][:2]) @ np.array([[c, s_], [-s_, c]]).T
                f = float(np.mean((np.abs(rel[:, 0]) <= w["dims"][0] / 2 + S["stack_pad_m"])
                                  & (np.abs(rel[:, 1]) <= w["dims"][1] / 2 + S["stack_pad_m"])))
                if f > share:
                    best, share = w, f
            if best is not None and share > S["inside_ratio"]:
                log.info("%s (cut off at the image side) is part of %s", o["ids"], best["ids"])
                best["pts"] = np.vstack([best["pts"], o["pts"]])
                best["ids"] += o["ids"]
                best["nouns"] |= o["nouns"]
                best.update(self._fit(best["pts"], bg, self._best_view(best)))
            else:
                out.append(o)
        return out

    def _background(self, ctx):
        """Depth points of every view outside every object mask (the support, rails, unknown things)."""
        if getattr(self, "_bg", None) is not None and self._bg_n == len(self.views):
            return self._bg
        step, parts = self.S["point_step_px"] * 2, []
        for k, v in enumerate(self.views):
            m = np.zeros(v.depth.shape, bool)
            for (noun, kk), inst in self.inst.items():
                if kk == k:
                    for i in inst:
                        m |= i["mask"]
            vs, us = np.nonzero(~m[::step, ::step])
            parts.append(self._pixels_to_pelvis(ctx, v, us * step, vs * step))
        bg = np.vstack(parts) if parts else np.zeros((0, 3))
        self._bg, self._bg_n = bg[bg[:, 2] > self.cfg["perception"]["floor_below_m"]], len(self.views)
        return self._bg

    def _footprint(self, xy):
        """Minimum-area rectangle of points seen from above (p1-p99 trimmed), axes named so y (the squeeze axis) is
        the one closest to left-right: (cx, cy, dx, dy, yaw, raw a, raw b) in metres / radians / mm."""
        lo, hi = np.percentile(xy, 1, axis=0), np.percentile(xy, 99, axis=0)
        xy = xy[np.all((xy >= lo) & (xy <= hi), axis=1)]
        (mx, my), (a, b_), ang = cv2.minAreaRect((xy * 1000).astype(np.float32))
        yaw = math.radians(ang)
        ex = np.array([math.cos(yaw), math.sin(yaw)])
        if abs(ex[1]) > abs(ex[0]):
            yaw += math.pi / 2
            a, b_ = b_, a
        yaw = (yaw + math.pi / 2) % math.pi - math.pi / 2
        if yaw > math.pi / 4:
            yaw, a, b_ = yaw - math.pi / 2, b_, a
        elif yaw < -math.pi / 4:
            yaw, a, b_ = yaw + math.pi / 2, b_, a
        return mx / 1000, my / 1000, a / 1000 + self.S["edge_bias_m"], b_ / 1000 + self.S["edge_bias_m"], yaw, a, b_

    def _support(self, bg, cx, cy, dx, dy, top, pts):
        """Height of the surface in a ring around the footprint, below the top (else the object's lowest points)."""
        from .geometry import Geometry
        r = math.hypot(dx, dy) / 2
        d = np.hypot(bg[:, 0] - cx, bg[:, 1] - cy) if len(bg) else np.zeros(0)
        ring = bg[(d > r + 0.01) & (d < r + self.S["support_ring_m"]) & (bg[:, 2] < top - 0.02)] if len(bg) else bg
        support = Geometry.plane_height(ring[:, 2]) if len(ring) else None
        return float(np.percentile(pts[:, 2], 3)) if support is None else support

    def _fit(self, pts, bg, foot=None):
        """Oriented box of an object's points: top = 97th percentile, support = the surface around it, footprint =
        minimum-area rectangle seen from above of `foot` (its best single view: views disagree by 1-2 cm, and their
        union smeared a 14 cm box to 21 cm, 2026-10-29)."""
        from .geometry import Geometry
        z = pts[:, 2]
        top = Geometry.plane_height(z[z > np.percentile(z, 30)])  # its top face: the densest slab of its upper part (a dark,
        if top is None:                                   # glossy top threw ~10 % of its points up to 2.5 cm above it,
            top = float(np.percentile(z, 97))             # 2026-10-29: 9.1 cm box read 12.5 cm tall and 18 cm wide)
        cap = top + self.S["above_top_m"]
        pts = pts[pts[:, 2] <= cap]
        foot = pts if foot is None else foot[foot[:, 2] <= cap]
        cx, cy, dx, dy, yaw, a, b_ = self._footprint(foot[:, :2])
        support = self._support(bg, cx, cy, dx, dy, top, pts)
        # the mask's spill onto what the object stands on (a case rail, the table edge) is not the object: its
        # footprint is its body above the support (2026-10-29: a case rail 2.5 cm high beside a box made it 18-27 cm wide)
        body = foot[foot[:, 2] > support + max(self.S["body_above_support_m"], self.S["body_above_frac"] * (top - support))]
        if len(body) >= self.S["min_points"]:
            cx, cy, dx, dy, yaw, a, b_ = self._footprint(body[:, :2])
            support = self._support(bg, cx, cy, dx, dy, top, pts)
        # the shape: a box's top face fills its rectangle, a cylinder's ~79 %
        tp = pts[pts[:, 2] > top - 0.02, :2]
        fill = 0.0
        if len(tp) >= 5:
            fill = float(cv2.contourArea(cv2.convexHull((tp * 1000).astype(np.float32)))) / max(1.0, a * b_)
        shape = "box" if fill > 0.88 else ("round" if fill > 0.6 else "irregular")
        x0 = cx - (abs(math.cos(yaw)) * dx + abs(math.sin(yaw)) * dy) / 2
        sup = bg[(np.abs(bg[:, 2] - support) < 0.012) & (np.abs(bg[:, 1] - cy) < 0.2) & (bg[:, 0] < cx)] if len(bg) else bg
        sfront = float(np.percentile(sup[:, 0], 2)) if len(sup) > 30 else x0 - 0.10
        return {"center": [float(cx), float(cy), float((top + support) / 2)], "dims": [float(dx), float(dy), float(top - support)],
                "yaw": float(yaw), "top_z": top, "support_z": float(support), "front_x": float(x0),
                "support_front_x": min(sfront, float(x0)), "shape": shape, "fill": round(fill, 2),
                "bottom_z": float(np.percentile(pts[:, 2], 3))}

    def entities(self, ctx, nouns):
        """Objects resting on each other form one stack entity (its top is where things are put); the rest are
        single entities."""
        S = self.S
        objs = self.objects(ctx, nouns)
        parent = list(range(len(objs)))

        def root(i):
            while parent[i] != i:
                i = parent[i]
            return i
        on = {}
        for i, up in enumerate(objs):
            for j, low in enumerate(objs):
                if i == j or low["top_z"] > up["top_z"] - 0.02 or abs(up["bottom_z"] - low["top_z"]) > S["stack_gap_m"]:
                    continue
                c, s_ = math.cos(-up["yaw"]), math.sin(-up["yaw"])
                rel = (low["pts"][:, :2] - up["center"][:2]) @ np.array([[c, s_], [-s_, c]]).T
                share = float(np.mean((np.abs(rel[:, 0]) <= up["dims"][0] / 2 + S["stack_pad_m"])
                                      & (np.abs(rel[:, 1]) <= up["dims"][1] / 2 + S["stack_pad_m"])))
                if share > S["stack_inside"]:
                    on[i] = j
                    parent[root(i)] = root(j)
        groups = {}
        for i in range(len(objs)):
            groups.setdefault(root(i), []).append(i)
        out = []
        for members in groups.values():
            members.sort(key=lambda i: objs[i]["top_z"])
            base, top = objs[members[0]], objs[members[-1]]
            if len(members) > 1:                         # each upper object rests on the one below it
                for i in members[1:]:
                    objs[i]["support_z"] = objs[on[i]]["top_z"] if i in on else objs[i]["support_z"]
            e = dict(top)
            e.update({"center": [top["center"][0], top["center"][1], (top["top_z"] + base["support_z"]) / 2],
                      "dims": [top["dims"][0], top["dims"][1], top["top_z"] - base["support_z"]],
                      "support_z": base["support_z"], "support_front_x": min(objs[i]["support_front_x"] for i in members),
                      "front_x": min(objs[i]["front_x"] for i in members), "stack_n": len(members),
                      "score": max(objs[i]["score"] for i in members), "objs": [objs[i] for i in members],
                      "nouns": set().union(*(objs[i]["nouns"] for i in members))})
            out.append(e)
        return out

    # ---------------------------------------------------------------- what the words refer to
    def find(self, ctx, label):
        """The entity `label` names (stored like the old find's result, for the arm skills)."""
        from .skills import _pelvis_yaw
        noun, quals = parse(label)
        for k in range(len(self.views)):
            self._segment(ctx, noun, k)
        nouns = sorted({nn for (nn, k) in self.inst})
        cands = [e for e in self.entities(ctx, nouns) if noun in e["nouns"]]
        for e in cands:
            log.info("find '%s': %s at (%.3f, %.3f), %s cm, top %.3f, %s, score %.2f, views %s", label,
                     f"stack of {e['stack_n']}" if e["stack_n"] > 1 else e["shape"], e["center"][0], e["center"][1],
                     [round(100 * d, 1) for d in e["dims"]], e["top_z"], f"fill {e['fill']}", e["score"],
                     sorted(set(k for o in e["objs"] for k in o["views"])))
        if not cands:
            raise SkillError(f"no '{noun}' seen on the table (views at waist {[round(v.yaw_deg) for v in self.views]} deg)")
        pick = None
        if "left" in quals:
            pick = max(cands, key=lambda e: e["center"][1])
        elif "right" in quals:
            pick = min(cands, key=lambda e: e["center"][1])
        elif {"nearest", "closest"} & set(quals):
            pick = min(cands, key=lambda e: math.hypot(*e["center"][:2]))
        elif "farthest" in quals:
            pick = max(cands, key=lambda e: math.hypot(*e["center"][:2]))
        elif {"middle", "center", "centre"} & set(quals):
            pick = min(cands, key=lambda e: abs(e["center"][1]))
        elif len(cands) == 1:
            pick = cands[0]
        else:                                           # never guess between similar objects (OK-Robot's top failure)
            where = ", ".join(f"({e['center'][0]:.2f}, {e['center'][1]:+.2f})" for e in cands)
            raise SkillError(f"'{label}' matches {len(cands)} objects at {where} m: say which one "
                             f"(left / right / nearest / farthest / middle)")
        def far(o, k):                                   # a real mask of it in view k (not a stray sliver) touches the top
            ms = [m for m in o["members"] if m["view"] == k]
            return any(m["far_cut"] and len(m["pts"]) >= 0.25 * max(len(x["pts"]) for x in ms) for m in ms)
        # seen to its far side: in a view showing a good part of it (half the points of its best view; 2026-10-29: a
        # sliver of a stack at an image side passed and the stack's unseen far 3 cm stayed an obstacle under the
        # place) no mask of it touches the top
        size = lambda o, k: sum(len(m["pts"]) for m in o["members"] if m["view"] == k)
        unseen = [o for o in pick["objs"]
                  if not any(not far(o, k) and size(o, k) >= 0.5 * max(size(o, j) for j in o["views"]) for k in o["views"])]
        if unseen:                                       # its far part is above the top of every view: size unknown
            raise SkillError(f"the camera cannot see the far side of '{label}' (it runs out of the top of every view, "
                             f"top {max(o['top_z'] for o in unseen):.2f} m): it is too tall for the head camera's view (if the "
                             f"camera points further down than it should, scripts/camera_tilt.py shows by how much)")
        for k, v in ctx.bb["objects"].items():          # two names must not be one object
            if k != label and math.hypot(v["center"][0] - pick["center"][0], v["center"][1] - pick["center"][1]) < 0.05:
                raise SkillError(f"'{label}' is the same object as '{k}'")
        keys = ("center", "dims", "yaw", "top_z", "support_z", "front_x", "support_front_x", "shape", "stack_n")
        b = {k: pick[k] for k in keys}
        b["distance_m"] = math.hypot(*b["center"][:2])
        b["noun"], b["t"] = noun, time.time()
        b["imu_yaw"] = self.views[0].imu_yaw if self.views[0].imu_yaw is not None else _pelvis_yaw(ctx)
        # the straight view's own measurement of it, for the check before playing
        o0 = [o for o in pick["objs"] if 0 in o["views"] and o["cut"].get(0) == 0]
        b["view0"] = self._fit(o0[-1]["views"][0]["pts"], self._background(ctx))["center"][:2] if o0 else None
        ctx.bb["objects"][label] = b
        self.used[label] = b
        self.record["found"][label] = {k: v for k, v in b.items()}
        self._annotate()
        return {"object": label, "center_m": [round(v, 3) for v in b["center"]], "size_m": [round(v, 3) for v in b["dims"]],
                "distance_m": round(b["distance_m"], 3), "shape": b["shape"], "stack": b["stack_n"],
                "candidates": len(cands)}

    def surface(self, ctx, label):
        """A table-like surface: the dominant horizontal plane in front of the robot (straight view)."""
        from .geometry import Geometry
        bg = self._background(ctx)
        bg = bg[(bg[:, 0] > 0.1) & (bg[:, 0] < 0.9) & (np.abs(bg[:, 1]) < 0.5)]
        top = Geometry.plane_height(bg[:, 2]) if len(bg) else None
        if top is None:
            raise SkillError(f"saw no '{label}' surface")
        s = bg[np.abs(bg[:, 2] - top) < 0.02]
        x0, x1 = np.percentile(s[:, 0], [2, 98])
        y0, y1 = np.percentile(s[:, 1], [2, 98])
        ctx.bb["surfaces"][label] = {"top_z": top, "front_x": float(x0), "back_x": float(x1),
                                     "y_range": [float(y0), float(y1)], "t": time.time()}
        return {"surface": label, "front_edge_m": round(float(x0), 3), "height_m": round(top, 3)}

    # ---------------------------------------------------------------- the map of this task
    def ensure_map(self, ctx, cuts):
        """The obstacle map fused from every view, built once per task; every found object is cut out (planned as
        its fitted box) and so is the support around it (the table slab)."""
        if self.map_built or ctx.motion.planner is None:
            return
        exclude = [cuboid(k, v["center"], v["dims"], v.get("yaw", 0.0)) for k, v in ctx.bb["objects"].items()] + cuts
        r = ctx.motion.build_map_views(self.views, ctx.geometry, exclude)
        self.map_built = True
        log.info("task map from %d views: %s", len(self.views), r.get("pixels"))

    # ---------------------------------------------------------------- closed-loop checks while playing
    def check_unchanged(self, ctx):
        """Before playing: the straight view again; every object it saw whole must be where Generate measured it
        (turned by how far the robot has turned since). Else the plan is stale: Generate again."""
        from .skills import _pelvis_yaw
        self.capture_straight(ctx)
        probe = TaskScene(ctx, self.text, run_dir=os.path.join(self.run_dir, "play_check") if self.run_dir else None)
        probe.capture(ctx, 0.0)
        now = _pelvis_yaw(ctx)
        report = {}
        for label, b in self.used.items():
            if b.get("view0") is None:
                report[label] = "not whole in the straight view: checked at the line-up"
                continue
            x, y = b["view0"]
            if now is not None and b.get("imu_yaw") is not None:
                d = now - b["imu_yaw"]
                x, y = math.cos(-d) * x - math.sin(-d) * y, math.sin(-d) * x + math.cos(-d) * y
            seen = [o for o in probe.objects(ctx, [b["noun"]]) if 0 in o["views"] and o["cut"].get(0) == 0]
            if not seen:
                raise SkillError(f"'{label}' is no longer seen where Generate found it: press Generate again")
            o = min(seen, key=lambda o: math.hypot(o["center"][0] - x, o["center"][1] - y))
            moved = math.hypot(o["center"][0] - x, o["center"][1] - y)
            report[label] = round(100 * moved, 1)
            log.info("play check '%s': %.1f cm from where Generate measured it", label, 100 * moved)
            if moved > self.S["moved_tol_m"]:
                raise SkillError(f"'{label}' moved {100 * moved:.1f} cm since Generate (limit "
                                 f"{100 * self.S['moved_tol_m']:.1f}): press Generate again")
        self.record["play_check"] = report
        self.save()
        return report

    def measure_held(self, ctx, noun, palms_mid, above_z):
        """The held object's centre (x, y) as the camera sees it now: the instance of its noun above above_z (the
        target's top) nearest the palms' midpoint, its top face's rectangle. Camera to camera with the target: the
        arm calibration and the object's place in the grip cancel out."""
        from .skills import _waist_settled
        _waist_settled(ctx)
        f = ctx.camera.fine(self.S["align_frames"])
        q = ctx.motion.q_now(commanded=False)
        from .skills import _pelvis_yaw
        probe = TaskScene(ctx, self.text, record=False)
        probe.views.append(View(f.rgb, f.depth, f.depth_scale, f.intrinsics, q,
                                ctx.geometry.torso_in_pelvis(ctx.motion.waist_of(q)), _pelvis_yaw(ctx)))
        inst = [i for i in probe._segment(ctx, noun, 0) if i["ok"]]
        inst = [i for i in inst if float(np.percentile(i["pts"][:, 2], 20)) > above_z]   # held over the target
        if not inst:
            return None
        i = min(inst, key=lambda i: float(np.hypot(*(np.median(i["pts"][:, :2], axis=0) - palms_mid[:2]))))
        p = i["pts"]
        top = p[p[:, 2] > np.percentile(p[:, 2], 97) - 0.02, :2]
        if len(top) < 30:
            return None
        lo, hi = np.percentile(top, 2, axis=0), np.percentile(top, 98, axis=0)
        top = top[np.all((top >= lo) & (top <= hi), axis=1)]
        (mx, my), _, _ = cv2.minAreaRect((top * 1000).astype(np.float32))
        return np.array([mx / 1000, my / 1000])

    def held_top(self, ctx, noun, palms_mid):
        """Top height of the object between the palms as the camera sees it now (None if not seen): whether a held
        object really rose does not depend on how the hands deform (2026-10-29: Inspire hands lifted a box while
        their palm points closed 1.7 cm on its flexing walls, read as a slip)."""
        from .geometry import Geometry
        from .skills import _pelvis_yaw
        f = ctx.camera.fine(self.S["align_frames"])
        q = ctx.motion.q_now(commanded=False)
        probe = TaskScene(ctx, self.text, record=False)
        probe.views.append(View(f.rgb, f.depth, f.depth_scale, f.intrinsics, q,
                                ctx.geometry.torso_in_pelvis(ctx.motion.waist_of(q)), _pelvis_yaw(ctx)))
        inst = [i for i in probe._segment(ctx, noun, 0) if i["ok"]]
        near = lambda i: float(np.hypot(*(np.median(i["pts"][:, :2], axis=0) - np.asarray(palms_mid[:2]))))
        inst = [i for i in inst if near(i) < self.S["held_near_m"]]
        if not inst:
            return None
        z = min(inst, key=near)["pts"][:, 2]
        top = Geometry.plane_height(z[z > np.percentile(z, 30)])
        return float(np.percentile(z, 97)) if top is None else float(top)

    # ---------------------------------------------------------------- record, replay, forget
    def note(self, **kw):
        self.record.update(kw)
        self.save()

    def save(self):
        if not self.run_dir:
            return
        with open(os.path.join(self.run_dir, "scene.json"), "w") as fh:
            json.dump(self.record, fh, indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))

    def _annotate(self):
        """The straight view with every found entity named (shown in the UI next to the plan)."""
        if not self.run_dir or not self.views or self.geo is None:
            return
        v = self.views[0]
        img = v.rgb.copy()
        for (noun, k), inst in self.inst.items():
            for i in inst if k == 0 else []:
                if i["ok"]:
                    img[i["mask"]] = (0.55 * img[i["mask"]] + 0.45 * np.array((60, 160, 60))).astype(np.uint8)
        Ti, kk = np.linalg.inv(v.T), v.intrinsics
        for label, b in self.used.items():
            pt = Ti[:3, :3] @ np.array([b["center"][0], b["center"][1], b["top_z"]]) + Ti[:3, 3]   # torso frame
            cam = self.geo.R.T @ (pt - self.geo.t)                                              # optical frame
            if cam[2] <= 0.05:
                continue
            u = int(np.clip(kk["fx"] * cam[0] / cam[2] + kk["ppx"], 5, img.shape[1] - 5))
            w = int(np.clip(kk["fy"] * cam[1] / cam[2] + kk["ppy"], 15, img.shape[0] - 5))
            cv2.circle(img, (u, w), 6, (0, 0, 255), -1)
            cv2.putText(img, label, (max(0, min(u + 8, img.shape[1] - 9 * len(label))), w - 6), 0, 0.55, (0, 0, 255), 2)
        cv2.imwrite(os.path.join(self.run_dir, "scene.png"), img)

    @classmethod
    def load(cls, ctx, run_dir):
        """A recorded task scene (offline replay): the same views, so the same instances, objects and map."""
        rec = json.load(open(os.path.join(run_dir, "scene.json")))
        from .geometry import Geometry                  # the camera pose used live (before 2026-10-29: the URDF's)
        ctx.geometry.use_mount(rec.get("camera") or Geometry(fix=False).mount())
        s = cls(ctx, rec["text"], run_dir=os.path.join(run_dir, "replay"))
        for v in rec["views"]:
            z = np.load(os.path.join(run_dir, v["file"]))
            s.views.append(View(z["rgb"], z["depth"], v["depth_scale"], v["intrinsics"],
                                {int(i): a for i, a in v["q"].items()}, np.array(v["T"]), v["imu_yaw"],
                                clean=rec.get("depth_clean", False)))
        s.recorded = rec
        return s

    def forget(self, ctx):
        """The task is over and the hands are down: forget the scene, its map and the grip calibration."""
        ctx.forget_positions()
        ctx.bb.pop("calib", None)
        self.note(forgotten=time.time())
        log.info("task scene forgotten (%s)", self.run_dir)

