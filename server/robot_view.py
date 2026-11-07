"""Robot's view (camera panel 3): the last frame the robot looked at, drawn with what it made of it.

Frozen on the exact frame the robot used, never a live guess. Layers:
  detector boxes (OWLv2) with scores (the chosen one thick), SAM 3 masks of the task scene (grey + reason when unusable),
  the depth points each fit used (top face green, sides yellow, the surface it stands on blue), the fitted 3D box (cyan),
  a surface's front edge (orange), and the grasp: grip centre (red) and the two palm contacts (magenta).
The caption bar says what was looked for, when, and the result in the user's terms (cm from the chest front, height above
the floor). Drawing never changes what the robot does: every publish is wrapped so a drawing error is only logged.
"""
import asyncio
import logging
import math
import threading
import time

import cv2
import numpy as np

from common import config

log = logging.getLogger("robot_view")

# BGR
GREEN, YELLOW, BLUE, CYAN = (60, 200, 60), (0, 215, 255), (255, 140, 40), (255, 255, 0)
RED, MAGENTA, ORANGE, GREY, WHITE = (0, 0, 255), (255, 0, 255), (0, 140, 255), (150, 150, 150), (255, 255, 255)
MASK_COLOURS = [(90, 200, 90), (220, 160, 60), (80, 140, 230), (200, 90, 200), (60, 200, 200), (230, 220, 80)]
CHEST_X = 0.084            # chest front in the torso frame (torso mesh, the front of the chest)
TORSO_ABOVE_FLOOR = 0.80   # STANDARDS 0: torso origin above the floor


class RobotView:
    def __init__(self, geometry):
        self.geo = geometry
        self.cfg = config.load()
        self.lock = threading.Lock()
        self.snap = None           # what the robot looked at last
        self.jpg, self.seq = None, 0

    # ---------------------------------------------------------------- publishing (called from the skills)
    def look(self, view, T, title):
        """Start a new picture: `view` has rgb, depth, depth_scale, intrinsics (a camera Frame or a scene View);
        T = torso in pelvis when it was taken. Layers are added with the add_* calls, each re-renders."""
        if view is None:
            return
        with self.lock:
            self.snap = {"view": view, "T": np.array(T, float), "R": self.geo.R.copy(), "t": self.geo.t.copy(),
                         "title": title, "time": time.strftime("%H:%M:%S"), "layers": [], "lines": []}
        self._render()

    def note(self, text):
        """A caption line (the result, or why it failed)."""
        self._add(None, text)

    def dets(self, rects, chosen=None, label=""):
        """Detector boxes [(x, y, w, h), score]; `chosen` is the rect that was used."""
        self._add(("dets", list(rects), chosen, label), None)

    def masks(self, instances):
        """Task scene instances of this view: {noun, score, mask, ok, why}."""
        self._add(("masks", list(instances)), None)

    def box(self, b, label, chosen=True):
        """A fitted box {center, dims, yaw, top_z, support_z} (pelvis frame): its points, its 3D outline and a caption."""
        line = None
        if chosen:
            line = (f"{label}: centre {self._from_chest(b['center'][0]):.1f} cm ahead of the chest, "
                    f"{abs(100 * b['center'][1]):.1f} cm {'left' if b['center'][1] >= 0 else 'right'} | "
                    f"{' x '.join(f'{100 * d:.1f}' for d in b['dims'])} cm | top {self._above_floor(b['top_z']):.1f} cm above the floor")
        self._add(("box", dict(b), label, chosen), line)

    def surface(self, s, label):
        """A surface {top_z, front_x, y_range}: its top points and its front edge."""
        self._add(("surface", dict(s), label),
                  f"{label}: front edge {self._from_chest(s['front_x']):.1f} cm from the chest, "
                  f"top {self._above_floor(s['top_z']):.1f} cm above the floor"
                  + (f", {100 * (s['y_range'][1] - s['y_range'][0]):.0f} cm wide in view" if s.get("y_range") else "")
                  + (" (front face only: its top is out of view)" if s.get("partial") else ""))

    def grasp(self, g):
        """The grasp of _grasp_geometry: grip centre and the two palm contacts on the squeeze faces."""
        ux, uy = -math.sin(g["byaw"]), math.cos(g["byaw"])
        pts = {"grip": [g["cx"], g["cy"], g["zg"]],
               "L": [g["cx"] + ux * g["half"], g["cy"] + uy * g["half"], g["zg"]],
               "R": [g["cx"] - ux * g["half"], g["cy"] - uy * g["half"], g["zg"]]}
        with self.lock:                       # only the latest grasp (planning tries several, the margin check re-plans)
            if self.snap is not None:
                self.snap["layers"] = [l for l in self.snap["layers"] if l[0] != "grasp"]
                self.snap["lines"] = [l for l in self.snap["lines"] if not l.startswith("grasp '")]
        self._add(("grasp", pts),
                  f"grasp '{g['object']}': palms {100 * g['width']:.1f} cm apart, "
                  f"{100 * (g['zg'] - g['support_z']):.1f} cm above the surface it stands on")

    # ---------------------------------------------------------------- stream
    async def mjpeg(self):
        """The current picture, re-sent every second (a browser shows an MJPEG frame only when more follow)."""
        last, t_sent = None, 0.0
        while True:
            jpg, seq = self.jpg, self.seq
            if jpg is None:
                jpg = await asyncio.to_thread(self._placeholder)
            if seq != last or time.monotonic() - t_sent > 1.0:
                last, t_sent = seq, time.monotonic()
                yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpg + b"\r\n"
            await asyncio.sleep(0.2)

    # ---------------------------------------------------------------- internals
    def _add(self, layer, line):
        with self.lock:
            if self.snap is None:
                return
            if layer:
                self.snap["layers"].append(layer)
            if line:
                self.snap["lines"].append(line)
        self._render()

    def _from_chest(self, x):
        T = self.snap["T"]
        return 100 * (x - (T[0, 0] * CHEST_X + T[0, 3]))

    def _above_floor(self, z):
        return 100 * (z - (self.snap["T"][2, 3] - TORSO_ABOVE_FLOOR))

    def _project(self, s, P):
        """Pelvis-frame points (N,3) -> pixel (N,2) and a valid flag (in front of the camera)."""
        P = np.atleast_2d(np.asarray(P, float))
        Ti = np.linalg.inv(s["T"])
        torso = P @ Ti[:3, :3].T + Ti[:3, 3]
        cam = (torso - s["t"]) @ s["R"]                     # optical frame (points() does cam @ R.T + t)
        k = s["view"].intrinsics
        ok = cam[:, 2] > 0.05
        z = np.where(ok, cam[:, 2], 1.0)
        return np.stack([k["fx"] * cam[:, 0] / z + k["ppx"], k["fy"] * cam[:, 1] / z + k["ppy"]], axis=1), ok

    def _points(self, s, rect):
        """Depth pixels inside rect -> (us, vs, pelvis-frame points)."""
        v = s["view"]
        h, w = v.depth.shape
        x0, y0 = max(0, int(rect[0])), max(0, int(rect[1]))
        x1, y1 = min(w, int(rect[0] + rect[2])), min(h, int(rect[1] + rect[3]))
        if x1 <= x0 or y1 <= y0:
            return np.zeros(0, int), np.zeros(0, int), np.zeros((0, 3))
        vs, us = np.mgrid[y0:y1, x0:x1]
        z = v.depth[vs, us].astype(np.float32) * v.depth_scale
        keep = z > 0.05
        us, vs, z = us[keep], vs[keep], z[keep]
        k = v.intrinsics
        cam = np.stack([(us - k["ppx"]) * z / k["fx"], (vs - k["ppy"]) * z / k["fy"], z], axis=1)
        p = cam @ s["R"].T + s["t"]
        return us, vs, p @ s["T"][:3, :3].T + s["T"][:3, 3]

    @staticmethod
    def _tint(img, us, vs, colour, a=0.55):
        if len(us):
            img[vs, us] = ((1 - a) * img[vs, us] + a * np.array(colour)).astype(np.uint8)

    def _corners(self, b):
        (cx, cy, _), (dx, dy, _), yaw = b["center"], b["dims"], b.get("yaw", 0.0)
        c, sn = math.cos(yaw), math.sin(yaw)
        out = []
        for z in (b["support_z"], b["top_z"]):
            for sx, sy in ((1, 1), (1, -1), (-1, -1), (-1, 1)):
                lx, ly = sx * dx / 2, sy * dy / 2
                out.append([cx + c * lx - sn * ly, cy + sn * lx + c * ly, z])
        return np.array(out)

    def _draw_box(self, img, s, b, label, chosen):
        uv, ok = self._project(s, self._corners(b))
        if not ok.all():
            return
        if chosen:                                          # the points the fit stands on, classified like fit_box
            x0, y0 = uv.min(axis=0) - 25
            x1, y1 = uv.max(axis=0) + 25
            us, vs, p = self._points(s, (x0, y0, x1 - x0, y1 - y0))
            (cx, cy, _), (dx, dy, _), yaw = b["center"], b["dims"], b.get("yaw", 0.0)
            d = p[:, :2] - np.array([cx, cy])
            lx = d[:, 0] * math.cos(yaw) + d[:, 1] * math.sin(yaw)
            ly = -d[:, 0] * math.sin(yaw) + d[:, 1] * math.cos(yaw)
            inside = (np.abs(lx) <= dx / 2 + 0.01) & (np.abs(ly) <= dy / 2 + 0.01)
            z = p[:, 2]
            top = inside & (np.abs(z - b["top_z"]) < 0.02)
            side = inside & (z > b["support_z"] + 0.01) & (z < b["top_z"] - 0.02)
            sup = ~inside & (np.abs(z - b["support_z"]) < 0.012)
            self._tint(img, us[sup], vs[sup], BLUE, 0.35)
            self._tint(img, us[side], vs[side], YELLOW)
            self._tint(img, us[top], vs[top], GREEN)
        pts = uv.astype(int)
        colour, width = (CYAN, 2) if chosen else (GREY, 1)
        for i in range(4):
            for a, c in ((i, (i + 1) % 4), (i + 4, (i + 1) % 4 + 4), (i, i + 4)):
                cv2.line(img, tuple(pts[a]), tuple(pts[c]), colour, width, cv2.LINE_AA)
        cu, ok = self._project(s, [[b["center"][0], b["center"][1], b["top_z"]]])
        if ok[0]:
            u, v = cu[0].astype(int)
            cv2.circle(img, (u, v), 5, RED, -1)
            self._text(img, label, (u + 8, v - 6), RED if chosen else GREY)

    def _draw_surface(self, img, s, f, label):
        y0, y1 = f.get("y_range") or (-0.25, 0.25)
        x0 = f["front_x"]
        x1 = f.get("back_x", x0 + 0.4)
        corner = [[x0, y0, f["top_z"]], [x0, y1, f["top_z"]], [x1, y1, f["top_z"]], [x1, y0, f["top_z"]]]
        uv, ok = self._project(s, corner)
        if ok.all():
            lo, hi = uv.min(axis=0) - 5, uv.max(axis=0) + 5
            us, vs, p = self._points(s, (lo[0], lo[1], hi[0] - lo[0], hi[1] - lo[1]))
            on = (np.abs(p[:, 2] - f["top_z"]) < 0.02) & (p[:, 0] >= x0 - 0.01) & (p[:, 1] >= y0 - 0.01) & (p[:, 1] <= y1 + 0.01)
            self._tint(img, us[on], vs[on], BLUE, 0.4)
            a, b = uv[0].astype(int), uv[1].astype(int)
            cv2.line(img, tuple(a), tuple(b), ORANGE, 3, cv2.LINE_AA)
            self._text(img, f"{label} front edge", (int(min(a[0], b[0])) + 6, int(max(a[1], b[1])) - 8), ORANGE)

    def _draw_grasp(self, img, s, pts):
        uv, ok = self._project(s, [pts["L"], pts["grip"], pts["R"]])
        if not ok.all():
            return
        p = uv.astype(int)
        cv2.line(img, tuple(p[0]), tuple(p[2]), MAGENTA, 2, cv2.LINE_AA)
        for i, name in ((0, "L palm"), (2, "R palm")):
            cv2.circle(img, tuple(p[i]), 6, MAGENTA, -1)
            self._text(img, name, (p[i][0] - 20, p[i][1] + 20), MAGENTA)
        cv2.drawMarker(img, tuple(p[1]), RED, cv2.MARKER_CROSS, 16, 2)

    def _draw_dets(self, img, rects, chosen, label):
        for r, score in rects:
            x, y, w, h = (int(v) for v in r)
            sel = chosen is not None and tuple(int(v) for v in chosen) == (x, y, w, h)
            cv2.rectangle(img, (x, y), (x + w, y + h), WHITE if sel else GREY, 2 if sel else 1)
            self._text(img, f"{label} {score:.2f}", (x + 3, y + h - 6), WHITE if sel else GREY, 0.45)

    def _draw_masks(self, img, instances):
        for j, i in enumerate(instances):
            m = i.get("mask")
            if m is None or not m.any():
                continue
            colour = MASK_COLOURS[j % len(MASK_COLOURS)] if i.get("ok") else GREY
            img[m] = (0.55 * img[m] + 0.45 * np.array(colour)).astype(np.uint8)
            cs, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(img, cs, -1, colour, 1)
            ys, xs = np.nonzero(m)
            txt = f"{i.get('noun', '')} {i.get('score', 0):.2f}" + ("" if i.get("ok") else f" ({i.get('why', 'unused')})")
            self._text(img, txt, (int(xs.min()) + 3, int(ys.min()) + 14), colour, 0.42)

    @staticmethod
    def _text(img, txt, org, colour, scale=0.5):
        x = int(np.clip(org[0], 2, img.shape[1] - 8 * len(txt) * scale - 2))
        y = int(np.clip(org[1], 12, img.shape[0] - 4))
        cv2.putText(img, txt, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(img, txt, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, colour, 1, cv2.LINE_AA)

    def _caption(self, img, s):
        lines = []
        for k, t in enumerate([f"{s['time']}  {s['title']}"] + s["lines"][-4:]):   # wrap long lines at ~92 characters
            while len(t) > 92:
                cut = t.rfind(" ", 0, 92)
                cut = cut if cut > 40 else 92
                lines.append((k, t[:cut]))
                t = "   " + t[cut:].lstrip()
            lines.append((k, t))
        lines = lines[:8]
        hgt = 8 + 17 * len(lines)
        band = img[:hgt].astype(np.float32)
        img[:hgt] = (0.35 * band).astype(np.uint8)
        for j, (k, t) in enumerate(lines):
            cv2.putText(img, t, (6, 17 + 17 * j), cv2.FONT_HERSHEY_SIMPLEX, 0.45, WHITE if k == 0 else (200, 255, 200), 1, cv2.LINE_AA)

    def _render(self):
        try:
            with self.lock:
                s = self.snap
                if s is None:
                    return
                layers, img = list(s["layers"]), s["view"].rgb.copy()
            for layer in layers:              # masks first, points/boxes on top
                if layer[0] == "masks":
                    self._draw_masks(img, layer[1])
            for layer in layers:
                kind = layer[0]
                if kind == "dets":
                    self._draw_dets(img, layer[1], layer[2], layer[3])
                elif kind == "surface":
                    self._draw_surface(img, s, layer[1], layer[2])
            for layer in layers:
                if layer[0] == "box":
                    self._draw_box(img, s, layer[1], layer[2], layer[3])
            for layer in layers:
                if layer[0] == "grasp":
                    self._draw_grasp(img, s, layer[1])
            self._caption(img, s)
            ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, self.cfg["image_client"]["stream_quality"]])
            if ok:
                self.jpg, self.seq = buf.tobytes(), self.seq + 1
        except Exception:                     # drawing must never stop the robot
            log.exception("robot view: drawing failed")

    def _placeholder(self):
        img = np.full((480, 640, 3), 40, np.uint8)
        cv2.putText(img, "Robot's view: nothing looked at yet", (120, 230), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 1, cv2.LINE_AA)
        cv2.putText(img, "it appears when the robot looks (find, task scene, pick)", (90, 262), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (160, 160, 160), 1, cv2.LINE_AA)
        return cv2.imencode(".jpg", img)[1].tobytes()


def publish(ctx, fn, *a, **kw):
    """Call a RobotView method if this context has one (the UI server does; scripts and replays do not).
    Never raises: a drawing problem must not change what the robot does."""
    view = getattr(ctx, "view", None)
    if view is None:
        return
    try:
        getattr(view, fn)(*a, **kw)
    except Exception:
        log.exception("robot view: %s failed", fn)
