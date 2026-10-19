"""Detect & track: OWLv2 finds each requested object once, then OpenCV KCF follows it on every frame.
The distance (cm) is the median aligned depth inside the central part of the box."""
import base64
import threading
import time
from contextlib import contextmanager

import cv2
import numpy as np

from common import config


class Tracking:
    def __init__(self, camera, link, vision):
        self.cfg = config.load()
        self.camera, self.link, self.vision = camera, link, vision
        self.items = []
        self.lock = threading.Lock()
        camera.overlays.append(self.draw)
        threading.Thread(target=self._loop, daemon=True, name="tracking").start()

    # ---------------------------------------------------------------- api
    def detect(self, text):
        """Comma-separated labels. Empty text clears all tracking. One box per label (the best score)."""
        labels = [t.strip() for t in text.split(",") if t.strip()]
        if not labels:
            self.clear()
            return {"cleared": True, "items": []}
        frame = self.camera.latest()
        if frame is None:
            raise RuntimeError("camera is offline")
        ok, jpg = cv2.imencode(".jpg", frame.rgb, [cv2.IMWRITE_JPEG_QUALITY, 95])
        with self.processing("owl"):
            dets = self.vision.call("detect", {"labels": labels, "image_b64": base64.b64encode(jpg).decode()})["detections"]
        items = []
        for label in labels:
            best = next((d for d in dets if d["label"] == label), None)
            if best is None:
                items.append({"label": label, "found": False})
                continue
            x1, y1, x2, y2 = best["box"]
            rect = (int(x1), int(y1), max(2, int(x2 - x1)), max(2, int(y2 - y1)))
            tracker = cv2.TrackerKCF_create()
            tracker.init(frame.rgb, rect)
            it = {"label": label, "found": True, "score": best["score"], "rect": rect, "ok": True, "tracker": tracker}
            it.update(self.measure(frame, rect))
            items.append(it)
        with self.lock:
            self.items = items
        return {"cleared": False, "items": self.status()}

    def clear(self):
        with self.lock:
            self.items = []

    def status(self):
        with self.lock:
            return [{k: v for k, v in it.items() if k != "tracker"} for it in self.items]

    @contextmanager
    def processing(self, source):
        """Blink the robot head light while a model is working (ignored if no robot is connected)."""
        self.link.send("processing", {"source": source, "on": True})
        try:
            yield
        finally:
            self.link.send("processing", {"source": source, "on": False})

    # ---------------------------------------------------------------- internals
    def measure(self, frame, rect):
        """Distance (cm) and 3D point (m, camera optical frame) of the box centre from aligned depth."""
        x, y, w, h = rect
        f = self.cfg["tracking"]["depth_roi"]
        cx, cy = x + w / 2.0, y + h / 2.0
        x0, x1 = int(max(0, cx - w * f / 2)), int(min(frame.depth.shape[1], cx + w * f / 2))
        y0, y1 = int(max(0, cy - h * f / 2)), int(min(frame.depth.shape[0], cy + h * f / 2))
        roi = frame.depth[y0:y1, x0:x1]
        valid = roi[roi > 0]
        if valid.size < self.cfg["tracking"]["min_depth_px"]:
            return {"dist_cm": None, "xyz_m": None}
        z = float(np.median(valid)) * frame.depth_scale
        k = frame.intrinsics
        xyz = [(cx - k["ppx"]) * z / k["fx"], (cy - k["ppy"]) * z / k["fy"], z]
        return {"dist_cm": round(z * 100, 1), "xyz_m": [round(v, 3) for v in xyz]}

    def _loop(self):
        last_seq = None
        while True:
            frame = self.camera.latest()
            with self.lock:
                active = [it for it in self.items if it.get("found") and it.get("ok")]
            if frame is None or frame.seq == last_seq or not active:
                time.sleep(0.01)
                continue
            last_seq = frame.seq
            for it in active:
                ok, rect = it["tracker"].update(frame.rgb)
                with self.lock:
                    it["ok"] = bool(ok)
                    if ok:
                        it["rect"] = tuple(int(v) for v in rect)
                        it.update(self.measure(frame, it["rect"]))

    def draw(self, img):
        with self.lock:
            items = [dict(it) for it in self.items if it.get("found")]
        for it in items:
            x, y, w, h = it["rect"]
            color = (60, 220, 60) if it["ok"] else (0, 0, 255)
            cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
            dist = f"{it['dist_cm']:.0f} cm" if it.get("dist_cm") else "-- cm"
            text = f"{it['label']} {dist}" if it["ok"] else f"{it['label']} lost"
            cv2.putText(img, text, (x + 2, max(14, y - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(img, text, (x + 2, max(14, y - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1, cv2.LINE_AA)
