"""Server-side image client: subscribes to the robot image server, keeps the newest RGB + raw depth,
and serves them to the browser as MJPEG (RGB, and an 8-bit colour view of depth).

Other modules (OWLv2, cuRobo) read frames with latest(). Overlays (detections, tracking) are drawn
by callables registered in `overlays`, only on the browser stream.
"""
import asyncio
import json
import threading
import time

import cv2
import numpy as np
import zmq

from common import config


class Frame:
    __slots__ = ("seq", "t", "rgb", "depth", "depth_scale", "intrinsics")

    def __init__(self, meta, rgb, depth):
        self.seq, self.t = meta["seq"], meta["t"]
        self.depth_scale, self.intrinsics = meta["depth_scale"], meta["intrinsics"]
        self.rgb, self.depth = rgb, depth   # BGR uint8 (H,W,3); uint16 (H,W) aligned to RGB

    def depth_m(self):
        return self.depth.astype(np.float32) * self.depth_scale


class ImageClient:
    def __init__(self):
        self.cfg = config.load()
        self.host = None
        self.frame = None
        self.recv_t = 0.0
        self.fps = 0.0
        self.overlays = []          # callables(bgr_image) -> None, drawn on the RGB stream
        self._stop = threading.Event()
        self._thread = None

    def start(self, host):
        if host == self.host and self._thread and self._thread.is_alive():
            return
        self.stop()
        self.host, self._stop = host, threading.Event()
        self._thread = threading.Thread(target=self._run, args=(host, self._stop), daemon=True, name="image_client")
        self._thread.start()

    def stop(self):
        self._stop.set()
        self.host, self.frame = None, None

    def online(self):
        return self.frame is not None and time.time() - self.recv_t < self.cfg["image_client"]["stale_s"]

    def latest(self):
        return self.frame if self.online() else None

    def status(self):
        f = self.frame
        return {"online": self.online(), "fps": round(self.fps, 1), "seq": f.seq if f else None,
                "size": [f.rgb.shape[1], f.rgb.shape[0]] if f else None,
                "intrinsics": f.intrinsics if f else None}

    def _run(self, host, stop):
        sock = zmq.Context.instance().socket(zmq.SUB)
        sock.setsockopt(zmq.SUBSCRIBE, b"")
        sock.setsockopt(zmq.RCVHWM, 2)
        sock.setsockopt(zmq.LINGER, 0)
        sock.connect(f"tcp://{host}:{self.cfg['camera']['zmq_port']}")
        poller = zmq.Poller()
        poller.register(sock, zmq.POLLIN)
        last = None
        while not stop.is_set():
            if not poller.poll(200):
                continue
            meta, jpg, png = sock.recv_multipart()
            rgb = cv2.imdecode(np.frombuffer(jpg, np.uint8), cv2.IMREAD_COLOR)
            depth = cv2.imdecode(np.frombuffer(png, np.uint8), cv2.IMREAD_UNCHANGED)
            if rgb is None or depth is None:
                continue
            now = time.time()
            if last:
                self.fps = 0.9 * self.fps + 0.1 / max(now - last, 1e-3)
            last = now
            self.frame, self.recv_t = Frame(json.loads(meta), rgb, depth), now
        sock.close()

    # ---------------------------------------------------------------- browser streams
    def _jpeg(self, kind):
        f = self.frame
        if f is None:
            return None
        if kind == "rgb":
            img = f.rgb.copy()
            for draw in list(self.overlays):
                draw(img)
        else:
            max_m = self.cfg["image_client"]["depth_view_max_m"]
            d8 = np.clip(f.depth_m() / max_m * 255.0, 0, 255).astype(np.uint8)
            img = cv2.applyColorMap(d8, cv2.COLORMAP_TURBO)
            img[f.depth == 0] = 0
        ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, self.cfg["image_client"]["stream_quality"]])
        return buf.tobytes() if ok else None

    async def mjpeg(self, kind):
        period = 1.0 / self.cfg["camera"]["fps"]
        last_seq = None
        while True:
            f = self.frame
            if f is not None and f.seq != last_seq and self.online():
                last_seq = f.seq
                jpg = await asyncio.to_thread(self._jpeg, kind)
                if jpg:
                    yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpg + b"\r\n"
            await asyncio.sleep(period / 2)
