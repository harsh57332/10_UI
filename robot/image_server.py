"""Robot-side image server: D435i RGB + depth aligned to RGB -> ZMQ PUB.

Message (multipart): [meta json, rgb jpeg, depth png (uint16, lossless)]
meta: seq, t (capture time), width, height, depth_scale (m per unit), intrinsics (of the RGB camera)

Run: python -m robot.image_server   (systemd service, see scripts/robot.sh)
"""
import json
import logging
import time

import cv2
import numpy as np
import pyrealsense2 as rs
import zmq

from common import config

log = logging.getLogger("camera")


def open_camera(c):
    pipe, rc = rs.pipeline(), rs.config()
    rc.enable_stream(rs.stream.color, c["width"], c["height"], rs.format.bgr8, c["capture_fps"])
    rc.enable_stream(rs.stream.depth, c["width"], c["height"], rs.format.z16, c["capture_fps"])
    profile = pipe.start(rc)
    i = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
    intr = {"fx": i.fx, "fy": i.fy, "ppx": i.ppx, "ppy": i.ppy, "model": str(i.model), "coeffs": list(i.coeffs)}
    scale = profile.get_device().first_depth_sensor().get_depth_scale()
    return pipe, rs.align(rs.stream.color), intr, scale


def hardware_reset():
    for d in rs.context().query_devices():
        log.warning("hardware reset of %s", d.get_info(rs.camera_info.serial_number))
        d.hardware_reset()


def set_preset(profile, value):
    sensor = profile.get_device().first_depth_sensor()
    if sensor.supports(rs.option.visual_preset):
        sensor.set_option(rs.option.visual_preset, float(value))


def main():
    cfg = config.load()
    c = cfg["camera"]
    ctl = zmq.Context.instance().socket(zmq.REP)
    ctl.bind(f"tcp://*:{c['control_port']}")
    mode = "coarse"
    logging.basicConfig(level=cfg["logging"]["level"], format="%(asctime)s %(name)s %(levelname)s %(message)s")
    sock = zmq.Context().socket(zmq.PUB)
    sock.setsockopt(zmq.SNDHWM, 2)
    sock.bind(f"tcp://*:{c['zmq_port']}")
    period, half_frame = 1.0 / c["fps"], 0.5 / c["capture_fps"]
    jpg_args = [cv2.IMWRITE_JPEG_QUALITY, c["jpeg_quality"]]
    png_args = [cv2.IMWRITE_PNG_COMPRESSION, c["depth_png_level"]]
    seq = 0

    while True:
        pipe = None
        try:
            pipe, align, intr, scale = open_camera(c)
            profile = pipe.get_active_profile()
            set_preset(profile, c["presets"][mode])
            log.info("streaming %dx%d @ %d fps on :%d", c["width"], c["height"], c["fps"], c["zmq_port"])
            next_t = 0.0
            while True:
                frames = pipe.wait_for_frames(c["frame_timeout_ms"])
                try:                                 # precision switch from the server (non-blocking)
                    req = ctl.recv_json(flags=zmq.NOBLOCK)
                    if req.get("preset") in c["presets"]:
                        mode = req["preset"]
                        set_preset(profile, c["presets"][mode])
                    ctl.send_json({"preset": mode})
                except zmq.Again:
                    pass
                now = time.monotonic()               # scheduling only; meta "t" stays wall time
                if now + half_frame < next_t:        # decimate capture_fps -> fps
                    continue
                next_t = max(next_t + period, now - period)
                frames = align.process(frames)
                color, depth = frames.get_color_frame(), frames.get_depth_frame()
                if not color or not depth:
                    continue
                ok1, jpg = cv2.imencode(".jpg", np.asanyarray(color.get_data()), jpg_args)
                ok2, png = cv2.imencode(".png", np.asanyarray(depth.get_data()), png_args)
                if not (ok1 and ok2):
                    continue
                seq += 1
                meta = {"seq": seq, "t": time.time(), "width": c["width"], "height": c["height"], "preset": mode,
                        "depth_scale": scale, "intrinsics": intr}
                sock.send_multipart([json.dumps(meta).encode(), jpg.tobytes(), png.tobytes()])
        except Exception as e:
            log.error("camera error: %s", e)
            try:
                if pipe:
                    pipe.stop()
            except Exception:
                pass
            try:
                hardware_reset()
            except Exception as e2:
                log.error("hardware reset failed: %s", e2)
            time.sleep(c["restart_gap_s"])


if __name__ == "__main__":
    main()
