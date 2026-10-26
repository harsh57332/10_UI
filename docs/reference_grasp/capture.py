"""Capture the head camera (fine depth: median of 15 frames + RGB) and the joint state into this folder.
Run: ~/miniconda3/envs/g1ui_server/bin/python docs/reference_grasp/capture.py [tag]"""
import json
import sys
import time
from pathlib import Path

import cv2
import httpx
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from common import config                      # noqa: E402
from server.image_client import ImageClient    # noqa: E402

out = Path(__file__).resolve().parent
tag = sys.argv[1] if len(sys.argv) > 1 else "ref"
cfg = config.load()
cam = ImageClient()
cam.start(cfg["robot"]["host"])
t0 = time.time()
while cam.latest() is None and time.time() - t0 < 10:
    time.sleep(0.1)
f = cam.fine(15)
cv2.imwrite(str(out / f"{tag}_rgb.png"), f.rgb)
cv2.imwrite(str(out / f"{tag}_depth_u16.png"), f.depth)
d8 = np.clip(f.depth_m() / 1.5 * 255.0, 0, 255).astype(np.uint8)
cv2.imwrite(str(out / f"{tag}_depth_color.png"), cv2.applyColorMap(d8, cv2.COLORMAP_TURBO))
state = httpx.get(f"http://127.0.0.1:{cfg['server']['port']}/api/status", timeout=5).json()["state"]
json.dump({"seq": f.seq, "t": f.t, "depth_scale": f.depth_scale, "intrinsics": f.intrinsics,
           "joints_deg": {j["name"]: j["deg"] for j in state["joints"]}, "imu": state["imu"]},
          open(out / f"{tag}_meta.json", "w"), indent=1)
cam.stop()
print("saved", tag, "depth_scale", f.depth_scale, "valid depth px", int((f.depth > 0).sum()))
