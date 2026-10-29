"""Open-vocabulary vision service: OWLv2 boxes (the UI's detect & track) and SAM 3 masks (manipulation).

POST /infer  {"id", "action": "detect", "params": {"labels": ["black box", ...], "image_b64": "<jpeg>",
                                                   "threshold": 0.15 (optional)}}
  -> result {"detections": [{"label", "score", "box": [x1, y1, x2, y2]}], "size": [w, h]}
POST /infer  {"id", "action": "segment", "params": {"texts": ["box", ...], "image_b64": "<png/jpeg>"}}
  -> result {"instances": [{"text", "score", "box": [x1, y1, x2, y2], "mask_png": "<b64 png, 255 = object>"}],
             "size": [w, h]}
SAM 3 returns every WHOLE instance of each text (not one box per detection), so an object's mask covers all of
its visible faces. Boxes are in pixels of the input image. Run: scripts/svc.sh start vision
"""
import base64
import logging
import threading
import time

import cv2
import numpy as np
import torch
import uvicorn
from fastapi import FastAPI
from transformers import Owlv2ForObjectDetection, Owlv2Processor, Sam3Model, Sam3Processor

from common import config, envelope
from common.envelope import Request

CFG = config.load()
V = CFG["services"]["vision"]
log = logging.getLogger("vision")
app = FastAPI(title="g1ui-vision")
_model = _processor = None
_sam = _sam_proc = None
_lock = threading.Lock()   # one inference at a time on the GPU


def _load():
    global _model, _processor
    _processor = Owlv2Processor.from_pretrained(V["model"])
    _model = Owlv2ForObjectDetection.from_pretrained(V["model"], torch_dtype=getattr(torch, V["dtype"]))
    _model = _model.to(V["device"]).eval()
    global _sam, _sam_proc
    _sam_proc = Sam3Processor.from_pretrained(V["sam3_model"])
    _sam = Sam3Model.from_pretrained(V["sam3_model"], torch_dtype=getattr(torch, V["sam3_dtype"])).to(V["device"]).eval()


@torch.inference_mode()
def detect(labels, bgr, threshold):
    h, w = bgr.shape[:2]
    queries = [V["prompt"].format(label) for label in labels]
    inputs = _processor(text=[queries], images=bgr[..., ::-1].copy(), return_tensors="pt").to(V["device"])
    if "pixel_values" in inputs:
        inputs["pixel_values"] = inputs["pixel_values"].to(getattr(torch, V["dtype"]))
    out = _model(**inputs)
    # OWLv2 pads the image to a square at the bottom/right, so scale boxes to that square.
    side = max(h, w)
    res = _processor.post_process_grounded_object_detection(
        out, threshold=threshold, target_sizes=torch.tensor([[side, side]], device=V["device"]))[0]
    dets = []
    for score, idx, box in zip(res["scores"].tolist(), res["labels"].tolist(), res["boxes"].tolist()):
        x1, y1, x2, y2 = box
        dets.append({"label": labels[idx], "score": round(score, 3),
                     "box": [max(0.0, x1), max(0.0, y1), min(float(w), x2), min(float(h), y2)]})
    return sorted(dets, key=lambda d: -d["score"])


@torch.inference_mode()
def segment(texts, bgr):
    """Every instance of each text: score, box and full-resolution mask (PNG, 255 = object)."""
    from PIL import Image
    img = Image.fromarray(bgr[..., ::-1].copy())
    out = []
    for text in texts:
        inputs = _sam_proc(images=img, text=text, return_tensors="pt").to(V["device"])
        inputs["pixel_values"] = inputs["pixel_values"].to(getattr(torch, V["sam3_dtype"]))
        res = _sam_proc.post_process_instance_segmentation(
            _sam(**inputs), threshold=V["sam3_threshold"], mask_threshold=V["sam3_mask_threshold"],
            target_sizes=inputs.get("original_sizes").tolist())[0]
        for m, sc, b in zip(res["masks"], res["scores"].tolist(), res["boxes"].tolist()):
            ok, png = cv2.imencode(".png", m.cpu().numpy().astype(np.uint8) * 255)
            out.append({"text": text, "score": round(float(sc), 3), "box": [round(float(v), 1) for v in b],
                        "mask_png": base64.b64encode(png.tobytes()).decode()})
    return out


@app.post("/infer")
def infer(req: Request):
    t0 = time.time()
    if req.action not in ("detect", "segment"):
        return envelope.failed(req.id, f"unknown action '{req.action}'", t0, status="rejected")
    if req.action == "segment":
        texts = [str(x).strip() for x in req.params.get("texts", []) if str(x).strip()]
        img = cv2.imdecode(np.frombuffer(base64.b64decode(req.params.get("image_b64", "")), np.uint8), cv2.IMREAD_COLOR)
        if not texts or img is None:
            return envelope.failed(req.id, "'texts' is empty or 'image_b64' unreadable", t0, status="rejected")
        with _lock:
            inst = segment(texts, img)
        return envelope.done(req.id, {"instances": inst, "size": [img.shape[1], img.shape[0]]}, t0)
    labels = [str(x).strip() for x in req.params.get("labels", []) if str(x).strip()]
    if not labels:
        return envelope.failed(req.id, "'labels' is empty", t0, status="rejected")
    img = cv2.imdecode(np.frombuffer(base64.b64decode(req.params.get("image_b64", "")), np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        return envelope.failed(req.id, "could not decode 'image_b64'", t0, status="rejected")
    with _lock:
        dets = detect(labels, img, float(req.params.get("threshold", V["threshold"])))
    return envelope.done(req.id, {"detections": dets, "size": [img.shape[1], img.shape[0]]}, t0)


@app.get("/health")
def health():
    return {"service": "vision", "model": V["model"], "sam3": V["sam3_model"], "ready": _sam is not None}


def main():
    logging.basicConfig(level=CFG["logging"]["level"], format="%(asctime)s %(name)s %(levelname)s %(message)s")
    _load()
    dummy = np.zeros((480, 640, 3), np.uint8)
    detect(["box"], dummy, 0.5)            # warm-up
    segment(["box"], dummy)
    log.info("OWLv2 + SAM 3 ready on :%s", V["port"])
    uvicorn.run(app, host=V["host"], port=V["port"], log_level="warning")


if __name__ == "__main__":
    main()
