"""Server: serves the UI and relays its requests to the robot Agent.

Run: scripts/run_server.sh
"""
import base64
import subprocess
import sys
from pathlib import Path

import httpx
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from common import config, joints

from .image_client import ImageClient
from .robot_link import AgentLink

CFG = config.load()
STATIC = Path(__file__).parent / "static"
FFMPEG = str(Path(sys.executable).parent / "ffmpeg")

app = FastAPI(title="g1ui-server")
app.mount("/static", StaticFiles(directory=STATIC), name="static")
link = AgentLink()
camera = ImageClient()


class Cmd(BaseModel):
    action: str
    params: dict = {}
    wait: bool = False


class Host(BaseModel):
    host: str


@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")


@app.get("/api/ui-config")
def ui_config():
    return {
        "modes": CFG["modes"],
        "locomotion": CFG["locomotion"],
        "audio": {"tts_max_chars": CFG["audio"]["tts_max_chars"], "file_max_mb": CFG["audio"]["file_max_mb"]},
        "ui": CFG["ui"],
        "poll_ms": CFG["server"]["state_poll_ms"],
        "joint_groups": joints.GROUPS,
        "end_effector": CFG["end_effector"],
    }


# ---------------------------------------------------------------- connection
@app.post("/api/discover")
async def discover():
    return {"found": await link.discover()}


@app.post("/api/connect")
def connect(h: Host):
    try:
        health = link.connect(h.host)
        camera.start(h.host)
        return {"connected": True, "host": h.host, "health": health}
    except httpx.HTTPError as e:
        raise HTTPException(502, f"no agent at {h.host}: {e}")


@app.post("/api/disconnect")
def disconnect():
    link.disconnect()
    camera.stop()
    return {"connected": False}


@app.get("/api/status")
def status():
    if not link.host:
        return {"connected": False}
    try:
        return {"connected": True, "host": link.host, "state": link.state(), "camera": camera.status()}
    except (httpx.HTTPError, ValueError) as e:
        return {"connected": True, "host": link.host, "state": None, "camera": camera.status(), "error": str(e)}


# ---------------------------------------------------------------- camera
@app.get("/video/{kind}.mjpg")
def video(kind: str):
    if kind not in ("rgb", "depth"):
        raise HTTPException(404)
    return StreamingResponse(camera.mjpeg(kind), media_type="multipart/x-mixed-replace; boundary=frame")


# ---------------------------------------------------------------- commands
@app.post("/api/cmd")
def cmd(c: Cmd):
    return link.send(c.action, c.params, wait=c.wait)


@app.post("/api/audio")
async def audio(file: UploadFile = File(...)):
    max_bytes = CFG["audio"]["file_max_mb"] * 1024 * 1024
    data = await file.read(max_bytes + 1)
    if len(data) > max_bytes:
        raise HTTPException(413, f"audio file is over {CFG['audio']['file_max_mb']} MB")
    # any audio format -> 16 kHz mono int16 PCM, which the robot speaker expects
    p = subprocess.run([FFMPEG, "-hide_banner", "-loglevel", "error", "-i", "pipe:0", "-f", "s16le",
                        "-acodec", "pcm_s16le", "-ac", "1", "-ar", str(CFG["audio"]["sample_rate"]), "pipe:1"],
                       input=data, capture_output=True)
    if p.returncode != 0 or not p.stdout:
        raise HTTPException(400, f"could not decode audio: {p.stderr.decode()[:300]}")
    return link.send("play_audio", {"pcm_b64": base64.b64encode(p.stdout).decode()})
