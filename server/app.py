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

from .ai import AIControl, MotionPlanning
from .geometry import Geometry
from .image_client import ImageClient
from .llm import LLM
from .motion import ArmMotion, SkillError
from .robot_link import AgentLink
from .services import Service, ServiceError
from .skills import Ctx
from .tracking import Tracking

CFG = config.load()
STATIC = Path(__file__).parent / "static"
FFMPEG = str(Path(sys.executable).parent / "ffmpeg")

app = FastAPI(title="g1ui-server")
app.mount("/static", StaticFiles(directory=STATIC), name="static")
link = AgentLink()
camera = ImageClient()
vision = Service("vision")
tracking = Tracking(camera, link, vision)
llm = LLM()
motion = ArmMotion(link, Service("planner"), tracking.processing)
ctx = Ctx(link, camera, vision, Geometry(), motion, tracking.processing)
plan6 = MotionPlanning(ctx, llm)
ai = AIControl(ctx, llm)


class Cmd(BaseModel):
    action: str
    params: dict = {}
    wait: bool = False


class Host(BaseModel):
    host: str


class Text(BaseModel):
    text: str = ""


class Task(BaseModel):
    text: str
    mode: str = "step"


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
        "arms": {"max_speed_dps": CFG["arms"]["max_speed_dps"], "groups": {
            "waist": joints.GROUPS["waist"], "left_arm": joints.GROUPS["left_arm"], "right_arm": joints.GROUPS["right_arm"]}},
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
        return {"connected": True, "host": link.host, "state": link.state(), "camera": camera.status(),
                "tracking": tracking.status(), "plan6": plan6.status(), "ai": ai.status()}
    except (httpx.HTTPError, ValueError) as e:
        return {"connected": True, "host": link.host, "state": None, "camera": camera.status(),
                "tracking": tracking.status(), "error": str(e)}


# ---------------------------------------------------------------- camera
@app.get("/video/{kind}.mjpg")
def video(kind: str):
    if kind not in ("rgb", "depth"):
        raise HTTPException(404)
    return StreamingResponse(camera.mjpeg(kind), media_type="multipart/x-mixed-replace; boundary=frame")


@app.post("/api/detect")
def detect(t: Text):
    """Comma-separated object names (max ui.detect_text_max chars). Empty text clears tracking."""
    if len(t.text) > CFG["ui"]["detect_text_max"]:
        raise HTTPException(400, f"text is over {CFG['ui']['detect_text_max']} characters")
    try:
        return tracking.detect(t.text)
    except (ServiceError, RuntimeError) as e:
        raise HTTPException(503, str(e))


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


# ---------------------------------------------------------------- section 6: motion planning
def _nl(text):
    if not text.strip() or len(text) > CFG["ui"]["nl_text_max"]:
        raise HTTPException(400, f"text must be 1..{CFG['ui']['nl_text_max']} characters")


@app.post("/api/plan6/generate")
def plan6_generate(t: Text):
    _nl(t.text)
    try:
        return plan6.generate(t.text)
    except (SkillError, ServiceError) as e:
        raise HTTPException(409, str(e))


@app.post("/api/plan6/play")
def plan6_play():
    try:
        return plan6.play()
    except SkillError as e:
        raise HTTPException(409, str(e))


# ---------------------------------------------------------------- section 7: AI control
@app.post("/api/ai/start")
def ai_start(t: Task):
    _nl(t.text)
    if t.mode not in ("step", "auto"):
        raise HTTPException(400, "mode must be step or auto")
    try:
        return ai.start(t.text, t.mode)
    except SkillError as e:
        raise HTTPException(409, str(e))


@app.post("/api/ai/next")
def ai_next():
    ai.next()
    return ai.status()


@app.post("/api/ai/abort")
def ai_abort():
    ai.abort()
    return ai.status()


@app.post("/api/ai/mode")
def ai_mode(t: Task):
    ai.set_mode(t.mode if t.mode in ("step", "auto") else "step")
    return ai.status()


@app.get("/api/services")
def services_health():
    return {"vision": vision.health(), "planner": motion.planner.health(), "llm": llm.health() is not None}


class SkillCall(BaseModel):
    skill: str
    params: dict = {}


@app.post("/api/skill")
def skill_call(c: SkillCall):
    """Run one skill now (testing / manual use). Shares the busy lock with sections 6 and 7."""
    from .ai import Busy
    from .skills import run_skill
    if not Busy.lock.acquire(blocking=False):
        raise HTTPException(409, f"robot is busy ({Busy.owner})")
    Busy.owner = f"skill {c.skill}"
    try:
        return {"ok": True, "result": run_skill(ctx, c.skill, c.params), "known": ctx.summary()}
    except (SkillError, ServiceError, TypeError) as e:
        return {"ok": False, "error": str(e), "known": ctx.summary()}
    finally:
        Busy.owner = None
        Busy.lock.release()
