"""Run one section-6 task under supervision: Generate, Play, then follow it step by step (with timestamps) and
collect the server's log lines from the run. The report is printed and saved to logs/runs/<time>.txt.
Run: ~/miniconda3/envs/g1ui_server/bin/python -m scripts.supervised_run [--gen] [--marginal-ok] "pick up the right box ..."
  --gen          Generate only (scan, scene, plan, margin), do not play
  --marginal-ok  play even if the margin check says MARGINAL (default: stop before playing)
"""
import json
import subprocess
import sys
import threading
import time
from pathlib import Path

import httpx

from common import config

cfg = config.load()
url = f"http://127.0.0.1:{cfg['server']['port']}"
flags = {a for a in sys.argv[1:] if a.startswith("--")}
text = " ".join(a for a in sys.argv[1:] if not a.startswith("--")) or \
    "pick up right side box from both the hands and stack it upon left side box"
log_path = config.path("logs/server.log")
out_dir = config.path("logs/runs")
out_dir.mkdir(parents=True, exist_ok=True)
lines = []


def say(msg):
    line = f"{time.strftime('%H:%M:%S')} {msg}"
    print(line, flush=True)
    lines.append(line)


log_start = log_path.stat().st_size if log_path.exists() else 0
rec, recording = [], threading.Event()


def record():
    """Arm + waist joints at ~10 Hz while the plan plays: measured, commanded, torque, temperature."""
    with httpx.Client(timeout=2) as c:
        while recording.is_set():
            try:
                s = c.get(f"{url}/api/status").json()
                st, a = s["state"], s["state"]["arms"]
                rec.append({"t": round(time.time(), 2), "step": s["plan6"]["current"], "weight": a.get("weight"),
                            "imu": st.get("imu"),
                            "cmd": a.get("q_cmd_deg"), "ff": (a.get("ff") or {}).get("tau"),
                            "j": {j["index"]: [j["deg"], j.get("tau"), j["temp"]] for j in st["joints"] if j["index"] >= 12}})
            except Exception:
                pass
            time.sleep(0.1)
temps = lambda s: max(j["temp"] for j in s["joints"] if 15 <= j["index"] <= 28)
st = httpx.get(f"{url}/api/status", timeout=5).json()["state"]
say(f"start: mode {st['mode']}, arm motors max {temps(st)} C, battery {(st.get('battery') or {}).get('soc')} %")
t0 = time.time()
r = httpx.post(f"{url}/api/plan6/generate", json={"text": text}, timeout=180)
if r.status_code != 200:
    say(f"GENERATE REFUSED after {time.time() - t0:.1f} s: {r.json().get('detail')}")
else:
    g = r.json()
    say(f"generated in {time.time() - t0:.1f} s: {[(n['skill'], n['params']) for n in g['llm_plan']]}")
    say(f"   {len(g['steps'] or [])} steps, ~{g['duration_s']} s of motion, heat {g['heat']}")
    sc, m = g.get("scene") or {}, g.get("margin") or {}
    say(f"   scene {sc.get('dir')}: views {sc.get('views_deg')} deg")
    for k, o in (sc.get("objects") or {}).items():
        say(f"     {k}: {o}")
    say(f"   margin: {'ROBUST' if m.get('robust') else 'MARGINAL'} ({m.get('seconds')} s) "
        + "; ".join(f"{k}: {v[:120]}" for k, v in (m.get("cases") or {}).items() if v != "ok"))
    if "--gen" in flags or (not m.get("robust") and "--marginal-ok" not in flags):
        say("not playing (" + ("--gen" if "--gen" in flags else "MARGINAL plan; --marginal-ok to play it") + ")")
        Path(out_dir / f"{time.strftime('%Y%m%d_%H%M%S')}.txt").write_text("\n".join(lines) + "\n")
        sys.exit(0)
    recording.set()
    threading.Thread(target=record, daemon=True).start()
    r = httpx.post(f"{url}/api/plan6/play", timeout=30)
    if r.status_code != 200:
        say(f"PLAY REFUSED: {r.json().get('detail')}")
    else:
        t1, last = time.time(), None
        while True:
            s = httpx.get(f"{url}/api/status", timeout=5).json()
            p = s["plan6"]
            if p["current"] != last and p["current"]:
                last = p["current"]
                say(f"  [{time.time() - t1:5.1f} s] {last}   (arms max {temps(s['state'])} C)")
                if "check grip after lift" in last:     # the holding pose, seen by the head camera
                    hold_tag = "run_" + time.strftime("%H%M%S") + "_holding"
                    subprocess.Popen([sys.executable, str(config.path("docs/reference_grasp/capture.py")), hold_tag],
                                     stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                    say(f"  holding pose photo: docs/reference_grasp/{hold_tag}_rgb.png")
                for key, name in (("let go", "placed"), ("move the hands up", "released")):   # how it landed / what the open hands left
                    if last.endswith(key):
                        tag2 = "run_" + time.strftime("%H%M%S") + "_" + name
                        subprocess.Popen([sys.executable, str(config.path("docs/reference_grasp/capture.py")), tag2],
                                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                        say(f"  {name} photo: docs/reference_grasp/{tag2}_rgb.png")
            if p["state"] in ("done", "failed"):
                say(f"RESULT: {p['state'].upper()} after {time.time() - t1:.1f} s" + (f": {p['error']}" if p["error"] else ""))
                break
            time.sleep(0.3)
recording.clear()
tag = "run_" + time.strftime("%H%M%S")          # the scene after the run, before anyone touches it
subprocess.run([sys.executable, str(config.path("docs/reference_grasp/capture.py")), tag], capture_output=True)
say(f"photo after the run: docs/reference_grasp/{tag}_rgb.png")
with open(log_path) as f:
    f.seek(log_start)
    server = [l.rstrip() for l in f if " skills " in l or " motion " in l or "WARNING" in l or "ERROR" in l]
say(f"server log during the run ({len(server)} lines):")
for l in server:
    print("   " + l)
    lines.append("   " + l)
name = out_dir / f"{time.strftime('%Y%m%d_%H%M%S')}.txt"
name.write_text("\n".join(lines) + "\n")
if rec:
    json.dump(rec, open(name.with_suffix(".joints.json"), "w"))
    print(f"joint recording: {name.with_suffix('.joints.json')} ({len(rec)} samples)")
print(f"saved {name}")
