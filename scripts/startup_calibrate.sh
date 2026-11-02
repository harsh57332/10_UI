#!/usr/bin/env bash
# Run after the services are up and the robot is powered on and standing still (svc.sh start runs it in the background).
# Connects the server to the robot, then checks what the robot must know about itself and writes logs/startup_report.txt:
#   camera angle  scripts.camera_tilt --auto (camera accelerometer + torso IMU + level surfaces; rewrites the config's fix only if it moved
#                 > camera_mount.auto_tolerance_deg, then restarts the server and connects again)
#   hands         the agent detects them (dummy / Inspire) and the server follows
#   legs / body   FSM mode, torso IMU, waist lock, motor temperatures, battery
# Depth camera intrinsics are the sensor's own factory values (read at start); its mount position is the URDF's.
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"; cd "$ROOT"
PY="${CONDA_DIR:-$HOME/miniconda3}/envs/g1ui_server/bin/python"
REPORT="$ROOT/logs/startup_report.txt"
HOST=$("$PY" -c "from common import config; print(config.load()['robot']['host'])")
PORT=$("$PY" -c "from common import config; print(config.load()['server']['port'])")
connect() {
  for _ in $(seq 1 40); do
    curl -s -m5 -X POST "localhost:$PORT/api/connect" -H 'content-type: application/json' -d "{\"host\":\"$HOST\"}" | grep -q '"connected":true' && return 0
    sleep 3
  done
  return 1
}
{
  echo "startup calibration $(date '+%F %T')"
  if ! connect; then echo "robot $HOST not reachable: nothing calibrated (power it on, then run scripts/startup_calibrate.sh)"; exit 0; fi
  for _ in $(seq 1 40); do                                                  # the robot must stand in manipulation mode (fsm 501), still
    curl -s -m5 "localhost:$PORT/api/status" | "$PY" -c "import json,sys; s=json.load(sys.stdin).get('state') or {}; sys.exit(0 if s.get('mode')=='manipulation' and s.get('fsm_id')==501 else 1)" && break
    sleep 3
  done
  sleep 10                                                                 # settled; the agent's first state / hands detection
  "$PY" -m scripts.camera_tilt --auto 2>&1 | grep -E "camera looks|fix \(|CALIBRATION|no accelerometer|moving|moved"
  rc=${PIPESTATUS[0]}
  if [ "$rc" = 3 ]; then
    echo "camera angle changed: restarting the server to load it"; "$ROOT/scripts/svc.sh" restart server >/dev/null; connect || echo "server did not reconnect"
  elif [ "$rc" != 0 ]; then
    echo "camera angle NOT checked (stand the robot still and run scripts/startup_calibrate.sh again)"
  fi
  sleep 3
  curl -s -m5 "localhost:$PORT/api/status" | "$PY" -c "
import json, sys
s = json.load(sys.stdin).get('state')
if not s: print('robot state unavailable'); sys.exit()
t = {j['name']: j['temp'] for j in s['joints'] if j['index'] in (14, 15, 22)}
print('mode', s['mode'], 'fsm', s['fsm_id'], '| hands', (s.get('end_effector') or {}).get('type'), '| torso rpy', s['imu']['torso_rpy'],
      '| waist locked', s['arms'].get('waist_held'), '| temps', t, '| battery', (s.get('battery') or {}).get('soc'), '%')"
} 2>&1 | tee "$REPORT"
