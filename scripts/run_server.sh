#!/usr/bin/env bash
# Start the server (UI + relay to the Agent). Bind address and port come from config/config.yaml.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PY="${G1UI_SERVER_PY:-$HOME/miniconda3/envs/g1ui_server/bin/python}"
cd "$ROOT"
BIND=$("$PY" -c "from common import config; print(config.load()['server']['bind'])")
PORT=$("$PY" -c "from common import config; print(config.load()['server']['port'])")
exec "$PY" -m uvicorn server.app:app --host "$BIND" --port "$PORT" --log-level warning
