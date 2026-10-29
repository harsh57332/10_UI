#!/usr/bin/env bash
# Start/stop the server-side processes in the background (pid + log files in logs/).
#   scripts/svc.sh start|stop|restart|status|logs [server|vision|planner|llm|all]
# The UI server runs from env g1ui_server; model services use services.<name>.conda_env/module from the config.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
CONDA="${CONDA_DIR:-$HOME/miniconda3}"
SPY="$CONDA/envs/g1ui_server/bin/python"
LOGS="$ROOT/logs"; mkdir -p "$LOGS"
cd "$ROOT"

names() { "$SPY" -c "from common import config; print(' '.join(['server'] + list(config.load()['services'])))"; }
cmdline() {  # cmdline <name> -> the command to run
  if [ "$1" = server ]; then echo "$ROOT/scripts/run_server.sh"; return; fi
  "$SPY" -c "
from common import config
s = config.load()['services']['$1']
print(s['cmd'] if 'cmd' in s else '$CONDA/envs/' + s['conda_env'] + '/bin/python -m ' + s['module'])"
}
running() { [ -f "$LOGS/$1.pid" ] && kill -0 "$(cat "$LOGS/$1.pid")" 2>/dev/null; }

start() {
  if running "$1"; then echo "$1 already running (pid $(cat "$LOGS/$1.pid"))"; return; fi
  setsid nohup $(cmdline "$1") >> "$LOGS/$1.log" 2>&1 &
  echo $! > "$LOGS/$1.pid"
  echo "$1 started (pid $!, log logs/$1.log)"
}
stop() {   # TERM, then KILL after 5 s (the server can hang in shutdown and keep its port: two servers, 2026-10-29)
  if running "$1"; then
    local pid; pid=$(cat "$LOGS/$1.pid")
    kill -- -"$pid" 2>/dev/null || kill "$pid"
    for _ in $(seq 1 10); do kill -0 "$pid" 2>/dev/null || break; sleep 0.5; done
    if kill -0 "$pid" 2>/dev/null; then kill -9 -- -"$pid" 2>/dev/null || kill -9 "$pid"; echo "$1 killed (did not stop)"; else echo "$1 stopped"; fi
  fi
  rm -f "$LOGS/$1.pid"
}
status() { if running "$1"; then echo "$1: running (pid $(cat "$LOGS/$1.pid"))"; else echo "$1: stopped"; fi; }

ACTION="${1:-status}"; TARGET="${2:-all}"
LIST=$([ "$TARGET" = all ] && names || echo "$TARGET")
for n in $LIST; do
  case "$ACTION" in
    start) start "$n" ;;
    stop) stop "$n" ;;
    restart) stop "$n"; sleep 1; start "$n" ;;
    status) status "$n" ;;
    logs) echo "== $n"; tail -n 30 "$LOGS/$n.log" 2>/dev/null || true ;;
    *) echo "usage: $0 start|stop|restart|status|logs [server|vision|planner|llm|all]"; exit 1 ;;
  esac
done
