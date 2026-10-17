#!/usr/bin/env bash
# Manage the robot side from the server. All values come from config/config.yaml (+ .env).
#   scripts/robot.sh deploy                                   copy common/ robot/ config/ assets/robot/ to the robot
#   scripts/robot.sh install                                  deploy + install/enable the systemd services
#   scripts/robot.sh start|stop|restart|status [agent|camera] default: both
#   scripts/robot.sh logs [agent|camera] [n]
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PY="${G1UI_SERVER_PY:-$HOME/miniconda3/envs/g1ui_server/bin/python}"
set -a; [ -f "$ROOT/.env" ] && . "$ROOT/.env"; set +a

cfg() { (cd "$ROOT" && "$PY" -c "from common import config; c=config.load(); print(eval('c' + ''.join(f'[{k!r}]' for k in '$1'.split('.'))))"); }
HOST=$(cfg robot.host); USER_=$(cfg robot.ssh_user); APP=$(cfg robot.app_dir)
CONDA=$(cfg robot.conda_dir); ENV=$(cfg robot.conda_env); DEPS=$(cfg robot.deps_dir)
AGENT=$(cfg agent.service_name); CAMERA=$(cfg camera.service_name)
R="$USER_@$HOST"

svc() { case "${1:-all}" in agent) echo "$AGENT" ;; camera) echo "$CAMERA" ;; *) echo "$AGENT $CAMERA" ;; esac; }

deploy() {
  ssh "$R" "mkdir -p $APP/assets"
  rsync -a --delete --exclude __pycache__ "$ROOT/common" "$ROOT/robot" "$ROOT/config" "$R:$APP/"
  rsync -a --delete "$ROOT/assets/robot" "$R:$APP/assets/"
  echo "deployed to $R:$APP"
}

unit() {  # unit <service> <python module> <restart gap s>
  local envp="$CONDA/envs/$ENV"
  ssh "$R" "cat > /tmp/$1.service" <<UNIT
[Unit]
Description=G1 UI $2
After=network-online.target
Wants=network-online.target

[Service]
User=$USER_
WorkingDirectory=$APP
Environment=PYTHONNOUSERSITE=1
Environment=PYTHONUNBUFFERED=1
Environment=CYCLONEDDS_HOME=$DEPS/cyclonedds
Environment=LD_LIBRARY_PATH=$DEPS/cyclonedds/lib:$envp/lib
ExecStart=$envp/bin/python -m $2
Restart=always
RestartSec=$3

[Install]
WantedBy=multi-user.target
UNIT
  ssh "$R" "echo '$ROBOT_SUDO_PW' | sudo -S -p '' sh -c 'mv /tmp/$1.service /etc/systemd/system/ && systemctl daemon-reload && systemctl enable $1 && systemctl restart $1'"
  echo "service $1 installed and started"
}

install() {
  deploy
  unit "$AGENT" robot.agent 3
  unit "$CAMERA" robot.image_server "$(cfg camera.restart_gap_s)"
}

sctl() { ssh "$R" "echo '$ROBOT_SUDO_PW' | sudo -S -p '' systemctl $1 $(svc "$2")"; }

case "${1:-}" in
  deploy)  deploy ;;
  install) install ;;
  start|stop|restart) sctl "$1" "${2:-all}" ;;
  status)  for u in $(svc "${2:-all}"); do ssh "$R" "systemctl --no-pager status $u | head -6"; done ;;
  logs)    ssh "$R" "journalctl -u $(svc "${2:-agent}") --no-pager -n ${3:-80}" ;;
  *) echo "usage: $0 deploy|install|start|stop|restart|status [agent|camera]|logs [agent|camera] [n]"; exit 1 ;;
esac
