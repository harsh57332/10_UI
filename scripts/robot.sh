#!/usr/bin/env bash
# Manage the robot side from the server. All values come from config/config.yaml (+ .env).
#   scripts/robot.sh deploy     copy common/ robot/ config/ assets/robot/ to the robot
#   scripts/robot.sh install    deploy + install/enable the systemd Agent service
#   scripts/robot.sh start|stop|restart|status|logs
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PY="${G1UI_SERVER_PY:-$HOME/miniconda3/envs/g1ui_server/bin/python}"
set -a; [ -f "$ROOT/.env" ] && . "$ROOT/.env"; set +a

cfg() { (cd "$ROOT" && "$PY" -c "from common import config; c=config.load(); print(eval('c' + ''.join(f'[{k!r}]' for k in '$1'.split('.'))))"); }
HOST=$(cfg robot.host); USER_=$(cfg robot.ssh_user); APP=$(cfg robot.app_dir)
CONDA=$(cfg robot.conda_dir); ENV=$(cfg robot.conda_env); DEPS=$(cfg robot.deps_dir)
SVC=$(cfg agent.service_name)
R="$USER_@$HOST"

deploy() {
  ssh "$R" "mkdir -p $APP/assets"
  rsync -a --delete --exclude __pycache__ "$ROOT/common" "$ROOT/robot" "$ROOT/config" "$R:$APP/"
  rsync -a --delete "$ROOT/assets/robot" "$R:$APP/assets/"
  echo "deployed to $R:$APP"
}

install() {
  deploy
  local envp="$CONDA/envs/$ENV"
  ssh "$R" "cat > /tmp/$SVC.service" <<EOF
[Unit]
Description=G1 UI Agent
After=network-online.target
Wants=network-online.target

[Service]
User=$USER_
WorkingDirectory=$APP
Environment=PYTHONNOUSERSITE=1
Environment=PYTHONUNBUFFERED=1
Environment=CYCLONEDDS_HOME=$DEPS/cyclonedds
Environment=LD_LIBRARY_PATH=$DEPS/cyclonedds/lib:$envp/lib
ExecStart=$envp/bin/python -m robot.agent
Restart=always
RestartSec=3

[Install]
WantedBy=multi-user.target
EOF
  ssh "$R" "echo '$ROBOT_SUDO_PW' | sudo -S -p '' sh -c 'mv /tmp/$SVC.service /etc/systemd/system/ && systemctl daemon-reload && systemctl enable --now $SVC && systemctl restart $SVC'"
  echo "service $SVC installed and started"
}

sctl() { ssh "$R" "echo '$ROBOT_SUDO_PW' | sudo -S -p '' systemctl $1 $SVC"; }

case "${1:-}" in
  deploy)  deploy ;;
  install) install ;;
  start|stop|restart) sctl "$1" ;;
  status)  ssh "$R" "systemctl --no-pager status $SVC | head -15" ;;
  logs)    ssh "$R" "journalctl -u $SVC --no-pager -n ${2:-80}" ;;
  *) echo "usage: $0 deploy|install|start|stop|restart|status|logs [n]"; exit 1 ;;
esac
