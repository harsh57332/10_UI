#!/usr/bin/env bash
# One-time robot setup, run FROM THE SERVER (needs SSH key access to the robot and internet on the robot).
# Installs Miniforge, the robot env, CycloneDDS (C, 0.10.x) built into deps/, unitree_sdk2_python and
# librealsense (conda-forge build: the PyPI aarch64 wheels need glibc 2.32, Ubuntu 20.04 has 2.31).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PY="${G1UI_SERVER_PY:-$HOME/miniconda3/envs/g1ui_server/bin/python}"
cfg() { (cd "$ROOT" && "$PY" -c "from common import config; c=config.load(); print(eval('c' + ''.join(f'[{k!r}]' for k in '$1'.split('.'))))"); }
R="$(cfg robot.ssh_user)@$(cfg robot.host)"
CONDA=$(cfg robot.conda_dir); ENV=$(cfg robot.conda_env); DEPS=$(cfg robot.deps_dir)

ssh "$R" "CONDA=$CONDA ENV=$ENV DEPS=$DEPS bash -s" <<'EOF'
set -euo pipefail
export PYTHONNOUSERSITE=1
[ -x "$CONDA/bin/conda" ] || { curl -sSL -o /tmp/mf.sh https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-aarch64.sh && bash /tmp/mf.sh -b -p "$CONDA" && rm /tmp/mf.sh; }
C="$CONDA/bin/conda"; E="$CONDA/envs/$ENV"
[ -d "$E" ] || $C create -y -q -n "$ENV" python=3.10
$C install -y -q -n "$ENV" -c conda-forge libstdcxx "librealsense=2.58.3=cpu_py310*"

mkdir -p "$DEPS" && cd "$DEPS"
if [ ! -f cyclonedds/lib/libddsc.so ]; then
  [ -d cyclonedds-src ] || git clone -q --depth 1 -b releases/0.10.x https://github.com/eclipse-cyclonedds/cyclonedds cyclonedds-src
  cmake -S cyclonedds-src -B cyclonedds-src/build -DCMAKE_INSTALL_PREFIX="$DEPS/cyclonedds" \
        -DBUILD_EXAMPLES=OFF -DBUILD_TESTING=OFF -DBUILD_IDLC_TESTING=OFF >/dev/null
  cmake --build cyclonedds-src/build -j8 --target install >/dev/null
fi
export CYCLONEDDS_HOME="$DEPS/cyclonedds"
$C env config vars set -n "$ENV" PYTHONNOUSERSITE=1 CYCLONEDDS_HOME="$CYCLONEDDS_HOME" \
   LD_LIBRARY_PATH="$CYCLONEDDS_HOME/lib:$E/lib" >/dev/null

[ -d unitree_sdk2_python ] || git clone -q --depth 1 https://github.com/unitreerobotics/unitree_sdk2_python
"$E/bin/pip" install -q cyclonedds==0.10.2
"$E/bin/pip" install -q -e unitree_sdk2_python
"$E/bin/pip" install -q fastapi "uvicorn[standard]" pyzmq opencv-python-headless numpy pyyaml
LD_LIBRARY_PATH="$CYCLONEDDS_HOME/lib:$E/lib" "$E/bin/python" -c \
  "import unitree_sdk2py, pyrealsense2, cv2, zmq, fastapi; print('robot env OK')"
EOF
