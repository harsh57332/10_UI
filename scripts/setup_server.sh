#!/usr/bin/env bash
# One-time server setup: the UI/relay env. Uses conda-forge only (the defaults channel needs a ToS accept).
set -euo pipefail
CONDA="${CONDA_DIR:-$HOME/miniconda3}"
"$CONDA/bin/conda" env list | grep -q '^g1ui_server ' || \
  "$CONDA/bin/conda" create -y -q -n g1ui_server --override-channels -c conda-forge python=3.12 ffmpeg
P="$CONDA/envs/g1ui_server/bin"
"$P/pip" install -q fastapi "uvicorn[standard]" httpx pyyaml numpy opencv-contrib-python-headless pyzmq python-multipart py_trees
"$P/python" -c "import fastapi, httpx, cv2, zmq, yaml; assert hasattr(cv2, 'TrackerKCF_create'); print('server env OK')"
