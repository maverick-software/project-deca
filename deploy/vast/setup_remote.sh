#!/usr/bin/env bash
# Provision a freshly-rented Vast.ai GPU box for the Decadic brain + body.
# Invoked by the control plane over ssh (decadic/api/vast/controller.py).
#
# Reads (optional) env: ENCODER, WHISPER_MODEL.
#   - Installs MuJoCo/EGL system libs so the egocentric camera renders headless.
#   - Installs the project WITHOUT touching the image's CUDA torch (--no-deps),
#     then the remaining non-torch runtime deps.
#   - Prewarms the frozen CLIP + Whisper weights when ENCODER=hf so the first
#     cognitive cycle does not pay the ~1 GB download.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
APP_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$APP_ROOT"

echo "[setup] APP_ROOT=$APP_ROOT"
echo "[setup] installing system libraries (EGL / MuJoCo / tmux)"
export DEBIAN_FRONTEND=noninteractive
apt-get update -y
apt-get install -y --no-install-recommends \
  libgl1 libegl1 libglfw3 libosmesa6 libglib2.0-0 git tmux ca-certificates

echo "[setup] installing python deps (preserving the image's CUDA torch)"
python -m pip install --no-input --upgrade pip
python -m pip install --no-input -e . --no-deps
python -m pip install --no-input \
  "fastapi>=0.115" "uvicorn[standard]>=0.30" "pydantic>=2.7" "numpy>=1.26" \
  "transformers>=4.40" "Pillow>=10.0" "httpx>=0.27" "mujoco>=3.1" "websockets>=12"

echo "[setup] torch / CUDA check"
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"

if [ "${ENCODER:-hf}" = "hf" ]; then
  echo "[setup] prewarming frozen CLIP + Whisper (${WHISPER_MODEL:-openai/whisper-small})"
  export DECADIC_ENCODER_MODE=hf
  export DECADIC_WHISPER_MODEL="${WHISPER_MODEL:-openai/whisper-small}"
  export MUJOCO_GL=egl
  python - <<'PY' || echo "[setup] prewarm skipped (encoders will download on first cycle)"
import torch
from decadic.nn.frozen_encoders import FrozenSensoryEncoders
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
FrozenSensoryEncoders(mode="hf", device=dev, proprio_dim_out=64)
print("[setup] prewarm ok on", dev)
PY
fi

echo "[setup] done"
