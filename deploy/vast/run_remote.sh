#!/usr/bin/env bash
# Start the Decadic brain server on the rented GPU box, inside a detached tmux
# session so it survives the provisioning ssh connection. Invoked by the control
# plane (decadic/api/vast/controller.py); the scene/body is started afterwards
# via POST /environment through the tunnel (the adapter the supervisor spawns
# inherits MUJOCO_GL=egl from this server process, so the camera renders
# headless). This script only launches the server and returns immediately.
#
# Reads (optional) env: PRESET, ENCODER, WHISPER_MODEL.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
APP_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
mkdir -p "$APP_ROOT/logs"

PRESET="${PRESET:-full}"
ENCODER="${ENCODER:-hf}"
WHISPER_MODEL="${WHISPER_MODEL:-openai/whisper-small}"

# Generate an inner launch script so all env is baked in cleanly (no quoting
# games through tmux). Values are expanded here, at run_remote.sh time.
INNER="$SCRIPT_DIR/_run_server.sh"
cat > "$INNER" <<EOF
#!/usr/bin/env bash
cd "$APP_ROOT"
export DECADIC_DEVICE=cuda
export MUJOCO_GL=egl
export DECADIC_ENCODER_MODE="$ENCODER"
export DECADIC_NEURAL_PRESET="$PRESET"
export DECADIC_WHISPER_MODEL="$WHISPER_MODEL"
export DECADIC_N_ACTUATORS=21
export DECADIC_PLASTICITY_ENABLED=1
export DECADIC_SPARSE_ENABLED=1
export DECADIC_GROWTH_ENABLED=1
export DECADIC_CURRICULUM_MODE=legacy
export DECADIC_SELF_HOST=127.0.0.1
export DECADIC_SELF_PORT=8765
# Optional live stream: the body adapter the server spawns inherits this env,
# and auto-enables the RTMP spectator stream when a target is set. Provide a
# YouTube stream key (or full RTMP URL) in the environment that invokes this
# script and it flows through to the body untouched; empty => streaming off.
export DECADIC_YT_STREAM_KEY="${DECADIC_YT_STREAM_KEY:-}"
export DECADIC_STREAM_RTMP="${DECADIC_STREAM_RTMP:-}"
export DECADIC_STREAM_CAMERA="${DECADIC_STREAM_CAMERA:-}"
export DECADIC_STREAM_SIZE="${DECADIC_STREAM_SIZE:-1280x720}"
export DECADIC_STREAM_FPS="${DECADIC_STREAM_FPS:-30}"
export DECADIC_STREAM_BITRATE="${DECADIC_STREAM_BITRATE:-4500k}"
export DECADIC_STREAM_VCODEC="${DECADIC_STREAM_VCODEC:-libx264}"
exec python -m uvicorn decadic.api.app:app --host 0.0.0.0 --port 8765
EOF
chmod +x "$INNER"

echo "[run] (re)starting server tmux session 'decadic' (preset=$PRESET encoder=$ENCODER)"
tmux kill-session -t decadic 2>/dev/null || true
tmux new-session -d -s decadic "bash '$INNER' >> '$APP_ROOT/logs/server.log' 2>&1"

echo "[run] server launching in tmux session 'decadic'; logs: $APP_ROOT/logs/server.log"
