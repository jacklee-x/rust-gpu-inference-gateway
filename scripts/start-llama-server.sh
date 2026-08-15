#!/usr/bin/env bash
set -euo pipefail

# start-llama-server.sh
# ---------------------
# Linux/macOS counterpart of scripts/start-llama-server.ps1.
# Starts llama.cpp llama-server (the real GPU inference core) for local
# development, then point the Rust gateway at it with CORE_URL and
# CORE_PROTOCOL=llama-chat.
#
# Prerequisites:
#   - llama.cpp built with CUDA support (GGML_CUDA=ON), llama-server on
#     PATH or passed via --llama-server
#   - a GGUF model file (default: models/qwen2.5-0.5b-instruct-q4_k_m.gguf)
#
# Usage:
#   ./scripts/start-llama-server.sh [PORT] [MODEL] [LAYERS] [CTX]
# Example:
#   ./scripts/start-llama-server.sh 8081 models/qwen2.5-0.5b-instruct-q4_k_m.gguf 99 2048
#
# Afterwards start the gateway (in another terminal):
#   CORE_URL=http://127.0.0.1:8081 CORE_PROTOCOL=llama-chat cargo run --release

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
DEV_DIR="$ROOT_DIR/.dev"
mkdir -p "$DEV_DIR"

LOG_LLAMA="$DEV_DIR/llama-server.log"
PID_LLAMA="$DEV_DIR/llama-server.pid"

# CLI arguments: [PORT] [MODEL] [LAYERS] [CTX]
PORT=${1:-8081}
MODEL=${2:-"$ROOT_DIR/models/qwen2.5-0.5b-instruct-q4_k_m.gguf"}
LAYERS=${3:-99}   # layers offloaded to GPU (-ngl)
CTX=${4:-2048}    # context window size (-c)
LLAMA_SERVER=${LLAMA_SERVER:-llama-server}

# Model path: an absolute path is used as-is; a relative path is resolved
# against the project root.
case "$MODEL" in
  /*) MODEL_PATH="$MODEL" ;;
  *)  MODEL_PATH="$ROOT_DIR/$MODEL" ;;
esac

if [ ! -f "$MODEL_PATH" ]; then
  echo "Model file not found: $MODEL_PATH" >&2
  exit 1
fi

# Helper: wait for http 200 on URL (same logic as start-dev.sh)
wait_for_health() {
  local url="$1"
  local timeout_secs=${2:-120}
  local interval=${3:-3}
  local elapsed=0

  while [ "$(echo "$elapsed < $timeout_secs" | bc -l)" -eq 1 ]; do
    if curl -sSf --fail --connect-timeout 2 "$url" >/dev/null 2>&1; then
      return 0
    fi
    sleep "$interval"
    elapsed=$(echo "$elapsed + $interval" | bc -l)
  done
  return 1
}

# If a previous llama-server instance is still running, stop it first so
# the port is free and we never end up with two cores fighting.
if [ -f "$PID_LLAMA" ]; then
  old_pid=$(cat "$PID_LLAMA")
  if kill -0 "$old_pid" 2>/dev/null; then
    echo "Stopping previous llama-server (PID $old_pid)..."
    kill "$old_pid" || true
  fi
  rm -f "$PID_LLAMA"
fi

# All GGUF layers are offloaded to the GPU (--gpu-layers); the alias
# matches the gateway model registry entry so /v1/chat/completions
# requests are served under that name.
echo "Starting llama-server on 127.0.0.1:$PORT (logs: $LOG_LLAMA)"
nohup "$LLAMA_SERVER" \
  --model "$MODEL_PATH" \
  --host 127.0.0.1 \
  --port "$PORT" \
  --gpu-layers "$LAYERS" \
  --ctx-size "$CTX" \
  --alias qwen2.5-0.5b-instruct \
  >"$LOG_LLAMA" 2>&1 &
LLAMA_PID=$!
echo "$LLAMA_PID" > "$PID_LLAMA"

HEALTH_URL="http://127.0.0.1:$PORT/health"
if wait_for_health "$HEALTH_URL" 120 3; then
  echo "llama-server healthy at $HEALTH_URL (PID: $LLAMA_PID)"
  echo
  echo "Start the gateway in another terminal:"
  echo "  CORE_URL=http://127.0.0.1:$PORT CORE_PROTOCOL=llama-chat cargo run --release"
  echo "Stop the core later with: kill $LLAMA_PID (or use .dev/llama-server.pid)"
else
  echo "llama-server did not become healthy within 120s. See $LOG_LLAMA" >&2
  kill "$LLAMA_PID" 2>/dev/null || true
  rm -f "$PID_LLAMA"
  exit 1
fi