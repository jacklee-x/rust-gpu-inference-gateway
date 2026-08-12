#!/usr/bin/env bash
set -euo pipefail

# start-dev.sh (enhanced)
# Starts the C++ inference core and the Rust gateway for local development.
# Waits for each service to report healthy via /health before returning.
# On failure the script attempts to stop any started processes (rollback).

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
DEV_DIR="$ROOT_DIR/.dev"
CPP_DIR="$ROOT_DIR/cpp_inference"

mkdir -p "$DEV_DIR"

LOG_CORE="$DEV_DIR/core.log"
LOG_GATEWAY="$DEV_DIR/gateway.log"
PID_CORE="$DEV_DIR/core.pid"
PID_GATEWAY="$DEV_DIR/gateway.pid"
PIDS_FILE="$DEV_DIR/pids"

# Helper: wait for http 200 on URL
wait_for_health() {
  local url="$1"
  local timeout_secs=${2:-15}
  local interval=${3:-0.5}
  local elapsed=0

  while [ $(echo "$elapsed < $timeout_secs" | bc -l) -eq 1 ]; do
    if curl -sSf --fail --connect-timeout 2 "$url" >/dev/null 2>&1; then
      return 0
    fi
    sleep "$interval"
    elapsed=$(echo "$elapsed + $interval" | bc -l)
  done
  return 1
}

rollback() {
  echo "Rolling back started processes..."
  [ -f "$PID_GATEWAY" ] && { pid=$(cat "$PID_GATEWAY") && kill "$pid" 2>/dev/null || true; rm -f "$PID_GATEWAY"; }
  [ -f "$PID_CORE" ] && { pid=$(cat "$PID_CORE") && kill "$pid" 2>/dev/null || true; rm -f "$PID_CORE"; }
  rm -f "$PIDS_FILE"
  echo "Rolled back. See logs: $LOG_CORE , $LOG_GATEWAY"
}

# Build C++ core if needed
cd "$CPP_DIR"
if [ ! -x "./inference_core" ]; then
  echo "Building C++ inference core..."
  ./build.sh
fi

CORE_PORT=${1:-8081}
GATEWAY_PORT=${2:-8080}

# Start C++ core
echo "Starting C++ inference core on port $CORE_PORT (logs: $LOG_CORE)"
nohup ./inference_core --port "$CORE_PORT" >"$LOG_CORE" 2>&1 &
CORE_PID=$!
echo "$CORE_PID" > "$PID_CORE"

CORE_HEALTH_URL="http://127.0.0.1:$CORE_PORT/health"
if wait_for_health "$CORE_HEALTH_URL" 15 0.5; then
  echo "C++ core healthy at $CORE_HEALTH_URL"
else
  echo "C++ core failed to become healthy within timeout" >&2
  rollback
  exit 1
fi

# Start Rust gateway
cd "$ROOT_DIR"
export PORT="$GATEWAY_PORT"
echo "Starting Rust gateway on port $GATEWAY_PORT (logs: $LOG_GATEWAY)"
nohup cargo run >"$LOG_GATEWAY" 2>&1 &
GATEWAY_PID=$!
echo "$GATEWAY_PID" > "$PID_GATEWAY"

GATEWAY_HEALTH_URL="http://127.0.0.1:$GATEWAY_PORT/health"
if wait_for_health "$GATEWAY_HEALTH_URL" 15 0.5; then
  echo "Rust gateway healthy at $GATEWAY_HEALTH_URL"
else
  echo "Rust gateway failed to become healthy within timeout" >&2
  rollback
  exit 1
fi

# Record pids and ports
cat > "$PIDS_FILE" <<EOF
CORE_PID=$CORE_PID
GATEWAY_PID=$GATEWAY_PID
CORE_PORT=$CORE_PORT
GATEWAY_PORT=$GATEWAY_PORT
EOF

echo "Started C++ core (PID: $CORE_PID) and Rust gateway (PID: $GATEWAY_PID)"
echo "Logs: $LOG_CORE  $LOG_GATEWAY"
exit 0
