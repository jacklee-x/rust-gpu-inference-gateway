#!/usr/bin/env bash
set -euo pipefail

# start-dev.sh
# Starts the C++ inference core and the Rust gateway for local development.
# Writes PIDs into .dev/pids (in project root) so stop-dev.sh can stop them.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
DEV_DIR="$ROOT_DIR/.dev"
CPP_DIR="$ROOT_DIR/cpp_inference"

mkdir -p "$DEV_DIR"

# Build or ensure C++ core binary exists
cd "$CPP_DIR"
if [ ! -x "./inference_core" ]; then
  echo "Building C++ inference core..."
  ./build.sh
fi

# Start C++ core in background
CORE_PORT=${1:-8081}
echo "Starting C++ inference core on port $CORE_PORT"
nohup ./inference_core --port "$CORE_PORT" > "$DEV_DIR/core.log" 2>&1 &
CORE_PID=$!
echo $CORE_PID > "$DEV_DIR/core.pid"
sleep 0.5

# Start Rust gateway
cd "$ROOT_DIR"
GATEWAY_PORT=${2:-8080}
# Allow overriding via env
export PORT=$GATEWAY_PORT
echo "Starting Rust gateway on port $GATEWAY_PORT"
nohup cargo run > "$DEV_DIR/gateway.log" 2>&1 &
GATEWAY_PID=$!
echo $GATEWAY_PID > "$DEV_DIR/gateway.pid"

# Record both PIDs in a single file for convenience
cat > "$DEV_DIR/pids" <<EOF
CORE_PID=$CORE_PID
GATEWAY_PID=$GATEWAY_PID
CORE_PORT=$CORE_PORT
GATEWAY_PORT=$GATEWAY_PORT
EOF

sleep 0.5

echo "Started C++ core (PID: $CORE_PID) and Rust gateway (PID: $GATEWAY_PID)"
echo "Logs: $DEV_DIR/core.log  $DEV_DIR/gateway.log"
