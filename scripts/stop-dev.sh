#!/usr/bin/env bash
set -euo pipefail

# stop-dev.sh
# Stops processes started by start-dev.sh using PIDs recorded under .dev/

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
DEV_DIR="$ROOT_DIR/.dev"

if [ ! -d "$DEV_DIR" ]; then
  echo "No $DEV_DIR directory found; nothing to stop." >&2
  exit 0
fi

PID_FILE="$DEV_DIR/pids"
CORE_PID_FILE="$DEV_DIR/core.pid"
GATEWAY_PID_FILE="$DEV_DIR/gateway.pid"

kill_if_exists() {
  local pidfile="$1"
  if [ -f "$pidfile" ]; then
    local pid
    pid=$(cat "$pidfile")
    if [ -n "$pid" ] 2>/dev/null; then
      if kill -0 "$pid" 2>/dev/null; then
        echo "Stopping PID $pid"
        kill "$pid" || kill -9 "$pid" || true
        sleep 0.2
      else
        echo "PID $pid not running"
      fi
    fi
    rm -f "$pidfile"
  fi
}

kill_if_exists "$CORE_PID_FILE"
kill_if_exists "$GATEWAY_PID_FILE"

# clean pids file
if [ -f "$PID_FILE" ]; then
  rm -f "$PID_FILE"
fi

echo "Stopped dev services (if any) and cleaned $DEV_DIR. Logs remain for inspection."