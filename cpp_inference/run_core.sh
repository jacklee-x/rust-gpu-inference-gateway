#!/usr/bin/env bash
set -euo pipefail

# Run the compiled inference core. Usage: ./run_core.sh [port]
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

PORT=${1:-8081}
OUT="./inference_core"

if [ ! -x "$OUT" ]; then
  if [ -f "$OUT" ]; then
    echo "Found binary $OUT but it is not executable; attempting to set +x"
    chmod +x "$OUT" || true
  else
    echo "Binary $OUT not found. Run ./build.sh first." >&2
    exit 1
  fi
fi

echo "Starting inference core on 127.0.0.1:${PORT} (foreground). Use Ctrl-C to stop."
"$OUT" --port "$PORT"
