#!/usr/bin/env bash
set -euo pipefail

# Portable build script for the C++ inference core.
# Prefers nvcc (CUDA) if available, otherwise falls back to g++/clang++.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

SRC="inference_core.cpp"
OUT="inference_core"

echo "Building C++ inference core from $SRC -> $OUT"

# Try nvcc first (CUDA path)
if command -v nvcc >/dev/null 2>&1; then
  echo "nvcc found: building CUDA-enabled binary"
  nvcc -std=c++17 -O2 "$SRC" -o "$OUT"
  echo "Built $OUT with nvcc"
  exit 0
fi

# Try g++
if command -v g++ >/dev/null 2>&1; then
  echo "g++ found: building native binary"
  g++ -std=c++17 -O2 "$SRC" -o "$OUT" -pthread
  echo "Built $OUT with g++"
  exit 0
fi

# Try clang++
if command -v clang++ >/dev/null 2>&1; then
  echo "clang++ found: building native binary"
  clang++ -std=c++17 -O2 "$SRC" -o "$OUT" -pthread
  echo "Built $OUT with clang++"
  exit 0
fi

echo "No supported C++ compiler found. Install nvcc, g++, or clang++ and re-run this script." >&2
exit 1
