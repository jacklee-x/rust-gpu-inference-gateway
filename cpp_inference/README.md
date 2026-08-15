# C++ Inference Core

This directory contains the first actual C++ inference core for the gateway.

## Purpose

The Rust gateway does not run model execution directly. Instead, it forwards requests to a dedicated local inference service on port `127.0.0.1:8081`.

The C++ core is intentionally split into two layers:

- a real C++ HTTP service that exposes `GET /health` and `POST /infer`
- an optional CUDA execution path that is enabled when the project is built with a CUDA-capable toolchain (`__CUDACC__`)
- a CPU fallback path that keeps the project runnable in non-GPU environments

## Current behavior

The current implementation is still a lightweight prototype, not a production LLM engine. It performs a deterministic C++-side inference transformation on the input text and returns a JSON response compatible with the Rust gateway.

Important points:

- The service is HTTP-based and matches the Rust gateway contract.
- It can be built with MSVC on Windows or `g++` on Linux/macOS.
- If compiled with CUDA, it exposes a CUDA hook where the actual GPU kernel can be inserted.
- If CUDA is not available, the CPU fallback is used so the end-to-end pipeline still runs.

## Build

From this directory:

```bash
./build.sh
```

or, on Linux/macOS:

```bash
g++ -std=c++17 -O2 inference_core.cpp -o inference_core -pthread
```

On Windows, use a MinGW or MSVC toolchain; the source uses Winsock2 (`winsock2.h`) and links `ws2_32.lib`.

## Run

```bash
./run_core.sh
```

The server binds to `127.0.0.1:8081` and waits for inference requests.

## API contract

`POST /infer`

```json
{
  "model": "llama-7b",
  "input": "hello world",
  "options": {
    "max_tokens": 32,
    "temperature": 0.8,
    "top_p": 0.95
  }
}
```

Response example:

```json
{
  "request_id": "cpp-1710000000000",
  "model": "llama-7b",
  "output": "C++ CPU inference result for model=llama-7b ...",
  "usage": {
    "latency_ms": 5,
    "tokens": 32
  },
  "status": "ok"
}
```

## Roadmap

The next major steps are:

1. replace the CPU fallback with a real CUDA kernel path and model loader
2. integrate a small ONNX Runtime or TensorRT path for production inference
3. add request batching and model management (dynamic/loaded model status)</think>

> Note: `GET /metrics` and `GET /models` are served by the Rust gateway; the core provides `GET /health` for orchestration.
