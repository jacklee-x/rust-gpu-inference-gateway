# Run Guide

This page explains how to run the current first-version project and verify its behavior.

## Prerequisites

- Rust toolchain installed (`rustup`, `cargo`)
- Python 3.10+ installed
- `pip` available

## 1. Build and start the C++ inference core

The current version uses a real C++ inference service on `127.0.0.1:8081` instead of the earlier Python mock server.

From the project root, open one terminal and run:

```bash
cd cpp_inference
./build.sh
./run_core.sh
```

This C++ core listens on `127.0.0.1:8081` and exposes `/health` and `/infer` for the Rust gateway.

## 2. Start the Rust gateway

Open another terminal and run from the project root:

```bash
cargo run --release
```

The gateway listens on `127.0.0.1:8080`.

## 3. Verify the health endpoint

Open another terminal and run:

```bash
curl http://127.0.0.1:8080/health
```

Expected output:

```json
{"status":"healthy"}
```

## 4. Verify the inference endpoint

Send a sample request.

### macOS / Linux / Git Bash / WSL

```bash
curl -X POST http://127.0.0.1:8080/infer \
  -H "Content-Type: application/json" \
  -d '{"model":"llama-7b","input":"Hello world","options":{"max_tokens":32}}'
```

### PowerShell

```powershell
Invoke-RestMethod -Uri http://127.0.0.1:8080/infer -Method Post -ContentType "application/json" -Body '{"model":"llama-7b","input":"Hello world","options":{"max_tokens":32}}'
```

### Windows cmd / PowerShell using curl.exe

```powershell
curl.exe -X POST http://127.0.0.1:8080/infer -H "Content-Type: application/json" -d '{"model":"llama-7b","input":"Hello world","options":{"max_tokens":32}}'
```

Expected result contains fields like `request_id`, `model`, `output`, `usage`, and `status`.

> Note: this version uses a queued worker pool. The Rust gateway accepts requests into a bounded queue and dispatches them to worker tasks that call the C++ inference core on port `8081`. If the queue is full, you may receive a `503` response with `{"status":"queue_full"}`.

## 4. Run the Python demo client

Install dependencies.

### PowerShell / Windows

If `py` exists on your machine, use:

```powershell
py -m pip install -r python/requirements.txt
```

If `py` does not exist, use the Python executable directly:

```powershell
python -m pip install -r python/requirements.txt
```

If `python` also does not exist, install Python from https://www.python.org/downloads/ or enable it from the Microsoft Store.

Run the demo client from the repository root.

### PowerShell / Windows

If `py` exists:

```powershell
py .\python\demo\client.py --model llama-7b --input "Hello world"
```

If `python` exists:

```powershell
python .\python\demo\client.py --model llama-7b --input "Hello world"
```

### macOS / Linux

```bash
python3 python/demo/client.py --model llama-7b --input "Hello world"
```

This script sends a request to the Rust gateway and prints the returned JSON.

## 5. Run the Python benchmark

Run the benchmark script:

```bash
python python/benchmark/benchmark.py
```

It sends 10 requests to the gateway and prints per-request latency and average latency.

## 6. What this version does

At present, the system is a working end-to-end proof of concept. It supports:

- a Rust HTTP gateway on `127.0.0.1:8080`
- a C++ inference service on `127.0.0.1:8081`
- `GET /health` on both layers
- `POST /infer` through the Rust gateway to the C++ core
- worker-pool request scheduling and timeouts in the Rust layer

The C++ core is not yet a production LLM runtime, but it is a real C++ service with an optional CUDA-ready execution path and a CPU fallback.

## 7. Next steps

The parts still planned for later versions are:

- GPU model loading and execution with a true CUDA kernel
- ONNX Runtime / TensorRT integration
- Prometheus-formatted metrics
- `GET /models` endpoint and model registry support
- container orchestration and production deployment examples
