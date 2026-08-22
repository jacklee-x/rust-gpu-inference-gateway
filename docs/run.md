# Run Guide

This page explains how to run the current first-version project and verify its behavior.

## Prerequisites

- Rust toolchain installed (`rustup`, `cargo`)
- Python 3.10+ installed
- `pip` available

## GPU inference mode (llama.cpp + CUDA) — recommended

The current version supports **real GPU inference** through llama.cpp's
`llama-server`, which replaces the C++ mock core as the backend. The
gateway talks to it with `CORE_PROTOCOL=llama-chat` (OpenAI-compatible
`/v1/chat/completions`).

### 0.1 Build llama.cpp with CUDA support

```bash
# Linux
git clone https://github.com/ggml-org/llama.cpp.git
cd llama.cpp
cmake -B build -DGGML_CUDA=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release --target llama-server -j
```

Windows (MSVC + nvcc) notes:

- CUDA Toolkit 13.x, VS 2022 (C++ workload) and CMake are required.
- Add the CUDA toolkit to PATH or set `CUDA_PATH` before configuring
  (MSBuild otherwise fails with "The CUDA Toolkit directory '' does
  not exist").
- Use a target architecture matching your GPU, e.g. for an RTX 2070
  (Turing, sm_75): `-DCMAKE_CUDA_ARCHITECTURES=75`.
- At runtime the process needs the CUDA `bin/x64` directory on PATH
  (cuBLAS DLLs); the start script below does this automatically.

### 0.2 Download a GGUF model

```bash
mkdir -p models
curl -L -o models/qwen2.5-0.5b-instruct-q4_k_m.gguf \
  https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF/resolve/main/qwen2.5-0.5b-instruct-q4_k_m.gguf
```

(If huggingface.co is unreachable, use the mirror https://hf-mirror.com
with the same path.)

### 0.3 Start llama-server

Single-model mode:

Windows:

```powershell
.\scripts\start-llama-server.ps1 -LlamaServer C:\path\to\llama-server.exe
```

Linux/macOS:

```bash
./scripts/start-llama-server.sh 8081
```

Multi-model (router) mode — every GGUF in the directory is served in
its own lazy-loaded child process and listed by `GET /v1/models`:

```powershell
.\scripts\start-llama-server.ps1 -ModelsDir ..\models -LlamaServer C:\path\to\llama-server.exe
```

```bash
LLAMA_MODELS_DIR=models ./scripts/start-llama-server.sh 8081
```

Both scripts wait for `/health` and print the next command to run. They
load all model layers onto the GPU (`--gpu-layers 99`).

### 0.3.1 Dynamic model registry

In `llama-chat` mode the gateway does **not** trust its static model
list. A background task pulls the real model list from llama-server's
`GET /v1/models` immediately on startup and then every
`REGISTRY_REFRESH_SECS` seconds (default 30), replacing the registry:

- `GET /models` always reflects what the core actually serves, including
  per-model `status` (`loaded` / `unloaded` / `failed`, ...) as reported
  by llama-server.
- `POST /infer` validates model names (and aliases) against that list;
  requests for unknown models get `400 unknown model`.
- A failed refresh only logs a warning and keeps the previous snapshot
  (a briefly-unreachable core never empties the registry).
- In `CORE_PROTOCOL=infer` (legacy) mode the static default registry is
  used, unchanged.

### 0.4 Start the gateway in llama-chat mode

```bash
CORE_URL=http://127.0.0.1:8081 CORE_PROTOCOL=llama-chat cargo run --release
```

(The defaults are `CORE_URL=http://127.0.0.1:8081` and
`CORE_PROTOCOL=infer`; only `CORE_PROTOCOL=llama-chat` needs to be set
for GPU mode.)

### 0.5 Verify GPU inference end to end

1. Health:

   ```bash
   curl http://127.0.0.1:8080/health
   # {"status":"healthy"}
   ```

2. Inference through the gateway (use the registered model name):

   ```bash
   curl -X POST http://127.0.0.1:8080/infer -H "Content-Type: application/json" \
     -d '{"model":"qwen2.5-0.5b-instruct","input":"What is the capital of France?","options":{"max_tokens":32}}'
   ```

   Expected fields: `request_id`, `model`, `output` (a real model
   response such as "The capital of France is Paris."), `usage` with
   `latency_ms` and `tokens`, `status: "ok"`.

3. Confirm the work really runs on the GPU (both processes should appear
   with compute type `C`):

   ```bash
   nvidia-smi
   # | 0 ... llama-server.exe ...  C (or llama-server) |
   ```

4. Automated test (requires the gateway on port 8080):

   ```bash
   python -m pytest python/tests -v
   # python/tests/test_end_to_end.py::test_infer_endpoint PASSED
   ```

5. `GET /models` shows the loaded GPU model:
   `{"models":[{"name":"qwen2.5-0.5b-instruct","status":"loaded","device":"gpu0","backend":"llama.cpp-cuda"}, ...]}`

6. Worker pool dynamic scaling:

   ```bash
   # unit tests cover pool growth/shrink and bounds:
   cargo test --release
   # test result: ok. 5 passed (worker_pool::tests::*)

   # load test: fire ~40 concurrent requests while sampling /metrics
   # e.g. "python python/benchmark/benchmark.py" with a small
   # concurrent client, or curl in parallel from several terminals:
   for i in $(seq 1 40); do curl -s -X POST http://127.0.0.1:8080/infer \
     -H "Content-Type: application/json" \
     -d '{"model":"qwen2.5-0.5b-instruct","input":"hello","options":{"max_tokens":8}}' & done; wait

   # while the load runs, watch the pool gauge:
   watch -n 0.2 'curl -s http://127.0.0.1:8080/metrics | grep inference_worker_pool_size'
   ```

   Expected: `inference_worker_pool_size` rises from `1`
   (`MIN_CONCURRENCY`) to above the minimum while the queue is deep
   (up to `inference_worker_pool_max`, i.e. `MAX_CONCURRENCY`), then
   settles back to `1` once the queue drains. All concurrent requests
   still return `status: "ok"`.

---

## CPU mode (C++ mock core) — legacy

This describes the original mock-based path using `inference_core.cpp`
(simple uppercase transform, optional toy CUDA branch, CPU fallback).
It is still supported via the default `CORE_PROTOCOL=infer`.

## 1. Build and start the C++ inference core

The current version uses a real C++ inference service on `127.0.0.1:8081` instead of the earlier Python mock server.

From the project root, open one terminal and run:

```bash
cd cpp_inference
./build.sh
./run_core.sh
```

or use the helper script to start both services (recommended):

```bash
# from project root: ./scripts/start-dev.sh [CORE_PORT] [GATEWAY_PORT]
./scripts/start-dev.sh 8081 8080
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

## 6. Check metrics and models

The gateway exposes Prometheus-formatted metrics:

```bash
curl http://127.0.0.1:8080/metrics
```

Metrics include request counters by status (`ok` / `error` / `timeout` / `queue_full` / `invalid_request`), the in-flight gauge, latency summary, a latency histogram, and the adaptive worker pool gauges (`inference_worker_pool_size`, `inference_worker_pool_max`).

List known models:

```bash
curl http://127.0.0.1:8080/models
```

`POST /infer` validates the `model` field against the registry and returns `400 {"status":"error","message":"unknown model '...'"}` for unknown names.

### Tracing (optional OpenTelemetry export)

By default the gateway logs to stdout only. Set `OTEL_EXPORTER_OTLP_ENDPOINT`
(for example `http://localhost:4318`) to additionally export spans over
OTLP/HTTP protobuf to `{endpoint}/v1/traces`. Each `/infer` request produces
an `infer_request` span (`model`, `protocol`, `outcome`) containing a
`core_call` child span with `core_latency_ms`, and outgoing core calls carry
W3C `traceparent` headers so traces can span gateway -> core.

Correlation ids: send your own `x-request-id` header (a valid UUID) to reuse
it as the `request_id` across services; the gateway always echoes the final
id back in the `x-request-id` response header.

## 7. What this version does

At present, the system is a working end-to-end proof of concept. It supports:

- a Rust HTTP gateway on `127.0.0.1:8080`
- a C++ inference service on `127.0.0.1:8081`
- `GET /health` on both layers
- `POST /infer` through the Rust gateway to the C++ core
- worker-pool request scheduling and timeouts in the Rust layer
- Prometheus-compatible `GET /metrics` and a model registry `GET /models`
- per-request UUID `request_id` correlation across gateway logs and responses
- environment-variable configuration (`CORE_URL`, `CORE_PROTOCOL`, `MIN_CONCURRENCY`, `MAX_CONCURRENCY`, `QUEUE_CAPACITY`, `REQUEST_TIMEOUT_SECS`, `BIND_ADDR` / `PORT`)
- adaptive worker pool (`MIN_CONCURRENCY`..`MAX_CONCURRENCY`): concurrency grows with queue depth and collapses when idle; exposed in `/metrics` (`inference_worker_pool_size`, `inference_worker_pool_max`)
- Docker images for both services and a Compose file in `deploy/`

The C++ core is not yet a production LLM runtime, but it is a real C++ service with an optional CUDA-ready execution path and a CPU fallback.

## 8. Next steps

The parts still planned for later versions are:

- GPU model loading and execution with a true CUDA kernel
- ONNX Runtime / TensorRT integration
- request batching and dynamic model registry (backed by storage or the core)
- gRPC API and Kubernetes deployment examples
