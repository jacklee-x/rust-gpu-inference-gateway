# rust-gpu-inference-gateway

A Rust-based inference gateway with a C++/CUDA GPU inference core and Python tooling for model preparation, demo clients, and benchmarking.

## Core Features

- Rust gateway for HTTP/gRPC inference requests
- Task queue and worker pool for request scheduling
- C++ inference core for GPU-backed model execution
- Prometheus-compatible metrics endpoint (`GET /metrics`)
- Model registry and discovery endpoint (`GET /models`), dynamically synced with the inference core in `llama-chat` mode (multi-model router support)
- Traceable per-request `request_id` (UUID) stamped by the gateway
- Environment-variable based configuration (`CORE_URL`, `CORE_PROTOCOL` (`infer` | `llama-chat`), `MIN_CONCURRENCY`, `MAX_CONCURRENCY`, `QUEUE_CAPACITY`, `REQUEST_TIMEOUT_SECS`, ...)
- Adaptive worker pool: concurrency grows from `MIN_CONCURRENCY` to `MAX_CONCURRENCY` with the request queue depth and collapses when idle (visible in `/metrics` as `inference_worker_pool_size`)
- Python tooling for demo clients, model preparation, and benchmark scripts
- Production-style engineering features: configuration, logging, health checks, metrics, and Docker support

## Design Goals

1. Build a lightweight, end-to-end inference service that demonstrates cross-language engineering ability.
2. Showcase Rust async backend design, C++ GPU inference integration, and Python automation.
3. Provide a reproducible demo path from request to GPU inference result.
4. Keep the first version minimal and extendable.

## Architecture Overview

The project is composed of three main layers:

1. **Rust Gateway**
   - Entry point for client requests
   - HTTP/gRPC API
   - Request validation
   - Task scheduling
   - Observability

2. **C++ GPU Inference Core**
   - Handles model loading and execution
   - Executes inference on GPU via CUDA/TensorRT/ONNX Runtime or `llama.cpp`
   - Returns inference results to the Rust worker

3. **Python Tooling**
   - Demo clients for end-to-end testing
   - Model preparation and conversion scripts
   - Benchmark and load test scripts

## Quick Start

### GPU mode (llama.cpp + CUDA, recommended)

Requires a CUDA-capable NVIDIA GPU, CUDA Toolkit, CMake and a built
`llama-server` (see `docs/run.md` §0.1 for build steps), plus a GGUF
model in `models/` (see §0.2 for the download command).

1. Start llama-server (waits for /health, prints the next command):

   ```bash
   ./scripts/start-llama-server.sh 8081            # Linux/macOS
   .\scripts\start-llama-server.ps1                # Windows
   ```

2. Start the gateway in llama-chat mode:

   ```bash
   CORE_URL=http://127.0.0.1:8081 CORE_PROTOCOL=llama-chat cargo run --release
   ```

3. Send an inference request:

   ```bash
   curl -X POST http://127.0.0.1:8080/infer -H "Content-Type: application/json" \
     -d '{"model":"qwen2.5-0.5b-instruct","input":"What is the capital of France?","options":{"max_tokens":32}}'
   ```

   The response contains a real model-generated `output` plus
   `request_id`, `usage.latency_ms` (from the GPU core) and `status`.
   Full verification steps (health, nvidia-smi, pytest) are in
   `docs/run.md` §0.5.

### CPU mode (mock C++ core, legacy)

1. Clone the repository:

```bash
git clone https://github.com/jacklee-x/rust-gpu-inference-gateway.git
cd rust-gpu-inference-gateway
```

2. Build the C++ inference core:

```bash
cd cpp_inference
./build.sh
```

3. Start the C++ inference core in one terminal:

```bash
cd cpp_inference
./run_core.sh
```

or use the helper script (recommended) to start both services:

```bash
# from project root: ./scripts/start-dev.sh [CORE_PORT] [GATEWAY_PORT]
./scripts/start-dev.sh 8081 8080
```

4. Build the Rust gateway (if not using start-dev.sh which builds automatically):

```bash
cargo build --release
```

5. Run the Rust gateway in another terminal (if not using start-dev.sh):

```bash
cargo run --release
```

6. Verify the health endpoint:

```bash
curl http://127.0.0.1:8080/health
```

7. Verify the inference endpoint in PowerShell:

```powershell
Invoke-RestMethod -Uri http://127.0.0.1:8080/infer -Method Post -ContentType "application/json" -Body '{"model":"llama-7b","input":"Hello world","options":{"max_tokens":32}}'
```

Alternatively, use `curl.exe` in PowerShell:

```powershell
curl.exe -X POST http://127.0.0.1:8080/infer -H "Content-Type: application/json" -d '{"model":"llama-7b","input":"Hello world","options":{"max_tokens":32}}'
```

This version uses a queued worker pool and a real C++ inference core. The core exposes `GET /health` and `POST /infer` on port `8081`, and the Rust gateway calls that service before returning the final response. If the queue is full, the gateway returns `503` with `{"status":"queue_full"}`.

8. Send a test inference request with Python:

```powershell
# If the py launcher exists
py -m pip install -r python/requirements.txt
py .\python\demo\client.py --input "Hello from demo"
```

If `py` is not available, use the Python executable directly:

```powershell
python -m pip install -r python/requirements.txt
python .\python\demo\client.py --input "Hello from demo"
```

If neither `py` nor `python` work, install Python from https://www.python.org/downloads/ and make sure it is added to PATH.

9. Measure latency with the benchmark script:

```bash
python python/benchmark/benchmark.py
```

10. Start both services with Docker Compose (CPU fallback path):

```bash
docker compose -f deploy/docker-compose.yml up --build
```

The compose file builds the Rust gateway and the C++ inference core, wires them together (`CORE_URL=http://inference_core:8081`), and adds health checks.

For more detailed instructions, see `docs/run.md`.

## Project Structure

```text
rust-gpu-inference-gateway/
├── README.md
├── Cargo.toml
├── src/
├── cpp_inference/
├── python/
├── deploy/
├── docs/
└── .github/
```

## MVP API Overview

The first version provides a simple inference API:

- `POST /infer`
- `GET /health`
- `GET /metrics`
- `GET /models`

The API is designed to be easy to call from Python clients and to return clear inference results.

## Roadmap

### Phase 1
- [x] Implement Rust gateway with basic endpoints
- [x] Create a worker queue and task scheduler
- [x] Integrate a real C++ inference core with a CUDA-ready path and CPU fallback
- [x] Add Python demo client and benchmark scripts
- [x] Containerize the service with Docker (gateway + C++ core images and Compose)

### Phase 2
- [x] GPU inference support with a real CUDA kernel / model loader (llama.cpp + CUDA, `CORE_PROTOCOL=llama-chat`)
- [x] Model management: `GET /models` and in-memory model registry
- [x] Prometheus-formatted metrics endpoint
- [ ] Expand metrics and tracing (tracing layer, request headers, ...)

### Phase 3
- [x] Add multi-model support (llama-server router mode, dynamic model registry synced from the core; scaling still open)
- [ ] Add Kubernetes deployment examples
- [x] gRPC API and protobuf definitions (`proto/inference.proto` + tonic service on `:50051`)
- [ ] Solana/zk proof-of-concept integration

## Contributing

Contributions are welcome. Please open issues or pull requests for:

- bug fixes
- documentation improvements
- demo workflows
- performance improvements

## Contact

For questions or suggestions, open an issue in the repository.
