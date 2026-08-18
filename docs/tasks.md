# First Version Task Breakdown

## Phase 1: MVP Implementation

### Task 1: Project setup
- Initialize Rust workspace and project structure
- Create `README.md` and documentation skeleton
- Create `docs/architecture.md` and `docs/api.md`

### Task 2: Rust gateway core
- Build a simple Axum HTTP server
- Implement `POST /infer`, `GET /health`, `GET /metrics`, and `GET /models`
- Add request validation (model registry checks, JSON errors) and JSON serialization
- Add structured logging with `tracing` (per-request `request_id` spans)
- Add configuration support (env vars: `BIND_ADDR` / `PORT`, `CORE_URL`, `MAX_CONCURRENCY`, `QUEUE_CAPACITY`, `REQUEST_TIMEOUT_SECS`)

### Task 3: Task queue and worker
- Implement an asynchronous task queue using `tokio::mpsc`
- Add a worker pool for request execution
- Support request timeout and error handling
- Make the worker interface pluggable for the inference core

### Task 4: C++ inference core
- Create C++ project scaffolding in `cpp_inference/`
- Implement an HTTP-based C++ core that exposes `/health` and `/infer`
- Add a CUDA-enabled execution branch and a CPU fallback for local development
- Integrate Rust with the C++ service via a simple HTTP RPC boundary
- Verify end-to-end Rust -> C++ -> Rust flow

### Task 5: Python demo and benchmark
- Add `python/demo/client.py` to send inference requests
- Add `python/benchmark/benchmark.py` to measure latency
- Add `python/tools/generate_input.py` for synthetic inputs
- Add `python/tests/test_end_to_end.py` for basic validation

### Task 6: Containerization and deployment
- Add Dockerfile for Rust gateway (`deploy/Dockerfile.gateway`)
- Add Dockerfile for C++ inference core (`cpp_inference/Dockerfile`)
- Add `deploy/docker-compose.yml` for local startup (healthchecks, dependency wiring)
- Document startup steps in `README.md`

## Phase 2: GPU and model support

- [x] Replace the C++ stub with a real GPU inference implementation (llama.cpp + CUDA via `CORE_PROTOCOL=llama-chat`)
- [x] Add `GET /models` endpoint
- [x] Add model registry abstraction (`src/models.rs`, in-memory)
- [x] Expand metrics (Prometheus text format via `src/metrics.rs`) and tracing (request_id spans)

## Phase 3: Extension and polish

- [x] Multi-model support (llama-server router mode + dynamic registry synced from `GET /v1/models`)
- [x] Worker pool dynamic scaling (adaptive `MIN_CONCURRENCY`..`MAX_CONCURRENCY`, resized from queue depth, gauges in `/metrics`)
- Add Docker Compose or Kubernetes deployment examples
- Add optional gRPC API and protobuf definitions
- Add a Solana/zk proof-of-concept integration path
