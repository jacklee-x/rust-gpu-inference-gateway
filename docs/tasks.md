# First Version Task Breakdown

## Phase 1: MVP Implementation

### Task 1: Project setup
- Initialize Rust workspace and project structure
- Create `README.md` and documentation skeleton
- Create `docs/architecture.md` and `docs/api.md`

### Task 2: Rust gateway core
- Build a simple Axum HTTP server
- Implement `POST /infer`, `GET /health`, and `GET /metrics`
- Add request validation and JSON serialization
- Add structured logging with `tracing`
- Add configuration support

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
- Add Dockerfile for Rust gateway
- Add Dockerfile for C++ inference core
- Add `deploy/docker-compose.yml` for local startup
- Document startup steps in `README.md`

## Phase 2: GPU and model support

- Replace the C++ stub with a real GPU inference implementation
- Add model loading, preparation, and conversion
- Add a `GET /models` endpoint
- Add model registry and simple storage abstractions
- Expand metrics and tracing

## Phase 3: Extension and polish

- Add multi-model support and worker pool scaling
- Add Docker Compose or Kubernetes deployment examples
- Add optional gRPC API and protobuf definitions
- Add a Solana/zk proof-of-concept integration path
