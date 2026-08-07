# rust-gpu-inference-gateway

A Rust-based inference gateway with a C++/CUDA GPU inference core and Python tooling for model preparation, demo clients, and benchmarking.

## Core Features

- Rust gateway for HTTP/gRPC inference requests
- Task queue and worker pool for request scheduling
- C++ inference core for GPU-backed model execution
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

1. Clone the repository:

```bash
git clone https://github.com/jacklee-x/rust-gpu-inference-gateway.git
cd rust-gpu-inference-gateway
```

2. Build the Rust gateway:

```bash
cargo build --release
```

3. Run the Rust gateway:

```bash
cargo run --release
```

4. Verify the health endpoint:

```bash
curl http://127.0.0.1:8080/health
```

5. Verify the inference endpoint in PowerShell:

```powershell
Invoke-RestMethod -Uri http://127.0.0.1:8080/infer -Method Post -ContentType "application/json" -Body '{"model":"llama-7b","input":"Hello world","options":{"max_tokens":32}}'
```

Alternatively, use `curl.exe` in PowerShell:

```powershell
curl.exe -X POST http://127.0.0.1:8080/infer -H "Content-Type: application/json" -d '{"model":"llama-7b","input":"Hello world","options":{"max_tokens":32}}'
```

6. Send a test inference request with Python:

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

6. Measure latency with the benchmark script:

```bash
python python/benchmark/benchmark.py
```

7. Note: the `cpp_inference` directory currently contains the planned inference core. The current Rust service returns a stub response for testing the request path.

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

The first version will provide a simple inference API:

- `POST /infer`
- `GET /health`
- `GET /metrics`
- `GET /models`

The API is designed to be easy to call from Python clients and to return clear inference results.

## Roadmap

### Phase 1
- Implement Rust gateway with basic endpoints
- Create a worker queue and task scheduler
- Integrate a minimal C++ inference core stub
- Add Python demo client and benchmark scripts
- Containerize the service with Docker

### Phase 2
- Add GPU inference support
- Add model management and storage abstractions
- Improve observability with Prometheus and tracing

### Phase 3
- Add multi-model support
- Add Docker Compose or Kubernetes deployment examples
- Extend with Solana/zk proof-of-concept integration

## Contributing

Contributions are welcome. Please open issues or pull requests for:

- bug fixes
- documentation improvements
- demo workflows
- performance improvements

## Contact

For questions or suggestions, open an issue in the repository.
