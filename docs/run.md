# Run Guide

This page explains how to run the current first-version project and verify its behavior.

## Prerequisites

- Rust toolchain installed (`rustup`, `cargo`)
- Python 3.10+ installed
- `pip` available

## 1. Start the Rust gateway

From the project root:

```bash
cargo run --release
```

The service listens on `127.0.0.1:8080`.

## 2. Verify the health endpoint

Open another terminal and run:

```bash
curl http://127.0.0.1:8080/health
```

Expected output:

```json
{"status":"healthy"}
```

## 3. Verify the inference endpoint

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

## 4. Run the Python demo client

Install dependencies.

### PowerShell / Windows

```powershell
py -m pip install -r python/requirements.txt
```

### macOS / Linux

```bash
python3 -m pip install -r python/requirements.txt
```

Run the demo client from the repository root.

### PowerShell / Windows

```powershell
py .\python\demo\client.py --model llama-7b --input "Hello world"
```

If `py` is not available, use:

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

## 6. What this first version does

At present, the Rust gateway is a working HTTP service. It supports:

- `GET /health`
- `GET /metrics` (placeholder text)
- `POST /infer`

The `POST /infer` endpoint currently returns a stub response. It does not yet perform real GPU inference.

## 7. What is not implemented yet

The following pieces are planned but not yet implemented:

- real C++ GPU inference core
- worker pool with real scheduling
- GPU model loading and execution
- Prometheus-formatted metrics
- `GET /models` endpoint

## 8. Next step

After you verify the current service works, the next development step is to implement the `cpp_inference` core and connect it to the Rust worker.
