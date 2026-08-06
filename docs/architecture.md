# Architecture

## Overview

This project is a lightweight inference service that combines a Rust gateway with a C++ GPU inference core and Python tooling.

The first version focuses on an end-to-end demo that demonstrates the following:

- Rust async backend for request handling and scheduling
- C++ inference core for GPU-backed model execution
- Python tooling for demo, model preparation, and benchmarking
- Production-style engineering features such as metrics, health checks, and Docker support

## System Components

### Client

A consumer of the inference API. The client sends inference requests to the Rust gateway and receives inference responses.

### Rust Gateway

The Rust gateway is the main entry point for inference traffic. It is responsible for:

- Receiving and validating requests
- Exposing HTTP/gRPC endpoints
- Enqueuing inference tasks
- Managing a worker pool
- Emitting logs and metrics
- Reporting health status

### Scheduler / Queue

A simple asynchronous task queue that dispatches incoming inference requests to configurable worker threads. It is designed as a single-machine queue for the first version.

### Worker

Workers consume tasks from the scheduler and execute inference workloads by invoking the C++ inference core.

Each worker is responsible for:

- Picking tasks from the queue
- Calling the inference core via an FFI, subprocess, or RPC boundary
- Handling execution errors and retries
- Returning inference results

### C++ GPU Inference Core

The C++ component is responsible for actual model execution on GPU. It can be implemented in one of the following ways:

- `llama.cpp` / `ggml` for a fast proof-of-concept
- ONNX Runtime for model inference
- TensorRT for optimized GPU execution

The core should expose a minimal interface for the Rust worker to load models, execute inference, and return outputs.

### Python Tooling

Python scripts provide developer tooling and demo support:

- Demo client to drive the inference API
- Benchmarking and load testing scripts
- Model preparation, conversion, and registry utilities

## Request Flow

1. Client sends a request to `POST /infer`.
2. Rust gateway validates the request and writes it into the task queue.
3. A worker picks up the task and calls the C++ inference core.
4. The inference core runs the model on GPU and returns the result.
5. The worker constructs the final response and sends it back to the gateway.
6. The gateway responds to the client.

## Data Flow

- Request payloads are serialized in JSON or protobuf.
- Inference payloads flow through the Rust gateway and into the worker.
- GPU execution happens in the C++ process or shared library.
- Results are returned to the client and optionally stored for debugging.

## Deployment Model

The initial deployment model is a single host with the following processes:

- Rust gateway process
- C++ inference core process or shared library
- Python tooling executed separately for demos and benchmarks

Future versions may support:

- Docker-based deployment
- Docker Compose for local development
- Kubernetes deployment examples

## Observability

The Rust gateway should expose:

- Health check endpoint (`GET /health`)
- Prometheus metrics endpoint (`GET /metrics`)
- Structured tracing and log output

The Python tooling will help demonstrate request latency and throughput.

## Design Decisions

### Why Rust for the gateway?

Rust provides a strong foundation for safe, high-performance asynchronous services. It is a good fit for the gateway layer because of its ability to handle concurrency, provide low-overhead execution, and integrate with a robust observability stack.

### Why C++ for the inference core?

C++ is the industry standard for GPU inference engines and existing libraries such as TensorRT, ONNX Runtime, and `llama.cpp` are either native or easiest to integrate from C++.

### Why Python tooling?

Python is widely used in model preparation and evaluation. Python tooling will provide easy-to-use demo clients, benchmark scripts, and conversion utilities.

### Why a minimal first version?

A small first version keeps the project achievable and allows us to demonstrate a working end-to-end flow quickly. The first version avoids the complexity of distributed deployment and multi-model orchestration.
