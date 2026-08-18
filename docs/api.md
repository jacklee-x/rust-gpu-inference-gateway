# API Design

This document describes the MVP API for the `rust-gpu-inference-gateway` project.

## REST API

### POST /infer

Submit an inference request.

#### Request

```json
{
  "model": "qwen2.5-0.5b-instruct",
  "input": "Hello world",
  "options": {
    "max_tokens": 32,
    "temperature": 0.8,
    "top_p": 0.95
  }
}
```

#### Response

```json
{
  "request_id": "uuid",
  "model": "qwen2.5-0.5b-instruct",
  "output": "Hello! How can I help you today?",
  "usage": {
    "latency_ms": 301,
    "tokens": 32
  },
  "status": "ok"
}
```

#### Notes

- `model` identifies the model to use for inference and is validated against the model registry. Unknown models return `400` with `{"status":"error","message":"unknown model '...'"}`.
- `input` carries the prompt or input payload.
- `options` may be extended over time.
- `request_id` is optional: when absent, the gateway generates a UUID and stamps it on the request. The final response always carries the gateway-assigned `request_id`.
- This endpoint should return a status code of `200` when the request is accepted and processed successfully.

### GET /health

Simple health check endpoint.

#### Response

```json
{ "status": "healthy" }
```

### GET /metrics

Prometheus-compatible metrics endpoint (text exposition format `0.0.4`) used by observability tooling.

Metrics exposed:

- `inference_requests_total` — accepted requests (counter)
- `inference_requests_in_flight` — currently processing (gauge)
- `inference_responses_total{status="ok|error|timeout|queue_full|invalid_request"}` (counter)
- `inference_latency_ms_sum` / `inference_latency_ms_count` (counter)
- `inference_latency_ms_bucket{le="..."}` — latency histogram (counter)

### GET /models

List the models that are currently loaded or available. In `llama-chat`
mode this list is synced from the inference core (`GET /v1/models` of
llama-server) every `REGISTRY_REFRESH_SECS` seconds, so it always
reflects the real models served, with live `status` values
(`loaded` / `unloaded` / `failed`, ...). In the legacy `infer` mode a
static in-memory snapshot is returned.

#### Response (llama-chat mode, router with two models)

```json
{
  "models": [
    { "name": "qwen2.5-0.5b-instruct-q4_k_m", "status": "loaded", "device": "gpu0", "backend": "llama.cpp-cuda" },
    { "name": "qwen2.5-1.5b-instruct-q4_k_m", "status": "loaded", "device": "gpu0", "backend": "llama.cpp-cuda" }
  ]
}
```

### GET /requests/{id}

Optional endpoint to query the status of an inference request when asynchronous processing is implemented.

#### Response

```json
{
  "request_id": "uuid",
  "status": "completed",
  "output": "...",
  "latency_ms": 123
}
```

## gRPC API (Optional Future Work)

The first version will use a REST interface, with gRPC added later if needed.

### Service Definition

```protobuf
syntax = "proto3";

package inference;

service InferService {
  rpc Infer(InferRequest) returns (InferResponse);
  rpc Health(HealthRequest) returns (HealthResponse);
  rpc ListModels(ListModelsRequest) returns (ListModelsResponse);
}

message InferRequest {
  string model = 1;
  string input = 2;
  InferOptions options = 3;
}

message InferOptions {
  int32 max_tokens = 1;
  float temperature = 2;
  float top_p = 3;
}

message InferResponse {
  string request_id = 1;
  string model = 2;
  string output = 3;
  Usage usage = 4;
  string status = 5;
}

message Usage {
  int32 latency_ms = 1;
  int32 tokens = 2;
}

message HealthRequest {}
message HealthResponse { string status = 1; }
message ListModelsRequest {}
message ListModelsResponse {
  repeated ModelInfo models = 1;
}

message ModelInfo {
  string name = 1;
  string status = 2;
  string device = 3;
}
```

## API Design Principles

- Keep the MVP simple and easy to call from Python.
- Use JSON for the REST API to lower friction for demo and benchmark tooling.
- Provide clear observability through health and metrics endpoints.
- Design the API so it can be extended later with async request handling, batching, and gRPC.
