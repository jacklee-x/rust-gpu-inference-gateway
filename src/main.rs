use axum::{
    extract::Json,
    http::StatusCode,
    response::IntoResponse,
    routing::{get, post},
    Extension, Router,
};
use reqwest::Client;
use serde::{Deserialize, Serialize};
use std::net::SocketAddr;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::{
    net::TcpListener,
    sync::{mpsc, oneshot, Semaphore},
    time::timeout,
};
use tracing::{debug, error, info, info_span, warn};
use tracing_subscriber::EnvFilter;
use uuid::Uuid;

mod metrics;
mod models;

use metrics::{Metrics, Outcome};
use models::ModelRegistry;

// Request and response payload types for the HTTP API.
// These structs are serialized/deserialized as JSON by axum/serde.
// Keep these stable: they form the wire contract between clients, the
// Rust gateway, and the C++ inference core.

/// Client-facing request body for POST /infer
/// - model: logical model name (validated against the model registry)
/// - input: input text to run inference on
/// - options: optional inference tuning parameters
/// - request_id: optional correlation id; the gateway stamps one when
///   absent so every request is traceable end to end.
#[derive(Serialize, Deserialize, Clone)]
pub struct InferRequest {
    pub model: String,
    pub input: String,
    pub options: Option<InferOptions>,
    #[serde(default)]
    pub request_id: Option<String>,
}

/// Optional parameters to control inference behavior.
/// Not all fields are used by the simple prototype core; they are
/// included for forward-compatibility with a real inference engine.
#[derive(Serialize, Deserialize, Clone)]
pub struct InferOptions {
    pub max_tokens: Option<u32>,
    pub temperature: Option<f32>,
    pub top_p: Option<f32>,
}

/// Gateway / core response shown to clients.
/// - request_id: correlation id assigned by the gateway
/// - output: textual result from the inference engine
/// - usage: lightweight usage information (latency, token counts)
/// - status: textual status ("ok", "error", ...)
#[derive(Serialize, Deserialize, Clone)]
pub struct InferResponse {
    pub request_id: String,
    pub model: String,
    pub output: String,
    pub usage: Usage,
    pub status: String,
}

/// Minimal usage metrics attached to responses. This is intentionally
/// small and human-readable for the demo phase.
#[derive(Serialize, Deserialize, Clone)]
pub struct Usage {
    pub latency_ms: u128,
    pub tokens: u32,
}

// A JobResult is the worker's return type: either a successful
// InferResponse or a String describing the error.
type JobResult = Result<InferResponse, String>;

// ResponseSender is the oneshot channel used by a worker to deliver
// the result back to the request-handling task.
type ResponseSender = oneshot::Sender<JobResult>;

// Job captures a single inference request plus the channel for the
// worker to send back the result. Jobs are enqueued into a bounded
// mpsc channel to provide simple backpressure at the gateway.
struct Job {
    request: InferRequest,
    response_tx: ResponseSender,
}

// AppState is shared (Arc) into handlers via axum's Extension layer.
// It contains the sending side of the job queue, runtime configuration,
// the metrics registry, and the model registry.
#[derive(Clone)]
struct AppState {
    job_sender: mpsc::Sender<Job>,
    request_timeout_secs: u64,
    metrics: Arc<Metrics>,
    model_registry: Arc<ModelRegistry>,
}

// Runtime configuration, all of it overridable via environment variables
// so the same binary works locally, in Docker Compose, and in CI.
struct Config {
    bind_addr: SocketAddr,
    core_url: String,
    core_protocol: CoreProtocol,
    max_concurrency: usize,
    queue_capacity: usize,
    request_timeout_secs: u64,
}

// Selects the wire protocol used to talk to the inference core.
// - Infer: legacy custom JSON wire format (POST {core}/infer), used by
//   the mock C++ core and the Python mock server.
// - LlamaChat: llama.cpp llama-server OpenAI-compatible API
//   (POST {core}/v1/chat/completions) for real GPU inference.
#[derive(Clone, Copy, PartialEq, Eq)]
enum CoreProtocol {
    Infer,
    LlamaChat,
}

impl CoreProtocol {
    // Reads CORE_PROTOCOL from the environment. Accepts several aliases
    // for the llama-server API; anything unrecognized falls back to the
    // legacy "infer" protocol so existing deployments keep working.
    fn from_env() -> Self {
        match std::env::var("CORE_PROTOCOL")
            .unwrap_or_default()
            .trim()
            .to_ascii_lowercase()
            .as_str()
        {
            "llama-chat" | "llama" | "openai" => CoreProtocol::LlamaChat,
            _ => CoreProtocol::Infer,
        }
    }

    fn as_str(&self) -> &'static str {
        match self {
            CoreProtocol::Infer => "infer",
            CoreProtocol::LlamaChat => "llama-chat",
        }
    }
}

impl Config {
    fn from_env() -> Self {
        let bind_addr = std::env::var("BIND_ADDR")
            .ok()
            .or_else(|| {
                std::env::var("PORT")
                    .ok()
                    .map(|port| format!("127.0.0.1:{}", port))
            })
            .unwrap_or_else(|| "127.0.0.1:8080".to_string());

        let bind_addr: SocketAddr = bind_addr.parse().unwrap_or_else(|_| {
            eprintln!("Invalid BIND_ADDR/PORT: {}", bind_addr);
            std::process::exit(1);
        });

        let core_url =
            std::env::var("CORE_URL").unwrap_or_else(|_| "http://127.0.0.1:8081".to_string());

        let core_protocol = CoreProtocol::from_env();

        let max_concurrency = parse_env_usize("MAX_CONCURRENCY", 4);
        let queue_capacity = parse_env_usize("QUEUE_CAPACITY", 100);
        let request_timeout_secs = parse_env_u64("REQUEST_TIMEOUT_SECS", 10);

        Config {
            bind_addr,
            core_url,
            core_protocol,
            max_concurrency,
            queue_capacity,
            request_timeout_secs,
        }
    }
}

fn parse_env_usize(name: &str, default: usize) -> usize {
    match std::env::var(name) {
        Ok(value) => match value.trim().parse::<usize>() {
            Ok(parsed) if parsed > 0 => parsed,
            _ => {
                warn!(var = name, value = %value, %default, "invalid value, using default");
                default
            }
        },
        Err(_) => default,
    }
}

fn parse_env_u64(name: &str, default: u64) -> u64 {
    match std::env::var(name) {
        Ok(value) => match value.trim().parse::<u64>() {
            Ok(parsed) if parsed > 0 => parsed,
            _ => {
                warn!(var = name, value = %value, %default, "invalid value, using default");
                default
            }
        },
        Err(_) => default,
    }
}

// The entry point: initializes tracing, the task queue and dispatcher,
// and the axum HTTP router. The function is annotated with
// #[tokio::main] to provide an async runtime for the server.
#[tokio::main]
async fn main() {
    tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::from_default_env())
        .init();

    let config = Config::from_env();

    // Create a bounded channel used as the task queue. The queue size
    // (queue_capacity) provides simple backpressure: when full, try_send
    // will fail and the HTTP handler will return 503 to the client.
    let (job_sender, job_receiver) = mpsc::channel::<Job>(config.queue_capacity);

    // Single HTTP client shared by workers to call the inference core
    // over a local RPC boundary. Cloning reqwest::Client is cheap.
    let inference_core_client = Client::new();

    let metrics = Arc::new(Metrics::default());
    let model_registry = Arc::new(ModelRegistry::default());

    // Spawn the dispatcher loop in the background; it will accept jobs
    // from the queue and spawn worker tasks up to max_concurrency.
    tokio::spawn(dispatcher_loop(
        job_receiver,
        config.max_concurrency,
        inference_core_client,
        config.core_url.clone(),
        config.core_protocol,
    ));

    let state = AppState {
        job_sender,
        request_timeout_secs: config.request_timeout_secs,
        metrics,
        model_registry,
    };

    let app = Router::new()
        .route("/health", get(health_handler))
        .route("/metrics", get(metrics_handler))
        .route("/models", get(models_handler))
        .route("/infer", post(infer_handler))
        .layer(Extension(Arc::new(state)));

    let addr = config.bind_addr;
    info!(
        %addr,
        core_url = %config.core_url,
        core_protocol = config.core_protocol.as_str(),
        "starting inference gateway"
    );

    let listener = match TcpListener::bind(addr).await {
        Ok(listener) => listener,
        Err(err) => {
            // More friendly error message than panic
            if err.kind() == std::io::ErrorKind::AddrInUse {
                eprintln!(
                    "Failed to bind to {}: address already in use (is another instance running?).",
                    addr
                );
                eprintln!(
                    "On Windows: run `Get-NetTCPConnection -LocalPort {}` or use Task Manager to stop the process. On Linux: `ss -ltnp | grep {}`.",
                    addr.port(),
                    addr.port()
                );
            } else {
                eprintln!("Failed to bind to {}: {}", addr, err);
            }
            std::process::exit(1);
        }
    };

    if let Err(err) = axum::serve(listener, app).await {
        error!("server error: {}", err);
        std::process::exit(1);
    }
}

// dispatcher_loop runs indefinitely consuming jobs from the queue.
// Implementation notes:
// - A semaphore is used to cap the number of concurrently-running
//   worker tasks (worker_count). This behaves like a worker pool.
// - For each job, a worker task is spawned that calls the inference core
//   and forwards the result back to the original requester via a
//   oneshot channel.
async fn dispatcher_loop(
    mut job_receiver: mpsc::Receiver<Job>,
    worker_count: usize,
    client: Client,
    core_url: String,
    core_protocol: CoreProtocol,
) {
    let semaphore = Arc::new(Semaphore::new(worker_count));
    while let Some(job) = job_receiver.recv().await {
        let permit = semaphore.clone().acquire_owned().await.unwrap();
        let client = client.clone();
        let core_url = core_url.clone();

        tokio::spawn(async move {
            debug!(
                request_id = %job.request.request_id.as_deref().unwrap_or("unknown"),
                request_model = %job.request.model,
                "dispatching inference job to worker"
            );
            let result = call_inference_core(&client, &core_url, core_protocol, job.request).await;
            if job.response_tx.send(result).is_err() {
                error!("client response channel dropped before worker completed");
            }
            drop(permit);
        });
    }
}

async fn health_handler() -> impl IntoResponse {
    (
        StatusCode::OK,
        Json(serde_json::json!({"status": "healthy"})),
    )
}

// Prometheus text exposition format (version 0.0.4), served without
// external dependencies by the metrics module.
async fn metrics_handler(Extension(state): Extension<Arc<AppState>>) -> impl IntoResponse {
    (
        StatusCode::OK,
        [(
            axum::http::header::CONTENT_TYPE,
            "text/plain; version=0.0.4; charset=utf-8",
        )],
        state.metrics.render(),
    )
}

// GET /models: list the models known to the gateway.
async fn models_handler(Extension(state): Extension<Arc<AppState>>) -> impl IntoResponse {
    (
        StatusCode::OK,
        Json(serde_json::json!({ "models": state.model_registry.list() })),
    )
}

// HTTP handler for POST /infer
// Responsibilities:
// 1. Validate the request against the model registry
// 2. Assign a traceable request_id and stamp it on the outgoing payload
// 3. Create a oneshot channel for the worker to send back the result
// 4. Try to enqueue the Job. If the bounded queue is full, return 503 quickly
// 5. Wait for the worker's response with a timeout; map outcomes to HTTP codes
// 6. Record Prometheus metrics for every accepted request
async fn infer_handler(
    Extension(state): Extension<Arc<AppState>>,
    Json(mut payload): Json<InferRequest>,
) -> impl IntoResponse {
    if !state.model_registry.contains(&payload.model) {
        state.metrics.note_validation_error();
        warn!(model = %payload.model, "unknown model in request");
        return (
            StatusCode::BAD_REQUEST,
            Json(serde_json::json!({
                "status": "error",
                "message": format!("unknown model '{}'", payload.model)
            })),
        );
    }

    let request_id = Uuid::new_v4().to_string();
    payload.request_id = Some(request_id.clone());
    let span = info_span!("infer_request", request_id = %request_id, model = %payload.model);
    let _enter = span.enter();
    info!("inference request received");

    let (response_tx, response_rx) = oneshot::channel();
    let job = Job {
        request: payload,
        response_tx,
    };

    if state.job_sender.try_send(job).is_err() {
        state.metrics.note_queue_full();
        warn!(%request_id, "queue full, rejecting request");
        return (
            StatusCode::SERVICE_UNAVAILABLE,
            Json(serde_json::json!({"status": "queue_full"})),
        );
    }

    let started = Instant::now();
    state.metrics.begin_request();

    let timeout_dur = Duration::from_secs(state.request_timeout_secs);
    let response = timeout(timeout_dur, response_rx).await;
    let elapsed_ms = started.elapsed().as_millis() as u64;

    match response {
        Ok(Ok(Ok(mut infer_resp))) => {
            infer_resp.request_id = request_id.clone();
            state.metrics.finish_request(Outcome::Ok, elapsed_ms);
            info!(
                %request_id,
                core_latency_ms = %infer_resp.usage.latency_ms,
                "inference completed"
            );
            let value = serde_json::to_value(infer_resp)
                .unwrap_or(serde_json::json!({"status":"ok","output":"serialization_error"}));
            (StatusCode::OK, Json(value))
        }
        Ok(Ok(Err(err_msg))) => {
            state.metrics.finish_request(Outcome::Error, elapsed_ms);
            error!(%request_id, %err_msg, "inference failed");
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(serde_json::json!({"status":"error","message": err_msg})),
            )
        }
        Ok(Err(_)) => {
            state.metrics.finish_request(Outcome::Error, elapsed_ms);
            error!(%request_id, "worker task cancelled");
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(serde_json::json!({"status":"error","message": "worker task cancelled"})),
            )
        }
        Err(_) => {
            state
                .metrics
                .finish_request(Outcome::Timeout, timeout_dur.as_millis() as u64);
            warn!(%request_id, timeout_secs = %state.request_timeout_secs, "inference timed out");
            (
                StatusCode::GATEWAY_TIMEOUT,
                Json(serde_json::json!({"status":"timeout"})),
            )
        }
    }
}

// call_inference_core performs the synchronous (from the worker's
// perspective) network call to the external inference core. The function
// returns a JobResult which is either the parsed InferResponse or a
// String describing the error. The core address is configurable
// (CORE_URL) and defaults to the local C++ service on 127.0.0.1:8081;
// in later iterations this could be a UNIX socket, shared-memory/FFI
// call, or gRPC endpoint. The wire protocol is selected by
// CORE_PROTOCOL ("infer" for the legacy mock core, "llama-chat" for a
// llama.cpp llama-server instance).
async fn call_inference_core(
    client: &Client,
    core_url: &str,
    protocol: CoreProtocol,
    req: InferRequest,
) -> JobResult {
    match protocol {
        CoreProtocol::Infer => call_infer_protocol(client, core_url, req).await,
        CoreProtocol::LlamaChat => call_llama_chat_protocol(client, core_url, req).await,
    }
}

// Legacy custom wire format: POST {core}/infer with the full
// InferRequest JSON, expecting an InferResponse JSON in return.
async fn call_infer_protocol(client: &Client, core_url: &str, req: InferRequest) -> JobResult {
    let infer_url = format!("{}/infer", core_url.trim_end_matches('/'));
    let response = client
        .post(&infer_url)
        .json(&req)
        .send()
        .await
        .map_err(|err| format!("rpc error: {}", err))?;

    if !response.status().is_success() {
        let status = response.status();
        let text = response
            .text()
            .await
            .unwrap_or_else(|_| "no body".to_string());
        return Err(format!("core returned {}: {}", status, text));
    }

    response
        .json::<InferResponse>()
        .await
        .map_err(|err| format!("failed to decode core response: {}", err))
}

// llama.cpp llama-server OpenAI-compatible format:
// POST {core}/v1/chat/completions, expecting choices[0].message.content
// plus usage and timings for latency/token reporting.
async fn call_llama_chat_protocol(
    client: &Client,
    core_url: &str,
    req: InferRequest,
) -> JobResult {
    let chat_url = format!("{}/v1/chat/completions", core_url.trim_end_matches('/'));

    let mut body = serde_json::json!({
        "model": req.model,
        "messages": [{"role": "user", "content": req.input}],
    });
    if let Some(opts) = &req.options {
        if let Some(max_tokens) = opts.max_tokens {
            body["max_tokens"] = serde_json::json!(max_tokens);
        }
        if let Some(temperature) = opts.temperature {
            body["temperature"] = serde_json::json!(temperature);
        }
        if let Some(top_p) = opts.top_p {
            body["top_p"] = serde_json::json!(top_p);
        }
    }

    let response = client
        .post(&chat_url)
        .json(&body)
        .send()
        .await
        .map_err(|err| format!("rpc error: {}", err))?;

    if !response.status().is_success() {
        let status = response.status();
        let text = response
            .text()
            .await
            .unwrap_or_else(|_| "no body".to_string());
        return Err(format!("core returned {}: {}", status, text));
    }

    let parsed: LlamaChatResponse = response
        .json()
        .await
        .map_err(|err| format!("failed to decode llama-server response: {}", err))?;

    let output = parsed
        .choices
        .first()
        .map(|choice| choice.message.content.clone())
        .unwrap_or_default();

    let tokens = parsed
        .usage
        .as_ref()
        .and_then(|usage| usage.completion_tokens.or(usage.total_tokens))
        .unwrap_or(0);

    // llama-server reports generation latency in ms as timings.predicted_ms.
    let latency_ms = parsed
        .timings
        .as_ref()
        .and_then(|timings| timings.predicted_ms.or(timings.total_ms))
        .unwrap_or(0.0) as u128;

    Ok(InferResponse {
        request_id: req.request_id.unwrap_or_default(),
        model: req.model,
        output,
        usage: Usage {
            latency_ms,
            tokens,
        },
        status: "ok".to_string(),
    })
}

// Minimal deserialization targets for the llama-server OpenAI-compatible
// chat completions response. Only the fields the gateway reuses are
// declared; the rest of the JSON is ignored.
#[derive(Deserialize)]
struct LlamaChatResponse {
    choices: Vec<LlamaChatChoice>,
    usage: Option<LlamaChatUsage>,
    timings: Option<LlamaChatTimings>,
}

#[derive(Deserialize)]
struct LlamaChatChoice {
    message: LlamaChatMessage,
}

#[derive(Deserialize)]
struct LlamaChatMessage {
    content: String,
}

#[derive(Deserialize)]
struct LlamaChatUsage {
    completion_tokens: Option<u32>,
    total_tokens: Option<u32>,
}

#[derive(Deserialize)]
struct LlamaChatTimings {
    predicted_ms: Option<f64>,
    total_ms: Option<f64>,
}
