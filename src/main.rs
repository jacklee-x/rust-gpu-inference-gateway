use axum::{extract::Json, http::StatusCode, response::IntoResponse, routing::{get, post}, Router, Extension};
use serde::{Deserialize, Serialize};
use std::net::SocketAddr;
use std::sync::Arc;
use tokio::{net::TcpListener, sync::Semaphore, time::{timeout, Duration}};
use tracing_subscriber::EnvFilter;

#[derive(Deserialize, Clone)]
pub struct InferRequest {
    pub model: String,
    pub input: String,
    pub options: Option<InferOptions>,
}

#[derive(Deserialize, Clone)]
pub struct InferOptions {
    pub max_tokens: Option<u32>,
    pub temperature: Option<f32>,
    pub top_p: Option<f32>,
}

#[derive(Serialize, Clone)]
pub struct InferResponse {
    pub request_id: String,
    pub model: String,
    pub output: String,
    pub usage: Usage,
    pub status: String,
}

#[derive(Serialize, Clone)]
pub struct Usage {
    pub latency_ms: u128,
    pub tokens: u32,
}

#[derive(Clone)]
struct AppState {
    // limit concurrent inferences (acts like a worker pool size)
    concurrency_limit: Arc<Semaphore>,
    // request timeout seconds
    request_timeout_secs: u64,
}

#[tokio::main]
async fn main() {
    tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::from_default_env())
        .init();

    // configuration (simple for MVP)
    let max_concurrency = 4usize; // adjust as needed
    let request_timeout_secs = 10u64;

    let state = AppState {
        concurrency_limit: Arc::new(Semaphore::new(max_concurrency)),
        request_timeout_secs,
    };

    let app = Router::new()
        .route("/health", get(health_handler))
        .route("/metrics", get(metrics_handler))
        .route("/infer", post(infer_handler))
        .layer(Extension(Arc::new(state)));

    let addr = SocketAddr::from(([127, 0, 0, 1], 8080));
    tracing::info!(%addr, "starting inference gateway");
    let listener = TcpListener::bind(addr).await.unwrap();
    axum::serve(listener, app).await.unwrap();
}

async fn health_handler() -> impl IntoResponse {
    (StatusCode::OK, Json(serde_json::json!({"status": "healthy"})))
}

async fn metrics_handler() -> impl IntoResponse {
    // placeholder: replace with Prometheus metrics later
    (StatusCode::OK, "# metrics will be added later\n")
}

async fn infer_handler(Extension(state): Extension<Arc<AppState>>, Json(payload): Json<InferRequest>) -> impl IntoResponse {
    // Try to acquire a permit immediately to limit concurrency
    match state.concurrency_limit.clone().try_acquire_owned() {
        Ok(permit) => {
            // We got a permit — run the inference with timeout
            let timeout_dur = Duration::from_secs(state.request_timeout_secs);
            let fut = call_inference_stub(payload.clone());
            match timeout(timeout_dur, fut).await {
                Ok(Ok(infer_resp)) => {
                    // permit drops here when goes out of scope
                    drop(permit);
                    // unify response type as JSON Value
                    let v = serde_json::to_value(infer_resp).unwrap_or(serde_json::json!({"status":"ok","output":"serialization_error"}));
                    (StatusCode::OK, Json(v))
                }
                Ok(Err(e)) => {
                    drop(permit);
                    (StatusCode::INTERNAL_SERVER_ERROR, Json(serde_json::json!({"status":"error","message": e})))
                }
                Err(_) => {
                    // timeout
                    drop(permit);
                    (StatusCode::GATEWAY_TIMEOUT, Json(serde_json::json!({"status":"timeout"})))
                }
            }
        }
        Err(_) => {
            // no permits available -> return 503 to apply backpressure
            (StatusCode::SERVICE_UNAVAILABLE, Json(serde_json::json!({"status":"queue_full"})))
        }
    }
}

// Minimal async stub for inference call — replace with RPC/FFI to C++ core later
async fn call_inference_stub(req: InferRequest) -> Result<InferResponse, String> {
    // simulate some processing delay
    let sleep_ms = 50u64;
    tokio::time::sleep(Duration::from_millis(sleep_ms)).await;

    let resp = InferResponse {
        request_id: uuid::Uuid::new_v4().to_string(),
        model: req.model,
        output: format!("inferred (stub): {}", req.input),
        usage: Usage { latency_ms: sleep_ms as u128, tokens: req.options.and_then(|o| o.max_tokens).unwrap_or(0) },
        status: "ok".to_string(),
    };
    Ok(resp)
}
