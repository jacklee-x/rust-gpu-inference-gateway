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
use tokio::{
    net::TcpListener,
    sync::{mpsc, oneshot, Semaphore},
    time::{timeout, Duration},
};
use tracing::{debug, error, info};
use tracing_subscriber::EnvFilter;

// Request payload for the /infer endpoint.
#[derive(Serialize, Deserialize, Clone)]
pub struct InferRequest {
    pub model: String,
    pub input: String,
    pub options: Option<InferOptions>,
}

// Optional inference parameters.
#[derive(Serialize, Deserialize, Clone)]
pub struct InferOptions {
    pub max_tokens: Option<u32>,
    pub temperature: Option<f32>,
    pub top_p: Option<f32>,
}

// Response payload returned by the inference endpoint.
#[derive(Serialize, Deserialize, Clone)]
pub struct InferResponse {
    pub request_id: String,
    pub model: String,
    pub output: String,
    pub usage: Usage,
    pub status: String,
}

#[derive(Serialize, Deserialize, Clone)]
pub struct Usage {
    pub latency_ms: u128,
    pub tokens: u32,
}

type JobResult = Result<InferResponse, String>;

type ResponseSender = oneshot::Sender<JobResult>;

struct Job {
    request: InferRequest,
    response_tx: ResponseSender,
}

#[derive(Clone)]
struct AppState {
    job_sender: mpsc::Sender<Job>,
    request_timeout_secs: u64,
}

#[tokio::main]
async fn main() {
    tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::from_default_env())
        .init();

    let max_concurrency = 4usize;
    let queue_capacity = 100usize;
    let request_timeout_secs = 10u64;

    let (job_sender, job_receiver) = mpsc::channel::<Job>(queue_capacity);
    let inference_core_client = Client::new();
    tokio::spawn(dispatcher_loop(
        job_receiver,
        max_concurrency,
        inference_core_client,
    ));

    let state = AppState {
        job_sender,
        request_timeout_secs,
    };

    let app = Router::new()
        .route("/health", get(health_handler))
        .route("/metrics", get(metrics_handler))
        .route("/infer", post(infer_handler))
        .layer(Extension(Arc::new(state)));

    let addr = SocketAddr::from(([127, 0, 0, 1], 8080));
    info!(%addr, "starting inference gateway");
    let listener = TcpListener::bind(addr).await.unwrap();
    axum::serve(listener, app).await.unwrap();
}

async fn dispatcher_loop(
    mut job_receiver: mpsc::Receiver<Job>,
    worker_count: usize,
    client: Client,
) {
    let semaphore = Arc::new(Semaphore::new(worker_count));
    while let Some(job) = job_receiver.recv().await {
        let permit = semaphore.clone().acquire_owned().await.unwrap();
        let client = client.clone();

        tokio::spawn(async move {
            debug!(request_model = %job.request.model, "dispatching inference job to worker");
            let result = call_inference_core(&client, job.request).await;
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

async fn metrics_handler() -> impl IntoResponse {
    // placeholder: replace with Prometheus metrics later
    (StatusCode::OK, "# metrics will be added later\n")
}

async fn infer_handler(
    Extension(state): Extension<Arc<AppState>>,
    Json(payload): Json<InferRequest>,
) -> impl IntoResponse {
    let (response_tx, response_rx) = oneshot::channel();
    let job = Job {
        request: payload.clone(),
        response_tx,
    };

    if let Err(_) = state.job_sender.try_send(job) {
        return (
            StatusCode::SERVICE_UNAVAILABLE,
            Json(serde_json::json!({"status": "queue_full"})),
        );
    }

    let timeout_dur = Duration::from_secs(state.request_timeout_secs);
    match timeout(timeout_dur, response_rx).await {
        Ok(Ok(Ok(infer_resp))) => {
            let v = serde_json::to_value(infer_resp)
                .unwrap_or(serde_json::json!({"status":"ok","output":"serialization_error"}));
            (StatusCode::OK, Json(v))
        }
        Ok(Ok(Err(err_msg))) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(serde_json::json!({"status":"error","message": err_msg})),
        ),
        Ok(Err(_)) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(serde_json::json!({"status":"error","message": "worker task cancelled"})),
        ),
        Err(_) => (
            StatusCode::GATEWAY_TIMEOUT,
            Json(serde_json::json!({"status":"timeout"})),
        ),
    }
}

async fn call_inference_core(client: &Client, req: InferRequest) -> JobResult {
    let core_url = "http://127.0.0.1:8081/infer";
    let response = client
        .post(core_url)
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
