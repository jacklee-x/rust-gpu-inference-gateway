use axum::{extract::Json, http::StatusCode, response::IntoResponse, routing::get, routing::post, Router};
use serde::{Deserialize, Serialize};
use std::net::SocketAddr;
use tokio::net::TcpListener;
use tracing_subscriber::EnvFilter;

#[derive(Deserialize)]
struct InferRequest {
    model: String,
    input: String,
    options: Option<InferOptions>,
}

#[derive(Deserialize)]
struct InferOptions {
    max_tokens: Option<u32>,
    temperature: Option<f32>,
    top_p: Option<f32>,
}

#[derive(Serialize)]
struct InferResponse {
    request_id: String,
    model: String,
    output: String,
    usage: Usage,
    status: String,
}

#[derive(Serialize)]
struct Usage {
    latency_ms: u128,
    tokens: u32,
}

#[tokio::main]
async fn main() {
    tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::from_default_env())
        .init();

    let app = Router::new()
        .route("/health", get(health_handler))
        .route("/metrics", get(metrics_handler))
        .route("/infer", post(infer_handler));

    let addr = SocketAddr::from(([127, 0, 0, 1], 8080));
    tracing::info!(%addr, "starting inference gateway");
    let listener = TcpListener::bind(addr).await.unwrap();
    axum::serve(listener, app).await.unwrap();
}

async fn health_handler() -> impl IntoResponse {
    (StatusCode::OK, Json(serde_json::json!({"status": "healthy"})))
}

async fn metrics_handler() -> impl IntoResponse {
    (StatusCode::OK, "# metrics will be added later\n")
}

async fn infer_handler(Json(payload): Json<InferRequest>) -> impl IntoResponse {
    let response = InferResponse {
        request_id: uuid::Uuid::new_v4().to_string(),
        model: payload.model,
        output: format!("stub response for input: {}", payload.input),
        usage: Usage {
            latency_ms: 0,
            tokens: payload.options.and_then(|opts| opts.max_tokens).unwrap_or(0),
        },
        status: "ok".to_string(),
    };
    (StatusCode::OK, Json(response))
}
