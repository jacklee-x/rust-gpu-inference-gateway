//! Optional OpenTelemetry distributed-tracing support.
//!
//! Export is strictly OPT-IN: spans are shipped over OTLP/HTTP (protobuf)
//! only when `OTEL_EXPORTER_OTLP_ENDPOINT` is set to a non-empty value.
//! Without it the gateway behaves exactly as before (stdout logs only)
//! and never contacts a telemetry backend, so local runs, tests, and CI
//! are unaffected.
//!
//! Spans follow W3C Trace Context: incoming `traceparent` headers start
//! server spans as children, and outgoing calls to the inference core
//! carry a fresh `traceparent` so traces can span gateway -> core.

use opentelemetry::global;
use opentelemetry::propagation::Injector;
use opentelemetry::trace::TracerProvider as _;
use opentelemetry::KeyValue;
use opentelemetry_otlp::SpanExporter;
use opentelemetry_otlp::WithExportConfig;
use opentelemetry_sdk::{
    runtime::Tokio,
    trace::{RandomIdGenerator, Sampler, TracerProvider},
    Resource,
};
use tracing_opentelemetry::OpenTelemetryLayer;
use tracing_subscriber::Registry;

/// Concrete layer type produced by [`init_otel`], so callers can hold it
/// in an Option without boxing.
pub type OtelLayer = OpenTelemetryLayer<Registry, opentelemetry_sdk::trace::Tracer>;

/// Keeps the tracer provider alive for the process lifetime; shutting it
/// down flushes any spans still buffered by the batch exporter.
pub struct TelemetryGuard {
    provider: TracerProvider,
}

impl TelemetryGuard {
    pub fn shutdown(&self) {
        let _ = self.provider.shutdown();
    }
}

/// Build the OpenTelemetry layer plus its lifecycle guard.
///
/// Returns `(None, None)` when `OTEL_EXPORTER_OTLP_ENDPOINT` is unset or
/// empty; prints a warning and degrades to no-export if the exporter
/// cannot be constructed. The stdout logging subscriber stays enabled in
/// every case.
pub fn init_otel() -> (Option<OtelLayer>, Option<TelemetryGuard>) {
    let endpoint = match std::env::var("OTEL_EXPORTER_OTLP_ENDPOINT") {
        Ok(value) => value.trim().to_string(),
        Err(_) => return (None, None),
    };
    if endpoint.is_empty() {
        return (None, None);
    }

    // Standard OTLP/HTTP signal path; tolerate endpoints that already
    // include it.
    let traces_url = if endpoint.ends_with("/v1/traces") {
        endpoint.clone()
    } else {
        format!("{}/v1/traces", endpoint.trim_end_matches('/'))
    };

    let exporter = match SpanExporter::builder()
        .with_http()
        .with_endpoint(traces_url)
        .with_protocol(opentelemetry_otlp::Protocol::HttpBinary)
        .build()
    {
        Ok(exporter) => exporter,
        Err(err) => {
            eprintln!(
                "OpenTelemetry disabled: failed to build OTLP exporter: {}",
                err
            );
            return (None, None);
        }
    };

    let resource = Resource::new(vec![KeyValue::new(
        "service.name",
        "rust-gpu-inference-gateway",
    )]);

    let provider = TracerProvider::builder()
        .with_batch_exporter(exporter, Tokio)
        .with_resource(resource)
        .with_sampler(Sampler::ParentBased(Box::new(Sampler::TraceIdRatioBased(
            1.0,
        ))))
        .with_id_generator(RandomIdGenerator::default())
        .build();

    // Register globally so W3C trace-context extraction/injection works
    // from anywhere, and keep a clone alive for graceful shutdown.
    global::set_tracer_provider(provider.clone());
    global::set_text_map_propagator(opentelemetry_sdk::propagation::TraceContextPropagator::new());

    let tracer = provider.tracer("rust-gpu-inference-gateway");
    (
        Some(tracing_opentelemetry::layer().with_tracer(tracer)),
        Some(TelemetryGuard { provider }),
    )
}

/// Inject the active W3C trace context (`traceparent`/`tracestate`) into
/// outgoing HTTP headers so downstream services continue the same trace.
/// A no-op when no span is active or OpenTelemetry is disabled.
pub(crate) fn inject_trace_context(headers: &mut reqwest::header::HeaderMap) {
    struct ReqwestInjector<'a>(&'a mut reqwest::header::HeaderMap);

    impl Injector for ReqwestInjector<'_> {
        fn set(&mut self, key: &str, value: String) {
            if let (Ok(name), Ok(header_value)) = (
                reqwest::header::HeaderName::from_bytes(key.as_bytes()),
                reqwest::header::HeaderValue::from_str(&value),
            ) {
                self.0.insert(name, header_value);
            }
        }
    }

    global::get_text_map_propagator(|propagator| {
        propagator.inject(&mut ReqwestInjector(headers));
    });
}
