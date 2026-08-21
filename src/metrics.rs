//! Dependency-free Prometheus-compatible metrics registry.
//!
//! Keeps counters in lock-free atomics and renders them in the
//! Prometheus text exposition format (version 0.0.4) on request.
//! A small mutex-protected histogram records request latency buckets.
//! This is intentionally lightweight: a real deployment may swap this
//! module for the `prometheus` crate or a tracing-layer exporter.

use std::fmt::Write as _;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Mutex;

/// Latency bucket upper bounds in milliseconds (inclusive).
const LATENCY_BUCKETS_MS: [u64; 9] = [1, 5, 10, 25, 50, 100, 250, 500, 1000];

/// Final outcome of a processed inference request.
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Outcome {
    /// Inference completed and a response was returned to the client.
    Ok,
    /// The core reported an error or the worker was cancelled.
    Error,
    /// The gateway gave up waiting for the worker.
    Timeout,
}

/// Process-wide gateway metrics.
#[derive(Default)]
pub struct Metrics {
    requests_total: AtomicU64,
    ok_total: AtomicU64,
    error_total: AtomicU64,
    timeout_total: AtomicU64,
    queue_full_total: AtomicU64,
    validation_errors_total: AtomicU64,
    in_flight: AtomicU64,
    latency_ms_sum: AtomicU64,
    latency_count: AtomicU64,
    latency_histogram: Mutex<[u64; LATENCY_BUCKETS_MS.len()]>,
    // Adaptive worker pool gauges, updated by the dispatcher on resize.
    pool_workers_current: AtomicU64,
    pool_workers_max: AtomicU64,
}

impl Metrics {
    /// Mark an inference request as accepted and in flight.
    pub fn begin_request(&self) {
        self.requests_total.fetch_add(1, Ordering::Relaxed);
        self.in_flight.fetch_add(1, Ordering::Relaxed);
    }

    /// Record the outcome and end-to-end latency (ms) of a request.
    pub fn finish_request(&self, outcome: Outcome, latency_ms: u64) {
        self.in_flight.fetch_sub(1, Ordering::Relaxed);
        match outcome {
            Outcome::Ok => self.ok_total.fetch_add(1, Ordering::Relaxed),
            Outcome::Error => self.error_total.fetch_add(1, Ordering::Relaxed),
            Outcome::Timeout => self.timeout_total.fetch_add(1, Ordering::Relaxed),
        };
        self.latency_ms_sum.fetch_add(latency_ms, Ordering::Relaxed);
        self.latency_count.fetch_add(1, Ordering::Relaxed);

        let mut histogram = self.latency_histogram.lock().unwrap();
        for (bucket, bound) in histogram.iter_mut().zip(LATENCY_BUCKETS_MS.iter()) {
            if latency_ms <= *bound {
                *bucket += 1;
            }
        }
    }

    /// Count a request rejected because the task queue is full.
    pub fn note_queue_full(&self) {
        self.queue_full_total.fetch_add(1, Ordering::Relaxed);
    }

    /// Count a request rejected during validation (unknown model, ...).
    pub fn note_validation_error(&self) {
        self.validation_errors_total.fetch_add(1, Ordering::Relaxed);
    }

    /// Record the current adaptive worker pool size and its upper bound.
    /// Called at startup and whenever the dispatcher resizes the pool.
    pub fn set_pool_workers(&self, current: usize, max: usize) {
        self.pool_workers_current
            .store(current as u64, Ordering::Relaxed);
        self.pool_workers_max.store(max as u64, Ordering::Relaxed);
    }

    /// Render all metrics in the Prometheus text format.
    pub fn render(&self) -> String {
        let mut out = String::new();
        let count = self.latency_count.load(Ordering::Relaxed);
        let histogram = self.latency_histogram.lock().unwrap();

        family(
            &mut out,
            "inference_requests_total",
            "Total inference requests accepted by the gateway.",
            "counter",
        );
        writeln!(
            out,
            "inference_requests_total {}",
            self.requests_total.load(Ordering::Relaxed)
        )
        .unwrap();
        out.push('\n');

        family(
            &mut out,
            "inference_requests_in_flight",
            "Inference requests currently being processed.",
            "gauge",
        );
        writeln!(
            out,
            "inference_requests_in_flight {}",
            self.in_flight.load(Ordering::Relaxed)
        )
        .unwrap();
        out.push('\n');

        family(
            &mut out,
            "inference_worker_pool_size",
            "Current adaptive worker pool size (number of concurrent workers).",
            "gauge",
        );
        writeln!(
            out,
            "inference_worker_pool_size {}",
            self.pool_workers_current.load(Ordering::Relaxed)
        )
        .unwrap();
        family(
            &mut out,
            "inference_worker_pool_max",
            "Maximum allowed worker pool size (MAX_CONCURRENCY).",
            "gauge",
        );
        writeln!(
            out,
            "inference_worker_pool_max {}",
            self.pool_workers_max.load(Ordering::Relaxed)
        )
        .unwrap();
        out.push('\n');

        family(
            &mut out,
            "inference_responses_total",
            "Inference responses by terminal status.",
            "counter",
        );
        for (label, value) in [
            ("ok", self.ok_total.load(Ordering::Relaxed)),
            ("error", self.error_total.load(Ordering::Relaxed)),
            ("timeout", self.timeout_total.load(Ordering::Relaxed)),
            ("queue_full", self.queue_full_total.load(Ordering::Relaxed)),
            (
                "invalid_request",
                self.validation_errors_total.load(Ordering::Relaxed),
            ),
        ] {
            writeln!(
                out,
                "inference_responses_total{{status=\"{}\"}} {}",
                label, value
            )
            .unwrap();
        }
        out.push('\n');

        family(
            &mut out,
            "inference_latency_ms_sum",
            "Total end-to-end latency across completed requests (ms).",
            "counter",
        );
        writeln!(
            out,
            "inference_latency_ms_sum {}",
            self.latency_ms_sum.load(Ordering::Relaxed)
        )
        .unwrap();
        writeln!(out, "inference_latency_ms_count {}", count).unwrap();
        out.push('\n');

        family(
            &mut out,
            "inference_latency_ms_bucket",
            "End-to-end latency distribution (ms).",
            "counter",
        );
        for (bucket, bound) in histogram.iter().zip(LATENCY_BUCKETS_MS.iter()) {
            writeln!(
                out,
                "inference_latency_ms_bucket{{le=\"{}\"}} {}",
                bound, bucket
            )
            .unwrap();
        }
        writeln!(out, "inference_latency_ms_bucket{{le=\"+Inf\"}} {}", count).unwrap();
        out.push('\n');

        out
    }
}

/// Emit a # HELP / # TYPE pair for a metric family.
fn family(out: &mut String, name: &str, help: &str, metric_type: &str) {
    writeln!(out, "# HELP {} {}", name, help).unwrap();
    writeln!(out, "# TYPE {} {}", name, metric_type).unwrap();
}
