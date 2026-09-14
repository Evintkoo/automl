//! Zero autoscaling state and metrics
//!
//! Tracks idle duration, draining status, and exposes Prometheus-format metrics
//! for external autoscalers (e.g. Kubernetes HPA, Railway, Fly.io) to implement
//! scale-to-zero behavior.

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

/// Scaling state tracked via atomics for lock-free access from handlers.
pub struct ScalingState {
    /// Unix timestamp (ms) of the last request
    last_request_at: AtomicU64,
    /// Unix timestamp (ms) when the server started
    started_at: u64,
    /// Total requests served (monotonic counter)
    total_requests: AtomicU64,
    /// Whether the server is currently idle
    is_idle: AtomicBool,
    /// Whether the server is draining (shutting down gracefully)
    draining: AtomicBool,
    /// Seconds of inactivity before the server is considered idle
    idle_timeout_secs: u64,
}

impl ScalingState {
    pub fn new() -> Self {
        let now_ms = Self::now_ms();
        let idle_timeout_secs: u64 = std::env::var("SCALE_IDLE_TIMEOUT_SECS")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(300);

        Self {
            last_request_at: AtomicU64::new(now_ms),
            started_at: now_ms,
            total_requests: AtomicU64::new(0),
            is_idle: AtomicBool::new(false),
            draining: AtomicBool::new(false),
            idle_timeout_secs,
        }
    }

    /// Record that a request was received — resets idle timer.
    pub fn record_request(&self) {
        self.last_request_at.store(Self::now_ms(), Ordering::Relaxed);
        self.total_requests.fetch_add(1, Ordering::Relaxed);
        self.is_idle.store(false, Ordering::Relaxed);
    }

    /// Check whether the idle timeout has elapsed since the last request.
    pub fn check_idle(&self) -> bool {
        let elapsed_ms = Self::now_ms().saturating_sub(self.last_request_at.load(Ordering::Relaxed));
        let idle = elapsed_ms >= self.idle_timeout_secs * 1000;
        self.is_idle.store(idle, Ordering::Relaxed);
        idle
    }

    /// Seconds since the last request.
    pub fn idle_duration_secs(&self) -> f64 {
        let elapsed_ms = Self::now_ms().saturating_sub(self.last_request_at.load(Ordering::Relaxed));
        elapsed_ms as f64 / 1000.0
    }

    /// Seconds since the server started.
    pub fn uptime_secs(&self) -> f64 {
        let elapsed_ms = Self::now_ms().saturating_sub(self.started_at);
        elapsed_ms as f64 / 1000.0
    }

    /// Signal the server to drain (stop accepting new work and wait for inflight to finish).
    pub fn begin_drain(&self) {
        self.draining.store(true, Ordering::SeqCst);
    }

    /// Whether the server is currently draining.
    pub fn is_draining(&self) -> bool {
        self.draining.load(Ordering::SeqCst)
    }

    /// Total requests served.
    pub fn total_requests(&self) -> u64 {
        self.total_requests.load(Ordering::Relaxed)
    }

    fn now_ms() -> u64 {
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as u64
    }
}

/// JSON-serializable scaling status for the `/api/scaling/status` endpoint.
#[derive(serde::Serialize)]
pub struct ScalingStatus {
    pub state: &'static str,
    pub idle_duration_secs: f64,
    pub uptime_secs: f64,
    pub inflight: u64,
    pub queue_depth: usize,
    pub total_requests: u64,
    pub should_scale_to_zero: bool,
}

/// Render Prometheus text-format metrics.
pub fn prometheus_metrics(
    scaling: &ScalingState,
    inflight: u64,
    queue_depth: usize,
    total: u64,
) -> String {
    let idle = scaling.idle_duration_secs();
    let draining = if scaling.is_draining() { 1 } else { 0 };

    format!(
        "# HELP automl_requests_total Total inference requests served.\n\
         # TYPE automl_requests_total counter\n\
         automl_requests_total {total}\n\
         # HELP automl_inflight_requests Currently inflight requests.\n\
         # TYPE automl_inflight_requests gauge\n\
         automl_inflight_requests {inflight}\n\
         # HELP automl_queue_depth Requests waiting in the batch queue.\n\
         # TYPE automl_queue_depth gauge\n\
         automl_queue_depth {queue_depth}\n\
         # HELP automl_idle_duration_seconds Seconds since last request.\n\
         # TYPE automl_idle_duration_seconds gauge\n\
         automl_idle_duration_seconds {idle:.3}\n\
         # HELP automl_draining Whether the server is draining.\n\
         # TYPE automl_draining gauge\n\
         automl_draining {draining}\n\
         # HELP automl_uptime_seconds Server uptime in seconds.\n\
         # TYPE automl_uptime_seconds gauge\n\
         automl_uptime_seconds {uptime:.3}\n",
        total = total,
        inflight = inflight,
        queue_depth = queue_depth,
        idle = idle,
        draining = draining,
        uptime = scaling.uptime_secs(),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_scaling_state_basics() {
        let state = ScalingState::new();
        assert!(!state.is_draining());
        assert_eq!(state.total_requests(), 0);

        state.record_request();
        assert_eq!(state.total_requests(), 1);
        assert!(!state.check_idle()); // just recorded, not idle

        state.begin_drain();
        assert!(state.is_draining());
    }

    #[test]
    fn test_prometheus_metrics_format() {
        let state = ScalingState::new();
        state.record_request();
        let output = prometheus_metrics(&state, 5, 10, 1);
        assert!(output.contains("automl_requests_total 1"));
        assert!(output.contains("automl_inflight_requests 5"));
        assert!(output.contains("automl_queue_depth 10"));
    }
}
