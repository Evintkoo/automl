//! System Monitoring
//!
//! Collects system-level metrics such as CPU, memory, and disk I/O.

use parking_lot::RwLock;
use serde::{Serialize, Deserialize};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use sysinfo::System;

/// A snapshot of system metrics at a point in time
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SystemMetrics {
    /// Timestamp when metrics were collected (seconds since UNIX epoch)
    pub timestamp: f64,
    /// Overall CPU usage percentage (0–100)
    pub cpu_usage_pct: f64,
    /// Memory usage percentage (0–100)
    pub memory_usage_pct: f64,
    /// Memory currently used in megabytes
    pub memory_used_mb: f64,
    /// Memory available in megabytes
    pub memory_available_mb: f64,
    /// Cumulative disk bytes read
    pub disk_read_bytes: u64,
    /// Cumulative disk bytes written
    pub disk_write_bytes: u64,
}

/// Monitors system resources and maintains a history of metrics snapshots
pub struct SystemMonitor {
    /// Collection interval in seconds
    pub interval_secs: f64,
    /// Flag to control collection loop
    pub running: Arc<AtomicBool>,
    /// History of collected metrics
    pub metrics_history: Arc<RwLock<Vec<SystemMetrics>>>,
}

impl SystemMonitor {
    /// Maximum number of metrics snapshots retained in history before the
    /// oldest entries are evicted, to prevent unbounded memory growth when
    /// `collect_and_store` is called on every tick for the process lifetime.
    const MAX_METRICS_HISTORY: usize = 1000;

    /// Create a new SystemMonitor with the given collection interval
    pub fn new(interval_secs: f64) -> Self {
        Self {
            interval_secs,
            running: Arc::new(AtomicBool::new(false)),
            metrics_history: Arc::new(RwLock::new(Vec::new())),
        }
    }

    /// Collect a single snapshot of system metrics
    pub fn collect_metrics() -> SystemMetrics {
        let mut sys = System::new_all();
        sys.refresh_all();

        let total_memory = sys.total_memory() as f64;
        let used_memory = sys.used_memory() as f64;
        let available_memory = sys.available_memory() as f64;
        let memory_usage_pct = if total_memory > 0.0 {
            (used_memory / total_memory) * 100.0
        } else {
            0.0
        };

        let cpu_usage = if sys.cpus().is_empty() {
            0.0
        } else {
            sys.cpus().iter().map(|c| c.cpu_usage()).sum::<f32>() as f64
                / sys.cpus().len() as f64
        };

        // Aggregate disk I/O from process-level stats
        let mut disk_read_bytes: u64 = 0;
        let mut disk_write_bytes: u64 = 0;

        let processes = sys.processes();
        for (_pid, process) in processes {
            let disk_usage = process.disk_usage();
            disk_read_bytes = disk_read_bytes.saturating_add(disk_usage.read_bytes);
            disk_write_bytes = disk_write_bytes.saturating_add(disk_usage.written_bytes);
        }

        let timestamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs_f64();

        SystemMetrics {
            timestamp,
            cpu_usage_pct: cpu_usage,
            memory_usage_pct,
            memory_used_mb: used_memory / (1024.0 * 1024.0),
            memory_available_mb: available_memory / (1024.0 * 1024.0),
            disk_read_bytes,
            disk_write_bytes,
        }
    }

    /// Collect metrics and store them in history
    pub fn collect_and_store(&self) -> SystemMetrics {
        let metrics = Self::collect_metrics();
        {
            let mut history = self.metrics_history.write();
            history.push(metrics.clone());
            if history.len() > Self::MAX_METRICS_HISTORY {
                let excess = history.len() - Self::MAX_METRICS_HISTORY;
                history.drain(0..excess);
            }
        }
        metrics
    }

    /// Get the most recently collected metrics snapshot
    pub fn get_latest_metrics(&self) -> Option<SystemMetrics> {
        self.metrics_history.read().last().cloned()
    }

    /// Get the last N metrics snapshots
    pub fn get_metrics_history(&self, last_n: usize) -> Vec<SystemMetrics> {
        let history = self.metrics_history.read();
        let start = history.len().saturating_sub(last_n);
        history[start..].to_vec()
    }

    /// Mark the monitor as running
    pub fn start(&self) {
        self.running.store(true, Ordering::Relaxed);
    }

    /// Signal the monitor to stop
    pub fn stop(&self) {
        self.running.store(false, Ordering::Relaxed);
    }

    /// Check if the monitor is running
    pub fn is_running(&self) -> bool {
        self.running.load(Ordering::Relaxed)
    }

    /// Clear all stored metrics history
    pub fn clear_history(&self) {
        self.metrics_history.write().clear();
    }
}

impl Default for SystemMonitor {
    fn default() -> Self {
        Self::new(5.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_collect_metrics() {
        let metrics = SystemMonitor::collect_metrics();
        // CPU usage should be in valid range
        assert!(metrics.cpu_usage_pct >= 0.0);
        // Memory values should be non-negative
        assert!(metrics.memory_used_mb >= 0.0);
        assert!(metrics.memory_available_mb >= 0.0);
        assert!(metrics.timestamp > 0.0);
    }

    #[test]
    fn test_collect_and_store() {
        let monitor = SystemMonitor::new(1.0);
        assert!(monitor.get_latest_metrics().is_none());

        monitor.collect_and_store();
        assert!(monitor.get_latest_metrics().is_some());
    }

    #[test]
    fn test_metrics_history() {
        let monitor = SystemMonitor::new(1.0);
        for _ in 0..5 {
            monitor.collect_and_store();
        }

        let history = monitor.get_metrics_history(3);
        assert_eq!(history.len(), 3);

        let all = monitor.get_metrics_history(100);
        assert_eq!(all.len(), 5);
    }

    #[test]
    fn test_start_stop() {
        let monitor = SystemMonitor::new(1.0);
        assert!(!monitor.is_running());
        monitor.start();
        assert!(monitor.is_running());
        monitor.stop();
        assert!(!monitor.is_running());
    }
}
