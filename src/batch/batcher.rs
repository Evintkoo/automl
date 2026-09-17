//! Dynamic Batcher Implementation
//!
//! Intelligent request batching with adaptive sizing for optimal hardware utilization.
//! All mutable statistics are consolidated under a single lock to reduce overhead.

use std::collections::{HashMap, VecDeque};
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Condvar, Mutex};
use std::thread::JoinHandle;
use std::time::Instant;

use super::priority::Priority;

/// Configuration for dynamic batcher
#[derive(Debug, Clone)]
pub struct BatcherConfig {
    /// Maximum batch size
    pub max_batch_size: usize,
    /// Maximum wait time in milliseconds before forcing batch processing
    pub max_wait_time_ms: u64,
    /// Maximum queue size
    pub max_queue_size: usize,
    /// Enable adaptive batch sizing
    pub enable_adaptive_sizing: bool,
    /// Adaptation interval in seconds
    pub adaptation_interval_secs: f64,
    /// Number of priority levels
    pub priority_levels: usize,
}

impl Default for BatcherConfig {
    fn default() -> Self {
        Self {
            max_batch_size: 64,
            max_wait_time_ms: 10,
            max_queue_size: 1000,
            enable_adaptive_sizing: true,
            adaptation_interval_secs: 5.0,
            priority_levels: 4,
        }
    }
}

/// Statistics for the dynamic batcher
#[derive(Debug, Clone, Default)]
pub struct BatcherStats {
    /// Number of batches processed
    pub processed_batches: u64,
    /// Number of requests processed
    pub processed_requests: u64,
    /// Maximum observed queue size
    pub max_observed_queue_size: usize,
    /// Average batch size
    pub avg_batch_size: f64,
    /// Average wait time in milliseconds
    pub avg_wait_time_ms: f64,
    /// Average processing time in milliseconds
    pub avg_processing_time_ms: f64,
    /// Current optimal batch size
    pub current_optimal_batch_size: usize,
    /// Throughput (requests per second)
    pub throughput: f64,
}

/// Enhanced statistics including latency and request type info
#[derive(Debug, Clone, Default)]
pub struct EnhancedBatcherStats {
    /// Base statistics
    pub base: BatcherStats,
    /// Distribution of request types
    pub request_type_distribution: HashMap<String, usize>,
    /// Average latency in milliseconds
    pub avg_latency_ms: f64,
    /// 95th percentile latency in milliseconds
    pub p95_latency_ms: f64,
    /// 99th percentile latency in milliseconds
    pub p99_latency_ms: f64,
    /// History of batch sizes
    pub batch_size_history: Vec<usize>,
}

/// A request to be batched
#[derive(Debug)]
pub struct BatchRequest<T> {
    /// The request data
    pub data: T,
    /// Priority level
    pub priority: Priority,
    /// Timestamp when request was created
    pub created_at: Instant,
}

impl<T> BatchRequest<T> {
    /// Create a new batch request
    pub fn new(data: T, priority: Priority) -> Self {
        Self {
            data,
            priority,
            created_at: Instant::now(),
        }
    }

    /// Create a normal priority request
    pub fn normal(data: T) -> Self {
        Self::new(data, Priority::Normal)
    }

    /// Get the age of this request in milliseconds
    pub fn age_ms(&self) -> f64 {
        self.created_at.elapsed().as_secs_f64() * 1000.0
    }
}

/// Result of forming a batch
#[derive(Debug)]
pub struct FormedBatch<T> {
    /// The batch items
    pub items: Vec<T>,
    /// Average wait time for items in this batch
    pub avg_wait_time_ms: f64,
    /// Time taken to form the batch
    pub formation_time_ms: f64,
}

/// Inner mutable statistics protected by a single lock.
/// Consolidates what was previously 6 separate Arc<Mutex<>> fields.
struct BatcherInner {
    batch_sizes: VecDeque<usize>,
    batch_wait_times: VecDeque<f64>,
    batch_processing_times: VecDeque<f64>,
    batch_performance_history: VecDeque<f64>,
    request_type_counts: HashMap<String, usize>,
    latency_history: Vec<f64>,
    last_adaptation_time: Instant,
}

impl BatcherInner {
    fn new() -> Self {
        Self {
            batch_sizes: VecDeque::with_capacity(100),
            batch_wait_times: VecDeque::with_capacity(100),
            batch_processing_times: VecDeque::with_capacity(100),
            batch_performance_history: VecDeque::with_capacity(50),
            request_type_counts: HashMap::new(),
            latency_history: Vec::with_capacity(200),
            last_adaptation_time: Instant::now(),
        }
    }

    /// Push a value into a bounded VecDeque
    #[inline]
    fn push_bounded(deque: &mut VecDeque<f64>, value: f64, max: usize) {
        deque.push_back(value);
        if deque.len() > max {
            deque.pop_front();
        }
    }

    /// Compute average of a VecDeque<f64>
    #[inline]
    fn avg(deque: &VecDeque<f64>) -> f64 {
        if deque.is_empty() {
            0.0
        } else {
            deque.iter().sum::<f64>() / deque.len() as f64
        }
    }

    /// Compute average of a VecDeque<usize>
    #[inline]
    fn avg_usize(deque: &VecDeque<usize>) -> f64 {
        if deque.is_empty() {
            0.0
        } else {
            deque.iter().sum::<usize>() as f64 / deque.len() as f64
        }
    }
}

/// Dynamic batcher with intelligent batching strategies.
///
/// All mutable statistics are consolidated under `inner` (single Mutex)
/// to reduce lock acquisition overhead. Atomic counters use `Relaxed`
/// ordering since they are independent monotonic counters.
pub struct DynamicBatcher<T: Send + 'static> {
    config: BatcherConfig,

    // Multi-priority queues (separate lock — held during enqueue/form_batch)
    priority_queues: Arc<Mutex<Vec<VecDeque<BatchRequest<T>>>>>,

    // Adaptive batch sizing
    current_optimal_batch_size: AtomicUsize,

    // Control
    running: AtomicBool,
    batch_trigger: Arc<(Mutex<bool>, Condvar)>,

    // Lock-free counters (Relaxed — they are independent monotonic stats)
    processed_batches: AtomicU64,
    processed_requests: AtomicU64,
    max_observed_queue_size: AtomicUsize,

    // All mutable statistics under one lock
    inner: Mutex<BatcherInner>,

    // Worker
    #[allow(dead_code)]
    worker_thread: Arc<Mutex<Option<JoinHandle<()>>>>,

    // Latency-adaptive batch sizing config
    latency_target_ms: f64,
    min_batch_size: usize,
    max_batch_size: usize,
}

impl<T: Send + 'static> DynamicBatcher<T> {
    /// Create a new dynamic batcher
    pub fn new(config: BatcherConfig) -> Self {
        let num_priorities = config.priority_levels;
        let max_batch_size = config.max_batch_size;

        let priority_queues: Vec<VecDeque<BatchRequest<T>>> = (0..num_priorities)
            .map(|_| VecDeque::new())
            .collect();

        Self {
            config,
            priority_queues: Arc::new(Mutex::new(priority_queues)),
            current_optimal_batch_size: AtomicUsize::new(max_batch_size),
            running: AtomicBool::new(false),
            batch_trigger: Arc::new((Mutex::new(false), Condvar::new())),
            processed_batches: AtomicU64::new(0),
            processed_requests: AtomicU64::new(0),
            max_observed_queue_size: AtomicUsize::new(0),
            inner: Mutex::new(BatcherInner::new()),
            worker_thread: Arc::new(Mutex::new(None)),
            latency_target_ms: 100.0,
            min_batch_size: 1,
            max_batch_size,
        }
    }

    /// Enqueue a request for batching
    pub fn enqueue(&self, request: BatchRequest<T>) -> bool {
        let priority_idx = (request.priority as usize).min(self.config.priority_levels - 1);

        let mut queues = match self.priority_queues.lock() {
            Ok(q) => q,
            Err(_) => return false,
        };

        // Check total queue size
        let total_size: usize = queues.iter().map(|q| q.len()).sum();
        if total_size >= self.config.max_queue_size {
            return false;
        }

        queues[priority_idx].push_back(request);

        // Update max observed queue size (Relaxed: independent stat counter)
        let new_total = total_size + 1;
        self.max_observed_queue_size.fetch_max(new_total, Ordering::Relaxed);

        // Signal batch trigger if we have enough items
        let optimal_size = self.current_optimal_batch_size.load(Ordering::Relaxed);
        if new_total >= optimal_size {
            let (ref lock, ref cvar) = *self.batch_trigger;
            if let Ok(_guard) = lock.lock() {
                cvar.notify_one();
            }
        }

        true
    }

    /// Enqueue data with normal priority
    pub fn enqueue_normal(&self, data: T) -> bool {
        self.enqueue(BatchRequest::normal(data))
    }

    /// Get the total queue size
    pub fn queue_size(&self) -> usize {
        self.priority_queues
            .lock()
            .map(|q| q.iter().map(|pq| pq.len()).sum())
            .unwrap_or(0)
    }

    /// Form a batch from the queues
    pub fn form_batch(&self) -> Option<FormedBatch<T>> {
        let start = Instant::now();
        let optimal_size = self.current_optimal_batch_size.load(Ordering::Relaxed);

        let mut queues = match self.priority_queues.lock() {
            Ok(q) => q,
            Err(_) => return None,
        };

        let mut items = Vec::with_capacity(optimal_size);
        let mut total_wait_time = 0.0;

        // Collect from highest to lowest priority
        for queue in queues.iter_mut() {
            while items.len() < optimal_size {
                match queue.pop_front() {
                    Some(request) => {
                        total_wait_time += request.age_ms();
                        items.push(request.data);
                    }
                    None => break,
                }
            }
            if items.len() >= optimal_size {
                break;
            }
        }

        if items.is_empty() {
            return None;
        }

        let avg_wait_time_ms = total_wait_time / items.len() as f64;
        let formation_time_ms = start.elapsed().as_secs_f64() * 1000.0;

        // Update statistics (single lock acquisition)
        if let Ok(mut inner) = self.inner.lock() {
            inner.batch_sizes.push_back(items.len());
            if inner.batch_sizes.len() > 100 {
                inner.batch_sizes.pop_front();
            }
            BatcherInner::push_bounded(&mut inner.batch_wait_times, avg_wait_time_ms, 100);
        }

        Some(FormedBatch {
            items,
            avg_wait_time_ms,
            formation_time_ms,
        })
    }

    /// Record processing time for a batch
    pub fn record_processing_time(&self, processing_time_ms: f64) {
        self.processed_batches.fetch_add(1, Ordering::Relaxed);

        if let Ok(mut inner) = self.inner.lock() {
            BatcherInner::push_bounded(&mut inner.batch_processing_times, processing_time_ms, 100);

            inner.batch_performance_history.push_back(processing_time_ms);
            if inner.batch_performance_history.len() > 50 {
                inner.batch_performance_history.pop_front();
            }
        }

        // Check if we should adapt
        if self.config.enable_adaptive_sizing {
            self.maybe_adapt();
        }
    }

    /// Maybe adapt batch size based on performance
    fn maybe_adapt(&self) {
        let should_adapt = self.inner.lock()
            .map(|inner| inner.last_adaptation_time.elapsed().as_secs_f64() >= self.config.adaptation_interval_secs)
            .unwrap_or(false);

        if should_adapt {
            self.adapt_batch_size();
        }
    }

    /// Adapt batch size based on performance history
    fn adapt_batch_size(&self) {
        let mut inner = match self.inner.lock() {
            Ok(g) => g,
            Err(_) => return,
        };

        if inner.batch_performance_history.len() < 10 {
            return;
        }

        let avg_time = BatcherInner::avg(&inner.batch_performance_history);
        let target_time = self.config.max_wait_time_ms as f64;
        let current_size = self.current_optimal_batch_size.load(Ordering::Relaxed);

        let new_size = if avg_time < target_time * 0.5 {
            ((current_size as f64) * 1.1).ceil() as usize
        } else if avg_time > target_time * 0.9 {
            ((current_size as f64) * 0.9).floor() as usize
        } else {
            current_size
        };

        let new_size = new_size.max(1).min(self.config.max_batch_size);
        self.current_optimal_batch_size.store(new_size, Ordering::Relaxed);
        inner.last_adaptation_time = Instant::now();
    }

    /// Get batcher statistics
    pub fn stats(&self) -> BatcherStats {
        let processed_batches = self.processed_batches.load(Ordering::Relaxed);
        let processed_requests = self.processed_requests.load(Ordering::Relaxed);

        let (avg_batch_size, avg_wait_time_ms, avg_processing_time_ms) = self.inner
            .lock()
            .map(|inner| (
                BatcherInner::avg_usize(&inner.batch_sizes),
                BatcherInner::avg(&inner.batch_wait_times),
                BatcherInner::avg(&inner.batch_processing_times),
            ))
            .unwrap_or((0.0, 0.0, 0.0));

        let throughput = if avg_processing_time_ms > 0.0 {
            (avg_batch_size * 1000.0) / avg_processing_time_ms
        } else {
            0.0
        };

        BatcherStats {
            processed_batches,
            processed_requests,
            max_observed_queue_size: self.max_observed_queue_size.load(Ordering::Relaxed),
            avg_batch_size,
            avg_wait_time_ms,
            avg_processing_time_ms,
            current_optimal_batch_size: self.current_optimal_batch_size.load(Ordering::Relaxed),
            throughput,
        }
    }

    /// Track a request type for diversity-based batching
    pub fn track_request_type(&self, request_type: &str) {
        if let Ok(mut inner) = self.inner.lock() {
            *inner.request_type_counts.entry(request_type.to_string()).or_insert(0) += 1;
        }
    }

    /// Returns true if the current batch has requests of multiple types
    pub fn has_diverse_requests(&self) -> bool {
        self.inner.lock()
            .map(|inner| inner.request_type_counts.len() > 1)
            .unwrap_or(false)
    }

    /// Record a latency observation for adaptive batch sizing
    pub fn record_latency(&self, latency_ms: f64) {
        if let Ok(mut inner) = self.inner.lock() {
            inner.latency_history.push(latency_ms);
            // Keep last 200 observations — use truncate pattern to avoid O(n) drain
            if inner.latency_history.len() > 200 {
                let start = inner.latency_history.len() - 200;
                inner.latency_history.copy_within(start.., 0);
                inner.latency_history.truncate(200);
            }
        }
    }

    /// Adapt batch size based on observed latency relative to target
    pub fn adapt_batch_size_for_latency(&self) {
        let avg_latency = match self.inner.lock() {
            Ok(inner) => {
                if inner.latency_history.is_empty() {
                    return;
                }
                inner.latency_history.iter().sum::<f64>() / inner.latency_history.len() as f64
            }
            Err(_) => return,
        };

        let current_size = self.current_optimal_batch_size.load(Ordering::Relaxed);

        let new_size = if avg_latency > self.latency_target_ms {
            (current_size as f64 * 0.8).floor() as usize
        } else if avg_latency < self.latency_target_ms * 0.5 {
            (current_size as f64 * 1.2).ceil() as usize
        } else {
            current_size
        };

        let new_size = new_size.max(self.min_batch_size).min(self.max_batch_size);
        self.current_optimal_batch_size.store(new_size, Ordering::Relaxed);
    }

    /// Get enhanced statistics including latency and request type info.
    ///
    /// Acquires `inner` exactly once so the base stats (avg batch size/wait/processing
    /// time) and the enhanced fields (latency percentiles, request-type distribution,
    /// batch size history) are all read from the same consistent snapshot — previously
    /// `stats()` took its own lock and then this method re-locked separately, so a
    /// concurrent `form_batch`/`record_*` call in between could produce a response
    /// mixing metrics from two different points in time.
    pub fn get_enhanced_stats(&self) -> EnhancedBatcherStats {
        let processed_batches = self.processed_batches.load(Ordering::Relaxed);
        let processed_requests = self.processed_requests.load(Ordering::Relaxed);

        let (
            avg_batch_size,
            avg_wait_time_ms,
            avg_processing_time_ms,
            request_type_distribution,
            avg_latency_ms,
            p95_latency_ms,
            p99_latency_ms,
            batch_size_history,
        ) = self.inner
            .lock()
            .map(|inner| {
                let avg_batch_size = BatcherInner::avg_usize(&inner.batch_sizes);
                let avg_wait_time_ms = BatcherInner::avg(&inner.batch_wait_times);
                let avg_processing_time_ms = BatcherInner::avg(&inner.batch_processing_times);

                let dist = inner.request_type_counts.clone();

                let (avg, p95, p99) = if inner.latency_history.is_empty() {
                    (0.0, 0.0, 0.0)
                } else {
                    let avg = inner.latency_history.iter().sum::<f64>() / inner.latency_history.len() as f64;
                    let mut sorted = inner.latency_history.clone();
                    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
                    let p95_idx = ((sorted.len() as f64 * 0.95).ceil() as usize).min(sorted.len()) - 1;
                    let p99_idx = ((sorted.len() as f64 * 0.99).ceil() as usize).min(sorted.len()) - 1;
                    (avg, sorted[p95_idx], sorted[p99_idx])
                };

                let sizes: Vec<usize> = inner.batch_sizes.iter().copied().collect();
                (avg_batch_size, avg_wait_time_ms, avg_processing_time_ms, dist, avg, p95, p99, sizes)
            })
            .unwrap_or_default();

        let throughput = if avg_processing_time_ms > 0.0 {
            (avg_batch_size * 1000.0) / avg_processing_time_ms
        } else {
            0.0
        };

        let base = BatcherStats {
            processed_batches,
            processed_requests,
            max_observed_queue_size: self.max_observed_queue_size.load(Ordering::Relaxed),
            avg_batch_size,
            avg_wait_time_ms,
            avg_processing_time_ms,
            current_optimal_batch_size: self.current_optimal_batch_size.load(Ordering::Relaxed),
            throughput,
        };

        EnhancedBatcherStats {
            base,
            request_type_distribution,
            avg_latency_ms,
            p95_latency_ms,
            p99_latency_ms,
            batch_size_history,
        }
    }

    /// Check if the batcher is running
    pub fn is_running(&self) -> bool {
        self.running.load(Ordering::Relaxed)
    }

    /// Stop the batcher
    pub fn stop(&self) {
        self.running.store(false, Ordering::Relaxed);

        // Wake up the condition variable
        let (ref _lock, ref cvar) = *self.batch_trigger;
        cvar.notify_all();
    }

    /// Clear all queues
    pub fn clear(&self) {
        if let Ok(mut queues) = self.priority_queues.lock() {
            for queue in queues.iter_mut() {
                queue.clear();
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dynamic_batcher_enqueue() {
        let config = BatcherConfig::default();
        let batcher: DynamicBatcher<i32> = DynamicBatcher::new(config);

        for i in 0..10 {
            assert!(batcher.enqueue_normal(i));
        }

        assert_eq!(batcher.queue_size(), 10);
    }

    #[test]
    fn test_dynamic_batcher_form_batch() {
        let mut config = BatcherConfig::default();
        config.max_batch_size = 5;
        let batcher: DynamicBatcher<i32> = DynamicBatcher::new(config);

        for i in 0..10 {
            batcher.enqueue_normal(i);
        }

        let batch = batcher.form_batch().unwrap();
        assert_eq!(batch.items.len(), 5);
        assert_eq!(batcher.queue_size(), 5);
    }

    #[test]
    fn test_dynamic_batcher_priority() {
        let config = BatcherConfig::default();
        let batcher: DynamicBatcher<&str> = DynamicBatcher::new(config);

        batcher.enqueue(BatchRequest::new("low", Priority::Low));
        batcher.enqueue(BatchRequest::new("critical", Priority::Critical));
        batcher.enqueue(BatchRequest::new("normal", Priority::Normal));

        let batch = batcher.form_batch().unwrap();

        // Critical should come first
        assert_eq!(batch.items[0], "critical");
    }

    #[test]
    fn test_track_request_type() {
        let config = BatcherConfig::default();
        let batcher: DynamicBatcher<i32> = DynamicBatcher::new(config);

        batcher.track_request_type("inference");
        batcher.track_request_type("inference");
        batcher.track_request_type("training");

        let stats = batcher.get_enhanced_stats();
        assert_eq!(stats.request_type_distribution.get("inference"), Some(&2));
        assert_eq!(stats.request_type_distribution.get("training"), Some(&1));
    }

    #[test]
    fn test_has_diverse_requests() {
        let config = BatcherConfig::default();
        let batcher: DynamicBatcher<i32> = DynamicBatcher::new(config);

        assert!(!batcher.has_diverse_requests());

        batcher.track_request_type("inference");
        assert!(!batcher.has_diverse_requests());

        batcher.track_request_type("training");
        assert!(batcher.has_diverse_requests());
    }

    #[test]
    fn test_adapt_batch_size_for_latency_decrease() {
        let mut config = BatcherConfig::default();
        config.max_batch_size = 100;
        let batcher: DynamicBatcher<i32> = DynamicBatcher::new(config);

        // Record high latencies (above target of 100ms)
        for _ in 0..10 {
            batcher.record_latency(200.0);
        }

        let before = batcher.current_optimal_batch_size.load(Ordering::Relaxed);
        batcher.adapt_batch_size_for_latency();
        let after = batcher.current_optimal_batch_size.load(Ordering::Relaxed);

        assert!(after < before, "Batch size should decrease when latency is high");
    }

    #[test]
    fn test_adapt_batch_size_for_latency_increase() {
        let mut config = BatcherConfig::default();
        config.max_batch_size = 200;
        let batcher: DynamicBatcher<i32> = DynamicBatcher::new(config);

        // Set a smaller starting size
        batcher.current_optimal_batch_size.store(50, Ordering::Relaxed);

        // Record low latencies (below 0.5 * target of 100ms = 50ms)
        for _ in 0..10 {
            batcher.record_latency(10.0);
        }

        batcher.adapt_batch_size_for_latency();
        let after = batcher.current_optimal_batch_size.load(Ordering::Relaxed);

        assert!(after > 50, "Batch size should increase when latency is well under target");
    }

    #[test]
    fn test_enhanced_stats_latency_percentiles() {
        let config = BatcherConfig::default();
        let batcher: DynamicBatcher<i32> = DynamicBatcher::new(config);

        // Record 100 latencies: 1.0, 2.0, ..., 100.0
        for i in 1..=100 {
            batcher.record_latency(i as f64);
        }

        let stats = batcher.get_enhanced_stats();
        assert!((stats.avg_latency_ms - 50.5).abs() < 0.01);
        assert!(stats.p95_latency_ms >= 95.0);
        assert!(stats.p99_latency_ms >= 99.0);
        assert!(stats.batch_size_history.is_empty());
    }
}
