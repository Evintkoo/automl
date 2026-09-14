//! Batch Processor Implementation
//!
//! High-performance async batch processor with adaptive sizing and priority queuing.

use std::collections::VecDeque;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, RwLock};
use std::thread::JoinHandle;
use std::time::Instant;

use crate::error::{AutoMLError, Result};
use super::priority::{Priority, PriorityQueue};

/// Configuration for batch processor
#[derive(Debug, Clone)]
pub struct BatchProcessorConfig {
    /// Initial batch size
    pub initial_batch_size: usize,
    /// Minimum batch size
    pub min_batch_size: usize,
    /// Maximum batch size
    pub max_batch_size: usize,
    /// Maximum queue size
    pub max_queue_size: usize,
    /// Batch timeout in milliseconds
    pub batch_timeout_ms: u64,
    /// Number of worker threads
    pub num_workers: usize,
    /// Enable adaptive batching
    pub adaptive_batching: bool,
    /// Enable priority queue
    pub enable_priority_queue: bool,
    /// Enable monitoring
    pub enable_monitoring: bool,
    /// Monitoring window size
    pub monitoring_window: usize,
    /// Memory warning threshold (percentage)
    pub memory_warning_threshold: f64,
    /// Enable health monitoring
    pub enable_health_monitoring: bool,
    /// Health check interval in milliseconds
    pub health_check_interval_ms: u64,
}

impl Default for BatchProcessorConfig {
    fn default() -> Self {
        Self {
            initial_batch_size: 100,
            min_batch_size: 50,
            max_batch_size: 200,
            max_queue_size: 1000,
            batch_timeout_ms: 1000,
            num_workers: 4,
            adaptive_batching: true,
            enable_priority_queue: true,
            enable_monitoring: true,
            monitoring_window: 100,
            memory_warning_threshold: 70.0,
            enable_health_monitoring: true,
            health_check_interval_ms: 5000,
        }
    }
}

/// Statistics for batch processing
#[derive(Debug, Clone, Default)]
pub struct BatchStats {
    /// Total batches processed
    pub batches_processed: u64,
    /// Total items processed
    pub items_processed: u64,
    /// Average batch size
    pub avg_batch_size: f64,
    /// Average processing time in milliseconds
    pub avg_processing_time_ms: f64,
    /// Current queue size
    pub queue_size: usize,
    /// Throughput (items per second)
    pub throughput: f64,
    /// Error count
    pub error_count: u64,
}

/// Result of batch processing
#[derive(Debug)]
pub struct BatchResult<T> {
    /// Processed items
    pub items: Vec<T>,
    /// Processing time in milliseconds
    pub processing_time_ms: f64,
    /// Batch size
    pub batch_size: usize,
    /// Any errors that occurred
    pub errors: Vec<String>,
}

impl<T> BatchResult<T> {
    /// Create a successful batch result
    pub fn success(items: Vec<T>, processing_time_ms: f64) -> Self {
        let batch_size = items.len();
        Self {
            items,
            processing_time_ms,
            batch_size,
            errors: Vec::new(),
        }
    }
    
    /// Create a batch result with errors
    pub fn with_errors(items: Vec<T>, processing_time_ms: f64, errors: Vec<String>) -> Self {
        let batch_size = items.len();
        Self {
            items,
            processing_time_ms,
            batch_size,
            errors,
        }
    }
}

/// High-performance batch processor with adaptive batching and priority queuing
pub struct BatchProcessor<T: Send + 'static> {
    config: BatchProcessorConfig,
    queue: Arc<Mutex<PriorityQueue<T>>>,
    current_batch_size: AtomicUsize,
    running: AtomicBool,
    paused: AtomicBool,
    
    // Statistics
    batches_processed: AtomicU64,
    items_processed: AtomicU64,
    error_count: AtomicU64,
    total_processing_time_ns: AtomicU64,
    
    // Processing times for adaptive batching
    processing_times: Arc<RwLock<VecDeque<f64>>>,
    
    // Worker handles
    workers: Arc<Mutex<Vec<JoinHandle<()>>>>,
}

impl<T: Send + 'static> BatchProcessor<T> {
    /// Create a new batch processor with the given configuration
    pub fn new(config: BatchProcessorConfig) -> Self {
        let initial_batch_size = config.initial_batch_size;
        let max_queue_size = config.max_queue_size;
        let monitoring_window = config.monitoring_window;
        
        Self {
            config,
            queue: Arc::new(Mutex::new(PriorityQueue::new(max_queue_size))),
            current_batch_size: AtomicUsize::new(initial_batch_size),
            running: AtomicBool::new(false),
            paused: AtomicBool::new(false),
            batches_processed: AtomicU64::new(0),
            items_processed: AtomicU64::new(0),
            error_count: AtomicU64::new(0),
            total_processing_time_ns: AtomicU64::new(0),
            processing_times: Arc::new(RwLock::new(VecDeque::with_capacity(monitoring_window))),
            workers: Arc::new(Mutex::new(Vec::new())),
        }
    }
    
    /// Enqueue an item with the given priority
    pub fn enqueue(&self, item: T, priority: Priority) -> Result<()> {
        let mut queue = self.queue.lock().map_err(|_| {
            AutoMLError::ProcessingError("Failed to acquire queue lock".to_string())
        })?;
        
        if queue.is_full() {
            return Err(AutoMLError::ProcessingError("Queue is full".to_string()));
        }
        
        queue.push(item, priority);
        Ok(())
    }
    
    /// Enqueue an item with normal priority
    pub fn enqueue_normal(&self, item: T) -> Result<()> {
        self.enqueue(item, Priority::Normal)
    }
    
    /// Get the current queue size
    pub fn queue_size(&self) -> usize {
        self.queue.lock().map(|q| q.len()).unwrap_or(0)
    }
    
    /// Check if the processor is running
    pub fn is_running(&self) -> bool {
        self.running.load(Ordering::Relaxed)
    }
    
    /// Pause the processor
    pub fn pause(&self) {
        self.paused.store(true, Ordering::Relaxed);
    }
    
    /// Resume the processor
    pub fn resume(&self) {
        self.paused.store(false, Ordering::Relaxed);
    }
    
    /// Check if the processor is paused
    pub fn is_paused(&self) -> bool {
        self.paused.load(Ordering::Relaxed)
    }
    
    /// Get the current optimal batch size
    pub fn current_batch_size(&self) -> usize {
        self.current_batch_size.load(Ordering::Relaxed)
    }
    
    /// Get processing statistics
    pub fn stats(&self) -> BatchStats {
        let batches = self.batches_processed.load(Ordering::Relaxed);
        let items = self.items_processed.load(Ordering::Relaxed);
        let total_time_ns = self.total_processing_time_ns.load(Ordering::Relaxed);
        
        let avg_batch_size = if batches > 0 {
            items as f64 / batches as f64
        } else {
            0.0
        };
        
        let avg_processing_time_ms = if batches > 0 {
            (total_time_ns as f64 / batches as f64) / 1_000_000.0
        } else {
            0.0
        };
        
        let total_time_sec = total_time_ns as f64 / 1_000_000_000.0;
        let throughput = if total_time_sec > 0.0 {
            items as f64 / total_time_sec
        } else {
            0.0
        };
        
        BatchStats {
            batches_processed: batches,
            items_processed: items,
            avg_batch_size,
            avg_processing_time_ms,
            queue_size: self.queue_size(),
            throughput,
            error_count: self.error_count.load(Ordering::Relaxed),
        }
    }
    
    /// Collect a batch of items from the queue
    pub fn collect_batch(&self) -> Vec<T> {
        let batch_size = self.current_batch_size.load(Ordering::Relaxed);
        
        let mut queue = match self.queue.lock() {
            Ok(q) => q,
            Err(_) => return Vec::new(),
        };
        
        queue.drain(batch_size)
    }
    
    /// Process a batch with the given function
    pub fn process_batch<F, R>(&self, items: Vec<T>, process_fn: F) -> BatchResult<R>
    where
        F: Fn(Vec<T>) -> Vec<R>,
    {
        let start = Instant::now();
        let batch_size = items.len();
        
        let results = process_fn(items);
        
        let elapsed = start.elapsed();
        let processing_time_ms = elapsed.as_secs_f64() * 1000.0;
        
        // Update statistics
        self.batches_processed.fetch_add(1, Ordering::Relaxed);
        self.items_processed.fetch_add(batch_size as u64, Ordering::Relaxed);
        self.total_processing_time_ns.fetch_add(elapsed.as_nanos() as u64, Ordering::Relaxed);
        
        // Record processing time for adaptive batching
        if self.config.adaptive_batching {
            if let Ok(mut times) = self.processing_times.write() {
                times.push_back(processing_time_ms);
                while times.len() > self.config.monitoring_window {
                    times.pop_front();
                }
            }
            self.adapt_batch_size(processing_time_ms);
        }
        
        BatchResult::success(results, processing_time_ms)
    }
    
    /// Adapt batch size based on processing performance
    fn adapt_batch_size(&self, processing_time_ms: f64) {
        let target_time_ms = self.config.batch_timeout_ms as f64 * 0.8; // Target 80% of timeout
        let current_size = self.current_batch_size.load(Ordering::Relaxed);
        
        let new_size = if processing_time_ms < target_time_ms * 0.5 {
            // Processing is fast, increase batch size
            (current_size as f64 * 1.2).ceil() as usize
        } else if processing_time_ms > target_time_ms {
            // Processing is slow, decrease batch size
            (current_size as f64 * 0.8).floor() as usize
        } else {
            current_size
        };
        
        let new_size = new_size
            .max(self.config.min_batch_size)
            .min(self.config.max_batch_size);
        
        self.current_batch_size.store(new_size, Ordering::Relaxed);
    }
    
    /// Shutdown the processor and wait for workers to finish
    pub fn shutdown(&self) {
        self.running.store(false, Ordering::Relaxed);
        
        // Wait for workers to finish
        if let Ok(mut workers) = self.workers.lock() {
            for handle in workers.drain(..) {
                let _ = handle.join();
            }
        }
    }
    
    /// Clear all items from the queue
    pub fn clear(&self) {
        if let Ok(mut queue) = self.queue.lock() {
            queue.clear();
        }
    }
}

impl<T: Send + 'static> Drop for BatchProcessor<T> {
    fn drop(&mut self) {
        self.shutdown();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_batch_processor_basic() {
        let config = BatchProcessorConfig::default();
        let processor: BatchProcessor<i32> = BatchProcessor::new(config);
        
        // Enqueue items
        for i in 0..10 {
            processor.enqueue_normal(i).unwrap();
        }
        
        assert_eq!(processor.queue_size(), 10);
        
        // Collect batch
        let batch = processor.collect_batch();
        assert_eq!(batch.len(), 10);
        assert_eq!(processor.queue_size(), 0);
    }
    
    #[test]
    fn test_batch_processor_priority() {
        let config = BatchProcessorConfig::default();
        let processor: BatchProcessor<&str> = BatchProcessor::new(config);
        
        processor.enqueue("low", Priority::Low).unwrap();
        processor.enqueue("critical", Priority::Critical).unwrap();
        processor.enqueue("normal", Priority::Normal).unwrap();
        
        let batch = processor.collect_batch();
        
        // Should be ordered by priority
        assert_eq!(batch[0], "critical");
        assert_eq!(batch[1], "normal");
        assert_eq!(batch[2], "low");
    }
    
    #[test]
    fn test_batch_processor_stats() {
        let config = BatchProcessorConfig::default();
        let processor: BatchProcessor<i32> = BatchProcessor::new(config);
        
        for i in 0..50 {
            processor.enqueue_normal(i).unwrap();
        }
        
        let batch = processor.collect_batch();
        let _result = processor.process_batch(batch, |items| items);
        
        let stats = processor.stats();
        assert_eq!(stats.batches_processed, 1);
        assert_eq!(stats.items_processed, 50);
    }
}
