//! Memory Pool Implementation
//!
//! NUMA-aware memory buffer pooling for optimal memory locality
//! and reduced allocation overhead.

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, RwLock, Weak};

/// Buffer size category for pooling
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BufferCategory {
    /// Small buffers (< 1KB)
    Small,
    /// Medium buffers (1KB - 1MB)
    Medium,
    /// Large buffers (> 1MB)
    Large,
}

impl BufferCategory {
    /// Categorize a buffer size
    pub fn from_size(size_bytes: usize) -> Self {
        if size_bytes < 1024 {
            BufferCategory::Small
        } else if size_bytes < 1024 * 1024 {
            BufferCategory::Medium
        } else {
            BufferCategory::Large
        }
    }
}

/// Statistics for the memory pool
#[derive(Debug, Clone, Default)]
pub struct PoolStats {
    /// Number of cache hits (buffer reused)
    pub hits: u64,
    /// Number of cache misses (new allocation)
    pub misses: u64,
    /// Number of buffers currently in pool
    pub pooled_buffers: usize,
    /// Total bytes in pooled buffers
    pub pooled_bytes: usize,
    /// Hit rate
    pub hit_rate: f64,
}

/// A pooled buffer that returns to the pool when dropped
pub struct PooledBuffer {
    /// The buffer data
    data: Vec<f64>,
    /// Weak reference to the pool for return (avoids Arc cycle)
    pool: Option<Weak<MemoryPool>>,
    /// Shape of the buffer
    shape: Vec<usize>,
}

impl PooledBuffer {
    /// Create a new pooled buffer without a pool reference
    pub fn new(shape: Vec<usize>) -> Self {
        let size: usize = shape.iter().product();
        Self {
            data: vec![0.0; size],
            pool: None,
            shape,
        }
    }

    /// Create a pooled buffer with weak pool reference
    fn with_pool(data: Vec<f64>, shape: Vec<usize>, pool: &Arc<MemoryPool>) -> Self {
        Self {
            data,
            pool: Some(Arc::downgrade(pool)),
            shape,
        }
    }
    
    /// Get the shape of this buffer
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }
    
    /// Get the total size
    pub fn len(&self) -> usize {
        self.data.len()
    }
    
    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }
    
    /// Get a reference to the data
    pub fn as_slice(&self) -> &[f64] {
        &self.data
    }
    
    /// Get a mutable reference to the data
    pub fn as_mut_slice(&mut self) -> &mut [f64] {
        &mut self.data
    }
    
    /// Fill with zeros
    pub fn zero(&mut self) {
        self.data.fill(0.0);
    }
    
    /// Fill with a value
    pub fn fill(&mut self, value: f64) {
        self.data.fill(value);
    }
    
    /// Get a value at index
    pub fn get(&self, index: usize) -> Option<f64> {
        self.data.get(index).copied()
    }
    
    /// Set a value at index
    pub fn set(&mut self, index: usize, value: f64) {
        if let Some(elem) = self.data.get_mut(index) {
            *elem = value;
        }
    }
    
    /// Convert to Vec (consumes the buffer, prevents return to pool)
    pub fn into_vec(mut self) -> Vec<f64> {
        self.pool = None;
        std::mem::take(&mut self.data)
    }
}

impl Drop for PooledBuffer {
    fn drop(&mut self) {
        if let Some(weak) = self.pool.take() {
            if let Some(pool) = weak.upgrade() {
                let data = std::mem::take(&mut self.data);
                let shape = std::mem::take(&mut self.shape);
                pool.return_buffer(data, shape);
            }
        }
    }
}

impl std::ops::Index<usize> for PooledBuffer {
    type Output = f64;
    
    fn index(&self, index: usize) -> &Self::Output {
        &self.data[index]
    }
}

impl std::ops::IndexMut<usize> for PooledBuffer {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.data[index]
    }
}

/// Buffer pool key
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct BufferKey {
    shape: Vec<usize>,
}

/// NUMA-aware memory pool for buffer management
pub struct MemoryPool {
    /// Maximum number of buffers to pool per shape
    max_buffers_per_shape: usize,
    /// Maximum number of distinct shape buckets tracked per category. Bounds
    /// total memory under high shape variance (e.g. variable-length time
    /// series, dynamic batching) by LRU-evicting the least-recently-used
    /// shape bucket once this cap would be exceeded.
    max_distinct_shapes: usize,
    /// Pool storage organized by category and shape
    pools: RwLock<HashMap<BufferCategory, HashMap<BufferKey, Vec<Vec<f64>>>>>,
    /// Recency order of shape keys per category, least-recently-used at the front
    shape_order: RwLock<HashMap<BufferCategory, Vec<BufferKey>>>,
    /// Statistics
    hits: AtomicU64,
    misses: AtomicU64,
}

impl MemoryPool {
    /// Default cap on distinct shapes tracked per category before LRU eviction kicks in
    const DEFAULT_MAX_DISTINCT_SHAPES: usize = 64;

    /// Create a new memory pool
    pub fn new(max_buffers_per_shape: usize) -> Arc<Self> {
        Self::with_max_distinct_shapes(max_buffers_per_shape, Self::DEFAULT_MAX_DISTINCT_SHAPES)
    }

    /// Create a new memory pool with an explicit cap on the number of distinct
    /// shapes tracked per category (LRU-evicted once exceeded)
    pub fn with_max_distinct_shapes(max_buffers_per_shape: usize, max_distinct_shapes: usize) -> Arc<Self> {
        Arc::new(Self {
            max_buffers_per_shape,
            max_distinct_shapes: max_distinct_shapes.max(1),
            pools: RwLock::new(HashMap::new()),
            shape_order: RwLock::new(HashMap::new()),
            hits: AtomicU64::new(0),
            misses: AtomicU64::new(0),
        })
    }
    
    /// Get a buffer with the given shape
    pub fn get_buffer(self: &Arc<Self>, shape: Vec<usize>) -> PooledBuffer {
        let size: usize = shape.iter().product();
        let size_bytes = size * std::mem::size_of::<f64>();
        let category = BufferCategory::from_size(size_bytes);
        let key = BufferKey { shape: shape.clone() };
        
        // Try to get from pool
        if let Ok(mut pools) = self.pools.write() {
            if let Some(category_pool) = pools.get_mut(&category) {
                if let Some(buffers) = category_pool.get_mut(&key) {
                    if let Some(data) = buffers.pop() {
                        self.hits.fetch_add(1, Ordering::Relaxed);
                        return PooledBuffer::with_pool(data, shape, self);
                    }
                }
            }
        }

        // Allocate new buffer
        self.misses.fetch_add(1, Ordering::Relaxed);
        let data = vec![0.0; size];
        PooledBuffer::with_pool(data, shape, self)
    }
    
    /// Return a buffer to the pool
    fn return_buffer(&self, data: Vec<f64>, shape: Vec<usize>) {
        let size_bytes = data.len() * std::mem::size_of::<f64>();
        let category = BufferCategory::from_size(size_bytes);
        let key = BufferKey { shape };

        if let Ok(mut pools) = self.pools.write() {
            let category_pool = pools.entry(category).or_insert_with(HashMap::new);

            // If this is a brand-new shape and we're already tracking the max
            // number of distinct shapes for this category, evict the
            // least-recently-used shape bucket to keep memory bounded.
            if !category_pool.contains_key(&key) && category_pool.len() >= self.max_distinct_shapes {
                if let Ok(mut order) = self.shape_order.write() {
                    if let Some(order_list) = order.get_mut(&category) {
                        if !order_list.is_empty() {
                            let lru_key = order_list.remove(0);
                            category_pool.remove(&lru_key);
                        }
                    }
                }
            }

            let buffers = category_pool.entry(key.clone()).or_insert_with(Vec::new);

            if buffers.len() < self.max_buffers_per_shape {
                buffers.push(data);
            }
            // Otherwise, let the buffer be dropped
        }

        // Record this shape as the most-recently-used for its category.
        if let Ok(mut order) = self.shape_order.write() {
            let order_list = order.entry(category).or_insert_with(Vec::new);
            order_list.retain(|k| k != &key);
            order_list.push(key);
        }
    }
    
    /// Get a zeroed buffer
    pub fn get_zeros(self: &Arc<Self>, shape: Vec<usize>) -> PooledBuffer {
        let mut buffer = self.get_buffer(shape);
        buffer.zero();
        buffer
    }
    
    /// Get a buffer filled with ones
    pub fn get_ones(self: &Arc<Self>, shape: Vec<usize>) -> PooledBuffer {
        let mut buffer = self.get_buffer(shape);
        buffer.fill(1.0);
        buffer
    }
    
    /// Clear all pooled buffers
    pub fn clear(&self) {
        if let Ok(mut pools) = self.pools.write() {
            pools.clear();
        }
        if let Ok(mut order) = self.shape_order.write() {
            order.clear();
        }
    }
    
    /// Get pool statistics
    pub fn stats(&self) -> PoolStats {
        let hits = self.hits.load(Ordering::Relaxed);
        let misses = self.misses.load(Ordering::Relaxed);
        let total = hits + misses;
        
        let (pooled_buffers, pooled_bytes) = self.pools
            .read()
            .map(|pools| {
                let mut count = 0usize;
                let mut bytes = 0usize;
                for category_pool in pools.values() {
                    for (key, buffers) in category_pool {
                        count += buffers.len();
                        let size: usize = key.shape.iter().product();
                        bytes += buffers.len() * size * std::mem::size_of::<f64>();
                    }
                }
                (count, bytes)
            })
            .unwrap_or((0, 0));
        
        PoolStats {
            hits,
            misses,
            pooled_buffers,
            pooled_bytes,
            hit_rate: if total > 0 { hits as f64 / total as f64 } else { 0.0 },
        }
    }
    
    /// Get the number of pooled buffers
    pub fn pooled_count(&self) -> usize {
        self.pools
            .read()
            .map(|pools| {
                pools.values()
                    .flat_map(|cp| cp.values())
                    .map(|b| b.len())
                    .sum()
            })
            .unwrap_or(0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_memory_pool_basic() {
        let pool = MemoryPool::new(10);
        
        let mut buffer = pool.get_buffer(vec![10, 10]);
        assert_eq!(buffer.len(), 100);
        assert_eq!(buffer.shape(), &[10, 10]);
        
        buffer[0] = 1.0;
        assert_eq!(buffer[0], 1.0);
    }
    
    #[test]
    fn test_buffer_reuse() {
        let pool = MemoryPool::new(10);
        
        // Get a buffer
        {
            let buffer = pool.get_buffer(vec![10, 10]);
            assert_eq!(buffer.len(), 100);
        } // Buffer returned to pool
        
        // Get another buffer with same shape - should be reused
        {
            let _buffer = pool.get_buffer(vec![10, 10]);
        }
        
        let stats = pool.stats();
        assert_eq!(stats.hits, 1); // Second get was a hit
        assert_eq!(stats.misses, 1); // First get was a miss
    }
    
    #[test]
    fn test_get_zeros() {
        let pool = MemoryPool::new(10);
        
        let buffer = pool.get_zeros(vec![5, 5]);
        assert!(buffer.as_slice().iter().all(|&x| x == 0.0));
    }
    
    #[test]
    fn test_get_ones() {
        let pool = MemoryPool::new(10);
        
        let buffer = pool.get_ones(vec![5, 5]);
        assert!(buffer.as_slice().iter().all(|&x| x == 1.0));
    }
    
    #[test]
    fn test_into_vec() {
        let pool = MemoryPool::new(10);
        
        let mut buffer = pool.get_buffer(vec![5]);
        buffer[0] = 42.0;
        
        let vec = buffer.into_vec();
        assert_eq!(vec[0], 42.0);
        assert_eq!(vec.len(), 5);
    }

    #[test]
    fn test_distinct_shapes_bounded() {
        // Cap distinct shapes per category at 3 - a workload with many unique
        // shapes should never accumulate more than that many shape buckets.
        let pool = MemoryPool::with_max_distinct_shapes(10, 3);

        for n in 1..=20usize {
            // Each shape is unique and small enough to land in the Small category.
            let buffer = pool.get_buffer(vec![n]);
            drop(buffer); // returned to the pool, tracked by shape
        }

        let pools = pool.pools.read().unwrap();
        for category_pool in pools.values() {
            assert!(category_pool.len() <= 3, "distinct shapes exceeded cap: {}", category_pool.len());
        }
    }
}
