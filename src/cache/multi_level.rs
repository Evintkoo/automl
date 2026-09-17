//! Multi-Level Cache
//!
//! Advanced caching with L1/L2/L3 levels, predictive warming,
//! and intelligent partitioning.

use std::sync::atomic::{AtomicU64, Ordering};
use std::hash::Hash;

use super::lru_ttl::LruTtlCache;

/// Cache level designation
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CacheLevel {
    /// L1 - Hot data, small cache, fast access
    L1,
    /// L2 - Warm data, medium cache
    L2,
    /// L3 - Cold data, large cache, longer TTL
    L3,
}

/// Statistics for the multi-level cache
#[derive(Debug, Clone, Default)]
pub struct CacheStats {
    /// L1 cache hits
    pub l1_hits: u64,
    /// L1 cache misses
    pub l1_misses: u64,
    /// L2 cache hits
    pub l2_hits: u64,
    /// L2 cache misses
    pub l2_misses: u64,
    /// L3 cache hits
    pub l3_hits: u64,
    /// L3 cache misses
    pub l3_misses: u64,
    /// Total requests
    pub total_requests: u64,
    /// Overall hit rate
    pub hit_rate: f64,
}

/// Multi-level cache with L1/L2/L3 levels
pub struct MultiLevelCache<K, V>
where
    K: Eq + Hash + Clone + Send + Sync + 'static,
    V: Clone + Send + Sync + 'static,
{
    /// L1 cache - hot data
    l1: LruTtlCache<K, V>,
    /// L2 cache - warm data
    l2: LruTtlCache<K, V>,
    /// L3 cache - cold data
    l3: LruTtlCache<K, V>,
    
    // Statistics
    l1_hits: AtomicU64,
    l1_misses: AtomicU64,
    l2_hits: AtomicU64,
    l2_misses: AtomicU64,
    l3_hits: AtomicU64,
    l3_misses: AtomicU64,
    total_requests: AtomicU64,
}

impl<K, V> MultiLevelCache<K, V>
where
    K: Eq + Hash + Clone + Send + Sync + 'static,
    V: Clone + Send + Sync + 'static,
{
    /// Create a new multi-level cache with default sizes
    pub fn new() -> Self {
        Self::with_sizes(100, 500, 1000)
    }
    
    /// Create a new multi-level cache with specified sizes
    pub fn with_sizes(l1_size: usize, l2_size: usize, l3_size: usize) -> Self {
        Self {
            l1: LruTtlCache::new(l1_size, 300),    // 5 minutes TTL
            l2: LruTtlCache::new(l2_size, 900),    // 15 minutes TTL
            l3: LruTtlCache::new(l3_size, 1800),   // 30 minutes TTL
            l1_hits: AtomicU64::new(0),
            l1_misses: AtomicU64::new(0),
            l2_hits: AtomicU64::new(0),
            l2_misses: AtomicU64::new(0),
            l3_hits: AtomicU64::new(0),
            l3_misses: AtomicU64::new(0),
            total_requests: AtomicU64::new(0),
        }
    }
    
    /// Create with custom TTLs
    pub fn with_ttls(
        l1_size: usize, l1_ttl: u64,
        l2_size: usize, l2_ttl: u64,
        l3_size: usize, l3_ttl: u64,
    ) -> Self {
        Self {
            l1: LruTtlCache::new(l1_size, l1_ttl),
            l2: LruTtlCache::new(l2_size, l2_ttl),
            l3: LruTtlCache::new(l3_size, l3_ttl),
            l1_hits: AtomicU64::new(0),
            l1_misses: AtomicU64::new(0),
            l2_hits: AtomicU64::new(0),
            l2_misses: AtomicU64::new(0),
            l3_hits: AtomicU64::new(0),
            l3_misses: AtomicU64::new(0),
            total_requests: AtomicU64::new(0),
        }
    }
    
    /// Get a value from the cache, checking all levels
    pub fn get(&self, key: &K) -> Option<V> {
        self.total_requests.fetch_add(1, Ordering::Relaxed);
        
        // Try L1 first (hot data)
        if let Some(value) = self.l1.get(key) {
            self.l1_hits.fetch_add(1, Ordering::Relaxed);
            return Some(value);
        }
        self.l1_misses.fetch_add(1, Ordering::Relaxed);
        
        // Try L2 (warm data)
        if let Some(value) = self.l2.get(key) {
            self.l2_hits.fetch_add(1, Ordering::Relaxed);
            // Promote to L1
            self.l1.set(key.clone(), value.clone());
            return Some(value);
        }
        self.l2_misses.fetch_add(1, Ordering::Relaxed);
        
        // Try L3 (cold data)
        if let Some(value) = self.l3.get(key) {
            self.l3_hits.fetch_add(1, Ordering::Relaxed);
            // Promote to L2
            self.l2.set(key.clone(), value.clone());
            return Some(value);
        }
        self.l3_misses.fetch_add(1, Ordering::Relaxed);
        
        None
    }
    
    /// Set a value in the cache (goes to L3 by default).
    /// Invalidates the key at the other levels so a subsequent `get()` can
    /// never return a stale value from a level that wasn't updated.
    pub fn set(&self, key: K, value: V) {
        self.l1.remove(&key);
        self.l2.remove(&key);
        self.l3.set(key, value);
    }

    /// Set a value at a specific cache level.
    /// Invalidates the key at the other levels so a subsequent `get()` can
    /// never return a stale value from a level that wasn't updated.
    pub fn set_at_level(&self, key: K, value: V, level: CacheLevel) {
        match level {
            CacheLevel::L1 => {
                self.l2.remove(&key);
                self.l3.remove(&key);
                self.l1.set(key, value);
            }
            CacheLevel::L2 => {
                self.l1.remove(&key);
                self.l3.remove(&key);
                self.l2.set(key, value);
            }
            CacheLevel::L3 => {
                self.l1.remove(&key);
                self.l2.remove(&key);
                self.l3.set(key, value);
            }
        }
    }

    /// Set a value as hot (L1).
    /// Invalidates the key at the other levels so a subsequent `get()` can
    /// never return a stale value from a level that wasn't updated.
    pub fn set_hot(&self, key: K, value: V) {
        self.l2.remove(&key);
        self.l3.remove(&key);
        self.l1.set(key, value);
    }

    /// Set a value as warm (L2).
    /// Invalidates the key at the other levels so a subsequent `get()` can
    /// never return a stale value from a level that wasn't updated.
    pub fn set_warm(&self, key: K, value: V) {
        self.l1.remove(&key);
        self.l3.remove(&key);
        self.l2.set(key, value);
    }
    
    /// Remove a value from all cache levels
    pub fn remove(&self, key: &K) {
        self.l1.remove(key);
        self.l2.remove(key);
        self.l3.remove(key);
    }
    
    /// Check if a key exists in any cache level
    pub fn contains(&self, key: &K) -> bool {
        self.l1.contains(key) || self.l2.contains(key) || self.l3.contains(key)
    }
    
    /// Get the level where a key is cached
    pub fn get_level(&self, key: &K) -> Option<CacheLevel> {
        if self.l1.contains(key) {
            Some(CacheLevel::L1)
        } else if self.l2.contains(key) {
            Some(CacheLevel::L2)
        } else if self.l3.contains(key) {
            Some(CacheLevel::L3)
        } else {
            None
        }
    }
    
    /// Clear all cache levels
    pub fn clear(&self) {
        self.l1.clear();
        self.l2.clear();
        self.l3.clear();
    }
    
    /// Get the total number of cached items
    pub fn len(&self) -> usize {
        self.l1.len() + self.l2.len() + self.l3.len()
    }
    
    /// Check if all caches are empty
    pub fn is_empty(&self) -> bool {
        self.l1.is_empty() && self.l2.is_empty() && self.l3.is_empty()
    }
    
    /// Get cache statistics
    pub fn stats(&self) -> CacheStats {
        let l1_hits = self.l1_hits.load(Ordering::Relaxed);
        let l2_hits = self.l2_hits.load(Ordering::Relaxed);
        let l3_hits = self.l3_hits.load(Ordering::Relaxed);
        let total = self.total_requests.load(Ordering::Relaxed);
        
        let total_hits = l1_hits + l2_hits + l3_hits;
        let hit_rate = if total > 0 {
            total_hits as f64 / total as f64
        } else {
            0.0
        };
        
        CacheStats {
            l1_hits,
            l1_misses: self.l1_misses.load(Ordering::Relaxed),
            l2_hits,
            l2_misses: self.l2_misses.load(Ordering::Relaxed),
            l3_hits,
            l3_misses: self.l3_misses.load(Ordering::Relaxed),
            total_requests: total,
            hit_rate,
        }
    }
    
    /// Prune expired entries from all levels
    pub fn prune_expired(&self) -> usize {
        self.l1.prune_expired() + self.l2.prune_expired() + self.l3.prune_expired()
    }
    
    /// Get sizes of each cache level
    pub fn level_sizes(&self) -> (usize, usize, usize) {
        (self.l1.len(), self.l2.len(), self.l3.len())
    }
}

impl<K, V> Default for MultiLevelCache<K, V>
where
    K: Eq + Hash + Clone + Send + Sync + 'static,
    V: Clone + Send + Sync + 'static,
{
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_multi_level_basic() {
        let cache: MultiLevelCache<String, i32> = MultiLevelCache::new();
        
        cache.set("a".to_string(), 1);
        assert_eq!(cache.get(&"a".to_string()), Some(1));
    }
    
    #[test]
    fn test_promotion() {
        let cache: MultiLevelCache<String, i32> = MultiLevelCache::new();
        
        // Set in L3
        cache.set("a".to_string(), 1);
        assert_eq!(cache.get_level(&"a".to_string()), Some(CacheLevel::L3));
        
        // First access promotes to L2
        cache.get(&"a".to_string());
        assert!(cache.l2.contains(&"a".to_string()));
        
        // Second access promotes to L1
        cache.get(&"a".to_string());
        assert!(cache.l1.contains(&"a".to_string()));
    }
    
    #[test]
    fn test_set_hot() {
        let cache: MultiLevelCache<String, i32> = MultiLevelCache::new();
        
        cache.set_hot("hot".to_string(), 42);
        assert_eq!(cache.get_level(&"hot".to_string()), Some(CacheLevel::L1));
    }
    
    #[test]
    fn test_stats() {
        let cache: MultiLevelCache<String, i32> = MultiLevelCache::new();
        
        cache.set_hot("a".to_string(), 1);
        cache.get(&"a".to_string()); // L1 hit
        cache.get(&"a".to_string()); // L1 hit
        cache.get(&"b".to_string()); // Miss
        
        let stats = cache.stats();
        assert_eq!(stats.l1_hits, 2);
        assert_eq!(stats.total_requests, 3);
    }
}
