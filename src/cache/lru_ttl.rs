//! LRU Cache with TTL Support
//!
//! A thread-safe LRU cache with time-to-live expiration.
//! Uses a single coordinated lock to eliminate lock overhead,
//! improve cache locality, and prevent deadlocks.

use std::collections::HashMap;
use std::hash::Hash;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::RwLock;
use std::time::{Duration, Instant};

/// A cache entry with value and metadata
#[derive(Debug, Clone)]
pub struct CacheEntry<V> {
    /// The cached value
    pub value: V,
    /// When the entry was created
    pub created_at: Instant,
    /// When the entry was last accessed
    pub last_accessed: Instant,
    /// Number of times this entry was accessed
    pub access_count: u64,
}

impl<V> CacheEntry<V> {
    /// Create a new cache entry
    pub fn new(value: V) -> Self {
        let now = Instant::now();
        Self {
            value,
            created_at: now,
            last_accessed: now,
            access_count: 1,
        }
    }

    /// Check if this entry has expired
    pub fn is_expired(&self, ttl: Duration) -> bool {
        self.created_at.elapsed() > ttl
    }

    /// Get the age of this entry
    pub fn age(&self) -> Duration {
        self.created_at.elapsed()
    }
}

/// Node in the LRU linked list
struct LruNode<K> {
    key: K,
    prev: Option<usize>,
    next: Option<usize>,
}

/// Inner state protected by a single lock
struct CacheInner<K, V> {
    /// The cache storage
    cache: HashMap<K, (usize, CacheEntry<V>)>,
    /// LRU order tracking (index -> node)
    lru_list: Vec<LruNode<K>>,
    /// Head of LRU list (most recently used)
    head: Option<usize>,
    /// Tail of LRU list (least recently used)
    tail: Option<usize>,
    /// Free list for recycling nodes
    free_list: Vec<usize>,
}

impl<K: Eq + Hash + Clone, V: Clone> CacheInner<K, V> {
    fn new(max_size: usize) -> Self {
        Self {
            cache: HashMap::with_capacity(max_size),
            lru_list: Vec::with_capacity(max_size),
            head: None,
            tail: None,
            free_list: Vec::new(),
        }
    }

    fn allocate_node(&mut self, key: K) -> usize {
        if let Some(idx) = self.free_list.pop() {
            if idx < self.lru_list.len() {
                self.lru_list[idx] = LruNode {
                    key,
                    prev: None,
                    next: None,
                };
                return idx;
            }
        }
        let idx = self.lru_list.len();
        self.lru_list.push(LruNode {
            key,
            prev: None,
            next: None,
        });
        idx
    }

    fn free_node(&mut self, idx: usize) {
        self.free_list.push(idx);
    }

    fn push_to_head(&mut self, idx: usize) {
        if idx >= self.lru_list.len() {
            return;
        }
        if let Some(old_head) = self.head {
            if old_head < self.lru_list.len() {
                self.lru_list[old_head].prev = Some(idx);
            }
            self.lru_list[idx].next = Some(old_head);
            self.lru_list[idx].prev = None;
        } else {
            self.tail = Some(idx);
            self.lru_list[idx].next = None;
            self.lru_list[idx].prev = None;
        }
        self.head = Some(idx);
    }

    fn remove_node(&mut self, idx: usize) {
        if idx >= self.lru_list.len() {
            return;
        }
        let prev = self.lru_list[idx].prev;
        let next = self.lru_list[idx].next;

        if let Some(prev_idx) = prev {
            if prev_idx < self.lru_list.len() {
                self.lru_list[prev_idx].next = next;
            }
        } else {
            self.head = next;
        }

        if let Some(next_idx) = next {
            if next_idx < self.lru_list.len() {
                self.lru_list[next_idx].prev = prev;
            }
        } else {
            self.tail = prev;
        }

        self.lru_list[idx].prev = None;
        self.lru_list[idx].next = None;
    }

    fn move_to_head(&mut self, idx: usize) {
        self.remove_node(idx);
        self.push_to_head(idx);
    }

    fn evict_lru(&mut self) {
        let key_to_remove = self.tail.and_then(|idx| {
            if idx < self.lru_list.len() {
                Some(self.lru_list[idx].key.clone())
            } else {
                None
            }
        });
        if let Some(key) = key_to_remove {
            self.remove_entry(&key);
        }
    }

    fn remove_entry(&mut self, key: &K) -> Option<V> {
        if let Some((node_idx, entry)) = self.cache.remove(key) {
            self.remove_node(node_idx);
            self.free_node(node_idx);
            Some(entry.value)
        } else {
            None
        }
    }
}

/// LRU Cache with TTL support
///
/// All internal state is protected by a single `RwLock`, eliminating the
/// overhead and deadlock risk of multiple independent locks.
pub struct LruTtlCache<K, V>
where
    K: Eq + Hash + Clone,
    V: Clone,
{
    /// Maximum cache size
    max_size: usize,
    /// Time-to-live for entries
    ttl: Duration,
    /// All mutable state under one lock
    inner: RwLock<CacheInner<K, V>>,
    /// Statistics (lock-free atomics for hot-path counters)
    hits: AtomicU64,
    misses: AtomicU64,
}

impl<K, V> LruTtlCache<K, V>
where
    K: Eq + Hash + Clone,
    V: Clone,
{
    /// Create a new LRU-TTL cache
    pub fn new(max_size: usize, ttl_seconds: u64) -> Self {
        Self {
            max_size,
            ttl: Duration::from_secs(ttl_seconds),
            inner: RwLock::new(CacheInner::new(max_size)),
            hits: AtomicU64::new(0),
            misses: AtomicU64::new(0),
        }
    }

    /// Get an entry from the cache
    pub fn get(&self, key: &K) -> Option<V> {
        let mut inner = self.inner.write().ok()?;

        if let Some((node_idx, entry)) = inner.cache.get(key) {
            if entry.is_expired(self.ttl) {
                // Entry expired — remove it
                let key_clone = key.clone();
                inner.remove_entry(&key_clone);
                drop(inner);
                self.misses.fetch_add(1, Ordering::Relaxed);
                return None;
            }

            let value = entry.value.clone();
            let node_idx = *node_idx;

            // Update LRU order and access metadata
            inner.move_to_head(node_idx);
            if let Some((_, entry)) = inner.cache.get_mut(key) {
                entry.last_accessed = Instant::now();
                entry.access_count += 1;
            }

            drop(inner);
            self.hits.fetch_add(1, Ordering::Relaxed);
            Some(value)
        } else {
            drop(inner);
            self.misses.fetch_add(1, Ordering::Relaxed);
            None
        }
    }

    /// Set an entry in the cache
    pub fn set(&self, key: K, value: V) {
        let mut inner = match self.inner.write() {
            Ok(g) => g,
            Err(_) => return,
        };

        // If key already exists, update it
        if let Some((node_idx, entry)) = inner.cache.get_mut(&key) {
            entry.value = value;
            entry.last_accessed = Instant::now();
            let idx = *node_idx;
            inner.move_to_head(idx);
            return;
        }

        // If at capacity, evict LRU entry
        if inner.cache.len() >= self.max_size {
            inner.evict_lru();
        }

        // Add new entry
        let entry = CacheEntry::new(value);
        let node_idx = inner.allocate_node(key.clone());
        inner.cache.insert(key, (node_idx, entry));
        inner.push_to_head(node_idx);
    }

    /// Remove an entry from the cache
    pub fn remove(&self, key: &K) -> Option<V> {
        let mut inner = self.inner.write().ok()?;
        inner.remove_entry(key)
    }

    /// Check if a key exists in the cache and has not expired
    pub fn contains(&self, key: &K) -> bool {
        self.inner.read()
            .map(|g| {
                g.cache.get(key)
                    .map(|(_, entry)| !entry.is_expired(self.ttl))
                    .unwrap_or(false)
            })
            .unwrap_or(false)
    }

    /// Get the current cache size
    pub fn len(&self) -> usize {
        self.inner.read().map(|g| g.cache.len()).unwrap_or(0)
    }

    /// Check if the cache is empty
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Clear the cache
    pub fn clear(&self) {
        if let Ok(mut inner) = self.inner.write() {
            inner.cache.clear();
            inner.lru_list.clear();
            inner.head = None;
            inner.tail = None;
            inner.free_list.clear();
        }
    }

    /// Get cache statistics
    pub fn stats(&self) -> (u64, u64, f64) {
        let hits = self.hits.load(Ordering::Relaxed);
        let misses = self.misses.load(Ordering::Relaxed);
        let total = hits + misses;
        let hit_rate = if total > 0 {
            hits as f64 / total as f64
        } else {
            0.0
        };
        (hits, misses, hit_rate)
    }

    /// Prune expired entries
    pub fn prune_expired(&self) -> usize {
        let mut inner = match self.inner.write() {
            Ok(g) => g,
            Err(_) => return 0,
        };

        let ttl = self.ttl;
        let keys_to_remove: Vec<K> = inner.cache.iter()
            .filter(|(_, (_, entry))| entry.is_expired(ttl))
            .map(|(k, _)| k.clone())
            .collect();

        let count = keys_to_remove.len();
        for key in keys_to_remove {
            inner.remove_entry(&key);
        }
        count
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::thread;

    #[test]
    fn test_lru_cache_basic() {
        let cache: LruTtlCache<String, i32> = LruTtlCache::new(3, 60);

        cache.set("a".to_string(), 1);
        cache.set("b".to_string(), 2);
        cache.set("c".to_string(), 3);

        assert_eq!(cache.get(&"a".to_string()), Some(1));
        assert_eq!(cache.get(&"b".to_string()), Some(2));
        assert_eq!(cache.get(&"c".to_string()), Some(3));
    }

    #[test]
    fn test_lru_eviction() {
        let cache: LruTtlCache<String, i32> = LruTtlCache::new(3, 60);

        cache.set("a".to_string(), 1);
        cache.set("b".to_string(), 2);
        cache.set("c".to_string(), 3);

        // Access "a" to make it recently used
        cache.get(&"a".to_string());

        // Add "d", should evict "b" (LRU)
        cache.set("d".to_string(), 4);

        assert_eq!(cache.get(&"a".to_string()), Some(1));
        assert_eq!(cache.get(&"b".to_string()), None); // Evicted
        assert_eq!(cache.get(&"c".to_string()), Some(3));
        assert_eq!(cache.get(&"d".to_string()), Some(4));
    }

    #[test]
    fn test_ttl_expiration() {
        let cache: LruTtlCache<String, i32> = LruTtlCache::new(10, 1); // 1 second TTL

        cache.set("a".to_string(), 1);
        assert_eq!(cache.get(&"a".to_string()), Some(1));

        // Wait for expiration
        thread::sleep(Duration::from_secs(2));

        assert_eq!(cache.get(&"a".to_string()), None);
    }

    #[test]
    fn test_cache_stats() {
        let cache: LruTtlCache<String, i32> = LruTtlCache::new(10, 60);

        cache.set("a".to_string(), 1);
        cache.get(&"a".to_string()); // Hit
        cache.get(&"a".to_string()); // Hit
        cache.get(&"b".to_string()); // Miss

        let (hits, misses, hit_rate) = cache.stats();
        assert_eq!(hits, 2);
        assert_eq!(misses, 1);
        assert!((hit_rate - 0.666).abs() < 0.01);
    }
}
