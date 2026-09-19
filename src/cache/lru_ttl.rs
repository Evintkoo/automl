//! LRU Cache with TTL Support
//!
//! A thread-safe, sharded LRU cache with time-to-live expiration.

use std::collections::hash_map::DefaultHasher;
use std::collections::HashMap;
use std::hash::{Hash, Hasher};
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

/// Inner state protected by a single lock: one shard's worth of the cache.
struct CacheInner<K, V> {
    /// This shard's share of the cache's total capacity
    max_size: usize,
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
            max_size,
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

/// Target entries per shard when sizing the shard count. A cache stays a
/// single shard (byte-for-byte the original unsharded design, with exact
/// global LRU order) until it's large enough for cross-key lock contention
/// between concurrent callers to plausibly matter.
const SHARD_TARGET_ENTRIES: usize = 64;
/// Upper bound on shard count — enough to spread contention across a typical
/// server's concurrent request load without fragmenting capacity too finely.
const MAX_SHARDS: usize = 16;

/// LRU Cache with TTL support
///
/// Sharded by key hash into independent, separately-locked partitions, so
/// concurrent callers touching different keys (e.g. concurrent prediction
/// requests keyed by distinct input hashes) don't contend on the same lock.
/// Eviction is exact LRU *within* a shard, not globally across the whole
/// cache — the standard trade every high-throughput concurrent cache makes
/// (Caffeine, Moka, memcached's slab classes), which doesn't matter for a
/// performance-only cache like this one.
///
/// Shard count scales with capacity (see `SHARD_TARGET_ENTRIES`,
/// `MAX_SHARDS`) and collapses to a single shard for small caches, which
/// keeps small-cache behavior — including exact global LRU order — identical
/// to the original unsharded design.
pub struct LruTtlCache<K, V>
where
    K: Eq + Hash + Clone,
    V: Clone,
{
    /// Time-to-live for entries
    ttl: Duration,
    /// Independent, separately-locked partitions of the cache
    shards: Vec<RwLock<CacheInner<K, V>>>,
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
        let num_shards = (max_size / SHARD_TARGET_ENTRIES).clamp(1, MAX_SHARDS);
        let shard_max_size = max_size.div_ceil(num_shards);
        let shards = (0..num_shards)
            .map(|_| RwLock::new(CacheInner::new(shard_max_size)))
            .collect();

        Self {
            ttl: Duration::from_secs(ttl_seconds),
            shards,
            hits: AtomicU64::new(0),
            misses: AtomicU64::new(0),
        }
    }

    /// Which shard a key belongs to. Stable for the cache's lifetime since
    /// the shard count never changes after construction.
    fn shard_for(&self, key: &K) -> &RwLock<CacheInner<K, V>> {
        let mut hasher = DefaultHasher::new();
        key.hash(&mut hasher);
        let idx = (hasher.finish() as usize) % self.shards.len();
        &self.shards[idx]
    }

    /// Get an entry from the cache
    pub fn get(&self, key: &K) -> Option<V> {
        let mut inner = self.shard_for(key).write().ok()?;

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
        let mut inner = match self.shard_for(&key).write() {
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

        // If this shard is at capacity, evict its LRU entry
        if inner.cache.len() >= inner.max_size {
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
        let mut inner = self.shard_for(key).write().ok()?;
        inner.remove_entry(key)
    }

    /// Check if a key exists in the cache and has not expired
    pub fn contains(&self, key: &K) -> bool {
        self.shard_for(key)
            .read()
            .map(|g| {
                g.cache.get(key)
                    .map(|(_, entry)| !entry.is_expired(self.ttl))
                    .unwrap_or(false)
            })
            .unwrap_or(false)
    }

    /// Get the current cache size (summed across all shards)
    pub fn len(&self) -> usize {
        self.shards
            .iter()
            .map(|s| s.read().map(|g| g.cache.len()).unwrap_or(0))
            .sum()
    }

    /// Check if the cache is empty
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Clear the cache
    pub fn clear(&self) {
        for shard in &self.shards {
            if let Ok(mut inner) = shard.write() {
                inner.cache.clear();
                inner.lru_list.clear();
                inner.head = None;
                inner.tail = None;
                inner.free_list.clear();
            }
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

    /// Prune expired entries (across all shards)
    pub fn prune_expired(&self) -> usize {
        let ttl = self.ttl;
        let mut count = 0;

        for shard in &self.shards {
            let mut inner = match shard.write() {
                Ok(g) => g,
                Err(_) => continue,
            };

            let keys_to_remove: Vec<K> = inner.cache.iter()
                .filter(|(_, (_, entry))| entry.is_expired(ttl))
                .map(|(k, _)| k.clone())
                .collect();

            count += keys_to_remove.len();
            for key in keys_to_remove {
                inner.remove_entry(&key);
            }
        }

        count
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
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
        // Capacity 3 stays a single shard (below SHARD_TARGET_ENTRIES), so
        // eviction order is exact global LRU, identical to the pre-sharding
        // design.
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

    #[test]
    fn test_sharded_cache_stores_and_retrieves_all_keys() {
        // Capacity well above SHARD_TARGET_ENTRIES so this actually spans
        // multiple shards; every key must still round-trip correctly
        // regardless of which shard it hashes into.
        let cache: LruTtlCache<usize, usize> = LruTtlCache::new(1000, 60);

        for i in 0..500 {
            cache.set(i, i * 10);
        }
        for i in 0..500 {
            assert_eq!(cache.get(&i), Some(i * 10), "key {i} did not round-trip");
        }
        assert_eq!(cache.len(), 500);

        cache.remove(&250);
        assert_eq!(cache.get(&250), None);
        assert_eq!(cache.len(), 499);
    }

    #[test]
    fn test_sharded_cache_concurrent_access() {
        let cache: Arc<LruTtlCache<usize, usize>> = Arc::new(LruTtlCache::new(2000, 60));
        let mut handles = Vec::new();

        for t in 0..8 {
            let cache = Arc::clone(&cache);
            handles.push(thread::spawn(move || {
                for i in 0..200 {
                    let key = t * 200 + i;
                    cache.set(key, key * 2);
                    assert_eq!(cache.get(&key), Some(key * 2));
                }
            }));
        }

        for h in handles {
            h.join().unwrap();
        }

        // Every key was distinct across threads and within total capacity,
        // so nothing should have been evicted.
        assert_eq!(cache.len(), 1600);
    }

    #[test]
    fn test_sharded_cache_respects_approximate_capacity() {
        // Total capacity is spread across shards (ceil-divided), so it may
        // overshoot slightly, but must never grow unbounded.
        let cache: LruTtlCache<usize, usize> = LruTtlCache::new(500, 60);

        for i in 0..5000 {
            cache.set(i, i);
        }

        assert!(
            cache.len() <= 500 + MAX_SHARDS,
            "sharded cache grew far beyond its configured capacity: len={}",
            cache.len()
        );
    }
}
