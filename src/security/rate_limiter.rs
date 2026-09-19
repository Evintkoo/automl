use serde::{Serialize, Deserialize};
use std::collections::{HashMap, VecDeque};
use std::time::Instant;
use dashmap::DashMap;
use parking_lot::Mutex;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RateLimitAlgorithm {
    SlidingWindow,
    TokenBucket,
    FixedWindow,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RateLimitConfig {
    pub requests_per_window: u32,
    pub window_seconds: u64,
    pub algorithm: RateLimitAlgorithm,
    pub burst_size: u32,
}

impl Default for RateLimitConfig {
    fn default() -> Self {
        Self {
            requests_per_window: 100,
            window_seconds: 60,
            algorithm: RateLimitAlgorithm::TokenBucket,
            burst_size: 10,
        }
    }
}

#[derive(Debug)]
struct ClientState {
    /// Timestamps of requests within the current sliding window, oldest first
    /// (always pushed in increasing-time order at the back). A `VecDeque` lets
    /// stale entries be popped from the front in O(1) amortized time instead
    /// of rescanning/rebuilding the whole collection via `retain` on every
    /// request.
    request_times: VecDeque<Instant>,
    tokens: f64,
    last_refill: Instant,
    window_start: Instant,
    window_count: u32,
    total_requests: usize,
    blocked_requests: usize,
}

impl ClientState {
    fn new(burst_size: u32) -> Self {
        let now = Instant::now();
        Self {
            request_times: VecDeque::new(),
            tokens: burst_size as f64,
            last_refill: now,
            window_start: now,
            window_count: 0,
            total_requests: 0,
            blocked_requests: 0,
        }
    }
}

/// Per-client state protected by a lightweight Mutex.
/// Using DashMap<String, Mutex<ClientState>> instead of Mutex<HashMap<...>> means
/// different clients never block each other — only concurrent requests from the
/// *same* client contend on the same lock.
#[derive(Debug)]
pub struct RateLimiter {
    config: RateLimitConfig,
    clients: DashMap<String, Mutex<ClientState>>,
}

impl RateLimiter {
    pub fn new(config: RateLimitConfig) -> Self {
        Self {
            config,
            clients: DashMap::new(),
        }
    }

    pub fn is_allowed(&self, client_id: &str) -> bool {
        // Evict stale entries when the map grows large.
        // We do this outside of any per-client lock to avoid blocking other clients.
        if self.clients.len() > 10_000 {
            let window = std::time::Duration::from_secs(self.config.window_seconds * 2);
            let now = Instant::now();
            self.clients.retain(|_, cell| {
                let state = cell.lock();
                now.duration_since(state.window_start) < window
            });
        }

        let entry = self.clients
            .entry(client_id.to_string())
            .or_insert_with(|| Mutex::new(ClientState::new(self.config.burst_size)));

        let mut state = entry.value().lock();
        state.total_requests += 1;

        let allowed = match self.config.algorithm {
            RateLimitAlgorithm::SlidingWindow => self.check_sliding_window(&mut state),
            RateLimitAlgorithm::TokenBucket   => self.check_token_bucket(&mut state),
            RateLimitAlgorithm::FixedWindow   => self.check_fixed_window(&mut state),
        };

        if !allowed {
            state.blocked_requests += 1;
        }
        allowed
    }

    pub fn get_remaining(&self, client_id: &str) -> u32 {
        let entry = match self.clients.get(client_id) {
            Some(e) => e,
            None => return self.config.requests_per_window,
        };
        let mut state = entry.value().lock();
        match self.config.algorithm {
            RateLimitAlgorithm::SlidingWindow => {
                let now = Instant::now();
                let window = std::time::Duration::from_secs(self.config.window_seconds);
                while let Some(&front) = state.request_times.front() {
                    if now.duration_since(front) < window {
                        break;
                    }
                    state.request_times.pop_front();
                }
                let count = state.request_times.len() as u32;
                self.config.requests_per_window.saturating_sub(count)
            }
            RateLimitAlgorithm::TokenBucket => {
                self.refill_tokens(&mut state);
                state.tokens as u32
            }
            RateLimitAlgorithm::FixedWindow => {
                self.config.requests_per_window.saturating_sub(state.window_count)
            }
        }
    }

    pub fn reset_client(&self, client_id: &str) {
        self.clients.remove(client_id);
    }

    pub fn get_metrics(&self) -> HashMap<String, usize> {
        let mut metrics = HashMap::new();
        let mut total_requests: usize = 0;
        let mut total_blocked: usize = 0;

        for entry in self.clients.iter() {
            let state = entry.value().lock();
            total_requests += state.total_requests;
            total_blocked += state.blocked_requests;
            metrics.insert(format!("client_{}_requests", entry.key()), state.total_requests);
            metrics.insert(format!("client_{}_blocked", entry.key()), state.blocked_requests);
        }

        metrics.insert("total_requests".to_string(), total_requests);
        metrics.insert("total_blocked".to_string(), total_blocked);
        metrics.insert("active_clients".to_string(), self.clients.len());
        metrics
    }

    fn check_sliding_window(&self, state: &mut ClientState) -> bool {
        let now = Instant::now();
        let window = std::time::Duration::from_secs(self.config.window_seconds);
        while let Some(&front) = state.request_times.front() {
            if now.duration_since(front) < window {
                break;
            }
            state.request_times.pop_front();
        }

        if (state.request_times.len() as u32) < self.config.requests_per_window {
            state.request_times.push_back(now);
            true
        } else {
            false
        }
    }

    fn check_token_bucket(&self, state: &mut ClientState) -> bool {
        self.refill_tokens(state);
        if state.tokens >= 1.0 {
            state.tokens -= 1.0;
            true
        } else {
            false
        }
    }

    fn check_fixed_window(&self, state: &mut ClientState) -> bool {
        let now = Instant::now();
        let window = std::time::Duration::from_secs(self.config.window_seconds);

        if now.duration_since(state.window_start) >= window {
            state.window_start = now;
            state.window_count = 0;
        }

        if state.window_count < self.config.requests_per_window {
            state.window_count += 1;
            true
        } else {
            false
        }
    }

    fn refill_tokens(&self, state: &mut ClientState) {
        let now = Instant::now();
        let elapsed = now.duration_since(state.last_refill).as_secs_f64();
        let rate = self.config.requests_per_window as f64 / self.config.window_seconds as f64;
        state.tokens = (state.tokens + elapsed * rate).min(self.config.burst_size as f64);
        state.last_refill = now;
    }
}
