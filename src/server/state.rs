//! Application state management

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;
use dashmap::DashMap;
use tokio::sync::RwLock;
use uuid::Uuid;
use polars::prelude::*;
use crate::cache::LruTtlCache;
use crate::inference::{InferenceConfig, InferenceEngine, InferenceStats};
use crate::security::{SecurityManager, SecurityConfig, RateLimiter, RateLimitConfig, AuditTrail};
use crate::preprocessing::DataPreprocessor;
use crate::training::TrainEngine;
use crate::provenance::ProvenanceTracker;
use crate::privacy::{RetentionManager, DataClassification};

use super::ServerConfig;

/// Training job status
#[derive(Debug, Clone, serde::Serialize)]
pub enum JobStatus {
    Pending,
    Running {
        progress: f64,
        message: String,
        /// Partial results streamed as they arrive (e.g. leaderboard entries, trial results)
        #[serde(skip_serializing_if = "Option::is_none")]
        partial_results: Option<Vec<serde_json::Value>>,
    },
    Completed { metrics: serde_json::Value },
    Failed { error: String },
}

/// Training job information
#[derive(Debug, Clone)]
pub struct TrainingJob {
    pub id: String,
    pub status: JobStatus,
    pub config: serde_json::Value,
    pub created_at: chrono::DateTime<chrono::Utc>,
    pub model_path: Option<PathBuf>,
    /// Identity fingerprint of the request that created this job (dataset + request
    /// body), used to detect duplicate/retried submissions before spawning a second
    /// expensive run. `None` for job kinds that don't participate in dedup.
    pub fingerprint: Option<String>,
}

/// Trained model information
#[derive(Debug, Clone, serde::Serialize)]
pub struct ModelInfo {
    pub id: String,
    pub name: String,
    pub task_type: String,
    pub metrics: serde_json::Value,
    pub created_at: String,
    pub path: PathBuf,
}

/// Cached evaluation data for Metrics Deep Dive insights
#[derive(Debug, Clone)]
pub struct InsightsEvaluation {
    pub task: crate::training::TaskType,
    pub classes: Vec<String>,
    pub y_true: Vec<f64>,
    pub y_pred: Vec<f64>,
    pub y_prob: Option<Vec<Vec<f64>>>,
}

/// Uploaded dataset information
#[derive(Debug, Clone)]
pub struct DatasetInfo {
    pub id: String,
    pub name: String,
    pub path: PathBuf,
    pub rows: usize,
    pub columns: usize,
    pub column_names: Vec<String>,
    pub dtypes: Vec<String>,
    /// When this dataset was stored — used for oldest-first eviction and to give
    /// `RetentionManager` a stable notion of dataset age.
    pub created_at: chrono::DateTime<chrono::Utc>,
}

/// In-memory model registry for caching loaded inference engines
pub struct ModelRegistry {
    engines: DashMap<String, Arc<InferenceEngine>>,
    max_cached: usize,
    /// Shared prediction cache across all models
    prediction_cache: Option<Arc<LruTtlCache<u64, Vec<f64>>>>,
}

impl ModelRegistry {
    pub fn new(max_cached: usize) -> Self {
        Self {
            engines: DashMap::new(),
            max_cached,
            prediction_cache: None,
        }
    }

    /// Create a new model registry with a shared prediction cache
    pub fn with_prediction_cache(mut self, cache: Arc<LruTtlCache<u64, Vec<f64>>>) -> Self {
        self.prediction_cache = Some(cache);
        self
    }

    /// Insert a pre-built engine into the registry (e.g. right after training).
    /// The shared prediction cache is injected automatically.
    pub fn insert(&self, model_id: String, mut engine: InferenceEngine) {
        if let Some(ref cache) = self.prediction_cache {
            engine.set_prediction_cache(Arc::clone(cache));
        }
        // Evict least-used entry if at capacity
        if self.engines.len() >= self.max_cached {
            let least_used = self.engines.iter()
                .min_by_key(|e| e.value().stats().total_predictions)
                .map(|e| e.key().clone());
            if let Some(key) = least_used {
                self.engines.remove(&key);
            }
        }
        self.engines.insert(model_id, Arc::new(engine));
    }

    /// Get a cached inference engine, or load from disk and cache it.
    /// Uses DashMap for lock-free fast-path reads (no async overhead on cache hit).
    pub async fn get_or_load(
        &self,
        model_id: &str,
        model_path: &str,
    ) -> crate::error::Result<Arc<InferenceEngine>> {
        // Fast path: DashMap read — no async overhead
        if let Some(engine) = self.engines.get(model_id) {
            return Ok(Arc::clone(engine.value()));
        }

        // Slow path: load from disk (blocking I/O, done without holding any lock)
        let mut engine = InferenceEngine::load(InferenceConfig::new(), None, model_path)?;

        // Inject shared prediction cache
        if let Some(ref cache) = self.prediction_cache {
            engine.set_prediction_cache(Arc::clone(cache));
        }

        engine.warmup()?;
        let engine = Arc::new(engine);

        // Evict least-used entry if at capacity
        if self.engines.len() >= self.max_cached {
            let least_used = self.engines.iter()
                .min_by_key(|e| e.value().stats().total_predictions)
                .map(|e| e.key().clone());
            if let Some(key) = least_used {
                self.engines.remove(&key);
            }
        }

        // entry().or_insert() is atomic — if another task raced us, we get their Arc back
        let entry = self.engines.entry(model_id.to_string()).or_insert_with(|| Arc::clone(&engine));
        Ok(Arc::clone(entry.value()))
    }

    /// Evict a model from the cache
    pub fn evict(&self, model_id: &str) {
        self.engines.remove(model_id);
    }

    /// Clear all cached models
    pub fn clear(&self) {
        self.engines.clear();
    }

    /// Number of currently cached models
    pub fn cached_count(&self) -> usize {
        self.engines.len()
    }

    /// Get stats for a cached model
    pub fn get_model_stats(&self, model_id: &str) -> Option<InferenceStats> {
        self.engines.get(model_id).map(|e| e.value().stats())
    }
}

/// Application state shared across handlers
pub struct AppState {
    pub config: ServerConfig,
    pub datasets: RwLock<HashMap<String, DatasetInfo>>,
    pub jobs: RwLock<HashMap<String, TrainingJob>>,
    /// `tokio::spawn` handles for in-flight training-style jobs, keyed by job id.
    /// Stored (rather than discarded) so a future cancel-job endpoint can abort a
    /// running job; nothing currently calls `.abort()` on these.
    pub job_handles: DashMap<String, tokio::task::JoinHandle<()>>,
    pub models: DashMap<String, ModelInfo>,
    /// The model id `/predict`-family handlers default to when a request
    /// doesn't specify its own `model_id` — mirrors `current_dataset_id`.
    /// Every insert into `models` unconditionally overwrites this with its
    /// own id, which is correct because each live insert stamps its
    /// `created_at` with `Utc::now()` at insert time, so insertion order and
    /// created_at order coincide; disk-restored models (whose `created_at`
    /// is the file's mtime, not insertion order) are handled separately by
    /// seeding this from a one-time max-by-created_at scan right after
    /// `restore_models_from_disk` at startup. Without this cache, every
    /// prediction request that omits `model_id` did a full linear scan of
    /// every registered model to find the newest one.
    pub latest_model_id: RwLock<Option<String>>,
    /// Actual per-dataset DataFrame storage, keyed by dataset id. Kept separate from
    /// `datasets` (metadata only) so that multiple datasets uploaded/imported
    /// concurrently can coexist instead of one silently clobbering another via a
    /// single shared "current" buffer.
    pub dataset_frames: RwLock<HashMap<String, DataFrame>>,
    /// The dataset id operations default to when a request doesn't specify its own
    /// `dataset_id` — single source of truth for "the currently active dataset".
    pub current_dataset_id: RwLock<Option<String>>,
    pub preprocessor: RwLock<Option<DataPreprocessor>>,
    pub model_registry: ModelRegistry,
    /// Shared prediction cache for all inference engines
    pub prediction_cache: Arc<LruTtlCache<u64, Vec<f64>>>,
    /// Store trained engines for post-training analysis (explainability, etc.)
    pub train_engines: DashMap<String, TrainEngine>,
    /// Cache of evaluation data (y_true/y_pred) for Metrics Deep Dive
    pub insights_cache: DashMap<String, InsightsEvaluation>,
    /// Quality reports produced by QualityPipeline, keyed by model_id.
    pub quality_reports: dashmap::DashMap<String, crate::quality::QualityReport>,
    /// Fitted OOD detectors, keyed by model_id.
    pub ood_detectors: dashmap::DashMap<String, crate::quality::post_training::OodDetector>,
    /// Fitted conformal predictors for regression, keyed by model_id.
    pub conformal_predictors: dashmap::DashMap<String, crate::quality::post_training::ConformalPredictor>,
    /// Store completed hyperopt studies for history
    pub completed_studies: RwLock<Vec<serde_json::Value>>,
    /// Ring buffer of recent prediction latencies in microseconds (capped at 1000).
    pub prediction_latencies: std::sync::Mutex<std::collections::VecDeque<u64>>,
    /// Security manager for auth, input validation, audit logging
    pub security_manager: Arc<SecurityManager>,
    /// Rate limiter for API throttling
    pub rate_limiter: Arc<RateLimiter>,
    /// Data provenance tracker (ISO 5259, 5338)
    pub provenance_tracker: ProvenanceTracker,
    /// Tamper-evident audit trail (ISO 27001, TR 24028)
    pub audit_trail: AuditTrail,
    /// Data retention manager (ISO 27701)
    pub retention_manager: RetentionManager,
}

impl AppState {
    pub fn new(config: ServerConfig) -> Self {
        // Initialize security from env vars
        let security_config = SecurityConfig {
            api_keys: std::env::var("SECURITY_API_KEYS")
                .map(|s| s.split(',').map(|k| k.trim().to_string()).filter(|k| !k.is_empty()).collect())
                .unwrap_or_default(),
            jwt_secret: std::env::var("SECURITY_JWT_SECRET").ok(),
            jwt_expiry_secs: std::env::var("SECURITY_JWT_EXPIRY")
                .ok().and_then(|s| s.parse().ok()).unwrap_or(3600),
            enable_api_key_auth: std::env::var("SECURITY_ENABLE_API_KEY")
                .map(|v| v == "true" || v == "1").unwrap_or(false),
            enable_jwt_auth: std::env::var("SECURITY_ENABLE_JWT")
                .map(|v| v == "true" || v == "1").unwrap_or(false),
            blocked_ips: std::env::var("SECURITY_BLOCKED_IPS")
                .map(|s| s.split(',').map(|ip| ip.trim().to_string()).filter(|ip| !ip.is_empty()).collect())
                .unwrap_or_default(),
            audit_log_enabled: std::env::var("SECURITY_AUDIT_LOG")
                .map(|v| v != "false" && v != "0").unwrap_or(true),
        };

        let rate_limit_config = RateLimitConfig {
            requests_per_window: std::env::var("SECURITY_RATE_LIMIT")
                .ok().and_then(|s| s.parse().ok()).unwrap_or(200),
            window_seconds: std::env::var("SECURITY_RATE_WINDOW")
                .ok().and_then(|s| s.parse().ok()).unwrap_or(60),
            burst_size: std::env::var("SECURITY_RATE_BURST")
                .ok().and_then(|s| s.parse().ok()).unwrap_or(200),
            algorithm: crate::security::RateLimitAlgorithm::TokenBucket,
            ..Default::default()
        };

        let prediction_cache = Arc::new(LruTtlCache::new(1000, 300));

        Self {
            config,
            datasets: RwLock::new(HashMap::new()),
            jobs: RwLock::new(HashMap::new()),
            job_handles: DashMap::new(),
            models: DashMap::new(),
            latest_model_id: RwLock::new(None),
            dataset_frames: RwLock::new(HashMap::new()),
            current_dataset_id: RwLock::new(None),
            preprocessor: RwLock::new(None),
            model_registry: ModelRegistry::new(10)
                .with_prediction_cache(Arc::clone(&prediction_cache)),
            prediction_cache,
            train_engines: DashMap::new(),
            insights_cache: DashMap::new(),
            quality_reports: dashmap::DashMap::new(),
            ood_detectors: dashmap::DashMap::new(),
            conformal_predictors: dashmap::DashMap::new(),
            completed_studies: RwLock::new(Vec::new()),
            prediction_latencies: std::sync::Mutex::new(std::collections::VecDeque::new()),
            security_manager: Arc::new(SecurityManager::new(security_config)),
            rate_limiter: Arc::new(RateLimiter::new(rate_limit_config)),
            provenance_tracker: ProvenanceTracker::default(),
            audit_trail: AuditTrail::default(),
            retention_manager: RetentionManager::new(),
        }
    }

    /// Maximum number of completed jobs to retain in memory
    const MAX_JOBS: usize = 1000;
    /// Maximum number of train engines to retain
    const MAX_TRAIN_ENGINES: usize = 50;
    /// Maximum number of completed studies to retain
    const MAX_COMPLETED_STUDIES: usize = 100;
    /// Maximum number of full datasets (metadata + in-memory DataFrame) to retain at
    /// once. Storing every uploaded dataset's actual data (rather than just the
    /// single shared `current_data` buffer the old design used) trades correctness
    /// for memory, so this caps unbounded growth; the oldest datasets (by
    /// `created_at`, excluding whatever is currently active) are evicted first.
    const MAX_STORED_DATASETS: usize = 50;

    pub fn generate_id() -> String {
        // Use full UUID to avoid collisions (previously only 8 hex chars = 32 bits)
        Uuid::new_v4().to_string()
    }

    /// Store a dataset and return its ID.
    ///
    /// Inserts metadata (`datasets`) and the actual DataFrame (`dataset_frames`)
    /// before flipping `current_dataset_id` to point at the new id. Doing the writes
    /// in that order — and letting each write-guard's `Drop` run (releasing the lock)
    /// strictly before the next lock is acquired — establishes a happens-before chain
    /// through the locks' own acquire/release semantics: any reader that observes the
    /// new `current_dataset_id` value is guaranteed to also observe the corresponding
    /// `datasets`/`dataset_frames` entries, because the reader must acquire (and thus
    /// synchronize with the release of) the `current_dataset_id` lock before it can
    /// acquire the `dataset_frames`/`datasets` locks to look the id up. This avoids
    /// needing a single mega-mutex across three independent maps while still ruling
    /// out the "current_dataset_id points at an id resolve_dataset can't find yet"
    /// race.
    pub async fn store_dataset(&self, name: String, df: DataFrame, path: PathBuf) -> String {
        let id = Self::generate_id();

        let column_names: Vec<String> = df.get_column_names().iter().map(|s| s.to_string()).collect();
        let dtypes: Vec<String> = df.dtypes().iter().map(|d| format!("{:?}", d)).collect();

        let info = DatasetInfo {
            id: id.clone(),
            name,
            path,
            rows: df.height(),
            columns: df.width(),
            column_names,
            dtypes,
            created_at: chrono::Utc::now(),
        };

        // Register with the retention manager so `/api/compliance/retention-status`
        // (`retention_manager.get_expired()`) has something to actually enforce.
        // Previously nothing in the crate ever called `register`, so retention was
        // permanently a no-op regardless of configuration. Uploaded user data is
        // treated as Internal by default — it isn't public, but we don't have enough
        // signal at upload time to justify the stricter Confidential/Restricted
        // 90/30-day windows; PII scanning (see `src/privacy/mod.rs::PiiScanner`) is a
        // separate, explicit opt-in step that can reclassify a dataset later if needed.
        self.retention_manager.register(&id, DataClassification::Internal);

        // Store metadata and data together, then flip the "current" pointer last (see
        // doc comment above for why this ordering is sufficient for atomicity from a
        // reader's perspective).
        self.datasets.write().await.insert(id.clone(), info);
        self.dataset_frames.write().await.insert(id.clone(), df);
        *self.current_dataset_id.write().await = Some(id.clone());

        id
    }

    /// Resolve a DataFrame by an optional client-supplied `dataset_id`.
    ///
    /// - `Some(id)`: look up that exact dataset; 404 if it isn't found (unknown or
    ///   evicted id).
    /// - `None`: fall back to whatever `store_dataset` most recently marked as
    ///   "current". This is the default path used by every existing caller that
    ///   doesn't yet pass its own `dataset_id`, so its error (a "No data loaded" 404)
    ///   is preserved byte-for-byte from the old `current_data.read().await.as_ref()
    ///   == None` behavior to avoid changing existing tests/clients.
    pub async fn resolve_dataset(&self, dataset_id: Option<&str>) -> super::error::Result<DataFrame> {
        self.resolve_dataset_with_id(dataset_id).await.map(|(_, df)| df)
    }

    /// Like `resolve_dataset`, but also returns the id that was actually resolved.
    /// Needed by callers that mutate the DataFrame in place (preprocessing, auto-clean)
    /// and must write the result back to the *same* dataset id they read from — which
    /// may have been implicitly resolved from `current_dataset_id` rather than passed
    /// explicitly.
    pub async fn resolve_dataset_with_id(&self, dataset_id: Option<&str>) -> super::error::Result<(String, DataFrame)> {
        use super::error::ServerError;
        let id = match dataset_id {
            Some(id) => id.to_string(),
            None => self.current_dataset_id.read().await.clone()
                .ok_or_else(|| ServerError::NotFound("No data loaded".to_string()))?,
        };
        let df = self.dataset_frames.read().await.get(&id).cloned()
            .ok_or_else(|| ServerError::NotFound(format!("Dataset not found: {}", id)))?;
        Ok((id, df))
    }

    /// Replace the stored DataFrame for `dataset_id` (used after an in-place
    /// transform like preprocessing/auto-clean) and refresh its cached metadata
    /// (row/column counts, column names, dtypes) to match.
    pub async fn update_dataset_frame(&self, dataset_id: &str, df: DataFrame) {
        let rows = df.height();
        let columns = df.width();
        let column_names: Vec<String> = df.get_column_names().iter().map(|s| s.to_string()).collect();
        let dtypes: Vec<String> = df.dtypes().iter().map(|d| format!("{:?}", d)).collect();

        self.dataset_frames.write().await.insert(dataset_id.to_string(), df);

        if let Some(info) = self.datasets.write().await.get_mut(dataset_id) {
            info.rows = rows;
            info.columns = columns;
            info.column_names = column_names;
            info.dtypes = dtypes;
        }
    }

    /// Evict completed/failed jobs and old train engines to prevent unbounded memory growth.
    /// Called automatically after adding new entries.
    pub async fn evict_stale_entries(&self) {
        // Evict jobs: only acquire write lock if over limit
        {
            let jobs_len = self.jobs.read().await.len();
            if jobs_len > Self::MAX_JOBS {
                let mut jobs = self.jobs.write().await;
                if jobs.len() > Self::MAX_JOBS {
                    let to_remove = jobs.len() - Self::MAX_JOBS;
                    // Single pass to find oldest completed/failed jobs (O(n) not O(n log n))
                    let mut removable: Vec<(String, chrono::DateTime<chrono::Utc>)> = jobs.iter()
                        .filter(|(_, j)| matches!(j.status, JobStatus::Completed { .. } | JobStatus::Failed { .. }))
                        .map(|(id, j)| (id.clone(), j.created_at))
                        .collect();
                    if !removable.is_empty() {
                        // Partial sort: only bring the `to_remove` oldest to front
                        let pivot = to_remove.min(removable.len()) - 1;
                        removable.select_nth_unstable_by_key(pivot, |(_, t)| *t);
                        for (id, _) in removable.into_iter().take(to_remove) {
                            jobs.remove(&id);
                            // Job handles for completed/failed jobs are already
                            // finished (or about to be dropped); drop the JoinHandle
                            // too so it doesn't linger in the map forever.
                            self.job_handles.remove(&id);
                        }
                    }
                }
            }
        }

        // Evict train_engines and insights_cache in lockstep: same keys removed from both
        if self.train_engines.len() > Self::MAX_TRAIN_ENGINES {
            let to_remove = self.train_engines.len() - Self::MAX_TRAIN_ENGINES;
            let keys: Vec<String> = self.train_engines.iter()
                .take(to_remove)
                .map(|entry| entry.key().clone())
                .collect();
            for key in &keys {
                self.train_engines.remove(key);
                self.insights_cache.remove(key);
            }
        }

        // Truncate completed studies
        {
            let mut studies = self.completed_studies.write().await;
            if studies.len() > Self::MAX_COMPLETED_STUDIES {
                let drain_count = studies.len() - Self::MAX_COMPLETED_STUDIES;
                studies.drain(..drain_count);
            }
        }

        // Evict datasets: (a) oldest-first once over MAX_STORED_DATASETS, and (b) any
        // whose retention policy has expired (`retention_manager.get_expired()`).
        // The currently-active dataset is never evicted out from under a caller.
        {
            let current = self.current_dataset_id.read().await.clone();

            let over_cap: Vec<String> = {
                let datasets = self.datasets.read().await;
                if datasets.len() > Self::MAX_STORED_DATASETS {
                    let to_remove = datasets.len() - Self::MAX_STORED_DATASETS;
                    let mut candidates: Vec<(String, chrono::DateTime<chrono::Utc>)> = datasets.iter()
                        .filter(|(id, _)| Some(id.as_str()) != current.as_deref())
                        .map(|(id, info)| (id.clone(), info.created_at))
                        .collect();
                    candidates.sort_by_key(|(_, t)| *t);
                    candidates.into_iter().take(to_remove).map(|(id, _)| id).collect()
                } else {
                    Vec::new()
                }
            };
            for id in &over_cap {
                self.datasets.write().await.remove(id);
                self.dataset_frames.write().await.remove(id);
                self.retention_manager.mark_deleted(id, "evicted: over MAX_STORED_DATASETS cap");
            }

            let expired = self.retention_manager.get_expired();
            for record in expired {
                if Some(record.dataset_id.as_str()) == current.as_deref() {
                    continue;
                }
                let removed_info = self.datasets.write().await.remove(&record.dataset_id);
                self.dataset_frames.write().await.remove(&record.dataset_id);
                self.retention_manager.mark_deleted(&record.dataset_id, "retention period expired");
                if let Some(info) = removed_info {
                    if let Err(e) = std::fs::remove_file(&info.path) {
                        tracing::warn!(
                            dataset_id = %record.dataset_id,
                            path = ?info.path,
                            error = %e,
                            "Failed to delete expired dataset file from disk (best-effort)"
                        );
                    }
                }
            }
        }
    }

    /// Get system information
    pub fn get_system_info(&self) -> serde_json::Value {
        use sysinfo::System;

        let mut sys = System::new_all();
        sys.refresh_all();

        // Calculate overall CPU usage
        let cpu_count = sys.cpus().len().max(1);
        let cpu_usage: f32 = sys.cpus().iter().map(|c| c.cpu_usage()).sum::<f32>() / cpu_count as f32;

        let total_memory = sys.total_memory().max(1); // avoid division by zero

        let (cache_hits, cache_misses, cache_hit_rate) = self.prediction_cache.stats();

        serde_json::json!({
            "cpu_count": sys.cpus().len(),
            "cpu_usage": cpu_usage,
            "total_memory_gb": total_memory as f64 / 1024.0 / 1024.0 / 1024.0,
            "used_memory_gb": sys.used_memory() as f64 / 1024.0 / 1024.0 / 1024.0,
            "memory_usage_percent": (sys.used_memory() as f64 / total_memory as f64) * 100.0,
            "prediction_cache": {
                "size": self.prediction_cache.len(),
                "hits": cache_hits,
                "misses": cache_misses,
                "hit_rate": cache_hit_rate,
            },
        })
    }

    /// Push a prediction latency (microseconds) into the ring buffer.
    /// Evicts the oldest entry once the buffer reaches 1000 entries.
    pub fn record_prediction_latency(&self, micros: u64) {
        let mut q = self.prediction_latencies.lock().unwrap();
        if q.len() >= 1000 {
            q.pop_front();
        }
        q.push_back(micros);
    }
}

/// Scan `models_dir` for `model_*.json` files and populate `models` with
/// restored `ModelInfo` entries. Called once at server startup (synchronously,
/// before axum::serve). Skips autotune_ and optimized_ files. Logs and skips
/// malformed files without panicking.
pub fn restore_models_from_disk(models_dir: &str, models: &dashmap::DashMap<String, ModelInfo>) {
    use crate::training::TrainEngine;
    use tracing::{info, warn};

    let dir = match std::fs::read_dir(models_dir) {
        Ok(d) => d,
        Err(e) => {
            warn!(models_dir = %models_dir, error = %e, "Cannot read models directory during restore");
            return;
        }
    };

    for entry in dir.flatten() {
        let path = entry.path();
        if path.extension().and_then(|e| e.to_str()) != Some("json") { continue; }

        let stem = match path.file_stem().and_then(|s| s.to_str()) {
            Some(s) => s.to_string(),
            None => continue,
        };

        if stem.starts_with("autotune_") || stem.starts_with("optimized_") { continue; }
        if !stem.starts_with("model_") { continue; }

        let path_str = match path.to_str() {
            Some(s) => s.to_string(),
            None => continue,
        };

        let engine = match TrainEngine::load(&path_str) {
            Ok(e) => e,
            Err(err) => {
                warn!(path = %path_str, error = %err, "Skipping model file — failed to load");
                continue;
            }
        };

        let display_type = {
            let parts: Vec<&str> = stem.splitn(3, '_').collect();
            if parts.len() >= 2 { parts[1].to_string() } else { stem.clone() }
        };

        let metrics = engine.metrics()
            .cloned()
            .and_then(|m| serde_json::to_value(m).ok())
            .unwrap_or(serde_json::Value::Null);

        let created_at = std::fs::metadata(&path)
            .ok()
            .and_then(|m| m.modified().ok())
            .map(|t| chrono::DateTime::<chrono::Utc>::from(t).to_rfc3339())
            .unwrap_or_else(|| chrono::Utc::now().to_rfc3339());

        let task_type = format!("{:?}", engine.config().task_type);

        let model_info = ModelInfo {
            id: stem.clone(),
            name: format!("{} (restored)", display_type),
            task_type,
            metrics,
            created_at,
            path: path.clone(),
        };

        models.insert(stem.clone(), model_info);
        info!(model_id = %stem, path = %path_str, "Restored model from disk");
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::training::{TrainEngine, TrainingConfig, TaskType, ModelType};
    use polars::prelude::*;

    fn make_test_engine() -> TrainEngine {
        let df = df! {
            "x" => [1.0f64, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
            "y" => [0.0f64, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0],
        }.unwrap();
        let config = TrainingConfig::new(TaskType::BinaryClassification, "y")
            .with_model(ModelType::LogisticRegression);
        let mut engine = TrainEngine::new(config);
        engine.fit(&df).unwrap();
        engine
    }

    #[test]
    fn test_restore_models_from_disk_loads_valid_model() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("model_LogisticRegression_abc123.json");
        let engine = make_test_engine();
        engine.save(path.to_str().unwrap()).unwrap();

        let models: DashMap<String, ModelInfo> = DashMap::new();
        restore_models_from_disk(dir.path().to_str().unwrap(), &models);

        assert_eq!(models.len(), 1);
        let entry = models.iter().next().unwrap();
        assert_eq!(entry.key(), "model_LogisticRegression_abc123");
        assert!(entry.value().name.contains("restored"));
        assert_eq!(entry.value().path, path);
    }

    #[test]
    fn test_restore_models_from_disk_skips_autotune() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("autotune_RandomForest_xyz.json");
        let engine = make_test_engine();
        engine.save(path.to_str().unwrap()).unwrap();

        let models: DashMap<String, ModelInfo> = DashMap::new();
        restore_models_from_disk(dir.path().to_str().unwrap(), &models);
        assert_eq!(models.len(), 0);
    }

    #[test]
    fn test_restore_models_from_disk_skips_malformed_json() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(
            dir.path().join("model_Bad_zzz.json"),
            b"not valid json {{{"
        ).unwrap();

        let models: DashMap<String, ModelInfo> = DashMap::new();
        // Must not panic
        restore_models_from_disk(dir.path().to_str().unwrap(), &models);
        assert_eq!(models.len(), 0);
    }
}

#[cfg(test)]
mod quality_tests {
    use super::*;
    use crate::quality::QualityReport;

    #[test]
    fn test_app_state_stores_and_retrieves_quality_report() {
        let reports: dashmap::DashMap<String, QualityReport> = dashmap::DashMap::new();
        let report = QualityReport { model_id: "m1".into(), overall_quality_score: 0.85, ..Default::default() };
        reports.insert("m1".to_string(), report);
        let retrieved = reports.get("m1").unwrap();
        assert_eq!(retrieved.model_id, "m1");
        assert!((retrieved.overall_quality_score - 0.85).abs() < 1e-10);
    }
}

#[cfg(test)]
mod tests_latency {
    use std::collections::VecDeque;

    fn percentile_us(sorted: &[u64], p: f64) -> f64 {
        if sorted.is_empty() { return 0.0; }
        let idx = ((sorted.len() as f64 - 1.0) * p / 100.0).round() as usize;
        sorted[idx] as f64 / 1000.0
    }

    #[test]
    fn test_percentile_empty() {
        assert_eq!(percentile_us(&[], 50.0), 0.0);
    }

    #[test]
    fn test_percentile_single() {
        // 5000 µs = 5.0 ms
        assert!((percentile_us(&[5000], 50.0) - 5.0).abs() < 1e-9);
    }

    #[test]
    fn test_percentile_multiple() {
        let mut vals = vec![1000u64, 5000, 3000, 2000, 4000];
        vals.sort_unstable();
        // sorted: [1000, 2000, 3000, 4000, 5000]
        // P50 → index 2 → 3000 µs = 3.0 ms
        assert!((percentile_us(&vals, 50.0) - 3.0).abs() < 1e-9);
        // P99 → index 4 → 5000 µs = 5.0 ms
        assert!((percentile_us(&vals, 99.0) - 5.0).abs() < 1e-9);
    }

    #[test]
    fn test_ring_buffer_cap() {
        let mut q: VecDeque<u64> = VecDeque::new();
        for i in 0..1100u64 {
            if q.len() >= 1000 { q.pop_front(); }
            q.push_back(i);
        }
        assert_eq!(q.len(), 1000);
        // First 100 were evicted; front should be 100
        assert_eq!(*q.front().unwrap(), 100u64);
    }
}
