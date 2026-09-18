//! Inference engine implementation
//!
//! High-performance inference with:
//! - Parallel batch processing via rayon
//! - In-memory model caching (Arc-shared)
//! - Latency tracking via PerformanceMetrics
//! - Memory-aware adaptive batch sizing
//! - Model warmup for consistent latency
//! - Classification probability support

use crate::cache::LruTtlCache;
use crate::error::{AutoMLError, Result};
use crate::monitoring::PerformanceMetrics;
use crate::preprocessing::DataPreprocessor;
use crate::training::TrainEngine;
use super::InferenceConfig;
use ndarray::{Array1, Array2};
use polars::prelude::*;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Instant;

/// Inference statistics snapshot
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InferenceStats {
    pub total_predictions: u64,
    pub avg_latency_ms: f64,
    pub p50_latency_ms: f64,
    pub p95_latency_ms: f64,
    pub p99_latency_ms: f64,
    pub throughput_per_sec: f64,
    pub error_count: u64,
}

/// Maximum number of rows to cache (larger batches bypass cache)
const MAX_CACHEABLE_ROWS: usize = 100;

/// High-performance inference engine
pub struct InferenceEngine {
    config: InferenceConfig,
    preprocessor: Option<Arc<DataPreprocessor>>,
    model: Option<Arc<TrainEngine>>,
    is_loaded: bool,
    metrics: Arc<PerformanceMetrics>,
    is_warmed_up: bool,
    /// Cache for prediction results, keyed by hash of input features
    prediction_cache: Option<Arc<LruTtlCache<u64, Vec<f64>>>>,
    /// Cache for preprocessed DataFrames, keyed by hash of raw input
    preprocessing_cache: Option<Arc<LruTtlCache<u64, DataFrame>>>,
    /// Tracks whether the last prediction was a cache hit
    last_cache_hit: AtomicBool,
    /// Dedicated rayon thread pool sized per `config.n_workers`, built once
    /// (lazily, on first use — see `get_or_build_worker_pool`) and reused by
    /// every subsequent `predict_parallel` call. This used to be rebuilt
    /// from scratch (spawning a brand new OS thread pool) on every single
    /// large-batch prediction; `config.n_workers` never changes after
    /// construction (there's no setter on `InferenceEngine` for it), so
    /// there was nothing to gain from rebuilding it per call.
    worker_pool: std::sync::OnceLock<Option<Arc<rayon::ThreadPool>>>,
}

impl std::fmt::Debug for InferenceEngine {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("InferenceEngine")
            .field("config", &self.config)
            .field("is_loaded", &self.is_loaded)
            .field("is_warmed_up", &self.is_warmed_up)
            .field("prediction_cache_enabled", &self.prediction_cache.is_some())
            .finish()
    }
}

/// Compute a fast hash of a float slice using XxHash64
fn hash_f64_slice(data: &[f64]) -> u64 {
    use xxhash_rust::xxh3::xxh3_64;
    // Hash the raw bytes of the f64 slice directly (safe: deterministic bit representation)
    let byte_slice = unsafe {
        std::slice::from_raw_parts(data.as_ptr() as *const u8, data.len() * std::mem::size_of::<f64>())
    };
    xxh3_64(byte_slice)
}

/// Compute hash of a DataFrame's numeric content
fn hash_dataframe(df: &DataFrame) -> u64 {
    use xxhash_rust::xxh3::xxh3_64;
    let mut bytes = Vec::with_capacity(df.height() * df.width() * 8);
    for col in df.get_columns() {
        if let Ok(ca) = col.f64() {
            for val in ca.into_no_null_iter() {
                bytes.extend_from_slice(&val.to_bits().to_le_bytes());
            }
        } else {
            // For non-f64 columns, hash the string representation
            for i in 0..col.len() {
                let s = format!("{:?}", col.get(i));
                bytes.extend_from_slice(s.as_bytes());
            }
        }
    }
    xxh3_64(&bytes)
}

impl InferenceEngine {
    /// Create a new inference engine
    pub fn new(config: InferenceConfig) -> Self {
        let prediction_cache = if config.cache_preprocessing {
            Some(Arc::new(LruTtlCache::new(
                config.prediction_cache_size,
                config.prediction_cache_ttl_secs,
            )))
        } else {
            None
        };

        Self {
            prediction_cache,
            preprocessing_cache: None, // initialized lazily when preprocessor is set
            config,
            preprocessor: None,
            model: None,
            is_loaded: false,
            metrics: Arc::new(PerformanceMetrics::new(10000)),
            is_warmed_up: false,
            last_cache_hit: AtomicBool::new(false),
            worker_pool: std::sync::OnceLock::new(),
        }
    }

    /// Get the cached dedicated thread pool (building it on first call), or
    /// `None` if `config.n_workers` isn't set (meaning "use rayon's global
    /// pool"). A build failure is surfaced as an error on whichever call
    /// triggers it, same as when this was rebuilt on every call.
    fn get_or_build_worker_pool(&self) -> Result<Option<Arc<rayon::ThreadPool>>> {
        if let Some(pool) = self.worker_pool.get() {
            return Ok(pool.clone());
        }

        let pool = match self.config.n_workers {
            Some(n_workers) => Some(Arc::new(
                rayon::ThreadPoolBuilder::new()
                    .num_threads(n_workers)
                    .build()
                    .map_err(|e| AutoMLError::InferenceError(format!("Thread pool error: {}", e)))?,
            )),
            None => None,
        };

        // Benign race: if another thread already initialized this concurrently,
        // both pools are equally valid (built from the same fixed config) —
        // just use whichever one we personally have rather than erroring.
        let _ = self.worker_pool.set(pool.clone());
        Ok(pool)
    }

    /// Load a preprocessor
    pub fn with_preprocessor(mut self, preprocessor: DataPreprocessor) -> Self {
        self.preprocessor = Some(Arc::new(preprocessor));
        // Initialize preprocessing cache when we have a preprocessor and caching is enabled
        if self.config.cache_preprocessing {
            self.preprocessing_cache = Some(Arc::new(LruTtlCache::new(
                self.config.prediction_cache_size,
                self.config.prediction_cache_ttl_secs,
            )));
        }
        self
    }

    /// Load a trained model
    pub fn with_model(mut self, model: TrainEngine) -> Self {
        self.model = Some(Arc::new(model));
        self.is_loaded = true;
        self
    }

    /// Load from files
    pub fn load(
        config: InferenceConfig,
        preprocessor_path: Option<&str>,
        model_path: &str,
    ) -> Result<Self> {
        config.validate()?;
        let mut engine = Self::new(config);

        if let Some(path) = preprocessor_path {
            let preprocessor = DataPreprocessor::load(path)?;
            engine.preprocessor = Some(Arc::new(preprocessor));
            // Initialize preprocessing cache when preprocessor is loaded
            if engine.config.cache_preprocessing {
                engine.preprocessing_cache = Some(Arc::new(LruTtlCache::new(
                    engine.config.prediction_cache_size,
                    engine.config.prediction_cache_ttl_secs,
                )));
            }
        }

        let model = TrainEngine::load(model_path)?;
        engine.model = Some(Arc::new(model));
        engine.is_loaded = true;

        Ok(engine)
    }

    /// Warm up the engine by running a dummy prediction to prime caches and JIT paths
    pub fn warmup(&mut self) -> Result<()> {
        if !self.is_loaded {
            return Err(AutoMLError::ModelNotFitted);
        }
        let model = self.model.as_ref().ok_or(AutoMLError::ModelNotFitted)?;
        let n_features = model.feature_names().len();
        if n_features == 0 {
            return Ok(());
        }

        // Create a small dummy DataFrame with 1 row
        let columns: Vec<Column> = model.feature_names().iter().map(|name| {
            let series = Series::new(name.as_str().into(), &[0.0_f64]);
            series.into()
        }).collect();

        let df = DataFrame::new(columns)
            .map_err(|e| AutoMLError::DataError(e.to_string()))?;

        let processed = if let Some(ref preprocessor) = self.preprocessor {
            preprocessor.transform(&df)?
        } else {
            df
        };

        // Run a prediction to warm up internal paths
        let _ = model.predict(&processed);
        self.is_warmed_up = true;
        Ok(())
    }

    /// Check if the engine has been warmed up
    pub fn is_warmed_up(&self) -> bool {
        self.is_warmed_up
    }

    /// Get a reference to the underlying performance metrics
    pub fn performance_metrics(&self) -> &PerformanceMetrics {
        &self.metrics
    }

    /// Get inference statistics snapshot
    pub fn stats(&self) -> InferenceStats {
        InferenceStats {
            total_predictions: self.metrics.total_items(),
            avg_latency_ms: self.metrics.avg_latency(),
            p50_latency_ms: self.metrics.p50_latency(),
            p95_latency_ms: self.metrics.p95_latency(),
            p99_latency_ms: self.metrics.p99_latency(),
            throughput_per_sec: self.metrics.throughput(),
            error_count: self.metrics.total_errors(),
        }
    }

    /// Check if the engine is ready for predictions
    pub fn is_loaded(&self) -> bool {
        self.is_loaded
    }

    /// Get the current configuration
    pub fn config(&self) -> &InferenceConfig {
        &self.config
    }

    /// Get the model reference (if loaded)
    pub fn model(&self) -> Option<&TrainEngine> {
        self.model.as_deref()
    }

    /// Get prediction cache statistics: (hits, misses, hit_rate)
    pub fn cache_stats(&self) -> (u64, u64, f64) {
        match &self.prediction_cache {
            Some(cache) => cache.stats(),
            None => (0, 0, 0.0),
        }
    }

    /// Check if the last prediction was served from cache
    pub fn last_prediction_was_cached(&self) -> bool {
        self.last_cache_hit.load(Ordering::Relaxed)
    }

    /// Set an external shared prediction cache
    pub fn set_prediction_cache(&mut self, cache: Arc<LruTtlCache<u64, Vec<f64>>>) {
        self.prediction_cache = Some(cache);
    }

    /// Get the prediction cache (if enabled)
    pub fn prediction_cache(&self) -> Option<&Arc<LruTtlCache<u64, Vec<f64>>>> {
        self.prediction_cache.as_ref()
    }

    /// Make predictions on a DataFrame
    pub fn predict(&self, df: &DataFrame) -> Result<Array1<f64>> {
        let start = Instant::now();
        self.last_cache_hit.store(false, Ordering::Relaxed);

        if !self.is_loaded {
            self.metrics.record_error();
            return Err(AutoMLError::ModelNotFitted);
        }

        let model = self.model.as_ref().ok_or(AutoMLError::ModelNotFitted)?;
        let n_rows = df.height();

        // Check prediction cache for small inputs
        let input_hash = if n_rows <= MAX_CACHEABLE_ROWS {
            let h = hash_dataframe(df);
            if let Some(ref cache) = self.prediction_cache {
                if let Some(cached) = cache.get(&h) {
                    self.last_cache_hit.store(true, Ordering::Relaxed);
                    let latency_ms = start.elapsed().as_secs_f64() * 1000.0;
                    self.metrics.record_latency(latency_ms);
                    self.metrics.record_items(cached.len() as u64);
                    return Ok(Array1::from_vec(cached));
                }
            }
            Some(h)
        } else {
            None
        };

        // Check preprocessing cache or compute
        let processed = if let Some(ref preprocessor) = self.preprocessor {
            if let (Some(hash), Some(ref pp_cache)) = (input_hash, &self.preprocessing_cache) {
                if let Some(cached_df) = pp_cache.get(&hash) {
                    cached_df
                } else {
                    let transformed = preprocessor.transform(df)?;
                    pp_cache.set(hash, transformed.clone());
                    transformed
                }
            } else {
                preprocessor.transform(df)?
            }
        } else {
            df.clone()
        };

        let result = if processed.height() <= self.config.batch_size {
            model.predict(&processed)
        } else if self.should_use_parallel(processed.height()) {
            self.predict_parallel(&processed, model)
        } else {
            self.predict_batched(&processed, model)
        };

        let latency_ms = start.elapsed().as_secs_f64() * 1000.0;
        match &result {
            Ok(preds) => {
                self.metrics.record_latency(latency_ms);
                self.metrics.record_items(preds.len() as u64);
                // Store in prediction cache
                if let (Some(hash), Some(ref cache)) = (input_hash, &self.prediction_cache) {
                    cache.set(hash, preds.to_vec());
                }
            }
            Err(_) => self.metrics.record_error(),
        }

        result
    }

    /// Make predictions on raw feature array
    pub fn predict_array(&self, x: &Array2<f64>) -> Result<Array1<f64>> {
        let start = Instant::now();
        self.last_cache_hit.store(false, Ordering::Relaxed);

        if !self.is_loaded {
            self.metrics.record_error();
            return Err(AutoMLError::ModelNotFitted);
        }

        let model = self.model.as_ref().ok_or(AutoMLError::ModelNotFitted)?;

        // Get feature names from model
        let feature_names = model.feature_names();

        if x.ncols() != feature_names.len() {
            self.metrics.record_error();
            return Err(AutoMLError::ShapeError {
                expected: format!("{} features", feature_names.len()),
                actual: format!("{} features", x.ncols()),
            });
        }

        let n_rows = x.nrows();

        // Check prediction cache for small inputs
        let input_hash = if n_rows <= MAX_CACHEABLE_ROWS {
            // Hash directly from contiguous slice if possible, avoiding allocation
            let h = if let Some(slice) = x.as_slice() {
                hash_f64_slice(slice)
            } else {
                let flat: Vec<f64> = x.iter().copied().collect();
                hash_f64_slice(&flat)
            };
            if let Some(ref cache) = self.prediction_cache {
                if let Some(cached) = cache.get(&h) {
                    self.last_cache_hit.store(true, Ordering::Relaxed);
                    let latency_ms = start.elapsed().as_secs_f64() * 1000.0;
                    self.metrics.record_latency(latency_ms);
                    self.metrics.record_items(cached.len() as u64);
                    return Ok(Array1::from_vec(cached));
                }
            }
            Some(h)
        } else {
            None
        };

        // Convert Array2 to DataFrame
        let df = self.array_to_dataframe(x, feature_names)?;

        // Apply preprocessing with cache (mirrors predict() logic)
        let processed = if let Some(ref preprocessor) = self.preprocessor {
            if let (Some(hash), Some(ref pp_cache)) = (input_hash, &self.preprocessing_cache) {
                if let Some(cached_df) = pp_cache.get(&hash) {
                    cached_df
                } else {
                    let transformed = preprocessor.transform(&df)?;
                    pp_cache.set(hash, transformed.clone());
                    transformed
                }
            } else {
                preprocessor.transform(&df)?
            }
        } else {
            df
        };

        let result = if processed.height() <= self.config.batch_size {
            model.predict(&processed)
        } else if self.should_use_parallel(processed.height()) {
            self.predict_parallel(&processed, model)
        } else {
            self.predict_batched(&processed, model)
        };

        let latency_ms = start.elapsed().as_secs_f64() * 1000.0;
        match &result {
            Ok(preds) => {
                self.metrics.record_latency(latency_ms);
                self.metrics.record_items(preds.len() as u64);
                // Store in prediction cache
                if let (Some(hash), Some(ref cache)) = (input_hash, &self.prediction_cache) {
                    cache.set(hash, preds.to_vec());
                }
            }
            Err(_) => self.metrics.record_error(),
        }

        result
    }

    /// Get probability predictions (for classification) using the model's native predict_proba
    pub fn predict_proba(&self, df: &DataFrame) -> Result<Array2<f64>> {
        let start = Instant::now();

        if !self.is_loaded {
            self.metrics.record_error();
            return Err(AutoMLError::ModelNotFitted);
        }

        let model = self.model.as_ref().ok_or(AutoMLError::ModelNotFitted)?;

        // Apply preprocessing if available
        let processed = if let Some(ref preprocessor) = self.preprocessor {
            preprocessor.transform(df)?
        } else {
            df.clone()
        };

        let result = model.predict_proba(&processed);

        let latency_ms = start.elapsed().as_secs_f64() * 1000.0;
        match &result {
            Ok(proba) => {
                self.metrics.record_latency(latency_ms);
                self.metrics.record_items(proba.nrows() as u64);
            }
            Err(_) => self.metrics.record_error(),
        }

        result
    }

    /// Get probability predictions on raw feature array
    pub fn predict_proba_array(&self, x: &Array2<f64>) -> Result<Array2<f64>> {
        let start = Instant::now();

        if !self.is_loaded {
            self.metrics.record_error();
            return Err(AutoMLError::ModelNotFitted);
        }

        let model = self.model.as_ref().ok_or(AutoMLError::ModelNotFitted)?;
        let feature_names = model.feature_names();

        if x.ncols() != feature_names.len() {
            self.metrics.record_error();
            return Err(AutoMLError::ShapeError {
                expected: format!("{} features", feature_names.len()),
                actual: format!("{} features", x.ncols()),
            });
        }

        let result = model.predict_proba_array(x);

        let latency_ms = start.elapsed().as_secs_f64() * 1000.0;
        match &result {
            Ok(proba) => {
                self.metrics.record_latency(latency_ms);
                self.metrics.record_items(proba.nrows() as u64);
            }
            Err(_) => self.metrics.record_error(),
        }

        result
    }

    /// Predict with classification threshold applied (returns 0/1 labels)
    pub fn predict_with_threshold(&self, df: &DataFrame) -> Result<Array1<f64>> {
        let proba = self.predict_proba(df)?;
        let threshold = self.config.classification_threshold;
        // Use the positive class column (last column)
        let pos_col = proba.ncols() - 1;
        Ok(Array1::from_iter(
            proba.column(pos_col).iter().map(|&p| if p >= threshold { 1.0 } else { 0.0 })
        ))
    }

    /// Stream predictions for large datasets
    pub fn predict_streaming<'a>(
        &'a self,
        df: &'a DataFrame,
    ) -> Result<impl Iterator<Item = Result<Array1<f64>>> + 'a> {
        if !self.is_loaded {
            return Err(AutoMLError::ModelNotFitted);
        }

        let model = self.model.as_ref().ok_or(AutoMLError::ModelNotFitted)?;
        // Defensive clamp: guard against a misconfigured/deserialized config with
        // stream_chunk_size == 0, which would otherwise panic on division below.
        let chunk_size = self.config.stream_chunk_size.max(1);
        let n_chunks = (df.height() + chunk_size - 1) / chunk_size;

        Ok((0..n_chunks).map(move |i| {
            let start_time = Instant::now();
            let start = i * chunk_size;
            let end = (start + chunk_size).min(df.height());
            let chunk = df.slice(start as i64, end - start);

            // Apply preprocessing
            let processed = if let Some(ref preprocessor) = self.preprocessor {
                preprocessor.transform(&chunk)?
            } else {
                chunk
            };

            let result = model.predict(&processed);
            let latency_ms = start_time.elapsed().as_secs_f64() * 1000.0;
            self.metrics.record_latency(latency_ms);
            if let Ok(ref preds) = result {
                self.metrics.record_items(preds.len() as u64);
            }
            result
        }))
    }

    // =========================================================================
    // Private helpers
    // =========================================================================

    /// Determine if parallel batch processing should be used
    fn should_use_parallel(&self, n_rows: usize) -> bool {
        let n_workers = self.config.n_workers.unwrap_or_else(|| {
            rayon::current_num_threads()
        });
        // Use parallel when we have enough rows to fill multiple workers
        n_workers > 1 && n_rows > self.config.batch_size * 2
    }

    /// Compute adaptive batch size based on available memory
    pub(crate) fn adaptive_batch_size(&self, n_rows: usize, n_cols: usize) -> usize {
        if let Some(max_bytes) = self.config.max_memory_bytes {
            let bytes_per_row = n_cols * std::mem::size_of::<f64>();
            // Use at most 50% of memory budget for a single batch
            let max_rows = (max_bytes / 2) / bytes_per_row.max(1);
            max_rows.max(1).min(self.config.batch_size).min(n_rows)
        } else {
            self.config.batch_size.min(n_rows)
        }
    }

    /// Convert Array2 to DataFrame using model feature names
    fn array_to_dataframe(&self, x: &Array2<f64>, feature_names: &[String]) -> Result<DataFrame> {
        // For single-row predictions, use a fast path that avoids per-column allocation
        let n_cols = x.ncols();
        let columns: Vec<Column> = if x.nrows() == 1 {
            // Single row: build each column from a one-element slice
            (0..n_cols)
                .map(|j| {
                    let val = x[[0, j]];
                    Column::new(feature_names[j].as_str().into(), &[val])
                })
                .collect()
        } else {
            // Multi-row: use standard column-major extraction
            (0..n_cols)
                .map(|j| {
                    let col_data: Vec<f64> = x.column(j).iter().copied().collect();
                    Column::new(feature_names[j].as_str().into(), col_data)
                })
                .collect()
        };
        DataFrame::new(columns).map_err(|e| AutoMLError::DataError(e.to_string()))
    }

    /// Sequential batch prediction
    fn predict_batched(&self, df: &DataFrame, model: &TrainEngine) -> Result<Array1<f64>> {
        let n_rows = df.height();
        let n_cols = df.width();
        // Defensive clamp: adaptive_batch_size can return 0 when config.batch_size == 0
        // (e.g. a deserialized config that skipped validation); avoid division by zero.
        let batch_size = self.adaptive_batch_size(n_rows, n_cols).max(1);
        let n_batches = (n_rows + batch_size - 1) / batch_size;

        let mut all_predictions = Vec::with_capacity(n_rows);

        for i in 0..n_batches {
            let start = i * batch_size;
            let end = (start + batch_size).min(n_rows);
            let batch = df.slice(start as i64, end - start);

            let predictions = model.predict(&batch)?;
            all_predictions.extend(predictions.iter().cloned());
        }

        Ok(Array1::from_vec(all_predictions))
    }

    /// Parallel batch prediction using rayon
    fn predict_parallel(&self, df: &DataFrame, model: &TrainEngine) -> Result<Array1<f64>> {
        let n_rows = df.height();
        let n_cols = df.width();
        // Defensive clamp: see predict_batched for rationale.
        let batch_size = self.adaptive_batch_size(n_rows, n_cols).max(1);
        let n_batches = (n_rows + batch_size - 1) / batch_size;

        // Reuse the cached thread pool sized per `config.n_workers`.
        let pool = self.get_or_build_worker_pool()?;

        // Prepare batch slices
        let batches: Vec<DataFrame> = (0..n_batches)
            .map(|i| {
                let start = i * batch_size;
                let end = (start + batch_size).min(n_rows);
                df.slice(start as i64, end - start)
            })
            .collect();

        // Run predictions in parallel
        let predict_fn = |batches: &[DataFrame]| -> Result<Vec<Array1<f64>>> {
            batches
                .par_iter()
                .map(|batch| model.predict(batch))
                .collect::<Result<Vec<_>>>()
        };

        let batch_results = if let Some(ref pool) = pool {
            pool.install(|| predict_fn(&batches))?
        } else {
            predict_fn(&batches)?
        };

        // Concatenate results in order
        let mut all_predictions = Vec::with_capacity(n_rows);
        for preds in batch_results {
            all_predictions.extend(preds.iter().cloned());
        }

        Ok(Array1::from_vec(all_predictions))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::training::{TrainingConfig, TaskType};

    fn create_regression_data() -> DataFrame {
        DataFrame::new(vec![
            Series::new("feature1".into(), &[1.0, 2.0, 3.0, 4.0, 5.0]).into(),
            Series::new("feature2".into(), &[1.5, 3.1, 2.8, 7.2, 5.5]).into(),
            Series::new("target".into(), &[3.0, 6.0, 9.0, 12.0, 15.0]).into(),
        ])
        .unwrap()
    }

    fn create_classification_data() -> DataFrame {
        DataFrame::new(vec![
            Series::new("feature1".into(), &[1.0, 2.0, 3.0, 8.0, 9.0, 10.0]).into(),
            Series::new("feature2".into(), &[1.5, 2.5, 3.5, 8.5, 9.5, 10.5]).into(),
            Series::new("target".into(), &[0.0, 0.0, 0.0, 1.0, 1.0, 1.0]).into(),
        ])
        .unwrap()
    }

    fn train_regression_model() -> TrainEngine {
        let df = create_regression_data();
        let config = TrainingConfig::new(TaskType::Regression, "target");
        let mut model = TrainEngine::new(config);
        model.fit(&df).unwrap();
        model
    }

    fn train_classification_model() -> TrainEngine {
        let df = create_classification_data();
        let config = TrainingConfig::new(TaskType::BinaryClassification, "target");
        let mut model = TrainEngine::new(config);
        model.fit(&df).unwrap();
        model
    }

    #[test]
    fn test_engine_creation() {
        let config = InferenceConfig::new();
        let engine = InferenceEngine::new(config);
        assert!(!engine.is_loaded());
    }

    #[test]
    fn test_inference_pipeline() {
        let df = create_regression_data();
        let model = train_regression_model();

        let config = InferenceConfig::new().with_batch_size(2);
        let engine = InferenceEngine::new(config).with_model(model);

        let predictions = engine.predict(&df).unwrap();
        assert_eq!(predictions.len(), 5);
    }

    #[test]
    fn test_predict_array() {
        let df = create_regression_data();
        let model = train_regression_model();

        let config = InferenceConfig::new();
        let engine = InferenceEngine::new(config).with_model(model);

        let x = ndarray::array![[1.0, 1.5], [2.0, 3.1]];
        let predictions = engine.predict_array(&x).unwrap();
        assert_eq!(predictions.len(), 2);
    }

    #[test]
    fn test_warmup() {
        let model = train_regression_model();
        let config = InferenceConfig::new();
        let mut engine = InferenceEngine::new(config).with_model(model);

        assert!(!engine.is_warmed_up());
        engine.warmup().unwrap();
        assert!(engine.is_warmed_up());
    }

    #[test]
    fn test_latency_tracking() {
        let df = create_regression_data();
        let model = train_regression_model();

        let config = InferenceConfig::new();
        let engine = InferenceEngine::new(config).with_model(model);

        assert_eq!(engine.stats().total_predictions, 0);

        engine.predict(&df).unwrap();
        let stats = engine.stats();
        assert!(stats.total_predictions > 0);
        assert!(stats.avg_latency_ms >= 0.0);
    }

    #[test]
    fn test_parallel_batch_prediction() {
        let model = train_regression_model();

        // Create larger dataset to trigger parallel processing
        let n = 100;
        let f1: Vec<f64> = (0..n).map(|i| i as f64).collect();
        let f2: Vec<f64> = (0..n).map(|i| i as f64 * 1.5).collect();
        let target: Vec<f64> = (0..n).map(|i| i as f64 * 3.0).collect();
        let df = DataFrame::new(vec![
            Series::new("feature1".into(), &f1).into(),
            Series::new("feature2".into(), &f2).into(),
            Series::new("target".into(), &target).into(),
        ]).unwrap();

        let config = InferenceConfig::new()
            .with_batch_size(10)
            .with_n_workers(2);
        let engine = InferenceEngine::new(config).with_model(model);

        let predictions = engine.predict(&df).unwrap();
        assert_eq!(predictions.len(), n);
    }

    #[test]
    fn test_memory_aware_batch_sizing() {
        let model = train_regression_model();
        let config = InferenceConfig::new()
            .with_batch_size(1000)
            .with_max_memory(1024); // Very small memory budget
        let engine = InferenceEngine::new(config).with_model(model);

        // adaptive_batch_size should limit batch size due to memory constraint
        let batch_size = engine.adaptive_batch_size(1000, 10);
        assert!(batch_size < 1000);
    }

    #[test]
    fn test_predict_not_loaded() {
        let config = InferenceConfig::new();
        let engine = InferenceEngine::new(config);

        let df = create_regression_data();
        assert!(engine.predict(&df).is_err());
    }

    #[test]
    fn test_predict_proba_classification() {
        let df = create_classification_data();
        let model = train_classification_model();

        let config = InferenceConfig::new();
        let engine = InferenceEngine::new(config).with_model(model);

        let proba = engine.predict_proba(&df);
        // predict_proba may fail for some model types that don't support it
        // but should not panic
        if let Ok(proba) = proba {
            assert_eq!(proba.nrows(), 6);
            assert_eq!(proba.ncols(), 2);
            // Probabilities should sum to ~1
            for i in 0..proba.nrows() {
                let sum = proba[[i, 0]] + proba[[i, 1]];
                assert!((sum - 1.0).abs() < 1e-6);
            }
        }
    }

    #[test]
    fn test_predict_with_threshold() {
        let df = create_classification_data();
        let model = train_classification_model();

        let config = InferenceConfig::new();
        let engine = InferenceEngine::new(config).with_model(model);

        let result = engine.predict_with_threshold(&df);
        if let Ok(labels) = result {
            assert_eq!(labels.len(), 6);
            for &v in labels.iter() {
                assert!(v == 0.0 || v == 1.0);
            }
        }
    }

    #[test]
    fn test_streaming_predictions() {
        let df = create_regression_data();
        let model = train_regression_model();

        let config = InferenceConfig::new().with_streaming(2);
        let engine = InferenceEngine::new(config).with_model(model);

        let chunks: Vec<_> = engine.predict_streaming(&df).unwrap().collect();
        let total: usize = chunks.iter().filter_map(|r| r.as_ref().ok()).map(|a| a.len()).sum();
        assert_eq!(total, 5);
    }

    #[test]
    fn test_prediction_cache_hit_miss() {
        let df = create_regression_data();
        let model = train_regression_model();

        let config = InferenceConfig::new();
        let engine = InferenceEngine::new(config).with_model(model);

        // First call should be a miss
        let preds1 = engine.predict(&df).unwrap();
        assert!(!engine.last_prediction_was_cached());

        // Second call with same data should be a hit
        let preds2 = engine.predict(&df).unwrap();
        assert!(engine.last_prediction_was_cached());

        // Results should be identical
        assert_eq!(preds1.to_vec(), preds2.to_vec());

        // Cache stats should reflect 1 hit, 1 miss
        let (hits, misses, hit_rate) = engine.cache_stats();
        assert_eq!(hits, 1);
        assert_eq!(misses, 1);
        assert!((hit_rate - 0.5).abs() < 0.01);
    }

    #[test]
    fn test_prediction_cache_array() {
        let model = train_regression_model();

        let config = InferenceConfig::new();
        let engine = InferenceEngine::new(config).with_model(model);

        let x = ndarray::array![[1.0, 1.5], [2.0, 3.1]];

        // First call: miss
        let preds1 = engine.predict_array(&x).unwrap();
        assert!(!engine.last_prediction_was_cached());

        // Second call: hit
        let preds2 = engine.predict_array(&x).unwrap();
        assert!(engine.last_prediction_was_cached());

        assert_eq!(preds1.to_vec(), preds2.to_vec());
    }

    #[test]
    fn test_cache_bypass_for_large_batches() {
        let model = train_regression_model();

        // Create dataset larger than MAX_CACHEABLE_ROWS (100)
        let n = 150;
        let f1: Vec<f64> = (0..n).map(|i| i as f64).collect();
        let f2: Vec<f64> = (0..n).map(|i| i as f64 * 1.5).collect();
        let target: Vec<f64> = (0..n).map(|i| i as f64 * 3.0).collect();
        let df = DataFrame::new(vec![
            Series::new("feature1".into(), &f1).into(),
            Series::new("feature2".into(), &f2).into(),
            Series::new("target".into(), &target).into(),
        ]).unwrap();

        let config = InferenceConfig::new();
        let engine = InferenceEngine::new(config).with_model(model);

        engine.predict(&df).unwrap();
        engine.predict(&df).unwrap();

        // Neither call should hit the cache (batch too large)
        let (hits, _misses, _hit_rate) = engine.cache_stats();
        assert_eq!(hits, 0);
    }

    #[test]
    fn test_cache_ttl_expiry() {
        let df = create_regression_data();
        let model = train_regression_model();

        // 1 second TTL
        let config = InferenceConfig::new().with_prediction_cache(100, 1);
        let engine = InferenceEngine::new(config).with_model(model);

        engine.predict(&df).unwrap();
        assert!(!engine.last_prediction_was_cached());

        // Wait for TTL to expire
        std::thread::sleep(std::time::Duration::from_secs(2));

        engine.predict(&df).unwrap();
        assert!(!engine.last_prediction_was_cached()); // expired, so miss
    }

    #[test]
    fn test_cache_disabled() {
        let df = create_regression_data();
        let model = train_regression_model();

        let mut config = InferenceConfig::new();
        config.cache_preprocessing = false;
        let engine = InferenceEngine::new(config).with_model(model);

        engine.predict(&df).unwrap();
        engine.predict(&df).unwrap();

        // No cache, so always miss
        assert!(!engine.last_prediction_was_cached());
        let (hits, misses, _) = engine.cache_stats();
        assert_eq!(hits, 0);
        assert_eq!(misses, 0); // no cache means no stats tracked
    }
}
