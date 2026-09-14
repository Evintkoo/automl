# AutoML - Rust Migration Plan

## Overview

**Project**: Evintkoo/automl  
**Target Branch**: `rust-dev` (new branch from `main`)  
**Current Stack**: Python 3.9+ with FastAPI, Gradio, scikit-learn, XGBoost, LightGBM, Optuna  
**Target Stack**: Rust with Python bindings via PyO3  
**Rust Edition**: 2021  
**MSRV**: 1.75.0+

---

## Table of Contents

1. [Project Analysis](#project-analysis)
2. [Migration Strategy](#migration-strategy)
3. [Detailed Workplan](#workplan)
4. [Rust Architecture](#rust-architecture)
5. [Module Specifications](#module-specifications)
6. [Testing Strategy](#testing-strategy)
7. [Performance Benchmarks](#performance-benchmarks)
8. [CI/CD Pipeline](#cicd-pipeline)
9. [Dependencies](#dependencies)
10. [Risk Mitigation](#risk-mitigation)

---

## Project Analysis

### Current Architecture

The codebase is a production-ready AutoML framework (~1.5M+ lines including dependencies) with:

| Module | Size | Description | Migration Priority |
|--------|------|-------------|-------------------|
| `modules/engine/` | ~600KB | Core ML engine (training, inference, preprocessing) | **P0 - Critical** |
| `modules/optimizer/` | ~156KB | Hyperparameter optimization (HyperOptX, ASHT) | **P0 - Critical** |
| `modules/api/` | ~400KB | FastAPI REST endpoints | P1 - High |
| `modules/device_optimizer.py` | ~68KB | Hardware optimization | P1 - High |
| `modules/model/` | ~9KB | Model management | P2 - Medium |
| `modules/configs.py` | ~60KB | Configuration management | P2 - Medium |
| `app.py` | ~135KB | Gradio UI | P3 - Keep Python |

### Key Components to Migrate

#### Tier 1: Performance-Critical (Must Migrate)

1. **Core Engine** (`modules/engine/`)
   - `train_engine.py` (130KB) - Main training logic
   - `inference_engine.py` (84KB) - Inference pipeline
   - `data_preprocessor.py` (79KB) - Data preprocessing
   - `quantizer.py` (47KB) - Model quantization
   - `batch_processor.py` (38KB) - Batch processing
   - `memory_aware_processor.py` (33KB) - Memory management
   - `adaptive_hyperopt.py` (31KB) - Adaptive HPO
   - `streaming_pipeline.py` (28KB) - Streaming data
   - `optimized_data_loader.py` (28KB) - Data loading

2. **Optimizer** (`modules/optimizer/`)
   - `hyperoptx.py` (124KB) - Custom hyperparameter optimizer
   - `asht.py` (32KB) - Adaptive successive halving

#### Tier 2: Performance-Beneficial (Should Migrate)

3. **Performance Utilities**
   - `jit_compiler.py` - Replace with native Rust compilation
   - `simd_optimizer.py` - Native SIMD via std::simd or packed_simd
   - `mixed_precision.py` - Native f16/bf16 support
   - Cache modules (lru_ttl_cache, multi_level_cache, memory_pool)

#### Tier 3: Optional Migration

4. **Support Modules**
   - `experiment_tracker.py` (23KB) - Experiment tracking
   - `logging_config.py` - Structured logging (use tracing)

---

## Migration Strategy

### Approach: **Incremental Hybrid Migration with Feature Flags**

```
┌─────────────────────────────────────────────────────────────────┐
│                        Python Layer                              │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────┐  │
│  │  Gradio UI  │  │  FastAPI    │  │  Python Orchestration   │  │
│  │  (app.py)   │  │  (modules/  │  │  (config, logging)      │  │
│  │             │  │   api/)     │  │                         │  │
│  └──────┬──────┘  └──────┬──────┘  └───────────┬─────────────┘  │
│         │                │                      │                │
│         └────────────────┼──────────────────────┘                │
│                          │                                       │
│                    ┌─────▼─────┐                                 │
│                    │   PyO3    │                                 │
│                    │ Bindings  │                                 │
│                    └─────┬─────┘                                 │
└──────────────────────────┼──────────────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────────┐
│                        Rust Layer                                │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │                    automl-core                              ││
│  │  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌────────────────┐  ││
│  │  │  Data    │ │ Training │ │Inference │ │   Optimizer    │  ││
│  │  │Processing│ │  Engine  │ │  Engine  │ │  (HyperOptX)   │  ││
│  │  └──────────┘ └──────────┘ └──────────┘ └────────────────┘  ││
│  │  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌────────────────┐  ││
│  │  │  Cache   │ │  Memory  │ │   SIMD   │ │   Parallel     │  ││
│  │  │  System  │ │   Pool   │ │   Ops    │ │   Executor     │  ││
│  │  └──────────┘ └──────────┘ └──────────┘ └────────────────┘  ││
│  └─────────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────────┘
```

### Feature Flags for Gradual Rollout

```python
# In Python code
import os
USE_RUST_BACKEND = os.environ.get("AUTOML_USE_RUST", "1") == "1"

if USE_RUST_BACKEND:
    from automl_rust import DataPreprocessor, TrainEngine
else:
    from modules.engine.data_preprocessor import DataPreprocessor
    from modules.engine.train_engine import TrainEngine
```

### Benefits of Rust Migration
- **Performance**: 10-100x faster numerical operations
- **Memory Safety**: Eliminate memory leaks and race conditions  
- **Concurrency**: True parallelism without GIL limitations
- **Binary Distribution**: Single binary deployment option
- **SIMD**: Native vectorized operations via portable_simd
- **Predictable Latency**: No GC pauses

---

## Workplan

### Phase 1: Project Setup & Infrastructure

**Objective**: Establish Rust development environment and CI/CD pipeline

- [ ] **1.1 Branch Setup**
  - [ ] Create `rust-dev` branch from `main`
  - [ ] Set up branch protection rules
  - [ ] Configure PR template for Rust changes

- [ ] **1.2 Rust Workspace Initialization**
  - [ ] Create root `Cargo.toml` with workspace configuration
  - [ ] Set up workspace members structure
  - [ ] Configure shared dependencies and features
  - [ ] Set up `rust-toolchain.toml` (pin to stable 1.75+)
  - [ ] Configure `.cargo/config.toml` for optimizations

- [ ] **1.3 PyO3/Maturin Setup**
  - [ ] Add maturin as build backend in `pyproject.toml`
  - [ ] Configure PyO3 features (abi3, extension-module)
  - [ ] Set up Python stub generation (.pyi files)
  - [ ] Configure wheel metadata

- [ ] **1.4 Development Environment**
  - [ ] Create `Makefile.toml` (cargo-make) for task automation
  - [ ] Set up pre-commit hooks (rustfmt, clippy)
  - [ ] Configure VS Code/rust-analyzer settings
  - [ ] Create Docker dev container with Rust toolchain

- [ ] **1.5 CI/CD Pipeline** (see [CI/CD Section](#cicd-pipeline))
  - [ ] GitHub Actions for Rust builds
  - [ ] Cross-compilation for Linux/macOS/Windows
  - [ ] Automated wheel building and publishing
  - [ ] Integration with existing Python CI

---

### Phase 2: Core Data Structures & Utilities

**Objective**: Build foundational Rust types that mirror Python data structures

- [ ] **2.1 Core Types**
  ```rust
  // Key types to implement
  pub struct AutoMLArray<T> { /* ndarray wrapper */ }
  pub struct AutoMLDataFrame { /* polars wrapper */ }
  pub struct FeatureColumn { /* column metadata + data */ }
  pub enum DType { Float32, Float64, Int32, Int64, String, Categorical, DateTime }
  ```
  - [ ] Generic array type with SIMD-friendly memory layout
  - [ ] DataFrame abstraction over Polars
  - [ ] Column types with metadata (nullable, categorical levels, etc.)
  - [ ] Efficient string interning for categorical data

- [ ] **2.2 Serialization Layer**
  - [ ] Implement `serde` for all core types
  - [ ] Model artifact format (compatible with existing .pkl)
  - [ ] Efficient binary format for large arrays
  - [ ] JSON config serialization

- [ ] **2.3 Memory Management**
  ```rust
  pub struct MemoryPool {
      arena: bumpalo::Bump,
      stats: MemoryStats,
  }
  
  pub struct LruCache<K, V> {
      capacity: usize,
      ttl: Option<Duration>,
      map: HashMap<K, CacheEntry<V>>,
  }
  ```
  - [ ] Arena allocator for temporary batch data
  - [ ] LRU cache with TTL support
  - [ ] Memory-mapped file support for large datasets
  - [ ] Memory pressure monitoring

- [ ] **2.4 Configuration System**
  ```rust
  #[derive(Debug, Clone, Serialize, Deserialize)]
  pub struct TrainConfig {
      pub task_type: TaskType,
      pub cv_folds: u32,
      pub early_stopping_rounds: Option<u32>,
      pub time_budget_seconds: Option<u64>,
      // ... 
  }
  ```
  - [ ] Typed configuration structs with validation
  - [ ] Environment variable overrides
  - [ ] YAML/TOML config file support
  - [ ] Runtime config hot-reloading

---

### Phase 3: Data Processing Engine

**Objective**: Port all data preprocessing to Rust with zero-copy where possible

- [ ] **3.1 Data Loading** (`optimized_data_loader.rs`)
  ```rust
  pub trait DataLoader: Send + Sync {
      fn load(&self, path: &Path) -> Result<AutoMLDataFrame>;
      fn load_chunked(&self, path: &Path, chunk_size: usize) -> ChunkedReader;
  }
  
  pub struct CsvLoader { /* config */ }
  pub struct ParquetLoader { /* config */ }
  pub struct ArrowLoader { /* config */ }
  ```
  - [ ] CSV loading with type inference
  - [ ] Parquet/Arrow native loading
  - [ ] Chunked/streaming loading for large files
  - [ ] Parallel column parsing
  - [ ] Memory-mapped loading option

- [ ] **3.2 Data Preprocessor** (`data_preprocessor.rs`)
  ```rust
  pub struct DataPreprocessor {
      numeric_transformer: NumericTransformer,
      categorical_encoder: CategoricalEncoder,
      imputer: Imputer,
      scaler: Scaler,
      fitted: bool,
  }
  
  impl DataPreprocessor {
      pub fn fit(&mut self, df: &AutoMLDataFrame) -> Result<()>;
      pub fn transform(&self, df: &AutoMLDataFrame) -> Result<AutoMLDataFrame>;
      pub fn fit_transform(&mut self, df: &AutoMLDataFrame) -> Result<AutoMLDataFrame>;
      pub fn inverse_transform(&self, df: &AutoMLDataFrame) -> Result<AutoMLDataFrame>;
  }
  ```
  
  - [ ] **Numeric Processing**
    - [ ] Missing value detection (NaN, None, special values)
    - [ ] Outlier detection (IQR, Z-score, Isolation Forest)
    - [ ] Outlier handling (clip, remove, impute)
    - [ ] Log/power transforms
    - [ ] Binning/discretization
  
  - [ ] **Categorical Encoding**
    - [ ] Label encoding with unknown handling
    - [ ] One-hot encoding (sparse output)
    - [ ] Target encoding with smoothing
    - [ ] Frequency encoding
    - [ ] Hash encoding (for high-cardinality)
    - [ ] Binary encoding
  
  - [ ] **Imputation**
    - [ ] Mean/median/mode imputation
    - [ ] KNN imputation
    - [ ] Iterative imputation (MICE-like)
    - [ ] Constant value imputation
  
  - [ ] **Scaling**
    - [ ] StandardScaler (z-score)
    - [ ] MinMaxScaler
    - [ ] RobustScaler (IQR-based)
    - [ ] MaxAbsScaler
    - [ ] Normalizer (L1/L2)

- [ ] **3.3 Batch Processor** (`batch_processor.rs`)
  ```rust
  pub struct BatchProcessor {
      batch_size: usize,
      memory_limit: usize,
      strategy: BatchStrategy,
  }
  
  pub enum BatchStrategy {
      Fixed(usize),
      Dynamic { min: usize, max: usize },
      MemoryAware { target_memory: usize },
  }
  ```
  - [ ] Dynamic batch sizing based on memory
  - [ ] Parallel batch processing with rayon
  - [ ] Batch statistics accumulation
  - [ ] Progress tracking with callbacks

- [ ] **3.4 Streaming Pipeline** (`streaming_pipeline.rs`)
  ```rust
  pub struct StreamingPipeline<T> {
      stages: Vec<Box<dyn PipelineStage<T>>>,
      buffer_size: usize,
  }
  
  pub trait PipelineStage<T>: Send + Sync {
      fn process(&self, input: T) -> Result<T>;
  }
  ```
  - [ ] Async streaming with tokio channels
  - [ ] Backpressure handling
  - [ ] Pipeline stage composition
  - [ ] Error recovery and retry logic

---

### Phase 4: Training Engine

**Objective**: Port training orchestration and ML algorithm integration

- [ ] **4.1 Training Engine Core** (`train_engine.rs`)
  ```rust
  pub struct TrainEngine {
      config: TrainConfig,
      preprocessor: DataPreprocessor,
      model_registry: ModelRegistry,
      experiment_tracker: ExperimentTracker,
  }
  
  impl TrainEngine {
      pub async fn train(&mut self, data: TrainData) -> Result<TrainedModel>;
      pub fn cross_validate(&self, data: &TrainData, cv: CV) -> Result<CVResults>;
      pub fn evaluate(&self, model: &TrainedModel, data: &TestData) -> Result<Metrics>;
  }
  ```
  
  - [ ] **Cross-Validation**
    - [ ] K-Fold CV
    - [ ] Stratified K-Fold
    - [ ] Time Series Split
    - [ ] Group K-Fold
    - [ ] Repeated K-Fold
    - [ ] Leave-One-Out CV
  
  - [ ] **Training Loop**
    - [ ] Early stopping with patience
    - [ ] Learning rate scheduling
    - [ ] Checkpoint saving
    - [ ] Progress callbacks
    - [ ] Time budget enforcement
  
  - [ ] **Metrics Computation**
    - [ ] Classification: accuracy, precision, recall, F1, AUC-ROC, AUC-PR, log-loss
    - [ ] Regression: MSE, RMSE, MAE, MAPE, R², adjusted R²
    - [ ] Custom metric support

- [ ] **4.2 Model Registry** (`model_registry.rs`)
  ```rust
  pub trait Model: Send + Sync {
      fn fit(&mut self, X: &Array2<f64>, y: &Array1<f64>) -> Result<()>;
      fn predict(&self, X: &Array2<f64>) -> Result<Array1<f64>>;
      fn predict_proba(&self, X: &Array2<f64>) -> Result<Array2<f64>>;
      fn feature_importances(&self) -> Option<Array1<f64>>;
  }
  
  pub struct ModelRegistry {
      models: HashMap<String, Box<dyn ModelFactory>>,
  }
  ```
  - [ ] XGBoost integration via xgboost-rs
  - [ ] LightGBM integration via lightgbm-rs
  - [ ] Native Rust implementations:
    - [ ] Linear/Logistic Regression
    - [ ] Decision Tree
    - [ ] Random Forest
    - [ ] Gradient Boosting
  - [ ] Model serialization/deserialization

- [ ] **4.3 Adaptive Hyperparameter Optimization** (`adaptive_hyperopt.rs`)
  ```rust
  pub struct AdaptiveHPO {
      sampler: Box<dyn Sampler>,
      pruner: Box<dyn Pruner>,
      history: TrialHistory,
  }
  
  pub trait Sampler {
      fn sample(&self, space: &SearchSpace, history: &TrialHistory) -> Params;
  }
  
  pub trait Pruner {
      fn should_prune(&self, trial: &Trial, step: usize, value: f64) -> bool;
  }
  ```
  - [ ] TPE (Tree-structured Parzen Estimator) sampler
  - [ ] CMA-ES sampler
  - [ ] Random sampler
  - [ ] Grid sampler
  - [ ] Median pruner
  - [ ] Hyperband pruner

---

### Phase 5: Inference Engine

**Objective**: High-performance inference with minimal latency

- [ ] **5.1 Inference Engine** (`inference_engine.rs`)
  ```rust
  pub struct InferenceEngine {
      model_cache: LruCache<String, LoadedModel>,
      preprocessor_cache: LruCache<String, DataPreprocessor>,
      thread_pool: rayon::ThreadPool,
      config: InferenceConfig,
  }
  
  impl InferenceEngine {
      pub fn predict(&self, model_id: &str, input: PredictInput) -> Result<PredictOutput>;
      pub fn predict_batch(&self, model_id: &str, inputs: Vec<PredictInput>) -> Result<Vec<PredictOutput>>;
      pub async fn predict_stream(&self, model_id: &str, stream: impl Stream<Item = PredictInput>) -> impl Stream<Item = Result<PredictOutput>>;
  }
  ```
  
  - [ ] **Model Loading**
    - [ ] Lazy model loading
    - [ ] Model warm-up
    - [ ] Model versioning
    - [ ] Hot model swapping
  
  - [ ] **Prediction Paths**
    - [ ] Single prediction (optimized for latency)
    - [ ] Batch prediction (optimized for throughput)
    - [ ] Streaming prediction
    - [ ] Async prediction
  
  - [ ] **Caching**
    - [ ] Model cache with LRU eviction
    - [ ] Preprocessor cache
    - [ ] Prediction cache (optional)

- [ ] **5.2 Model Quantization** (`quantizer.rs`)
  ```rust
  pub struct Quantizer {
      precision: QuantizationPrecision,
      calibration_method: CalibrationMethod,
  }
  
  pub enum QuantizationPrecision {
      Int8,
      Int16,
      Float16,
      BFloat16,
  }
  
  pub enum CalibrationMethod {
      MinMax,
      Percentile(f64),
      Entropy,
  }
  ```
  - [ ] Post-training quantization
  - [ ] Calibration data collection
  - [ ] Quantized inference
  - [ ] Accuracy validation

---

### Phase 6: Optimizer (HyperOptX)

**Objective**: Port the custom hyperparameter optimization framework

- [ ] **6.1 Search Space** (`search_space.rs`)
  ```rust
  pub enum Distribution {
      Uniform { low: f64, high: f64 },
      LogUniform { low: f64, high: f64 },
      IntUniform { low: i64, high: i64 },
      Categorical { choices: Vec<String> },
      Conditional { parent: String, condition: Condition, space: Box<SearchSpace> },
  }
  
  pub struct SearchSpace {
      params: HashMap<String, Distribution>,
  }
  ```

- [ ] **6.2 HyperOptX Engine** (`hyperoptx.rs`)
  ```rust
  pub struct HyperOptX {
      search_space: SearchSpace,
      sampler: Box<dyn Sampler>,
      pruner: Box<dyn Pruner>,
      study: Study,
      parallel_trials: usize,
  }
  
  impl HyperOptX {
      pub fn optimize<F>(&mut self, objective: F, n_trials: usize) -> Result<BestParams>
      where
          F: Fn(&Params) -> Result<f64> + Send + Sync;
      
      pub async fn optimize_async<F, Fut>(&mut self, objective: F, n_trials: usize) -> Result<BestParams>
      where
          F: Fn(Params) -> Fut + Send + Sync,
          Fut: Future<Output = Result<f64>> + Send;
  }
  ```
  
  - [ ] **Samplers**
    - [ ] TPE with multivariate support
    - [ ] GP-based Bayesian optimization
    - [ ] CMA-ES
    - [ ] NSGA-II (multi-objective)
  
  - [ ] **Pruners**
    - [ ] Median pruner
    - [ ] Percentile pruner
    - [ ] Hyperband
    - [ ] Successive Halving (ASHA)
  
  - [ ] **Study Management**
    - [ ] Trial history persistence (SQLite/JSON)
    - [ ] Distributed optimization support
    - [ ] Warm-starting from previous studies

- [ ] **6.3 ASHT** (`asht.rs`)
  ```rust
  pub struct ASHT {
      min_resource: u64,
      max_resource: u64,
      reduction_factor: f64,
      brackets: Vec<Bracket>,
  }
  ```
  - [ ] Adaptive resource allocation
  - [ ] Bracket management
  - [ ] Promotion logic
  - [ ] Early termination

---

### Phase 7: Performance Utilities

**Objective**: Low-level optimizations for maximum performance

- [ ] **7.1 SIMD Operations** (`simd.rs`)
  ```rust
  #[cfg(target_arch = "x86_64")]
  use std::arch::x86_64::*;
  
  pub fn dot_product_simd(a: &[f32], b: &[f32]) -> f32;
  pub fn euclidean_distance_simd(a: &[f32], b: &[f32]) -> f32;
  pub fn sum_simd(a: &[f32]) -> f32;
  pub fn mean_simd(a: &[f32]) -> f32;
  pub fn variance_simd(a: &[f32]) -> f32;
  ```
  - [ ] AVX2/AVX-512 implementations
  - [ ] NEON implementations (ARM)
  - [ ] Fallback scalar implementations
  - [ ] Runtime feature detection

- [ ] **7.2 Parallel Execution** (`parallel.rs`)
  ```rust
  pub struct ParallelExecutor {
      thread_pool: rayon::ThreadPool,
      task_queue: crossbeam::deque::Injector<Task>,
  }
  ```
  - [ ] Custom thread pool with work stealing
  - [ ] Async task execution
  - [ ] CPU affinity control
  - [ ] NUMA-aware allocation

- [ ] **7.3 Memory Optimization**
  - [ ] Custom allocator (mimalloc/jemalloc)
  - [ ] Memory-mapped arrays
  - [ ] Copy-on-write semantics
  - [ ] Zero-copy buffer sharing

---

### Phase 8: Python Bindings (PyO3)

**Objective**: Seamless Python integration with minimal overhead

- [ ] **8.1 Module Structure**
  ```rust
  #[pymodule]
  fn automl_rust(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
      m.add_class::<PyDataPreprocessor>()?;
      m.add_class::<PyTrainEngine>()?;
      m.add_class::<PyInferenceEngine>()?;
      m.add_class::<PyHyperOptX>()?;
      Ok(())
  }
  ```

- [ ] **8.2 Data Conversion**
  ```rust
  impl<'py> FromPyObject<'py> for AutoMLArray<f64> {
      fn extract(ob: &'py PyAny) -> PyResult<Self> {
          let array: PyReadonlyArray2<f64> = ob.extract()?;
          // Zero-copy when possible
          Ok(AutoMLArray::from_numpy(array))
      }
  }
  ```
  - [ ] NumPy array ↔ ndarray (zero-copy where possible)
  - [ ] Pandas DataFrame ↔ Polars DataFrame
  - [ ] Python dict ↔ Rust HashMap
  - [ ] Python list ↔ Rust Vec
  - [ ] Async Python ↔ Tokio

- [ ] **8.3 Error Handling**
  ```rust
  #[derive(Debug, thiserror::Error)]
  pub enum AutoMLError {
      #[error("Data processing error: {0}")]
      DataError(String),
      #[error("Training error: {0}")]
      TrainError(String),
      // ...
  }
  
  impl From<AutoMLError> for PyErr {
      fn from(err: AutoMLError) -> PyErr {
          PyRuntimeError::new_err(err.to_string())
      }
  }
  ```

- [ ] **8.4 Type Stubs**
  - [ ] Generate .pyi files for IDE support
  - [ ] Full type annotations
  - [ ] Docstrings

---

### Phase 9: API Integration

**Objective**: Update Python API layer to use Rust backend

- [ ] **9.1 FastAPI Integration**
  - [ ] Update endpoints to use Rust modules
  - [ ] Benchmark response times
  - [ ] Maintain backward compatibility
  - [ ] Add feature flag for fallback

- [ ] **9.2 Optional: Native HTTP Server**
  ```rust
  // Using Axum
  async fn predict(
      State(engine): State<Arc<InferenceEngine>>,
      Json(input): Json<PredictRequest>,
  ) -> Result<Json<PredictResponse>, AppError> {
      let output = engine.predict(&input.model_id, input.data)?;
      Ok(Json(PredictResponse { predictions: output }))
  }
  ```
  - [ ] Axum-based HTTP server
  - [ ] gRPC support (tonic)
  - [ ] WebSocket support for streaming

---

### Phase 10: Documentation & Release

- [ ] **10.1 Documentation**
  - [ ] Rust API docs (rustdoc)
  - [ ] Python API docs update
  - [ ] Architecture documentation
  - [ ] Migration guide

- [ ] **10.2 Release**
  - [ ] Semantic versioning
  - [ ] Changelog generation
  - [ ] PyPI publishing
  - [ ] crates.io publishing (core library)

---

## Rust Architecture

### Project Structure

```
automl/
├── Cargo.toml                          # Workspace root
├── rust-toolchain.toml                 # Pin Rust version
├── .cargo/
│   └── config.toml                     # Build optimizations
│
├── rust/
│   ├── automl-core/                   # Core library (no Python deps)
│   │   ├── Cargo.toml
│   │   └── src/
│   │       ├── lib.rs                  # Public API
│   │       ├── error.rs                # Error types
│   │       │
│   │       ├── data/                   # Data structures
│   │       │   ├── mod.rs
│   │       │   ├── array.rs            # N-dimensional arrays
│   │       │   ├── dataframe.rs        # DataFrame abstraction
│   │       │   ├── column.rs           # Column types
│   │       │   ├── dtype.rs            # Data types
│   │       │   └── series.rs           # Series type
│   │       │
│   │       ├── preprocessing/          # Data preprocessing
│   │       │   ├── mod.rs
│   │       │   ├── preprocessor.rs     # Main preprocessor
│   │       │   ├── numeric.rs          # Numeric transforms
│   │       │   ├── categorical.rs      # Categorical encoding
│   │       │   ├── imputer.rs          # Missing value imputation
│   │       │   ├── scaler.rs           # Feature scaling
│   │       │   ├── outlier.rs          # Outlier handling
│   │       │   └── text.rs             # Text preprocessing
│   │       │
│   │       ├── training/               # Training engine
│   │       │   ├── mod.rs
│   │       │   ├── engine.rs           # Main training engine
│   │       │   ├── cv.rs               # Cross-validation
│   │       │   ├── metrics.rs          # Evaluation metrics
│   │       │   ├── callbacks.rs        # Training callbacks
│   │       │   ├── early_stopping.rs   # Early stopping
│   │       │   └── checkpoint.rs       # Model checkpointing
│   │       │
│   │       ├── inference/              # Inference engine
│   │       │   ├── mod.rs
│   │       │   ├── engine.rs           # Main inference engine
│   │       │   ├── batch.rs            # Batch inference
│   │       │   ├── stream.rs           # Streaming inference
│   │       │   └── quantizer.rs        # Model quantization
│   │       │
│   │       ├── models/                 # ML models
│   │       │   ├── mod.rs
│   │       │   ├── traits.rs           # Model traits
│   │       │   ├── registry.rs         # Model registry
│   │       │   ├── linear.rs           # Linear models
│   │       │   ├── tree.rs             # Decision trees
│   │       │   ├── ensemble.rs         # Ensemble methods
│   │       │   ├── xgboost.rs          # XGBoost wrapper
│   │       │   └── lightgbm.rs         # LightGBM wrapper
│   │       │
│   │       ├── optimizer/              # Hyperparameter optimization
│   │       │   ├── mod.rs
│   │       │   ├── hyperoptx.rs        # Main optimizer
│   │       │   ├── search_space.rs     # Search space definitions
│   │       │   ├── sampler/
│   │       │   │   ├── mod.rs
│   │       │   │   ├── tpe.rs          # TPE sampler
│   │       │   │   ├── cmaes.rs        # CMA-ES sampler
│   │       │   │   ├── random.rs       # Random sampler
│   │       │   │   └── grid.rs         # Grid sampler
│   │       │   ├── pruner/
│   │       │   │   ├── mod.rs
│   │       │   │   ├── median.rs       # Median pruner
│   │       │   │   ├── hyperband.rs    # Hyperband pruner
│   │       │   │   └── asha.rs         # ASHA pruner
│   │       │   ├── study.rs            # Study management
│   │       │   └── trial.rs            # Trial management
│   │       │
│   │       ├── io/                     # Data I/O
│   │       │   ├── mod.rs
│   │       │   ├── csv.rs              # CSV loading
│   │       │   ├── parquet.rs          # Parquet loading
│   │       │   ├── arrow.rs            # Arrow loading
│   │       │   └── model.rs            # Model serialization
│   │       │
│   │       ├── cache/                  # Caching utilities
│   │       │   ├── mod.rs
│   │       │   ├── lru.rs              # LRU cache
│   │       │   ├── ttl.rs              # TTL cache
│   │       │   └── memory_pool.rs      # Memory pool
│   │       │
│   │       └── utils/                  # Utilities
│   │           ├── mod.rs
│   │           ├── simd.rs             # SIMD operations
│   │           ├── parallel.rs         # Parallel utilities
│   │           ├── stats.rs            # Statistical functions
│   │           └── math.rs             # Math utilities
│   │
│   ├── automl-python/                 # PyO3 bindings
│   │   ├── Cargo.toml
│   │   └── src/
│   │       ├── lib.rs                  # Module definition
│   │       ├── convert.rs              # Type conversions
│   │       ├── error.rs                # Python error handling
│   │       ├── preprocessing.rs        # Preprocessor bindings
│   │       ├── training.rs             # Training bindings
│   │       ├── inference.rs            # Inference bindings
│   │       ├── optimizer.rs            # Optimizer bindings
│   │       └── async_support.rs        # Async Python support
│   │
│   ├── automl-server/                 # Optional native server
│   │   ├── Cargo.toml
│   │   └── src/
│   │       ├── main.rs
│   │       ├── routes/
│   │       │   ├── mod.rs
│   │       │   ├── predict.rs
│   │       │   ├── train.rs
│   │       │   └── health.rs
│   │       └── state.rs
│   │
│   └── automl-cli/                    # Optional CLI
│       ├── Cargo.toml
│       └── src/
│           └── main.rs
│
├── python/
│   └── automl_rust/                   # Python package
│       ├── __init__.py
│       ├── __init__.pyi                # Type stubs
│       ├── preprocessing.pyi
│       ├── training.pyi
│       ├── inference.pyi
│       └── optimizer.pyi
│
├── pyproject.toml                      # Updated for maturin
├── modules/                            # Existing Python modules
├── tests/
│   ├── rust/                           # Rust tests
│   │   ├── unit/
│   │   ├── integration/
│   │   ├── property/
│   │   └── benchmark/
│   └── python/                         # Python integration tests
│       └── test_rust_backend.py
│
└── benches/                            # Criterion benchmarks
    ├── preprocessing.rs
    ├── training.rs
    ├── inference.rs
    └── optimizer.rs
```

### Cargo Workspace Configuration

```toml
# Cargo.toml (root)
[workspace]
resolver = "2"
members = [
    "rust/automl-core",
    "rust/automl-python",
    "rust/automl-server",
    "rust/automl-cli",
]

[workspace.package]
version = "0.2.0"
edition = "2021"
rust-version = "1.75"
license = "MIT"
repository = "https://github.com/Evintkoo/automl"

[workspace.dependencies]
# Core
ndarray = { version = "0.15", features = ["rayon", "serde"] }
polars = { version = "0.35", features = ["lazy", "parquet", "json", "csv"] }
arrow = "50.0"
rayon = "1.8"
tokio = { version = "1.35", features = ["full"] }
serde = { version = "1.0", features = ["derive"] }
serde_json = "1.0"
thiserror = "1.0"
anyhow = "1.0"
tracing = "0.1"
tracing-subscriber = "0.3"

# ML
xgboost = "0.1"
lightgbm = "0.2"

# Optimization
rand = "0.8"
rand_distr = "0.4"
statrs = "0.16"

# Python bindings
pyo3 = { version = "0.20", features = ["extension-module", "abi3-py39"] }
numpy = "0.20"

# Testing
proptest = "1.4"
criterion = { version = "0.5", features = ["html_reports"] }
approx = "0.5"

# Performance
mimalloc = { version = "0.1", features = ["local_dynamic_tls"] }

[profile.release]
lto = "thin"
codegen-units = 1
panic = "abort"
strip = true

[profile.bench]
lto = "thin"
codegen-units = 1
```

---

## Module Specifications

### 5.1 Data Preprocessor Specification

```rust
//! Data Preprocessor Module
//! 
//! Provides comprehensive data preprocessing capabilities including:
//! - Numeric feature transformations
//! - Categorical encoding
//! - Missing value imputation
//! - Feature scaling
//! - Outlier handling

use ndarray::{Array1, Array2};
use serde::{Deserialize, Serialize};

/// Main preprocessor that orchestrates all preprocessing steps
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataPreprocessor {
    /// Configuration for preprocessing
    config: PreprocessorConfig,
    /// Fitted state for each column
    column_states: HashMap<String, ColumnState>,
    /// Whether the preprocessor has been fitted
    fitted: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PreprocessorConfig {
    /// Strategy for handling numeric features
    pub numeric_strategy: NumericStrategy,
    /// Strategy for handling categorical features
    pub categorical_strategy: CategoricalStrategy,
    /// Strategy for handling missing values
    pub missing_strategy: MissingStrategy,
    /// Strategy for scaling features
    pub scaling_strategy: ScalingStrategy,
    /// Strategy for handling outliers
    pub outlier_strategy: OutlierStrategy,
    /// Whether to infer column types automatically
    pub auto_detect_types: bool,
    /// Maximum cardinality for categorical encoding
    pub max_cardinality: usize,
}

/// Numeric transformation strategies
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum NumericStrategy {
    /// Keep as-is
    Passthrough,
    /// Log transform (log1p for handling zeros)
    Log,
    /// Square root transform
    Sqrt,
    /// Box-Cox transform
    BoxCox { lambda: Option<f64> },
    /// Yeo-Johnson transform (handles negative values)
    YeoJohnson { lambda: Option<f64> },
    /// Quantile transform to uniform distribution
    QuantileUniform { n_quantiles: usize },
    /// Quantile transform to normal distribution
    QuantileNormal { n_quantiles: usize },
}

/// Categorical encoding strategies
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CategoricalStrategy {
    /// Integer label encoding
    Label { handle_unknown: UnknownHandler },
    /// One-hot encoding
    OneHot { 
        sparse: bool, 
        drop: OneHotDrop,
        handle_unknown: UnknownHandler,
    },
    /// Target encoding with smoothing
    Target { 
        smoothing: f64,
        min_samples: usize,
    },
    /// Frequency encoding
    Frequency,
    /// Binary encoding
    Binary,
    /// Hash encoding for high cardinality
    Hash { n_components: usize },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum UnknownHandler {
    Error,
    Ignore,
    InfrequentCategory,
    UseDefault(String),
}

/// Missing value imputation strategies
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MissingStrategy {
    /// Drop rows with missing values
    Drop,
    /// Impute with mean (numeric only)
    Mean,
    /// Impute with median (numeric only)
    Median,
    /// Impute with mode (most frequent)
    Mode,
    /// Impute with constant value
    Constant { value: serde_json::Value },
    /// K-nearest neighbors imputation
    Knn { n_neighbors: usize },
    /// Iterative imputation (MICE-like)
    Iterative { max_iter: usize, tol: f64 },
}

/// Feature scaling strategies
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ScalingStrategy {
    /// No scaling
    None,
    /// Standardization (z-score normalization)
    Standard,
    /// Min-max normalization to [0, 1]
    MinMax,
    /// Min-max normalization to custom range
    MinMaxRange { min: f64, max: f64 },
    /// Robust scaling using IQR
    Robust { 
        with_centering: bool,
        quantile_range: (f64, f64),
    },
    /// Max absolute scaling
    MaxAbs,
    /// L1/L2 normalization
    Normalizer { norm: Norm },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Norm {
    L1,
    L2,
    Max,
}

/// Outlier handling strategies
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum OutlierStrategy {
    /// No outlier handling
    None,
    /// Clip outliers to bounds
    Clip { 
        method: OutlierMethod,
        lower: Option<f64>,
        upper: Option<f64>,
    },
    /// Replace outliers with NaN (then impute)
    ToNaN { method: OutlierMethod },
    /// Remove outlier rows
    Remove { method: OutlierMethod },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum OutlierMethod {
    /// IQR-based detection
    IQR { factor: f64 },
    /// Z-score based detection
    ZScore { threshold: f64 },
    /// Percentile-based bounds
    Percentile { lower: f64, upper: f64 },
}
```

### 5.2 Training Engine Specification

```rust
//! Training Engine Module
//!
//! Provides model training orchestration including:
//! - Cross-validation
//! - Early stopping
//! - Hyperparameter optimization integration
//! - Experiment tracking

use std::sync::Arc;
use tokio::sync::mpsc;

/// Main training engine
pub struct TrainEngine {
    config: TrainConfig,
    preprocessor: DataPreprocessor,
    model_registry: Arc<ModelRegistry>,
    optimizer: Option<Box<dyn Optimizer>>,
    callbacks: Vec<Box<dyn TrainCallback>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainConfig {
    /// Type of ML task
    pub task_type: TaskType,
    /// Cross-validation configuration
    pub cv: CVConfig,
    /// Early stopping configuration
    pub early_stopping: Option<EarlyStoppingConfig>,
    /// Time budget in seconds (optional)
    pub time_budget: Option<u64>,
    /// Maximum number of models to train
    pub max_models: Option<usize>,
    /// Evaluation metric
    pub metric: Metric,
    /// Whether to optimize metric (maximize or minimize)
    pub direction: Direction,
    /// Number of parallel jobs
    pub n_jobs: Option<usize>,
    /// Random seed for reproducibility
    pub seed: Option<u64>,
    /// Verbosity level
    pub verbosity: Verbosity,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TaskType {
    BinaryClassification,
    MulticlassClassification,
    Regression,
    MultiOutputRegression,
    Ranking,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CVConfig {
    /// Type of cross-validation
    pub cv_type: CVType,
    /// Number of folds
    pub n_folds: usize,
    /// Whether to shuffle data before splitting
    pub shuffle: bool,
    /// Random seed for shuffling
    pub seed: Option<u64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CVType {
    KFold,
    StratifiedKFold,
    GroupKFold { group_column: String },
    TimeSeriesSplit { gap: usize },
    RepeatedKFold { n_repeats: usize },
    LeaveOneOut,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EarlyStoppingConfig {
    /// Number of rounds without improvement before stopping
    pub patience: usize,
    /// Minimum improvement to be considered significant
    pub min_delta: f64,
    /// Whether to restore best weights
    pub restore_best: bool,
    /// Metric to monitor
    pub monitor: String,
}

/// Training callbacks for customization
pub trait TrainCallback: Send + Sync {
    fn on_train_begin(&mut self, state: &TrainState) {}
    fn on_train_end(&mut self, state: &TrainState) {}
    fn on_epoch_begin(&mut self, epoch: usize, state: &TrainState) {}
    fn on_epoch_end(&mut self, epoch: usize, state: &TrainState, metrics: &Metrics) {}
    fn on_fold_begin(&mut self, fold: usize, state: &TrainState) {}
    fn on_fold_end(&mut self, fold: usize, state: &TrainState, metrics: &Metrics) {}
}

/// Built-in callbacks
pub struct ProgressCallback { /* progress bar state */ }
pub struct CheckpointCallback { path: PathBuf, save_best_only: bool }
pub struct EarlyStoppingCallback { config: EarlyStoppingConfig }
pub struct LoggingCallback { logger: tracing::Span }
pub struct MetricHistoryCallback { history: Vec<Metrics> }

/// Training result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainResult {
    /// Best model found
    pub best_model: TrainedModel,
    /// Cross-validation results
    pub cv_results: CVResults,
    /// Training history
    pub history: TrainHistory,
    /// Total training time
    pub train_time: Duration,
    /// Best parameters (if HPO was used)
    pub best_params: Option<HashMap<String, serde_json::Value>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CVResults {
    pub fold_scores: Vec<f64>,
    pub mean_score: f64,
    pub std_score: f64,
    pub fold_times: Vec<Duration>,
}
```

### 5.3 HyperOptX Specification

```rust
//! HyperOptX - Advanced Hyperparameter Optimization
//!
//! Features:
//! - Multiple sampling strategies (TPE, CMA-ES, Random, Grid)
//! - Pruning for early termination
//! - Parallel trial execution
//! - Warm-starting from previous studies

/// Main HyperOptX optimizer
pub struct HyperOptX {
    search_space: SearchSpace,
    sampler: Box<dyn Sampler>,
    pruner: Option<Box<dyn Pruner>>,
    study: Study,
    config: HyperOptXConfig,
}

#[derive(Debug, Clone)]
pub struct HyperOptXConfig {
    /// Number of parallel trials
    pub n_parallel: usize,
    /// Maximum number of trials
    pub n_trials: usize,
    /// Time budget in seconds
    pub timeout: Option<Duration>,
    /// Direction of optimization
    pub direction: Direction,
    /// Seed for reproducibility
    pub seed: Option<u64>,
}

/// Search space definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchSpace {
    params: HashMap<String, Parameter>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Parameter {
    /// Continuous uniform distribution
    Float {
        low: f64,
        high: f64,
        log: bool,
        step: Option<f64>,
    },
    /// Integer uniform distribution
    Int {
        low: i64,
        high: i64,
        log: bool,
        step: Option<i64>,
    },
    /// Categorical parameter
    Categorical {
        choices: Vec<serde_json::Value>,
    },
    /// Conditional parameter (depends on parent)
    Conditional {
        parent: String,
        condition: Condition,
        param: Box<Parameter>,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Condition {
    Equals(serde_json::Value),
    In(Vec<serde_json::Value>),
    GreaterThan(f64),
    LessThan(f64),
}

/// Sampler trait for different sampling strategies
pub trait Sampler: Send + Sync {
    /// Sample a new set of parameters
    fn sample(&self, space: &SearchSpace, study: &Study) -> Result<Params>;
    
    /// Update sampler state with trial result
    fn on_trial_complete(&mut self, trial: &Trial);
}

/// TPE (Tree-structured Parzen Estimator) sampler
pub struct TPESampler {
    n_startup_trials: usize,
    n_ei_candidates: usize,
    gamma: f64,
    multivariate: bool,
    consider_prior: bool,
    prior_weight: f64,
}

/// CMA-ES sampler for continuous optimization
pub struct CMAESSampler {
    sigma0: f64,
    n_startup_trials: usize,
    restart_strategy: RestartStrategy,
}

/// Pruner trait for early termination
pub trait Pruner: Send + Sync {
    /// Check if trial should be pruned
    fn should_prune(&self, trial: &Trial, step: usize, value: f64) -> bool;
}

/// Median pruner
pub struct MedianPruner {
    n_startup_trials: usize,
    n_warmup_steps: usize,
    interval_steps: usize,
    n_min_trials: usize,
}

/// Hyperband pruner
pub struct HyperbandPruner {
    min_resource: usize,
    max_resource: usize,
    reduction_factor: usize,
}

/// ASHA (Asynchronous Successive Halving)
pub struct ASHAPruner {
    min_resource: usize,
    reduction_factor: f64,
    brackets: usize,
}

/// Study for managing optimization state
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Study {
    study_name: String,
    direction: Direction,
    trials: Vec<Trial>,
    best_trial: Option<usize>,
    user_attrs: HashMap<String, serde_json::Value>,
}

/// Individual trial
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Trial {
    trial_id: usize,
    params: Params,
    values: Option<Vec<f64>>,
    intermediate_values: Vec<(usize, f64)>,
    state: TrialState,
    start_time: DateTime<Utc>,
    end_time: Option<DateTime<Utc>>,
    user_attrs: HashMap<String, serde_json::Value>,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum TrialState {
    Running,
    Complete,
    Pruned,
    Fail,
    Waiting,
}
```

```toml
[dependencies]
# Core
ndarray = "0.15"                 # N-dimensional arrays
rayon = "1.8"                    # Parallelism
serde = { version = "1.0", features = ["derive"] }
serde_json = "1.0"

# Data handling
polars = "0.35"                  # DataFrame operations
arrow = "50.0"                   # Arrow format support
csv = "1.3"                      # CSV parsing
parquet = "50.0"                 # Parquet support

# ML
linfa = "0.7"                    # ML algorithms
smartcore = "0.3"                # Additional ML
xgboost = "0.1"                  # XGBoost bindings

# Optimization
argmin = "0.8"                   # Optimization framework
rand = "0.8"                     # RNG

# Python bindings
pyo3 = { version = "0.20", features = ["extension-module"] }
numpy = "0.20"                   # NumPy interop

# Async
tokio = { version = "1.35", features = ["full"] }

# Performance
packed_simd = "0.3"              # SIMD operations
mimalloc = "0.1"                 # Fast allocator
```

---

## Testing Strategy

### Overview

The testing strategy follows a pyramid approach with extensive automation:

```
                    ┌─────────────┐
                    │   E2E       │  ← Python integration tests
                   ┌┴─────────────┴┐
                   │  Integration  │  ← Cross-module tests
                  ┌┴───────────────┴┐
                  │   Property      │  ← Proptest fuzz tests
                 ┌┴─────────────────┴┐
                 │       Unit        │  ← Module-level tests
                └─────────────────────┘
```

### 6.1 Unit Tests

Located in `tests/rust/unit/` and inline with source code.

```rust
// Example: tests/rust/unit/preprocessing/test_scaler.rs

use automl_core::preprocessing::{Scaler, ScalingStrategy};
use ndarray::array;
use approx::assert_abs_diff_eq;

mod standard_scaler {
    use super::*;

    #[test]
    fn test_fit_calculates_mean_and_std() {
        let data = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]];
        let mut scaler = Scaler::new(ScalingStrategy::Standard);
        
        scaler.fit(&data).unwrap();
        
        assert_abs_diff_eq!(scaler.mean().unwrap(), array![3.0, 4.0], epsilon = 1e-10);
        assert_abs_diff_eq!(scaler.std().unwrap(), array![1.632993, 1.632993], epsilon = 1e-5);
    }

    #[test]
    fn test_transform_normalizes_data() {
        let data = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]];
        let mut scaler = Scaler::new(ScalingStrategy::Standard);
        scaler.fit(&data).unwrap();
        
        let transformed = scaler.transform(&data).unwrap();
        
        // Mean should be ~0, std should be ~1
        let col_mean = transformed.mean_axis(ndarray::Axis(0)).unwrap();
        let col_std = transformed.std_axis(ndarray::Axis(0), 0.0);
        
        assert_abs_diff_eq!(col_mean, array![0.0, 0.0], epsilon = 1e-10);
        assert_abs_diff_eq!(col_std, array![1.0, 1.0], epsilon = 1e-10);
    }

    #[test]
    fn test_inverse_transform_recovers_original() {
        let data = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]];
        let mut scaler = Scaler::new(ScalingStrategy::Standard);
        scaler.fit(&data).unwrap();
        
        let transformed = scaler.transform(&data).unwrap();
        let recovered = scaler.inverse_transform(&transformed).unwrap();
        
        assert_abs_diff_eq!(recovered, data, epsilon = 1e-10);
    }

    #[test]
    fn test_handles_zero_variance_columns() {
        let data = array![[1.0, 5.0], [1.0, 5.0], [1.0, 5.0]];
        let mut scaler = Scaler::new(ScalingStrategy::Standard);
        scaler.fit(&data).unwrap();
        
        let transformed = scaler.transform(&data).unwrap();
        
        // Zero variance columns should become 0
        assert!(transformed.iter().all(|&x| x == 0.0 || x.is_nan() == false));
    }

    #[test]
    fn test_unfitted_transform_returns_error() {
        let data = array![[1.0, 2.0]];
        let scaler = Scaler::new(ScalingStrategy::Standard);
        
        let result = scaler.transform(&data);
        
        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), AutoMLError::NotFitted(_)));
    }
}

mod minmax_scaler {
    use super::*;

    #[test]
    fn test_scales_to_0_1_range() {
        let data = array![[0.0], [50.0], [100.0]];
        let mut scaler = Scaler::new(ScalingStrategy::MinMax);
        scaler.fit(&data).unwrap();
        
        let transformed = scaler.transform(&data).unwrap();
        
        assert_abs_diff_eq!(transformed, array![[0.0], [0.5], [1.0]], epsilon = 1e-10);
    }

    #[test]
    fn test_scales_to_custom_range() {
        let data = array![[0.0], [50.0], [100.0]];
        let mut scaler = Scaler::new(ScalingStrategy::MinMaxRange { min: -1.0, max: 1.0 });
        scaler.fit(&data).unwrap();
        
        let transformed = scaler.transform(&data).unwrap();
        
        assert_abs_diff_eq!(transformed, array![[-1.0], [0.0], [1.0]], epsilon = 1e-10);
    }
}

mod robust_scaler {
    use super::*;

    #[test]
    fn test_uses_median_and_iqr() {
        // Data with outlier
        let data = array![[1.0], [2.0], [3.0], [4.0], [5.0], [1000.0]];
        let mut scaler = Scaler::new(ScalingStrategy::Robust {
            with_centering: true,
            quantile_range: (25.0, 75.0),
        });
        scaler.fit(&data).unwrap();
        
        // Median should be ~3.5, IQR should be ~3
        assert_abs_diff_eq!(scaler.center().unwrap()[0], 3.5, epsilon = 0.1);
    }

    #[test]
    fn test_resistant_to_outliers() {
        let normal_data = array![[1.0], [2.0], [3.0], [4.0], [5.0]];
        let outlier_data = array![[1.0], [2.0], [3.0], [4.0], [1000.0]];
        
        let mut scaler_normal = Scaler::new(ScalingStrategy::Robust {
            with_centering: true,
            quantile_range: (25.0, 75.0),
        });
        let mut scaler_outlier = Scaler::new(ScalingStrategy::Robust {
            with_centering: true,
            quantile_range: (25.0, 75.0),
        });
        
        scaler_normal.fit(&normal_data).unwrap();
        scaler_outlier.fit(&outlier_data).unwrap();
        
        // Centers should be similar despite outlier
        let center_diff = (scaler_normal.center().unwrap()[0] - scaler_outlier.center().unwrap()[0]).abs();
        assert!(center_diff < 1.0);
    }
}
```

### 6.2 Property-Based Tests (Proptest)

Located in `tests/rust/property/`

```rust
// tests/rust/property/test_preprocessing.rs

use proptest::prelude::*;
use ndarray::Array2;
use automl_core::preprocessing::{DataPreprocessor, PreprocessorConfig};

/// Generate random 2D arrays for testing
fn array_strategy(rows: usize, cols: usize) -> impl Strategy<Value = Array2<f64>> {
    prop::collection::vec(
        prop::num::f64::NORMAL,
        rows * cols
    ).prop_map(move |v| {
        Array2::from_shape_vec((rows, cols), v).unwrap()
    })
}

proptest! {
    /// Property: fit_transform followed by inverse_transform should recover original data
    #[test]
    fn prop_inverse_transform_recovers_original(
        data in array_strategy(10..100, 2..10)
    ) {
        let mut preprocessor = DataPreprocessor::new(PreprocessorConfig {
            scaling_strategy: ScalingStrategy::Standard,
            ..Default::default()
        });
        
        let transformed = preprocessor.fit_transform(&data)?;
        let recovered = preprocessor.inverse_transform(&transformed)?;
        
        // Allow small floating point errors
        for (orig, rec) in data.iter().zip(recovered.iter()) {
            if orig.is_finite() {
                prop_assert!((orig - rec).abs() < 1e-6, 
                    "Expected {} but got {}", orig, rec);
            }
        }
    }

    /// Property: transformed data should have mean ~0 and std ~1 for StandardScaler
    #[test]
    fn prop_standard_scaler_normalizes(
        data in array_strategy(100..1000, 1..5)
            .prop_filter("need variance", |d| {
                d.columns().into_iter().all(|c| c.std(0.0) > 1e-10)
            })
    ) {
        let mut preprocessor = DataPreprocessor::new(PreprocessorConfig {
            scaling_strategy: ScalingStrategy::Standard,
            ..Default::default()
        });
        
        let transformed = preprocessor.fit_transform(&data)?;
        
        for col in transformed.columns() {
            let mean = col.mean().unwrap();
            let std = col.std(0.0);
            
            prop_assert!((mean).abs() < 0.01, "Mean {} should be ~0", mean);
            prop_assert!((std - 1.0).abs() < 0.01, "Std {} should be ~1", std);
        }
    }

    /// Property: MinMaxScaler should produce values in [0, 1]
    #[test]
    fn prop_minmax_bounds(
        data in array_strategy(10..100, 1..5)
            .prop_filter("need range", |d| {
                d.columns().into_iter().all(|c| {
                    let min = c.iter().cloned().fold(f64::INFINITY, f64::min);
                    let max = c.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                    (max - min).abs() > 1e-10
                })
            })
    ) {
        let mut preprocessor = DataPreprocessor::new(PreprocessorConfig {
            scaling_strategy: ScalingStrategy::MinMax,
            ..Default::default()
        });
        
        let transformed = preprocessor.fit_transform(&data)?;
        
        for &val in transformed.iter() {
            if val.is_finite() {
                prop_assert!(val >= -1e-10 && val <= 1.0 + 1e-10,
                    "Value {} should be in [0, 1]", val);
            }
        }
    }

    /// Property: Label encoder should be deterministic
    #[test]
    fn prop_label_encoder_deterministic(
        categories in prop::collection::vec("[a-z]{1,5}", 10..50)
    ) {
        let mut encoder1 = LabelEncoder::new();
        let mut encoder2 = LabelEncoder::new();
        
        encoder1.fit(&categories)?;
        encoder2.fit(&categories)?;
        
        let encoded1 = encoder1.transform(&categories)?;
        let encoded2 = encoder2.transform(&categories)?;
        
        prop_assert_eq!(encoded1, encoded2);
    }

    /// Property: One-hot encoding should produce correct number of columns
    #[test]
    fn prop_onehot_column_count(
        categories in prop::collection::vec("[a-z]{1,3}", 10..100)
    ) {
        let unique_count = categories.iter().collect::<std::collections::HashSet<_>>().len();
        
        let mut encoder = OneHotEncoder::new(OneHotConfig {
            drop: OneHotDrop::None,
            ..Default::default()
        });
        encoder.fit(&categories)?;
        
        let encoded = encoder.transform(&categories)?;
        
        prop_assert_eq!(encoded.ncols(), unique_count,
            "Expected {} columns but got {}", unique_count, encoded.ncols());
    }
}
```

### 6.3 Integration Tests

Located in `tests/rust/integration/`

```rust
// tests/rust/integration/test_training_pipeline.rs

use automl_core::{
    preprocessing::DataPreprocessor,
    training::{TrainEngine, TrainConfig, CVConfig},
    models::ModelRegistry,
};
use tokio;

/// Helper to create test data
fn create_classification_data(n_samples: usize, n_features: usize) -> (Array2<f64>, Array1<i64>) {
    let mut rng = rand::thread_rng();
    let X = Array2::random((n_samples, n_features), rand_distr::Normal::new(0.0, 1.0).unwrap());
    let y = Array1::from_iter((0..n_samples).map(|i| if X[[i, 0]] > 0.0 { 1 } else { 0 }));
    (X, y)
}

mod training_pipeline {
    use super::*;

    #[tokio::test]
    async fn test_end_to_end_classification() {
        // Setup
        let (X, y) = create_classification_data(1000, 10);
        
        // Create pipeline
        let preprocessor = DataPreprocessor::default();
        let config = TrainConfig {
            task_type: TaskType::BinaryClassification,
            cv: CVConfig {
                cv_type: CVType::StratifiedKFold,
                n_folds: 5,
                shuffle: true,
                seed: Some(42),
            },
            metric: Metric::AucRoc,
            direction: Direction::Maximize,
            ..Default::default()
        };
        
        let mut engine = TrainEngine::new(config, preprocessor);
        
        // Train
        let result = engine.train(TrainData { X, y }).await.unwrap();
        
        // Assertions
        assert!(result.cv_results.mean_score > 0.5, "Should perform better than random");
        assert!(result.best_model.is_fitted());
        assert_eq!(result.cv_results.fold_scores.len(), 5);
    }

    #[tokio::test]
    async fn test_early_stopping_triggers() {
        let (X, y) = create_classification_data(1000, 10);
        
        let config = TrainConfig {
            early_stopping: Some(EarlyStoppingConfig {
                patience: 5,
                min_delta: 0.001,
                restore_best: true,
                monitor: "val_auc".to_string(),
            }),
            ..Default::default()
        };
        
        let mut engine = TrainEngine::new(config, DataPreprocessor::default());
        let result = engine.train(TrainData { X, y }).await.unwrap();
        
        // Early stopping should have triggered
        assert!(result.history.epochs_completed < 1000);
    }

    #[tokio::test]
    async fn test_model_serialization_roundtrip() {
        let (X, y) = create_classification_data(500, 5);
        
        let mut engine = TrainEngine::default();
        let result = engine.train(TrainData { X: X.clone(), y }).await.unwrap();
        
        // Serialize
        let serialized = result.best_model.to_bytes().unwrap();
        
        // Deserialize
        let loaded = TrainedModel::from_bytes(&serialized).unwrap();
        
        // Predictions should match
        let pred_original = result.best_model.predict(&X).unwrap();
        let pred_loaded = loaded.predict(&X).unwrap();
        
        assert_eq!(pred_original, pred_loaded);
    }
}

mod hyperparameter_optimization {
    use super::*;
    use automl_core::optimizer::{HyperOptX, SearchSpace, Parameter};

    #[tokio::test]
    async fn test_hyperoptx_finds_optimum() {
        // Simple quadratic function: minimize (x - 3)^2
        let search_space = SearchSpace::new()
            .add("x", Parameter::Float { low: -10.0, high: 10.0, log: false, step: None });
        
        let mut optimizer = HyperOptX::new(search_space, HyperOptXConfig {
            n_trials: 50,
            direction: Direction::Minimize,
            ..Default::default()
        });
        
        let result = optimizer.optimize(|params| {
            let x: f64 = params.get("x").unwrap();
            Ok((x - 3.0).powi(2))
        }, 50).await.unwrap();
        
        let best_x: f64 = result.best_params.get("x").unwrap();
        assert!((best_x - 3.0).abs() < 0.5, "Should find x close to 3, got {}", best_x);
    }

    #[tokio::test]
    async fn test_pruning_reduces_trials() {
        let search_space = SearchSpace::new()
            .add("x", Parameter::Float { low: 0.0, high: 100.0, log: false, step: None });
        
        let mut optimizer = HyperOptX::new(search_space, HyperOptXConfig {
            n_trials: 100,
            direction: Direction::Minimize,
            ..Default::default()
        }).with_pruner(MedianPruner::new(5, 2, 1, 5));
        
        let total_steps = std::sync::atomic::AtomicUsize::new(0);
        
        optimizer.optimize(|params| {
            let x: f64 = params.get("x").unwrap();
            // Report intermediate values
            for step in 0..10 {
                total_steps.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                let value = (x - 50.0).powi(2) + step as f64;
                // Check for pruning
                if optimizer.should_prune(step, value) {
                    return Err(AutoMLError::TrialPruned);
                }
            }
            Ok((x - 50.0).powi(2))
        }, 100).await.unwrap();
        
        // Pruning should reduce total steps
        let steps = total_steps.load(std::sync::atomic::Ordering::Relaxed);
        assert!(steps < 100 * 10, "Pruning should reduce steps, got {}", steps);
    }
}
```

### 6.4 Benchmark Tests

Located in `benches/` using Criterion.

```rust
// benches/preprocessing.rs

use criterion::{criterion_group, criterion_main, Criterion, BenchmarkId, Throughput};
use automl_core::preprocessing::{DataPreprocessor, PreprocessorConfig, ScalingStrategy};
use ndarray::Array2;
use rand::prelude::*;

fn generate_data(rows: usize, cols: usize) -> Array2<f64> {
    let mut rng = StdRng::seed_from_u64(42);
    Array2::from_shape_fn((rows, cols), |_| rng.gen_range(-100.0..100.0))
}

fn benchmark_standard_scaler(c: &mut Criterion) {
    let mut group = c.benchmark_group("StandardScaler");
    
    for size in [1_000, 10_000, 100_000, 1_000_000].iter() {
        let data = generate_data(*size, 10);
        group.throughput(Throughput::Elements(*size as u64 * 10));
        
        group.bench_with_input(
            BenchmarkId::new("fit_transform", size),
            &data,
            |b, data| {
                b.iter(|| {
                    let mut scaler = DataPreprocessor::new(PreprocessorConfig {
                        scaling_strategy: ScalingStrategy::Standard,
                        ..Default::default()
                    });
                    scaler.fit_transform(data).unwrap()
                });
            },
        );
        
        // Benchmark transform only (after fitting)
        let mut fitted_scaler = DataPreprocessor::new(PreprocessorConfig {
            scaling_strategy: ScalingStrategy::Standard,
            ..Default::default()
        });
        fitted_scaler.fit(&data).unwrap();
        
        group.bench_with_input(
            BenchmarkId::new("transform_only", size),
            &data,
            |b, data| {
                b.iter(|| fitted_scaler.transform(data).unwrap());
            },
        );
    }
    
    group.finish();
}

fn benchmark_categorical_encoding(c: &mut Criterion) {
    let mut group = c.benchmark_group("CategoricalEncoder");
    
    for cardinality in [10, 100, 1000].iter() {
        let categories: Vec<String> = (0..100_000)
            .map(|i| format!("cat_{}", i % cardinality))
            .collect();
        
        group.bench_with_input(
            BenchmarkId::new("label_encoding", cardinality),
            &categories,
            |b, cats| {
                b.iter(|| {
                    let mut encoder = LabelEncoder::new();
                    encoder.fit(cats).unwrap();
                    encoder.transform(cats).unwrap()
                });
            },
        );
        
        group.bench_with_input(
            BenchmarkId::new("onehot_encoding", cardinality),
            &categories,
            |b, cats| {
                b.iter(|| {
                    let mut encoder = OneHotEncoder::default();
                    encoder.fit(cats).unwrap();
                    encoder.transform(cats).unwrap()
                });
            },
        );
    }
    
    group.finish();
}

fn benchmark_inference(c: &mut Criterion) {
    let mut group = c.benchmark_group("Inference");
    group.sample_size(100);
    
    // Setup: train a model
    let train_data = generate_data(10_000, 50);
    let train_labels: Array1<i64> = Array1::from_iter(
        train_data.column(0).iter().map(|&x| if x > 0.0 { 1 } else { 0 })
    );
    
    let mut engine = TrainEngine::default();
    let trained = tokio::runtime::Runtime::new().unwrap().block_on(async {
        engine.train(TrainData { X: train_data, y: train_labels }).await.unwrap()
    });
    
    // Benchmark single prediction
    let single_input = generate_data(1, 50);
    group.bench_function("single_prediction", |b| {
        b.iter(|| trained.best_model.predict(&single_input).unwrap());
    });
    
    // Benchmark batch predictions
    for batch_size in [10, 100, 1000, 10000].iter() {
        let batch_input = generate_data(*batch_size, 50);
        group.throughput(Throughput::Elements(*batch_size as u64));
        
        group.bench_with_input(
            BenchmarkId::new("batch_prediction", batch_size),
            &batch_input,
            |b, input| {
                b.iter(|| trained.best_model.predict(input).unwrap());
            },
        );
    }
    
    group.finish();
}

criterion_group!(
    benches,
    benchmark_standard_scaler,
    benchmark_categorical_encoding,
    benchmark_inference,
);
criterion_main!(benches);
```

### 6.5 Python Integration Tests

Located in `tests/python/`

```python
# tests/python/test_rust_backend.py

import pytest
import numpy as np
import pandas as pd
from numpy.testing import assert_array_almost_equal, assert_array_equal

# Import both implementations for comparison
import automl_rust
from modules.engine.data_preprocessor import DataPreprocessor as PythonPreprocessor


class TestDataPreprocessorParity:
    """Test that Rust implementation matches Python behavior."""
    
    @pytest.fixture
    def sample_data(self):
        np.random.seed(42)
        return np.random.randn(1000, 10)
    
    @pytest.fixture
    def sample_df(self):
        np.random.seed(42)
        return pd.DataFrame({
            'numeric1': np.random.randn(1000),
            'numeric2': np.random.randn(1000) * 10 + 5,
            'category': np.random.choice(['A', 'B', 'C'], 1000),
        })
    
    def test_standard_scaler_parity(self, sample_data):
        """Rust StandardScaler should match Python sklearn."""
        # Python
        py_preprocessor = PythonPreprocessor(scaling='standard')
        py_result = py_preprocessor.fit_transform(sample_data)
        
        # Rust
        rust_preprocessor = automl_rust.DataPreprocessor(scaling='standard')
        rust_result = rust_preprocessor.fit_transform(sample_data)
        
        assert_array_almost_equal(rust_result, py_result, decimal=10)
    
    def test_minmax_scaler_parity(self, sample_data):
        """Rust MinMaxScaler should match Python sklearn."""
        py_preprocessor = PythonPreprocessor(scaling='minmax')
        py_result = py_preprocessor.fit_transform(sample_data)
        
        rust_preprocessor = automl_rust.DataPreprocessor(scaling='minmax')
        rust_result = rust_preprocessor.fit_transform(sample_data)
        
        assert_array_almost_equal(rust_result, py_result, decimal=10)
    
    def test_label_encoding_parity(self, sample_df):
        """Rust LabelEncoder should match Python sklearn."""
        categories = sample_df['category'].values
        
        py_preprocessor = PythonPreprocessor(categorical_encoding='label')
        py_result = py_preprocessor.fit_transform_categorical(categories)
        
        rust_preprocessor = automl_rust.DataPreprocessor(categorical_encoding='label')
        rust_result = rust_preprocessor.fit_transform_categorical(categories)
        
        assert_array_equal(rust_result, py_result)
    
    def test_inverse_transform_roundtrip(self, sample_data):
        """Inverse transform should recover original data."""
        rust_preprocessor = automl_rust.DataPreprocessor(scaling='standard')
        transformed = rust_preprocessor.fit_transform(sample_data)
        recovered = rust_preprocessor.inverse_transform(transformed)
        
        assert_array_almost_equal(recovered, sample_data, decimal=10)
    
    def test_handles_nan_values(self):
        """Should handle NaN values correctly."""
        data = np.array([[1.0, np.nan], [3.0, 4.0], [np.nan, 6.0]])
        
        rust_preprocessor = automl_rust.DataPreprocessor(
            scaling='standard',
            missing_strategy='mean'
        )
        result = rust_preprocessor.fit_transform(data)
        
        # Should have no NaN values after imputation
        assert not np.isnan(result).any()


class TestTrainingEngineParity:
    """Test that Rust training matches Python behavior."""
    
    @pytest.fixture
    def classification_data(self):
        np.random.seed(42)
        X = np.random.randn(1000, 10)
        y = (X[:, 0] > 0).astype(int)
        return X, y
    
    def test_cross_validation_scores(self, classification_data):
        """CV scores should be similar between implementations."""
        X, y = classification_data
        
        # Python
        py_engine = TrainEngine(cv_folds=5, metric='accuracy')
        py_result = py_engine.cross_validate(X, y)
        
        # Rust
        rust_engine = automl_rust.TrainEngine(cv_folds=5, metric='accuracy')
        rust_result = rust_engine.cross_validate(X, y)
        
        # Scores should be within 1% (allowing for numerical differences)
        assert abs(py_result['mean_score'] - rust_result['mean_score']) < 0.01
    
    @pytest.mark.asyncio
    async def test_async_training(self, classification_data):
        """Async training should work correctly."""
        X, y = classification_data
        
        rust_engine = automl_rust.TrainEngine()
        result = await rust_engine.train_async(X, y)
        
        assert result.best_model is not None
        assert result.cv_results.mean_score > 0.5


class TestPerformanceImprovement:
    """Verify Rust implementation is faster."""
    
    @pytest.fixture
    def large_data(self):
        np.random.seed(42)
        return np.random.randn(100_000, 50)
    
    def test_preprocessing_speedup(self, large_data, benchmark):
        """Rust preprocessing should be faster than Python."""
        rust_preprocessor = automl_rust.DataPreprocessor(scaling='standard')
        
        # Benchmark Rust
        rust_time = benchmark(lambda: rust_preprocessor.fit_transform(large_data.copy()))
        
        # Rust should be at least 5x faster
        # (actual comparison done in CI with both implementations)
        assert rust_time < 1.0  # Should complete in under 1 second
    
    def test_inference_latency(self, benchmark):
        """Single inference should have low latency."""
        # Setup: create trained model
        np.random.seed(42)
        X_train = np.random.randn(10000, 20)
        y_train = (X_train[:, 0] > 0).astype(int)
        
        engine = automl_rust.TrainEngine()
        model = engine.train(X_train, y_train).best_model
        
        # Benchmark single prediction
        single_input = np.random.randn(1, 20)
        
        def predict():
            return model.predict(single_input)
        
        result = benchmark(predict)
        
        # Single prediction should be under 1ms
        assert benchmark.stats.mean < 0.001


class TestErrorHandling:
    """Test error handling matches expected behavior."""
    
    def test_raises_on_unfitted_transform(self):
        """Should raise error when transforming unfitted preprocessor."""
        preprocessor = automl_rust.DataPreprocessor()
        data = np.random.randn(10, 5)
        
        with pytest.raises(RuntimeError, match="not fitted"):
            preprocessor.transform(data)
    
    def test_raises_on_shape_mismatch(self):
        """Should raise error on feature count mismatch."""
        preprocessor = automl_rust.DataPreprocessor()
        train_data = np.random.randn(100, 10)
        test_data = np.random.randn(10, 5)  # Different column count
        
        preprocessor.fit(train_data)
        
        with pytest.raises(ValueError, match="feature"):
            preprocessor.transform(test_data)
    
    def test_handles_empty_input(self):
        """Should handle empty arrays gracefully."""
        preprocessor = automl_rust.DataPreprocessor()
        empty_data = np.array([]).reshape(0, 5)
        
        with pytest.raises(ValueError, match="empty"):
            preprocessor.fit(empty_data)
```

### 6.6 Fuzz Testing

```rust
// tests/rust/fuzz/fuzz_targets/preprocessing.rs

#![no_main]
use libfuzzer_sys::fuzz_target;
use automl_core::preprocessing::{DataPreprocessor, PreprocessorConfig};
use ndarray::Array2;

fuzz_target!(|data: (Vec<f64>, usize, usize)| {
    let (values, rows, cols) = data;
    
    // Skip invalid dimensions
    if rows == 0 || cols == 0 || values.len() < rows * cols {
        return;
    }
    
    // Create array from fuzzed data
    let array = match Array2::from_shape_vec((rows.min(1000), cols.min(100)), 
        values.into_iter().take(rows * cols).collect()) {
        Ok(a) => a,
        Err(_) => return,
    };
    
    // Try various preprocessing configurations
    for scaling in [ScalingStrategy::Standard, ScalingStrategy::MinMax, ScalingStrategy::Robust] {
        let mut preprocessor = DataPreprocessor::new(PreprocessorConfig {
            scaling_strategy: scaling,
            ..Default::default()
        });
        
        // Should not panic
        let _ = preprocessor.fit_transform(&array);
    }
});
```

### 6.7 Test Coverage Requirements

| Module | Minimum Coverage | Target Coverage |
|--------|------------------|-----------------|
| `preprocessing/` | 90% | 95% |
| `training/` | 85% | 90% |
| `inference/` | 90% | 95% |
| `optimizer/` | 85% | 90% |
| `cache/` | 80% | 85% |
| `utils/` | 80% | 85% |
| **Overall** | **85%** | **90%** |

### 6.8 Test Commands

```bash
# Run all Rust tests
cargo test --workspace

# Run tests with output
cargo test --workspace -- --nocapture

# Run specific test module
cargo test --package automl-core preprocessing::

# Run property tests with more cases
PROPTEST_CASES=10000 cargo test --workspace

# Run benchmarks
cargo bench --workspace

# Generate coverage report
cargo tarpaulin --workspace --out Html

# Run Python integration tests
pytest tests/python/ -v

# Run fuzz tests (requires nightly)
cargo +nightly fuzz run preprocessing -- -max_total_time=300
```

---

## Performance Benchmarks

### Benchmark Targets

| Operation | Python Baseline | Rust Target | Improvement |
|-----------|-----------------|-------------|-------------|
| StandardScaler (1M rows × 10 cols) | ~500ms | <50ms | 10x |
| MinMaxScaler (1M rows × 10 cols) | ~400ms | <40ms | 10x |
| LabelEncoder (1M values, 1K categories) | ~200ms | <20ms | 10x |
| OneHotEncoder (1M values, 100 categories) | ~800ms | <80ms | 10x |
| Single Inference | ~5ms | <0.1ms | 50x |
| Batch Inference (10K rows) | ~50ms | <5ms | 10x |
| Cross-validation (5-fold, 100K rows) | ~30s | <5s | 6x |
| HPO Trial (single) | ~100ms | <10ms | 10x |

### Memory Targets

| Operation | Python Baseline | Rust Target | Reduction |
|-----------|-----------------|-------------|-----------|
| Load 1M row DataFrame | ~800MB | <400MB | 50% |
| Preprocessing Pipeline | ~1.2GB peak | <600MB peak | 50% |
| Model in Memory | ~100MB | <50MB | 50% |
| Inference (10K batch) | ~200MB | <50MB | 75% |

---

## CI/CD Pipeline

### GitHub Actions Workflow

```yaml
# .github/workflows/rust.yml
name: Rust CI

on:
  push:
    branches: [rust-dev, main]
    paths:
      - 'rust/**'
      - 'Cargo.toml'
      - 'Cargo.lock'
  pull_request:
    branches: [rust-dev, main]
    paths:
      - 'rust/**'

env:
  CARGO_TERM_COLOR: always
  RUST_BACKTRACE: 1

jobs:
  check:
    name: Check
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: dtolnay/rust-toolchain@stable
      - uses: Swatinem/rust-cache@v2
      - run: cargo check --workspace --all-features

  fmt:
    name: Rustfmt
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: dtolnay/rust-toolchain@stable
        with:
          components: rustfmt
      - run: cargo fmt --all -- --check

  clippy:
    name: Clippy
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: dtolnay/rust-toolchain@stable
        with:
          components: clippy
      - uses: Swatinem/rust-cache@v2
      - run: cargo clippy --workspace --all-features -- -D warnings

  test:
    name: Test Suite
    runs-on: ${{ matrix.os }}
    strategy:
      matrix:
        os: [ubuntu-latest, macos-latest, windows-latest]
        rust: [stable, beta]
    steps:
      - uses: actions/checkout@v4
      - uses: dtolnay/rust-toolchain@master
        with:
          toolchain: ${{ matrix.rust }}
      - uses: Swatinem/rust-cache@v2
      - run: cargo test --workspace --all-features
      - run: cargo test --workspace --all-features -- --ignored
        if: matrix.rust == 'stable'

  coverage:
    name: Code Coverage
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: dtolnay/rust-toolchain@stable
      - uses: Swatinem/rust-cache@v2
      - uses: taiki-e/install-action@cargo-tarpaulin
      - run: cargo tarpaulin --workspace --out Xml
      - uses: codecov/codecov-action@v3
        with:
          files: cobertura.xml
          fail_ci_if_error: true

  benchmark:
    name: Benchmarks
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: dtolnay/rust-toolchain@stable
      - uses: Swatinem/rust-cache@v2
      - run: cargo bench --workspace -- --noplot
      - uses: benchmark-action/github-action-benchmark@v1
        with:
          tool: 'cargo'
          output-file-path: target/criterion/*/benchmark.json
          github-token: ${{ secrets.GITHUB_TOKEN }}
          auto-push: true
          alert-threshold: '150%'
          comment-on-alert: true

  build-wheels:
    name: Build Python Wheels
    runs-on: ${{ matrix.os }}
    strategy:
      matrix:
        os: [ubuntu-latest, macos-latest, windows-latest]
        python: ['3.9', '3.10', '3.11', '3.12']
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: ${{ matrix.python }}
      - uses: dtolnay/rust-toolchain@stable
      - uses: Swatinem/rust-cache@v2
      - run: pip install maturin
      - run: maturin build --release --strip
      - uses: actions/upload-artifact@v4
        with:
          name: wheel-${{ matrix.os }}-py${{ matrix.python }}
          path: target/wheels/*.whl

  integration-test:
    name: Python Integration Tests
    needs: build-wheels
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: '3.11'
      - uses: actions/download-artifact@v4
        with:
          name: wheel-ubuntu-latest-py3.11
          path: dist/
      - run: pip install dist/*.whl pytest pytest-asyncio numpy pandas
      - run: pytest tests/python/ -v
```

---

## Dependencies

### Rust Dependencies (Cargo.toml)

```toml
[workspace.dependencies]
# ═══════════════════════════════════════════════════════════════════════════════
# CORE DEPENDENCIES
# ═══════════════════════════════════════════════════════════════════════════════

# N-dimensional arrays - core data structure
ndarray = { version = "0.15", features = ["rayon", "serde"] }
ndarray-stats = "0.5"  # Statistical functions for ndarray

# DataFrame operations - Polars is our primary choice for speed
polars = { version = "0.35", features = [
    "lazy",           # Lazy evaluation
    "parquet",        # Parquet I/O
    "json",           # JSON I/O  
    "csv",            # CSV I/O
    "ipc",            # Arrow IPC
    "dtype-full",     # All data types
    "streaming",      # Streaming support
    "performant",     # Performance optimizations
] }

# Apache Arrow for interop
arrow = { version = "50.0", features = ["ffi"] }
parquet = "50.0"

# ═══════════════════════════════════════════════════════════════════════════════
# ASYNC & PARALLELISM
# ═══════════════════════════════════════════════════════════════════════════════

# Async runtime
tokio = { version = "1.35", features = [
    "rt-multi-thread",
    "sync",
    "time",
    "macros",
    "fs",
] }

# Data parallelism
rayon = "1.8"

# Async channels
crossbeam = { version = "0.8", features = ["crossbeam-channel"] }

# ═══════════════════════════════════════════════════════════════════════════════
# SERIALIZATION
# ═══════════════════════════════════════════════════════════════════════════════

serde = { version = "1.0", features = ["derive"] }
serde_json = "1.0"
serde_yaml = "0.9"
toml = "0.8"
bincode = "1.3"  # Fast binary serialization

# ═══════════════════════════════════════════════════════════════════════════════
# ERROR HANDLING & LOGGING
# ═══════════════════════════════════════════════════════════════════════════════

thiserror = "1.0"
anyhow = "1.0"

# Structured logging
tracing = "0.1"
tracing-subscriber = { version = "0.3", features = ["env-filter", "json"] }

# ═══════════════════════════════════════════════════════════════════════════════
# MACHINE LEARNING
# ═══════════════════════════════════════════════════════════════════════════════

# XGBoost bindings
xgboost = "0.1"

# LightGBM bindings  
lightgbm = "0.2"

# Pure Rust ML (backup/simple cases)
linfa = "0.7"
linfa-linear = "0.7"
linfa-logistic = "0.7"
linfa-trees = "0.7"
smartcore = "0.3"

# ═══════════════════════════════════════════════════════════════════════════════
# NUMERICAL & STATISTICAL
# ═══════════════════════════════════════════════════════════════════════════════

# Random number generation
rand = "0.8"
rand_distr = "0.4"
rand_chacha = "0.3"

# Statistics
statrs = "0.16"

# Special functions
special = "0.10"

# Linear algebra (optional, for advanced ops)
nalgebra = { version = "0.32", optional = true }

# ═══════════════════════════════════════════════════════════════════════════════
# PYTHON BINDINGS
# ═══════════════════════════════════════════════════════════════════════════════

pyo3 = { version = "0.20", features = [
    "extension-module",
    "abi3-py39",        # Support Python 3.9+
    "multiple-pymethods",
] }
numpy = "0.20"  # NumPy interop

# ═══════════════════════════════════════════════════════════════════════════════
# PERFORMANCE
# ═══════════════════════════════════════════════════════════════════════════════

# Fast memory allocator
mimalloc = { version = "0.1", features = ["local_dynamic_tls"] }

# Memory mapping
memmap2 = "0.9"

# Compile-time string interning
string-interner = "0.14"

# ═══════════════════════════════════════════════════════════════════════════════
# UTILITIES
# ═══════════════════════════════════════════════════════════════════════════════

# Date/time
chrono = { version = "0.4", features = ["serde"] }

# Hash maps
hashbrown = "0.14"
indexmap = { version = "2.1", features = ["serde"] }

# UUID generation
uuid = { version = "1.6", features = ["v4", "serde"] }

# Path handling
camino = "1.1"

# ═══════════════════════════════════════════════════════════════════════════════
# TESTING (dev-dependencies)
# ═══════════════════════════════════════════════════════════════════════════════

[workspace.dev-dependencies]
# Property-based testing
proptest = "1.4"

# Benchmarking
criterion = { version = "0.5", features = ["html_reports"] }

# Approximate floating point comparison
approx = "0.5"

# Test utilities
rstest = "0.18"
test-case = "3.3"

# Async testing
tokio-test = "0.4"

# Snapshot testing
insta = { version = "1.34", features = ["yaml"] }
```

---

## Risk Mitigation

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Breaking API changes | Medium | High | Maintain Python facade, semantic versioning, deprecation warnings |
| Performance regression | Low | High | Benchmark at each PR, automated regression detection |
| Missing ML algorithm support | Medium | Medium | Keep Python fallback for rare algorithms, prioritize common ones |
| Build complexity | High | Medium | Pre-built wheels, Docker images, detailed build docs |
| Team learning curve | High | Medium | Documentation, pair programming, Rust book club |
| Memory safety bugs | Low | High | Extensive testing, miri for unsafe code, fuzzing |
| Cross-platform issues | Medium | Medium | CI on all platforms, platform-specific testing |
| Python/Rust interop bugs | Medium | High | Comprehensive integration tests, type stubs |

---

## Success Metrics

### Performance Metrics
- [ ] 5x+ improvement in data preprocessing throughput
- [ ] 10x+ improvement in single inference latency
- [ ] 3x+ improvement in batch inference throughput
- [ ] 50%+ reduction in peak memory usage
- [ ] <1ms p99 latency for single predictions

### Quality Metrics
- [ ] 100% Python API compatibility (drop-in replacement)
- [ ] All existing Python tests passing
- [ ] 90%+ Rust code coverage
- [ ] Zero `unsafe` blocks in public API
- [ ] All clippy warnings addressed

### Operational Metrics
- [ ] Binary wheel size under 50MB
- [ ] Build time under 10 minutes
- [ ] Documentation coverage 100%
- [ ] CI pipeline green on all platforms

---

## Notes & Future Considerations

### Keep in Python
- `app.py` (Gradio UI) - No performance benefit, complex Python ecosystem
- FastAPI endpoints - Thin layer, Python async works well
- Configuration parsing - Low impact, complex validation logic

### Future Enhancements
- **WASM compilation** - Enable browser-based inference
- **GPU support** - CUDA/ROCm bindings for training acceleration
- **Distributed training** - Integration with Ray or custom solution
- **Model quantization** - INT4/INT8 for edge deployment
- **ONNX export** - Cross-platform model deployment

### Migration Timeline Estimate
- Phase 1-2 (Setup & Data Structures): 2-3 weeks
- Phase 3 (Data Processing): 3-4 weeks  
- Phase 4 (Training Engine): 4-6 weeks
- Phase 5 (Inference Engine): 2-3 weeks
- Phase 6 (Optimizer): 3-4 weeks
- Phase 7 (Performance Utils): 2 weeks
- Phase 8 (Python Bindings): 2-3 weeks
- Phase 9-10 (Integration & Docs): 2-3 weeks

**Total: ~20-28 weeks for full migration**
