# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.5.0] - 2025-02-06

### Added

#### Single-Crate Architecture Migration
- Consolidated workspace (automl-core, automl-cli, automl-python, automl-server) into a unified single crate
- All modules now under `src/` with `automl` as the single package name

#### Performance & Infrastructure Modules
- **Batch Processing** (`src/batch/`) — Priority-based async batch processing with adaptive sizing
- **Multi-Level Cache** (`src/cache/`) — L1/L2/L3 cache with LRU+TTL eviction
- **Memory Management** (`src/memory/`) — Memory pooling, system memory monitoring, pressure detection
- **Streaming** (`src/streaming/`) — Streaming pipelines with backpressure control
- **Quantization** (`src/quantization/`) — INT8/UINT8/INT16/FP16 quantization with calibration
- **Monitoring** (`src/monitoring/`) — Performance metrics, latency histograms (p50/p95/p99), alert system
- **Experiment Tracking** (`src/tracking/`) — MLflow-like local experiment tracking with metrics history

#### Device & Adaptive Optimization
- **Device Optimizer** (`src/device/`) — CPU SIMD feature detection (AVX2, AVX512, NEON, FMA), NUMA topology, environment detection (Docker/VM/cloud), auto-tuned configs
- **Adaptive Preprocessing** (`src/adaptive/`) — Dataset profiling, automatic strategy selection based on data characteristics
- **Adaptive Hyperopt** (`src/adaptive/`) — Search space narrowing over time, convergence detection, promising region identification

#### Mixed Precision & Security
- **Mixed Precision** (`src/precision/`) — FP16/BF16 support detection, dynamic loss scaling, precision auto-selection
- **Security** (`src/security/`) — API key/JWT authentication, rate limiting, TLS management, secrets manager, security middleware

#### API Improvements
- Expanded prelude with all major types (device, adaptive, precision, security, NAS, architectures)
- Comprehensive doc comments on all modules

### Changed
- Renamed crate from `automl_core` to `automl`
- Updated all examples to use new crate name and correct APIs
- Fixed test data to avoid singular matrix issues in training/inference tests

### Removed
- Workspace structure (automl-core, automl-cli, automl-python, automl-server crates)
- Python bindings (PyO3) — pure Rust only

## [0.4.0] - 2024-02-03

### Added

#### Training Engine Improvements

- **Complete Model Integration**
  - TrainEngine now uses real model implementations instead of placeholders
  - Added `TrainedModel` enum to hold all model variants polymorphically
  - Full dispatch for all model types in `train_model()` and `predict_internal()`
  - Added `feature_importances()` method to TrainEngine

- **New Models**
  - Neural Network (MLP) for both classification and regression
  - Naive Bayes classifier (Gaussian and Multinomial variants)
  - Both models fully integrated into TrainEngine

- **Outlier Detection Module** (preprocessing/outlier.rs)
  - Detection methods: IQR, Z-Score, Percentile, Modified Z-Score
  - Handling strategies: Clip, ToNaN, Remove, ReplaceWithMean/Median
  - Scikit-learn compatible fit/transform interface

- **Inference Engine Enhancement**
  - Added `predict_array()` for direct ndarray input
  - Automatic DataFrame conversion using model's feature names

- **Examples**
  - `basic_training.rs` - Simple model training example
  - `preprocessing.rs` - Data preprocessing pipeline
  - `model_comparison.rs` - Comparing multiple models
  - `hyperopt.rs` - Hyperparameter optimization example

### Changed

- **Training Configuration Updates**
  - `n_estimators` and `learning_rate` now `Option<>` for flexibility
  - Added `max_iter`, `n_neighbors`, `random_seed` fields
  - New builder methods for all parameters

- **CLI Improvements**
  - Benchmark command now tests all available models
  - Added Gradient Boosting, KNN, Naive Bayes, Neural Network to benchmark

- **Python Bindings**
  - Added `NaiveBayes` to `PyModelType` enum
  - All new model types accessible from Python

### Fixed

- Flattened project structure (moved rust/ contents to root)
- Updated all documentation paths accordingly
- Fixed .gitignore for new project structure

## [0.3.0] - 2024-02-03

### Added

#### Pure Rust Implementation

This release completes the migration to pure Rust, removing all Python dependencies.

- **Web Server (automl-server)**
  - Axum 0.7 based HTTP server
  - REST API for data, training, and inference
  - Static file serving for web UI
  - CORS and compression middleware
  - Real-time system monitoring endpoints

- **Web UI (automl-web)**
  - htmx + Alpine.js reactive frontend
  - TailwindCSS styling
  - Tab-based navigation: Data, Config, Training, Monitor
  - Sample dataset loading (iris, diabetes, boston, wine)
  - Training progress visualization
  - System status monitoring

- **CLI Application (automl-cli)**
  - `train` - Train models on data files
  - `predict` - Make predictions using trained models
  - `preprocess` - Data preprocessing pipeline
  - `benchmark` - Compare multiple models
  - `info` - Show dataset information
  - `serve` - Start the web server

### Changed

- **Removed Python dependencies**
  - Deleted `python/` directory
  - Deleted `legacy/` directory
  - Deleted `pyproject.toml`
  - Updated README for pure Rust usage

### Technical Details

- Rust Edition 2021
- Axum 0.7 for web server
- Polars 0.46 for DataFrames
- sysinfo 0.30 for system monitoring
- tokio 1.x async runtime

## [0.2.0] - 2024-02-03

### Added

#### Rust Core Library (automl-core)

- **Preprocessing Engine**
  - `StandardScaler`, `MinMaxScaler`, `RobustScaler` for feature scaling
  - `OneHotEncoder`, `LabelEncoder`, `TargetEncoder` for categorical encoding
  - `MeanImputer`, `MedianImputer`, `ModeImputer` for missing values
  - `DataPreprocessor` pipeline for composable transformations

- **Training Engine**
  - K-Fold and Stratified K-Fold cross-validation
  - Metrics: R², MSE, RMSE, MAE, Accuracy, Precision, Recall, F1
  - Model registry with factory pattern

- **Models**
  - Linear Regression with L2 regularization (Ridge)
  - Logistic Regression
  - Decision Tree (classifier and regressor)
  - Random Forest (classifier and regressor)

- **Inference Engine**
  - Single and batch prediction
  - Quantization type definitions

- **Optimizer (HyperOptX)**
  - Search space with Float, Int, Categorical parameters
  - TPE (Tree-structured Parzen Estimator) sampler
  - Random and Grid samplers
  - Median, Percentile, and Hyperband pruners
  - ASHT (Adaptive Successive Halving)

- **Data I/O**
  - CSV loading with type inference
  - Parquet loading
  - Chunked/streaming for large files
  - File info inspection

- **Performance Utilities**
  - SIMD operations (sum, mean, variance, dot product, softmax)
  - Memory pool with allocation tracking
  - LRU cache with TTL support
  - Arena allocator (bumpalo)
  - Parallel execution with rayon

#### Python Bindings (automl-python)

- PyO3 bindings for all core modules
- NumPy array interop
- Pandas DataFrame ↔ Polars conversion
- Error handling with Python exceptions

### Technical Details

- Rust Edition 2021
- MSRV: 1.75.0
- Polars 0.46
- PyO3 0.20 with abi3-py39

## [0.1.0] - Initial Release

- Original Python implementation
- FastAPI REST API
- Gradio UI
- scikit-learn, XGBoost, LightGBM models
- Optuna hyperparameter optimization
