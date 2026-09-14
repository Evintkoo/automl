# 📚 AutoML Documentation

Welcome to the comprehensive documentation for **AutoML** — a high-performance automated machine learning framework written in pure Rust.

## 🎯 Quick Navigation

### 👋 New to AutoML?
- [📖 **Getting Started**](getting-started/) — Installation, setup, and first steps
- [🎓 **User Guides**](user-guides/) — Step-by-step tutorials for common tasks
- [💻 **Web Interface Guide**](user-guides/web-interface.md) — Using the built-in web UI

### 🚀 Deploy to Production
- [🐳 **Docker Deployment**](deployment/docker.md) — Containerized deployment

### 🔧 API Integration
- [📋 **API Reference**](api-reference/) — Complete REST API documentation

### 👩‍💻 Development
- [🛠️ **Development Setup**](development/) — Developer environment and contributing

### 📦 Module Documentation
- [🧩 **Module Docs**](modules/) — Detailed API docs for each module

---

## 🚀 What is AutoML?

AutoML is a comprehensive, production-ready machine learning platform written in pure Rust that provides:

- **Automated model training** with 8+ algorithm families
- **Hyperparameter optimization** (Bayesian, TPE, ASHT)
- **Data preprocessing** with adaptive strategy selection
- **High-performance inference** with batch processing and caching
- **Web server** with REST API and interactive UI
- **CLI** for scripting and automation

## Quick Start

### Installation
```bash
# Clone the repository
git clone https://github.com/Evintkoo/automl.git
cd automl

# Build
cargo build --release
```

### Basic Usage

#### Command Line Interface
```bash
# Interactive mode
cargo run --release

# Train a model
cargo run --release -- train --data data.csv --target label --model random_forest

# Start web server
cargo run --release -- serve --port 8080

# Dataset info
cargo run --release -- info --data data.csv

# Benchmark models
cargo run --release -- benchmark --data data.csv --target label
```

#### Rust API
```rust
use automl::prelude::*;
use polars::prelude::*;

// Load data
let df = CsvReadOptions::default()
    .try_into_reader_with_file_path(Some("data.csv".into()))?
    .finish()?;

// Configure and train
let config = TrainingConfig::new(TaskType::Regression, "target")
    .with_model(ModelType::RandomForest)
    .with_n_estimators(100)
    .with_max_depth(10);

let mut engine = TrainEngine::new(config);
engine.fit(&df)?;

// Predict
let predictions = engine.predict(&df)?;
```

## Architecture Overview

```
automl/
├── src/
│   ├── lib.rs                  # Library root with prelude
│   ├── main.rs                 # CLI entry point
│   ├── error.rs                # Error types
│   │
│   ├── preprocessing/          # Data scaling, encoding, imputation
│   ├── training/               # Model training (8+ algorithms)
│   ├── inference/              # High-performance inference
│   ├── optimizer/              # Hyperparameter optimization
│   │
│   ├── explainability/         # Feature importance, PDP
│   ├── ensemble/               # Voting, stacking, blending
│   ├── calibration/            # Probability calibration
│   ├── anomaly/                # Isolation Forest, LOF
│   ├── drift/                  # Data & concept drift detection
│   │
│   ├── feature_engineering/    # Polynomial, interactions, TF-IDF
│   ├── imputation/             # MICE, KNN, iterative
│   ├── synthetic/              # SMOTE, ADASYN
│   ├── timeseries/             # Time series features & CV
│   │
│   ├── batch/                  # Priority-based batch processing
│   ├── cache/                  # Multi-level LRU+TTL caching
│   ├── memory/                 # Memory pooling & monitoring
│   ├── streaming/              # Backpressure-controlled streaming
│   ├── quantization/           # INT8/FP16 quantization
│   ├── monitoring/             # Latency, throughput, alerts
│   ├── tracking/               # Experiment tracking
│   │
│   ├── device/                 # Hardware detection & auto-config
│   ├── adaptive/               # Adaptive preprocessing & hyperopt
│   ├── precision/              # Mixed precision (FP16/BF16)
│   ├── security/               # Auth, rate limiting, TLS
│   │
│   ├── nas/                    # Neural architecture search (DARTS)
│   ├── autopipeline/           # Automatic pipeline construction
│   ├── architectures/          # TabNet, FT-Transformer
│   ├── export/                 # ONNX, PMML serialization
│   ├── utils/                  # SIMD, parallel, metrics
│   │
│   ├── server/                 # Axum HTTP server + REST API
│   └── cli/                    # CLI with clap
│
├── examples/                   # Usage examples
├── tests/                      # Integration tests
├── benches/                    # Benchmarks (criterion)
├── automl-web/                # Web UI (htmx + Alpine.js)
└── docs/                       # Documentation
```

## Development

```bash
# Run tests
cargo test

# Run benchmarks
cargo bench

# Build release
cargo build --release

# Generate API docs
cargo doc --open
```

## Version Information

**Current Version:** v0.5.0

See [CHANGELOG.md](../CHANGELOG.md) for release history.

## License

MIT License — see [LICENSE](../LICENSE) for details.
