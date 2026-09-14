# AutoML

[![Rust](https://img.shields.io/badge/rust-1.75%2B-orange.svg)](https://www.rust-lang.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

🚀 **Pure Rust** high-performance AutoML framework with web UI and CLI.

## Quick Start

### Using the CLI

```bash
# Build the CLI
cargo build --release

# Train a model
./target/release/automl train --data data.csv --target label --model random_forest

# Benchmark multiple models
./target/release/automl benchmark --data data.csv --target label

# Show data info
./target/release/automl info --data data.csv
```

### Using the Web Server

```bash
# Build the server (same binary as the CLI)
cargo build --release

# Start the server
./target/release/automl serve --port 8080
# or just: ./target/release/automl   (defaults to serving on 0.0.0.0:8080)

# Open http://localhost:8080 in your browser
```

The web UI provides:
- 📊 Data upload and sample datasets (iris, diabetes, boston, wine)
- ⚙️ Configuration for task type, model, and preprocessing
- 🚀 One-click model training with progress tracking
- 📡 Real-time system monitoring

### API Examples

```bash
# Load sample dataset
curl http://localhost:8080/api/data/sample/iris

# Start training
curl -X POST http://localhost:8080/api/train \
  -H "Content-Type: application/json" \
  -d '{"target_column":"species","task_type":"classification","model_type":"random_forest"}'

# Check training status
curl http://localhost:8080/api/train/status/{job_id}

# Get system status
curl http://localhost:8080/api/system/status
```

## Features

| Feature | Description | Status |
|---------|-------------|--------|
| **Web Server** | Axum-based REST API | ✅ |
| **Web UI** | htmx + Alpine.js frontend | ✅ |
| **CLI** | Full-featured command line | ✅ |
| **Data Preprocessing** | Scalers, Encoders, Imputers | ✅ |
| **Training Engine** | Cross-validation, Metrics | ✅ |
| **Models** | Linear, Logistic, Trees, Random Forest | ✅ |
| **HyperOptX** | TPE, Random, Grid samplers | ✅ |
| **Pruners** | Median, Percentile, Hyperband | ✅ |
| **SIMD Operations** | Vectorized math operations | ✅ |

## Performance

Benchmarked against a Python/scikit-learn baseline on identical hardware (Railway deployment). Rust wins **18/18** head-to-head comparisons.

| Metric | Rust | Python | Speedup |
|--------|------|--------|---------|
| Page Load (TTFB) | 29 ms | 512 ms | **17x** |
| DOM Content Loaded | 46 ms | 3,676 ms | **80x** |
| Health Check API | 135 ms | 1,428 ms | **11x** |
| Iris Dataset Load | 113 ms | 26,808 ms | **237x** |
| Model Training (Iris) | 446 ms | 1,047 ms | **2.3x** |
| Single Prediction | 43 ms | 138 ms | **3.2x** |
| Tab Switch (UI) | 101 ms | 4,101 ms | **41x** |
| 10 Concurrent Requests | 29 ms avg | 16,531 ms avg | **564x** |

**Rust API aggregate** (57 endpoints): avg 60 ms, median 40 ms, 68% under 50 ms.

Full end-to-end pipeline (load + preprocess + train + predict + explain): **204 ms**.

| Operation | Typical Latency |
|-----------|-----------------|
| StandardScaler (1M×10) | ~45 ms |
| Random Forest fit (10K rows) | ~1.8 s |
| Random Forest fit (50K rows) | ~1.1 s |
| Batch Inference (10K) | ~4 ms |
| Full AutoML Pipeline | ~595 ms |
| Server startup | <1 s |

## Project Structure

```
automl/
├── src/                    # automl crate (lib + CLI/server binary)
│   ├── cli/                # clap CLI, incl. the `serve` subcommand
│   ├── server/              # Axum web server (mounted by `serve`)
│   └── ...                 # preprocessing, training, models, etc.
├── automl-web/             # Web frontend assets served by the server
├── benches/                 # Benchmarks
├── examples/                # Example code
├── docs/                    # Documentation
└── tests/                   # Integration tests
```

## Building from Source

### Prerequisites

- Rust 1.75+ ([rustup](https://rustup.rs))

### Build

```bash
# Build the crate (produces the `automl` binary, used as both CLI and server)
cargo build --release

# Run tests
cargo test

# Run the server
cargo run -- serve --port 8080

# Run the CLI
cargo run -- --help
```

You can also use the [Makefile](#makefile) for the common commands above.

### Build Binaries

```bash
# Optimized release build
cargo build --release

# Binary will be in:
# - target/release/automl   (run `automl serve` to start the web server)
```

## Makefile

Common tasks are wrapped in a `Makefile`:

```bash
make build      # cargo build --release
make server     # build (if needed) and run the server on :8080
make dev        # run the server via `cargo run` (debug build, faster iteration)
make cli        # show CLI help
make test       # cargo test
make bench      # cargo bench
make fmt        # cargo fmt
make lint       # cargo clippy
make clean      # cargo clean
```

Override the port/host with `PORT`/`HOST`, e.g. `make server PORT=9000`.

## API Reference

### REST Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/data/upload` | Upload CSV/JSON/Parquet |
| GET | `/api/data/sample/:name` | Load sample dataset |
| GET | `/api/data/preview` | Preview loaded data |
| GET | `/api/data/info` | Get dataset info |
| POST | `/api/train` | Start training job |
| GET | `/api/train/status/:id` | Get training status |
| POST | `/api/predict` | Make predictions |
| GET | `/api/system/status` | System metrics |

## CLI Reference

```bash
automl <COMMAND>

Commands:
  train       Train a model on data
  predict     Make predictions using a trained model
  preprocess  Preprocess data
  benchmark   Benchmark multiple models
  info        Show data information
  serve       Start the web server
  help        Print help
```

## Documentation

- [Changelog](CHANGELOG.md)

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

MIT License - see [LICENSE](LICENSE) for details.
