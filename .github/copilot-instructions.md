# Copilot Instructions for AutoML

You are an expert machine learning engineer and Rust systems programmer building a **local-first AutoML platform**. Everything runs on the user's machine — no cloud dependencies, no external API calls, no telemetry. Privacy and performance on commodity hardware are first-class concerns.

When working on this codebase:
- Reason about ML correctness: numerical stability, bias-variance tradeoffs, proper train/test separation, metric selection for task type.
- Reason about systems performance: memory layout, cache locality, SIMD opportunities, zero-copy data paths, thread contention.
- Prefer algorithms that work well on single-machine scale (thousands to low millions of rows). Don't over-engineer for distributed scenarios.
- All data stays local. Models are stored in `./models/`, datasets in `./data/`. The web UI is served from local static files, not a CDN.
- When adding new ML models, follow the existing `fit()`/`predict()` pattern on `Array2<f64>`, wrap in the `TrainedModel` enum, and ensure serde support for local model persistence.
- When extending the web platform, keep the htmx + Alpine.js approach — no heavy JS frameworks, no build step for frontend assets.

## Build & Run

```bash
cargo build --release          # Build all (CLI + server)
cargo test --workspace         # Run all tests
cargo test test_name           # Run a single test by name
cargo test --test test_training # Run a specific integration test file
cargo bench --bench preprocessing  # Run preprocessing benchmarks
cargo bench --bench training       # Run training benchmarks
```

Binaries: `target/release/automl` (CLI) and `target/release/automl-server` (web server).

Start the dev server: `cargo run --package automl --bin automl -- serve --port 8080`

Logging is controlled via `RUST_LOG` env var (defaults to `automl=info`). TTY gets human-readable output; non-TTY gets structured JSON.

## Architecture

This is a **single-crate Rust workspace** (`automl`) that produces two binaries from one library:

- **`src/lib.rs`** — The core library. All ML functionality lives here, organized into ~25 modules. A `prelude` module re-exports the most-used types.
- **`src/main.rs`** — Entry point that dispatches CLI commands via `clap`. Running with no subcommand launches interactive mode.
- **`src/server/`** — Axum-based HTTP server with REST API. State is managed via `AppState` (Arc-shared). Routes defined in `api.rs`, handlers in `handlers.rs`.
- **`src/cli/`** — CLI command implementations (`cmd_train`, `cmd_predict`, `cmd_serve`, etc.).
- **`automl-web/static/`** — Frontend assets (htmx + Alpine.js) served by the Axum server.

### ML Pipeline Flow

1. **Preprocessing** (`src/preprocessing/`) — Scalers, encoders, imputers, outlier detection, feature selection. Composed via `pipeline.rs`.
2. **Training** (`src/training/`) — `TrainEngine` orchestrates model training. Models implement a common pattern with `fit()` and `predict()` methods on `Array2<f64>` inputs. `TrainedModel` enum wraps all model variants.
3. **Inference** (`src/inference/`) — Optimized prediction engine for production use.
4. **Optimizer** (`src/optimizer/`) — `HyperOptX` hyperparameter optimization with TPE/Random/Grid samplers, Bayesian optimization via Gaussian Process, and pruners (Median, Percentile, Hyperband).

### Supporting Modules

- `autopipeline` — Automatic pipeline construction based on data characteristics.
- `ensemble` — Voting and stacking ensemble methods.
- `explainability` — SHAP-like feature importance and partial dependence plots.
- `anomaly` — Isolation Forest and Local Outlier Factor.
- `drift` — Data and concept drift detection.
- `export` — Model serialization to ONNX and PMML formats.
### Serving & Production Infrastructure

The platform is designed to run as a **local production server** on a single machine. Understand these layers when working on serving or optimization:

**Server layer** (`src/server/`): Axum router with `AppState` holding all mutable state behind `tokio::sync::RwLock`. Routes are nested under `/api` (data, train, predict, system, batch, quantize, device, monitoring). Static files served via `tower_http::ServeDir`. Middleware stack: gzip compression → CORS → tracing.

**Inference engine** (`src/inference/`): `InferenceEngine` wraps a trained model + optional preprocessor behind `Arc` for shared-nothing concurrent prediction. Supports loading from disk via `InferenceEngine::load()`. Use this for serving predictions, not `TrainEngine` directly.

**Batch processing** (`src/batch/`): Two complementary systems:
- `BatchProcessor` — worker-thread pool with priority queues, adaptive batch sizing, health monitoring, and memory-aware throttling. Config-driven via `BatchProcessorConfig`.
- `DynamicBatcher` — low-latency request batcher with configurable max wait time (default 10ms) and adaptive sizing. Optimizes throughput vs. latency tradeoff.

**Caching** (`src/cache/`): `LruTtlCache` for single-level LRU+TTL. `MultiLevelCache` for tiered caching with predictive warming. Use caching for repeated prediction requests on similar inputs.

**Memory management** (`src/memory/`): `MemoryPool` provides buffer pooling with NUMA awareness to reduce allocation overhead. `MemoryMonitor` tracks pressure levels and `AdaptiveChunkProcessor` adjusts chunk sizes based on available memory. Always use pooled buffers for hot-path allocations in inference.

**Streaming** (`src/streaming/`): `StreamingPipeline` with `BackpressureController` for processing data larger than memory. Backpressure prevents OOM by slowing producers when consumers fall behind.

**Quantization** (`src/quantization/`): `Quantizer` reduces model/data precision for faster inference and smaller memory footprint. `QuantizationCalibrator` determines optimal quantization parameters from calibration data. Expose via `/api/quantize` endpoints.

**Device optimization** (`src/device/`): `DeviceOptimizer` detects CPU capabilities (AVX2, AVX512, NEON), memory topology, and environment (Docker/VM/bare-metal). `auto_tune_configs()` generates optimal batch, preprocessing, inference, and training configs for the detected hardware. Use this to set sensible defaults instead of hardcoding.

**Monitoring** (`src/monitoring/`): `PerformanceMetrics` tracks latencies and throughput. `AlertManager` fires alerts on configurable conditions. `SystemMonitor` provides CPU/memory/disk stats. Exposed via `/api/monitoring/*` endpoints.

**Security** (`src/security/`): API key auth, JWT verification, rate limiting (token bucket and sliding window algorithms), TLS management, and secrets management. Configured via `SECURITY_*` env vars. Rate limiter is middleware-composable.

**Adaptive processing** (`src/adaptive/`): Auto-tunes preprocessing and hyperparameter optimization strategies based on dataset characteristics (size, dimensionality, sparsity). Smaller datasets get more exhaustive search; larger ones get faster heuristics.

**Mixed precision** (`src/precision/`): `MixedPrecisionManager` selects f32/f64 precision per operation based on numerical sensitivity. Use lower precision on non-critical paths (feature scaling) and full precision where it matters (gradient computation, loss calculation).

When building serving features:
- Profile with `cargo bench` before and after changes. Latency regressions on the inference path are blockers.
- Respect the memory budget — the platform must run on machines with as little as 4GB RAM. Use `MemoryMonitor` to gate expensive operations.
- Batch and cache aggressively. Single-request latency matters less than throughput on local deployments serving a web UI.
- Use `DeviceOptimizer::auto_tune_configs()` to derive defaults rather than hardcoding magic numbers for batch sizes, thread counts, or buffer sizes.
- Keep the server startup fast (<1s). Defer expensive initialization (model loading, cache warming) to first request or background tasks.

## Key Conventions

- **Error handling**: Use `AutoMLError` enum (in `src/error.rs`) and the `Result<T>` type alias. Each error variant is domain-specific (DataError, TrainingError, etc.). Convert external errors via `From` impls.
- **Data types**: `ndarray::Array2<f64>` is the standard matrix type for ML operations. `polars::DataFrame` is used for data loading and preprocessing. Convert between them at module boundaries.
- **Serialization**: All model structs derive `Serialize`/`Deserialize` (serde). Models are serialized with bincode for binary format.
- **Parallelism**: Uses `rayon` for data-parallel operations and `tokio` for async I/O (server, CLI async commands). Concurrent state uses `parking_lot` mutexes and `dashmap`.
- **Server config**: Reads from environment variables (`API_HOST`, `API_PORT`, `STATIC_DIR`, `DATA_DIR`, `MODELS_DIR`, `MAX_UPLOAD_SIZE`). See `.env.example` for all options.
- **Testing**: Integration tests are in `tests/`. Unit tests are co-located in source files using `#[cfg(test)]` modules. Dev dependencies include `criterion` (benchmarks), `proptest` (property testing), `rstest` (parameterized tests).
- **Release profile**: Uses fat LTO, single codegen unit, and `panic = "abort"` for maximum optimization.
