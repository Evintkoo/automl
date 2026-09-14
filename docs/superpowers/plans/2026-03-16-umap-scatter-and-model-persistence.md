# UMAP Streaming Scatter & Model Persistence Fix — Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix trained models disappearing on page refresh, and add a streaming UMAP scatter plot to the training results card for classification, regression, and clustering models.

**Architecture:** (1) At server startup, scan the `models/` directory and restore `ModelInfo` entries from saved JSON files so models survive restarts. (2) Extend `Umap` with a callback-based `fit_transform_with_cb` method that emits partial embeddings every 20 epochs. (3) Add a new SSE endpoint that bridges the blocking UMAP computation to an async stream via `tokio::sync::mpsc`. (4) After training completes in the frontend `ePoll` function, inject a canvas + `EventSource` that progressively renders the UMAP scatter coloured by class, target value, or cluster ID.

**Tech Stack:** Rust, Axum (SSE via `axum::response::sse`), Tokio (`spawn_blocking` + `mpsc`), vanilla JS + Canvas 2D API.

**Spec:** `docs/superpowers/specs/2026-03-16-umap-scatter-and-model-persistence-design.md`

**Security note:** Model IDs used to build DOM element IDs are server-generated strings (filename stems or UUIDs). Before inserting them into DOM attribute strings, sanitize with a replace: `modelId.replace(/[^a-zA-Z0-9_-]/g, '')`. This prevents any unexpected characters from reaching the DOM even if the value were somehow tampered.

**Pre-flight: Cargo.toml changes required before any task**

Make these changes to `Cargo.toml` first:

1. Add `"sse"` to axum features:
```toml
axum = { version = "0.7", features = ["multipart", "ws", "sse"] }
```

2. Add `async-stream`:
```toml
async-stream = "0.3"
```

3. Note: `tempfile = "3.15"` is already in `[dependencies]` (production) — do NOT add it again to `[dev-dependencies]`. Tests can use it as-is.

Run `cargo build` after these changes to verify they resolve before proceeding.

---

## Chunk 1: UMAP epoch callback

### Task 1: Add `fit_transform_with_cb` to `umap.rs`

**Files:**
- Modify: `src/visualization/umap.rs`

Context: `fit_transform` is at line 105. `optimize_layout` is at line 313. The inner epoch loop is the `for epoch in 0..n_epochs` loop at line 334. The existing tests are in the `#[cfg(test)]` block starting at line 451.

The plan: pull the `optimize_layout` body into a new `optimize_layout_with_cb` that fires a closure every `UMAP_STREAM_EMIT_EVERY` epochs. `fit_transform_with_cb` runs the three phases feeding results to the closure. `fit_transform` delegates to `fit_transform_with_cb` with `|_, _| {}`.

**Borrow safety note:** `on_epoch(epoch + 1, &embedding)` is called AFTER the inner `for edge in edges` loop body completes — at the bottom of the outer epoch loop. At that point, no other borrows of `embedding` are active. This is safe.

- [ ] **Step 1: Write the failing test**

Add inside the `#[cfg(test)]` block in `src/visualization/umap.rs`:

```rust
#[test]
fn test_fit_transform_with_cb_fires_callbacks() {
    let data: Vec<Vec<f64>> = (0..20)
        .map(|i| vec![i as f64, (i % 2) as f64 * 10.0])
        .collect();
    let config = UmapConfig { n_neighbors: 3, n_epochs: 40, ..Default::default() };
    let umap = Umap::new(config);

    let mut callback_count = 0usize;
    let mut last_points_len = 0usize;
    let result = umap.fit_transform_with_cb(&data, |_epoch, pts| {
        callback_count += 1;
        last_points_len = pts.len();
    }).unwrap();

    // 40 epochs / 20 per emit = 2 mid-run callbacks + 1 final = at least 3
    assert!(callback_count >= 3, "expected >= 3 callbacks, got {}", callback_count);
    assert_eq!(last_points_len, data.len());
    assert_eq!(result.len(), data.len());
}

#[test]
fn test_fit_transform_delegates_to_with_cb() {
    let data: Vec<Vec<f64>> = (0..10)
        .map(|i| vec![i as f64, 0.0, 1.0])
        .collect();
    let config = UmapConfig { n_neighbors: 3, n_epochs: 20, ..Default::default() };
    let umap = Umap::new(config);
    // fit_transform must still work (delegates to with_cb with no-op)
    let result = umap.fit_transform(&data).unwrap();
    assert_eq!(result.len(), data.len());
}
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
cd /Users/evintleovonzko/Documents/Works/AutoML/automl
cargo test test_fit_transform_with_cb_fires_callbacks test_fit_transform_delegates_to_with_cb 2>&1 | tail -20
```

Expected: compile error — `fit_transform_with_cb` not found.

- [ ] **Step 3: Add the constant and `fit_transform_with_cb` method**

In `src/visualization/umap.rs`, add after the `pub fn fit_transform` method (after line 174):

```rust
/// How many SGD epochs between callback emissions during streaming.
const UMAP_STREAM_EMIT_EVERY: usize = 20;

/// Run UMAP and emit partial embeddings to `on_epoch` every
/// `UMAP_STREAM_EMIT_EVERY` epochs and once after the final epoch.
/// The final return value is the fully-converged embedding.
pub fn fit_transform_with_cb<F>(
    &self,
    data: &[Vec<f64>],
    mut on_epoch: F,
) -> crate::error::Result<Vec<[f64; 2]>>
where
    F: FnMut(usize, &[[f64; 2]]),
{
    let n = data.len();
    if n < 3 {
        return Err(crate::error::AutoMLError::DataError(
            "UMAP requires at least 3 samples".to_string(),
        ));
    }

    let k = self.config.n_neighbors.min(n - 1);

    let (work_data, sample_indices) = if n > self.config.max_samples {
        let mut rng = ChaCha8Rng::seed_from_u64(self.config.random_state);
        let mut indices: Vec<usize> = (0..n).collect();
        for i in 0..self.config.max_samples {
            let j = rng.gen_range(i..n);
            indices.swap(i, j);
        }
        indices.truncate(self.config.max_samples);
        indices.sort_unstable();
        let sampled: Vec<Vec<f64>> = indices.iter().map(|&i| data[i].clone()).collect();
        (sampled, Some(indices))
    } else {
        (data.to_vec(), None)
    };

    let n_work = work_data.len();
    let (knn_indices, knn_distances) = self.compute_knn(&work_data, k);
    let edges = self.compute_fuzzy_set(&knn_indices, &knn_distances, k);

    let mut embedding = self.optimize_layout_with_cb(n_work, &edges, |epoch, emb| {
        on_epoch(epoch, emb);
    });

    if let Some(indices) = sample_indices {
        let mut full_embedding = vec![[0.0f64; 2]; n];
        for (sub_idx, &orig_idx) in indices.iter().enumerate() {
            full_embedding[orig_idx] = embedding[sub_idx];
        }
        for i in 0..n {
            if !indices.contains(&i) {
                let mut best_dist = f64::INFINITY;
                let mut best_idx = 0;
                for &si in &indices {
                    let d = SimdOps::squared_euclidean_distance(&data[i], &data[si]);
                    if d < best_dist {
                        best_dist = d;
                        best_idx = si;
                    }
                }
                let emb = full_embedding[best_idx];
                full_embedding[i] = [
                    emb[0] + (i as f64 * 0.001).sin() * 0.1,
                    emb[1] + (i as f64 * 0.001).cos() * 0.1,
                ];
            }
        }
        embedding = full_embedding;
    }

    // Final callback: always fires with the fully-converged embedding
    on_epoch(self.config.n_epochs, &embedding);

    Ok(embedding)
}
```

- [ ] **Step 4: Add `optimize_layout_with_cb` and update `optimize_layout`**

Replace the existing `fn optimize_layout` at line 313 with two methods:

```rust
/// SGD layout with epoch callbacks — fires on_epoch every UMAP_STREAM_EMIT_EVERY epochs.
fn optimize_layout_with_cb<F>(
    &self,
    n_samples: usize,
    edges: &[Edge],
    mut on_epoch: F,
) -> Vec<[f64; 2]>
where
    F: FnMut(usize, &[[f64; 2]]),
{
    let (a, b) = self.find_ab_params(self.config.spread, self.config.min_dist);
    let mut rng = ChaCha8Rng::seed_from_u64(self.config.random_state);
    let mut embedding: Vec<[f64; 2]> = (0..n_samples)
        .map(|_| [rng.gen_range(-10.0..10.0) * 0.01, rng.gen_range(-10.0..10.0) * 0.01])
        .collect();

    let n_epochs = self.config.n_epochs;
    let neg_rate = self.config.negative_sample_rate;
    let max_weight = edges.iter().map(|e| e.weight).fold(0.0_f64, f64::max);

    for epoch in 0..n_epochs {
        let alpha = self.config.learning_rate * (1.0 - epoch as f64 / n_epochs as f64);
        if alpha < 1e-8 { break; }

        for edge in edges {
            let epochs_per_sample = if edge.weight > 0.0 {
                max_weight / edge.weight
            } else {
                f64::INFINITY
            };
            if epoch as f64 % epochs_per_sample.max(1.0) >= 1.0 { continue; }

            let i = edge.i;
            let j = edge.j;
            let dy = [embedding[i][0] - embedding[j][0], embedding[i][1] - embedding[j][1]];
            let dist_sq = dy[0] * dy[0] + dy[1] * dy[1] + 1e-8;
            let grad_coeff = -2.0 * a * b * dist_sq.powf(b - 1.0) / (1.0 + a * dist_sq.powf(b));
            let gd0 = grad_coeff * dy[0];
            let gd1 = grad_coeff * dy[1];
            embedding[i][0] += alpha * gd0;
            embedding[i][1] += alpha * gd1;
            embedding[j][0] -= alpha * gd0;
            embedding[j][1] -= alpha * gd1;

            for _ in 0..neg_rate {
                let k = rng.gen_range(0..n_samples);
                if k == i { continue; }
                let dy_neg = [embedding[i][0] - embedding[k][0], embedding[i][1] - embedding[k][1]];
                let dsq_neg = dy_neg[0] * dy_neg[0] + dy_neg[1] * dy_neg[1] + 1e-8;
                let gc_neg = 2.0 * b / ((0.001 + dsq_neg) * (1.0 + a * dsq_neg.powf(b)));
                embedding[i][0] += alpha * gc_neg * dy_neg[0];
                embedding[i][1] += alpha * gc_neg * dy_neg[1];
            }

            embedding[i][0] = embedding[i][0].clamp(-10.0, 10.0);
            embedding[i][1] = embedding[i][1].clamp(-10.0, 10.0);
            embedding[j][0] = embedding[j][0].clamp(-10.0, 10.0);
            embedding[j][1] = embedding[j][1].clamp(-10.0, 10.0);
        }

        if (epoch + 1) % UMAP_STREAM_EMIT_EVERY == 0 {
            on_epoch(epoch + 1, &embedding);
        }
    }

    embedding
}

/// Phase 3: SGD layout optimization (no callbacks).
fn optimize_layout(&self, n_samples: usize, edges: &[Edge]) -> Vec<[f64; 2]> {
    self.optimize_layout_with_cb(n_samples, edges, |_, _| {})
}
```

Delete the old `optimize_layout` body entirely.

- [ ] **Step 5: Update `fit_transform` to delegate**

Replace the entire body of `fit_transform` with:

```rust
pub fn fit_transform(&self, data: &[Vec<f64>]) -> crate::error::Result<Vec<[f64; 2]>> {
    self.fit_transform_with_cb(data, |_, _| {})
}
```

- [ ] **Step 6: Run all umap tests**

```bash
cargo test --lib visualization::umap 2>&1 | tail -20
```

Expected: all 6 tests pass (4 existing + 2 new).

- [ ] **Step 7: Commit**

```bash
git add src/visualization/umap.rs
git commit -m "feat(umap): add fit_transform_with_cb with epoch streaming support"
```

---

## Chunk 2: Model persistence fix

### Task 2: Add `restore_models_from_disk` and call it on startup

**Files:**
- Modify: `src/server/state.rs`
- Modify: `src/server/mod.rs` (line 95, after `AppState::new`)
- Possibly modify: `src/training/engine.rs` (add `config()` accessor if missing)

Context: `AppState::new` is at `state.rs:208`. `ModelInfo` is at `state.rs:47`. In `mod.rs`, line 95 is `let state = Arc::new(AppState::new(config.clone()));`. `TrainEngine::load` is at `engine.rs:221`. Model files are named `model_{type}_{job_id}.json`.

- [ ] **Step 1: Write failing test**

Add to `src/server/state.rs` at the bottom:

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use crate::training::engine::TrainEngine;
    use crate::training::{TrainingConfig, TaskType, ModelType};
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
        restore_models_from_disk(dir.path().to_str().unwrap(), &models);
        assert_eq!(models.len(), 0);
    }
}
```

`tempfile = "3.15"` is already in `[dependencies]` in `Cargo.toml` — do NOT add it again. Tests can use it directly.

- [ ] **Step 2: Run tests to confirm they fail**

```bash
cargo test test_restore_models_from_disk 2>&1 | tail -20
```

Expected: compile error — `restore_models_from_disk` not found.

- [ ] **Step 3: Add `config()` accessor to `TrainEngine` in `engine.rs`**

`config` is a private field (`config: TrainingConfig` at line 125, no pub). Add this accessor to `impl TrainEngine` in `src/training/engine.rs` (after the existing `metrics()` method, around line 199):

```rust
/// Return the training configuration.
pub fn config(&self) -> &TrainingConfig {
    &self.config
}
```

Verify it compiles:
```bash
cargo build 2>&1 | grep "^error" | head -5
```

- [ ] **Step 4: Implement `restore_models_from_disk` in `state.rs`**

Add this free function to `src/server/state.rs` (outside any `impl` block):

```rust
/// Scan `models_dir` for `model_*.json` files and populate `models` with
/// restored `ModelInfo` entries. Called once at server startup (synchronously,
/// before axum::serve). Skips autotune_ and optimized_ files. Logs and skips
/// malformed files without panicking.
pub fn restore_models_from_disk(models_dir: &str, models: &dashmap::DashMap<String, ModelInfo>) {
    use crate::training::engine::TrainEngine;
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
```

- [ ] **Step 5: Call `restore_models_from_disk` in `mod.rs`**

In `src/server/mod.rs`, immediately after line 95:

```rust
let state = Arc::new(AppState::new(config.clone()));

// Restore previously-trained models from disk so they survive server restarts
crate::server::state::restore_models_from_disk(&config.models_dir, &state.models);
```

- [ ] **Step 6: Run restore tests**

```bash
cargo test test_restore_models_from_disk 2>&1 | tail -30
```

Expected: all 3 tests pass.

- [ ] **Step 7: Run full test suite**

```bash
cargo test 2>&1 | tail -20
```

Expected: no new failures.

- [ ] **Step 8: Commit**

```bash
git add src/server/state.rs src/server/mod.rs src/training/engine.rs Cargo.toml
git commit -m "fix(persistence): restore trained models from disk on server startup"
```

---

## Chunk 3: SSE streaming endpoint

### Task 3: Add `umap_stream_handler` in `handlers.rs` + route in `api.rs`

**Files:**
- Modify: `src/server/handlers.rs` (append new handler near the insights handlers ~line 7770)
- Modify: `src/server/api.rs:122` (add `.route("/visualization/umap/stream", get(...))`)

Context: Axum SSE requires `axum::response::sse::{Sse, Event, KeepAlive}` — `"sse"` feature was added in the pre-flight step. `async-stream = "0.3"` was also added. `futures-util` is already present as an Axum transitive dep. `polars::prelude::IdxCa` is used for row selection via `df.take(&idx_ca)`.

**Data source:** The handler reads the dataset from `state.current_data` (an `Arc<RwLock<Option<DataFrame>>>`). It loads `TrainEngine` from disk (using `TrainEngine::load`) only to recover `config().target_column` — the training engine is NOT used for inference in this handler. The feature matrix is built by dropping the target column from the sampled DataFrame.

- [ ] **Step 1: Add stub handler and route to verify compile**

In `handlers.rs` (near end of file):

```rust
#[derive(serde::Deserialize)]
pub struct UmapStreamParams {
    pub model_id: String,
    pub max_samples: Option<usize>,
}

pub async fn umap_stream_handler(
    axum::extract::State(_state): axum::extract::State<std::sync::Arc<crate::server::state::AppState>>,
    axum::extract::Query(_params): axum::extract::Query<UmapStreamParams>,
) -> impl axum::response::IntoResponse {
    axum::http::StatusCode::NOT_IMPLEMENTED
}
```

In `api.rs`, after the `/visualization/pca` route:
```rust
.route("/visualization/umap/stream", get(handlers::umap_stream_handler))
```

- [ ] **Step 2: Compile check**

```bash
cargo build 2>&1 | grep "^error" | head -10
```

Expected: clean.

- [ ] **Step 3: Implement full SSE handler**

Replace the stub. The full handler structure:

```rust
pub async fn umap_stream_handler(
    axum::extract::State(state): axum::extract::State<std::sync::Arc<crate::server::state::AppState>>,
    axum::extract::Query(params): axum::extract::Query<UmapStreamParams>,
) -> axum::response::sse::Sse<impl futures_util::Stream<Item = Result<axum::response::sse::Event, std::convert::Infallible>>> {
    use axum::response::sse::{Event, KeepAlive, Sse};
    use tokio::sync::mpsc;

    let model_id = params.model_id.clone();
    let max_samples = params.max_samples.unwrap_or(2000);

    // Helper closure to make a single-event error stream
    macro_rules! err_stream {
        ($msg:expr) => {{
            let payload = serde_json::json!({"error": $msg}).to_string();
            let s = futures_util::stream::once(
                async move { Ok(Event::default().data(payload)) }
            );
            return Sse::new(s).keep_alive(KeepAlive::default());
        }};
    }

    // 1. Look up model
    let model_info = match state.models.get(&model_id) {
        Some(m) => m.clone(),
        None => err_stream!("model not found"),
    };

    // 2. Load engine to recover target_column
    let engine = match crate::training::engine::TrainEngine::load(
        model_info.path.to_str().unwrap_or("")
    ) {
        Ok(e) => e,
        Err(e) => err_stream!(format!("failed to load model: {}", e)),
    };
    let target_col = engine.config().target_column.clone();
    let task_type_str = model_info.task_type.clone();
    let n_epochs = 200usize;

    // 3. Clone dataset
    let df = match state.current_data.read().await.clone() {
        Some(d) => d,
        None => err_stream!("no dataset loaded"),
    };

    // 4. Sample rows
    let n = df.height();
    let sample_size = max_samples.min(n);
    let indices: Vec<u32> = {
        use rand::seq::SliceRandom;
        let mut rng = rand::thread_rng();
        let mut idx: Vec<u32> = (0..n as u32).collect();
        idx.shuffle(&mut rng);
        idx.truncate(sample_size);
        idx.sort_unstable();
        idx
    };
    let idx_ca = polars::prelude::IdxCa::from_vec("idx".into(), indices.iter().map(|&i| i as polars::prelude::IdxSize).collect());
    let sampled_df = df.take(&idx_ca).unwrap_or_else(|_| df.clone());

    // 5. Extract labels
    let is_regression = task_type_str.contains("Regression") || task_type_str.contains("TimeSeries");
    let labels: Vec<serde_json::Value> = if let Ok(col) = sampled_df.column(&target_col) {
        col.cast(&polars::prelude::DataType::Float64).ok()
            .and_then(|s| s.f64().ok().map(|ca| {
                ca.into_iter().map(|v| {
                    let val = v.unwrap_or(0.0);
                    if is_regression {
                        serde_json::json!(val)
                    } else if task_type_str.contains("Clustering") {
                        serde_json::json!(format!("cluster_{}", val as i64))
                    } else {
                        serde_json::json!(format!("class_{}", val as i64))
                    }
                }).collect()
            })).unwrap_or_default()
    } else { vec![] };

    let value_range: Option<(f64, f64)> = if is_regression {
        let vals: Vec<f64> = labels.iter().filter_map(|v| v.as_f64()).collect();
        if vals.is_empty() { None } else {
            Some((
                vals.iter().cloned().fold(f64::INFINITY, f64::min),
                vals.iter().cloned().fold(f64::NEG_INFINITY, f64::max),
            ))
        }
    } else { None };

    // 6. Feature matrix (drop target column)
    let feature_df = sampled_df.drop(&target_col).unwrap_or_else(|_| sampled_df.clone());
    let feature_data: Vec<Vec<f64>> = (0..feature_df.height()).map(|row| {
        feature_df.get_columns().iter().map(|col| {
            col.cast(&polars::prelude::DataType::Float64).ok()
                .and_then(|s| s.f64().ok().map(|ca| ca.get(row).unwrap_or(0.0)))
                .unwrap_or(0.0)
        }).collect()
    }).collect();

    // 7. mpsc channel + spawn_blocking
    let (tx, mut rx) = mpsc::channel::<(usize, Vec<[f64; 2]>)>(4);

    tokio::task::spawn_blocking(move || {
        let config = crate::visualization::umap::UmapConfig {
            n_epochs,
            max_samples,
            ..Default::default()
        };
        let umap = crate::visualization::umap::Umap::new(config);
        let _ = umap.fit_transform_with_cb(&feature_data, |epoch, pts| {
            let _ = tx.blocking_send((epoch, pts.to_vec()));
        });
    });

    // 8. Async SSE stream
    let labels_arc = std::sync::Arc::new(labels);
    let tt = task_type_str;
    let vr = value_range;
    let is_reg = is_regression;

    let event_stream = async_stream::stream! {
        while let Some((epoch, pts)) = rx.recv().await {
            let progress = (epoch as f64 / n_epochs as f64).min(1.0);
            let done = progress >= 1.0;

            let points: Vec<serde_json::Value> = pts.iter().enumerate().map(|(i, p)| {
                let label = labels_arc.get(i).cloned().unwrap_or(serde_json::json!("?"));
                if is_reg {
                    serde_json::json!({"x": p[0], "y": p[1], "value": label})
                } else {
                    serde_json::json!({"x": p[0], "y": p[1], "label": label})
                }
            }).collect();

            let mut payload = serde_json::json!({
                "progress": progress,
                "done": done,
                "task_type": tt,
                "points": points,
            });
            if let Some((mn, mx)) = vr {
                payload["value_range"] = serde_json::json!([mn, mx]);
            }

            yield Ok::<Event, std::convert::Infallible>(
                Event::default().data(payload.to_string())
            );
        }
    };

    Sse::new(event_stream).keep_alive(KeepAlive::default())
}
```

If `async-stream` is missing from `Cargo.toml`, add:
```toml
async-stream = "0.3"
```

- [ ] **Step 4: Compile check**

```bash
cargo build 2>&1 | grep "^error" | head -20
```

Fix any API mismatches. Common ones:
- `IdxCa::from_vec` vs `IdxCa::new` — check the polars version in Cargo.lock
- `engine.config().task_type` — if `config` is a public field, use `engine.config.task_type` directly
- `futures_util::stream::once` vs `std::iter::once` — use `futures_util::stream::once(async { ... })`

- [ ] **Step 5: Smoke test**

```bash
cargo run -- serve &
sleep 3
# Upload a small CSV with a target column in the browser
# Train one model
# Then check the stream endpoint responds with SSE lines
```

- [ ] **Step 6: Commit**

```bash
git add src/server/handlers.rs src/server/api.rs Cargo.toml
git commit -m "feat(umap): add SSE streaming endpoint /api/visualization/umap/stream"
```

---

## Chunk 4: Frontend UMAP canvas

### Task 4: Inject UMAP canvas into training results card

**Files:**
- Modify: `src/server/handlers.rs` — embedded JS in `serve_index` handler

Context: The `ePoll` function is at line 2312. It's one long minified line. The completion branch sets `$('e-result').innerHTML = h` (metrics table), then calls `eDashTrainResult`, `eShowTrainViz`, `eNotify`, `eLogActivity`, `eDashModels`, then fetches `/api/models` to set `eLastModelId`.

**Security note:** Model IDs are sanitized before DOM injection with `.replace(/[^a-zA-Z0-9_-]/g, '')` to ensure no unintended characters reach element IDs or attribute strings.

The JS is embedded as a Rust string literal (between `r#"..."#` or similar). All edits must maintain valid Rust string escaping. The JS uses single quotes throughout — do not introduce double quotes in new JS.

- [ ] **Step 1: Add the `eUmapScatter` and `eStartUmap` helper functions**

Find where `eAtCharts` function ends in the JS (around line 2332–2333) and append these two functions immediately after, in the same minified style (single line each in the Rust source):

```javascript
function eUmapScatter(modelId,taskType,pts,vr,done){var c=document.getElementById('umap-s-'+modelId);if(!c)return;var dpr=window.devicePixelRatio||1;var W=c.offsetWidth||320,H=280;c.width=W*dpr;c.height=H*dpr;c.style.width=W+'px';c.style.height=H+'px';var ctx=c.getContext('2d');ctx.scale(dpr,dpr);ctx.clearRect(0,0,W,H);var PAL=['#0066f5','#f43f5e','#3abc3f','#ffa931','#8b5cf6','#06b6d4','#ec4899','#f97316','#14b8a6','#a78bfa'];var isReg=taskType==='Regression'||taskType==='TimeSeries';if(!pts||!pts.length)return;var xs=pts.map(function(p){return p.x}),ys=pts.map(function(p){return p.y});var xMin=Math.min.apply(null,xs),xMax=Math.max.apply(null,xs);var yMin=Math.min.apply(null,ys),yMax=Math.max.apply(null,ys);var pad=isReg?{t:16,r:64,b:16,l:16}:{t:16,r:16,b:16,l:16};var pW=W-pad.l-pad.r,pH=H-pad.t-pad.b;var xR=xMax-xMin||1,yR=yMax-yMin||1;function tx(v){return pad.l+(v-xMin)/xR*pW}function ty(v){return pad.t+pH-(v-yMin)/yR*pH}var labelMap={};var labelIdx=0;pts.forEach(function(p){if(isReg)return;var lbl=p.label||'?';if(labelMap[lbl]===undefined)labelMap[lbl]=labelIdx++});pts.forEach(function(p){var cx2=tx(p.x),cy2=ty(p.y);var col;if(isReg){var mn=vr?vr[0]:0,mx2=vr?vr[1]:1;var t=mx2>mn?(p.value-mn)/(mx2-mn):0.5;t=Math.max(0,Math.min(1,t));col='hsl('+Math.round((1-t)*240)+',80%,55%)'}else{col=PAL[(labelMap[p.label||'?']||0)%PAL.length]}ctx.beginPath();ctx.arc(cx2,cy2,3,0,Math.PI*2);ctx.globalAlpha=0.75;ctx.fillStyle=col;ctx.fill();ctx.globalAlpha=1});ctx.font='11px system-ui';ctx.textBaseline='middle';if(isReg){var gx=W-pad.r+8,gy=pad.t,gh=pH,gw=12;var grad=ctx.createLinearGradient(0,gy,0,gy+gh);grad.addColorStop(0,'hsl(0,80%,55%)');grad.addColorStop(1,'hsl(240,80%,55%)');ctx.fillStyle=grad;ctx.fillRect(gx,gy,gw,gh);ctx.fillStyle='#6a6f73';var mn3=vr?vr[0]:0,mx3=vr?vr[1]:1;ctx.textAlign='left';ctx.fillText(mx3.toFixed(1),gx+gw+3,gy);ctx.fillText(mn3.toFixed(1),gx+gw+3,gy+gh)}else{var ly=H-12-Object.keys(labelMap).length*16;Object.keys(labelMap).forEach(function(lbl,i){var ci=PAL[labelMap[lbl]%PAL.length];ctx.beginPath();ctx.arc(14,ly+i*16,5,0,Math.PI*2);ctx.fillStyle=ci;ctx.fill();ctx.fillStyle='#6a6f73';ctx.textAlign='left';ctx.fillText(lbl,22,ly+i*16)})}ctx.font='bold 11px system-ui';ctx.fillStyle='#9c9fa1';ctx.textAlign='left';ctx.textBaseline='top';ctx.fillText(done?'UMAP Embedding':'UMAP (computing\u2026)',8,4)}
function eStartUmap(rawId,taskType){var modelId=rawId.replace(/[^a-zA-Z0-9_-]/g,'');var wrap=document.getElementById('umap-wrap-'+modelId);var bar=document.getElementById('umap-bar-'+modelId);if(!wrap)return;if(!window._umapSrc)window._umapSrc={};if(window._umapSrc[modelId])window._umapSrc[modelId].close();var es=new EventSource('/api/visualization/umap/stream?model_id='+encodeURIComponent(rawId));window._umapSrc[modelId]=es;es.onmessage=function(e){var d;try{d=JSON.parse(e.data)}catch(ex){return}if(d.error){if(wrap){wrap.textContent='UMAP: '+d.error}es.close();return}if(bar)bar.style.width=Math.round((d.progress||0)*100)+'%';eUmapScatter(modelId,d.task_type,d.points,d.value_range,d.done);if(d.done){if(bar&&bar.parentNode)bar.parentNode.style.display='none';es.close()}};es.onerror=function(){if(bar&&bar.parentNode)bar.parentNode.style.display='none';var c=document.getElementById('umap-s-'+modelId);if(c){var ctx=c.getContext('2d');ctx.clearRect(0,0,c.width,c.height);ctx.fillStyle='#9c9fa1';ctx.font='12px system-ui';ctx.textAlign='center';ctx.textBaseline='middle';ctx.fillText('UMAP unavailable',c.width/2,c.height/2)}es.close()}}
```

- [ ] **Step 2: Modify the `ePoll` completion branch to start UMAP**

In the `ePoll` function (line 2312), find the models fetch at the end of the Completed branch:

```javascript
fetch('/api/models').then(function(r){return r.json()}).then(function(md){var models=md.models||[];if(models.length>0){eLastModelId=models[models.length-1].id;$('e-explain-btn').disabled=false}}).catch(function(){})
```

Also capture `cfg` before the fetch (it's already in scope as the `cfg` object sent to `/api/train`). Replace the models fetch with:

```javascript
var _trainCfg=cfg;fetch('/api/models').then(function(r){return r.json()}).then(function(md){var models=md.models||[];if(models.length>0){eLastModelId=models[models.length-1].id;$('e-explain-btn').disabled=false;var safeId=eLastModelId.replace(/[^a-zA-Z0-9_-]/g,'');var tt=_trainCfg&&_trainCfg.task_type?_trainCfg.task_type:'';var umapDiv=document.createElement('div');umapDiv.style.cssText='margin-top:16px';var barWrap=document.createElement('div');barWrap.style.cssText='margin-bottom:4px';var barTrack=document.createElement('div');barTrack.style.cssText='height:3px;background:#1a1b1c;border-radius:2px';var barFill=document.createElement('div');barFill.id='umap-bar-'+safeId;barFill.style.cssText='height:3px;background:#0066f5;border-radius:2px;width:0%;transition:width .3s';barTrack.appendChild(barFill);barWrap.appendChild(barTrack);var wrapDiv=document.createElement('div');wrapDiv.id='umap-wrap-'+safeId;var canvas=document.createElement('canvas');canvas.id='umap-s-'+safeId;canvas.style.cssText='width:100%;height:280px;display:block;border-radius:8px;background:#0d0e0f';wrapDiv.appendChild(canvas);umapDiv.appendChild(barWrap);umapDiv.appendChild(wrapDiv);$('e-result').appendChild(umapDiv);eStartUmap(eLastModelId,tt)}}).catch(function(){})
```

Note: Instead of setting `innerHTML` for the UMAP section (which would trigger the security hook), we use `createElement`/`appendChild` throughout. This matches the existing pattern used elsewhere in the Insights tab for canvas injection.

- [ ] **Step 3: Compile check**

```bash
cargo build 2>&1 | grep "^error" | head -10
```

Fix any Rust string-escaping issues.

- [ ] **Step 4: Manual end-to-end test**

```bash
cargo run -- serve &
sleep 3
# Open http://localhost:8080
# Upload iris.csv (classification)
# Train RandomForest on species column
# After completion: verify metrics table + UMAP canvas appear
# Verify scatter updates progressively (dots refine positions)
# Verify colored legend appears in corner

# Train a regression model (e.g. Boston housing, target = medv)
# Verify cool-to-warm gradient colormap + gradient bar legend

# Kill server (ctrl+c), restart: cargo run -- serve &
# Verify model appears in models list without retraining
```

- [ ] **Step 5: Commit**

```bash
git add src/server/handlers.rs
git commit -m "feat(frontend): add streaming UMAP scatter plot to training results card"
```

---

## Chunk 5: Final verification

### Task 5: Run full test suite and verify

- [ ] **Step 1: Run all Rust tests**

```bash
cargo test 2>&1 | tail -30
```

Expected: all pass.

- [ ] **Step 2: Check for compiler warnings**

```bash
cargo build 2>&1 | grep "warning:" | grep -v "unused import\|dead_code" | head -10
```

- [ ] **Step 3: Run E2E tests if available**

```bash
npx playwright test e2e/umap.spec.ts --reporter=line 2>&1 | tail -20
```

- [ ] **Step 4: Final commit**

```bash
git add -A
git commit -m "chore: final checks — UMAP streaming scatter and model persistence complete"
```
