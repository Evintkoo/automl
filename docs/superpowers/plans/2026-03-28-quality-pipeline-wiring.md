# Quality Pipeline Wiring Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Wire the four quality gates (pre-training, training, hyperopt, post-training) into all three training handlers so `state.quality_reports` and `state.ood_detectors` are populated every time a model is trained.

**Architecture:** Three changes across three files. (1) Make `TrainEngine::extract_features` public so the quality wiring code can extract the feature matrix. (2) Uncomment the `QualityPipeline` impl block and update `run_post_training` to return the `OodDetector` alongside the report. (3) Add a `run_quality_for_model` helper in handlers.rs and call it in `start_training`, `auto_tune`, and `run_automl_pipeline`.

**Tech Stack:** Rust, ndarray, polars, existing `QualityPipeline` / gate types in `src/quality/`.

---

## File Structure

| File | Change |
|---|---|
| `src/training/engine.rs` | `fn extract_features` → `pub fn extract_features` |
| `src/quality/mod.rs` | Uncomment `QualityPipeline` impl; update `run_post_training` to return `(QualityReport, OodDetector)` |
| `src/server/handlers.rs` | Add `run_quality_for_model` helper + call in three training paths |

---

### Task 1: Make `extract_features` public on `TrainEngine`

**Files:**
- Modify: `src/training/engine.rs:267`

- [ ] **Step 1: Locate the method**

Open `src/training/engine.rs` and find line 267:
```rust
    fn extract_features(&self, df: &DataFrame) -> Result<Array2<f64>> {
        Self::columns_to_array2(df, &self.feature_names)
    }
```

- [ ] **Step 2: Make the method public**

Change `fn extract_features` to `pub fn extract_features`:
```rust
    pub fn extract_features(&self, df: &DataFrame) -> Result<Array2<f64>> {
        Self::columns_to_array2(df, &self.feature_names)
    }
```

- [ ] **Step 3: Verify it compiles**

```bash
cd /Users/evintleovonzko/Documents/Works/AutoML/automl
cargo check 2>&1 | head -20
```
Expected: no errors related to this change.

- [ ] **Step 4: Commit**

```bash
git add src/training/engine.rs
git commit -m "feat(quality): make TrainEngine::extract_features pub for quality wiring"
```

---

### Task 2: Uncomment `QualityPipeline` impl and update `run_post_training`

**Files:**
- Modify: `src/quality/mod.rs:275-357`

The `QualityPipeline` impl is currently entirely commented out with a `// TODO:` banner. All gate implementations (`PreTrainingGate`, `TrainingGate`, `HyperoptGate`, `PostTrainingGate`) are already complete in their respective files.

The only change from the original commented code: `run_post_training` currently returns `QualityReport`. We need it to return `(QualityReport, OodDetector)` so the caller can store the detector in `state.ood_detectors`.

- [ ] **Step 1: Write the failing test**

Add this test to the `tests` module at the bottom of `src/quality/mod.rs`:

```rust
    #[test]
    fn test_quality_pipeline_run_pre_training() {
        use ndarray::array;
        let features = array![[1.0f64, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0], [9.0, 10.0]];
        let target = array![0.0f64, 1.0, 0.0, 1.0, 0.0];
        let names = vec!["a".to_string(), "b".to_string()];
        let pipeline = QualityPipeline::default();
        let mut ctx = QualityContext::default();
        let kept = pipeline.run_pre_training(&features, &target, &names, &mut ctx);
        assert!(kept.len() <= names.len());
        assert_eq!(ctx.n_samples, 5);
        assert_eq!(ctx.n_features, 2);
    }

    #[test]
    fn test_quality_pipeline_run_post_training_returns_ood_detector() {
        use ndarray::array;
        let features = array![[0.1f64, 0.2], [0.3, 0.4], [0.5, 0.6], [0.7, 0.8], [0.9, 1.0]];
        let raw_scores = vec![0.2f64, 0.4, 0.6, 0.8, 0.9];
        let labels = vec![0.0f64, 0.0, 1.0, 1.0, 1.0];
        let pipeline = QualityPipeline::default();
        let mut ctx = QualityContext::default();
        ctx.n_samples = 5;
        ctx.n_features = 2;
        let (report, detector) = pipeline.run_post_training(
            "test_model".to_string(), &raw_scores, &labels, &features, &mut ctx
        );
        assert_eq!(report.model_id, "test_model");
        assert!(detector.threshold >= 0.0);
        assert_eq!(detector.means.len(), 2); // 2 features
    }
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
cargo test test_quality_pipeline_run_pre_training test_quality_pipeline_run_post_training_returns_ood_detector 2>&1 | tail -20
```
Expected: FAIL — `QualityPipeline` has no `run_pre_training` or `run_post_training` methods yet.

- [ ] **Step 3: Replace the commented-out impl block**

In `src/quality/mod.rs`, find and replace the entire commented block (from `// TODO: uncomment when gate files are implemented` through the closing `// }` at line ~357) with:

```rust
impl QualityPipeline {
    /// Phase 1: Run pre-training gate. Returns indices of features to keep.
    /// Call before fitting any model.
    pub fn run_pre_training(
        &self,
        features: &ndarray::Array2<f64>,
        target: &ndarray::Array1<f64>,
        feature_names: &[String],
        ctx: &mut QualityContext,
    ) -> Vec<usize> {
        use crate::quality::pre_training::PreTrainingGate;
        let gate = PreTrainingGate {
            leakage_threshold: self.pre_training_leakage_threshold,
            cardinality_threshold: 50,
            shift_sigma: 3.0,
        };
        gate.run(features, target, feature_names, ctx)
    }

    /// Phase 2: Record training gate decisions into ctx.
    /// Call after all CV scores are known, before hyperopt.
    pub fn record_training_decisions(
        &self,
        ctx: &mut QualityContext,
        cv_scores: Vec<(String, f64, f64)>,
        ensemble_score: Option<f64>,
        group_column: Option<String>,
    ) {
        use crate::quality::training::{TrainingGate, AlgorithmSelector, CvStrategyChooser};
        let gate = TrainingGate {
            algorithm_selector: AlgorithmSelector::default(),
            cv_chooser: CvStrategyChooser { repeated_threshold: self.cv_small_dataset_threshold },
            top_k: self.ensemble_top_k,
            ensemble_margin: self.ensemble_margin,
        };
        gate.prepare(ctx, group_column);
        gate.finalize(ctx, cv_scores, ensemble_score);
    }

    /// Phase 3: Record hyperopt decisions into ctx.
    /// Call after hyperopt completes (or pass empty/false for non-hyperopt training).
    pub fn record_hyperopt_decisions(
        &self,
        ctx: &mut QualityContext,
        pareto_points: Vec<ParetoPoint>,
        converged: bool,
        trials_run: usize,
    ) {
        ctx.pareto_front = pareto_points;
        ctx.hyperopt_converged = converged;
        ctx.trials_run = trials_run;
    }

    /// Phase 4: Run post-training gate on calibration set.
    /// `raw_scores`: model's raw output on calibration set (probabilities for classification).
    /// `cal_targets`: true labels/values on calibration set.
    /// `features`: calibration feature matrix (for OOD detector).
    /// Returns the finalised `QualityReport` and the fitted `OodDetector`.
    pub fn run_post_training(
        &self,
        model_id: String,
        raw_scores: &[f64],
        cal_targets: &[f64],
        features: &ndarray::Array2<f64>,
        ctx: &mut QualityContext,
    ) -> (QualityReport, crate::quality::post_training::OodDetector) {
        use crate::quality::post_training::{PostTrainingGate, OodDetector};
        let gate = PostTrainingGate {
            ece_improvement_threshold: 0.005,
            conformal_coverage: self.conformal_coverage,
            ood_percentile: self.ood_percentile,
            n_ece_bins: 10,
        };
        let cal_labels: Vec<f64> = cal_targets.to_vec();
        gate.select_calibration(raw_scores, &cal_labels, ctx);

        let detector = OodDetector::fit(features, self.ood_percentile);
        ctx.ood_threshold = Some(detector.threshold);

        let report = QualityReport::from_context(model_id, ctx);
        (report, detector)
    }
}
```

- [ ] **Step 4: Run tests to confirm they pass**

```bash
cargo test test_quality_pipeline_run_pre_training test_quality_pipeline_run_post_training_returns_ood_detector 2>&1 | tail -20
```
Expected: PASS for both tests.

- [ ] **Step 5: Run full quality test suite**

```bash
cargo test --lib quality 2>&1 | tail -20
```
Expected: all quality tests pass.

- [ ] **Step 6: Commit**

```bash
git add src/quality/mod.rs
git commit -m "feat(quality): uncomment QualityPipeline impl; run_post_training returns (QualityReport, OodDetector)"
```

---

### Task 3: Add `run_quality_for_model` helper and wire into training handlers

**Files:**
- Modify: `src/server/handlers.rs`

This task adds a private helper function `run_quality_for_model` and three call sites.

- [ ] **Step 1: Write the failing test**

In `src/server/handlers.rs` (or a test file), confirm current behavior: quality reports are empty after training. Since this is an integration test that would require a running server, we instead verify via a unit test that checks the function exists and compiles. We'll rely on cargo check here.

Actually, we test by verifying `cargo test` still passes before our change (baseline):

```bash
cargo test --lib 2>&1 | tail -10
```
Expected: all tests pass (note the count so we can verify nothing breaks after).

- [ ] **Step 2: Add `run_quality_for_model` helper function**

Find the line in `src/server/handlers.rs` where `run_training_job` is defined (search for `async fn run_training_job`). Insert the new helper BEFORE `run_training_job`:

```rust
/// Run quality gates for a trained model and store results in AppState.
/// Errors are logged and silently discarded so they never block training completion.
fn run_quality_for_model(
    engine: &TrainEngine,
    df: &polars::prelude::DataFrame,
    task_type: &TaskType,
    model_id: &str,
    target_col: &str,
    state: &AppState,
) {
    use crate::quality::{QualityContext, QualityPipeline};
    use crate::quality::post_training::OodDetector;
    use polars::prelude::DataType;

    let pipeline = QualityPipeline::default();
    let mut ctx = QualityContext::default();

    // Extract feature matrix
    let x = match engine.extract_features(df) {
        Ok(x) => x,
        Err(e) => {
            warn!(model_id = %model_id, error = %e, "Quality pipeline: feature extraction failed");
            return;
        }
    };

    // Extract target as Array1<f64>
    let y: ndarray::Array1<f64> = match df.column(target_col) {
        Ok(series) => {
            match series.cast(&DataType::Float64) {
                Ok(s_f64) => s_f64
                    .f64()
                    .map(|ca| ca.into_iter().map(|v| v.unwrap_or(0.0)).collect())
                    .unwrap_or_default(),
                Err(e) => {
                    warn!(model_id = %model_id, error = %e, "Quality pipeline: target cast failed");
                    return;
                }
            }
        }
        Err(e) => {
            warn!(model_id = %model_id, error = %e, "Quality pipeline: target column not found");
            return;
        }
    };

    let feature_names = engine.feature_names().to_vec();

    // Gate 1: Pre-training (leakage, cardinality, distribution fingerprint)
    pipeline.run_pre_training(&x, &y, &feature_names, &mut ctx);

    // Gate 2: Training (algorithm selection + CV strategy; no CV scores for single-model training)
    pipeline.record_training_decisions(&mut ctx, vec![], None, None);

    // Gate 3: Hyperopt (not run for direct training; record as 0 trials, not converged)
    pipeline.record_hyperopt_decisions(&mut ctx, vec![], false, 0);

    // Gate 4: Post-training (calibration + OOD detector)
    // Calibration requires binary class probabilities; skip for regression/multi-class/clustering.
    let (report, detector) = match task_type {
        TaskType::BinaryClassification => {
            match engine.predict_proba(df) {
                Ok(proba) => {
                    // P(positive class) = column index 1 for binary; fall back to col 0 if only one column
                    let raw_scores: Vec<f64> = proba.outer_iter()
                        .map(|row| if row.len() > 1 { row[1] } else { row[0] })
                        .collect();
                    let labels = y.to_vec();
                    pipeline.run_post_training(model_id.to_string(), &raw_scores, &labels, &x, &mut ctx)
                }
                Err(_) => {
                    // Model does not expose probabilities — skip calibration, still fit OOD detector
                    let detector = OodDetector::fit(&x, pipeline.ood_percentile);
                    ctx.ood_threshold = Some(detector.threshold);
                    let report = crate::quality::QualityReport::from_context(model_id.to_string(), &ctx);
                    (report, detector)
                }
            }
        }
        _ => {
            // Regression, multi-class, time-series, clustering: skip calibration gate
            let detector = OodDetector::fit(&x, pipeline.ood_percentile);
            ctx.ood_threshold = Some(detector.threshold);
            let report = crate::quality::QualityReport::from_context(model_id.to_string(), &ctx);
            (report, detector)
        }
    };

    state.quality_reports.insert(model_id.to_string(), report);
    state.ood_detectors.insert(model_id.to_string(), detector);
    info!(model_id = %model_id, "Quality report stored");
}
```

- [ ] **Step 3: Wire into `start_training`**

In `start_training`, find the block that populates the insights cache (ends around line 895 with `state_clone.insights_cache.insert(...)`). The next statement after that block is `let model_info = ModelInfo { ... }`.

Insert the quality call between the insights cache insertion and the model info registration:

Find this exact code:
```rust
                // Register model in state
                let model_info = ModelInfo {
                    id: model_id.clone(),
                    name: format!("{} (job {})", model_type_name, job_id),
                    task_type: task_type_name,
                    metrics: metrics.clone(),
                    created_at: chrono::Utc::now().to_rfc3339(),
                    path: std::path::PathBuf::from(&model_path),
                };
                state_clone.models.insert(model_id, model_info);
```

Insert the quality call right before `// Register model in state`:
```rust
                // Quality pipeline
                run_quality_for_model(
                    &engine_for_insights,
                    &df,
                    &task_type_for_insights,
                    &model_id,
                    &target_col,
                    &state_clone,
                );

                // Register model in state
                let model_info = ModelInfo {
```

- [ ] **Step 4: Wire into `auto_tune`**

In `auto_tune`, find the outer result-collection loop (after all handles complete). Find this exact code:

```rust
                    let model_id = format!("autotune_{}_{}", name, job_id);
                    state_clone.train_engines.insert(model_id, engine);
```

Replace with:
```rust
                    let model_id = format!("autotune_{}_{}", name, job_id);
                    run_quality_for_model(
                        &engine,
                        &df,
                        &task_type,
                        &model_id,
                        &target_column,
                        &state_clone,
                    );
                    state_clone.train_engines.insert(model_id, engine);
```

- [ ] **Step 5: Wire into `run_automl_pipeline`**

In `run_automl_pipeline`, find the outer result-collection loop. Find this exact code:

```rust
                    // Store engine
                    let key = format!("automl_{}_{}", name, job_id_clone);
                    state_clone.train_engines.insert(key, engine.clone());
```

Replace with:
```rust
                    // Store engine
                    let key = format!("automl_{}_{}", name, job_id_clone);
                    run_quality_for_model(
                        &engine,
                        &df_clone,
                        &task_type,
                        &key,
                        &target_clone,
                        &state_clone,
                    );
                    state_clone.train_engines.insert(key, engine.clone());
```

- [ ] **Step 6: Verify it compiles**

```bash
cargo check 2>&1 | head -30
```
Expected: no errors.

If `TaskType` is not in scope for `run_quality_for_model`, add `use crate::training::TaskType;` at the top of the function body.

- [ ] **Step 7: Run all tests**

```bash
cargo test --lib 2>&1 | tail -20
```
Expected: same test count as before, all passing.

- [ ] **Step 8: Commit**

```bash
git add src/server/handlers.rs
git commit -m "feat(quality): wire quality pipeline into start_training, auto_tune, run_automl_pipeline"
```

---

## Self-Review

**Spec coverage:**
- ✅ Quality reports populated when models are trained (all 3 paths)
- ✅ OOD detectors stored for use at inference time
- ✅ Pre-training gate: leakage, cardinality, distribution fingerprint
- ✅ Training gate: algorithm selection + CV strategy recorded
- ✅ Hyperopt gate: recorded as 0 trials for direct training (not misleading)
- ✅ Post-training gate: calibration for binary classification; OOD for all task types
- ✅ Frontend already wired to `GET /api/quality/report/:model_id` — no frontend changes needed

**Type consistency:**
- `run_post_training` returns `(QualityReport, OodDetector)` — used consistently in both Task 2 tests and Task 3 helper
- `engine.extract_features(df)` returns `Result<Array2<f64>>` — matches ndarray usage in gate files
- `QualityPipeline::default()` uses `ood_percentile: 0.99` — same value passed to `OodDetector::fit`

**No placeholders:** All steps contain complete code.
