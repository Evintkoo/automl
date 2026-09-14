# Codebase Optimization Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Eliminate O(n³) algorithms, remove unnecessary DataFrame clones, add conservative column-parallel rayon parallelism, and reduce async lock contention across 8 files.

**Architecture:** File-by-file in severity order; `cargo check` after each task to catch borrow-checker issues early; full `cargo test` + `cargo bench` comparison at the end.

**Tech Stack:** Rust, Polars, ndarray + rayon feature, rayon, tokio, DashMap (already in Cargo.toml).

**Floating-point note:** Parallel reductions break floating-point associativity. Tiny non-deterministic deltas in correlation/skewness values are an accepted trade-off (the pipeline already has non-deterministic RFE output). Note this in relevant commit messages.

---

## Chunk 1: Algorithmic + Clone Fixes (Tasks 1–4)

---

### Pre-Task: Record benchmark baseline

Before making any changes, record the benchmark baseline for comparison after all tasks are complete.

- [ ] **Step 1: Record baseline**

Run: `cargo bench 2>&1 | tee bench_baseline.txt`

Expected: bench_baseline.txt created. The existing bench groups are named `preprocessing` and `scaling` (see `benches/preprocessing.rs`). The output lines to watch for are those groups' timings — there are no pre-existing per-operation benchmarks for correlation, encoding, or scaler individually. The baseline captures the current state; use it for end-to-end comparison after all 8 tasks are complete.

---

### Task 1: `src/preprocessing/feature_selection.rs` — O(n³) → O(n²)

**Files:**
- Modify: `src/preprocessing/feature_selection.rs`

The `CorrelationFilter::fit()` method (line 496) currently calls `mean_correlation()` (itself O(n)) for every pair in an O(n²) loop → O(n³) total. `discretize()` (line 277) makes three passes over data. The file already has `compute_correlation_matrix()` at line 588 — use it.

- [ ] **Step 1: Read the current `CorrelationFilter::fit()`, `mean_correlation()`, `discretize()`, `compute_correlation_matrix()`, and `pearson_correlation()`**

Read lines 276–294 and 496–640 to confirm signatures and current code before editing. Also grep for `pearson_correlation` to confirm it is a plain associated function (no `&self`):

Run: `sed -n '276,294p' src/preprocessing/feature_selection.rs && sed -n '496,640p' src/preprocessing/feature_selection.rs && grep -n "fn pearson_correlation" src/preprocessing/feature_selection.rs`

- [ ] **Step 2: Replace `CorrelationFilter::fit()` with precomputed-matrix version**

Replace the body of `fit()` (lines 496–535). The existing `compute_correlation_matrix()` at line 588 already builds the full NxN matrix — call it here and replace `mean_correlation()` calls with row-mean reads from the cached matrix:

```rust
pub fn fit(&mut self, x: &Array2<f64>) -> Result<()> {
    let n_features = x.ncols();
    let mut to_remove: HashSet<usize> = HashSet::new();
    let mut removed_pairs = Vec::new();

    // Precompute full correlation matrix once — O(n²) instead of O(n³)
    let corr_matrix = Self::compute_correlation_matrix(x);

    for i in 0..n_features {
        if to_remove.contains(&i) {
            continue;
        }
        for j in (i + 1)..n_features {
            if to_remove.contains(&j) {
                continue;
            }
            let corr = corr_matrix[[i, j]];
            if corr.abs() > self.threshold {
                // Row-mean from precomputed matrix (O(n) read, no recomputation)
                let n_active = n_features - to_remove.len();
                let mean_i: f64 = (0..n_features)
                    .filter(|&k| k != i && !to_remove.contains(&k))
                    .map(|k| corr_matrix[[i, k]].abs())
                    .sum::<f64>() / n_active.max(1) as f64;
                let mean_j: f64 = (0..n_features)
                    .filter(|&k| k != j && !to_remove.contains(&k))
                    .map(|k| corr_matrix[[j, k]].abs())
                    .sum::<f64>() / n_active.max(1) as f64;

                let remove_idx = if mean_i > mean_j { i } else { j };
                to_remove.insert(remove_idx);
                removed_pairs.push((i, j, corr));
            }
        }
    }

    let selected: Vec<usize> = (0..n_features)
        .filter(|i| !to_remove.contains(i))
        .collect();

    self.selected_features = Some(selected);
    self.removed_features = Some(removed_pairs);
    Ok(())
}
```

- [ ] **Step 3: Replace `discretize()` with single-pass version**

Replace the body of `discretize()` (lines 276–294):

```rust
fn discretize(x: &Array1<f64>, n_bins: usize) -> Vec<usize> {
    // Single pass: compute min, max, and bins together
    let mut min_val = f64::INFINITY;
    let mut max_val = f64::NEG_INFINITY;
    for &v in x.iter() {
        if v < min_val { min_val = v; }
        if v > max_val { max_val = v; }
    }
    let range = max_val - min_val;
    if range <= 0.0 {
        return vec![0; x.len()];
    }
    let bin_width = range / n_bins as f64;
    x.iter()
        .map(|&v| ((v - min_val) / bin_width) as usize.min(n_bins - 1))
        .collect()
}
```

- [ ] **Step 4: Add `par_iter` to RFE score computation in `fit_rfe()`**

The `fit_rfe()` method (lines 339–391) computes importance scores for all remaining features inside the `while` loop (lines 356–363). These per-feature scores are fully independent — parallelise the inner score computation with `par_iter`:

```rust
use rayon::prelude::*;

let mut scores: Vec<(usize, f64)> = remaining
    .par_iter()
    .map(|&idx| {
        let col = x.column(idx);
        let importance = Self::compute_correlation(&col.to_owned(), y).abs();
        (idx, importance)
    })
    .collect();
```

Note: `x.column(idx)` is a read-only borrow; the `col.to_owned()` call inside each iteration is safe across threads. The `remaining: HashSet<usize>` is only read here; mutation happens after `.collect()`.

- [ ] **Step 5: Delete `mean_correlation()` (now unused)**

Remove the `mean_correlation` method (lines 604–624) — it is no longer called after Step 2.

- [ ] **Step 6: Add parallelism to `compute_correlation_matrix()` outer loop**

Replace `compute_correlation_matrix()` (lines 588–602) with a parallel version using `axis_iter_mut`:

```rust
pub fn compute_correlation_matrix(x: &Array2<f64>) -> Array2<f64> {
    use ndarray::Axis;
    use rayon::prelude::*;

    let n = x.ncols();
    // Precompute owned columns to avoid repeated borrows in parallel closure
    let cols: Vec<ndarray::Array1<f64>> = (0..n).map(|i| x.column(i).to_owned()).collect();

    let mut corr = Array2::<f64>::eye(n);
    // Split by rows — each row slice is an independent mutable region
    corr.axis_iter_mut(Axis(0))
        .into_par_iter()
        .enumerate()
        .for_each(|(i, mut row)| {
            for j in 0..n {
                if i != j {
                    row[j] = Self::pearson_correlation(&cols[i], &cols[j]);
                }
            }
        });
    corr
}
```

Note: `axis_iter_mut` with `into_par_iter` requires `ndarray`'s `rayon` feature, which is already enabled in `Cargo.toml`.

- [ ] **Step 7: `cargo check`**

Run: `cargo check 2>&1 | grep -E "^error"`

Expected: no errors.

- [ ] **Step 8: Run feature_selection tests**

Run: `cargo test --lib preprocessing::feature_selection 2>&1 | tail -10`

Expected: all pass.

- [ ] **Step 9: Commit**

```bash
git add src/preprocessing/feature_selection.rs
git commit -m "perf: optimize CorrelationFilter O(n³)→O(n²), parallelize correlation matrix + RFE scores, single-pass discretize

NOTE: parallel reduction introduces non-deterministic float deltas (accepted trade-off)"
```

---

### Task 2: `src/preprocessing/encoder.rs` — Eliminate clone storm

**Files:**
- Modify: `src/preprocessing/encoder.rs`

Every `transform_*()` method clones the entire DataFrame and then calls `.with_column(...).clone()` in a loop — O(columns) full DataFrame allocations per call.

- [ ] **Step 1: Read current `transform_label`, `transform_target`, `transform_frequency`, `transform_binary`, `transform_hash`**

Run: `sed -n '209,435p' src/preprocessing/encoder.rs`

- [ ] **Step 2: Rewrite `transform_label()` to eliminate per-iteration clones**

The current code does `result = result.with_column(...).map_err(...)?.clone()` inside the loop — an extra full DataFrame clone after every `with_column` call. The fix: collect all new column values first, then do one `df.clone()` followed by N in-place `with_column` mutations with no intermediate clones.

The initial `df.clone()` is unavoidable for in-place column replacement (Polars requires an owned DataFrame). The saving is eliminating the N extra clones inside the loop (one per replaced column):

```rust
fn transform_label(&self, df: &DataFrame) -> Result<DataFrame> {
    // Compute all replacement columns first
    let mut new_cols: Vec<Series> = Vec::new();
    for (col_name, mapping) in &self.mappings {
        if let Ok(series) = df.column(col_name) {
            let ca = series.str().map_err(|e| AutoMLError::DataError(e.to_string()))?;
            let values: Vec<Option<i64>> = ca
                .into_iter()
                .map(|v| v.and_then(|s| mapping.get(s).map(|&i| i as i64)))
                .collect();
            new_cols.push(Series::new(col_name.clone().into(), values));
        }
    }
    // One clone + N in-place mutations — no clone after each with_column
    let mut result = df.clone();
    for col in new_cols {
        result.with_column(col).map_err(|e| AutoMLError::DataError(e.to_string()))?;
    }
    Ok(result)
}
```

- [ ] **Step 3: Rewrite `transform_target()` (lines 234–263) to collect-then-build**

`transform_target` uses `self.target_means` (not `self.mappings`) and emits `f64` values with a `global_mean` fallback. Apply the same collect-then-build pattern:

```rust
fn transform_target(&self, df: &DataFrame) -> Result<DataFrame> {
    let mut new_cols: Vec<Series> = Vec::new();
    for (col_name, means) in &self.target_means {
        if let Ok(series) = df.column(col_name) {
            let ca = series.str().map_err(|e| AutoMLError::DataError(e.to_string()))?;
            let global_mean: f64 = means.values().sum::<f64>() / means.len().max(1) as f64;
            let values: Vec<f64> = ca
                .into_iter()
                .map(|v| v.and_then(|s| means.get(s).copied()).unwrap_or(global_mean))
                .collect();
            new_cols.push(Series::new(col_name.clone().into(), values));
        }
    }
    let mut result = df.clone();
    for col in new_cols {
        result.with_column(col).map_err(|e| AutoMLError::DataError(e.to_string()))?;
    }
    Ok(result)
}
```

- [ ] **Step 4: Rewrite `transform_binary()` to accumulate columns, build once**

Binary encoding adds multiple new columns (one per bit) and drops the original. Collect all new bit columns first, then build the result:

```rust
fn transform_binary(&self, df: &DataFrame) -> Result<DataFrame> {
    let mut extra_cols: Vec<Series> = Vec::new();
    let mut cols_to_drop: Vec<String> = Vec::new();

    for (col_name, mapping) in &self.mappings {
        if let Ok(series) = df.column(col_name) {
            let ca = series.str().map_err(|e| AutoMLError::DataError(e.to_string()))?;
            let n_categories = mapping.len();
            if n_categories == 0 {
                cols_to_drop.push(col_name.clone());
                continue;
            }
            let n_bits = if n_categories <= 1 { 1 } else { (n_categories as f64).log2().ceil() as usize };

            // Collect all rows once into a Vec for multi-bit access
            let indices: Vec<Option<usize>> = ca
                .into_iter()
                .map(|v| v.and_then(|s| mapping.get(s).copied()))
                .collect();

            for bit_pos in 0..n_bits {
                let new_col_name = format!("{}_{}", col_name, bit_pos);
                let values: Vec<i32> = indices.iter()
                    .map(|opt| opt.map(|idx| ((idx >> bit_pos) & 1) as i32).unwrap_or(0))
                    .collect();
                extra_cols.push(Series::new(new_col_name.into(), values));
            }
            cols_to_drop.push(col_name.clone());
        }
    }

    // Build result: one clone + drop originals + add all new columns
    let mut result = df.clone();
    for col_name in &cols_to_drop {
        result = result.drop(col_name).map_err(|e| AutoMLError::DataError(e.to_string()))?;
    }
    // Build final DataFrame with new columns appended
    let mut all_cols: Vec<Series> = result.get_columns().iter().map(|c| c.as_materialized_series().clone()).collect();
    all_cols.extend(extra_cols);
    DataFrame::new(all_cols).map_err(|e| AutoMLError::DataError(e.to_string()))
}
```

- [ ] **Step 5: Rewrite `transform_hash()` — row-outer, component-inner**

```rust
fn transform_hash(&self, df: &DataFrame, n_components: usize) -> Result<DataFrame> {
    let mut extra_cols: Vec<Series> = Vec::new();
    let mut cols_to_drop: Vec<String> = Vec::new();

    for col_name in self.mappings.keys() {
        if let Ok(series) = df.column(col_name) {
            let ca = series.str().map_err(|e| AutoMLError::DataError(e.to_string()))?;

            // Row-outer, component-inner: each row read once for all components
            let mut component_values: Vec<Vec<f64>> = vec![Vec::with_capacity(ca.len()); n_components];
            for opt_val in ca.into_iter() {
                for comp_idx in 0..n_components {
                    let v = opt_val
                        .map(|s| {
                            let hash = self.hash_string(s, comp_idx);
                            if hash % 2 == 0 { 1.0 } else { -1.0 }
                        })
                        .unwrap_or(0.0);
                    component_values[comp_idx].push(v);
                }
            }
            for (comp_idx, values) in component_values.into_iter().enumerate() {
                let new_col_name = format!("{}_{}", col_name, comp_idx);
                extra_cols.push(Series::new(new_col_name.into(), values));
            }
            cols_to_drop.push(col_name.clone());
        }
    }

    let mut result = df.clone();
    for col_name in &cols_to_drop {
        result = result.drop(col_name).map_err(|e| AutoMLError::DataError(e.to_string()))?;
    }
    let mut all_cols: Vec<Series> = result.get_columns().iter().map(|c| c.as_materialized_series().clone()).collect();
    all_cols.extend(extra_cols);
    DataFrame::new(all_cols).map_err(|e| AutoMLError::DataError(e.to_string()))
}
```

- [ ] **Step 6: Add `par_iter` to `transform_frequency` column loop**

The frequency map and value generation per column are independent. Wrap the column loop in `par_iter`. Note: collect `raw_vals: Vec<Option<String>>` once before building the freq_map — do not call `ca.into_iter()` twice, as re-iteration behavior depends on the Polars version:

```rust
fn transform_frequency(&self, df: &DataFrame) -> Result<DataFrame> {
    use rayon::prelude::*;

    let new_cols: Vec<Result<Series>> = self.mappings.par_iter()
        .filter_map(|(col_name, _mapping)| {
            df.column(col_name).ok().map(|series| {
                let ca = series.str().map_err(|e| AutoMLError::DataError(e.to_string()))?;
                let total = ca.len() as f64;
                // Collect values once to allow two-pass logic safely
                let raw_vals: Vec<Option<String>> = ca.into_iter()
                    .map(|v| v.map(|s| s.to_string()))
                    .collect();
                let mut freq_map: HashMap<String, f64> = HashMap::new();
                for val in raw_vals.iter().flatten() {
                    *freq_map.entry(val.clone()).or_insert(0.0) += 1.0;
                }
                for v in freq_map.values_mut() { *v /= total; }
                let values: Vec<f64> = raw_vals.iter()
                    .map(|v| v.as_deref().and_then(|s| freq_map.get(s).copied()).unwrap_or(0.0))
                    .collect();
                Ok(Series::new(col_name.clone().into(), values))
            })
        })
        .collect();

    let new_cols: Vec<Series> = new_cols.into_iter().collect::<Result<Vec<_>>>()?;
    let mut result = df.clone();
    for col in new_cols {
        result.with_column(col).map_err(|e| AutoMLError::DataError(e.to_string()))?;
    }
    Ok(result)
}
```

- [ ] **Step 7: `cargo check`**

Run: `cargo check 2>&1 | grep -E "^error"`

Expected: no errors.

- [ ] **Step 8: Run encoder tests**

Run: `cargo test --lib preprocessing::encoder 2>&1 | tail -10`

Expected: all pass.

- [ ] **Step 9: Commit**

```bash
git add src/preprocessing/encoder.rs
git commit -m "perf(encoder): eliminate per-column DataFrame clone storm, row-outer hash encoding, parallelize frequency encoding"
```

---

### Task 3: `src/preprocessing/scaler.rs` — Remove residual clone + optimize quantile

**Files:**
- Modify: `src/preprocessing/scaler.rs`

`transform()` at line 104 already collects scaled columns into `Vec<Series>`, but still ends with `df.clone()` + N `with_column` mutations. `compute_params()` for `Robust` scaler calls `quantile()` twice.

- [ ] **Step 1: Read `transform()` and `compute_params()` for `Robust`**

Run: `sed -n '103,207p' src/preprocessing/scaler.rs`

- [ ] **Step 2: Replace `transform()` to build without `df.clone()`**

```rust
pub fn transform(&self, df: &DataFrame) -> Result<DataFrame> {
    if !self.is_fitted {
        return Err(AutoMLError::ModelNotFitted);
    }

    // Parallel: scale all columns independently
    let scaled_cols: Vec<Series> = self.params.par_iter()
        .filter_map(|(col_name, params)| {
            df.column(col_name).ok().map(|column| {
                let series = column.as_materialized_series();
                self.scale_series(series, params)
            })
        })
        .collect::<Result<Vec<Series>>>()?;

    // Build result without full DataFrame clone:
    // keep all columns from df, replace the scaled ones
    let scaled_names: std::collections::HashSet<&str> =
        self.params.keys().map(|s| s.as_str()).collect();
    let mut all_cols: Vec<Series> = df.get_columns()
        .iter()
        .filter(|c| !scaled_names.contains(c.name().as_str()))
        .map(|c| c.as_materialized_series().clone())
        .collect();
    all_cols.extend(scaled_cols);
    // Restore original column order
    let orig_order: Vec<&str> = df.get_column_names().into_iter().map(|s| s.as_str()).collect();
    let pos: std::collections::HashMap<&str, usize> = orig_order.iter()
        .enumerate()
        .map(|(i, &s)| (s, i))
        .collect();
    all_cols.sort_by_key(|c| pos.get(c.name().as_str()).copied().unwrap_or(usize::MAX));
    DataFrame::new(all_cols).map_err(|e| AutoMLError::DataError(e.to_string()))
}
```

Also replace `inverse_transform()` (lines 136–157) with the same build-without-clone pattern (this extends the spec's scope to cover the symmetric reverse path — same structural fix, same rationale):

```rust
pub fn inverse_transform(&self, df: &DataFrame) -> Result<DataFrame> {
    if !self.is_fitted {
        return Err(AutoMLError::ModelNotFitted);
    }

    // Parallel: unscale all columns independently
    let unscaled_cols: Vec<Series> = self.params.par_iter()
        .filter_map(|(col_name, params)| {
            df.column(col_name).ok().map(|column| {
                let series = column.as_materialized_series();
                self.unscale_series(series, params)
            })
        })
        .collect::<Result<Vec<Series>>>()?;

    // Build result without full DataFrame clone
    let scaled_names: std::collections::HashSet<&str> =
        self.params.keys().map(|s| s.as_str()).collect();
    let mut all_cols: Vec<Series> = df.get_columns()
        .iter()
        .filter(|c| !scaled_names.contains(c.name().as_str()))
        .map(|c| c.as_materialized_series().clone())
        .collect();
    all_cols.extend(unscaled_cols);
    // Restore original column order
    let orig_order: Vec<&str> = df.get_column_names().into_iter().map(|s| s.as_str()).collect();
    let pos: std::collections::HashMap<&str, usize> = orig_order.iter()
        .enumerate()
        .map(|(i, &s)| (s, i))
        .collect();
    all_cols.sort_by_key(|c| pos.get(c.name().as_str()).copied().unwrap_or(usize::MAX));
    DataFrame::new(all_cols).map_err(|e| AutoMLError::DataError(e.to_string()))
}
```

- [ ] **Step 3: Replace `Robust` quantile computation with sort-once**

Replace the `Robust` arm inside `compute_params()` (lines 182–190):

```rust
ScalerType::Robust => {
    // Sort once, read Q1 and Q3 from indices instead of two quantile() calls
    let mut vals: Vec<f64> = ca.into_no_null_iter().collect();
    vals.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let n = vals.len();
    let median = if n % 2 == 0 {
        (vals[n / 2 - 1] + vals[n / 2]) / 2.0
    } else {
        vals[n / 2]
    };
    let q1 = vals[n / 4];
    let q3 = vals[3 * n / 4];
    let iqr = q3 - q1;
    Ok(ScalerParams {
        center: median,
        scale: if iqr.abs() < SCALE_EPSILON { 1.0 } else { iqr },
    })
}
```

- [ ] **Step 4: `cargo check`**

Run: `cargo check 2>&1 | grep -E "^error"`

Expected: no errors.

- [ ] **Step 5: Run scaler tests**

Run: `cargo test --lib preprocessing::scaler 2>&1 | tail -10`

Expected: all pass.

- [ ] **Step 6: Commit**

```bash
git add src/preprocessing/scaler.rs
git commit -m "perf(scaler): build transformed DataFrame without full clone, parallel column scaling, sort-once for robust quantile

NOTE: robust Q1/Q3 now uses nearest-rank method (vals[n/4], vals[3n/4]) instead of
Polars linear-interpolation quantile() — tiny numeric difference on small datasets"
```

---

### Task 4: `src/preprocessing/pipeline.rs` — Imputer chain borrow + O(n) cast ordering

**Files:**
- Modify: `src/preprocessing/pipeline.rs`

`fit()` (lines 174–194) clones `df` when no imputer is present. `cast_numeric_to_f64()` (line 136) sorts columns with O(n log n) `sort_by_key` where direct-index placement gives O(n).

- [ ] **Step 1: Read `fit()` imputer chain and `cast_numeric_to_f64()`**

Run: `sed -n '105,200p' src/preprocessing/pipeline.rs`

- [ ] **Step 2: Fix imputer chain in `fit()` to avoid unnecessary clone**

Replace the two imputer-chain blocks (lines 174–197):

```rust
// Fit scaler for numeric columns — impute first if needed
if !self.numeric_columns.is_empty() {
    let mut scaler = Scaler::new(self.config.scaler_type.clone());
    let cols: Vec<&str> = self.numeric_columns.iter().map(|s| s.as_str()).collect();

    // Borrow imputed data; only materialise an owned DataFrame when transform occurs
    let imputed_owned: Option<DataFrame>;
    let imputed_ref: &DataFrame = if let Some(ref imputer) = self.numeric_imputer {
        imputed_owned = Some(imputer.transform(df)?);
        imputed_owned.as_ref().unwrap()
    } else {
        df  // borrow without cloning
    };

    scaler.fit(imputed_ref, &cols)?;
    self.scaler = Some(scaler);
}

// Fit encoder for categorical columns — impute first if needed
if !self.categorical_columns.is_empty() {
    let mut encoder = Encoder::new(self.config.encoder_type.clone());
    let cols: Vec<&str> = self.categorical_columns.iter().map(|s| s.as_str()).collect();

    let imputed_owned: Option<DataFrame>;
    let imputed_ref: &DataFrame = if let Some(ref imputer) = self.categorical_imputer {
        imputed_owned = Some(imputer.transform(df)?);
        imputed_owned.as_ref().unwrap()
    } else {
        df
    };

    encoder.fit(imputed_ref, &cols)?;
    self.encoder = Some(encoder);
}
```

- [ ] **Step 3: Replace `cast_numeric_to_f64()` sort with direct-index placement**

Note: `cast_numeric_to_f64` works with `Column` (Polars' lazy-vs-eager wrapper), not `Series`. The code uses `Vec<Option<Column>>` and `Vec<Column>`. This is consistent with the Polars version used in this project. `cargo check` will catch any type mismatch immediately.

Replace the final accumulation/sort block in `cast_numeric_to_f64` — read the function in Step 1 to find the exact lines (the `sort_by_key` call is the last statement before `DataFrame::new`). The replacement uses two placement passes instead of one sorted collection — this is correct and produces the same output order:

```rust
// Direct-index placement: O(n) instead of O(n log n) sort
let orig_order: Vec<&str> = df.get_column_names().into_iter().map(|s| s.as_str()).collect();
let pos_map: std::collections::HashMap<&str, usize> = orig_order.iter()
    .enumerate()
    .map(|(i, &s)| (s, i))
    .collect();

let mut ordered: Vec<Option<Column>> = vec![None; orig_order.len()];
// Place non-cast columns
for col in df.get_columns() {
    if !cast_names.contains(col.name().as_str()) {
        if let Some(&idx) = pos_map.get(col.name().as_str()) {
            ordered[idx] = Some(col.clone());
        }
    }
}
// Place cast columns
for col in columns_to_cast {
    if let Some(&idx) = pos_map.get(col.name().as_str()) {
        ordered[idx] = Some(col);
    }
}
let all_cols: Vec<Column> = ordered.into_iter().flatten().collect();
DataFrame::new(all_cols).map_err(|e| AutoMLError::DataError(e.to_string()))
```

- [ ] **Step 4: Read `compute_statistics()` before editing**

Verify the actual method signatures used inside `compute_statistics()`:

Run: `sed -n '558,580p' src/preprocessing/pipeline.rs && grep -n "fn from_numeric_series\|fn from_categorical_series" src/preprocessing/*.rs`

Expected: `FeatureStats::from_numeric_series` and `FeatureStats::from_categorical_series` are defined. Confirm the exact function names match before the next step.

- [ ] **Step 5: Parallelize `compute_statistics()` using `par_iter` over column names**

`compute_statistics()` (lines 558–578) has two sequential loops over `self.numeric_columns` and `self.categorical_columns`. Replace both loops with parallel collection then sequential insertion into `self.feature_stats`:

```rust
fn compute_statistics(&mut self, df: &DataFrame) -> Result<()> {
    use rayon::prelude::*;
    self.feature_stats.clear();

    // Combine numeric and categorical column names, tagged by type
    let all_columns: Vec<(&str, bool)> = self.numeric_columns.iter()
        .map(|s| (s.as_str(), true))
        .chain(self.categorical_columns.iter().map(|s| (s.as_str(), false)))
        .collect();

    // Collect stats in parallel — each column is independent
    let stats: Vec<Result<(String, FeatureStats)>> = all_columns.par_iter()
        .filter_map(|(col_name, is_numeric)| {
            df.column(col_name).ok().map(|column| {
                let series = column.as_materialized_series();
                let fs = if *is_numeric {
                    FeatureStats::from_numeric_series(col_name, series)?
                } else {
                    FeatureStats::from_categorical_series(col_name, series)?
                };
                Ok((col_name.to_string(), fs))
            })
        })
        .collect();

    // Apply to self sequentially (cannot mutate self in parallel)
    for result in stats {
        let (name, stat) = result?;
        self.feature_stats.insert(name, stat);
    }
    Ok(())
}
```

- [ ] **Step 6: `cargo check`**

Run: `cargo check 2>&1 | grep -E "^error"`

Expected: no errors.

- [ ] **Step 7: Run pipeline tests**

Run: `cargo test --lib preprocessing::pipeline 2>&1 | tail -10`

Expected: all pass.

- [ ] **Step 8: Commit**

```bash
git add src/preprocessing/pipeline.rs
git commit -m "perf(pipeline): borrow DataFrame through imputer chain, O(n) cast ordering via index map, parallel compute_statistics"
```

---

## Chunk 2: Feature Engineering, Server, and Final Verification (Tasks 5–8)

---

### Task 5: `src/feature_engineering/interactions.rs` — Cache-hostile loop + correlation precompute

**Files:**
- Modify: `src/feature_engineering/interactions.rs`

`transform()` (line 181) uses element-wise `result[[i,j]]` access, destroying cache locality. `AutoCrossing::correlation()` (line 283) recomputes column means/stds per pair call.

- [ ] **Step 1: Read `transform()`, `get_feature_names()`, `AutoCrossing::correlation()` and `AutoCrossing::fit()`**

Run: `sed -n '181,340p' src/feature_engineering/interactions.rs`

- [ ] **Step 2: Replace `transform()` with column-slice assignments**

```rust
fn transform(&self, x: &Array2<f64>) -> Result<Array2<f64>> {
    use ndarray::{s, Axis};
    use rayon::prelude::*;

    let crossings = self.crossings.as_ref().ok_or_else(|| {
        AutoMLError::ValidationError("Transformer not fitted".to_string())
    })?;

    let n_original = if self.include_original { x.ncols() } else { 0 };
    let n_output = n_original + crossings.len();
    let mut result = Array2::zeros((x.nrows(), n_output));

    // Copy original columns with column-slice assignment (cache-friendly)
    if self.include_original {
        for j in 0..x.ncols() {
            result.slice_mut(s![.., j]).assign(&x.slice(s![.., j]));
        }
    }

    // Compute interaction columns in parallel
    // Each crossing writes to a distinct output column index
    result
        .axis_iter_mut(Axis(1))
        .into_par_iter()
        .skip(n_original)
        .zip(crossings.par_iter())
        .for_each(|(mut out_col, crossing)| {
            let col_a = x.slice(s![.., crossing.feature_a]);
            let col_b = x.slice(s![.., crossing.feature_b]);
            for (o, (&a, &b)) in out_col.iter_mut().zip(col_a.iter().zip(col_b.iter())) {
                *o = crossing.interaction.apply(a, b);
            }
        });

    Ok(result)
}
```

Note: `axis_iter_mut(...).into_par_iter().skip(n)` skips the original columns. Verify the `skip` + `zip` alignment is correct for the specific ndarray version in use.

- [ ] **Step 3: Replace `AutoCrossing::correlation()` with precomputed-stats version**

Remove the per-call `Vec` allocations and mean/std recomputation. Add a helper that takes precomputed stats:

```rust
fn correlation_from_stats(
    x: &Array2<f64>,
    col_a: usize, mean_a: f64, std_a: f64,
    col_b: usize, mean_b: f64, std_b: f64,
) -> f64 {
    if std_a < 1e-10 || std_b < 1e-10 { return 0.0; }
    let n = x.nrows() as f64;
    let cov: f64 = x.column(col_a).iter()
        .zip(x.column(col_b).iter())
        .map(|(&a, &b)| (a - mean_a) * (b - mean_b))
        .sum::<f64>() / n;
    cov / (std_a * std_b)
}
```

- [ ] **Step 4: Update `AutoCrossing::fit()` to precompute all column means/stds once**

At the start of `fit()`, compute means and stds for all columns before the pairwise loop:

```rust
fn fit(&mut self, x: &Array2<f64>) -> Result<()> {
    let n_features = x.ncols();
    let n = x.nrows() as f64;

    // Precompute means and stds once
    let means: Vec<f64> = (0..n_features)
        .map(|i| x.column(i).iter().sum::<f64>() / n)
        .collect();
    let stds: Vec<f64> = (0..n_features)
        .map(|i| {
            let m = means[i];
            (x.column(i).iter().map(|&v| (v - m).powi(2)).sum::<f64>() / n).sqrt()
        })
        .collect();

    // Compute correlations using precomputed stats
    let mut pairs = Vec::new();
    for i in 0..n_features {
        for j in (i + 1)..n_features {
            let corr = Self::correlation_from_stats(
                x, i, means[i], stds[i], j, means[j], stds[j],
            ).abs();
            if corr >= self.correlation_threshold {
                pairs.push((i, j, corr));
            }
        }
    }

    // Sort by correlation descending, take top max_crossings
    pairs.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap_or(std::cmp::Ordering::Equal));
    let selected: Vec<(usize, usize)> = pairs.into_iter()
        .take(self.max_crossings)
        .map(|(i, j, _)| (i, j))
        .collect();

    // ... rest of fit() building the generator ...
```

- [ ] **Step 5: Fix `get_feature_names()` — use `extend_from_slice`**

Replace line 220 `names.extend(input_names.clone())` with `names.extend_from_slice(input_names)`.

- [ ] **Step 6: `cargo check`**

Run: `cargo check 2>&1 | grep -E "^error"`

Expected: no errors.

- [ ] **Step 7: Run interactions tests**

Run: `cargo test --lib feature_engineering 2>&1 | tail -10`

Expected: all pass.

- [ ] **Step 8: Commit**

```bash
git add src/feature_engineering/interactions.rs
git commit -m "perf(interactions): column-slice assignments for cache locality, precompute means/stds, parallel interaction columns"
```

---

### Task 6: `src/autopipeline/composer.rs` — Single-pass partition + HashSet membership

**Files:**
- Modify: `src/autopipeline/composer.rs`

`compose()` partitions columns into `high_missing` / `remaining_missing` using two filtered passes. `Vec::contains()` called for membership on `drop_columns`, `numeric_columns`, `categorical_columns`.

- [ ] **Step 1: Read the missing-value handling block and categorical filter block**

Run: `sed -n '195,280p' src/autopipeline/composer.rs`

- [ ] **Step 2: Replace repeated filtering with single-pass partition**

At the start of `compose()`, build `HashSet`s from the struct's Vec fields once:

```rust
use std::collections::HashSet;

let drop_set: HashSet<usize> = schema.drop_columns.iter().copied().collect();
let numeric_set: HashSet<usize> = schema.numeric_columns.iter().copied().collect();
let categorical_set: HashSet<usize> = schema.categorical_columns.iter().copied().collect();
```

Replace the multi-pass column-with-missing filtering (lines 204–264) with a single-pass partition:

```rust
// Single pass: partition columns with missing values into buckets
let (high_missing, remaining_missing): (Vec<usize>, Vec<usize>) = schema
    .columns
    .iter()
    .enumerate()
    .filter(|(i, col)| col.n_missing > 0 && !drop_set.contains(i))
    .partition(|(i, col)| col.pct_missing > self.config.missing_threshold);

let high_missing: Vec<usize> = high_missing.into_iter().map(|(i, _)| i).collect();
let remaining_missing: Vec<usize> = remaining_missing.into_iter().map(|(i, _)| i).collect();
```

Replace all subsequent `schema.drop_columns.contains(i)` / `schema.numeric_columns.contains(&i)` / `schema.categorical_columns.contains(&i)` calls with `drop_set.contains(i)` / `numeric_set.contains(&i)` / `categorical_set.contains(&i)`.

- [ ] **Step 3: `cargo check`**

Run: `cargo check 2>&1 | grep -E "^error"`

Expected: no errors.

- [ ] **Step 4: Run autopipeline tests**

Run: `cargo test --lib autopipeline 2>&1 | tail -10`

Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add src/autopipeline/composer.rs
git commit -m "perf(composer): single-pass partition for missing-value buckets, HashSet for O(1) membership checks"
```

---

### Task 7: `src/server/state.rs` — DashMap migration + O(n) eviction

**Files:**
- Modify: `src/server/state.rs`

Both `train_engines` and `models` are `RwLock<HashMap<String, ...>>`. DashMap is already in `Cargo.toml` (line 71). `evict_stale_entries()` uses O(n log n) sort to find oldest jobs.

- [ ] **Step 1: Read `AppState` struct, `evict_stale_entries`, and all access sites for both `train_engines` and `models`**

Run: `grep -n "train_engines\|models" src/server/state.rs src/server/handlers.rs | head -60`

- [ ] **Step 2: Change `train_engines` and `models` fields to `DashMap`**

In `state.rs`, add the import and change both fields:

```rust
use dashmap::DashMap;

// In AppState struct — change both:
pub train_engines: DashMap<String, TrainEngine>,
pub models: DashMap<String, TrainedModel>,   // (confirm actual value type from the struct)

// In AppState::new() — change both:
train_engines: DashMap::new(),
models: DashMap::new(),
```

Note: confirm the actual type parameter for `models` in Step 1 before applying.

- [ ] **Step 3: Update all `train_engines` and `models` access sites in `state.rs` and `handlers.rs`**

DashMap uses synchronous access (no `.await`). Replace patterns for both fields:

| Old | New |
|-----|-----|
| `self.train_engines.write().await.insert(k, v)` | `self.train_engines.insert(k, v)` |
| `self.train_engines.read().await.get(k)` | `self.train_engines.get(k)` |
| `self.train_engines.write().await.remove(k)` | `self.train_engines.remove(k)` |
| `self.train_engines.read().await.len()` | `self.train_engines.len()` |
| `self.train_engines.read().await.iter()` | `self.train_engines.iter()` |
| `self.models.write().await.insert(k, v)` | `self.models.insert(k, v)` |
| `self.models.read().await.get(k)` | `self.models.get(k)` |
| `self.models.write().await.remove(k)` | `self.models.remove(k)` |
| `self.models.read().await.len()` | `self.models.len()` |

- [ ] **Step 4: Update `evict_stale_entries()` to use `min_by_key` for jobs (O(n) instead of O(n log n) sort)**

```rust
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
                // Partial sort: only bring the `to_remove` oldest to front
                removable.select_nth_unstable_by_key(to_remove.min(removable.len().saturating_sub(1)), |(_, t)| *t);
                for (id, _) in removable.into_iter().take(to_remove) {
                    jobs.remove(&id);
                }
            }
        }
    }

    // Evict train_engines: DashMap — no lock needed
    if self.train_engines.len() > Self::MAX_TRAIN_ENGINES {
        let to_remove = self.train_engines.len() - Self::MAX_TRAIN_ENGINES;
        let keys: Vec<String> = self.train_engines.iter()
            .take(to_remove)
            .map(|entry| entry.key().clone())
            .collect();
        for key in keys {
            self.train_engines.remove(&key);
        }
    }

    // Truncate completed studies
    let mut studies = self.completed_studies.write().await;
    if studies.len() > Self::MAX_COMPLETED_STUDIES {
        let drain_count = studies.len() - Self::MAX_COMPLETED_STUDIES;
        studies.drain(..drain_count);
    }
}
```

Note: `select_nth_unstable_by_key` is available on `Vec` (it's `slice::select_nth_unstable_by_key`). If the `to_remove` value equals or exceeds `removable.len()`, guard with `.min(removable.len().saturating_sub(1))`.

- [ ] **Step 5: `cargo check`**

Run: `cargo check 2>&1 | grep -E "^error"`

Expected: no errors.

- [ ] **Step 6: Run state-related tests**

Run: `cargo test --lib server 2>&1 | tail -10`

Expected: all pass.

- [ ] **Step 7: Commit**

```bash
git add src/server/state.rs
git commit -m "perf(state): migrate train_engines to DashMap (no RwLock), O(n) job eviction via select_nth_unstable"
```

---

### Task 8: `src/server/handlers.rs` — Single-pass skew/kurt + clone-then-drop lock

**Files:**
- Modify: `src/server/handlers.rs`

`analyze_data()` and `get_data_quality()` both compute skewness and kurtosis with two separate iterator passes. Both hold `state.current_data.read()` for the entire CPU-intensive function body.

- [ ] **Step 1: Find the skew/kurt double-pass in `analyze_data`**

Run: `grep -n "sum3\|sum4\|into_no_null_iter" src/server/handlers.rs | head -20`

- [ ] **Step 2: Merge skewness and kurtosis into single pass in `analyze_data`**

Replace the two-pass block (around line 341–352):

```rust
// Single pass computing both sum3 and sum4
let (skewness, kurtosis) = if let Some(ca) = ca {
    let n = ca.len() as f64 - ca.null_count() as f64;
    if n > 3.0 && std > 0.0 {
        let (sum3, sum4) = ca.into_no_null_iter().fold((0.0_f64, 0.0_f64), |(s3, s4), v| {
            let z = (v - mean) / std;
            let z3 = z * z * z;
            (s3 + z3, s4 + z3 * z)
        });
        let skew = sum3 * n / ((n - 1.0) * (n - 2.0));
        let kurt = n * (n + 1.0) / ((n - 1.0) * (n - 2.0) * (n - 3.0)) * sum4
            - 3.0 * (n - 1.0).powi(2) / ((n - 2.0) * (n - 3.0));
        (skew, kurt)
    } else { (0.0, 0.0) }
} else { (0.0, 0.0) };
```

- [ ] **Step 3: Apply the same single-pass fix in `get_data_quality`**

Find the equivalent block in `get_data_quality` (around line 6099). Apply the same fold pattern.

- [ ] **Step 4: Narrow the lock scope in `analyze_data` — clone-then-drop**

Replace lines 267–277:

```rust
pub async fn analyze_data(
    State(state): State<Arc<AppState>>,
) -> Result<Json<serde_json::Value>> {
    // Clone the DataFrame and immediately drop the read lock.
    // This prevents blocking writers for the full duration of CPU-intensive analysis.
    let df = {
        let data = state.current_data.read().await;
        data.as_ref()
            .ok_or_else(|| ServerError::NotFound("No data loaded".to_string()))?
            .clone()   // cheap Arc-based clone of the DataFrame
    };
    // `data` guard dropped here — lock released

    let n_rows = df.height();
    let n_cols = df.width();
    let memory_bytes = df.estimated_size();
    // ... rest of function uses `df` directly, not `data.as_ref()` ...
```

- [ ] **Step 5: Apply the same clone-then-drop to `get_data_quality`**

Find `get_data_quality` (around line 5898). Apply the same pattern: clone `df` from the read lock immediately, drop the guard, proceed with `df`.

- [ ] **Step 6: `cargo check`**

Run: `cargo check 2>&1 | grep -E "^error"`

Expected: no errors.

- [ ] **Step 7: Run handlers tests**

Run: `cargo test --test test_web_ui 2>&1 | tail -10`

Expected: all pass.

- [ ] **Step 8: Commit**

```bash
git add src/server/handlers.rs
git commit -m "perf(handlers): single-pass skew+kurt computation, clone-then-drop lock in analyze_data and get_data_quality"
```

---

## Final Verification

- [ ] **Step 1: Full release build**

Run: `cargo build --release 2>&1 | grep -E "^error"`

Expected: no errors.

- [ ] **Step 2: Full test suite**

Run: `cargo test 2>&1 | grep -E "test result"`

Expected: all test suites report `0 failed`.

- [ ] **Step 3: Benchmark comparison**

Run: `cargo bench 2>&1 | grep -E "time:|thrpt:"`

Compare against baseline recorded before starting Task 1. Key metrics to check:
- `preprocessing/correlation_filter` — expect significant speedup (O(n³) → O(n²))
- `preprocessing/scaler_transform` — expect modest improvement (no full clone)
- `preprocessing/encoder_transform` — expect improvement (no per-column clone)

- [ ] **Step 4: Final commit + push**

```bash
git push origin main
```

---

## Benchmark Results — 2026-03-15

Recorded after all 8 optimization tasks completed and pushed to `main`.

> **Note:** No pre-optimization baseline was available (pre-existing compile errors in bench files prevented running benchmarks before the changes). Bench files fixed as part of this session (`Vec<Series>` → `Vec<Column>` for Polars API, `engine.fit()` return value handling).

### Preprocessing — `fit_transform` (DataPreprocessor, 10 cols, StandardScaler)

| rows    | time (median) | range               |
|---------|--------------|---------------------|
| 1,000   | 2.942 ms     | [2.466 ms – 3.464 ms] |
| 10,000  | 8.657 ms     | [7.834 ms – 9.598 ms] |
| 100,000 | 39.069 ms    | [36.028 ms – 42.750 ms] |

### Scaling — individual scalers (DataPreprocessor, 10,000 rows, 20 cols)

| scaler   | time (median) | range                 |
|----------|--------------|------------------------|
| Standard | 14.152 ms    | [12.678 ms – 15.902 ms] |
| MinMax   | 11.709 ms    | [10.990 ms – 12.484 ms] |
| Robust   | 18.863 ms    | [17.695 ms – 20.113 ms] |

### Training — `fit` (TrainEngine, regression, 10 features)

| rows   | time (median) | range                   |
|--------|--------------|--------------------------|
| 1,000  | 171.13 µs    | [166.17 µs – 181.47 µs] |
| 5,000  | 1.1038 ms    | [879.76 µs – 1.3709 ms] |
| 10,000 | 1.6895 ms    | [1.6382 ms – 1.7233 ms] |

### Prediction — `predict` (trained on 5,000 rows, 10 features)

| rows   | time (median) | range                    |
|--------|--------------|---------------------------|
| 100    | 12.241 µs    | [11.474 µs – 13.334 µs]  |
| 1,000  | 100.49 µs    | [96.738 µs – 104.56 µs]  |
| 10,000 | 1.0734 ms    | [998.00 µs – 1.1726 ms]  |
