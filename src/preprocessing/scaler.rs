//! Feature scaling implementations

use crate::error::{AutoMLError, Result};
use polars::prelude::*;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Minimum scale value to prevent numerical instability.
/// Any computed scale (std, range, IQR, max_abs) below this threshold
/// is treated as zero-variance and replaced with 1.0.
const SCALE_EPSILON: f64 = 1e-12;

/// Type of scaler to use
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ScalerType {
    /// Standard scaling (z-score normalization): (x - mean) / std
    Standard,
    /// Min-Max scaling: (x - min) / (max - min), mapped to feature_range
    MinMax,
    /// Robust scaling using median and IQR
    Robust,
    /// Max absolute scaling: x / max(|x|)
    MaxAbs,
    /// No scaling
    None,
}

/// Parameters for a fitted scaler
#[derive(Debug, Clone, Serialize, Deserialize)]
struct ScalerParams {
    center: f64,    // mean, min, or median
    scale: f64,     // std, range, or IQR
}

/// Feature scaler with configurable output range and clipping
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Scaler {
    scaler_type: ScalerType,
    params: HashMap<String, ScalerParams>,
    is_fitted: bool,
    /// Output range for MinMax scaler: (min, max). Default: (0.0, 1.0)
    feature_range: (f64, f64),
    /// Whether to clip transformed values to the expected output range.
    /// For MinMax: clips to feature_range. For MaxAbs: clips to [-1, 1].
    /// For Standard/Robust: no clipping (unbounded by nature).
    clip: bool,
}

impl Scaler {
    /// Create a new scaler
    pub fn new(scaler_type: ScalerType) -> Self {
        Self {
            scaler_type,
            params: HashMap::new(),
            is_fitted: false,
            feature_range: (0.0, 1.0),
            clip: false,
        }
    }

    /// Create a new scaler with a custom output range for MinMax scaling.
    ///
    /// # Panics
    /// Panics if `range_min >= range_max`.
    pub fn with_feature_range(mut self, range_min: f64, range_max: f64) -> Self {
        assert!(
            range_min < range_max,
            "feature_range min ({}) must be less than max ({})",
            range_min,
            range_max
        );
        self.feature_range = (range_min, range_max);
        self
    }

    /// Enable or disable output clipping.
    ///
    /// When enabled, transformed values are clamped to the expected output
    /// range so that unseen data outside the training distribution doesn't
    /// produce extreme scaled values.
    pub fn with_clip(mut self, clip: bool) -> Self {
        self.clip = clip;
        self
    }

    /// Fit the scaler to the data
    pub fn fit(&mut self, df: &DataFrame, columns: &[&str]) -> Result<&mut Self> {
        for col_name in columns {
            let column = df
                .column(col_name)
                .map_err(|_| AutoMLError::FeatureNotFound(col_name.to_string()))?;
            let series = column.as_materialized_series();

            let params = self.compute_params(series)?;
            self.params.insert(col_name.to_string(), params);
        }

        self.is_fitted = true;
        Ok(self)
    }

    /// Transform the data
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
        let mut all_cols: Vec<Column> = df.get_columns()
            .iter()
            .filter(|c| !scaled_names.contains(c.name().as_str()))
            .cloned()
            .collect();
        all_cols.extend(scaled_cols.into_iter().map(|s| s.into()));
        // Restore original column order
        let orig_order: Vec<&str> = df.get_column_names().into_iter().map(|s| s.as_str()).collect();
        let pos: std::collections::HashMap<&str, usize> = orig_order.iter()
            .enumerate()
            .map(|(i, &s)| (s, i))
            .collect();
        all_cols.sort_by_key(|c| pos.get(c.name().as_str()).copied().unwrap_or(usize::MAX));
        DataFrame::new(all_cols).map_err(|e| AutoMLError::DataError(e.to_string()))
    }

    /// Fit and transform in one step
    pub fn fit_transform(&mut self, df: &DataFrame, columns: &[&str]) -> Result<DataFrame> {
        self.fit(df, columns)?;
        self.transform(df)
    }

    /// Inverse transform the data
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
        let mut all_cols: Vec<Column> = df.get_columns()
            .iter()
            .filter(|c| !scaled_names.contains(c.name().as_str()))
            .cloned()
            .collect();
        all_cols.extend(unscaled_cols.into_iter().map(|s| s.into()));
        // Restore original column order
        let orig_order: Vec<&str> = df.get_column_names().into_iter().map(|s| s.as_str()).collect();
        let pos: std::collections::HashMap<&str, usize> = orig_order.iter()
            .enumerate()
            .map(|(i, &s)| (s, i))
            .collect();
        all_cols.sort_by_key(|c| pos.get(c.name().as_str()).copied().unwrap_or(usize::MAX));
        DataFrame::new(all_cols).map_err(|e| AutoMLError::DataError(e.to_string()))
    }

    fn compute_params(&self, series: &Series) -> Result<ScalerParams> {
        let ca = series
            .f64()
            .map_err(|e| AutoMLError::DataError(e.to_string()))?;

        match self.scaler_type {
            ScalerType::Standard => {
                // Polars' mean()/std() propagate float NaN (unlike null, which is
                // skipped), so a single NaN would otherwise poison the whole-column
                // statistic. Explicitly drop NaNs before aggregating.
                let clean = ca
                    .filter(&ca.is_not_nan())
                    .map_err(|e| AutoMLError::DataError(e.to_string()))?;
                let mean = clean.mean().unwrap_or(0.0);
                let std = clean.std(1).unwrap_or(1.0);
                Ok(ScalerParams {
                    center: mean,
                    scale: if std.abs() < SCALE_EPSILON { 1.0 } else { std },
                })
            }
            ScalerType::MinMax => {
                // Same NaN-propagation issue as Standard above.
                let clean = ca
                    .filter(&ca.is_not_nan())
                    .map_err(|e| AutoMLError::DataError(e.to_string()))?;
                let min = clean.min().unwrap_or(0.0);
                let max = clean.max().unwrap_or(1.0);
                let range = max - min;
                Ok(ScalerParams {
                    center: min,
                    scale: if range.abs() < SCALE_EPSILON { 1.0 } else { range },
                })
            }
            ScalerType::Robust => {
                // Order statistics for Q1/median/Q3: uses nearest-rank (vals[n/4],
                // vals[3n/4]) instead of Polars QuantileMethod::Linear — small
                // numeric difference on datasets with n < ~20, negligible otherwise.
                // `into_no_null_iter` only skips Polars null, not float NaN, so
                // NaN must be filtered explicitly to avoid poisoning the selection
                // (and therefore the median/IQR) below.
                let mut vals: Vec<f64> = ca.into_no_null_iter().filter(|v| !v.is_nan()).collect();
                if vals.is_empty() {
                    return Ok(ScalerParams { center: 0.0, scale: 1.0 });
                }
                let n = vals.len();
                let cmp = |a: &f64, b: &f64| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal);

                // Three independent quickselects (each O(n) average) instead of one
                // full O(n log n) sort — we only ever read 2-3 order-statistic slots.
                let mid = n / 2;
                vals.select_nth_unstable_by(mid, cmp);
                let upper_median = vals[mid];
                let median = if n % 2 == 0 {
                    let lower_median = vals[..mid]
                        .iter()
                        .copied()
                        .fold(f64::NEG_INFINITY, f64::max);
                    (lower_median + upper_median) / 2.0
                } else {
                    upper_median
                };

                let q1_idx = n / 4;
                vals.select_nth_unstable_by(q1_idx, cmp);
                let q1 = vals[q1_idx];

                let q3_idx = 3 * n / 4;
                vals.select_nth_unstable_by(q3_idx, cmp);
                let q3 = vals[q3_idx];
                let iqr = q3 - q1;
                Ok(ScalerParams {
                    center: median,
                    scale: if iqr.abs() < SCALE_EPSILON { 1.0 } else { iqr },
                })
            }
            ScalerType::MaxAbs => {
                // Note: f64::max returns the non-NaN operand when either side is
                // NaN, so this fold already ignores NaN values without needing an
                // explicit filter (unlike ca.min()/ca.max() used elsewhere above).
                let max_abs = ca
                    .into_iter()
                    .filter_map(|v| v.map(|x| x.abs()))
                    .fold(0.0f64, |a, b| a.max(b));
                Ok(ScalerParams {
                    center: 0.0,
                    scale: if max_abs < SCALE_EPSILON { 1.0 } else { max_abs },
                })
            }
            ScalerType::None => Ok(ScalerParams {
                center: 0.0,
                scale: 1.0,
            }),
        }
    }

    fn scale_series(&self, series: &Series, params: &ScalerParams) -> Result<Series> {
        let ca = series
            .f64()
            .map_err(|e| AutoMLError::DataError(e.to_string()))?;

        let (range_min, range_max) = self.feature_range;
        let is_minmax = self.scaler_type == ScalerType::MinMax;
        let clip = self.clip;

        let scaled: Float64Chunked = ca
            .into_iter()
            .map(|opt| {
                opt.map(|v| {
                    let mut s = (v - params.center) / params.scale;

                    // For MinMax, remap from [0,1] to [range_min, range_max]
                    if is_minmax {
                        s = s * (range_max - range_min) + range_min;
                    }

                    // Apply clipping if enabled
                    if clip {
                        s = match self.scaler_type {
                            ScalerType::MinMax => s.clamp(range_min, range_max),
                            ScalerType::MaxAbs => s.clamp(-1.0, 1.0),
                            _ => s, // Standard/Robust are unbounded
                        };
                    }

                    s
                })
            })
            .collect();

        Ok(scaled.with_name(series.name().clone()).into_series())
    }

    fn unscale_series(&self, series: &Series, params: &ScalerParams) -> Result<Series> {
        let ca = series
            .f64()
            .map_err(|e| AutoMLError::DataError(e.to_string()))?;

        let (range_min, range_max) = self.feature_range;
        let is_minmax = self.scaler_type == ScalerType::MinMax;

        let unscaled: Float64Chunked = ca
            .into_iter()
            .map(|opt| {
                opt.map(|v| {
                    let mut s = v;

                    // For MinMax, reverse the range mapping first
                    if is_minmax {
                        s = (s - range_min) / (range_max - range_min);
                    }

                    s * params.scale + params.center
                })
            })
            .collect();

        Ok(unscaled.with_name(series.name().clone()).into_series())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_df(name: &str, values: &[f64]) -> DataFrame {
        DataFrame::new(vec![
            Series::new(name.into(), values).into(),
        ])
        .unwrap()
    }

    // ── Standard scaler ──────────────────────────────────────────────

    #[test]
    fn test_standard_scaler() {
        let df = make_df("a", &[1.0, 2.0, 3.0, 4.0, 5.0]);
        let mut scaler = Scaler::new(ScalerType::Standard);
        let result = scaler.fit_transform(&df, &["a"]).unwrap();

        let col = result.column("a").unwrap().f64().unwrap();
        let mean: f64 = col.mean().unwrap();
        assert!(mean.abs() < 1e-10, "Mean should be ~0, got {}", mean);
    }

    #[test]
    fn test_standard_scaler_unit_variance() {
        let df = make_df("a", &[1.0, 2.0, 3.0, 4.0, 5.0]);
        let mut scaler = Scaler::new(ScalerType::Standard);
        let result = scaler.fit_transform(&df, &["a"]).unwrap();

        let col = result.column("a").unwrap().f64().unwrap();
        let std = col.std(1).unwrap();
        assert!(
            (std - 1.0).abs() < 1e-10,
            "Standard scaler should produce unit variance, got std={}",
            std
        );
    }

    // ── MinMax scaler ────────────────────────────────────────────────

    #[test]
    fn test_minmax_scaler() {
        let df = make_df("a", &[1.0, 2.0, 3.0, 4.0, 5.0]);
        let mut scaler = Scaler::new(ScalerType::MinMax);
        let result = scaler.fit_transform(&df, &["a"]).unwrap();

        let col = result.column("a").unwrap().f64().unwrap();
        assert!((col.min().unwrap() - 0.0).abs() < 1e-10);
        assert!((col.max().unwrap() - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_minmax_custom_range() {
        let df = make_df("a", &[1.0, 2.0, 3.0, 4.0, 5.0]);
        let mut scaler = Scaler::new(ScalerType::MinMax)
            .with_feature_range(-1.0, 1.0);
        let result = scaler.fit_transform(&df, &["a"]).unwrap();

        let col = result.column("a").unwrap().f64().unwrap();
        let min_val = col.min().unwrap();
        let max_val = col.max().unwrap();
        assert!(
            (min_val - (-1.0)).abs() < 1e-10,
            "MinMax(-1,1) min should be -1, got {}",
            min_val
        );
        assert!(
            (max_val - 1.0).abs() < 1e-10,
            "MinMax(-1,1) max should be 1, got {}",
            max_val
        );
    }

    #[test]
    fn test_minmax_custom_range_inverse() {
        let df = make_df("a", &[10.0, 20.0, 30.0, 40.0, 50.0]);
        let mut scaler = Scaler::new(ScalerType::MinMax)
            .with_feature_range(-1.0, 1.0);
        let scaled = scaler.fit_transform(&df, &["a"]).unwrap();
        let unscaled = scaler.inverse_transform(&scaled).unwrap();

        let original = df.column("a").unwrap().f64().unwrap();
        let restored = unscaled.column("a").unwrap().f64().unwrap();
        for (o, r) in original.into_iter().zip(restored.into_iter()) {
            assert!(
                (o.unwrap() - r.unwrap()).abs() < 1e-10,
                "Inverse transform should recover original values"
            );
        }
    }

    // ── Clipping ─────────────────────────────────────────────────────

    #[test]
    fn test_minmax_clip_out_of_range() {
        // Fit on [0, 100], then transform a value of 150 and -50
        let train = make_df("a", &[0.0, 25.0, 50.0, 75.0, 100.0]);
        let mut scaler = Scaler::new(ScalerType::MinMax).with_clip(true);
        scaler.fit(&train, &["a"]).unwrap();

        let test = make_df("a", &[-50.0, 50.0, 150.0]);
        let result = scaler.transform(&test).unwrap();

        let col = result.column("a").unwrap().f64().unwrap();
        let values: Vec<f64> = col.into_iter().map(|v| v.unwrap()).collect();
        assert!(
            (values[0] - 0.0).abs() < 1e-10,
            "Clipped -50 should be 0.0, got {}",
            values[0]
        );
        assert!(
            (values[1] - 0.5).abs() < 1e-10,
            "50 should be 0.5, got {}",
            values[1]
        );
        assert!(
            (values[2] - 1.0).abs() < 1e-10,
            "Clipped 150 should be 1.0, got {}",
            values[2]
        );
    }

    #[test]
    fn test_minmax_no_clip_allows_extrapolation() {
        let train = make_df("a", &[0.0, 100.0]);
        let mut scaler = Scaler::new(ScalerType::MinMax); // clip=false by default
        scaler.fit(&train, &["a"]).unwrap();

        let test = make_df("a", &[150.0]);
        let result = scaler.transform(&test).unwrap();

        let val = result.column("a").unwrap().f64().unwrap().get(0).unwrap();
        assert!(
            (val - 1.5).abs() < 1e-10,
            "Without clip, 150 on [0,100] should be 1.5, got {}",
            val
        );
    }

    #[test]
    fn test_maxabs_clip() {
        let train = make_df("a", &[-10.0, 0.0, 10.0]);
        let mut scaler = Scaler::new(ScalerType::MaxAbs).with_clip(true);
        scaler.fit(&train, &["a"]).unwrap();

        let test = make_df("a", &[-20.0, 5.0, 20.0]);
        let result = scaler.transform(&test).unwrap();

        let col = result.column("a").unwrap().f64().unwrap();
        let values: Vec<f64> = col.into_iter().map(|v| v.unwrap()).collect();
        assert!(
            (values[0] - (-1.0)).abs() < 1e-10,
            "Clipped -20 should be -1.0, got {}",
            values[0]
        );
        assert!(
            (values[2] - 1.0).abs() < 1e-10,
            "Clipped 20 should be 1.0, got {}",
            values[2]
        );
    }

    // ── Epsilon / near-zero variance ─────────────────────────────────

    #[test]
    fn test_constant_feature_standard() {
        // All same values → std=0 → should not produce NaN/Inf
        let df = make_df("a", &[5.0, 5.0, 5.0, 5.0]);
        let mut scaler = Scaler::new(ScalerType::Standard);
        let result = scaler.fit_transform(&df, &["a"]).unwrap();

        let col = result.column("a").unwrap().f64().unwrap();
        for v in col.into_iter() {
            let val = v.unwrap();
            assert!(val.is_finite(), "Constant feature should not produce Inf/NaN, got {}", val);
        }
    }

    #[test]
    fn test_near_zero_variance_no_overflow() {
        // Extremely small variance should be treated as zero
        let df = make_df("a", &[1.0, 1.0 + 1e-15, 1.0 - 1e-15]);
        let mut scaler = Scaler::new(ScalerType::Standard);
        let result = scaler.fit_transform(&df, &["a"]).unwrap();

        let col = result.column("a").unwrap().f64().unwrap();
        for v in col.into_iter() {
            let val = v.unwrap();
            assert!(
                val.is_finite() && val.abs() < 1e6,
                "Near-zero variance should not produce extreme values, got {}",
                val
            );
        }
    }

    #[test]
    fn test_constant_feature_minmax() {
        let df = make_df("a", &[42.0, 42.0, 42.0]);
        let mut scaler = Scaler::new(ScalerType::MinMax);
        let result = scaler.fit_transform(&df, &["a"]).unwrap();

        let col = result.column("a").unwrap().f64().unwrap();
        for v in col.into_iter() {
            assert!(v.unwrap().is_finite());
        }
    }

    // ── Robust scaler ────────────────────────────────────────────────

    #[test]
    fn test_robust_scaler_outlier_resistance() {
        // Robust scaler should center on median and scale by IQR,
        // so extreme outliers shouldn't affect the bulk of the data much
        let mut values: Vec<f64> = (1..=100).map(|x| x as f64).collect();
        values.push(10000.0); // extreme outlier

        let df = make_df("a", &values);
        let mut scaler = Scaler::new(ScalerType::Robust);
        let result = scaler.fit_transform(&df, &["a"]).unwrap();

        let col = result.column("a").unwrap().f64().unwrap();
        // The median of 1..=100 is ~50.5, IQR ~50
        // Values near the median should be close to 0
        let val_at_50 = col.get(49).unwrap(); // original value 50
        assert!(
            val_at_50.abs() < 0.5,
            "Value near median should be close to 0, got {}",
            val_at_50
        );
    }

    // ── MaxAbs scaler ────────────────────────────────────────────────

    #[test]
    fn test_maxabs_scaler() {
        let df = make_df("a", &[-10.0, -5.0, 0.0, 5.0, 10.0]);
        let mut scaler = Scaler::new(ScalerType::MaxAbs);
        let result = scaler.fit_transform(&df, &["a"]).unwrap();

        let col = result.column("a").unwrap().f64().unwrap();
        assert!((col.min().unwrap() - (-1.0)).abs() < 1e-10);
        assert!((col.max().unwrap() - 1.0).abs() < 1e-10);
    }

    // ── Inverse transform ────────────────────────────────────────────

    #[test]
    fn test_inverse_transform() {
        let df = make_df("a", &[1.0, 2.0, 3.0, 4.0, 5.0]);
        let mut scaler = Scaler::new(ScalerType::Standard);
        let scaled = scaler.fit_transform(&df, &["a"]).unwrap();
        let unscaled = scaler.inverse_transform(&scaled).unwrap();

        let original = df.column("a").unwrap().f64().unwrap();
        let restored = unscaled.column("a").unwrap().f64().unwrap();
        for (o, r) in original.into_iter().zip(restored.into_iter()) {
            assert!((o.unwrap() - r.unwrap()).abs() < 1e-10);
        }
    }

    #[test]
    fn test_inverse_transform_all_scaler_types() {
        let df = make_df("a", &[10.0, 20.0, 30.0, 40.0, 50.0]);

        for scaler_type in &[
            ScalerType::Standard,
            ScalerType::MinMax,
            ScalerType::Robust,
            ScalerType::MaxAbs,
            ScalerType::None,
        ] {
            let mut scaler = Scaler::new(scaler_type.clone());
            let scaled = scaler.fit_transform(&df, &["a"]).unwrap();
            let unscaled = scaler.inverse_transform(&scaled).unwrap();

            let original = df.column("a").unwrap().f64().unwrap();
            let restored = unscaled.column("a").unwrap().f64().unwrap();
            for (o, r) in original.into_iter().zip(restored.into_iter()) {
                assert!(
                    (o.unwrap() - r.unwrap()).abs() < 1e-8,
                    "Inverse transform failed for {:?}",
                    scaler_type
                );
            }
        }
    }

    // ── Multi-column ─────────────────────────────────────────────────

    #[test]
    fn test_multi_column_scaling() {
        let df = DataFrame::new(vec![
            Series::new("a".into(), &[1.0, 2.0, 3.0]).into(),
            Series::new("b".into(), &[100.0, 200.0, 300.0]).into(),
            Series::new("c".into(), &[-1.0, 0.0, 1.0]).into(),
        ])
        .unwrap();

        let mut scaler = Scaler::new(ScalerType::Standard);
        let result = scaler.fit_transform(&df, &["a", "b", "c"]).unwrap();

        // Each column should be independently standardized
        for col_name in &["a", "b", "c"] {
            let col = result.column(*col_name).unwrap().f64().unwrap();
            let mean = col.mean().unwrap();
            assert!(
                mean.abs() < 1e-10,
                "Column {} mean should be ~0, got {}",
                col_name,
                mean
            );
        }
    }

    // ── Large-scale data ─────────────────────────────────────────────

    #[test]
    fn test_large_scale_data() {
        // Test with a large range of values to verify numerical stability
        let values: Vec<f64> = (0..10000).map(|x| x as f64 * 1000.0).collect();
        let df = make_df("a", &values);

        let mut scaler = Scaler::new(ScalerType::MinMax);
        let result = scaler.fit_transform(&df, &["a"]).unwrap();

        let col = result.column("a").unwrap().f64().unwrap();
        let min_val = col.min().unwrap();
        let max_val = col.max().unwrap();
        assert!(
            (min_val - 0.0).abs() < 1e-10,
            "Large data MinMax min should be 0, got {}",
            min_val
        );
        assert!(
            (max_val - 1.0).abs() < 1e-10,
            "Large data MinMax max should be 1, got {}",
            max_val
        );

        // Verify all values are in range
        for v in col.into_iter() {
            let val = v.unwrap();
            assert!(val >= -1e-10 && val <= 1.0 + 1e-10, "Value out of range: {}", val);
        }
    }

    #[test]
    fn test_negative_values_minmax() {
        let df = make_df("a", &[-100.0, -50.0, 0.0, 50.0, 100.0]);
        let mut scaler = Scaler::new(ScalerType::MinMax);
        let result = scaler.fit_transform(&df, &["a"]).unwrap();

        let col = result.column("a").unwrap().f64().unwrap();
        assert!((col.min().unwrap() - 0.0).abs() < 1e-10);
        assert!((col.max().unwrap() - 1.0).abs() < 1e-10);
    }

    // ── NaN handling ─────────────────────────────────────────────────

    #[test]
    fn test_nan_passthrough() {
        let df = make_df("a", &[1.0, f64::NAN, 3.0, 4.0, 5.0]);
        let mut scaler = Scaler::new(ScalerType::Standard);
        let result = scaler.fit_transform(&df, &["a"]).unwrap();

        let col = result.column("a").unwrap().f64().unwrap();
        // NaN should remain NaN (not turned into a scaled value)
        assert!(col.get(1).is_none() || col.get(1).map_or(false, |v| v.is_nan()));
    }
}
