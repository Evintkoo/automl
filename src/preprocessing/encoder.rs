//! Categorical encoding implementations

use crate::error::{AutoMLError, Result};
use polars::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::hash::{Hash, Hasher};
use std::collections::hash_map::DefaultHasher;

/// Type of encoder to use
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum EncoderType {
    /// One-hot encoding
    OneHot,
    /// Label encoding (ordinal)
    Label,
    /// Target encoding (mean of target per category)
    Target,
    /// Binary encoding
    Binary,
    /// Frequency encoding
    Frequency,
    /// Leave-one-out encoding
    LeaveOneOut,
    /// Hash encoding (feature hashing) - good for high cardinality
    Hash { n_components: usize },
}

/// Categorical encoder
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Encoder {
    encoder_type: EncoderType,
    // Maps column name -> (category -> encoded value/index)
    mappings: HashMap<String, HashMap<String, usize>>,
    // For target encoding: column name -> (category -> target mean)
    target_means: HashMap<String, HashMap<String, f64>>,
    // For target/leave-one-out encoding: column name -> (category -> sum of target)
    target_sums: HashMap<String, HashMap<String, f64>>,
    // For target/leave-one-out encoding: column name -> (category -> count)
    target_counts: HashMap<String, HashMap<String, usize>>,
    // For leave-one-out encoding: column name -> row-aligned raw target values
    // captured at `fit_with_target` time, used to exclude a row's own target
    // value when `transform` is later called on that SAME data.
    fit_target_values: HashMap<String, Vec<Option<f64>>>,
    // For frequency encoding: column name -> (category -> frequency)
    frequencies: HashMap<String, HashMap<String, f64>>,
    is_fitted: bool,
}

impl Encoder {
    /// Create a new encoder
    pub fn new(encoder_type: EncoderType) -> Self {
        Self {
            encoder_type,
            mappings: HashMap::new(),
            target_means: HashMap::new(),
            target_sums: HashMap::new(),
            target_counts: HashMap::new(),
            fit_target_values: HashMap::new(),
            frequencies: HashMap::new(),
            is_fitted: false,
        }
    }

    /// Fit the encoder to the data
    pub fn fit(&mut self, df: &DataFrame, columns: &[&str]) -> Result<&mut Self> {
        for col_name in columns {
            let column = df
                .column(col_name)
                .map_err(|_| AutoMLError::FeatureNotFound(col_name.to_string()))?;
            let series = column.as_materialized_series();

            let mapping = self.build_mapping(series)?;
            self.mappings.insert(col_name.to_string(), mapping);
        }

        self.is_fitted = true;
        Ok(self)
    }

    /// Fit with target for target encoding
    pub fn fit_with_target(
        &mut self,
        df: &DataFrame,
        columns: &[&str],
        target: &Series,
    ) -> Result<&mut Self> {
        if !matches!(self.encoder_type, EncoderType::Target | EncoderType::LeaveOneOut) {
            return self.fit(df, columns);
        }

        let target_values = target
            .f64()
            .map_err(|e| AutoMLError::DataError(e.to_string()))?;

        for col_name in columns {
            let column = df
                .column(col_name)
                .map_err(|_| AutoMLError::FeatureNotFound(col_name.to_string()))?;
            let series = column.as_materialized_series();

            // Build regular mapping
            let mapping = self.build_mapping(series)?;
            self.mappings.insert(col_name.to_string(), mapping);

            // Build target sums/counts/means
            let (means, sums, counts) = self.compute_target_stats(series, &target_values)?;
            self.target_means.insert(col_name.to_string(), means);
            self.target_sums.insert(col_name.to_string(), sums);
            self.target_counts.insert(col_name.to_string(), counts);

            // Capture the row-aligned raw target values so a later `transform`
            // call on this SAME data can compute genuine leave-one-out means
            // (excluding each row's own target value). If `transform` is later
            // called on different/unseen data, these are simply unused and the
            // fit-time category means are used instead (see
            // `transform_leave_one_out`).
            let ca = series
                .str()
                .map_err(|e| AutoMLError::DataError(e.to_string()))?;
            let row_targets: Vec<Option<f64>> = ca
                .into_iter()
                .zip(target_values.into_iter())
                .map(|(cat, t)| if cat.is_some() { t } else { None })
                .collect();
            self.fit_target_values
                .insert(col_name.to_string(), row_targets);
        }

        self.is_fitted = true;
        Ok(self)
    }

    /// Transform the data
    pub fn transform(&self, df: &DataFrame) -> Result<DataFrame> {
        if !self.is_fitted {
            return Err(AutoMLError::ModelNotFitted);
        }

        match &self.encoder_type {
            EncoderType::OneHot => self.transform_onehot(df),
            EncoderType::Label => self.transform_label(df),
            EncoderType::Target => self.transform_target(df),
            EncoderType::Frequency => self.transform_frequency(df),
            EncoderType::Binary => self.transform_binary(df),
            EncoderType::Hash { n_components } => self.transform_hash(df, *n_components),
            EncoderType::LeaveOneOut => self.transform_leave_one_out(df),
        }
    }

    /// Fit and transform in one step
    pub fn fit_transform(&mut self, df: &DataFrame, columns: &[&str]) -> Result<DataFrame> {
        self.fit(df, columns)?;
        self.transform(df)
    }

    /// Fit and transform `EncoderType::Target`/`EncoderType::LeaveOneOut` columns
    /// using k-fold out-of-fold means, for encoding the TRAINING data itself
    /// without leaking each row's own target value into its own encoded value.
    ///
    /// This also fits the encoder on the FULL data (populating `target_means`,
    /// `target_sums`, `target_counts`), so a later `.transform(df)` call on
    /// genuinely new/unseen data (not this training data) correctly uses the
    /// full-data category means - no OOF split is needed there since there's
    /// no leakage risk when the target of the new rows was never used to fit.
    pub fn fit_transform_target_oof(
        &mut self,
        df: &DataFrame,
        columns: &[&str],
        target: &Series,
        n_folds: usize,
    ) -> Result<DataFrame> {
        if !matches!(self.encoder_type, EncoderType::Target | EncoderType::LeaveOneOut) {
            return self.fit_transform(df, columns);
        }

        // Fit on the full data first so mappings/target stats used by future
        // `transform()` calls (on new/unseen data) reflect the complete data.
        self.fit_with_target(df, columns, target)?;

        let n_rows = df.height();
        let n_folds = n_folds.clamp(2, n_rows.max(2));

        let target_values = target
            .f64()
            .map_err(|e| AutoMLError::DataError(e.to_string()))?;
        let row_targets: Vec<Option<f64>> = target_values.into_iter().collect();

        let mut new_cols: Vec<Series> = Vec::new();
        for col_name in columns {
            let column = df
                .column(col_name)
                .map_err(|_| AutoMLError::FeatureNotFound(col_name.to_string()))?;
            let series = column.as_materialized_series();
            let ca = series
                .str()
                .map_err(|e| AutoMLError::DataError(e.to_string()))?;
            let categories: Vec<Option<String>> =
                ca.into_iter().map(|v| v.map(|s| s.to_string())).collect();

            let total_sums = self.target_sums.get(*col_name).cloned().unwrap_or_default();
            let total_counts = self
                .target_counts
                .get(*col_name)
                .cloned()
                .unwrap_or_default();
            let total_sum_all: f64 = total_sums.values().sum();
            let total_count_all: usize = total_counts.values().sum();
            let global_mean: f64 = if total_count_all == 0 {
                0.0
            } else {
                total_sum_all / total_count_all as f64
            };

            // Per-fold sums/counts, so the out-of-fold aggregate for row `i` is
            // simply `total - fold_of(i)`.
            let mut fold_sums: Vec<HashMap<String, f64>> = vec![HashMap::new(); n_folds];
            let mut fold_counts: Vec<HashMap<String, usize>> = vec![HashMap::new(); n_folds];
            for (row, cat) in categories.iter().enumerate() {
                if let (Some(c), Some(t)) = (cat, row_targets.get(row).copied().flatten()) {
                    let f = row % n_folds;
                    *fold_sums[f].entry(c.clone()).or_insert(0.0) += t;
                    *fold_counts[f].entry(c.clone()).or_insert(0) += 1;
                }
            }

            let mut encoded: Vec<f64> = Vec::with_capacity(n_rows);
            for row in 0..n_rows {
                let value = match &categories[row] {
                    Some(c) => {
                        let f = row % n_folds;
                        let total_sum = total_sums.get(c).copied().unwrap_or(0.0);
                        let total_count = total_counts.get(c).copied().unwrap_or(0);
                        let fold_sum = fold_sums[f].get(c).copied().unwrap_or(0.0);
                        let fold_count = fold_counts[f].get(c).copied().unwrap_or(0);

                        let oof_sum = total_sum - fold_sum;
                        let oof_count = total_count.saturating_sub(fold_count);

                        if oof_count > 0 {
                            oof_sum / oof_count as f64
                        } else {
                            // This category only appears within this row's own
                            // fold - no out-of-fold data to estimate from, so
                            // fall back to the global mean rather than leaking.
                            global_mean
                        }
                    }
                    None => global_mean,
                };
                encoded.push(value);
            }

            new_cols.push(Series::new(col_name.to_string().into(), encoded));
        }

        let mut result = df.clone();
        for col in new_cols {
            result
                .with_column(col)
                .map_err(|e| AutoMLError::DataError(e.to_string()))?;
        }
        Ok(result)
    }

    fn build_mapping(&self, series: &Series) -> Result<HashMap<String, usize>> {
        let mut mapping = HashMap::new();
        let ca = series
            .str()
            .map_err(|e| AutoMLError::DataError(e.to_string()))?;

        let mut idx = 0usize;
        for val in ca.into_iter().flatten() {
            if !mapping.contains_key(val) {
                mapping.insert(val.to_string(), idx);
                idx += 1;
            }
        }

        Ok(mapping)
    }

    /// Compute per-category target sums, counts, and means from the fit data.
    fn compute_target_stats(
        &self,
        series: &Series,
        target: &Float64Chunked,
    ) -> Result<(
        HashMap<String, f64>,
        HashMap<String, f64>,
        HashMap<String, usize>,
    )> {
        let mut sums: HashMap<String, f64> = HashMap::new();
        let mut counts: HashMap<String, usize> = HashMap::new();

        let ca = series
            .str()
            .map_err(|e| AutoMLError::DataError(e.to_string()))?;

        for (cat, target_val) in ca.into_iter().zip(target.into_iter()) {
            if let (Some(c), Some(t)) = (cat, target_val) {
                *sums.entry(c.to_string()).or_insert(0.0) += t;
                *counts.entry(c.to_string()).or_insert(0) += 1;
            }
        }

        let means: HashMap<String, f64> = sums
            .iter()
            .map(|(k, sum)| {
                let count = counts.get(k).unwrap_or(&1);
                (k.clone(), sum / *count as f64)
            })
            .collect();

        Ok((means, sums, counts))
    }

    fn transform_onehot(&self, df: &DataFrame) -> Result<DataFrame> {
        let mut result = df.clone();

        for (col_name, mapping) in &self.mappings {
            if let Ok(series) = df.column(col_name) {
                let ca = series
                    .str()
                    .map_err(|e| AutoMLError::DataError(e.to_string()))?;

                // Create binary column for each category
                for (category, _) in mapping {
                    let new_col_name = format!("{}_{}", col_name, category);
                    let values: Vec<i32> = ca
                        .into_iter()
                        .map(|v| if v == Some(category.as_str()) { 1 } else { 0 })
                        .collect();

                    let new_series = Series::new(new_col_name.into(), values);
                    result = result
                        .with_column(new_series)
                        .map_err(|e| AutoMLError::DataError(e.to_string()))?
                        .clone();
                }

                // Drop original column
                result = result
                    .drop(col_name)
                    .map_err(|e| AutoMLError::DataError(e.to_string()))?;
            }
        }

        Ok(result)
    }

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

    /// Genuine leave-one-out target encoding: each row is encoded using the
    /// mean of all OTHER rows in the same category (excluding that row's own
    /// target value), i.e. `(category_sum - own_target) / (category_count - 1)`.
    ///
    /// This only has a row's own target value available when `transform` is
    /// called on the SAME data that was passed to `fit_with_target` (detected
    /// heuristically by matching row count against the captured fit-time
    /// target values). When called on different/unseen data - or when a
    /// row's category has only a single occurrence, making `count - 1 == 0`
    /// - there is nothing to leak/exclude, so it falls back to the category's
    /// fit-time mean, or the global mean for an unseen category.
    fn transform_leave_one_out(&self, df: &DataFrame) -> Result<DataFrame> {
        let mut new_cols: Vec<Series> = Vec::new();

        for (col_name, sums) in &self.target_sums {
            let counts = match self.target_counts.get(col_name) {
                Some(c) => c,
                None => continue,
            };
            let series = match df.column(col_name) {
                Ok(s) => s.as_materialized_series(),
                Err(_) => continue,
            };
            let ca = series
                .str()
                .map_err(|e| AutoMLError::DataError(e.to_string()))?;

            let global_mean: f64 = {
                let total_sum: f64 = sums.values().sum();
                let total_count: usize = counts.values().sum();
                if total_count == 0 { 0.0 } else { total_sum / total_count as f64 }
            };

            let fit_targets = self.fit_target_values.get(col_name);
            let same_data = fit_targets.map(|v| v.len()) == Some(series.len());

            let values: Vec<f64> = ca
                .into_iter()
                .enumerate()
                .map(|(row, cat)| {
                    let Some(c) = cat else { return global_mean };
                    let Some(&count) = counts.get(c) else { return global_mean };
                    let Some(&sum) = sums.get(c) else { return global_mean };

                    if same_data && count > 1 {
                        if let Some(own) = fit_targets.and_then(|v| v.get(row).copied().flatten())
                        {
                            return (sum - own) / (count as f64 - 1.0);
                        }
                    }
                    // Unseen data, singleton category, or missing own target:
                    // fall back to the (non-leaky) category mean.
                    sum / count as f64
                })
                .collect();

            new_cols.push(Series::new(col_name.clone().into(), values));
        }

        let mut result = df.clone();
        for col in new_cols {
            result
                .with_column(col)
                .map_err(|e| AutoMLError::DataError(e.to_string()))?;
        }
        Ok(result)
    }

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
        let mut all_cols: Vec<Column> = result.get_columns().to_vec();
        all_cols.extend(extra_cols.into_iter().map(|s| s.into()));
        DataFrame::new(all_cols).map_err(|e| AutoMLError::DataError(e.to_string()))
    }

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
        let mut all_cols: Vec<Column> = result.get_columns().to_vec();
        all_cols.extend(extra_cols.into_iter().map(|s| s.into()));
        DataFrame::new(all_cols).map_err(|e| AutoMLError::DataError(e.to_string()))
    }

    /// Hash a string to a bucket index using murmur-like hashing
    fn hash_string(&self, s: &str, seed: usize) -> usize {
        let mut hasher = DefaultHasher::new();
        seed.hash(&mut hasher);
        s.hash(&mut hasher);
        hasher.finish() as usize
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_label_encoding() {
        let df = DataFrame::new(vec![Series::new(
            "category".into(),
            &["a", "b", "c", "a", "b"],
        ).into()])
        .unwrap();

        let mut encoder = Encoder::new(EncoderType::Label);
        let result = encoder.fit_transform(&df, &["category"]).unwrap();

        let col = result.column("category").unwrap().i64().unwrap();
        // All values should be encoded as integers
        assert!(col.into_iter().all(|v| v.is_some()));
    }

    #[test]
    fn test_onehot_encoding() {
        let df = DataFrame::new(vec![Series::new(
            "category".into(),
            &["a", "b", "c", "a", "b"],
        ).into()])
        .unwrap();

        let mut encoder = Encoder::new(EncoderType::OneHot);
        let result = encoder.fit_transform(&df, &["category"]).unwrap();

        // Should have created new columns and dropped original
        assert!(result.column("category").is_err());
        assert_eq!(result.width(), 3); // a, b, c columns
    }
}
