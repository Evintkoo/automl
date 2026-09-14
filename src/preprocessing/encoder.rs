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

            // Build target means
            let means = self.compute_target_means(series, &target_values)?;
            self.target_means.insert(col_name.to_string(), means);
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
            EncoderType::LeaveOneOut => self.transform_label(df), // Simplified LOO
        }
    }

    /// Fit and transform in one step
    pub fn fit_transform(&mut self, df: &DataFrame, columns: &[&str]) -> Result<DataFrame> {
        self.fit(df, columns)?;
        self.transform(df)
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

    fn compute_target_means(
        &self,
        series: &Series,
        target: &Float64Chunked,
    ) -> Result<HashMap<String, f64>> {
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
            .into_iter()
            .map(|(k, sum)| {
                let count = counts.get(&k).unwrap_or(&1);
                (k, sum / *count as f64)
            })
            .collect();

        Ok(means)
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
