//! Data preprocessing pipeline

use crate::error::{AutoMLError, Result};
use super::{
    config::PreprocessingConfig,
    imputer::{Imputer, ImputeStrategy},
    scaler::{Scaler, ScalerType},
    encoder::Encoder,
    FeatureStats, ColumnType,
};
use polars::prelude::*;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::Instant;

/// Running statistics for incremental (partial) fitting using Welford's algorithm
#[derive(Debug, Clone, Serialize, Deserialize)]
struct RunningStats {
    count: usize,
    mean: f64,
    m2: f64,   // sum of squares of differences from the current mean
    min: f64,
    max: f64,
}

impl RunningStats {
    fn new() -> Self {
        Self {
            count: 0,
            mean: 0.0,
            m2: 0.0,
            min: f64::INFINITY,
            max: f64::NEG_INFINITY,
        }
    }

    /// Update with a single value using Welford's online algorithm
    fn update(&mut self, value: f64) {
        self.count += 1;
        let delta = value - self.mean;
        self.mean += delta / self.count as f64;
        let delta2 = value - self.mean;
        self.m2 += delta * delta2;
        if value < self.min { self.min = value; }
        if value > self.max { self.max = value; }
    }

    fn variance(&self) -> f64 {
        if self.count < 2 { 0.0 } else { self.m2 / (self.count - 1) as f64 }
    }

    fn std(&self) -> f64 {
        self.variance().sqrt()
    }
}

/// Main data preprocessing pipeline
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataPreprocessor {
    config: PreprocessingConfig,
    numeric_columns: Vec<String>,
    categorical_columns: Vec<String>,
    numeric_imputer: Option<Imputer>,
    categorical_imputer: Option<Imputer>,
    scaler: Option<Scaler>,
    encoder: Option<Encoder>,
    feature_stats: HashMap<String, FeatureStats>,
    is_fitted: bool,
    /// Timing: seconds spent in last fit call
    fit_time: Option<f64>,
    /// Timing: seconds spent in last transform call
    transform_time: Option<f64>,
    /// Total number of samples (rows) processed across fit/transform/partial_fit
    samples_processed: usize,
    /// Per-feature running statistics for partial_fit (Welford's algorithm)
    running_stats: HashMap<String, RunningStats>,
}

impl DataPreprocessor {
    /// Create a new preprocessor with default configuration
    pub fn new() -> Self {
        Self::with_config(PreprocessingConfig::default())
    }

    /// Create a new preprocessor with custom configuration
    pub fn with_config(config: PreprocessingConfig) -> Self {
        Self {
            config,
            numeric_columns: Vec::new(),
            categorical_columns: Vec::new(),
            numeric_imputer: None,
            categorical_imputer: None,
            scaler: None,
            encoder: None,
            feature_stats: HashMap::new(),
            is_fitted: false,
            fit_time: None,
            transform_time: None,
            samples_processed: 0,
            running_stats: HashMap::new(),
        }
    }

    /// Cast all numeric (integer) columns to Float64 for consistent processing.
    /// Collects casted columns first, then applies them in a single pass to
    /// avoid redundant full-DataFrame clones per column.
    fn cast_numeric_to_f64(df: &DataFrame) -> Result<DataFrame> {
        let columns_to_cast: Vec<Column> = df.get_columns().iter()
            .filter_map(|col| {
                match col.dtype() {
                    DataType::Int8 | DataType::Int16 | DataType::Int32 | DataType::Int64 |
                    DataType::UInt8 | DataType::UInt16 | DataType::UInt32 | DataType::UInt64 |
                    DataType::Float32 => {
                        col.cast(&DataType::Float64).ok()
                    }
                    _ => None,
                }
            })
            .collect();

        if columns_to_cast.is_empty() {
            return Ok(df.clone());
        }

        // Build all columns (keep originals, replace casted ones)
        let cast_names: std::collections::HashSet<&str> = columns_to_cast.iter()
            .map(|c| c.name().as_str())
            .collect();

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
    }

    pub fn fit(&mut self, df: &DataFrame) -> Result<&mut Self> {
        let start = Instant::now();

        // Cast numeric columns to Float64
        let df = &Self::cast_numeric_to_f64(df)?;

        // Detect column types
        self.detect_column_types(df)?;

        // Compute feature statistics
        self.compute_statistics(df)?;

        // Fit imputers
        if !self.numeric_columns.is_empty() {
            let mut imputer = Imputer::new(self.config.numeric_impute_strategy.clone());
            let cols: Vec<&str> = self.numeric_columns.iter().map(|s| s.as_str()).collect();
            imputer.fit(df, &cols)?;
            self.numeric_imputer = Some(imputer);
        }

        if !self.categorical_columns.is_empty() {
            let mut imputer = Imputer::new(self.config.categorical_impute_strategy.clone());
            let cols: Vec<&str> = self.categorical_columns.iter().map(|s| s.as_str()).collect();
            imputer.fit(df, &cols)?;
            self.categorical_imputer = Some(imputer);
        }

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

        self.is_fitted = true;
        self.samples_processed += df.height();
        self.fit_time = Some(start.elapsed().as_secs_f64());
        Ok(self)
    }

    /// Transform the data
    pub fn transform(&self, df: &DataFrame) -> Result<DataFrame> {
        let start = Instant::now();

        if !self.is_fitted {
            return Err(AutoMLError::ModelNotFitted);
        }

        // Cast numeric columns to Float64
        let mut result = Self::cast_numeric_to_f64(df)?;

        // Apply numeric imputation
        if let Some(ref imputer) = self.numeric_imputer {
            result = imputer.transform(&result)?;
        }

        // Apply categorical imputation
        if let Some(ref imputer) = self.categorical_imputer {
            result = imputer.transform(&result)?;
        }

        // Apply scaling
        if let Some(ref scaler) = self.scaler {
            result = scaler.transform(&result)?;
        }

        // Apply encoding
        if let Some(ref encoder) = self.encoder {
            result = encoder.transform(&result)?;
        }

        // Note: transform_time is stored per-call but self is &self, so we
        // cannot mutate here. Timing is available via fit_transform path.
        let _elapsed = start.elapsed().as_secs_f64();

        Ok(result)
    }

    /// Fit and transform in one step
    pub fn fit_transform(&mut self, df: &DataFrame) -> Result<DataFrame> {
        self.fit(df)?;
        let start = Instant::now();
        let result = self.transform(df)?;
        self.transform_time = Some(start.elapsed().as_secs_f64());
        // Note: samples_processed already incremented in fit(), don't double-count
        Ok(result)
    }

    /// Reverse the transformation, recovering approximate original values.
    ///
    /// Applies inverse scaling (using stored params) on numeric columns.
    /// For `Vec<Vec<f64>>` input, each inner vec is one row of numeric features
    /// in the same order as `self.numeric_columns`.
    pub fn reverse_transform(&self, data: &[Vec<f64>]) -> std::result::Result<Vec<Vec<f64>>, String> {
        if !self.is_fitted {
            return Err("Preprocessor is not fitted".to_string());
        }

        let scaler = match &self.scaler {
            Some(s) => s,
            None => return Ok(data.to_vec()), // no scaling was applied
        };

        let n_cols = self.numeric_columns.len();
        let mut result = Vec::with_capacity(data.len());

        // Build a DataFrame from the input so we can use Scaler::inverse_transform
        let columns: Vec<Column> = (0..n_cols)
            .map(|col_idx| {
                let col_name = &self.numeric_columns[col_idx];
                let values: Vec<f64> = data.iter().map(|row| {
                    if col_idx < row.len() { row[col_idx] } else { 0.0 }
                }).collect();
                Column::new(col_name.clone().into(), values)
            })
            .collect();

        let df = DataFrame::new(columns).map_err(|e| e.to_string())?;
        let unscaled = scaler.inverse_transform(&df).map_err(|e| e.to_string())?;

        // Convert back to Vec<Vec<f64>>
        for row_idx in 0..unscaled.height() {
            let mut row = Vec::with_capacity(n_cols);
            for col_name in &self.numeric_columns {
                let val = unscaled
                    .column(col_name)
                    .map_err(|e| e.to_string())?
                    .f64()
                    .map_err(|e| e.to_string())?
                    .get(row_idx)
                    .unwrap_or(0.0);
                row.push(val);
            }
            result.push(row);
        }

        Ok(result)
    }

    /// Incrementally update running statistics without reprocessing all data.
    ///
    /// Uses Welford's online algorithm to maintain per-feature mean, variance,
    /// min, and max. Each inner vec is one row of numeric features in the same
    /// order as `self.numeric_columns`.
    pub fn partial_fit(&mut self, data: &[Vec<f64>]) -> std::result::Result<(), String> {
        if !self.is_fitted {
            // If column names aren't known yet, we can't do partial_fit
            if self.numeric_columns.is_empty() {
                return Err("Preprocessor must be fitted at least once before partial_fit".to_string());
            }
        }

        let n_cols = self.numeric_columns.len();

        for row in data {
            for col_idx in 0..n_cols.min(row.len()) {
                let col_name = &self.numeric_columns[col_idx];
                let stats = self
                    .running_stats
                    .entry(col_name.clone())
                    .or_insert_with(RunningStats::new);
                stats.update(row[col_idx]);
            }
        }

        // Update feature_stats from running stats
        for col_name in &self.numeric_columns {
            if let Some(rs) = self.running_stats.get(col_name) {
                let fs = self
                    .feature_stats
                    .entry(col_name.clone())
                    .or_insert_with(|| FeatureStats::new(col_name, ColumnType::Numeric));
                fs.mean = Some(rs.mean);
                fs.std = Some(rs.std());
                fs.min = Some(rs.min);
                fs.max = Some(rs.max);
                fs.count = rs.count;
            }
        }

        self.samples_processed += data.len();
        self.is_fitted = true;
        Ok(())
    }

    /// Analyze the data and auto-select the best preprocessing strategy.
    ///
    /// Examines sparsity, outlier ratio, and skewness to choose scaler type,
    /// outlier handling, and imputation strategy.
    pub fn optimize_for_dataset(&mut self, data: &[Vec<f64>]) -> std::result::Result<(), String> {
        if data.is_empty() {
            return Err("Data is empty".to_string());
        }

        let n_rows = data.len();
        let n_cols = data.first().map(|r| r.len()).unwrap_or(0);
        if n_cols == 0 {
            return Err("Data has no features".to_string());
        }

        let mut total_zeros = 0usize;
        let mut total_elements = 0usize;
        let mut col_means = vec![0.0f64; n_cols];
        let mut col_mins = vec![f64::INFINITY; n_cols];
        let mut col_maxs = vec![f64::NEG_INFINITY; n_cols];

        // First pass: compute means, mins, maxs, sparsity
        for row in data {
            for (j, &val) in row.iter().enumerate() {
                if j >= n_cols { break; }
                col_means[j] += val;
                if val < col_mins[j] { col_mins[j] = val; }
                if val > col_maxs[j] { col_maxs[j] = val; }
                if val == 0.0 { total_zeros += 1; }
                total_elements += 1;
            }
        }
        for m in &mut col_means { *m /= n_rows as f64; }

        // Second pass: variance, skewness, and outlier count (merged into single scan)
        let mut col_var = vec![0.0f64; n_cols];
        let mut col_skew_num = vec![0.0f64; n_cols];
        // We'll use Welford-like approach: compute variance first, then derive std inline
        // But we need std for outlier detection... so we do a two-step within this pass:
        // Step 1: accumulate variance/skewness sums
        for row in data {
            for (j, &val) in row.iter().enumerate() {
                if j >= n_cols { break; }
                let diff = val - col_means[j];
                col_var[j] += diff * diff;
                col_skew_num[j] += diff * diff * diff;
            }
        }

        let mut avg_abs_skewness = 0.0f64;
        let mut col_std = vec![0.0f64; n_cols];
        for j in 0..n_cols {
            col_var[j] /= (n_rows - 1).max(1) as f64;
            col_std[j] = col_var[j].sqrt();
            if col_std[j] > 1e-12 {
                let skew = (col_skew_num[j] / n_rows as f64) / (col_std[j] * col_std[j] * col_std[j]);
                avg_abs_skewness += skew.abs();
            }
        }
        avg_abs_skewness /= n_cols as f64;

        // Count outliers (> 3 std from mean) — reuses col_std from above
        // Combined into iteration over pre-computed means/stds without a separate data pass
        let outlier_count: usize = data.iter()
            .flat_map(|row| row.iter().enumerate().take(n_cols))
            .filter(|&(j, &val)| col_std[j] > 1e-12 && (val - col_means[j]).abs() > 3.0 * col_std[j])
            .count();

        let sparsity = total_zeros as f64 / total_elements.max(1) as f64;
        let outlier_ratio = outlier_count as f64 / total_elements.max(1) as f64;

        // Auto-select scaler
        if outlier_ratio > 0.05 {
            self.config.scaler_type = ScalerType::Robust;
        } else if sparsity > 0.5 {
            self.config.scaler_type = ScalerType::MaxAbs;
        } else if avg_abs_skewness > 1.0 {
            self.config.scaler_type = ScalerType::MinMax;
        } else {
            self.config.scaler_type = ScalerType::Standard;
        }

        // Auto-select outlier handling
        if outlier_ratio > 0.01 {
            self.config.handle_outliers = true;
            self.config.outlier_threshold = if outlier_ratio > 0.05 { 2.5 } else { 3.0 };
        } else {
            self.config.handle_outliers = false;
        }

        // Auto-select imputation (check for NaN in data)
        let has_nan = data.iter().any(|row| row.iter().any(|v| v.is_nan()));
        if has_nan {
            if avg_abs_skewness > 1.0 {
                self.config.numeric_impute_strategy = ImputeStrategy::Median;
            } else {
                self.config.numeric_impute_strategy = ImputeStrategy::Mean;
            }
        }

        Ok(())
    }

    /// Return preprocessing statistics as a JSON-compatible map.
    pub fn get_statistics(&self) -> HashMap<String, serde_json::Value> {
        let mut stats = HashMap::new();

        // Per-feature statistics
        let mut means = HashMap::new();
        let mut stds = HashMap::new();
        let mut mins = HashMap::new();
        let mut maxs = HashMap::new();
        let mut null_counts = HashMap::new();
        let mut feature_types = HashMap::new();

        for (name, fs) in &self.feature_stats {
            if let Some(m) = fs.mean { means.insert(name.clone(), m); }
            if let Some(s) = fs.std { stds.insert(name.clone(), s); }
            if let Some(mn) = fs.min { mins.insert(name.clone(), mn); }
            if let Some(mx) = fs.max { maxs.insert(name.clone(), mx); }
            null_counts.insert(name.clone(), fs.null_count);
            feature_types.insert(name.clone(), format!("{:?}", fs.dtype));
        }

        stats.insert("means".to_string(), serde_json::to_value(&means).unwrap_or_default());
        stats.insert("stds".to_string(), serde_json::to_value(&stds).unwrap_or_default());
        stats.insert("mins".to_string(), serde_json::to_value(&mins).unwrap_or_default());
        stats.insert("maxs".to_string(), serde_json::to_value(&maxs).unwrap_or_default());
        stats.insert("null_counts".to_string(), serde_json::to_value(&null_counts).unwrap_or_default());
        stats.insert("feature_types".to_string(), serde_json::to_value(&feature_types).unwrap_or_default());
        stats.insert("is_fitted".to_string(), serde_json::Value::Bool(self.is_fitted));
        stats.insert("numeric_columns".to_string(), serde_json::to_value(&self.numeric_columns).unwrap_or_default());
        stats.insert("categorical_columns".to_string(), serde_json::to_value(&self.categorical_columns).unwrap_or_default());

        stats
    }

    /// Return performance / timing metrics.
    pub fn get_performance_metrics(&self) -> HashMap<String, f64> {
        let mut metrics = HashMap::new();
        if let Some(ft) = self.fit_time {
            metrics.insert("fit_time".to_string(), ft);
        }
        if let Some(tt) = self.transform_time {
            metrics.insert("transform_time".to_string(), tt);
        }
        metrics.insert("total_samples_processed".to_string(), self.samples_processed as f64);
        metrics
    }

    /// Get feature statistics
    pub fn feature_stats(&self) -> &HashMap<String, FeatureStats> {
        &self.feature_stats
    }

    /// Get numeric column names
    pub fn numeric_columns(&self) -> &[String] {
        &self.numeric_columns
    }

    /// Get categorical column names
    pub fn categorical_columns(&self) -> &[String] {
        &self.categorical_columns
    }

    /// Save the preprocessor to a file
    pub fn save(&self, path: &str) -> Result<()> {
        let json = serde_json::to_string_pretty(self)?;
        std::fs::write(path, json)?;
        Ok(())
    }

    /// Load a preprocessor from a file
    pub fn load(path: &str) -> Result<Self> {
        let json = std::fs::read_to_string(path)?;
        let preprocessor: Self = serde_json::from_str(&json)?;
        Ok(preprocessor)
    }

    fn detect_column_types(&mut self, df: &DataFrame) -> Result<()> {
        self.numeric_columns.clear();
        self.categorical_columns.clear();

        for col in df.get_columns() {
            let name = col.name().to_string();
            
            match col.dtype() {
                DataType::Int8 | DataType::Int16 | DataType::Int32 | DataType::Int64 |
                DataType::UInt8 | DataType::UInt16 | DataType::UInt32 | DataType::UInt64 |
                DataType::Float32 | DataType::Float64 => {
                    self.numeric_columns.push(name);
                }
                DataType::String | DataType::Categorical(_, _) => {
                    self.categorical_columns.push(name);
                }
                _ => {
                    // Try to infer type from data
                    if col.str().is_ok() {
                        self.categorical_columns.push(name);
                    }
                }
            }
        }

        Ok(())
    }

    fn compute_statistics(&mut self, df: &DataFrame) -> Result<()> {
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
}

impl Default for DataPreprocessor {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::preprocessing::EncoderType;

    fn create_test_dataframe() -> DataFrame {
        DataFrame::new(vec![
            Series::new("age".into(), &[25.0, 30.0, 35.0, 40.0, 45.0]).into(),
            Series::new("income".into(), &[50000.0, 60000.0, 70000.0, 80000.0, 90000.0]).into(),
            Series::new("city".into(), &["NYC", "LA", "NYC", "SF", "LA"]).into(),
        ])
        .unwrap()
    }

    #[test]
    fn test_preprocessor_creation() {
        let preprocessor = DataPreprocessor::new();
        assert!(!preprocessor.is_fitted);
    }

    #[test]
    fn test_column_detection() {
        let df = create_test_dataframe();
        let mut preprocessor = DataPreprocessor::new();
        preprocessor.fit(&df).unwrap();

        assert_eq!(preprocessor.numeric_columns().len(), 2);
        assert_eq!(preprocessor.categorical_columns().len(), 1);
        assert!(preprocessor.numeric_columns().contains(&"age".to_string()));
        assert!(preprocessor.numeric_columns().contains(&"income".to_string()));
        assert!(preprocessor.categorical_columns().contains(&"city".to_string()));
    }

    #[test]
    fn test_fit_transform() {
        let df = create_test_dataframe();
        let mut preprocessor = DataPreprocessor::with_config(
            PreprocessingConfig::new()
                .with_scaler(ScalerType::Standard)
                .with_encoder(EncoderType::OneHot),
        );

        let result = preprocessor.fit_transform(&df).unwrap();
        
        // Original numeric columns should be scaled
        assert!(result.column("age").is_ok());
        assert!(result.column("income").is_ok());
        
        // Original categorical column should be removed (one-hot)
        assert!(result.column("city").is_err());
    }

    #[test]
    fn test_feature_statistics() {
        let df = create_test_dataframe();
        let mut preprocessor = DataPreprocessor::new();
        preprocessor.fit(&df).unwrap();

        let stats = preprocessor.feature_stats();
        
        let age_stats = stats.get("age").unwrap();
        assert_eq!(age_stats.dtype, ColumnType::Numeric);
        assert!(age_stats.mean.is_some());
        
        let city_stats = stats.get("city").unwrap();
        assert_eq!(city_stats.dtype, ColumnType::Categorical);
        assert!(city_stats.categories.is_some());
    }
}
