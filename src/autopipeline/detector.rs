//! Automatic data type detection

use crate::error::Result;
use crate::preprocessing::ColumnType;
use serde::{Deserialize, Serialize};

/// Detected column information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ColumnInfo {
    /// Column name
    pub name: String,
    /// Detected data type
    pub dtype: ColumnType,
    /// Number of unique values
    pub n_unique: usize,
    /// Number of missing values
    pub n_missing: usize,
    /// Percentage missing
    pub pct_missing: f64,
    /// Is constant (single value)
    pub is_constant: bool,
    /// Is ID-like (high cardinality, unique)
    pub is_id_like: bool,
    /// Recommended encoding
    pub recommended_encoding: Option<String>,
    /// Statistics
    pub stats: ColumnStats,
}

/// Column statistics
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ColumnStats {
    /// Mean (for numeric)
    pub mean: Option<f64>,
    /// Standard deviation (for numeric)
    pub std: Option<f64>,
    /// Minimum (for numeric)
    pub min: Option<f64>,
    /// Maximum (for numeric)
    pub max: Option<f64>,
    /// Median (for numeric)
    pub median: Option<f64>,
    /// Skewness (for numeric)
    pub skewness: Option<f64>,
    /// Kurtosis (for numeric)
    pub kurtosis: Option<f64>,
    /// Most frequent value (for categorical)
    pub mode: Option<String>,
    /// Mode frequency
    pub mode_frequency: Option<usize>,
}

/// Detected data schema
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DetectedSchema {
    /// Column information
    pub columns: Vec<ColumnInfo>,
    /// Number of rows
    pub n_rows: usize,
    /// Number of columns
    pub n_cols: usize,
    /// Numeric column indices
    pub numeric_columns: Vec<usize>,
    /// Categorical column indices
    pub categorical_columns: Vec<usize>,
    /// Datetime column indices
    pub datetime_columns: Vec<usize>,
    /// Text column indices
    pub text_columns: Vec<usize>,
    /// Columns to drop (ID-like or constant)
    pub drop_columns: Vec<usize>,
    /// Overall data quality score (0-1)
    pub quality_score: f64,
}

impl DetectedSchema {
    /// Get columns that need imputation
    pub fn columns_needing_imputation(&self) -> Vec<usize> {
        self.columns
            .iter()
            .enumerate()
            .filter(|(i, col)| col.n_missing > 0 && !self.drop_columns.contains(i))
            .map(|(i, _)| i)
            .collect()
    }

    /// Get columns with high cardinality
    pub fn high_cardinality_columns(&self, threshold: usize) -> Vec<usize> {
        self.categorical_columns
            .iter()
            .filter(|&&i| self.columns[i].n_unique > threshold)
            .copied()
            .collect()
    }

    /// Get columns with skewed distributions
    pub fn skewed_columns(&self, threshold: f64) -> Vec<usize> {
        self.numeric_columns
            .iter()
            .filter(|&&i| {
                self.columns[i]
                    .stats
                    .skewness
                    .map(|s| s.abs() > threshold)
                    .unwrap_or(false)
            })
            .copied()
            .collect()
    }
}

/// Automatic data type detector
pub struct DataTypeDetector {
    /// Threshold for considering column as categorical
    categorical_threshold: f64,
    /// Maximum unique values for categorical
    max_categorical_unique: usize,
    /// Minimum text length to consider as text
    #[allow(dead_code)]
    min_text_length: usize,
    /// ID column detection enabled
    detect_id_columns: bool,
}

impl DataTypeDetector {
    /// Create new detector with defaults
    pub fn new() -> Self {
        Self {
            categorical_threshold: 0.05,
            max_categorical_unique: 50,
            min_text_length: 50,
            detect_id_columns: true,
        }
    }

    /// Set categorical threshold
    pub fn with_categorical_threshold(mut self, threshold: f64) -> Self {
        self.categorical_threshold = threshold;
        self
    }

    /// Set max categorical unique values
    pub fn with_max_categorical_unique(mut self, max: usize) -> Self {
        self.max_categorical_unique = max;
        self
    }

    /// Detect schema from numeric data
    pub fn detect_from_array(&self, data: &ndarray::Array2<f64>, column_names: Option<&[String]>) -> Result<DetectedSchema> {
        let n_rows = data.nrows();
        let n_cols = data.ncols();

        let mut columns = Vec::with_capacity(n_cols);
        let mut numeric_columns = Vec::new();
        let mut categorical_columns = Vec::new();
        let mut drop_columns = Vec::new();

        for col_idx in 0..n_cols {
            let col = data.column(col_idx);
            let name = column_names
                .and_then(|names| names.get(col_idx))
                .map(|s| s.clone())
                .unwrap_or_else(|| format!("column_{}", col_idx));

            // Count missing (NaN) values
            let n_missing = col.iter().filter(|v| v.is_nan()).count();
            let pct_missing = n_missing as f64 / n_rows as f64;

            // Count unique values
            let mut unique_values: Vec<i64> = col
                .iter()
                .filter(|v| !v.is_nan())
                .map(|v| (v * 1000000.0) as i64)
                .collect();
            unique_values.sort();
            unique_values.dedup();
            let n_unique = unique_values.len();

            // Compute statistics
            let valid_values: Vec<f64> = col.iter().filter(|v| !v.is_nan()).copied().collect();
            let stats = self.compute_numeric_stats(&valid_values);

            // Determine if constant
            let is_constant = n_unique <= 1;

            // Determine if ID-like
            let is_id_like = self.detect_id_columns 
                && n_unique as f64 / n_rows as f64 > 0.95 
                && n_unique > 100;

            // Determine column type
            let unique_ratio = n_unique as f64 / n_rows.max(1) as f64;
            let dtype = if is_constant {
                drop_columns.push(col_idx);
                ColumnType::Numeric
            } else if is_id_like {
                drop_columns.push(col_idx);
                ColumnType::Numeric
            } else if n_unique <= self.max_categorical_unique && unique_ratio < self.categorical_threshold {
                categorical_columns.push(col_idx);
                ColumnType::Categorical
            } else {
                numeric_columns.push(col_idx);
                ColumnType::Numeric
            };

            // Determine recommended encoding
            let recommended_encoding = if dtype == ColumnType::Categorical {
                if n_unique <= 10 {
                    Some("onehot".to_string())
                } else {
                    Some("target".to_string())
                }
            } else {
                None
            };

            columns.push(ColumnInfo {
                name,
                dtype,
                n_unique,
                n_missing,
                pct_missing,
                is_constant,
                is_id_like,
                recommended_encoding,
                stats,
            });
        }

        // Compute quality score
        let quality_score = self.compute_quality_score(&columns, n_rows);

        Ok(DetectedSchema {
            columns,
            n_rows,
            n_cols,
            numeric_columns,
            categorical_columns,
            datetime_columns: vec![],
            text_columns: vec![],
            drop_columns,
            quality_score,
        })
    }

    fn compute_numeric_stats(&self, values: &[f64]) -> ColumnStats {
        if values.is_empty() {
            return ColumnStats::default();
        }

        let n = values.len() as f64;
        let mean = values.iter().sum::<f64>() / n;
        let variance = values.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / n;
        let std = variance.sqrt();

        let min = values.iter().cloned().fold(f64::INFINITY, f64::min);
        let max = values.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

        // Median
        let mut sorted = values.to_vec();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let median = if sorted.len() % 2 == 0 {
            (sorted[sorted.len() / 2 - 1] + sorted[sorted.len() / 2]) / 2.0
        } else {
            sorted[sorted.len() / 2]
        };

        // Skewness
        let skewness = if std > 0.0 {
            let m3 = values.iter().map(|v| ((v - mean) / std).powi(3)).sum::<f64>() / n;
            Some(m3)
        } else {
            None
        };

        // Kurtosis
        let kurtosis = if std > 0.0 {
            let m4 = values.iter().map(|v| ((v - mean) / std).powi(4)).sum::<f64>() / n - 3.0;
            Some(m4)
        } else {
            None
        };

        ColumnStats {
            mean: Some(mean),
            std: Some(std),
            min: Some(min),
            max: Some(max),
            median: Some(median),
            skewness,
            kurtosis,
            mode: None,
            mode_frequency: None,
        }
    }

    fn compute_quality_score(&self, columns: &[ColumnInfo], n_rows: usize) -> f64 {
        if columns.is_empty() {
            return 0.0;
        }

        let mut score = 1.0;

        // Penalize for missing values
        let avg_missing: f64 = columns.iter().map(|c| c.pct_missing).sum::<f64>() / columns.len() as f64;
        score -= avg_missing * 0.5;

        // Penalize for constant columns
        let pct_constant = columns.iter().filter(|c| c.is_constant).count() as f64 / columns.len() as f64;
        score -= pct_constant * 0.2;

        // Penalize for very small dataset
        if n_rows < 100 {
            score -= 0.2;
        } else if n_rows < 1000 {
            score -= 0.1;
        }

        score.max(0.0).min(1.0)
    }
}

impl Default for DataTypeDetector {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;

    #[test]
    fn test_detect_numeric() {
        let data = Array2::from_shape_vec(
            (10, 3),
            vec![
                1.0, 0.0, 100.0,
                2.0, 0.0, 200.0,
                3.0, 1.0, 300.0,
                4.0, 1.0, 400.0,
                5.0, 0.0, 500.0,
                6.0, 1.0, 600.0,
                7.0, 0.0, 700.0,
                8.0, 1.0, 800.0,
                9.0, 0.0, 900.0,
                10.0, 1.0, 1000.0,
            ],
        ).unwrap();

        let detector = DataTypeDetector::new();
        let schema = detector.detect_from_array(&data, None).unwrap();

        assert_eq!(schema.n_rows, 10);
        assert_eq!(schema.n_cols, 3);
        assert!(!schema.numeric_columns.is_empty());
    }

    #[test]
    fn test_detect_constant_column() {
        let data = Array2::from_shape_vec(
            (5, 2),
            vec![
                1.0, 5.0,
                2.0, 5.0,
                3.0, 5.0,
                4.0, 5.0,
                5.0, 5.0,
            ],
        ).unwrap();

        let detector = DataTypeDetector::new();
        let schema = detector.detect_from_array(&data, None).unwrap();

        // Column 1 is constant
        assert!(schema.columns[1].is_constant);
        assert!(schema.drop_columns.contains(&1));
    }

    #[test]
    fn test_quality_score() {
        let data = Array2::from_shape_vec(
            (100, 2),
            (0..200).map(|i| i as f64).collect(),
        ).unwrap();

        let detector = DataTypeDetector::new();
        let schema = detector.detect_from_array(&data, None).unwrap();

        assert!(schema.quality_score > 0.5);
    }
}
