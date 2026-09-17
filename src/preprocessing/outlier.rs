//! Outlier detection and handling
//!
//! Provides methods to detect and handle outliers in numeric data.

use crate::error::{AutoMLError, Result};
use polars::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Minimum spread (IQR or MAD) to use when computing outlier bounds. Mirrors
/// `SCALE_EPSILON` in `scaler.rs`: without this floor, a degenerate/single-row
/// fit collapses IQR or MAD to zero, so bounds become `(x, x)` and any future
/// value that merely differs from the fitted `x` gets flagged as an outlier.
const OUTLIER_SCALE_EPSILON: f64 = 1e-12;

/// Method for outlier detection
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum OutlierMethod {
    /// Interquartile Range method
    IQR { factor: f64 },
    /// Z-score (standard deviations from mean)
    ZScore { threshold: f64 },
    /// Percentile-based bounds
    Percentile { lower: f64, upper: f64 },
    /// Modified Z-score using median
    ModifiedZScore { threshold: f64 },
}

impl Default for OutlierMethod {
    fn default() -> Self {
        OutlierMethod::IQR { factor: 1.5 }
    }
}

/// Strategy for handling detected outliers
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum OutlierStrategy {
    /// Clip outliers to the boundary values
    Clip,
    /// Replace outliers with NaN (can then be imputed)
    ToNaN,
    /// Remove rows containing outliers
    Remove,
    /// Replace with mean
    ReplaceWithMean,
    /// Replace with median
    ReplaceWithMedian,
    /// Do nothing (detect only)
    None,
}

impl Default for OutlierStrategy {
    fn default() -> Self {
        OutlierStrategy::Clip
    }
}

/// Compute the median of an already-sorted slice.
///
/// For an even-length slice this averages the two middle elements rather than
/// returning the upper-middle one (`sorted[len/2]` alone is biased high).
fn median_of_sorted(sorted: &[f64]) -> f64 {
    let n = sorted.len();
    if n == 0 {
        return 0.0;
    }
    if n % 2 == 0 {
        (sorted[n / 2 - 1] + sorted[n / 2]) / 2.0
    } else {
        sorted[n / 2]
    }
}

/// Fitted bounds for a column
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OutlierBounds {
    pub lower: f64,
    pub upper: f64,
    pub mean: f64,
    pub median: f64,
}

/// Outlier detector and handler
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OutlierDetector {
    method: OutlierMethod,
    strategy: OutlierStrategy,
    columns: Option<Vec<String>>,
    bounds: HashMap<String, OutlierBounds>,
    is_fitted: bool,
}

impl OutlierDetector {
    /// Create a new outlier detector
    pub fn new(method: OutlierMethod, strategy: OutlierStrategy) -> Self {
        Self {
            method,
            strategy,
            columns: None,
            bounds: HashMap::new(),
            is_fitted: false,
        }
    }

    /// Create with IQR method
    pub fn iqr(factor: f64) -> Self {
        Self::new(
            OutlierMethod::IQR { factor },
            OutlierStrategy::Clip,
        )
    }

    /// Create with Z-score method
    pub fn zscore(threshold: f64) -> Self {
        Self::new(
            OutlierMethod::ZScore { threshold },
            OutlierStrategy::Clip,
        )
    }

    /// Set the handling strategy
    pub fn with_strategy(mut self, strategy: OutlierStrategy) -> Self {
        self.strategy = strategy;
        self
    }

    /// Set specific columns to process
    pub fn with_columns(mut self, columns: Vec<String>) -> Self {
        self.columns = Some(columns);
        self
    }

    /// Fit the detector to compute bounds
    pub fn fit(&mut self, df: &DataFrame) -> Result<&mut Self> {
        let columns: Vec<String> = match &self.columns {
            Some(cols) => cols.clone(),
            None => df
                .get_column_names()
                .iter()
                .filter(|name| {
                    df.column(name.as_str())
                        .map(|s| s.dtype().is_primitive_numeric())
                        .unwrap_or(false)
                })
                .map(|s| s.to_string())
                .collect(),
        };

        for col_name in &columns {
            let series = df
                .column(col_name.as_str())
                .map_err(|_| AutoMLError::FeatureNotFound(col_name.clone()))?;

            if let Ok(ca) = series.f64() {
                let bounds = self.compute_bounds(&ca)?;
                self.bounds.insert(col_name.clone(), bounds);
            }
        }

        self.is_fitted = true;
        Ok(self)
    }

    /// Transform the data by handling outliers
    pub fn transform(&self, df: &DataFrame) -> Result<DataFrame> {
        if !self.is_fitted {
            return Err(AutoMLError::ModelNotFitted);
        }

        if matches!(self.strategy, OutlierStrategy::None) {
            return Ok(df.clone());
        }

        let mut result = df.clone();

        for (col_name, bounds) in &self.bounds {
            if let Ok(series) = df.column(col_name.as_str()) {
                let transformed = self.transform_column(&series.as_series().unwrap(), bounds)?;
                result = result
                    .drop(col_name.as_str())
                    .map_err(|e| AutoMLError::DataError(e.to_string()))?;
                result
                    .hstack_mut(&[transformed.into()])
                    .map_err(|e| AutoMLError::DataError(e.to_string()))?;
            }
        }

        Ok(result)
    }

    /// Fit and transform in one step
    pub fn fit_transform(&mut self, df: &DataFrame) -> Result<DataFrame> {
        self.fit(df)?;
        self.transform(df)
    }

    /// Detect outliers and return a boolean mask
    pub fn detect(&self, df: &DataFrame) -> Result<DataFrame> {
        if !self.is_fitted {
            return Err(AutoMLError::ModelNotFitted);
        }

        let mut masks: Vec<Column> = Vec::new();

        for (col_name, bounds) in &self.bounds {
            if let Ok(col) = df.column(col_name.as_str()) {
                let mask = self.detect_in_column(&col.as_series().unwrap(), bounds)?;
                masks.push(Series::new(format!("{}_outlier", col_name).into(), mask).into());
            }
        }

        DataFrame::new(masks).map_err(|e| AutoMLError::DataError(e.to_string()))
    }

    /// Get the computed bounds
    pub fn bounds(&self) -> &HashMap<String, OutlierBounds> {
        &self.bounds
    }

    fn compute_bounds(&self, ca: &ChunkedArray<Float64Type>) -> Result<OutlierBounds> {
        let values: Vec<f64> = ca.into_iter().filter_map(|v| v).collect();
        
        if values.is_empty() {
            return Ok(OutlierBounds {
                lower: f64::NEG_INFINITY,
                upper: f64::INFINITY,
                mean: 0.0,
                median: 0.0,
            });
        }

        let mut sorted = values.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let mean = values.iter().sum::<f64>() / values.len() as f64;
        let median = median_of_sorted(&sorted);

        let (lower, upper) = match &self.method {
            OutlierMethod::IQR { factor } => {
                let q1_idx = sorted.len() / 4;
                let q3_idx = 3 * sorted.len() / 4;
                let q1 = sorted[q1_idx];
                let q3 = sorted[q3_idx];
                let iqr_raw = q3 - q1;
                let iqr = if iqr_raw.abs() < OUTLIER_SCALE_EPSILON {
                    OUTLIER_SCALE_EPSILON
                } else {
                    iqr_raw
                };
                (q1 - factor * iqr, q3 + factor * iqr)
            }
            OutlierMethod::ZScore { threshold } => {
                let std = (values.iter().map(|x| (x - mean).powi(2)).sum::<f64>()
                    / values.len() as f64)
                    .sqrt();
                (mean - threshold * std, mean + threshold * std)
            }
            OutlierMethod::Percentile { lower, upper } => {
                let lower_idx = ((sorted.len() as f64) * lower / 100.0).floor() as usize;
                let upper_idx = ((sorted.len() as f64) * upper / 100.0).ceil() as usize;
                (
                    sorted[lower_idx.min(sorted.len() - 1)],
                    sorted[upper_idx.min(sorted.len() - 1)],
                )
            }
            OutlierMethod::ModifiedZScore { threshold } => {
                // Modified Z-score uses median and MAD
                let mad_raw: f64 = sorted
                    .iter()
                    .map(|x| (x - median).abs())
                    .sum::<f64>()
                    / sorted.len() as f64;
                let mad = if mad_raw.abs() < OUTLIER_SCALE_EPSILON {
                    OUTLIER_SCALE_EPSILON
                } else {
                    mad_raw
                };
                let k = 1.4826; // Consistency constant for normal distribution
                (
                    median - threshold * k * mad,
                    median + threshold * k * mad,
                )
            }
        };

        Ok(OutlierBounds { lower, upper, mean, median })
    }

    fn transform_column(&self, series: &Series, bounds: &OutlierBounds) -> Result<Series> {
        let ca = series
            .f64()
            .map_err(|e| AutoMLError::DataError(e.to_string()))?;

        let transformed: Vec<Option<f64>> = ca
            .into_iter()
            .map(|opt_v| {
                opt_v.map(|v| {
                    if v < bounds.lower || v > bounds.upper {
                        match self.strategy {
                            OutlierStrategy::Clip => v.max(bounds.lower).min(bounds.upper),
                            OutlierStrategy::ToNaN => f64::NAN,
                            OutlierStrategy::ReplaceWithMean => bounds.mean,
                            OutlierStrategy::ReplaceWithMedian => bounds.median,
                            OutlierStrategy::Remove | OutlierStrategy::None => v,
                        }
                    } else {
                        v
                    }
                })
            })
            .collect();

        Ok(Series::new(series.name().clone(), transformed))
    }

    fn detect_in_column(&self, series: &Series, bounds: &OutlierBounds) -> Result<Vec<bool>> {
        let ca = series
            .f64()
            .map_err(|e| AutoMLError::DataError(e.to_string()))?;

        let mask: Vec<bool> = ca
            .into_iter()
            .map(|opt_v| {
                opt_v
                    .map(|v| v < bounds.lower || v > bounds.upper)
                    .unwrap_or(false)
            })
            .collect();

        Ok(mask)
    }
}

impl Default for OutlierDetector {
    fn default() -> Self {
        Self::iqr(1.5)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_df() -> DataFrame {
        DataFrame::new(vec![
            Series::new("normal".into(), &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]).into(),
            Series::new("with_outliers".into(), &[1.0, 2.0, 3.0, 4.0, 5.0, 100.0, 7.0, 8.0, 9.0, -50.0]).into(),
        ])
        .unwrap()
    }

    #[test]
    fn test_iqr_detection() {
        let df = create_test_df();
        
        let mut detector = OutlierDetector::iqr(1.5);
        detector.fit(&df).unwrap();
        
        let outliers = detector.detect(&df).unwrap();
        assert!(outliers.height() > 0);
    }

    #[test]
    fn test_zscore_detection() {
        let df = create_test_df();
        
        let mut detector = OutlierDetector::zscore(2.0);
        detector.fit(&df).unwrap();
        
        let bounds = detector.bounds();
        assert!(bounds.contains_key("with_outliers"));
    }

    #[test]
    fn test_clip_strategy() {
        let df = create_test_df();
        
        let mut detector = OutlierDetector::iqr(1.5)
            .with_strategy(OutlierStrategy::Clip)
            .with_columns(vec!["with_outliers".to_string()]);
        
        let transformed = detector.fit_transform(&df).unwrap();
        
        let col = transformed.column("with_outliers").unwrap();
        let values: Vec<f64> = col.f64().unwrap().into_iter().filter_map(|v| v).collect();
        
        // The extreme outliers should be clipped
        assert!(values.iter().all(|&v| v >= -20.0 && v <= 20.0));
    }

    #[test]
    fn test_percentile_method() {
        let df = create_test_df();
        
        let mut detector = OutlierDetector::new(
            OutlierMethod::Percentile { lower: 5.0, upper: 95.0 },
            OutlierStrategy::Clip,
        );
        
        detector.fit(&df).unwrap();
        let bounds = detector.bounds();
        
        assert!(bounds.get("normal").is_some());
    }

    #[test]
    fn test_bounds_computation() {
        let df = DataFrame::new(vec![
            Series::new("test".into(), &[1.0, 2.0, 3.0, 4.0, 5.0]).into(),
        ])
        .unwrap();

        let mut detector = OutlierDetector::iqr(1.5);
        detector.fit(&df).unwrap();

        let bounds = detector.bounds().get("test").unwrap();
        assert!((bounds.mean - 3.0).abs() < 0.01);
        assert!((bounds.median - 3.0).abs() < 0.01);
    }
}
