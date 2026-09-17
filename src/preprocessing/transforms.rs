//! Feature transformation implementations
//!
//! Provides log, power, box-cox, yeo-johnson transforms and binning/discretization.

use crate::error::{AutoMLError, Result};
use polars::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Type of transformation to apply
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum TransformType {
    /// Natural logarithm: log(x)
    Log,
    /// Log with offset: log(x + 1)
    Log1p,
    /// Log base 10
    Log10,
    /// Square root
    Sqrt,
    /// Power transform: x^power
    Power(f64),
    /// Box-Cox transform (requires positive data)
    BoxCox { lambda: Option<f64> },
    /// Yeo-Johnson transform (works with any data)
    YeoJohnson { lambda: Option<f64> },
    /// Reciprocal: 1/x
    Reciprocal,
    /// No transformation
    Identity,
}

impl Default for TransformType {
    fn default() -> Self {
        TransformType::Identity
    }
}

/// Parameters for a fitted transformer
#[derive(Debug, Clone, Serialize, Deserialize)]
struct TransformParams {
    transform_type: TransformType,
    lambda: Option<f64>,  // For Box-Cox/Yeo-Johnson
    shift: f64,           // Shift to make data positive if needed
}

/// Feature transformer for applying mathematical transforms
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Transformer {
    transform_type: TransformType,
    params: HashMap<String, TransformParams>,
    is_fitted: bool,
}

impl Transformer {
    /// Create a new transformer
    pub fn new(transform_type: TransformType) -> Self {
        Self {
            transform_type,
            params: HashMap::new(),
            is_fitted: false,
        }
    }

    /// Fit the transformer to the data
    pub fn fit(&mut self, df: &DataFrame, columns: &[&str]) -> Result<&mut Self> {
        for col_name in columns {
            let column = df
                .column(col_name)
                .map_err(|_| AutoMLError::FeatureNotFound(col_name.to_string()))?;
            let series = column.as_materialized_series();
            
            let values: Vec<f64> = series
                .f64()
                .map_err(|e| AutoMLError::DataError(e.to_string()))?
                .into_iter()
                .filter_map(|v| v)
                .collect();

            let params = self.compute_params(&values)?;
            self.params.insert(col_name.to_string(), params);
        }

        self.is_fitted = true;
        Ok(self)
    }

    /// Compute transformation parameters
    fn compute_params(&self, values: &[f64]) -> Result<TransformParams> {
        let min_val = values.iter().cloned().fold(f64::INFINITY, f64::min);
        
        // Compute shift to ensure positive values for Box-Cox
        let shift = if min_val <= 0.0 {
            -min_val + 1.0
        } else {
            0.0
        };

        let lambda = match &self.transform_type {
            TransformType::BoxCox { lambda: Some(l) } => Some(*l),
            TransformType::BoxCox { lambda: None } => {
                // Estimate optimal lambda using maximum likelihood
                Some(self.estimate_boxcox_lambda(values, shift))
            }
            TransformType::YeoJohnson { lambda: Some(l) } => Some(*l),
            TransformType::YeoJohnson { lambda: None } => {
                Some(self.estimate_yeojohnson_lambda(values))
            }
            _ => None,
        };

        Ok(TransformParams {
            transform_type: self.transform_type.clone(),
            lambda,
            shift,
        })
    }

    /// Estimate optimal Box-Cox lambda using grid search
    fn estimate_boxcox_lambda(&self, values: &[f64], shift: f64) -> f64 {
        let shifted: Vec<f64> = values.iter().map(|&v| v + shift).collect();
        
        let mut best_lambda = 1.0;
        let mut best_ll = f64::NEG_INFINITY;
        
        // Grid search over lambda values
        for lambda_int in -20..=20 {
            let lambda = lambda_int as f64 * 0.1;
            let ll = self.boxcox_log_likelihood(&shifted, lambda);
            if ll > best_ll {
                best_ll = ll;
                best_lambda = lambda;
            }
        }
        
        best_lambda
    }

    /// Box-Cox log-likelihood for parameter estimation
    fn boxcox_log_likelihood(&self, values: &[f64], lambda: f64) -> f64 {
        let n = values.len() as f64;
        
        let transformed: Vec<f64> = values.iter().map(|&x| {
            if lambda.abs() < 1e-10 {
                x.ln()
            } else {
                (x.powf(lambda) - 1.0) / lambda
            }
        }).collect();
        
        let mean = transformed.iter().sum::<f64>() / n;
        let variance = transformed.iter().map(|&t| (t - mean).powi(2)).sum::<f64>() / n;
        
        if variance <= 0.0 {
            return f64::NEG_INFINITY;
        }
        
        let log_jacobian: f64 = values.iter().map(|&x| x.ln()).sum();
        
        -n / 2.0 * variance.ln() + (lambda - 1.0) * log_jacobian
    }

    /// Estimate optimal Yeo-Johnson lambda
    fn estimate_yeojohnson_lambda(&self, values: &[f64]) -> f64 {
        let mut best_lambda = 1.0;
        let mut best_ll = f64::NEG_INFINITY;
        
        for lambda_int in -20..=20 {
            let lambda = lambda_int as f64 * 0.1;
            let ll = self.yeojohnson_log_likelihood(values, lambda);
            if ll > best_ll {
                best_ll = ll;
                best_lambda = lambda;
            }
        }
        
        best_lambda
    }

    /// Yeo-Johnson log-likelihood
    fn yeojohnson_log_likelihood(&self, values: &[f64], lambda: f64) -> f64 {
        let n = values.len() as f64;
        
        let transformed: Vec<f64> = values.iter().map(|&x| {
            self.yeojohnson_transform_value(x, lambda)
        }).collect();
        
        let mean = transformed.iter().sum::<f64>() / n;
        let variance = transformed.iter().map(|&t| (t - mean).powi(2)).sum::<f64>() / n;
        
        if variance <= 0.0 {
            return f64::NEG_INFINITY;
        }
        
        let log_jacobian: f64 = values.iter().map(|&x| {
            (x.abs() + 1.0).ln().copysign(x)
        }).sum();
        
        -n / 2.0 * variance.ln() + (lambda - 1.0) * log_jacobian
    }

    /// Transform the data
    pub fn transform(&self, df: &DataFrame) -> Result<DataFrame> {
        if !self.is_fitted {
            return Err(AutoMLError::ModelNotFitted);
        }

        let mut result = df.clone();

        for (col_name, params) in &self.params {
            if let Ok(column) = df.column(col_name) {
                let series = column.as_materialized_series();
                let transformed = self.transform_series(series, params)?;
                result
                    .with_column(transformed)
                    .map_err(|e| AutoMLError::DataError(e.to_string()))?;
            }
        }

        Ok(result)
    }

    /// Transform a single series
    fn transform_series(&self, series: &Series, params: &TransformParams) -> Result<Series> {
        let name = series.name().to_string();
        let values: Vec<Option<f64>> = series
            .f64()
            .map_err(|e| AutoMLError::DataError(e.to_string()))?
            .into_iter()
            .map(|v| v.map(|x| self.transform_value(x, params)))
            .collect();

        Ok(Series::new(name.into(), values))
    }

    /// Transform a single value
    fn transform_value(&self, x: f64, params: &TransformParams) -> f64 {
        match &params.transform_type {
            TransformType::Log => {
                if x > 0.0 { x.ln() } else { f64::NAN }
            }
            TransformType::Log1p => (x + 1.0).ln(),
            TransformType::Log10 => {
                if x > 0.0 { x.log10() } else { f64::NAN }
            }
            TransformType::Sqrt => {
                if x >= 0.0 { x.sqrt() } else { f64::NAN }
            }
            TransformType::Power(p) => x.powf(*p),
            TransformType::BoxCox { .. } => {
                let shifted = x + params.shift;
                let lambda = params.lambda.unwrap_or(1.0);
                if lambda.abs() < 1e-10 {
                    shifted.ln()
                } else {
                    (shifted.powf(lambda) - 1.0) / lambda
                }
            }
            TransformType::YeoJohnson { .. } => {
                let lambda = params.lambda.unwrap_or(1.0);
                self.yeojohnson_transform_value(x, lambda)
            }
            TransformType::Reciprocal => {
                if x.abs() > 1e-10 { 1.0 / x } else { f64::NAN }
            }
            TransformType::Identity => x,
        }
    }

    /// Yeo-Johnson transform for a single value
    fn yeojohnson_transform_value(&self, x: f64, lambda: f64) -> f64 {
        if x >= 0.0 {
            if lambda.abs() < 1e-10 {
                (x + 1.0).ln()
            } else {
                ((x + 1.0).powf(lambda) - 1.0) / lambda
            }
        } else {
            if (lambda - 2.0).abs() < 1e-10 {
                -((-x + 1.0).ln())
            } else {
                -(((-x + 1.0).powf(2.0 - lambda) - 1.0) / (2.0 - lambda))
            }
        }
    }

    /// Fit and transform in one step
    pub fn fit_transform(&mut self, df: &DataFrame, columns: &[&str]) -> Result<DataFrame> {
        self.fit(df, columns)?;
        self.transform(df)
    }

    /// Inverse transform (where applicable)
    pub fn inverse_transform(&self, df: &DataFrame) -> Result<DataFrame> {
        if !self.is_fitted {
            return Err(AutoMLError::ModelNotFitted);
        }

        let mut result = df.clone();

        for (col_name, params) in &self.params {
            if let Ok(column) = df.column(col_name) {
                let series = column.as_materialized_series();
                let inverse = self.inverse_transform_series(series, params)?;
                result
                    .with_column(inverse)
                    .map_err(|e| AutoMLError::DataError(e.to_string()))?;
            }
        }

        Ok(result)
    }

    /// Inverse transform a single series
    fn inverse_transform_series(&self, series: &Series, params: &TransformParams) -> Result<Series> {
        let name = series.name().to_string();
        let values: Vec<Option<f64>> = series
            .f64()
            .map_err(|e| AutoMLError::DataError(e.to_string()))?
            .into_iter()
            .map(|v| v.map(|y| self.inverse_transform_value(y, params)))
            .collect();

        Ok(Series::new(name.into(), values))
    }

    /// Inverse transform a single value
    fn inverse_transform_value(&self, y: f64, params: &TransformParams) -> f64 {
        match &params.transform_type {
            TransformType::Log => y.exp(),
            TransformType::Log1p => y.exp() - 1.0,
            TransformType::Log10 => 10.0_f64.powf(y),
            TransformType::Sqrt => y.powi(2),
            TransformType::Power(p) => y.powf(1.0 / p),
            TransformType::BoxCox { .. } => {
                let lambda = params.lambda.unwrap_or(1.0);
                let x = if lambda.abs() < 1e-10 {
                    y.exp()
                } else {
                    (y * lambda + 1.0).powf(1.0 / lambda)
                };
                x - params.shift
            }
            TransformType::YeoJohnson { .. } => {
                // Inverse Yeo-Johnson is complex, approximate with numerical methods
                self.inverse_yeojohnson(y, params.lambda.unwrap_or(1.0))
            }
            TransformType::Reciprocal => 1.0 / y,
            TransformType::Identity => y,
        }
    }

    /// Inverse Yeo-Johnson using Newton-Raphson
    fn inverse_yeojohnson(&self, y: f64, lambda: f64) -> f64 {
        // Use Newton-Raphson to find x such that yeo_johnson(x, lambda) = y
        let mut x = y;  // Initial guess
        
        for _ in 0..20 {
            let fx = self.yeojohnson_transform_value(x, lambda) - y;
            if fx.abs() < 1e-10 {
                break;
            }
            
            // Derivative approximation
            let h = 1e-8;
            let dfx = (self.yeojohnson_transform_value(x + h, lambda) - 
                      self.yeojohnson_transform_value(x - h, lambda)) / (2.0 * h);
            
            if dfx.abs() < 1e-10 {
                break;
            }
            
            x = x - fx / dfx;
        }
        
        x
    }
}

// ============================================================================
// Binning / Discretization
// ============================================================================

/// Strategy for creating bins
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum BinningStrategy {
    /// Equal-width bins
    Uniform,
    /// Equal-frequency bins (quantiles)
    Quantile,
    /// K-means based binning
    KMeans,
    /// Custom bin edges
    Custom(Vec<f64>),
}

impl Default for BinningStrategy {
    fn default() -> Self {
        BinningStrategy::Uniform
    }
}

/// Feature binner/discretizer
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Binner {
    strategy: BinningStrategy,
    n_bins: usize,
    bin_edges: HashMap<String, Vec<f64>>,
    encode: BinEncoding,
    is_fitted: bool,
}

/// How to encode binned values
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum BinEncoding {
    /// Return bin index (0, 1, 2, ...)
    Ordinal,
    /// Return bin label as string
    Label,
    /// Return one-hot encoded
    OneHot,
}

impl Default for BinEncoding {
    fn default() -> Self {
        BinEncoding::Ordinal
    }
}

impl Binner {
    /// Create a new binner.
    ///
    /// `n_bins` is clamped to a minimum of 1 (matching the convention used by
    /// e.g. `PolynomialFeatures::new`/`KNNImputer::new`), since `n_bins == 0`
    /// would leave `kmeans_bin_edges` indexing into an empty centroids/clusters
    /// vec on the first value processed.
    pub fn new(strategy: BinningStrategy, n_bins: usize) -> Self {
        Self {
            strategy,
            n_bins: n_bins.max(1),
            bin_edges: HashMap::new(),
            encode: BinEncoding::Ordinal,
            is_fitted: false,
        }
    }

    /// Set encoding type
    pub fn with_encoding(mut self, encode: BinEncoding) -> Self {
        self.encode = encode;
        self
    }

    /// Fit the binner to the data
    pub fn fit(&mut self, df: &DataFrame, columns: &[&str]) -> Result<&mut Self> {
        for col_name in columns {
            let column = df
                .column(col_name)
                .map_err(|_| AutoMLError::FeatureNotFound(col_name.to_string()))?;
            let series = column.as_materialized_series();
            
            let mut values: Vec<f64> = series
                .f64()
                .map_err(|e| AutoMLError::DataError(e.to_string()))?
                .into_iter()
                .filter_map(|v| v)
                .collect();

            let edges = self.compute_bin_edges(&mut values)?;
            self.bin_edges.insert(col_name.to_string(), edges);
        }

        self.is_fitted = true;
        Ok(self)
    }

    /// Compute bin edges based on strategy
    fn compute_bin_edges(&self, values: &mut Vec<f64>) -> Result<Vec<f64>> {
        if values.is_empty() {
            return Err(AutoMLError::DataError("Empty column".to_string()));
        }

        values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        
        let min_val = values[0];
        let max_val = values[values.len() - 1];

        let edges = match &self.strategy {
            BinningStrategy::Uniform => {
                let step = (max_val - min_val) / self.n_bins as f64;
                (0..=self.n_bins)
                    .map(|i| min_val + i as f64 * step)
                    .collect()
            }
            BinningStrategy::Quantile => {
                let mut edges = Vec::with_capacity(self.n_bins + 1);
                edges.push(min_val);
                
                for i in 1..self.n_bins {
                    let q = i as f64 / self.n_bins as f64;
                    let idx = (q * (values.len() - 1) as f64) as usize;
                    edges.push(values[idx]);
                }
                
                edges.push(max_val);
                edges
            }
            BinningStrategy::KMeans => {
                // Simple k-means for binning
                self.kmeans_bin_edges(values)
            }
            BinningStrategy::Custom(custom_edges) => {
                custom_edges.clone()
            }
        };

        Ok(edges)
    }

    /// K-means based bin edge computation
    fn kmeans_bin_edges(&self, values: &[f64]) -> Vec<f64> {
        // Initialize centroids uniformly
        let min_val = values[0];
        let max_val = values[values.len() - 1];
        let step = (max_val - min_val) / self.n_bins as f64;
        
        let mut centroids: Vec<f64> = (0..self.n_bins)
            .map(|i| min_val + (i as f64 + 0.5) * step)
            .collect();

        // Run k-means iterations
        for _ in 0..50 {
            // Assign points to nearest centroid
            let mut clusters: Vec<Vec<f64>> = vec![Vec::new(); self.n_bins];
            
            for &v in values {
                let nearest = centroids
                    .iter()
                    .enumerate()
                    .min_by(|(_, a), (_, b)| {
                        (v - *a).abs().partial_cmp(&(v - *b).abs()).unwrap()
                    })
                    .map(|(i, _)| i)
                    .unwrap_or(0);
                clusters[nearest].push(v);
            }

            // Update centroids
            let mut converged = true;
            for (i, cluster) in clusters.iter().enumerate() {
                if !cluster.is_empty() {
                    let new_centroid = cluster.iter().sum::<f64>() / cluster.len() as f64;
                    if (new_centroid - centroids[i]).abs() > 1e-6 {
                        converged = false;
                    }
                    centroids[i] = new_centroid;
                }
            }

            if converged {
                break;
            }
        }

        // Compute edges as midpoints between sorted centroids
        centroids.sort_by(|a, b| a.partial_cmp(b).unwrap());
        
        let mut edges = vec![min_val];
        for i in 0..centroids.len() - 1 {
            edges.push((centroids[i] + centroids[i + 1]) / 2.0);
        }
        edges.push(max_val);
        
        edges
    }

    /// Transform the data
    pub fn transform(&self, df: &DataFrame) -> Result<DataFrame> {
        if !self.is_fitted {
            return Err(AutoMLError::ModelNotFitted);
        }

        let mut result = df.clone();

        for (col_name, edges) in &self.bin_edges {
            if let Ok(column) = df.column(col_name) {
                let series = column.as_materialized_series();
                let binned = self.bin_series(series, edges)?;
                result
                    .with_column(binned)
                    .map_err(|e| AutoMLError::DataError(e.to_string()))?;
            }
        }

        Ok(result)
    }

    /// Bin a single series
    fn bin_series(&self, series: &Series, edges: &[f64]) -> Result<Series> {
        let name = series.name().to_string();
        
        let values: Vec<Option<f64>> = series
            .f64()
            .map_err(|e| AutoMLError::DataError(e.to_string()))?
            .into_iter()
            .map(|v| v.map(|x| self.find_bin(x, edges) as f64))
            .collect();

        Ok(Series::new(name.into(), values))
    }

    /// Find which bin a value belongs to
    fn find_bin(&self, value: f64, edges: &[f64]) -> usize {
        let n_bins = edges.len().saturating_sub(1);
        for (i, window) in edges.windows(2).enumerate() {
            // Half-open interval [edge_i, edge_{i+1}) for every bin except the
            // last, which is closed on both ends so the maximum value is
            // included. This is the standard quantile-binning convention: a
            // value exactly on an interior edge is assigned to the bin it
            // *starts*, not the one it ends, avoiding a systematic bias toward
            // lower bins when ties sit exactly on a computed boundary.
            let is_last = i == n_bins.saturating_sub(1);
            let in_bin = if is_last {
                value >= window[0] && value <= window[1]
            } else {
                value >= window[0] && value < window[1]
            };
            if in_bin {
                return i;
            }
        }
        // Edge case: value below the first edge or above the last - clamp to
        // the nearest bin.
        if !edges.is_empty() && value < edges[0] {
            0
        } else {
            n_bins.saturating_sub(1)
        }
    }

    /// Fit and transform in one step
    pub fn fit_transform(&mut self, df: &DataFrame, columns: &[&str]) -> Result<DataFrame> {
        self.fit(df, columns)?;
        self.transform(df)
    }

    /// Get bin edges for a column
    pub fn get_bin_edges(&self, column: &str) -> Option<&Vec<f64>> {
        self.bin_edges.get(column)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_df() -> DataFrame {
        DataFrame::new(vec![
            Series::new("values".into(), &[1.0, 2.0, 4.0, 8.0, 16.0, 32.0]).into(),
            Series::new("negative".into(), &[-2.0, -1.0, 0.0, 1.0, 2.0, 3.0]).into(),
        ]).unwrap()
    }

    #[test]
    fn test_log_transform() {
        let df = create_test_df();
        let mut transformer = Transformer::new(TransformType::Log);
        let result = transformer.fit_transform(&df, &["values"]).unwrap();
        
        let transformed = result.column("values").unwrap();
        let vals: Vec<f64> = transformed.f64().unwrap().into_iter().filter_map(|v| v).collect();
        
        assert!((vals[0] - 0.0).abs() < 0.01);  // log(1) = 0
        assert!((vals[1] - 0.693).abs() < 0.01);  // log(2) ≈ 0.693
    }

    #[test]
    fn test_yeojohnson_transform() {
        let df = create_test_df();
        let mut transformer = Transformer::new(TransformType::YeoJohnson { lambda: None });
        let result = transformer.fit_transform(&df, &["negative"]).unwrap();
        
        // Yeo-Johnson should work with negative values
        let transformed = result.column("negative").unwrap();
        let vals: Vec<f64> = transformed.f64().unwrap().into_iter().filter_map(|v| v).collect();
        assert_eq!(vals.len(), 6);
    }

    #[test]
    fn test_uniform_binning() {
        let df = DataFrame::new(vec![
            Series::new("x".into(), &[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]).into(),
        ]).unwrap();

        let mut binner = Binner::new(BinningStrategy::Uniform, 3);
        let result = binner.fit_transform(&df, &["x"]).unwrap();
        
        let binned = result.column("x").unwrap();
        let vals: Vec<f64> = binned.f64().unwrap().into_iter().filter_map(|v| v).collect();
        
        // Values 0-3 should be bin 0, 3-6 bin 1, 6-9 bin 2
        assert_eq!(vals[0], 0.0);  // 0 -> bin 0
        assert_eq!(vals[5], 1.0);  // 5 -> bin 1
        assert_eq!(vals[9], 2.0);  // 9 -> bin 2
    }

    #[test]
    fn test_quantile_binning() {
        let df = DataFrame::new(vec![
            Series::new("x".into(), &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]).into(),
        ]).unwrap();

        let mut binner = Binner::new(BinningStrategy::Quantile, 4);
        let result = binner.fit_transform(&df, &["x"]).unwrap();
        
        let binned = result.column("x").unwrap();
        let vals: Vec<f64> = binned.f64().unwrap().into_iter().filter_map(|v| v).collect();
        
        // Each bin should have roughly equal number of samples
        let bin_counts: Vec<usize> = (0..4).map(|b| vals.iter().filter(|&&v| v == b as f64).count()).collect();
        assert!(bin_counts.iter().all(|&c| c >= 2)); // At least 2 samples per bin
    }
}
