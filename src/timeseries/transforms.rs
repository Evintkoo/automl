//! Time series transformations

use crate::error::{AutoMLError, Result};
use ndarray::Array1;
use serde::{Deserialize, Serialize};

/// Differencing transformer
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Differencer {
    /// Order of differencing
    order: usize,
    /// Seasonal period (0 = no seasonal differencing)
    seasonal_period: usize,
    /// Stored values for inverse transform
    initial_values: Option<Vec<f64>>,
}

impl Differencer {
    /// Create new differencer
    pub fn new(order: usize) -> Self {
        Self {
            order: order.max(1),
            seasonal_period: 0,
            initial_values: None,
        }
    }

    /// Add seasonal differencing
    pub fn with_seasonal(mut self, period: usize) -> Self {
        self.seasonal_period = period;
        self
    }

    /// Apply differencing
    pub fn transform(&mut self, series: &Array1<f64>) -> Result<Array1<f64>> {
        let mut result = series.clone();
        let mut initial = Vec::new();

        // Regular differencing
        for _ in 0..self.order {
            initial.push(result[0]);
            result = self.diff_once(&result);
        }

        // Seasonal differencing
        if self.seasonal_period > 0 {
            for i in 0..self.seasonal_period.min(result.len()) {
                initial.push(result[i]);
            }
            result = self.seasonal_diff(&result);
        }

        self.initial_values = Some(initial);
        Ok(result)
    }

    /// Inverse differencing
    pub fn inverse_transform(&self, series: &Array1<f64>) -> Result<Array1<f64>> {
        let initial = self.initial_values.as_ref().ok_or_else(|| {
            AutoMLError::ValidationError("Differencer not fitted".to_string())
        })?;

        let mut result = series.clone();
        let mut init_idx = initial.len();

        // Inverse seasonal differencing
        if self.seasonal_period > 0 {
            init_idx -= self.seasonal_period.min(initial.len());
            let seasonal_init: Vec<f64> = initial[init_idx..].to_vec();
            result = self.inverse_seasonal_diff(&result, &seasonal_init);
        }

        // Inverse regular differencing
        for _i in (0..self.order).rev() {
            init_idx = init_idx.saturating_sub(1);
            let init_val = initial.get(init_idx).copied().unwrap_or(0.0);
            result = self.cumsum(&result, init_val);
        }

        Ok(result)
    }

    fn diff_once(&self, series: &Array1<f64>) -> Array1<f64> {
        let n = series.len();
        if n <= 1 {
            return Array1::zeros(0);
        }

        let mut result = Array1::zeros(n - 1);
        for i in 1..n {
            result[i - 1] = series[i] - series[i - 1];
        }
        result
    }

    fn seasonal_diff(&self, series: &Array1<f64>) -> Array1<f64> {
        let n = series.len();
        let period = self.seasonal_period;
        
        if n <= period {
            return Array1::zeros(0);
        }

        let mut result = Array1::zeros(n - period);
        for i in period..n {
            result[i - period] = series[i] - series[i - period];
        }
        result
    }

    fn cumsum(&self, series: &Array1<f64>, init: f64) -> Array1<f64> {
        let n = series.len();
        let mut result = Array1::zeros(n + 1);
        result[0] = init;

        for i in 0..n {
            result[i + 1] = result[i] + series[i];
        }
        result
    }

    fn inverse_seasonal_diff(&self, series: &Array1<f64>, init: &[f64]) -> Array1<f64> {
        let n = series.len();
        let period = self.seasonal_period;
        let mut result = Array1::zeros(n + period);

        // Fill initial values
        for (i, &v) in init.iter().enumerate() {
            result[i] = v;
        }

        // Reconstruct
        for i in 0..n {
            result[i + period] = series[i] + result[i];
        }
        result
    }
}

/// Seasonal decomposition methods
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
pub enum DecomposeMethod {
    /// Additive: y = trend + seasonal + residual
    Additive,
    /// Multiplicative: y = trend * seasonal * residual
    Multiplicative,
}

/// Seasonal decomposition result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DecomposeResult {
    /// Original series
    pub observed: Array1<f64>,
    /// Trend component
    pub trend: Array1<f64>,
    /// Seasonal component
    pub seasonal: Array1<f64>,
    /// Residual component
    pub residual: Array1<f64>,
    /// Decomposition method used
    pub method: DecomposeMethod,
}

/// Seasonal decomposer
#[derive(Debug, Clone)]
pub struct SeasonalDecomposer {
    /// Period of seasonality
    period: usize,
    /// Decomposition method
    method: DecomposeMethod,
    /// Seasonal pattern (fitted)
    seasonal_pattern: Option<Vec<f64>>,
}

impl SeasonalDecomposer {
    /// Create new decomposer
    pub fn new(period: usize, method: DecomposeMethod) -> Self {
        Self {
            period: period.max(2),
            method,
            seasonal_pattern: None,
        }
    }

    /// Decompose a time series
    pub fn decompose(&mut self, series: &Array1<f64>) -> Result<DecomposeResult> {
        let n = series.len();
        
        if n < self.period * 2 {
            return Err(AutoMLError::ValidationError(
                "Series too short for decomposition".to_string(),
            ));
        }

        // Extract trend using centered moving average
        let trend = self.moving_average(series);

        // Compute de-trended series
        let detrended = match self.method {
            DecomposeMethod::Additive => {
                let mut dt = Array1::zeros(n);
                for i in 0..n {
                    dt[i] = series[i] - trend[i];
                }
                dt
            }
            DecomposeMethod::Multiplicative => {
                let mut dt = Array1::zeros(n);
                for i in 0..n {
                    dt[i] = if trend[i].abs() > 1e-10 {
                        series[i] / trend[i]
                    } else {
                        1.0
                    };
                }
                dt
            }
        };

        // Compute seasonal component
        let seasonal = self.compute_seasonal(&detrended);
        self.seasonal_pattern = Some(
            (0..self.period).map(|i| seasonal[i]).collect()
        );

        // Compute residual
        let residual = match self.method {
            DecomposeMethod::Additive => {
                let mut res = Array1::zeros(n);
                for i in 0..n {
                    res[i] = series[i] - trend[i] - seasonal[i];
                }
                res
            }
            DecomposeMethod::Multiplicative => {
                let mut res = Array1::zeros(n);
                for i in 0..n {
                    let denom = trend[i] * seasonal[i];
                    res[i] = if denom.abs() > 1e-10 {
                        series[i] / denom
                    } else {
                        1.0
                    };
                }
                res
            }
        };

        Ok(DecomposeResult {
            observed: series.clone(),
            trend,
            seasonal,
            residual,
            method: self.method,
        })
    }

    /// Remove seasonality from series
    pub fn deseasonalize(&self, series: &Array1<f64>) -> Result<Array1<f64>> {
        let pattern = self.seasonal_pattern.as_ref().ok_or_else(|| {
            AutoMLError::ValidationError("Decomposer not fitted".to_string())
        })?;

        let n = series.len();
        let mut result = Array1::zeros(n);

        for i in 0..n {
            let seasonal_val = pattern[i % self.period];
            result[i] = match self.method {
                DecomposeMethod::Additive => series[i] - seasonal_val,
                DecomposeMethod::Multiplicative => {
                    if seasonal_val.abs() > 1e-10 {
                        series[i] / seasonal_val
                    } else {
                        series[i]
                    }
                }
            };
        }

        Ok(result)
    }

    /// Add seasonality back to series
    pub fn reseasonalize(&self, series: &Array1<f64>) -> Result<Array1<f64>> {
        let pattern = self.seasonal_pattern.as_ref().ok_or_else(|| {
            AutoMLError::ValidationError("Decomposer not fitted".to_string())
        })?;

        let n = series.len();
        let mut result = Array1::zeros(n);

        for i in 0..n {
            let seasonal_val = pattern[i % self.period];
            result[i] = match self.method {
                DecomposeMethod::Additive => series[i] + seasonal_val,
                DecomposeMethod::Multiplicative => series[i] * seasonal_val,
            };
        }

        Ok(result)
    }

    fn moving_average(&self, series: &Array1<f64>) -> Array1<f64> {
        let n = series.len();
        let window = self.period;
        let mut result = Array1::zeros(n);

        // Centered moving average
        let half = window / 2;
        
        for i in 0..n {
            let start = if i >= half { i - half } else { 0 };
            let end = (i + half + 1).min(n);
            
            let sum: f64 = series.slice(ndarray::s![start..end]).sum();
            let count = end - start;
            result[i] = sum / count as f64;
        }

        result
    }

    fn compute_seasonal(&self, detrended: &Array1<f64>) -> Array1<f64> {
        let n = detrended.len();
        let period = self.period;

        // Average values at each seasonal position
        let mut seasonal_means = vec![0.0; period];
        let mut counts = vec![0usize; period];

        for i in 0..n {
            let pos = i % period;
            seasonal_means[pos] += detrended[i];
            counts[pos] += 1;
        }

        for i in 0..period {
            if counts[i] > 0 {
                seasonal_means[i] /= counts[i] as f64;
            }
        }

        // Normalize seasonal component
        let mean_seasonal: f64 = seasonal_means.iter().sum::<f64>() / period as f64;
        
        match self.method {
            DecomposeMethod::Additive => {
                for s in &mut seasonal_means {
                    *s -= mean_seasonal;
                }
            }
            DecomposeMethod::Multiplicative => {
                for s in &mut seasonal_means {
                    *s /= mean_seasonal;
                }
            }
        }

        // Extend to full series length
        let mut result = Array1::zeros(n);
        for i in 0..n {
            result[i] = seasonal_means[i % period];
        }
        result
    }
}

/// General time series transformer
pub struct TimeSeriesTransformer {
    /// Whether series is log-transformed
    log_transform: bool,
    /// Box-Cox lambda
    box_cox_lambda: Option<f64>,
    /// Differencer
    differencer: Option<Differencer>,
    /// Seasonal decomposer
    decomposer: Option<SeasonalDecomposer>,
}

impl TimeSeriesTransformer {
    /// Create new transformer
    pub fn new() -> Self {
        Self {
            log_transform: false,
            box_cox_lambda: None,
            differencer: None,
            decomposer: None,
        }
    }

    /// Enable log transformation
    pub fn with_log_transform(mut self) -> Self {
        self.log_transform = true;
        self
    }

    /// Enable Box-Cox transformation
    pub fn with_box_cox(mut self, lambda: f64) -> Self {
        self.box_cox_lambda = Some(lambda);
        self
    }

    /// Enable differencing
    pub fn with_differencing(mut self, order: usize, seasonal_period: Option<usize>) -> Self {
        let mut diff = Differencer::new(order);
        if let Some(period) = seasonal_period {
            diff = diff.with_seasonal(period);
        }
        self.differencer = Some(diff);
        self
    }

    /// Enable seasonal decomposition
    pub fn with_decomposition(mut self, period: usize, method: DecomposeMethod) -> Self {
        self.decomposer = Some(SeasonalDecomposer::new(period, method));
        self
    }

    /// Apply all transformations
    pub fn transform(&mut self, series: &Array1<f64>) -> Result<Array1<f64>> {
        let mut result = series.clone();

        // Log transform
        if self.log_transform {
            result = result.mapv(|x| (x + 1.0).ln());
        }

        // Box-Cox transform
        if let Some(lambda) = self.box_cox_lambda {
            result = result.mapv(|x| self.box_cox_transform(x, lambda));
        }

        // Differencing
        if let Some(ref mut diff) = self.differencer {
            result = diff.transform(&result)?;
        }

        Ok(result)
    }

    /// Inverse transform
    pub fn inverse_transform(&self, series: &Array1<f64>) -> Result<Array1<f64>> {
        let mut result = series.clone();

        // Inverse differencing
        if let Some(ref diff) = self.differencer {
            result = diff.inverse_transform(&result)?;
        }

        // Inverse Box-Cox
        if let Some(lambda) = self.box_cox_lambda {
            result = result.mapv(|x| self.inverse_box_cox(x, lambda));
        }

        // Inverse log
        if self.log_transform {
            result = result.mapv(|x| x.exp() - 1.0);
        }

        Ok(result)
    }

    fn box_cox_transform(&self, x: f64, lambda: f64) -> f64 {
        if lambda.abs() < 1e-10 {
            (x + 1.0).ln()
        } else {
            ((x + 1.0).powf(lambda) - 1.0) / lambda
        }
    }

    fn inverse_box_cox(&self, y: f64, lambda: f64) -> f64 {
        if lambda.abs() < 1e-10 {
            y.exp() - 1.0
        } else {
            (y * lambda + 1.0).powf(1.0 / lambda) - 1.0
        }
    }
}

impl Default for TimeSeriesTransformer {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_differencer() {
        let mut diff = Differencer::new(1);
        let series = array![1.0, 3.0, 6.0, 10.0, 15.0];

        let diffed = diff.transform(&series).unwrap();
        
        // Differences: 2, 3, 4, 5
        assert!((diffed[0] - 2.0).abs() < 1e-6);
        assert!((diffed[1] - 3.0).abs() < 1e-6);
        assert!((diffed[2] - 4.0).abs() < 1e-6);
        assert!((diffed[3] - 5.0).abs() < 1e-6);

        // Inverse
        let recovered = diff.inverse_transform(&diffed).unwrap();
        assert!((recovered[0] - 1.0).abs() < 1e-6);
        assert!((recovered[4] - 15.0).abs() < 1e-6);
    }

    #[test]
    fn test_seasonal_decomposition() {
        // Create series with clear seasonality (period 4)
        let mut series_data = Vec::new();
        for i in 0..20 {
            let trend = i as f64;
            let seasonal = [10.0, 20.0, 15.0, 5.0][i % 4];
            series_data.push(trend + seasonal);
        }
        let series = Array1::from_vec(series_data);

        let mut decomposer = SeasonalDecomposer::new(4, DecomposeMethod::Additive);
        let result = decomposer.decompose(&series).unwrap();

        assert_eq!(result.trend.len(), 20);
        assert_eq!(result.seasonal.len(), 20);
        assert_eq!(result.residual.len(), 20);
    }

    #[test]
    fn test_transformer_chain() {
        let mut transformer = TimeSeriesTransformer::new()
            .with_log_transform()
            .with_differencing(1, None);

        let series = array![1.0, 2.0, 4.0, 8.0, 16.0];
        let transformed = transformer.transform(&series).unwrap();
        
        // Should be differenced log values
        assert!(transformed.len() < series.len());
    }
}
