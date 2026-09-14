//! Time series feature engineering

use crate::error::{AutoMLError, Result};
use ndarray::{Array1, Array2};
use serde::{Deserialize, Serialize};

/// Configuration for lag features
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LagConfig {
    /// Lag periods to create
    pub lags: Vec<usize>,
    /// Fill value for missing data
    pub fill_value: f64,
}

impl Default for LagConfig {
    fn default() -> Self {
        Self {
            lags: vec![1, 2, 3, 7, 14],
            fill_value: 0.0,
        }
    }
}

/// Configuration for rolling statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RollingConfig {
    /// Window sizes
    pub windows: Vec<usize>,
    /// Statistics to compute
    pub statistics: Vec<RollingStat>,
    /// Minimum periods for computation
    pub min_periods: usize,
}

impl Default for RollingConfig {
    fn default() -> Self {
        Self {
            windows: vec![3, 7, 14, 30],
            statistics: vec![RollingStat::Mean, RollingStat::Std, RollingStat::Min, RollingStat::Max],
            min_periods: 1,
        }
    }
}

/// Rolling statistics types
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
pub enum RollingStat {
    Mean,
    Std,
    Min,
    Max,
    Sum,
    Median,
    Skew,
    Kurt,
}

/// Configuration for time-based features
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimeFeatureConfig {
    /// Include lag features
    pub include_lags: bool,
    /// Lag configuration
    pub lag_config: LagConfig,
    /// Include rolling features
    pub include_rolling: bool,
    /// Rolling configuration
    pub rolling_config: RollingConfig,
    /// Include date/time features
    pub include_datetime: bool,
    /// Include difference features
    pub include_diff: bool,
    /// Difference orders
    pub diff_orders: Vec<usize>,
    /// Include exponential weighted features
    pub include_ewm: bool,
    /// EWM spans
    pub ewm_spans: Vec<usize>,
}

impl Default for TimeFeatureConfig {
    fn default() -> Self {
        Self {
            include_lags: true,
            lag_config: LagConfig::default(),
            include_rolling: true,
            rolling_config: RollingConfig::default(),
            include_datetime: true,
            include_diff: true,
            diff_orders: vec![1],
            include_ewm: true,
            ewm_spans: vec![7, 14, 30],
        }
    }
}

/// Time series feature generator
#[derive(Debug, Clone)]
pub struct TimeSeriesFeatures {
    config: TimeFeatureConfig,
    feature_names: Vec<String>,
}

impl TimeSeriesFeatures {
    /// Create new feature generator with config
    pub fn new(config: TimeFeatureConfig) -> Self {
        Self {
            config,
            feature_names: Vec::new(),
        }
    }

    /// Create with default config
    pub fn default() -> Self {
        Self::new(TimeFeatureConfig::default())
    }

    /// Generate all features from a single time series
    pub fn transform(&mut self, series: &Array1<f64>) -> Result<Array2<f64>> {
        let n = series.len();
        let mut features: Vec<Array1<f64>> = Vec::new();
        let mut names: Vec<String> = Vec::new();

        // Original series
        features.push(series.clone());
        names.push("original".to_string());

        // Lag features
        if self.config.include_lags {
            for &lag in &self.config.lag_config.lags {
                let lagged = self.create_lag(series, lag);
                features.push(lagged);
                names.push(format!("lag_{}", lag));
            }
        }

        // Rolling features
        if self.config.include_rolling {
            for &window in &self.config.rolling_config.windows {
                for stat in &self.config.rolling_config.statistics {
                    let rolled = self.create_rolling(series, window, *stat)?;
                    features.push(rolled);
                    names.push(format!("rolling_{}_{:?}", window, stat).to_lowercase());
                }
            }
        }

        // Difference features
        if self.config.include_diff {
            for &order in &self.config.diff_orders {
                let diffed = self.create_diff(series, order);
                features.push(diffed);
                names.push(format!("diff_{}", order));
            }
        }

        // Exponential weighted features
        if self.config.include_ewm {
            for &span in &self.config.ewm_spans {
                let ewm = self.create_ewm(series, span);
                features.push(ewm);
                names.push(format!("ewm_{}", span));
            }
        }

        self.feature_names = names;

        // Combine into matrix
        let n_features = features.len();
        let mut result = Array2::zeros((n, n_features));
        
        for (col, feature) in features.into_iter().enumerate() {
            result.column_mut(col).assign(&feature);
        }

        Ok(result)
    }

    /// Get feature names
    pub fn feature_names(&self) -> &[String] {
        &self.feature_names
    }

    /// Create lag feature
    fn create_lag(&self, series: &Array1<f64>, lag: usize) -> Array1<f64> {
        let n = series.len();
        let fill = self.config.lag_config.fill_value;
        
        let mut result = Array1::from_elem(n, fill);
        
        for i in lag..n {
            result[i] = series[i - lag];
        }
        
        result
    }

    /// Create rolling statistic
    fn create_rolling(&self, series: &Array1<f64>, window: usize, stat: RollingStat) -> Result<Array1<f64>> {
        let n = series.len();
        let min_periods = self.config.rolling_config.min_periods;
        let mut result = Array1::zeros(n);

        for i in 0..n {
            let start = if i + 1 >= window { i + 1 - window } else { 0 };
            let values: Vec<f64> = series.slice(ndarray::s![start..=i]).iter().copied().collect();

            if values.len() >= min_periods {
                result[i] = match stat {
                    RollingStat::Mean => values.iter().sum::<f64>() / values.len() as f64,
                    RollingStat::Sum => values.iter().sum(),
                    RollingStat::Min => values.iter().cloned().fold(f64::INFINITY, f64::min),
                    RollingStat::Max => values.iter().cloned().fold(f64::NEG_INFINITY, f64::max),
                    RollingStat::Std => {
                        let mean = values.iter().sum::<f64>() / values.len() as f64;
                        let variance = values.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / values.len() as f64;
                        variance.sqrt()
                    }
                    RollingStat::Median => {
                        let mut sorted = values.clone();
                        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
                        let mid = sorted.len() / 2;
                        if sorted.len() % 2 == 0 {
                            (sorted[mid - 1] + sorted[mid]) / 2.0
                        } else {
                            sorted[mid]
                        }
                    }
                    RollingStat::Skew => {
                        let n = values.len() as f64;
                        if n < 3.0 {
                            0.0
                        } else {
                            let mean = values.iter().sum::<f64>() / n;
                            let std = (values.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / n).sqrt();
                            if std == 0.0 {
                                0.0
                            } else {
                                let skew = values.iter().map(|v| ((v - mean) / std).powi(3)).sum::<f64>() / n;
                                skew
                            }
                        }
                    }
                    RollingStat::Kurt => {
                        let n = values.len() as f64;
                        if n < 4.0 {
                            0.0
                        } else {
                            let mean = values.iter().sum::<f64>() / n;
                            let std = (values.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / n).sqrt();
                            if std == 0.0 {
                                0.0
                            } else {
                                let kurt = values.iter().map(|v| ((v - mean) / std).powi(4)).sum::<f64>() / n - 3.0;
                                kurt
                            }
                        }
                    }
                };
            } else {
                result[i] = f64::NAN;
            }
        }

        Ok(result)
    }

    /// Create difference feature
    fn create_diff(&self, series: &Array1<f64>, order: usize) -> Array1<f64> {
        let n = series.len();
        let mut result = Array1::from_elem(n, f64::NAN);

        for i in order..n {
            result[i] = series[i] - series[i - order];
        }

        result
    }

    /// Create exponentially weighted mean
    fn create_ewm(&self, series: &Array1<f64>, span: usize) -> Array1<f64> {
        let n = series.len();
        let alpha = 2.0 / (span as f64 + 1.0);
        let mut result = Array1::zeros(n);

        result[0] = series[0];
        for i in 1..n {
            result[i] = alpha * series[i] + (1.0 - alpha) * result[i - 1];
        }

        result
    }
}

/// Extract date/time features from timestamps
#[allow(dead_code)]
#[derive(Debug, Clone)]
pub struct DateTimeFeatures {
    /// Include day of week (0-6)
    pub day_of_week: bool,
    /// Include day of month (1-31)
    pub day_of_month: bool,
    /// Include day of year (1-366)
    pub day_of_year: bool,
    /// Include week of year (1-53)
    pub week_of_year: bool,
    /// Include month (1-12)
    pub month: bool,
    /// Include quarter (1-4)
    pub quarter: bool,
    /// Include year
    pub year: bool,
    /// Include hour (0-23)
    pub hour: bool,
    /// Include minute (0-59)
    pub minute: bool,
    /// Include is_weekend flag
    pub is_weekend: bool,
    /// Include cyclical encoding
    pub cyclical: bool,
}

impl Default for DateTimeFeatures {
    fn default() -> Self {
        Self {
            day_of_week: true,
            day_of_month: true,
            day_of_year: false,
            week_of_year: true,
            month: true,
            quarter: true,
            year: true,
            hour: false,
            minute: false,
            is_weekend: true,
            cyclical: true,
        }
    }
}

#[allow(dead_code)]
impl DateTimeFeatures {
    /// Create features from Unix timestamps (seconds)
    pub fn from_timestamps(&self, timestamps: &[i64]) -> Result<Array2<f64>> {
        let n = timestamps.len();
        let mut features: Vec<Vec<f64>> = Vec::new();

        // Calculate features for each timestamp
        for &ts in timestamps {
            let mut row = Vec::new();
            
            // Convert to days since epoch and time components
            let days = ts / 86400;
            let day_of_week = ((days + 4) % 7) as f64; // Epoch was Thursday
            
            // Approximate month/day calculations
            let year_approx = 1970 + (days as f64 / 365.25) as i64;
            let day_of_year = (days % 365) as f64;
            let month_approx = ((day_of_year / 30.44) as i64 + 1).clamp(1, 12) as f64;
            let day_of_month = ((day_of_year % 30.44) as i64 + 1).clamp(1, 31) as f64;
            let week_of_year = (day_of_year / 7.0).floor() + 1.0;
            let quarter = ((month_approx - 1.0) / 3.0).floor() + 1.0;
            let hour = ((ts % 86400) / 3600) as f64;
            let minute = ((ts % 3600) / 60) as f64;
            let is_weekend = if day_of_week >= 5.0 { 1.0 } else { 0.0 };

            if self.day_of_week {
                row.push(day_of_week);
                if self.cyclical {
                    row.push((2.0 * std::f64::consts::PI * day_of_week / 7.0).sin());
                    row.push((2.0 * std::f64::consts::PI * day_of_week / 7.0).cos());
                }
            }
            if self.day_of_month {
                row.push(day_of_month);
                if self.cyclical {
                    row.push((2.0 * std::f64::consts::PI * day_of_month / 31.0).sin());
                    row.push((2.0 * std::f64::consts::PI * day_of_month / 31.0).cos());
                }
            }
            if self.day_of_year {
                row.push(day_of_year);
                if self.cyclical {
                    row.push((2.0 * std::f64::consts::PI * day_of_year / 365.0).sin());
                    row.push((2.0 * std::f64::consts::PI * day_of_year / 365.0).cos());
                }
            }
            if self.week_of_year {
                row.push(week_of_year);
                if self.cyclical {
                    row.push((2.0 * std::f64::consts::PI * week_of_year / 52.0).sin());
                    row.push((2.0 * std::f64::consts::PI * week_of_year / 52.0).cos());
                }
            }
            if self.month {
                row.push(month_approx);
                if self.cyclical {
                    row.push((2.0 * std::f64::consts::PI * month_approx / 12.0).sin());
                    row.push((2.0 * std::f64::consts::PI * month_approx / 12.0).cos());
                }
            }
            if self.quarter {
                row.push(quarter);
            }
            if self.year {
                row.push(year_approx as f64);
            }
            if self.hour {
                row.push(hour);
                if self.cyclical {
                    row.push((2.0 * std::f64::consts::PI * hour / 24.0).sin());
                    row.push((2.0 * std::f64::consts::PI * hour / 24.0).cos());
                }
            }
            if self.minute {
                row.push(minute);
            }
            if self.is_weekend {
                row.push(is_weekend);
            }

            features.push(row);
        }

        if features.is_empty() || features[0].is_empty() {
            return Err(AutoMLError::ValidationError(
                "No features generated".to_string(),
            ));
        }

        let n_features = features[0].len();
        let data: Vec<f64> = features.into_iter().flatten().collect();
        
        Array2::from_shape_vec((n, n_features), data).map_err(|e| {
            AutoMLError::ValidationError(format!("Failed to create feature array: {}", e))
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_lag_features() {
        let config = TimeFeatureConfig {
            include_lags: true,
            lag_config: LagConfig {
                lags: vec![1, 2],
                fill_value: 0.0,
            },
            include_rolling: false,
            include_datetime: false,
            include_diff: false,
            include_ewm: false,
            ..Default::default()
        };

        let mut generator = TimeSeriesFeatures::new(config);
        let series = array![1.0, 2.0, 3.0, 4.0, 5.0];

        let features = generator.transform(&series).unwrap();

        // Original + 2 lags = 3 features
        assert_eq!(features.ncols(), 3);
        
        // Check lag 1: [0, 1, 2, 3, 4]
        assert!((features[[0, 1]] - 0.0).abs() < 1e-6);
        assert!((features[[1, 1]] - 1.0).abs() < 1e-6);
        
        // Check lag 2: [0, 0, 1, 2, 3]
        assert!((features[[0, 2]] - 0.0).abs() < 1e-6);
        assert!((features[[2, 2]] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_rolling_mean() {
        let config = TimeFeatureConfig {
            include_lags: false,
            include_rolling: true,
            rolling_config: RollingConfig {
                windows: vec![3],
                statistics: vec![RollingStat::Mean],
                min_periods: 1,
            },
            include_datetime: false,
            include_diff: false,
            include_ewm: false,
            ..Default::default()
        };

        let mut generator = TimeSeriesFeatures::new(config);
        let series = array![1.0, 2.0, 3.0, 4.0, 5.0];

        let features = generator.transform(&series).unwrap();

        // Rolling mean of window 3 at position 4: (3+4+5)/3 = 4
        assert!((features[[4, 1]] - 4.0).abs() < 1e-6);
    }

    #[test]
    fn test_diff() {
        let config = TimeFeatureConfig {
            include_lags: false,
            include_rolling: false,
            include_datetime: false,
            include_diff: true,
            diff_orders: vec![1],
            include_ewm: false,
            ..Default::default()
        };

        let mut generator = TimeSeriesFeatures::new(config);
        let series = array![1.0, 3.0, 6.0, 10.0];

        let features = generator.transform(&series).unwrap();

        // Diff: [NaN, 2, 3, 4]
        assert!((features[[1, 1]] - 2.0).abs() < 1e-6);
        assert!((features[[2, 1]] - 3.0).abs() < 1e-6);
        assert!((features[[3, 1]] - 4.0).abs() < 1e-6);
    }

    #[test]
    fn test_ewm() {
        let config = TimeFeatureConfig {
            include_lags: false,
            include_rolling: false,
            include_datetime: false,
            include_diff: false,
            include_ewm: true,
            ewm_spans: vec![3],
            ..Default::default()
        };

        let mut generator = TimeSeriesFeatures::new(config);
        let series = array![1.0, 2.0, 3.0, 4.0, 5.0];

        let features = generator.transform(&series).unwrap();

        // EWM should smooth the values
        assert_eq!(features.ncols(), 2); // original + 1 ewm
        assert!(features[[0, 1]] == 1.0); // First value is same
    }
}
