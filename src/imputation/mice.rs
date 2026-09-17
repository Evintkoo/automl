//! MICE (Multiple Imputation by Chained Equations) imputer

use crate::error::{AutoMLError, Result};
use crate::imputation::{Imputer, InitialStrategy, is_missing};
use ndarray::{Array1, Array2};
use rand::prelude::*;
use serde::{Deserialize, Serialize};

/// Compute the median of an already-sorted slice, averaging the two middle
/// elements for an even-length slice instead of returning the upper-middle one.
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

/// MICE imputer using chained equations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MICEImputer {
    /// Maximum number of iterations
    max_iter: usize,
    /// Convergence tolerance
    tol: f64,
    /// Initial imputation strategy
    initial_strategy: InitialStrategy,
    /// Random seed
    seed: Option<u64>,
    /// Fitted statistics per feature
    feature_stats: Option<Vec<FeatureStats>>,
    /// Imputation order
    imputation_order: Option<Vec<usize>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct FeatureStats {
    mean: f64,
    median: f64,
    std: f64,
    missing_mask_fit: Vec<bool>,
    coefficients: Option<Vec<f64>>,
    intercept: f64,
}

impl MICEImputer {
    /// Create new MICE imputer
    pub fn new() -> Self {
        Self {
            max_iter: 10,
            tol: 1e-3,
            initial_strategy: InitialStrategy::Mean,
            seed: None,
            feature_stats: None,
            imputation_order: None,
        }
    }

    /// Set maximum iterations
    pub fn with_max_iter(mut self, max_iter: usize) -> Self {
        self.max_iter = max_iter.max(1);
        self
    }

    /// Set convergence tolerance
    pub fn with_tolerance(mut self, tol: f64) -> Self {
        self.tol = tol.max(1e-10);
        self
    }

    /// Set initial strategy
    pub fn with_initial_strategy(mut self, strategy: InitialStrategy) -> Self {
        self.initial_strategy = strategy;
        self
    }

    /// Set random seed
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = Some(seed);
        self
    }

    /// Compute initial imputation value
    fn initial_value(&self, column: &[f64]) -> f64 {
        let observed: Vec<f64> = column.iter()
            .filter(|v| !is_missing(**v))
            .copied()
            .collect();

        if observed.is_empty() {
            return 0.0;
        }

        match self.initial_strategy {
            InitialStrategy::Mean => {
                observed.iter().sum::<f64>() / observed.len() as f64
            }
            InitialStrategy::Median => {
                let mut sorted = observed.clone();
                sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
                median_of_sorted(&sorted)
            }
            InitialStrategy::MostFrequent => {
                // For continuous data, use mode approximation (most common bin)
                let mean = observed.iter().sum::<f64>() / observed.len() as f64;
                mean
            }
            InitialStrategy::Constant => 0.0,
        }
    }

    /// Simple linear regression for imputation
    fn fit_linear_regression(x: &Array2<f64>, y: &Array1<f64>) -> (Vec<f64>, f64) {
        let n = x.nrows() as f64;
        let p = x.ncols();

        if n < 2.0 || p == 0 {
            return (vec![0.0; p], y.mean().unwrap_or(0.0));
        }

        // Compute means
        let y_mean = y.mean().unwrap_or(0.0);
        let x_means: Vec<f64> = (0..p)
            .map(|j| x.column(j).mean().unwrap_or(0.0))
            .collect();

        // Center the data
        let y_centered: Vec<f64> = y.iter().map(|&yi| yi - y_mean).collect();
        
        // Compute coefficients using normal equations (simplified)
        let mut coefficients = vec![0.0; p];
        
        for j in 0..p {
            let x_col: Vec<f64> = x.column(j).iter().copied().collect();
            let x_centered: Vec<f64> = x_col.iter().map(|&xi| xi - x_means[j]).collect();
            
            let numerator: f64 = x_centered.iter()
                .zip(y_centered.iter())
                .map(|(&xi, &yi)| xi * yi)
                .sum();
            
            let denominator: f64 = x_centered.iter()
                .map(|&xi| xi * xi)
                .sum::<f64>()
                .max(1e-10);
            
            coefficients[j] = numerator / denominator;
        }

        // Compute intercept
        let intercept = y_mean - coefficients.iter()
            .zip(x_means.iter())
            .map(|(&c, &m)| c * m)
            .sum::<f64>();

        (coefficients, intercept)
    }

    /// Predict using linear regression
    fn predict_linear(x: &Array2<f64>, coefficients: &[f64], intercept: f64) -> Array1<f64> {
        let n = x.nrows();
        let mut predictions = Array1::zeros(n);

        for i in 0..n {
            let mut pred = intercept;
            for (j, &coef) in coefficients.iter().enumerate() {
                pred += coef * x[[i, j]];
            }
            predictions[i] = pred;
        }

        predictions
    }

    /// Perform one iteration of MICE
    fn mice_iteration(&self, data: &mut Array2<f64>, rng: &mut StdRng) -> f64 {
        let n_features = data.ncols();
        let mut total_change = 0.0;

        for target_col in 0..n_features {
            // Find missing indices for this column
            let missing_mask: Vec<bool> = data.column(target_col)
                .iter()
                .map(|&v| is_missing(v))
                .collect();

            let n_missing = missing_mask.iter().filter(|&&m| m).count();
            if n_missing == 0 {
                continue;
            }

            // Create feature matrix (all other columns) and target
            let observed_indices: Vec<usize> = missing_mask.iter()
                .enumerate()
                .filter(|(_, &m)| !m)
                .map(|(i, _)| i)
                .collect();

            let missing_indices: Vec<usize> = missing_mask.iter()
                .enumerate()
                .filter(|(_, &m)| m)
                .map(|(i, _)| i)
                .collect();

            if observed_indices.is_empty() {
                continue;
            }

            // Build feature matrix (exclude target column)
            let feature_cols: Vec<usize> = (0..n_features).filter(|&c| c != target_col).collect();
            
            let mut x_train = Array2::zeros((observed_indices.len(), feature_cols.len()));
            let mut y_train = Array1::zeros(observed_indices.len());

            for (i, &row_idx) in observed_indices.iter().enumerate() {
                for (j, &col_idx) in feature_cols.iter().enumerate() {
                    x_train[[i, j]] = data[[row_idx, col_idx]];
                }
                y_train[i] = data[[row_idx, target_col]];
            }

            // Fit regression model
            let (coefficients, intercept) = Self::fit_linear_regression(&x_train, &y_train);

            // Predict for missing values
            let mut x_test = Array2::zeros((missing_indices.len(), feature_cols.len()));
            for (i, &row_idx) in missing_indices.iter().enumerate() {
                for (j, &col_idx) in feature_cols.iter().enumerate() {
                    x_test[[i, j]] = data[[row_idx, col_idx]];
                }
            }

            let predictions = Self::predict_linear(&x_test, &coefficients, intercept);

            // Add noise based on residual variance
            let y_pred_train = Self::predict_linear(&x_train, &coefficients, intercept);
            let residual_var: f64 = y_train.iter()
                .zip(y_pred_train.iter())
                .map(|(&y, &yp)| (y - yp).powi(2))
                .sum::<f64>() / observed_indices.len().max(1) as f64;
            let residual_std = residual_var.sqrt();

            // Impute with noise
            for (i, &row_idx) in missing_indices.iter().enumerate() {
                let old_value = data[[row_idx, target_col]];
                let noise: f64 = rng.gen::<f64>() * 2.0 - 1.0; // Uniform noise
                let new_value = predictions[i] + noise * residual_std * 0.1;
                data[[row_idx, target_col]] = new_value;

                if !old_value.is_nan() {
                    total_change += (new_value - old_value).abs();
                }
            }
        }

        total_change
    }
}

impl Default for MICEImputer {
    fn default() -> Self {
        Self::new()
    }
}

impl Imputer for MICEImputer {
    fn fit(&mut self, x: &Array2<f64>) -> Result<()> {
        let n_features = x.ncols();
        let mut stats = Vec::with_capacity(n_features);

        for col_idx in 0..n_features {
            let column: Vec<f64> = x.column(col_idx).iter().copied().collect();
            let observed: Vec<f64> = column.iter()
                .filter(|v| !is_missing(**v))
                .copied()
                .collect();

            let mean = if observed.is_empty() {
                0.0
            } else {
                observed.iter().sum::<f64>() / observed.len() as f64
            };

            let mut sorted = observed.clone();
            sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            let median = median_of_sorted(&sorted);

            let variance = if observed.is_empty() {
                0.0
            } else {
                observed.iter().map(|&v| (v - mean).powi(2)).sum::<f64>() / observed.len() as f64
            };

            let missing_mask: Vec<bool> = column.iter().map(|v| is_missing(*v)).collect();

            stats.push(FeatureStats {
                mean,
                median,
                std: variance.sqrt(),
                missing_mask_fit: missing_mask,
                coefficients: None,
                intercept: 0.0,
            });
        }

        // Determine imputation order (by number of missing values)
        let mut order: Vec<(usize, usize)> = stats.iter()
            .enumerate()
            .map(|(i, s)| (i, s.missing_mask_fit.iter().filter(|&&m| m).count()))
            .collect();
        order.sort_by_key(|&(_, count)| count);
        self.imputation_order = Some(order.iter().map(|&(i, _)| i).collect());

        self.feature_stats = Some(stats);
        Ok(())
    }

    fn transform(&self, x: &Array2<f64>) -> Result<Array2<f64>> {
        let _stats = self.feature_stats.as_ref().ok_or_else(|| {
            AutoMLError::ValidationError("Imputer not fitted".to_string())
        })?;

        let mut result = x.clone();
        let n_features = x.ncols();

        // Initial imputation
        for col_idx in 0..n_features {
            let column: Vec<f64> = result.column(col_idx).iter().copied().collect();
            let initial_value = self.initial_value(&column);

            for (row_idx, &val) in column.iter().enumerate() {
                if is_missing(val) {
                    result[[row_idx, col_idx]] = initial_value;
                }
            }
        }

        // MICE iterations
        let mut rng = match self.seed {
            Some(seed) => StdRng::seed_from_u64(seed),
            None => StdRng::from_entropy(),
        };

        for _ in 0..self.max_iter {
            let change = self.mice_iteration(&mut result, &mut rng);
            if change < self.tol {
                break;
            }
        }

        Ok(result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mice_imputer() {
        let data = Array2::from_shape_vec(
            (5, 3),
            vec![
                1.0, 2.0, 3.0,
                f64::NAN, 5.0, 6.0,
                7.0, f64::NAN, 9.0,
                10.0, 11.0, f64::NAN,
                13.0, 14.0, 15.0,
            ],
        ).unwrap();

        let mut imputer = MICEImputer::new()
            .with_max_iter(5)
            .with_seed(42);

        let result = imputer.fit_transform(&data).unwrap();

        // Check no NaN values remain
        assert!(!result.iter().any(|&v| v.is_nan()));
    }

    #[test]
    fn test_mice_with_initial_median() {
        let data = Array2::from_shape_vec(
            (4, 2),
            vec![
                1.0, 10.0,
                2.0, f64::NAN,
                f64::NAN, 30.0,
                4.0, 40.0,
            ],
        ).unwrap();

        let mut imputer = MICEImputer::new()
            .with_initial_strategy(InitialStrategy::Median)
            .with_max_iter(3);

        let result = imputer.fit_transform(&data).unwrap();
        assert!(!result.iter().any(|&v| v.is_nan()));
    }
}
