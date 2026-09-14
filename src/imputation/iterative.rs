//! Iterative imputer using various estimators

use crate::error::{AutoMLError, Result};
use crate::imputation::{Imputer, InitialStrategy, is_missing};
use ndarray::{Array1, Array2};
use rand::prelude::*;
use serde::{Deserialize, Serialize};

/// Estimator type for iterative imputation
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ImputerEstimator {
    /// Linear regression
    Linear,
    /// Ridge regression
    Ridge,
    /// Decision tree (simple)
    Tree,
    /// Mean predictor
    Mean,
}

/// Iterative imputer (generalization of MICE)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IterativeImputer {
    /// Maximum iterations
    max_iter: usize,
    /// Convergence tolerance
    tol: f64,
    /// Estimator type
    estimator: ImputerEstimator,
    /// Initial strategy
    initial_strategy: InitialStrategy,
    /// Random seed
    seed: Option<u64>,
    /// Ridge regularization parameter
    ridge_alpha: f64,
    /// Feature means
    feature_means: Option<Array1<f64>>,
    /// Feature medians
    feature_medians: Option<Array1<f64>>,
    /// Missing indicator per feature
    missing_features: Option<Vec<bool>>,
}

impl IterativeImputer {
    /// Create new iterative imputer
    pub fn new(estimator: ImputerEstimator) -> Self {
        Self {
            max_iter: 10,
            tol: 1e-3,
            estimator,
            initial_strategy: InitialStrategy::Mean,
            seed: None,
            ridge_alpha: 1.0,
            feature_means: None,
            feature_medians: None,
            missing_features: None,
        }
    }

    /// Set max iterations
    pub fn with_max_iter(mut self, n: usize) -> Self {
        self.max_iter = n.max(1);
        self
    }

    /// Set tolerance
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

    /// Set ridge alpha
    pub fn with_ridge_alpha(mut self, alpha: f64) -> Self {
        self.ridge_alpha = alpha.max(0.0);
        self
    }

    /// Initial value for a column
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
                sorted[sorted.len() / 2]
            }
            InitialStrategy::MostFrequent => {
                observed.iter().sum::<f64>() / observed.len() as f64
            }
            InitialStrategy::Constant => 0.0,
        }
    }

    /// Fit linear model and return coefficients
    fn fit_linear(&self, x: &Array2<f64>, y: &Array1<f64>) -> (Vec<f64>, f64) {
        let n = x.nrows() as f64;
        let p = x.ncols();

        if n < 2.0 || p == 0 {
            return (vec![0.0; p], y.mean().unwrap_or(0.0));
        }

        let y_mean = y.mean().unwrap_or(0.0);
        let x_means: Vec<f64> = (0..p)
            .map(|j| x.column(j).mean().unwrap_or(0.0))
            .collect();

        let y_centered: Vec<f64> = y.iter().map(|&yi| yi - y_mean).collect();

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
                .sum::<f64>() + match self.estimator {
                    ImputerEstimator::Ridge => self.ridge_alpha,
                    _ => 0.0,
                };

            coefficients[j] = if denominator > 1e-10 {
                numerator / denominator
            } else {
                0.0
            };
        }

        let intercept = y_mean - coefficients.iter()
            .zip(x_means.iter())
            .map(|(&c, &m)| c * m)
            .sum::<f64>();

        (coefficients, intercept)
    }

    /// Predict using coefficients
    fn predict(&self, x: &Array2<f64>, coefficients: &[f64], intercept: f64) -> Array1<f64> {
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

    /// Fit simple decision tree stump and predict
    fn fit_tree_predict(&self, x_train: &Array2<f64>, y_train: &Array1<f64>, x_test: &Array2<f64>) -> Array1<f64> {
        // Very simple tree: find best split on one feature
        let n_train = x_train.nrows();
        let p = x_train.ncols();

        if n_train == 0 || p == 0 {
            return Array1::from_elem(x_test.nrows(), y_train.mean().unwrap_or(0.0));
        }

        let mut best_feature = 0;
        let mut best_threshold = 0.0;
        let mut best_mse = f64::MAX;
        let mut best_left_mean = 0.0;
        let mut best_right_mean = 0.0;

        for feat in 0..p {
            // Get unique values for thresholds
            let values: Vec<f64> = x_train.column(feat).iter().copied().collect();
            let mut thresholds: Vec<f64> = values.clone();
            thresholds.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            thresholds.dedup();

            for &threshold in &thresholds {
                let (left_indices, right_indices): (Vec<usize>, Vec<usize>) = 
                    (0..n_train).partition(|&i| x_train[[i, feat]] <= threshold);

                if left_indices.is_empty() || right_indices.is_empty() {
                    continue;
                }

                let left_mean = left_indices.iter()
                    .map(|&i| y_train[i])
                    .sum::<f64>() / left_indices.len() as f64;

                let right_mean = right_indices.iter()
                    .map(|&i| y_train[i])
                    .sum::<f64>() / right_indices.len() as f64;

                let mse: f64 = left_indices.iter()
                    .map(|&i| (y_train[i] - left_mean).powi(2))
                    .sum::<f64>()
                    + right_indices.iter()
                        .map(|&i| (y_train[i] - right_mean).powi(2))
                        .sum::<f64>();

                if mse < best_mse {
                    best_mse = mse;
                    best_feature = feat;
                    best_threshold = threshold;
                    best_left_mean = left_mean;
                    best_right_mean = right_mean;
                }
            }
        }

        // Predict
        let n_test = x_test.nrows();
        let mut predictions = Array1::zeros(n_test);

        for i in 0..n_test {
            predictions[i] = if x_test[[i, best_feature]] <= best_threshold {
                best_left_mean
            } else {
                best_right_mean
            };
        }

        predictions
    }

    /// Run one iteration
    fn iteration(&self, data: &mut Array2<f64>, _rng: &mut StdRng) -> f64 {
        let n_features = data.ncols();
        let mut total_change = 0.0;

        for target_col in 0..n_features {
            let missing_mask: Vec<bool> = data.column(target_col)
                .iter()
                .map(|&v| is_missing(v))
                .collect();

            let n_missing = missing_mask.iter().filter(|&&m| m).count();
            if n_missing == 0 {
                continue;
            }

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

            let feature_cols: Vec<usize> = (0..n_features).filter(|&c| c != target_col).collect();

            // Build training data
            let mut x_train = Array2::zeros((observed_indices.len(), feature_cols.len()));
            let mut y_train = Array1::zeros(observed_indices.len());

            for (i, &row_idx) in observed_indices.iter().enumerate() {
                for (j, &col_idx) in feature_cols.iter().enumerate() {
                    x_train[[i, j]] = data[[row_idx, col_idx]];
                }
                y_train[i] = data[[row_idx, target_col]];
            }

            // Build test data
            let mut x_test = Array2::zeros((missing_indices.len(), feature_cols.len()));
            for (i, &row_idx) in missing_indices.iter().enumerate() {
                for (j, &col_idx) in feature_cols.iter().enumerate() {
                    x_test[[i, j]] = data[[row_idx, col_idx]];
                }
            }

            // Predict
            let predictions = match self.estimator {
                ImputerEstimator::Linear | ImputerEstimator::Ridge => {
                    let (coef, intercept) = self.fit_linear(&x_train, &y_train);
                    self.predict(&x_test, &coef, intercept)
                }
                ImputerEstimator::Tree => {
                    self.fit_tree_predict(&x_train, &y_train, &x_test)
                }
                ImputerEstimator::Mean => {
                    let mean = y_train.mean().unwrap_or(0.0);
                    Array1::from_elem(missing_indices.len(), mean)
                }
            };

            // Update values
            for (i, &row_idx) in missing_indices.iter().enumerate() {
                let old_value = data[[row_idx, target_col]];
                let new_value = predictions[i];
                data[[row_idx, target_col]] = new_value;

                if !old_value.is_nan() {
                    total_change += (new_value - old_value).abs();
                }
            }
        }

        total_change
    }
}

impl Default for IterativeImputer {
    fn default() -> Self {
        Self::new(ImputerEstimator::Linear)
    }
}

impl Imputer for IterativeImputer {
    fn fit(&mut self, x: &Array2<f64>) -> Result<()> {
        let n_features = x.ncols();
        
        // Compute means and medians
        let mut means = Vec::with_capacity(n_features);
        let mut medians = Vec::with_capacity(n_features);
        let mut has_missing = Vec::with_capacity(n_features);

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
            let median = if sorted.is_empty() {
                0.0
            } else {
                sorted[sorted.len() / 2]
            };

            let missing = column.iter().any(|v| is_missing(*v));

            means.push(mean);
            medians.push(median);
            has_missing.push(missing);
        }

        self.feature_means = Some(Array1::from_vec(means));
        self.feature_medians = Some(Array1::from_vec(medians));
        self.missing_features = Some(has_missing);

        Ok(())
    }

    fn transform(&self, x: &Array2<f64>) -> Result<Array2<f64>> {
        if self.feature_means.is_none() {
            return Err(AutoMLError::ValidationError(
                "Imputer not fitted".to_string()
            ));
        }

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

        // Iterative refinement
        let mut rng = match self.seed {
            Some(seed) => StdRng::seed_from_u64(seed),
            None => StdRng::from_entropy(),
        };

        for _ in 0..self.max_iter {
            let change = self.iteration(&mut result, &mut rng);
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
    fn test_iterative_linear() {
        let data = Array2::from_shape_vec(
            (5, 3),
            vec![
                1.0, 2.0, 3.0,
                f64::NAN, 5.0, 6.0,
                7.0, f64::NAN, 9.0,
                10.0, 11.0, 12.0,
                13.0, 14.0, 15.0,
            ],
        ).unwrap();

        let mut imputer = IterativeImputer::new(ImputerEstimator::Linear)
            .with_max_iter(5)
            .with_seed(42);

        let result = imputer.fit_transform(&data).unwrap();
        assert!(!result.iter().any(|&v| v.is_nan()));
    }

    #[test]
    fn test_iterative_ridge() {
        let data = Array2::from_shape_vec(
            (4, 2),
            vec![
                1.0, 10.0,
                2.0, f64::NAN,
                f64::NAN, 30.0,
                4.0, 40.0,
            ],
        ).unwrap();

        let mut imputer = IterativeImputer::new(ImputerEstimator::Ridge)
            .with_ridge_alpha(0.1)
            .with_max_iter(5);

        let result = imputer.fit_transform(&data).unwrap();
        assert!(!result.iter().any(|&v| v.is_nan()));
    }

    #[test]
    fn test_iterative_tree() {
        let data = Array2::from_shape_vec(
            (6, 2),
            vec![
                1.0, 10.0,
                2.0, 20.0,
                3.0, 30.0,
                4.0, f64::NAN,
                f64::NAN, 50.0,
                6.0, 60.0,
            ],
        ).unwrap();

        let mut imputer = IterativeImputer::new(ImputerEstimator::Tree)
            .with_max_iter(3);

        let result = imputer.fit_transform(&data).unwrap();
        assert!(!result.iter().any(|&v| v.is_nan()));
    }
}
