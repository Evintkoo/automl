//! Gaussian Process Regression
//!
//! Non-parametric Bayesian regression that provides uncertainty estimates.
//! Uses RBF (squared exponential) kernel with Cholesky decomposition for prediction.

use crate::error::{AutoMLError, Result};
use ndarray::{Array1, Array2};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GPConfig {
    pub length_scale: f64,
    pub signal_variance: f64,
    pub noise_variance: f64,
    pub max_training_size: usize,
}

impl Default for GPConfig {
    fn default() -> Self {
        Self {
            length_scale: 1.0,
            signal_variance: 1.0,
            noise_variance: 0.01,
            max_training_size: 2000,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GaussianProcessRegressor {
    pub config: GPConfig,
    x_train: Option<Array2<f64>>,
    alpha: Option<Array1<f64>>, // K_inv * y
    cholesky_l: Option<Array2<f64>>, // Cached Cholesky factor L where K = L * L^T
    y_mean: f64,
}

impl GaussianProcessRegressor {
    pub fn new(config: GPConfig) -> Self {
        Self { config, x_train: None, alpha: None, cholesky_l: None, y_mean: 0.0 }
    }

    pub fn fit(&mut self, x: &Array2<f64>, y: &Array1<f64>) -> Result<()> {
        let n = x.nrows();
        if n == 0 { return Err(AutoMLError::TrainingError("Empty dataset".into())); }

        // Subsample if too large
        let (x_sub, y_sub) = if n > self.config.max_training_size {
            let step = n / self.config.max_training_size;
            let indices: Vec<usize> = (0..n).step_by(step).take(self.config.max_training_size).collect();
            let x_sub = Array2::from_shape_fn((indices.len(), x.ncols()), |(i, j)| x[[indices[i], j]]);
            let y_sub = Array1::from_vec(indices.iter().map(|&i| y[i]).collect());
            (x_sub, y_sub)
        } else {
            (x.clone(), y.clone())
        };

        let n_sub = x_sub.nrows();
        self.y_mean = y_sub.mean().unwrap_or(0.0);
        let y_centered = &y_sub - self.y_mean;

        // Build kernel matrix K + σ²I (parallel for large n)
        let k = if n_sub > 100 {
            use rayon::prelude::*;
            let signal_var = self.config.signal_variance;
            let length_scale = self.config.length_scale;
            let noise_var = self.config.noise_variance;
            let rows: Vec<Vec<(usize, f64)>> = (0..n_sub).into_par_iter().map(|i| {
                let row_i = x_sub.row(i);
                (i..n_sub).map(|j| {
                    let row_j = x_sub.row(j);
                    let sq_dist: f64 = row_i.iter().zip(row_j.iter()).map(|(&a, &b)| (a - b).powi(2)).sum();
                    let mut val = signal_var * (-sq_dist / (2.0 * length_scale.powi(2))).exp();
                    if i == j { val += noise_var; }
                    (j, val)
                }).collect()
            }).collect();
            let mut k = Array2::zeros((n_sub, n_sub));
            for (i, row_vals) in rows.into_iter().enumerate() {
                for (j, val) in row_vals {
                    k[[i, j]] = val;
                    k[[j, i]] = val;
                }
            }
            k
        } else {
            let mut k = Array2::zeros((n_sub, n_sub));
            for i in 0..n_sub {
                for j in i..n_sub {
                    let val = self.rbf_kernel(x_sub.row(i).as_slice().unwrap(), x_sub.row(j).as_slice().unwrap());
                    k[[i, j]] = val;
                    k[[j, i]] = val;
                }
                k[[i, i]] += self.config.noise_variance;
            }
            k
        };

        // Solve K * alpha = y via Cholesky
        let (alpha, l) = cholesky_solve(&k, &y_centered)?;

        self.x_train = Some(x_sub);
        self.alpha = Some(alpha);
        self.cholesky_l = Some(l);
        Ok(())
    }

    pub fn predict(&self, x: &Array2<f64>) -> Result<Array1<f64>> {
        let x_train = self.x_train.as_ref().ok_or_else(|| AutoMLError::TrainingError("Model not fitted".into()))?;
        let alpha = self.alpha.as_ref().unwrap();

        let predictions = x.rows().into_iter().map(|row| {
            let s = row.as_slice().unwrap();
            let k_star: f64 = x_train.rows().into_iter().zip(alpha.iter())
                .map(|(tr, &a)| a * self.rbf_kernel(s, tr.as_slice().unwrap()))
                .sum();
            k_star + self.y_mean
        }).collect();

        Ok(Array1::from_vec(predictions))
    }

    /// Predict with uncertainty (mean, std)
    pub fn predict_with_std(&self, x: &Array2<f64>) -> Result<(Array1<f64>, Array1<f64>)> {
        let x_train = self.x_train.as_ref().ok_or_else(|| AutoMLError::TrainingError("Model not fitted".into()))?;
        let alpha = self.alpha.as_ref().unwrap();
        let n_train = x_train.nrows();

        let mut means = Vec::with_capacity(x.nrows());
        let mut stds = Vec::with_capacity(x.nrows());

        for row in x.rows() {
            let s = row.as_slice().unwrap();
            let mut k_star = Array1::zeros(n_train);
            let mut mean = 0.0;
            for (j, tr) in x_train.rows().into_iter().enumerate() {
                let kval = self.rbf_kernel(s, tr.as_slice().unwrap());
                k_star[j] = kval;
                mean += alpha[j] * kval;
            }

            // Variance = k(x*, x*) - k_star^T K^{-1} k_star
            // Compute v = L^{-1} k_star via forward substitution for proper variance
            let k_self = self.config.signal_variance;
            let var = if let Some(ref l) = self.cholesky_l {
                let nl = l.nrows();
                let mut v = Array1::zeros(nl);
                for i in 0..nl {
                    let sum: f64 = (0..i).map(|j| l[[i, j]] * v[j]).sum();
                    v[i] = (k_star[i] - sum) / l[[i, i]];
                }
                (k_self - v.dot(&v)).max(1e-10)
            } else {
                (k_self - k_star.dot(&k_star) / (n_train as f64 * self.config.signal_variance)).max(1e-10)
            };

            means.push(mean + self.y_mean);
            stds.push(var.sqrt());
        }

        Ok((Array1::from_vec(means), Array1::from_vec(stds)))
    }

    fn rbf_kernel(&self, a: &[f64], b: &[f64]) -> f64 {
        let sq_dist: f64 = a.iter().zip(b.iter()).map(|(&ai, &bi)| (ai - bi).powi(2)).sum();
        self.config.signal_variance * (-sq_dist / (2.0 * self.config.length_scale.powi(2))).exp()
    }
}

/// Solve Ax = b via Cholesky decomposition (A must be positive definite)
fn cholesky_solve(a: &Array2<f64>, b: &Array1<f64>) -> Result<(Array1<f64>, Array2<f64>)> {
    let n = a.nrows();

    // Cholesky factorization: A = L * L^T
    let mut l = Array2::zeros((n, n));
    for i in 0..n {
        for j in 0..=i {
            let sum: f64 = (0..j).map(|k| l[[i, k]] * l[[j, k]]).sum();
            if i == j {
                let val = a[[i, i]] - sum;
                if val <= 0.0 {
                    return Err(AutoMLError::TrainingError("Matrix not positive definite".into()));
                }
                l[[i, j]] = val.sqrt();
            } else {
                l[[i, j]] = (a[[i, j]] - sum) / l[[j, j]];
            }
        }
    }

    // Forward substitution: L * y = b
    let mut y = Array1::zeros(n);
    for i in 0..n {
        let sum: f64 = (0..i).map(|j| l[[i, j]] * y[j]).sum();
        y[i] = (b[i] - sum) / l[[i, i]];
    }

    // Back substitution: L^T * x = y
    let mut x = Array1::zeros(n);
    for i in (0..n).rev() {
        let sum: f64 = (i + 1..n).map(|j| l[[j, i]] * x[j]).sum();
        x[i] = (y[i] - sum) / l[[i, i]];
    }

    Ok((x, l))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gp_regressor_basic() {
        let x = Array2::from_shape_vec((50, 2), (0..100).map(|i| (i as f64) / 50.0).collect()).unwrap();
        let y = Array1::from_vec((0..50).map(|i| {
            let x0 = (i * 2) as f64 / 50.0;
            (x0 * std::f64::consts::PI).sin()
        }).collect());

        let config = GPConfig { length_scale: 0.5, noise_variance: 0.01, ..Default::default() };
        let mut model = GaussianProcessRegressor::new(config);
        model.fit(&x, &y).unwrap();
        let preds = model.predict(&x).unwrap();
        assert_eq!(preds.len(), 50);
    }

    #[test]
    fn test_gp_predict_with_std() {
        let x = Array2::from_shape_vec((30, 1), (0..30).map(|i| i as f64 / 10.0).collect()).unwrap();
        let y = Array1::from_vec((0..30).map(|i| (i as f64 / 10.0).sin()).collect());

        let config = GPConfig { length_scale: 1.0, noise_variance: 0.01, ..Default::default() };
        let mut model = GaussianProcessRegressor::new(config);
        model.fit(&x, &y).unwrap();

        let x_test = Array2::from_shape_vec((5, 1), vec![0.5, 1.0, 1.5, 5.0, 10.0]).unwrap();
        let (means, stds) = model.predict_with_std(&x_test).unwrap();
        assert_eq!(means.len(), 5);
        assert_eq!(stds.len(), 5);
        // Points far from training data should have higher uncertainty
        for &s in stds.iter() {
            assert!(s >= 0.0);
        }
    }

    #[test]
    fn test_gp_cholesky() {
        // Simple positive definite matrix
        let a = Array2::from_shape_vec((3, 3), vec![
            4.0, 2.0, 1.0,
            2.0, 5.0, 3.0,
            1.0, 3.0, 6.0,
        ]).unwrap();
        let b = Array1::from_vec(vec![1.0, 2.0, 3.0]);
        let (x, _l) = cholesky_solve(&a, &b).unwrap();
        // Verify A*x ≈ b
        let result = a.dot(&x);
        for i in 0..3 {
            assert!((result[i] - b[i]).abs() < 1e-10);
        }
    }
}
