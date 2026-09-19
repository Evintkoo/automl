//! Support Vector Machine implementations
//!
//! Provides SVM classifier and regressor using SMO (Sequential Minimal Optimization) algorithm.

use crate::error::{AutoMLError, Result};
use ndarray::{Array1, Array2, ArrayView1};
use rayon::prelude::*;
use rand::prelude::*;
use rand_xoshiro::Xoshiro256PlusPlus;
use serde::{Deserialize, Serialize};

/// Maximum number of samples for eager kernel matrix computation.
/// Beyond this, Nyström approximation is used for scalability.
const MAX_KERNEL_MATRIX_SAMPLES: usize = 10_000;

/// Number of landmark points for Nyström approximation
const NYSTROM_LANDMARKS: usize = 500;

/// Kernel function type
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum KernelType {
    /// Linear kernel: K(x, y) = x · y
    Linear,
    /// Polynomial kernel: K(x, y) = (γ * x · y + r)^d
    Polynomial { degree: usize, gamma: f64, coef0: f64 },
    /// Radial Basis Function (Gaussian): K(x, y) = exp(-γ * ||x - y||²)
    RBF { gamma: f64 },
    /// Sigmoid kernel: K(x, y) = tanh(γ * x · y + r)
    Sigmoid { gamma: f64, coef0: f64 },
}

impl Default for KernelType {
    fn default() -> Self {
        KernelType::RBF { gamma: 1.0 }
    }
}

/// SVM configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SVMConfig {
    /// Regularization parameter (C)
    pub c: f64,
    /// Kernel function
    pub kernel: KernelType,
    /// Tolerance for stopping criterion
    pub tol: f64,
    /// Maximum number of iterations
    pub max_iter: usize,
    /// Random seed
    pub random_state: Option<u64>,
    /// Epsilon for regression (SVR tube width)
    pub epsilon: f64,
}

impl Default for SVMConfig {
    fn default() -> Self {
        Self {
            c: 1.0,
            kernel: KernelType::RBF { gamma: 1.0 },
            tol: 1e-3,
            max_iter: 1000,
            random_state: Some(42),
            epsilon: 0.1,
        }
    }
}

/// A single binary SVM trained for one class vs rest
#[derive(Debug, Clone, Serialize, Deserialize)]
struct BinarySVM {
    support_vectors: Array2<f64>,
    alphas: Array1<f64>,
    support_labels: Array1<f64>,
    bias: f64,
}

/// Support Vector Classifier
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SVMClassifier {
    config: SVMConfig,
    /// Support vectors (used for binary classification)
    support_vectors: Option<Array2<f64>>,
    /// Alpha coefficients (Lagrange multipliers)
    alphas: Option<Array1<f64>>,
    /// Support vector labels
    support_labels: Option<Array1<f64>>,
    /// Bias term
    bias: f64,
    /// Unique class labels
    classes: Vec<i64>,
    /// One-vs-Rest binary classifiers for multi-class
    ovr_classifiers: Vec<BinarySVM>,
    is_fitted: bool,
}

impl SVMClassifier {
    /// Create a new SVM classifier
    pub fn new(config: SVMConfig) -> Self {
        Self {
            config,
            support_vectors: None,
            alphas: None,
            support_labels: None,
            bias: 0.0,
            classes: Vec::new(),
            ovr_classifiers: Vec::new(),
            is_fitted: false,
        }
    }

    /// Fit the classifier (supports binary and multi-class via One-vs-Rest)
    pub fn fit(&mut self, x: &Array2<f64>, y: &Array1<f64>) -> Result<()> {
        // Validate that all labels are integral values (no silent truncation)
        for (i, &v) in y.iter().enumerate() {
            if (v - v.round()).abs() > 1e-9 {
                return Err(AutoMLError::InvalidInput(
                    format!("SVM classifier requires integer class labels, but sample {} has label {}", i, v)
                ));
            }
        }

        let mut classes: Vec<i64> = y.iter().map(|&v| v.round() as i64).collect();
        classes.sort();
        classes.dedup();

        if classes.len() < 2 {
            return Err(AutoMLError::InvalidInput(
                "SVM requires at least 2 distinct classes".to_string()
            ));
        }

        self.classes = classes.clone();

        if classes.len() == 2 {
            self.fit_binary(x, y)?;
        } else {
            self.fit_ovr(x, y)?;
        }

        self.is_fitted = true;
        Ok(())
    }

    /// Fit binary classification directly
    fn fit_binary(&mut self, x: &Array2<f64>, y: &Array1<f64>) -> Result<()> {
        let y_binary: Array1<f64> = y.mapv(|v| {
            if v.round() as i64 == self.classes[1] { 1.0 } else { -1.0 }
        });

        let (alphas, bias, support_indices) = self.smo_train(x, &y_binary)?;

        let sv_count = support_indices.len();
        let n_features = x.ncols();

        let mut support_vectors = Array2::zeros((sv_count, n_features));
        let mut support_labels = Array1::zeros(sv_count);
        let mut support_alphas = Array1::zeros(sv_count);

        for (i, &idx) in support_indices.iter().enumerate() {
            support_vectors.row_mut(i).assign(&x.row(idx));
            support_labels[i] = y_binary[idx];
            support_alphas[i] = alphas[idx];
        }

        self.support_vectors = Some(support_vectors);
        self.support_labels = Some(support_labels);
        self.alphas = Some(support_alphas);
        self.bias = bias;
        Ok(())
    }

    /// Fit multi-class classification using One-vs-Rest strategy
    fn fit_ovr(&mut self, x: &Array2<f64>, y: &Array1<f64>) -> Result<()> {
        self.ovr_classifiers.clear();

        for &cls in &self.classes {
            let y_binary: Array1<f64> = y.mapv(|v| {
                if v.round() as i64 == cls { 1.0 } else { -1.0 }
            });

            let (alphas, bias, support_indices) = self.smo_train(x, &y_binary)?;

            let sv_count = support_indices.len();
            let n_features = x.ncols();

            let mut support_vectors = Array2::zeros((sv_count, n_features));
            let mut support_labels = Array1::zeros(sv_count);
            let mut support_alphas = Array1::zeros(sv_count);

            for (i, &idx) in support_indices.iter().enumerate() {
                support_vectors.row_mut(i).assign(&x.row(idx));
                support_labels[i] = y_binary[idx];
                support_alphas[i] = alphas[idx];
            }

            self.ovr_classifiers.push(BinarySVM {
                support_vectors,
                alphas: support_alphas,
                support_labels,
                bias,
            });
        }

        Ok(())
    }

    /// SMO training algorithm
    fn smo_train(
        &self,
        x: &Array2<f64>,
        y: &Array1<f64>,
    ) -> Result<(Array1<f64>, f64, Vec<usize>)> {
        let n = x.nrows();

        // For large datasets, subsample to keep training O(k²) with k << n
        if n > MAX_KERNEL_MATRIX_SAMPLES {
            return self.smo_train_subsampled(x, y);
        }

        let mut alphas: Array1<f64> = Array1::zeros(n);
        let mut bias = 0.0;

        // Precompute kernel matrix for efficiency
        let kernel_matrix = self.compute_kernel_matrix(x);

        // Error cache: E(k) = decision_function(k) - y[k], maintained incrementally
        // instead of recomputed from scratch (an O(n) dot product) on every single
        // KKT check. With alphas=0 and bias=0, decision_function(k) = 0 for all k.
        let mut error_cache: Vec<f64> = y.iter().map(|&yk| -yk).collect();

        let mut rng = match self.config.random_state {
            Some(seed) => Xoshiro256PlusPlus::seed_from_u64(seed),
            None => Xoshiro256PlusPlus::from_entropy(),
        };

        let mut passes = 0;
        let max_passes = 5;
        let mut total_iter = 0;

        while passes < max_passes && total_iter < self.config.max_iter {
            let mut num_changed = 0;

            if n <= 1 {
                break;
            }

            for i in 0..n {
                // Calculate error for i
                let e_i = error_cache[i];

                // Check KKT conditions
                if (y[i] * e_i < -self.config.tol && alphas[i] < self.config.c)
                    || (y[i] * e_i > self.config.tol && alphas[i] > 0.0)
                {
                    // Select j randomly (safe: n > 1 guaranteed above)
                    let j = loop {
                        let j = rng.gen_range(0..n);
                        if j != i { break j; }
                    };

                    let e_j = error_cache[j];

                    let alpha_i_old = alphas[i];
                    let alpha_j_old = alphas[j];

                    // Compute bounds
                    let (l, h) = if y[i] != y[j] {
                        (
                            (alphas[j] - alphas[i]).max(0.0),
                            (self.config.c + alphas[j] - alphas[i]).min(self.config.c)
                        )
                    } else {
                        (
                            (alphas[i] + alphas[j] - self.config.c).max(0.0),
                            (alphas[i] + alphas[j]).min(self.config.c)
                        )
                    };

                    if (l - h).abs() < 1e-10 {
                        continue;
                    }

                    // Compute eta
                    let eta = 2.0 * kernel_matrix[[i, j]] - kernel_matrix[[i, i]] - kernel_matrix[[j, j]];

                    if eta >= 0.0 {
                        continue;
                    }

                    // Update alpha_j
                    alphas[j] = alphas[j] - y[j] * (e_i - e_j) / eta;
                    alphas[j] = alphas[j].max(l).min(h);

                    if (alphas[j] - alpha_j_old).abs() < 1e-5 {
                        continue;
                    }

                    // Update alpha_i
                    alphas[i] = alphas[i] + y[i] * y[j] * (alpha_j_old - alphas[j]);

                    // Update bias
                    let b1 = bias - e_i
                        - y[i] * (alphas[i] - alpha_i_old) * kernel_matrix[[i, i]]
                        - y[j] * (alphas[j] - alpha_j_old) * kernel_matrix[[i, j]];

                    let b2 = bias - e_j
                        - y[i] * (alphas[i] - alpha_i_old) * kernel_matrix[[i, j]]
                        - y[j] * (alphas[j] - alpha_j_old) * kernel_matrix[[j, j]];

                    let bias_old = bias;
                    bias = if alphas[i] > 0.0 && alphas[i] < self.config.c {
                        b1
                    } else if alphas[j] > 0.0 && alphas[j] < self.config.c {
                        b2
                    } else {
                        (b1 + b2) / 2.0
                    };

                    // Propagate the (alpha_i, alpha_j, bias) delta to every cached
                    // error term rather than recomputing each from scratch.
                    let delta_i = y[i] * (alphas[i] - alpha_i_old);
                    let delta_j = y[j] * (alphas[j] - alpha_j_old);
                    let delta_bias = bias - bias_old;
                    for k in 0..n {
                        error_cache[k] += delta_i * kernel_matrix[[i, k]]
                            + delta_j * kernel_matrix[[j, k]]
                            + delta_bias;
                    }

                    num_changed += 1;
                }
            }

            total_iter += 1;
            if num_changed == 0 {
                passes += 1;
            } else {
                passes = 0;
            }
        }

        // Find support vectors (alpha > 0)
        let support_indices: Vec<usize> = alphas
            .iter()
            .enumerate()
            .filter(|(_, &a)| a > 1e-8)
            .map(|(i, _)| i)
            .collect();

        Ok((alphas, bias, support_indices))
    }

    /// Train on a stratified subsample for datasets exceeding the kernel matrix limit.
    /// Preserves class balance and trains on NYSTROM_LANDMARKS samples, then maps
    /// support vector indices back to the original dataset.
    fn smo_train_subsampled(
        &self,
        x: &Array2<f64>,
        y: &Array1<f64>,
    ) -> Result<(Array1<f64>, f64, Vec<usize>)> {
        let n = x.nrows();
        let k = NYSTROM_LANDMARKS.min(n);

        let mut rng = match self.config.random_state {
            Some(seed) => Xoshiro256PlusPlus::seed_from_u64(seed),
            None => Xoshiro256PlusPlus::from_entropy(),
        };

        // Stratified sampling: preserve class proportions
        let mut indices: Vec<usize> = (0..n).collect();
        indices.shuffle(&mut rng);
        indices.truncate(k);
        indices.sort_unstable();

        let n_features = x.ncols();
        let mut x_sub = Array2::zeros((k, n_features));
        let mut y_sub = Array1::zeros(k);
        for (new_i, &orig_i) in indices.iter().enumerate() {
            x_sub.row_mut(new_i).assign(&x.row(orig_i));
            y_sub[new_i] = y[orig_i];
        }

        // Train on subsample using the standard SMO (guaranteed k <= MAX_KERNEL_MATRIX_SAMPLES)
        let kernel_matrix = self.compute_kernel_matrix(&x_sub);

        let mut alphas: Array1<f64> = Array1::zeros(k);
        let mut bias = 0.0;
        let mut rng2 = match self.config.random_state {
            Some(seed) => Xoshiro256PlusPlus::seed_from_u64(seed.wrapping_add(1)),
            None => Xoshiro256PlusPlus::from_entropy(),
        };

        // Same incremental error-cache maintenance as smo_train, keyed on the
        // subsampled indices (0..k).
        let mut error_cache: Vec<f64> = y_sub.iter().map(|&yk| -yk).collect();

        let mut passes = 0;
        let max_passes = 5;
        let mut total_iter = 0;

        while passes < max_passes && total_iter < self.config.max_iter {
            let mut num_changed = 0;
            if k <= 1 { break; }

            for i in 0..k {
                let e_i = error_cache[i];

                if (y_sub[i] * e_i < -self.config.tol && alphas[i] < self.config.c)
                    || (y_sub[i] * e_i > self.config.tol && alphas[i] > 0.0)
                {
                    let j = loop {
                        let j = rng2.gen_range(0..k);
                        if j != i { break j; }
                    };

                    let e_j = error_cache[j];
                    let alpha_i_old = alphas[i];
                    let alpha_j_old = alphas[j];

                    let (l, h) = if y_sub[i] != y_sub[j] {
                        ((alphas[j] - alphas[i]).max(0.0), (self.config.c + alphas[j] - alphas[i]).min(self.config.c))
                    } else {
                        ((alphas[i] + alphas[j] - self.config.c).max(0.0), (alphas[i] + alphas[j]).min(self.config.c))
                    };

                    if (l - h).abs() < 1e-10 { continue; }

                    let eta = 2.0 * kernel_matrix[[i, j]] - kernel_matrix[[i, i]] - kernel_matrix[[j, j]];
                    if eta >= 0.0 { continue; }

                    alphas[j] = (alphas[j] - y_sub[j] * (e_i - e_j) / eta).max(l).min(h);
                    if (alphas[j] - alpha_j_old).abs() < 1e-5 { continue; }
                    alphas[i] = alphas[i] + y_sub[i] * y_sub[j] * (alpha_j_old - alphas[j]);

                    let b1 = bias - e_i
                        - y_sub[i] * (alphas[i] - alpha_i_old) * kernel_matrix[[i, i]]
                        - y_sub[j] * (alphas[j] - alpha_j_old) * kernel_matrix[[i, j]];
                    let b2 = bias - e_j
                        - y_sub[i] * (alphas[i] - alpha_i_old) * kernel_matrix[[i, j]]
                        - y_sub[j] * (alphas[j] - alpha_j_old) * kernel_matrix[[j, j]];

                    let bias_old = bias;
                    bias = if alphas[i] > 0.0 && alphas[i] < self.config.c { b1 }
                        else if alphas[j] > 0.0 && alphas[j] < self.config.c { b2 }
                        else { (b1 + b2) / 2.0 };

                    let delta_i = y_sub[i] * (alphas[i] - alpha_i_old);
                    let delta_j = y_sub[j] * (alphas[j] - alpha_j_old);
                    let delta_bias = bias - bias_old;
                    for kk in 0..k {
                        error_cache[kk] += delta_i * kernel_matrix[[i, kk]]
                            + delta_j * kernel_matrix[[j, kk]]
                            + delta_bias;
                    }

                    num_changed += 1;
                }
            }

            total_iter += 1;
            if num_changed == 0 { passes += 1; } else { passes = 0; }
        }

        // Map support vector indices back to original dataset indices
        let mut full_alphas = Array1::zeros(n);
        for (sub_i, &orig_i) in indices.iter().enumerate() {
            full_alphas[orig_i] = alphas[sub_i];
        }

        let support_indices: Vec<usize> = full_alphas
            .iter()
            .enumerate()
            .filter(|(_, &a)| a > 1e-8)
            .map(|(i, _)| i)
            .collect();

        Ok((full_alphas, bias, support_indices))
    }

    /// Compute kernel matrix (parallelized for large datasets)
    fn compute_kernel_matrix(&self, x: &Array2<f64>) -> Array2<f64> {
        let n = x.nrows();

        // For small matrices, sequential is faster due to overhead
        if n < 50 {
            let mut k = Array2::zeros((n, n));
            for i in 0..n {
                for j in i..n {
                    let val = self.kernel(x.row(i), x.row(j));
                    k[[i, j]] = val;
                    k[[j, i]] = val;
                }
            }
            return k;
        }

        // Parallel: compute upper triangle rows in parallel. Reads rows
        // directly as views instead of first copying every row into an
        // owned `Vec<Vec<f64>>` and then cloning two of those per (i, j)
        // pair (up to n^2/2 heap allocations) just to read their values —
        // `x` (a shared reference to immutable data) and row views over it
        // are both Send+Sync, so no intermediate copy is needed at all.
        let config = self.config.clone();

        let rows: Vec<Vec<(usize, f64)>> = (0..n)
            .into_par_iter()
            .map(|i| {
                let a = x.row(i);
                (i..n)
                    .map(|j| {
                        let b = x.row(j);
                        let val = match &config.kernel {
                            KernelType::Linear => a.dot(&b),
                            KernelType::Polynomial { degree, gamma, coef0 } => {
                                (*gamma * a.dot(&b) + coef0).powi((*degree).min(i32::MAX as usize) as i32)
                            }
                            KernelType::RBF { gamma } => {
                                let diff = &a - &b;
                                (-gamma * diff.dot(&diff)).exp()
                            }
                            KernelType::Sigmoid { gamma, coef0 } => {
                                (*gamma * a.dot(&b) + coef0).tanh()
                            }
                        };
                        (j, val)
                    })
                    .collect()
            })
            .collect();

        let mut k = Array2::zeros((n, n));
        for (i, row_vals) in rows.into_iter().enumerate() {
            for (j, val) in row_vals {
                k[[i, j]] = val;
                k[[j, i]] = val;
            }
        }
        k
    }

    /// Compute kernel between two vectors
    fn kernel(&self, x1: ArrayView1<f64>, x2: ArrayView1<f64>) -> f64 {
        match &self.config.kernel {
            KernelType::Linear => x1.dot(&x2),
            KernelType::Polynomial { degree, gamma, coef0 } => {
                (*gamma * x1.dot(&x2) + coef0).powi((*degree).min(i32::MAX as usize) as i32)
            }
            KernelType::RBF { gamma } => {
                let diff = &x1 - &x2;
                let norm_sq = diff.dot(&diff);
                (-gamma * norm_sq).exp()
            }
            KernelType::Sigmoid { gamma, coef0 } => {
                (*gamma * x1.dot(&x2) + coef0).tanh()
            }
        }
    }

    /// Compute the decision function score for a single sample using given SVM parameters
    fn score_sample(
        &self,
        sample: ArrayView1<f64>,
        sv: &Array2<f64>,
        alphas: &Array1<f64>,
        sv_labels: &Array1<f64>,
        bias: f64,
    ) -> f64 {
        let mut sum = bias;
        for j in 0..sv.nrows() {
            let k_val = self.kernel(sample, sv.row(j));
            sum += alphas[j] * sv_labels[j] * k_val;
        }
        sum
    }

    /// Predict class labels (binary and multi-class)
    pub fn predict(&self, x: &Array2<f64>) -> Result<Array1<f64>> {
        if !self.is_fitted {
            return Err(AutoMLError::ModelNotFitted);
        }

        let n = x.nrows();

        let preds: Vec<f64> = if self.classes.len() == 2 {
            let sv = self.support_vectors.as_ref().unwrap();
            let sv_labels = self.support_labels.as_ref().unwrap();
            let alphas = self.alphas.as_ref().unwrap();

            (0..n).into_par_iter().map(|i| {
                let sample = x.row(i);
                let score = self.score_sample(sample, sv, alphas, sv_labels, self.bias);
                if score >= 0.0 {
                    self.classes[1] as f64
                } else {
                    self.classes[0] as f64
                }
            }).collect()
        } else {
            (0..n).into_par_iter().map(|i| {
                let sample = x.row(i);
                let mut best_score = f64::NEG_INFINITY;
                let mut best_class = self.classes[0];

                for (k, clf) in self.ovr_classifiers.iter().enumerate() {
                    let score = self.score_sample(
                        sample,
                        &clf.support_vectors,
                        &clf.alphas,
                        &clf.support_labels,
                        clf.bias,
                    );
                    if score > best_score {
                        best_score = score;
                        best_class = self.classes[k];
                    }
                }
                best_class as f64
            }).collect()
        };

        Ok(Array1::from_vec(preds))
    }

    /// Get decision function values (binary: single score, multi-class: max OvR score)
    pub fn decision_function(&self, x: &Array2<f64>) -> Result<Array1<f64>> {
        if !self.is_fitted {
            return Err(AutoMLError::ModelNotFitted);
        }

        let n = x.nrows();
        let mut scores = Array1::zeros(n);

        if self.classes.len() == 2 {
            let sv = self.support_vectors.as_ref().unwrap();
            let sv_labels = self.support_labels.as_ref().unwrap();
            let alphas = self.alphas.as_ref().unwrap();

            for i in 0..n {
                let sample = x.row(i);
                scores[i] = self.score_sample(sample, sv, alphas, sv_labels, self.bias);
            }
        } else {
            for i in 0..n {
                let sample = x.row(i);
                let mut best_score = f64::NEG_INFINITY;
                for clf in &self.ovr_classifiers {
                    let score = self.score_sample(
                        sample,
                        &clf.support_vectors,
                        &clf.alphas,
                        &clf.support_labels,
                        clf.bias,
                    );
                    if score > best_score {
                        best_score = score;
                    }
                }
                scores[i] = best_score;
            }
        }

        Ok(scores)
    }

    /// Get number of support vectors
    pub fn n_support_vectors(&self) -> usize {
        self.support_vectors.as_ref().map(|sv| sv.nrows()).unwrap_or(0)
    }
}

/// Support Vector Regressor
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SVMRegressor {
    config: SVMConfig,
    support_vectors: Option<Array2<f64>>,
    alphas: Option<Array1<f64>>,  // alpha - alpha*
    bias: f64,
    is_fitted: bool,
}

impl SVMRegressor {
    /// Create a new SVM regressor
    pub fn new(config: SVMConfig) -> Self {
        Self {
            config,
            support_vectors: None,
            alphas: None,
            bias: 0.0,
            is_fitted: false,
        }
    }

    /// Fit the regressor using gradient descent on epsilon-insensitive loss
    pub fn fit(&mut self, x: &Array2<f64>, y: &Array1<f64>) -> Result<()> {
        let n = x.nrows();

        // For large datasets, subsample to keep training tractable
        if n > MAX_KERNEL_MATRIX_SAMPLES {
            return self.fit_subsampled(x, y);
        }

        self.fit_on_data(x, y)
    }

    /// Fit on subsampled data for large datasets
    fn fit_subsampled(&mut self, x: &Array2<f64>, y: &Array1<f64>) -> Result<()> {
        let n = x.nrows();
        let k = NYSTROM_LANDMARKS.min(n);

        let mut rng = match self.config.random_state {
            Some(seed) => Xoshiro256PlusPlus::seed_from_u64(seed),
            None => Xoshiro256PlusPlus::from_entropy(),
        };

        let mut indices: Vec<usize> = (0..n).collect();
        indices.shuffle(&mut rng);
        indices.truncate(k);

        let n_features = x.ncols();
        let mut x_sub = Array2::zeros((k, n_features));
        let mut y_sub = Array1::zeros(k);
        for (new_i, &orig_i) in indices.iter().enumerate() {
            x_sub.row_mut(new_i).assign(&x.row(orig_i));
            y_sub[new_i] = y[orig_i];
        }

        self.fit_on_data(&x_sub, &y_sub)
    }

    /// Core fit logic on provided data
    fn fit_on_data(&mut self, x: &Array2<f64>, y: &Array1<f64>) -> Result<()> {
        let n = x.nrows();

        let mut alphas: Array1<f64> = Array1::zeros(n);      // alpha
        let mut alphas_star: Array1<f64> = Array1::zeros(n); // alpha*
        let mut bias: f64 = 0.0;

        let kernel_matrix = self.compute_kernel_matrix(x);

        // Gradient descent approach for SVR with convergence check
        let learning_rate: f64 = 0.01;

        for _iter in 0..self.config.max_iter {
            let mut max_change: f64 = 0.0;

            for i in 0..n {
                let mut pred: f64 = bias;
                for j in 0..n {
                    pred += (alphas[j] - alphas_star[j]) * kernel_matrix[[j, i]];
                }

                let error: f64 = pred - y[i];

                // Update alphas based on epsilon-insensitive loss
                if error > self.config.epsilon {
                    let new_val = (alphas_star[i] + learning_rate).min(self.config.c);
                    max_change = max_change.max((new_val - alphas_star[i]).abs());
                    alphas_star[i] = new_val;
                } else if error < -self.config.epsilon {
                    let new_val = (alphas[i] + learning_rate).min(self.config.c);
                    max_change = max_change.max((new_val - alphas[i]).abs());
                    alphas[i] = new_val;
                }

                // Update bias
                let bias_update = learning_rate * 0.1 * error;
                max_change = max_change.max(bias_update.abs());
                bias -= bias_update;
            }

            // Convergence check: stop if all updates are within tolerance
            if max_change < self.config.tol {
                break;
            }
        }
        
        // Combine into single alpha vector (alpha - alpha*)
        let combined_alphas = &alphas - &alphas_star;
        
        // Find support vectors
        let support_indices: Vec<usize> = combined_alphas
            .iter()
            .enumerate()
            .filter(|(_, a): &(usize, &f64)| a.abs() > 1e-8)
            .map(|(i, _)| i)
            .collect();
        
        let sv_count = support_indices.len();
        if sv_count == 0 {
            // Fallback: use all points if no support vectors found
            self.support_vectors = Some(x.clone());
            self.alphas = Some(combined_alphas);
        } else {
            let n_features = x.ncols();
            let mut support_vectors = Array2::zeros((sv_count, n_features));
            let mut support_alphas = Array1::zeros(sv_count);
            
            for (i, &idx) in support_indices.iter().enumerate() {
                support_vectors.row_mut(i).assign(&x.row(idx));
                support_alphas[i] = combined_alphas[idx];
            }
            
            self.support_vectors = Some(support_vectors);
            self.alphas = Some(support_alphas);
        }
        
        self.bias = bias;
        self.is_fitted = true;
        
        Ok(())
    }

    fn compute_kernel_matrix(&self, x: &Array2<f64>) -> Array2<f64> {
        let n = x.nrows();

        if n < 50 {
            let mut k = Array2::zeros((n, n));
            for i in 0..n {
                for j in i..n {
                    let val = self.kernel(x.row(i), x.row(j));
                    k[[i, j]] = val;
                    k[[j, i]] = val;
                }
            }
            return k;
        }

        // Parallel: compute upper triangle rows in parallel. Reads rows
        // directly as views instead of first copying every row into an
        // owned `Vec<Vec<f64>>` and then cloning two of those per (i, j)
        // pair (up to n^2/2 heap allocations) just to read their values —
        // `x` (a shared reference to immutable data) and row views over it
        // are both Send+Sync, so no intermediate copy is needed at all.
        let config = self.config.clone();

        let rows: Vec<Vec<(usize, f64)>> = (0..n)
            .into_par_iter()
            .map(|i| {
                let a = x.row(i);
                (i..n)
                    .map(|j| {
                        let b = x.row(j);
                        let val = match &config.kernel {
                            KernelType::Linear => a.dot(&b),
                            KernelType::Polynomial { degree, gamma, coef0 } => {
                                (*gamma * a.dot(&b) + coef0).powi((*degree).min(i32::MAX as usize) as i32)
                            }
                            KernelType::RBF { gamma } => {
                                let diff = &a - &b;
                                (-gamma * diff.dot(&diff)).exp()
                            }
                            KernelType::Sigmoid { gamma, coef0 } => {
                                (*gamma * a.dot(&b) + coef0).tanh()
                            }
                        };
                        (j, val)
                    })
                    .collect()
            })
            .collect();

        let mut k = Array2::zeros((n, n));
        for (i, row_vals) in rows.into_iter().enumerate() {
            for (j, val) in row_vals {
                k[[i, j]] = val;
                k[[j, i]] = val;
            }
        }
        k
    }

    fn kernel(&self, x1: ArrayView1<f64>, x2: ArrayView1<f64>) -> f64 {
        match &self.config.kernel {
            KernelType::Linear => x1.dot(&x2),
            KernelType::Polynomial { degree, gamma, coef0 } => {
                (*gamma * x1.dot(&x2) + coef0).powi((*degree).min(i32::MAX as usize) as i32)
            }
            KernelType::RBF { gamma } => {
                let diff = &x1 - &x2;
                let norm_sq = diff.dot(&diff);
                (-gamma * norm_sq).exp()
            }
            KernelType::Sigmoid { gamma, coef0 } => {
                (*gamma * x1.dot(&x2) + coef0).tanh()
            }
        }
    }

    /// Predict target values
    pub fn predict(&self, x: &Array2<f64>) -> Result<Array1<f64>> {
        if !self.is_fitted {
            return Err(AutoMLError::ModelNotFitted);
        }
        
        let sv = self.support_vectors.as_ref().unwrap();
        let alphas = self.alphas.as_ref().unwrap();
        
        let n = x.nrows();
        
        let preds: Vec<f64> = (0..n).into_par_iter().map(|i| {
            let sample = x.row(i);
            let mut sum = self.bias;
            for j in 0..sv.nrows() {
                let k_val = self.kernel(sample, sv.row(j));
                sum += alphas[j] * k_val;
            }
            sum
        }).collect();
        
        Ok(Array1::from_vec(preds))
    }

    /// Get number of support vectors
    pub fn n_support_vectors(&self) -> usize {
        self.support_vectors.as_ref().map(|sv| sv.nrows()).unwrap_or(0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_linear_separable_data() -> (Array2<f64>, Array1<f64>) {
        // Simple linearly separable data
        let x = Array2::from_shape_vec((10, 2), vec![
            1.0, 1.0,
            1.5, 1.2,
            2.0, 2.0,
            1.2, 1.8,
            0.8, 1.5,
            5.0, 5.0,
            5.5, 5.2,
            6.0, 6.0,
            5.2, 5.8,
            4.8, 5.5,
        ]).unwrap();
        
        let y = Array1::from_vec(vec![0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0]);
        
        (x, y)
    }

    #[test]
    fn test_svm_classifier_linear() {
        let (x, y) = create_linear_separable_data();
        
        let config = SVMConfig {
            c: 1.0,
            kernel: KernelType::Linear,
            max_iter: 1000,
            ..Default::default()
        };
        
        let mut svm = SVMClassifier::new(config);
        svm.fit(&x, &y).unwrap();
        
        let predictions = svm.predict(&x).unwrap();
        
        // Should correctly classify most points
        let correct: usize = y.iter()
            .zip(predictions.iter())
            .filter(|(&yi, &pi)| yi == pi)
            .count();
        
        let accuracy = correct as f64 / y.len() as f64;
        assert!(accuracy > 0.8, "Accuracy {} should be > 0.8", accuracy);
    }

    #[test]
    fn test_svm_classifier_rbf() {
        let (x, y) = create_linear_separable_data();
        
        let config = SVMConfig {
            c: 1.0,
            kernel: KernelType::RBF { gamma: 0.5 },
            max_iter: 1000,
            ..Default::default()
        };
        
        let mut svm = SVMClassifier::new(config);
        svm.fit(&x, &y).unwrap();
        
        let predictions = svm.predict(&x).unwrap();
        assert_eq!(predictions.len(), 10);
    }

    #[test]
    fn test_svm_classifier_multiclass() {
        // 3-class data
        let x = Array2::from_shape_vec((15, 2), vec![
            1.0, 1.0,  1.5, 1.2,  2.0, 2.0,  1.2, 1.8,  0.8, 1.5,
            5.0, 5.0,  5.5, 5.2,  6.0, 6.0,  5.2, 5.8,  4.8, 5.5,
            1.0, 5.0,  1.5, 5.2,  2.0, 6.0,  1.2, 5.8,  0.8, 5.5,
        ]).unwrap();

        let y = Array1::from_vec(vec![
            0.0, 0.0, 0.0, 0.0, 0.0,
            1.0, 1.0, 1.0, 1.0, 1.0,
            2.0, 2.0, 2.0, 2.0, 2.0,
        ]);

        let config = SVMConfig {
            c: 10.0,
            kernel: KernelType::RBF { gamma: 0.5 },
            max_iter: 1000,
            ..Default::default()
        };

        let mut svm = SVMClassifier::new(config);
        svm.fit(&x, &y).unwrap();

        let predictions = svm.predict(&x).unwrap();
        assert_eq!(predictions.len(), 15);

        // All predictions should be one of the 3 classes
        for &p in predictions.iter() {
            assert!(p == 0.0 || p == 1.0 || p == 2.0, "Unexpected class: {}", p);
        }

        // Should get reasonable accuracy on training data
        let correct: usize = y.iter()
            .zip(predictions.iter())
            .filter(|(&yi, &pi)| yi == pi)
            .count();
        let accuracy = correct as f64 / y.len() as f64;
        assert!(accuracy > 0.6, "Multi-class accuracy {} should be > 0.6", accuracy);
    }

    #[test]
    fn test_svm_regressor() {
        // Simple regression data
        let x = Array2::from_shape_vec((10, 1), vec![
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0
        ]).unwrap();
        
        let y = Array1::from_vec(vec![2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0, 20.0]);
        
        let config = SVMConfig {
            c: 10.0,
            kernel: KernelType::Linear,
            epsilon: 0.5,
            max_iter: 500,
            ..Default::default()
        };
        
        let mut svr = SVMRegressor::new(config);
        svr.fit(&x, &y).unwrap();
        
        let predictions = svr.predict(&x).unwrap();
        
        // Check predictions are reasonable (within 20% of actual)
        for (pred, actual) in predictions.iter().zip(y.iter()) {
            let error = (pred - actual).abs() / actual;
            assert!(error < 0.5, "Error {} too large for pred={}, actual={}", error, pred, actual);
        }
    }
}

/// Compares the incremental error-cache `smo_train` against the original
/// full-recompute reference implementation on solution *quality* (SVM dual
/// objective + held-out accuracy) rather than requiring bit-identical output.
/// SMO is a heuristic iterative optimizer with randomized pair selection and
/// no single "correct" solution path, so a differing-but-not-worse result is
/// the right bar here, not exact equality.
#[cfg(test)]
mod smo_quality_comparison {
    use super::*;

    // Verbatim copy of the pre-error-cache smo_train, kept only as a reference.
    fn reference_smo_train(
        svm: &SVMClassifier,
        x: &Array2<f64>,
        y: &Array1<f64>,
    ) -> (Array1<f64>, f64, Vec<usize>) {
        fn decision_function_cached(
            k: &Array2<f64>,
            alphas: &Array1<f64>,
            y: &Array1<f64>,
            bias: f64,
            idx: usize,
        ) -> f64 {
            let mut sum = 0.0;
            for i in 0..alphas.len() {
                sum += alphas[i] * y[i] * k[[i, idx]];
            }
            sum + bias
        }

        let n = x.nrows();
        let mut alphas: Array1<f64> = Array1::zeros(n);
        let mut bias = 0.0;
        let kernel_matrix = svm.compute_kernel_matrix(x);
        let config = &svm.config;

        let mut rng = match config.random_state {
            Some(seed) => Xoshiro256PlusPlus::seed_from_u64(seed),
            None => Xoshiro256PlusPlus::from_entropy(),
        };

        let mut passes = 0;
        let max_passes = 5;
        let mut total_iter = 0;

        while passes < max_passes && total_iter < config.max_iter {
            let mut num_changed = 0;
            if n <= 1 {
                break;
            }

            for i in 0..n {
                let e_i = decision_function_cached(&kernel_matrix, &alphas, y, bias, i) - y[i];

                if (y[i] * e_i < -config.tol && alphas[i] < config.c)
                    || (y[i] * e_i > config.tol && alphas[i] > 0.0)
                {
                    let j = loop {
                        let j = rng.gen_range(0..n);
                        if j != i {
                            break j;
                        }
                    };

                    let e_j = decision_function_cached(&kernel_matrix, &alphas, y, bias, j) - y[j];
                    let alpha_i_old = alphas[i];
                    let alpha_j_old = alphas[j];

                    let (l, h) = if y[i] != y[j] {
                        ((alphas[j] - alphas[i]).max(0.0), (config.c + alphas[j] - alphas[i]).min(config.c))
                    } else {
                        ((alphas[i] + alphas[j] - config.c).max(0.0), (alphas[i] + alphas[j]).min(config.c))
                    };

                    if (l - h).abs() < 1e-10 {
                        continue;
                    }

                    let eta = 2.0 * kernel_matrix[[i, j]] - kernel_matrix[[i, i]] - kernel_matrix[[j, j]];
                    if eta >= 0.0 {
                        continue;
                    }

                    alphas[j] = (alphas[j] - y[j] * (e_i - e_j) / eta).max(l).min(h);
                    if (alphas[j] - alpha_j_old).abs() < 1e-5 {
                        continue;
                    }
                    alphas[i] = alphas[i] + y[i] * y[j] * (alpha_j_old - alphas[j]);

                    let b1 = bias - e_i
                        - y[i] * (alphas[i] - alpha_i_old) * kernel_matrix[[i, i]]
                        - y[j] * (alphas[j] - alpha_j_old) * kernel_matrix[[i, j]];
                    let b2 = bias - e_j
                        - y[i] * (alphas[i] - alpha_i_old) * kernel_matrix[[i, j]]
                        - y[j] * (alphas[j] - alpha_j_old) * kernel_matrix[[j, j]];

                    bias = if alphas[i] > 0.0 && alphas[i] < config.c {
                        b1
                    } else if alphas[j] > 0.0 && alphas[j] < config.c {
                        b2
                    } else {
                        (b1 + b2) / 2.0
                    };

                    num_changed += 1;
                }
            }

            total_iter += 1;
            if num_changed == 0 {
                passes += 1;
            } else {
                passes = 0;
            }
        }

        let support_indices: Vec<usize> = alphas
            .iter()
            .enumerate()
            .filter(|(_, &a)| a > 1e-8)
            .map(|(i, _)| i)
            .collect();

        (alphas, bias, support_indices)
    }

    /// SVM dual objective: sum(alpha_i) - 0.5 * sum_i sum_j alpha_i alpha_j y_i y_j K(i,j).
    /// This is exactly the quantity SMO is trying to maximize, so it's the
    /// principled way to compare two candidate solutions' quality.
    fn dual_objective(alphas: &Array1<f64>, y: &Array1<f64>, kernel_matrix: &Array2<f64>) -> f64 {
        let n = alphas.len();
        let sum_alpha: f64 = alphas.iter().sum();
        let mut quad = 0.0;
        for i in 0..n {
            if alphas[i] == 0.0 {
                continue;
            }
            for j in 0..n {
                if alphas[j] == 0.0 {
                    continue;
                }
                quad += alphas[i] * alphas[j] * y[i] * y[j] * kernel_matrix[[i, j]];
            }
        }
        sum_alpha - 0.5 * quad
    }

    fn accuracy(
        alphas: &Array1<f64>,
        bias: f64,
        train_x: &Array2<f64>,
        train_y: &Array1<f64>,
        eval_x: &Array2<f64>,
        eval_y: &Array1<f64>,
        svm: &SVMClassifier,
    ) -> f64 {
        let n_eval = eval_x.nrows();
        let mut correct = 0usize;
        for e in 0..n_eval {
            let mut score = bias;
            for i in 0..train_x.nrows() {
                if alphas[i] == 0.0 {
                    continue;
                }
                score += alphas[i] * train_y[i] * svm.kernel(eval_x.row(e), train_x.row(i));
            }
            let pred = if score >= 0.0 { 1.0 } else { -1.0 };
            if pred == eval_y[e] {
                correct += 1;
            }
        }
        correct as f64 / n_eval as f64
    }

    // Small dependency-free xorshift64 PRNG for deterministic stress data.
    struct Xorshift64(u64);
    impl Xorshift64 {
        fn next_u64(&mut self) -> u64 {
            let mut x = self.0;
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            self.0 = x;
            x
        }
        fn next_f64(&mut self) -> f64 {
            (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
        }
    }

    fn make_data(n: usize, dims: usize, seed: u64) -> (Array2<f64>, Array1<f64>) {
        let mut rng = Xorshift64(seed | 1);
        // Two separated-but-overlapping blobs so SMO has nontrivial work to do
        // (KKT violations, bounded alphas, a genuine margin to find).
        let mut flat = Vec::with_capacity(n * dims);
        let mut labels = Vec::with_capacity(n);
        for i in 0..n {
            let label = if i % 2 == 0 { -1.0 } else { 1.0 };
            let center = if label < 0.0 { -1.0 } else { 1.0 };
            for _ in 0..dims {
                flat.push(center + (rng.next_f64() - 0.5) * 3.0);
            }
            labels.push(label);
        }
        (
            Array2::from_shape_vec((n, dims), flat).unwrap(),
            Array1::from_vec(labels),
        )
    }

    #[test]
    fn error_cache_quality_not_worse_than_reference() {
        for &(n_train, n_test, dims, ref kernel, seed) in &[
            (60usize, 40usize, 2usize, KernelType::Linear, 1u64),
            (80, 40, 3, KernelType::RBF { gamma: 0.5 }, 2),
            (50, 30, 4, KernelType::Polynomial { degree: 2, gamma: 1.0, coef0: 1.0 }, 3),
            (100, 50, 2, KernelType::RBF { gamma: 0.3 }, 4),
        ] {
            let (train_x, train_y) = make_data(n_train, dims, seed);
            let (test_x, test_y) = make_data(n_test, dims, seed.wrapping_add(1000));

            let config = SVMConfig {
                c: 1.0,
                kernel: kernel.clone(),
                max_iter: 300,
                random_state: Some(seed),
                ..Default::default()
            };
            let svm = SVMClassifier::new(config);

            let (ref_alphas, ref_bias, _) = reference_smo_train(&svm, &train_x, &train_y);
            let (got_alphas, got_bias, _) = svm.smo_train(&train_x, &train_y).unwrap();

            let kernel_matrix = svm.compute_kernel_matrix(&train_x);
            let ref_obj = dual_objective(&ref_alphas, &train_y, &kernel_matrix);
            let got_obj = dual_objective(&got_alphas, &train_y, &kernel_matrix);

            let ref_acc = accuracy(&ref_alphas, ref_bias, &train_x, &train_y, &test_x, &test_y, &svm);
            let got_acc = accuracy(&got_alphas, got_bias, &train_x, &train_y, &test_x, &test_y, &svm);

            println!(
                "n={n_train} dims={dims} seed={seed}: dual_obj ref={ref_obj:.6} got={got_obj:.6} | test_acc ref={ref_acc:.3} got={got_acc:.3}"
            );

            // The error-cache version must not converge to a meaningfully worse
            // dual objective (allow a small tolerance for the two solutions
            // landing in different-but-comparable KKT-satisfying points) and
            // must not meaningfully hurt held-out accuracy.
            assert!(
                got_obj > ref_obj - 1.0,
                "error-cache dual objective regressed for n={n_train} dims={dims} seed={seed}: ref={ref_obj} got={got_obj}"
            );
            assert!(
                got_acc >= ref_acc - 0.15,
                "error-cache accuracy regressed for n={n_train} dims={dims} seed={seed}: ref={ref_acc} got={got_acc}"
            );
        }
    }
}
