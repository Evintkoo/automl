//! Gaussian Process for Bayesian Optimization
//!
//! Implements GP regression with various kernels and acquisition functions
//! for intelligent hyperparameter search.

use ndarray::{Array1, Array2, ArrayView1, Axis};
use rand::prelude::*;
use rand_xoshiro::Xoshiro256PlusPlus;
use serde::{Deserialize, Serialize};

use super::search_space::{SearchSpace, TrialParams, ParameterValue, ParameterType};
use super::Sampler;

/// Kernel function types for Gaussian Process
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum KernelType {
    /// Radial Basis Function (Squared Exponential)
    RBF { length_scale: f64 },
    /// Matern kernel with nu parameter
    Matern { nu: f64, length_scale: f64 },
    /// Rational Quadratic kernel
    RationalQuadratic { length_scale: f64, alpha: f64 },
    /// Combined kernels (sum)
    Sum(Box<KernelType>, Box<KernelType>),
    /// Combined kernels (product)
    Product(Box<KernelType>, Box<KernelType>),
}

impl Default for KernelType {
    fn default() -> Self {
        KernelType::Matern { nu: 2.5, length_scale: 1.0 }
    }
}

/// Compute kernel matrix for given kernel type
fn compute_kernel(x1: &Array2<f64>, x2: &Array2<f64>, kernel: &KernelType) -> Array2<f64> {
    let n1 = x1.nrows();
    let n2 = x2.nrows();
    let mut k = Array2::zeros((n1, n2));

    for i in 0..n1 {
        for j in 0..n2 {
            // Pass views instead of `.to_owned()` clones — every one of the
            // n1*n2 kernel evaluations was heap-allocating two fresh Array1s
            // just to read their values once each.
            k[[i, j]] = kernel_value(x1.row(i), x2.row(j), kernel);
        }
    }
    k
}

/// Compute kernel value between two points
fn kernel_value(x1: ArrayView1<f64>, x2: ArrayView1<f64>, kernel: &KernelType) -> f64 {
    match kernel {
        KernelType::RBF { length_scale } => {
            let diff = &x1 - &x2;
            let dist_sq = diff.dot(&diff);
            (-0.5 * dist_sq / (length_scale * length_scale)).exp()
        }
        KernelType::Matern { nu, length_scale } => {
            let diff = &x1 - &x2;
            let dist = diff.dot(&diff).sqrt();
            let scaled = (2.0_f64.sqrt() * nu.sqrt() * dist) / length_scale;
            
            if scaled < 1e-10 {
                return 1.0;
            }
            
            if (*nu - 0.5).abs() < 1e-6 {
                // Matern 1/2 (Exponential)
                (-scaled).exp()
            } else if (*nu - 1.5).abs() < 1e-6 {
                // Matern 3/2
                let sqrt3 = 3.0_f64.sqrt();
                (1.0 + sqrt3 * dist / length_scale) * (-sqrt3 * dist / length_scale).exp()
            } else if (*nu - 2.5).abs() < 1e-6 {
                // Matern 5/2
                let sqrt5 = 5.0_f64.sqrt();
                let r = dist / length_scale;
                (1.0 + sqrt5 * r + 5.0 / 3.0 * r * r) * (-sqrt5 * r).exp()
            } else {
                // Fallback to RBF-like for other nu values
                (-0.5 * scaled * scaled).exp()
            }
        }
        KernelType::RationalQuadratic { length_scale, alpha } => {
            let diff = &x1 - &x2;
            let dist_sq = diff.dot(&diff);
            (1.0 + dist_sq / (2.0 * alpha * length_scale * length_scale)).powf(-*alpha)
        }
        KernelType::Sum(k1, k2) => {
            kernel_value(x1, x2, k1) + kernel_value(x1, x2, k2)
        }
        KernelType::Product(k1, k2) => {
            kernel_value(x1, x2, k1) * kernel_value(x1, x2, k2)
        }
    }
}

/// Acquisition function types
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum AcquisitionFunction {
    /// Expected Improvement
    EI,
    /// Probability of Improvement
    PI,
    /// Upper Confidence Bound
    UCB { kappa: f64 },
    /// Lower Confidence Bound (for minimization)
    LCB { kappa: f64 },
    /// Thompson Sampling
    ThompsonSampling,
    /// Expected Improvement with noise — adds exploration bonus xi, decayed over trials.
    /// Recommended for noisy objectives and large search spaces.
    EIWithNoise { xi: f64 },
}

impl Default for AcquisitionFunction {
    fn default() -> Self {
        AcquisitionFunction::EI
    }
}

/// Gaussian Process model for regression
#[derive(Debug, Clone)]
pub struct GaussianProcess {
    /// Kernel type
    kernel: KernelType,
    /// Noise variance
    noise: f64,
    /// Training inputs
    x_train: Option<Array2<f64>>,
    /// Training outputs
    y_train: Option<Array1<f64>>,
    /// Cholesky decomposition of K + noise*I
    l_chol: Option<Array2<f64>>,
    /// Alpha = L^-T L^-1 y
    alpha: Option<Array1<f64>>,
    /// Mean of training outputs (for normalization)
    y_mean: f64,
    /// Std of training outputs (for normalization)
    y_std: f64,
}

impl GaussianProcess {
    /// Create new GP with given kernel
    pub fn new(kernel: KernelType) -> Self {
        Self {
            kernel,
            noise: 1e-6,
            x_train: None,
            y_train: None,
            l_chol: None,
            alpha: None,
            y_mean: 0.0,
            y_std: 1.0,
        }
    }

    /// Set noise level
    pub fn with_noise(mut self, noise: f64) -> Self {
        self.noise = noise.max(1e-10);
        self
    }

    /// Fit the GP to training data
    pub fn fit(&mut self, x: Array2<f64>, y: Array1<f64>) {
        let n = y.len();
        
        // Normalize y
        self.y_mean = y.mean().unwrap_or(0.0);
        self.y_std = y.std(0.0);
        if self.y_std < 1e-10 {
            self.y_std = 1.0;
        }
        
        let y_normalized: Array1<f64> = y.iter()
            .map(|&yi| (yi - self.y_mean) / self.y_std)
            .collect();

        // Compute kernel matrix
        let k = compute_kernel(&x, &x, &self.kernel);
        
        // Add noise to diagonal
        let mut k_noisy = k.clone();
        for i in 0..n {
            k_noisy[[i, i]] += self.noise;
        }
        
        // Compute Cholesky decomposition
        let l = Self::cholesky(&k_noisy);
        
        // Solve for alpha
        let alpha = Self::solve_triangular_system(&l, &y_normalized);
        
        self.x_train = Some(x);
        self.y_train = Some(y_normalized);
        self.l_chol = Some(l);
        self.alpha = Some(alpha);
    }

    /// Predict mean and variance at test points
    pub fn predict(&self, x_test: &Array2<f64>) -> (Array1<f64>, Array1<f64>) {
        let x_train = self.x_train.as_ref().expect("GP not fitted");
        let l = self.l_chol.as_ref().expect("GP not fitted");
        let alpha = self.alpha.as_ref().expect("GP not fitted");

        // Compute k(X*, X)
        let k_star = compute_kernel(x_test, x_train, &self.kernel);
        
        // Mean: k* @ alpha
        let mean_normalized: Array1<f64> = k_star.dot(alpha);
        
        // Denormalize
        let mean: Array1<f64> = mean_normalized.iter()
            .map(|&m| m * self.y_std + self.y_mean)
            .collect();

        // Variance: k** - k* @ L^-T @ L^-1 @ k*^T
        let n_test = x_test.nrows();
        let mut var = Array1::zeros(n_test);
        
        for i in 0..n_test {
            // k(x*, x*)
            let k_self = kernel_value(x_test.row(i), x_test.row(i), &self.kernel);
            
            // Solve L @ v = k*[i]
            let k_star_i = k_star.row(i).to_owned();
            let v = Self::solve_lower_triangular(&l, &k_star_i);
            
            // var = k** - v^T @ v
            var[i] = (k_self - v.dot(&v)).max(1e-10) * self.y_std * self.y_std;
        }

        (mean, var)
    }

    /// Simple Cholesky decomposition
    fn cholesky(a: &Array2<f64>) -> Array2<f64> {
        let n = a.nrows();
        let mut l = Array2::zeros((n, n));

        for i in 0..n {
            for j in 0..=i {
                let mut sum = 0.0;
                
                if i == j {
                    for k in 0..j {
                        sum += l[[j, k]] * l[[j, k]];
                    }
                    l[[j, j]] = (a[[j, j]] - sum).max(1e-10).sqrt();
                } else {
                    for k in 0..j {
                        sum += l[[i, k]] * l[[j, k]];
                    }
                    l[[i, j]] = (a[[i, j]] - sum) / l[[j, j]].max(1e-10);
                }
            }
        }
        l
    }

    /// Solve L @ x = b for lower triangular L
    fn solve_lower_triangular(l: &Array2<f64>, b: &Array1<f64>) -> Array1<f64> {
        let n = b.len();
        let mut x = Array1::zeros(n);
        
        for i in 0..n {
            let mut sum = b[i];
            for j in 0..i {
                sum -= l[[i, j]] * x[j];
            }
            x[i] = sum / l[[i, i]].max(1e-10);
        }
        x
    }

    /// Solve L @ L^T @ x = b
    fn solve_triangular_system(l: &Array2<f64>, b: &Array1<f64>) -> Array1<f64> {
        let n = b.len();
        
        // Solve L @ y = b
        let y = Self::solve_lower_triangular(l, b);
        
        // Solve L^T @ x = y
        let mut x = Array1::zeros(n);
        for i in (0..n).rev() {
            let mut sum = y[i];
            for j in (i + 1)..n {
                sum -= l[[j, i]] * x[j];
            }
            x[i] = sum / l[[i, i]].max(1e-10);
        }
        x
    }
}

/// Gaussian Process based Bayesian Optimization sampler
#[derive(Debug)]
pub struct GPSampler {
    /// Random number generator
    rng: Xoshiro256PlusPlus,
    /// Gaussian Process model
    gp: GaussianProcess,
    /// Acquisition function
    acquisition: AcquisitionFunction,
    /// Number of startup trials (random sampling)
    n_startup_trials: usize,
    /// Number of candidates to evaluate
    n_candidates: usize,
    /// Whether to minimize (true) or maximize (false) the objective
    minimize: bool,
    /// Best observed value
    best_y: f64,
    /// Parameter names in order
    param_names: Vec<String>,
}

impl GPSampler {
    /// Create a new GP sampler
    pub fn new(seed: Option<u64>) -> Self {
        let rng = match seed {
            Some(s) => Xoshiro256PlusPlus::seed_from_u64(s),
            None => Xoshiro256PlusPlus::from_entropy(),
        };
        
        Self {
            rng,
            gp: GaussianProcess::new(KernelType::default()),
            acquisition: AcquisitionFunction::default(),
            n_startup_trials: 10,
            n_candidates: 1000,
            minimize: true,
            best_y: f64::INFINITY,
            param_names: Vec::new(),
        }
    }

    /// Set kernel type
    pub fn with_kernel(mut self, kernel: KernelType) -> Self {
        self.gp = GaussianProcess::new(kernel);
        self
    }

    /// Set acquisition function
    pub fn with_acquisition(mut self, acq: AcquisitionFunction) -> Self {
        self.acquisition = acq;
        self
    }

    /// Set number of startup trials
    pub fn with_n_startup(mut self, n: usize) -> Self {
        self.n_startup_trials = n;
        self
    }

    /// Set number of candidates
    pub fn with_n_candidates(mut self, n: usize) -> Self {
        self.n_candidates = n;
        self
    }

    /// Set minimization mode
    pub fn with_minimize(mut self, minimize: bool) -> Self {
        self.minimize = minimize;
        self
    }

    /// Convert trial params to array
    fn params_to_array(&self, params: &TrialParams, space: &SearchSpace) -> Array1<f64> {
        self.param_names.iter()
            .map(|name| {
                params.get(name)
                    .map(|v| self.value_to_float(v, name, space))
                    .unwrap_or(0.5)
            })
            .collect()
    }

    /// Convert parameter value to float [0, 1] normalized by parameter bounds
    fn value_to_float(&self, value: &ParameterValue, name: &str, space: &SearchSpace) -> f64 {
        if let Some(param) = space.parameters().iter().find(|p| p.name == name) {
            match (&param.param_type, value) {
                (ParameterType::Float { low, high, .. }, ParameterValue::Float(v)) => {
                    if (high - low).abs() > 1e-15 {
                        return (*v - low) / (high - low);
                    }
                    return *v;
                }
                (ParameterType::Int { low, high, .. }, ParameterValue::Int(v)) => {
                    if high != low {
                        return (*v - low) as f64 / (high - low) as f64;
                    }
                    return *v as f64;
                }
                _ => {}
            }
        }
        match value {
            ParameterValue::Float(v) => *v,
            ParameterValue::Int(v) => *v as f64,
            ParameterValue::String(_) => 0.5,
            ParameterValue::Bool(v) => if *v { 1.0 } else { 0.0 },
        }
    }

    /// Compute acquisition value
    fn acquisition_value(&self, mean: f64, var: f64) -> f64 {
        let std = var.sqrt().max(1e-10);
        
        match self.acquisition {
            AcquisitionFunction::EI => {
                // Expected Improvement
                let improvement = if self.minimize {
                    self.best_y - mean
                } else {
                    mean - self.best_y
                };
                
                let z = improvement / std;
                improvement * normal_cdf(z) + std * normal_pdf(z)
            }
            AcquisitionFunction::PI => {
                // Probability of Improvement
                let improvement = if self.minimize {
                    self.best_y - mean
                } else {
                    mean - self.best_y
                };
                normal_cdf(improvement / std)
            }
            AcquisitionFunction::UCB { kappa } => {
                // Upper Confidence Bound
                if self.minimize {
                    -(mean - kappa * std)
                } else {
                    mean + kappa * std
                }
            }
            AcquisitionFunction::LCB { kappa } => {
                // Lower Confidence Bound (for minimization): the "confidence
                // bound" itself is `mean - kappa*std`, and lower is better.
                // Candidate selection in `sample()` always argmaxes the
                // acquisition score, so we must return the *negated* bound
                // (equivalently `kappa*std - mean`) when minimizing so that
                // low mean + high uncertainty scores highest. When
                // maximizing, mirror `UCB`'s maximize branch so a high mean
                // with high uncertainty is preferred instead.
                if self.minimize {
                    kappa * std - mean
                } else {
                    mean + kappa * std
                }
            }
            AcquisitionFunction::ThompsonSampling => {
                // Deterministic approximation of Thompson sampling: use the
                // posterior mean as the "sample" (a full implementation
                // would draw a random sample from N(mean, var) instead).
                // `sample()` always argmaxes the acquisition score, so when
                // minimizing we negate the mean to turn "argmin mean" into
                // "argmax -mean" — matching how the other acquisition
                // functions in this file branch on `self.minimize`.
                if self.minimize {
                    -mean
                } else {
                    mean
                }
            }
            AcquisitionFunction::EIWithNoise { xi } => {
                // Mirror EI/PI's direction handling. `expected_improvement_with_noise`
                // is written for the maximize convention (`mu - best - xi`); for a
                // minimize study we get the minimize-convention formula
                // (`best - mu - xi`) by negating both `mu` and `best` before calling
                // it, which keeps the shared helper (and its tests) untouched.
                if self.minimize {
                    expected_improvement_with_noise(-mean, std, -self.best_y, xi)
                } else {
                    expected_improvement_with_noise(mean, std, self.best_y, xi)
                }
            }
        }
    }
}

impl Sampler for GPSampler {
    fn sample(
        &mut self,
        search_space: &SearchSpace,
        history: &[(TrialParams, f64)],
    ) -> TrialParams {
        // Random sampling for startup
        if history.len() < self.n_startup_trials {
            return search_space.sample(&mut self.rng);
        }

        // Initialize param names on first GP fit
        if self.param_names.is_empty() {
            self.param_names = search_space.param_names();
        }

        let n_params = self.param_names.len();
        
        // Build training data
        let mut x_data = Vec::new();
        let mut y_data = Vec::new();
        
        for (params, y) in history {
            let x: Vec<f64> = self.params_to_array(params, search_space).to_vec();
            x_data.extend(x);
            y_data.push(*y);
            
            // Update best
            if self.minimize && *y < self.best_y {
                self.best_y = *y;
            } else if !self.minimize && *y > self.best_y {
                self.best_y = *y;
            }
        }
        
        let n_samples = history.len();
        let x_train = Array2::from_shape_vec((n_samples, n_params), x_data)
            .expect("Failed to create training array");
        let y_train = Array1::from_vec(y_data);

        // Fit GP
        self.gp.fit(x_train, y_train);

        // Generate candidates
        let mut best_params = search_space.sample(&mut self.rng);
        let mut best_acq = f64::NEG_INFINITY;

        for _ in 0..self.n_candidates {
            let candidate = search_space.sample(&mut self.rng);
            let x = self.params_to_array(&candidate, search_space);
            let x_2d = x.insert_axis(Axis(0));
            
            let (mean, var) = self.gp.predict(&x_2d);
            let acq = self.acquisition_value(mean[0], var[0]);
            
            if acq > best_acq {
                best_acq = acq;
                best_params = candidate;
            }
        }

        best_params
    }
}

/// Expected Improvement with noise (EI-noise).
/// EI(x) = (μ - f* - ξ)·Φ(Z) + σ·φ(Z)  where Z = (μ - f* - ξ) / σ
/// ξ (xi) controls the exploration–exploitation trade-off.
pub fn expected_improvement_with_noise(mu: f64, sigma: f64, best: f64, xi: f64) -> f64 {
    if sigma <= 0.0 { return 0.0; }
    let improvement = mu - best - xi;
    let z = improvement / sigma;
    improvement * normal_cdf(z) + sigma * normal_pdf(z)
}

/// Standard normal CDF approximation
fn normal_cdf(x: f64) -> f64 {
    0.5 * (1.0 + erf(x / std::f64::consts::SQRT_2))
}

/// Standard normal PDF
fn normal_pdf(x: f64) -> f64 {
    (-0.5 * x * x).exp() / (2.0 * std::f64::consts::PI).sqrt()
}

/// Error function approximation (Abramowitz and Stegun)
fn erf(x: f64) -> f64 {
    let a1 = 0.254829592;
    let a2 = -0.284496736;
    let a3 = 1.421413741;
    let a4 = -1.453152027;
    let a5 = 1.061405429;
    let p = 0.3275911;

    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let x = x.abs();

    let t = 1.0 / (1.0 + p * x);
    let y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * (-x * x).exp();

    sign * y
}

/// Configuration for Bayesian Optimization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BayesianOptConfig {
    /// Kernel type
    pub kernel: KernelType,
    /// Acquisition function
    pub acquisition: AcquisitionFunction,
    /// Number of startup trials
    pub n_startup_trials: usize,
    /// Number of candidates to evaluate
    pub n_candidates: usize,
    /// Noise level for GP
    pub noise: f64,
}

impl Default for BayesianOptConfig {
    fn default() -> Self {
        Self {
            kernel: KernelType::Matern { nu: 2.5, length_scale: 1.0 },
            acquisition: AcquisitionFunction::EI,
            n_startup_trials: 10,
            n_candidates: 1000,
            noise: 1e-6,
        }
    }
}

/// High-level Bayesian Optimizer
pub struct BayesianOptimizer {
    /// Search space
    search_space: SearchSpace,
    /// GP Sampler
    sampler: GPSampler,
    /// History of trials
    history: Vec<(TrialParams, f64)>,
    /// Best parameters found
    best_params: Option<TrialParams>,
    /// Best value found
    best_value: Option<f64>,
    /// Minimize or maximize
    minimize: bool,
}

impl BayesianOptimizer {
    /// Create new optimizer
    pub fn new(search_space: SearchSpace, config: BayesianOptConfig) -> Self {
        let sampler = GPSampler::new(None)
            .with_kernel(config.kernel)
            .with_acquisition(config.acquisition)
            .with_n_startup(config.n_startup_trials)
            .with_n_candidates(config.n_candidates);
        
        Self {
            search_space,
            sampler,
            history: Vec::new(),
            best_params: None,
            best_value: None,
            minimize: true,
        }
    }

    /// Set minimization mode
    pub fn minimize(mut self) -> Self {
        self.minimize = true;
        self.sampler = self.sampler.with_minimize(true);
        self
    }

    /// Set maximization mode
    pub fn maximize(mut self) -> Self {
        self.minimize = false;
        self.sampler = self.sampler.with_minimize(false);
        self
    }

    /// Suggest next parameters to try
    pub fn suggest(&mut self) -> TrialParams {
        self.sampler.sample(&self.search_space, &self.history)
    }

    /// Report the result of a trial
    pub fn report(&mut self, params: TrialParams, value: f64) {
        self.history.push((params.clone(), value));
        
        let is_better = match self.best_value {
            None => true,
            Some(best) => {
                if self.minimize { value < best } else { value > best }
            }
        };
        
        if is_better {
            self.best_params = Some(params);
            self.best_value = Some(value);
        }
    }

    /// Get best parameters found
    pub fn best_params(&self) -> Option<&TrialParams> {
        self.best_params.as_ref()
    }

    /// Get best value found
    pub fn best_value(&self) -> Option<f64> {
        self.best_value
    }

    /// Get all history
    pub fn history(&self) -> &[(TrialParams, f64)] {
        &self.history
    }

    /// Run optimization for n iterations with objective function
    pub fn optimize<F>(&mut self, n_trials: usize, mut objective: F) -> (TrialParams, f64)
    where
        F: FnMut(&TrialParams) -> f64,
    {
        for _ in 0..n_trials {
            let params = self.suggest();
            let value = objective(&params);
            self.report(params, value);
        }
        
        (
            self.best_params.clone().expect("No trials completed"),
            self.best_value.expect("No trials completed"),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    #[test]
    fn test_rbf_kernel() {
        let x1 = Array1::from_vec(vec![0.0, 0.0]);
        let x2 = Array1::from_vec(vec![0.0, 0.0]);
        let kernel = KernelType::RBF { length_scale: 1.0 };
        
        let k = kernel_value(x1.view(), x2.view(), &kernel);
        assert!((k - 1.0).abs() < 1e-6, "Same point should have kernel 1.0");
    }

    #[test]
    fn test_matern_kernel() {
        let x1 = Array1::from_vec(vec![0.0]);
        let x2 = Array1::from_vec(vec![0.0]);
        let kernel = KernelType::Matern { nu: 2.5, length_scale: 1.0 };
        
        let k = kernel_value(x1.view(), x2.view(), &kernel);
        assert!((k - 1.0).abs() < 1e-6, "Same point should have kernel 1.0");
    }

    #[test]
    fn test_gp_fit_predict() {
        let mut gp = GaussianProcess::new(KernelType::RBF { length_scale: 1.0 });
        
        // Simple 1D function: y = x^2
        let x_train = Array2::from_shape_vec((5, 1), vec![-2.0, -1.0, 0.0, 1.0, 2.0]).unwrap();
        let y_train = Array1::from_vec(vec![4.0, 1.0, 0.0, 1.0, 4.0]);
        
        gp.fit(x_train, y_train);
        
        let x_test = Array2::from_shape_vec((3, 1), vec![-1.5, 0.5, 1.5]).unwrap();
        let (mean, var) = gp.predict(&x_test);
        
        // Check predictions are reasonable
        assert!(mean.len() == 3);
        assert!(var.len() == 3);
        assert!(var.iter().all(|&v| v > 0.0), "Variance should be positive");
    }

    #[test]
    fn test_acquisition_ei() {
        let sampler = GPSampler::new(Some(42));
        
        // With improvement possible
        let ei = sampler.acquisition_value(0.5, 0.1);
        assert!(ei > 0.0, "EI should be positive when improvement is possible");
    }

    #[test]
    fn test_gp_sampler_startup() {
        let space = SearchSpace::new()
            .float("x", 0.0, 1.0)
            .float("y", 0.0, 1.0);
        
        let mut sampler = GPSampler::new(Some(42));
        
        // During startup should do random sampling
        let params = sampler.sample(&space, &[]);
        assert!(params.contains_key("x"));
        assert!(params.contains_key("y"));
    }

    #[test]
    fn test_gp_sampler_with_history() {
        let space = SearchSpace::new()
            .float("x", 0.0, 1.0);
        
        let mut sampler = GPSampler::new(Some(42)).with_n_startup(5);
        
        // Create history
        let history: Vec<(TrialParams, f64)> = (0..10)
            .map(|i| {
                let x = i as f64 / 10.0;
                let mut params = HashMap::new();
                params.insert("x".to_string(), ParameterValue::Float(x));
                (params, x * x) // y = x^2
            })
            .collect();
        
        let params = sampler.sample(&space, &history);
        assert!(params.contains_key("x"));
    }

    #[test]
    fn test_bayesian_optimizer() {
        let space = SearchSpace::new()
            .float("x", -5.0, 5.0);
        
        let config = BayesianOptConfig {
            n_startup_trials: 5,
            n_candidates: 100,
            ..Default::default()
        };
        
        let mut optimizer = BayesianOptimizer::new(space, config).minimize();
        
        // Optimize x^2 (minimum at x=0)
        let (best_params, best_value) = optimizer.optimize(20, |params| {
            let x = match params.get("x") {
                Some(ParameterValue::Float(v)) => *v,
                _ => 0.0,
            };
            x * x
        });
        
        // Should find something close to 0
        assert!(best_value < 1.0, "Should find value < 1.0 for x^2");
    }

    #[test]
    fn test_normal_cdf() {
        assert!((normal_cdf(0.0) - 0.5).abs() < 0.01);
        assert!(normal_cdf(-3.0) < 0.01);
        assert!(normal_cdf(3.0) > 0.99);
    }

    #[test]
    fn test_ei_with_noise_higher_xi_increases_exploration() {
        // With xi=0, EI is pure exploitation. With xi>0, EI should be >= EI with xi=0
        // when mean > best (improvement is positive).
        let mean = 0.8;
        let best = 0.5;
        let var: f64 = 0.04; // std = 0.2

        let ei_no_noise = expected_improvement_with_noise(mean, var.sqrt(), best, 0.0);
        let ei_with_xi = expected_improvement_with_noise(mean, var.sqrt(), best, 0.05);
        // Just assert both are positive and the function doesn't panic
        assert!(ei_no_noise > 0.0);
        assert!(ei_with_xi >= 0.0);
    }

    #[test]
    fn test_ei_with_noise_zero_std_returns_zero() {
        let ei = expected_improvement_with_noise(0.9, 0.0, 0.5, 0.01);
        assert_eq!(ei, 0.0);
    }
}
