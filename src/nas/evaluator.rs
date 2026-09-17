//! Architecture Evaluator
//!
//! Provides utilities for evaluating neural architecture performance.

use ndarray::{Array1, Array2};
use rand::prelude::*;
use rand_xoshiro::Xoshiro256PlusPlus;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use super::search_space::NetworkArchitecture;
use crate::error::Result;

/// Evaluation configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvaluationConfig {
    /// Number of epochs for full evaluation
    pub epochs: usize,
    /// Number of epochs for quick evaluation (proxy)
    pub proxy_epochs: usize,
    /// Batch size
    pub batch_size: usize,
    /// Early stopping patience
    pub patience: usize,
    /// Whether to use early stopping
    pub early_stopping: bool,
    /// Validation split ratio
    pub val_split: f64,
    /// Number of cross-validation folds (0 = no CV)
    pub cv_folds: usize,
}

impl Default for EvaluationConfig {
    fn default() -> Self {
        Self {
            epochs: 100,
            proxy_epochs: 10,
            batch_size: 32,
            patience: 10,
            early_stopping: true,
            val_split: 0.2,
            cv_folds: 0,
        }
    }
}

/// Result of architecture evaluation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvaluationResult {
    /// Architecture identifier/hash
    pub arch_id: u64,
    /// Training loss
    pub train_loss: f64,
    /// Validation loss
    pub val_loss: f64,
    /// Validation accuracy (or other metric)
    pub val_metric: f64,
    /// Test metric (if available)
    pub test_metric: Option<f64>,
    /// Number of parameters
    pub num_params: usize,
    /// Training time in seconds
    pub train_time: f64,
    /// Number of FLOPs (estimated)
    pub flops: Option<usize>,
    /// Additional metrics
    pub extra_metrics: HashMap<String, f64>,
}

impl EvaluationResult {
    /// Create new result
    pub fn new(arch_id: u64) -> Self {
        Self {
            arch_id,
            train_loss: 0.0,
            val_loss: 0.0,
            val_metric: 0.0,
            test_metric: None,
            num_params: 0,
            train_time: 0.0,
            flops: None,
            extra_metrics: HashMap::new(),
        }
    }

    /// Set losses
    pub fn with_losses(mut self, train: f64, val: f64) -> Self {
        self.train_loss = train;
        self.val_loss = val;
        self
    }

    /// Set validation metric
    pub fn with_val_metric(mut self, metric: f64) -> Self {
        self.val_metric = metric;
        self
    }

    /// Set test metric
    pub fn with_test_metric(mut self, metric: f64) -> Self {
        self.test_metric = Some(metric);
        self
    }

    /// Set number of parameters
    pub fn with_num_params(mut self, n: usize) -> Self {
        self.num_params = n;
        self
    }

    /// Set training time
    pub fn with_train_time(mut self, t: f64) -> Self {
        self.train_time = t;
        self
    }

    /// Add extra metric
    pub fn add_metric(mut self, name: impl Into<String>, value: f64) -> Self {
        self.extra_metrics.insert(name.into(), value);
        self
    }
}

/// Architecture evaluator
pub struct ArchitectureEvaluator {
    /// Configuration
    config: EvaluationConfig,
    /// Evaluation cache
    cache: HashMap<u64, EvaluationResult>,
    /// Number of evaluations performed
    eval_count: usize,
}

impl ArchitectureEvaluator {
    /// Create new evaluator
    pub fn new(config: EvaluationConfig) -> Self {
        Self {
            config,
            cache: HashMap::new(),
            eval_count: 0,
        }
    }

    /// Create with default config
    pub fn default_evaluator() -> Self {
        Self::new(EvaluationConfig::default())
    }

    /// Estimate number of parameters in architecture
    pub fn estimate_params(&self, arch: &NetworkArchitecture) -> usize {
        let mut params = 0;
        let hidden = arch.hidden_dim;

        // Input projection
        params += arch.input_dim * hidden;

        // Each cell
        for cell in &arch.cells {
            for op in &cell.operations {
                let op_hidden = op.hidden_dim.unwrap_or(hidden);
                
                match op.op_type {
                    super::search_space::OperationType::Dense => {
                        params += hidden * op_hidden + op_hidden;
                    }
                    super::search_space::OperationType::MultiHeadAttention => {
                        let _heads = op.num_heads.unwrap_or(4);
                        // Q, K, V projections + output
                        params += 4 * hidden * op_hidden;
                    }
                    super::search_space::OperationType::LayerNorm | 
                    super::search_space::OperationType::BatchNorm => {
                        params += 2 * hidden; // gamma, beta
                    }
                    super::search_space::OperationType::Conv1D => {
                        let kernel = op.kernel_size.unwrap_or(3);
                        params += kernel * hidden * op_hidden + op_hidden;
                    }
                    _ => {}
                }
            }
        }

        // Output projection
        params += hidden * arch.output_dim + arch.output_dim;

        params
    }

    /// Estimate FLOPs for architecture
    pub fn estimate_flops(&self, arch: &NetworkArchitecture, seq_len: usize) -> usize {
        let mut flops = 0;
        let hidden = arch.hidden_dim;

        // Input projection
        flops += 2 * seq_len * arch.input_dim * hidden;

        // Each cell
        for cell in &arch.cells {
            for op in &cell.operations {
                let op_hidden = op.hidden_dim.unwrap_or(hidden);
                
                match op.op_type {
                    super::search_space::OperationType::Dense => {
                        flops += 2 * seq_len * hidden * op_hidden;
                    }
                    super::search_space::OperationType::MultiHeadAttention => {
                        // Attention: O(n^2 * d)
                        flops += 4 * seq_len * seq_len * hidden;
                    }
                    super::search_space::OperationType::Conv1D => {
                        let kernel = op.kernel_size.unwrap_or(3);
                        flops += 2 * seq_len * kernel * hidden * op_hidden;
                    }
                    _ => {}
                }
            }
        }

        // Output projection
        flops += 2 * seq_len * hidden * arch.output_dim;

        flops
    }

    /// Check cache for previous evaluation
    pub fn get_cached(&self, arch: &NetworkArchitecture) -> Option<&EvaluationResult> {
        let hash = arch.compute_hash();
        self.cache.get(&hash)
    }

    /// Add result to cache
    pub fn cache_result(&mut self, result: EvaluationResult) {
        self.cache.insert(result.arch_id, result);
    }

    /// Quick proxy evaluation (few epochs)
    pub fn proxy_evaluate(
        &mut self,
        arch: &NetworkArchitecture,
        x_train: &Array2<f64>,
        y_train: &Array1<f64>,
        x_val: &Array2<f64>,
        y_val: &Array1<f64>,
    ) -> Result<EvaluationResult> {
        let arch_id = arch.compute_hash();

        // Check cache
        if let Some(cached) = self.cache.get(&arch_id) {
            return Ok(cached.clone());
        }

        self.eval_count += 1;

        // Estimate parameters
        let num_params = self.estimate_params(arch);

        // Fast proxy training/evaluation that actually uses x/y (see
        // `proxy_fit_evaluate` for what this proxy is and isn't).
        let (train_loss, val_loss, val_metric) =
            self.proxy_fit_evaluate(arch, x_train, y_train, x_val, y_val, self.config.proxy_epochs);

        let result = EvaluationResult::new(arch_id)
            .with_losses(train_loss, val_loss)
            .with_val_metric(val_metric)
            .with_num_params(num_params);

        self.cache_result(result.clone());
        Ok(result)
    }

    /// Full evaluation with more epochs
    pub fn full_evaluate(
        &mut self,
        arch: &NetworkArchitecture,
        x_train: &Array2<f64>,
        y_train: &Array1<f64>,
        x_val: &Array2<f64>,
        y_val: &Array1<f64>,
    ) -> Result<EvaluationResult> {
        let arch_id = arch.compute_hash();
        self.eval_count += 1;

        let start_time = std::time::Instant::now();

        // Estimate parameters
        let num_params = self.estimate_params(arch);
        let _flops = self.estimate_flops(arch, x_train.nrows());

        // Fast proxy training/evaluation with more "epochs" than the quick
        // proxy path (still not full training of the candidate network).
        let (train_loss, val_loss, val_metric) =
            self.proxy_fit_evaluate(arch, x_train, y_train, x_val, y_val, self.config.epochs);

        let train_time = start_time.elapsed().as_secs_f64();

        let result = EvaluationResult::new(arch_id)
            .with_losses(train_loss, val_loss)
            .with_val_metric(val_metric)
            .with_num_params(num_params)
            .with_train_time(train_time);

        Ok(result)
    }

    /// Fast **proxy** evaluation that actually trains on `x_train`/`y_train`
    /// and evaluates on `x_val`/`y_val`, instead of a formula that ignores
    /// the data entirely.
    ///
    /// This is *not* full training of the candidate architecture (that would
    /// require constructing and back-propagating through the actual cell
    /// graph, which this evaluator does not do). Instead it fits a small,
    /// fast model directly with `ndarray`:
    /// - a fixed random projection of the raw features into
    ///   `arch.hidden_dim` (capped) "hidden units" followed by `tanh`,
    ///   giving the proxy at least some sensitivity to the architecture's
    ///   claimed capacity (a crude Extreme-Learning-Machine-style stand-in
    ///   for "a network with this many hidden units"), and
    /// - a linear readout on top of that projection, trained with a few
    ///   steps of full-batch gradient descent (least-squares/MSE loss) on a
    ///   subsample of the training data for speed.
    ///
    /// The returned `val_loss`/`val_metric` are computed from real
    /// predictions on the held-out `x_val`/`y_val`, so search loops built on
    /// this evaluator get a real (if crude and fast) data-dependent signal
    /// rather than one that only rewards architectures with more operations.
    fn proxy_fit_evaluate(
        &self,
        arch: &NetworkArchitecture,
        x_train: &Array2<f64>,
        y_train: &Array1<f64>,
        x_val: &Array2<f64>,
        y_val: &Array1<f64>,
        epochs: usize,
    ) -> (f64, f64, f64) {
        let n_train = x_train.nrows();
        let n_val = x_val.nrows();
        let input_dim = x_train.ncols();

        if n_train == 0 || input_dim == 0 {
            return (0.0, 0.0, 0.0);
        }

        // Seed deterministically off the architecture so repeated proxy
        // evaluations of the same architecture (outside the cache) are
        // reproducible.
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(arch.compute_hash());

        // Subsample the training set for speed — this is a fast proxy, not
        // full training.
        const MAX_SAMPLES: usize = 256;
        let sample_idx: Vec<usize> = if n_train > MAX_SAMPLES {
            let mut idx: Vec<usize> = (0..n_train).collect();
            idx.shuffle(&mut rng);
            idx.truncate(MAX_SAMPLES);
            idx
        } else {
            (0..n_train).collect()
        };
        let n_sub = sample_idx.len().max(1);

        // Random-feature projection sized off the architecture's hidden
        // dimension, so the proxy is at least somewhat architecture-aware.
        let feature_dim = arch.hidden_dim.max(1).min(64);
        let scale = 1.0 / (input_dim as f64).sqrt().max(1.0);
        let mut projection = Array2::<f64>::zeros((input_dim, feature_dim));
        for v in projection.iter_mut() {
            *v = (rng.gen::<f64>() - 0.5) * 2.0 * scale;
        }

        let feat_train = x_train.dot(&projection).mapv(f64::tanh);
        let feat_val = x_val.dot(&projection).mapv(f64::tanh);

        // Linear readout, trained with full-batch gradient descent on MSE
        // loss over the subsample.
        let mut w = Array1::<f64>::zeros(feature_dim);
        let mut b = 0.0f64;
        let lr = 0.1;

        for _ in 0..epochs.max(1) {
            let mut grad_w = Array1::<f64>::zeros(feature_dim);
            let mut grad_b = 0.0f64;

            for &i in &sample_idx {
                let feat = feat_train.row(i);
                let pred = feat.dot(&w) + b;
                let err = pred - y_train[i];
                grad_w = grad_w + &(feat.to_owned() * err);
                grad_b += err;
            }

            let n = n_sub as f64;
            w = w - (grad_w / n) * lr;
            b -= lr * grad_b / n;
        }

        // Training loss (MSE) on the subsample used to fit.
        let mut train_sq_err = 0.0;
        for &i in &sample_idx {
            let pred = feat_train.row(i).dot(&w) + b;
            let err = pred - y_train[i];
            train_sq_err += err * err;
        }
        let train_loss = train_sq_err / n_sub as f64;

        // Real validation loss/metric on the held-out split.
        if n_val == 0 {
            return (train_loss, train_loss, 0.0);
        }

        let mut val_sq_err = 0.0;
        for i in 0..n_val {
            let pred = feat_val.row(i).dot(&w) + b;
            let err = pred - y_val[i];
            val_sq_err += err * err;
        }
        let val_loss = val_sq_err / n_val as f64;

        // "Accuracy-like" metric: 1 - normalized MSE (an R^2-style score),
        // clamped to [0, 1] so it behaves like the accuracy metric callers
        // expect regardless of whether `y` holds regression targets or
        // 0/1 class labels.
        let y_mean = y_val.mean().unwrap_or(0.0);
        let y_var = y_val.iter().map(|v| (v - y_mean).powi(2)).sum::<f64>() / n_val as f64;
        let val_metric = if y_var > 1e-10 {
            (1.0 - val_loss / y_var).clamp(0.0, 1.0)
        } else if val_loss < 1e-6 {
            // Degenerate (near-constant) validation target that we matched.
            1.0
        } else {
            0.0
        };

        (train_loss, val_loss, val_metric)
    }

    /// Get number of evaluations
    pub fn eval_count(&self) -> usize {
        self.eval_count
    }

    /// Get cache size
    pub fn cache_size(&self) -> usize {
        self.cache.len()
    }

    /// Clear cache
    pub fn clear_cache(&mut self) {
        self.cache.clear();
    }
}

/// Multi-fidelity evaluator for efficient search
#[allow(dead_code)]
pub struct MultiFidelityEvaluator {
    /// Fidelity levels (epochs per level)
    fidelity_levels: Vec<usize>,
    /// Base evaluator
    evaluator: ArchitectureEvaluator,
    /// Results at each fidelity level
    results: HashMap<u64, Vec<EvaluationResult>>,
}

#[allow(dead_code)]
impl MultiFidelityEvaluator {
    /// Create new multi-fidelity evaluator
    pub fn new(fidelity_levels: Vec<usize>) -> Self {
        Self {
            fidelity_levels,
            evaluator: ArchitectureEvaluator::default_evaluator(),
            results: HashMap::new(),
        }
    }

    /// Create with Hyperband-style fidelities
    ///
    /// `eta` must be > 1: `eta == 0` would panic on integer division, and
    /// `eta == 1` would make `epochs /= eta` a no-op, so the loop below would
    /// never terminate. Invalid values are clamped up to the documented
    /// minimum of 2, with a warning.
    pub fn hyperband_style(max_epochs: usize, eta: usize) -> Self {
        let eta = if eta > 1 {
            eta
        } else {
            eprintln!(
                "warning: MultiFidelityEvaluator::hyperband_style requires eta > 1, got {}; clamping to 2",
                eta
            );
            2
        };

        let mut levels = Vec::new();
        let mut epochs = max_epochs;
        while epochs >= 1 {
            levels.push(epochs);
            epochs /= eta;
        }
        levels.reverse();
        Self::new(levels)
    }

    /// Evaluate at specific fidelity level
    pub fn evaluate_at_fidelity(
        &mut self,
        arch: &NetworkArchitecture,
        fidelity: usize,
        x_train: &Array2<f64>,
        y_train: &Array1<f64>,
        x_val: &Array2<f64>,
        y_val: &Array1<f64>,
    ) -> Result<EvaluationResult> {
        let epochs = self.fidelity_levels.get(fidelity)
            .copied()
            .unwrap_or(self.fidelity_levels.last().copied().unwrap_or(10));

        self.evaluator.config.proxy_epochs = epochs;
        self.evaluator.proxy_evaluate(arch, x_train, y_train, x_val, y_val)
    }

    /// Get fidelity levels
    pub fn levels(&self) -> &[usize] {
        &self.fidelity_levels
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::search_space::*;

    fn create_test_arch() -> NetworkArchitecture {
        NetworkArchitecture::new(10, 2)
            .with_hidden_dim(64)
            .with_num_layers(2)
            .add_cell(
                Cell::new(CellType::Normal)
                    .add_operation(Operation::dense(64), vec![0])
            )
    }

    #[test]
    fn test_evaluator_creation() {
        let evaluator = ArchitectureEvaluator::default_evaluator();
        assert_eq!(evaluator.eval_count(), 0);
    }

    #[test]
    fn test_estimate_params() {
        let evaluator = ArchitectureEvaluator::default_evaluator();
        let arch = create_test_arch();
        
        let params = evaluator.estimate_params(&arch);
        assert!(params > 0);
    }

    #[test]
    fn test_proxy_evaluate() {
        let mut evaluator = ArchitectureEvaluator::default_evaluator();
        let arch = create_test_arch();
        
        let x_train = Array2::zeros((100, 10));
        let y_train = Array1::zeros(100);
        let x_val = Array2::zeros((20, 10));
        let y_val = Array1::zeros(20);
        
        let result = evaluator.proxy_evaluate(&arch, &x_train, &y_train, &x_val, &y_val).unwrap();
        
        assert!(result.val_metric > 0.0);
        assert!(result.num_params > 0);
    }

    #[test]
    fn test_caching() {
        let mut evaluator = ArchitectureEvaluator::default_evaluator();
        let arch = create_test_arch();
        
        let x = Array2::zeros((100, 10));
        let y = Array1::zeros(100);
        
        // First evaluation
        let _result1 = evaluator.proxy_evaluate(&arch, &x, &y, &x, &y).unwrap();
        assert_eq!(evaluator.eval_count(), 1);
        
        // Should use cache
        let _result2 = evaluator.proxy_evaluate(&arch, &x, &y, &x, &y).unwrap();
        assert_eq!(evaluator.eval_count(), 1); // No new evaluation
    }

    #[test]
    fn test_multi_fidelity() {
        let evaluator = MultiFidelityEvaluator::hyperband_style(81, 3);
        
        // Should have levels: 1, 3, 9, 27, 81
        assert!(!evaluator.levels().is_empty());
    }

    #[test]
    fn test_evaluation_result_builder() {
        let result = EvaluationResult::new(12345)
            .with_losses(0.5, 0.4)
            .with_val_metric(0.85)
            .with_num_params(10000)
            .add_metric("f1_score", 0.82);
        
        assert_eq!(result.train_loss, 0.5);
        assert_eq!(result.val_metric, 0.85);
        assert_eq!(result.extra_metrics.get("f1_score"), Some(&0.82));
    }
}
