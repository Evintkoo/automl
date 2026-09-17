//! XGBoost-style gradient boosting with second-order approximation
//!
//! Key differences from standard gradient boosting:
//! - Uses both gradient (first derivative) and hessian (second derivative) of loss
//! - Regularized leaf weights: w* = -G / (H + lambda)
//! - Gain-based split scoring: Gain = 0.5 * [GL²/(HL+λ) + GR²/(HR+λ) - (GL+GR)²/(HL+HR+λ)] - γ
//! - Built-in L1 (alpha) and L2 (lambda) regularization
//! - Minimum child weight constraint

use crate::error::Result;
use ndarray::{Array1, Array2};
use rand::prelude::*;
use rand_xoshiro::Xoshiro256PlusPlus;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

/// XGBoost configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct XGBoostConfig {
    pub n_estimators: usize,
    pub learning_rate: f64,
    pub max_depth: usize,
    pub min_child_weight: f64,
    /// L2 regularization on leaf weights
    pub reg_lambda: f64,
    /// L1 regularization on leaf weights
    pub reg_alpha: f64,
    /// Minimum loss reduction to make a split (gamma)
    pub gamma: f64,
    pub subsample: f64,
    pub colsample_bytree: f64,
    pub random_state: Option<u64>,
}

impl Default for XGBoostConfig {
    fn default() -> Self {
        Self {
            n_estimators: 100,
            learning_rate: 0.3,
            max_depth: 6,
            min_child_weight: 1.0,
            reg_lambda: 1.0,
            reg_alpha: 0.0,
            gamma: 0.0,
            subsample: 1.0,
            colsample_bytree: 1.0,
            random_state: Some(42),
        }
    }
}

/// A single node in the XGBoost tree
#[derive(Debug, Clone, Serialize, Deserialize)]
enum XGBNode {
    Leaf { weight: f64 },
    Split {
        feature: usize,
        threshold: f64,
        left: Box<XGBNode>,
        right: Box<XGBNode>,
    },
}

impl XGBNode {
    fn predict(&self, sample: &[f64]) -> f64 {
        match self {
            XGBNode::Leaf { weight } => *weight,
            XGBNode::Split { feature, threshold, left, right } => {
                if sample[*feature] <= *threshold {
                    left.predict(sample)
                } else {
                    right.predict(sample)
                }
            }
        }
    }
}

/// Build an XGBoost tree using exact greedy split finding
fn build_xgb_tree(
    x: &Array2<f64>,
    grad: &Array1<f64>,
    hess: &Array1<f64>,
    indices: &[usize],
    feature_indices: &[usize],
    depth: usize,
    config: &XGBoostConfig,
) -> XGBNode {
    let n = indices.len();

    // Compute leaf weight with L1/L2 regularization
    let g_sum: f64 = indices.iter().map(|&i| grad[i]).sum();
    let h_sum: f64 = indices.iter().map(|&i| hess[i]).sum();

    let leaf_weight = compute_leaf_weight(g_sum, h_sum, config.reg_lambda, config.reg_alpha);

    // Stopping conditions
    if depth >= config.max_depth || n < 2 || h_sum < config.min_child_weight {
        return XGBNode::Leaf { weight: leaf_weight };
    }

    // Find best split across features (parallelized), pass pre-computed totals
    let best_split = feature_indices.par_iter().filter_map(|&f| {
        find_best_split_for_feature(x, grad, hess, indices, f, config, g_sum, h_sum)
    }).max_by(|a, b| a.3.partial_cmp(&b.3).unwrap_or(std::cmp::Ordering::Equal));

    match best_split {
        Some((feature, threshold, _split_pos, gain)) if gain > config.gamma => {
            let (left_idx, right_idx): (Vec<usize>, Vec<usize>) =
                indices.iter().partition(|&&i| x[[i, feature]] <= threshold);

            if left_idx.is_empty() || right_idx.is_empty() {
                return XGBNode::Leaf { weight: leaf_weight };
            }

            let left = build_xgb_tree(x, grad, hess, &left_idx, feature_indices, depth + 1, config);
            let right = build_xgb_tree(x, grad, hess, &right_idx, feature_indices, depth + 1, config);

            XGBNode::Split {
                feature,
                threshold,
                left: Box::new(left),
                right: Box::new(right),
            }
        }
        _ => XGBNode::Leaf { weight: leaf_weight },
    }
}

/// Optimal leaf weight with L1 (alpha) and L2 (lambda) regularization
fn compute_leaf_weight(g_sum: f64, h_sum: f64, lambda: f64, alpha: f64) -> f64 {
    if alpha > 0.0 {
        // Soft-threshold for L1
        let g_adj = if g_sum > alpha {
            g_sum - alpha
        } else if g_sum < -alpha {
            g_sum + alpha
        } else {
            return 0.0;
        };
        -g_adj / (h_sum + lambda)
    } else {
        -g_sum / (h_sum + lambda)
    }
}

/// Find best split for a single feature using exact greedy method
fn find_best_split_for_feature(
    x: &Array2<f64>,
    grad: &Array1<f64>,
    hess: &Array1<f64>,
    indices: &[usize],
    feature: usize,
    config: &XGBoostConfig,
    g_total: f64,
    h_total: f64,
) -> Option<(usize, f64, usize, f64)> {
    // Sort indices by feature value
    let mut sorted_indices: Vec<usize> = indices.to_vec();
    sorted_indices.sort_by(|&a, &b| {
        x[[a, feature]].partial_cmp(&x[[b, feature]]).unwrap_or(std::cmp::Ordering::Equal)
    });

    let mut g_left = 0.0;
    let mut h_left = 0.0;
    let mut best_gain = f64::NEG_INFINITY;
    let mut best_threshold = 0.0;
    let mut best_pos = 0;

    let lambda = config.reg_lambda;

    for (pos, &idx) in sorted_indices.iter().enumerate() {
        g_left += grad[idx];
        h_left += hess[idx];

        // Skip if next sample has same feature value (avoid identical split)
        if pos + 1 < sorted_indices.len() {
            let next_idx = sorted_indices[pos + 1];
            if (x[[idx, feature]] - x[[next_idx, feature]]).abs() < 1e-12 {
                continue;
            }
        }

        let g_right = g_total - g_left;
        let h_right = h_total - h_left;

        // Min child weight check
        if h_left < config.min_child_weight || h_right < config.min_child_weight {
            continue;
        }

        // XGBoost gain formula
        let gain = 0.5 * (
            (g_left * g_left) / (h_left + lambda)
            + (g_right * g_right) / (h_right + lambda)
            - (g_total * g_total) / (h_total + lambda)
        );

        if gain > best_gain {
            best_gain = gain;
            best_threshold = if pos + 1 < sorted_indices.len() {
                let next_idx = sorted_indices[pos + 1];
                (x[[idx, feature]] + x[[next_idx, feature]]) / 2.0
            } else {
                x[[idx, feature]]
            };
            best_pos = pos;
        }
    }

    if best_gain > f64::NEG_INFINITY {
        Some((feature, best_threshold, best_pos, best_gain))
    } else {
        None
    }
}

// ─── XGBoost Regressor ─────────────────────────────────────────────────────

/// XGBoost Regressor (squared error loss)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct XGBoostRegressor {
    config: XGBoostConfig,
    trees: Vec<XGBNode>,
    base_score: f64,
    feature_importances: Vec<f64>,
    n_features: usize,
}

impl XGBoostRegressor {
    pub fn new(config: XGBoostConfig) -> Self {
        Self {
            config,
            trees: Vec::new(),
            base_score: 0.0,
            feature_importances: Vec::new(),
            n_features: 0,
        }
    }

    pub fn fit(&mut self, x: &Array2<f64>, y: &Array1<f64>) -> Result<()> {
        let n_samples = x.nrows();
        let n_features = x.ncols();
        self.n_features = n_features;

        // Base prediction = mean(y)
        self.base_score = y.mean().unwrap_or(0.0);
        let mut preds = Array1::from_elem(n_samples, self.base_score);

        let mut rng = match self.config.random_state {
            Some(seed) => Xoshiro256PlusPlus::seed_from_u64(seed),
            None => Xoshiro256PlusPlus::from_entropy(),
        };

        self.trees.clear();

        for _ in 0..self.config.n_estimators {
            // Squared error: grad = pred - y, hess = 1.0
            let grad: Array1<f64> = &preds - y;
            let hess = Array1::from_elem(n_samples, 1.0);

            // Subsample rows
            let row_indices = subsample(&mut rng, n_samples, self.config.subsample);
            let col_indices = subsample(&mut rng, n_features, self.config.colsample_bytree);

            let tree = build_xgb_tree(x, &grad, &hess, &row_indices, &col_indices, 0, &self.config);

            // Update predictions for ALL rows, not just the ones in `row_indices` — subsample
            // only controls which rows are used to FIT this round's tree; every row's cached
            // prediction must still receive this round's contribution, otherwise the rows
            // left out of the sample keep a stale prediction that a fresh `predict()` call
            // would not reproduce (tree nodes reference the original/full feature space, so
            // this is safe to evaluate on every row regardless of `colsample_bytree`).
            // Matches lightgbm.rs/catboost.rs.
            for i in 0..n_samples {
                let row = x.row(i);
                preds[i] += self.config.learning_rate * tree.predict(row.as_slice().unwrap());
            }

            self.trees.push(tree);
        }

        Ok(())
    }

    pub fn predict(&self, x: &Array2<f64>) -> Result<Array1<f64>> {
        let n = x.nrows();
        let mut preds = Array1::from_elem(n, self.base_score);
        for i in 0..n {
            let sample = x.row(i);
            let s = sample.as_slice().unwrap();
            for tree in &self.trees {
                preds[i] += self.config.learning_rate * tree.predict(s);
            }
        }
        Ok(preds)
    }

    pub fn score(&self, x: &Array2<f64>, y: &Array1<f64>) -> Result<f64> {
        let p = self.predict(x)?;
        let ym = y.mean().unwrap_or(0.0);
        let ss_res = (&p - y).mapv(|v| v * v).sum();
        let ss_tot = y.mapv(|v| (v - ym).powi(2)).sum();
        Ok(if ss_tot == 0.0 { 1.0 } else { 1.0 - ss_res / ss_tot })
    }

    /// Compute feature importances by counting splits across all trees
    pub fn feature_importances(&self) -> Option<Array1<f64>> {
        if self.n_features == 0 { return None; }
        Some(xgb_tree_importances(&self.trees, self.n_features))
    }
}

/// Shared helper: compute split-count importances from XGBNode trees
fn xgb_tree_importances(trees: &[XGBNode], n_features: usize) -> Array1<f64> {
    let mut counts = vec![0.0f64; n_features];
    for tree in trees {
        xgb_count_splits(tree, &mut counts);
    }
    let total: f64 = counts.iter().sum();
    if total > 0.0 {
        for c in counts.iter_mut() { *c /= total; }
    }
    Array1::from_vec(counts)
}

fn xgb_count_splits(node: &XGBNode, counts: &mut [f64]) {
    match node {
        XGBNode::Leaf { .. } => {}
        XGBNode::Split { feature, left, right, .. } => {
            if *feature < counts.len() { counts[*feature] += 1.0; }
            xgb_count_splits(left, counts);
            xgb_count_splits(right, counts);
        }
    }
}

// ─── XGBoost Classifier ────────────────────────────────────────────────────

/// XGBoost Classifier (logistic loss with second-order approximation)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct XGBoostClassifier {
    config: XGBoostConfig,
    trees: Vec<XGBNode>,
    base_score: f64,
    n_features: usize,
}

impl XGBoostClassifier {
    pub fn new(config: XGBoostConfig) -> Self {
        Self {
            config,
            trees: Vec::new(),
            base_score: 0.0,
            n_features: 0,
        }
    }

    fn sigmoid(x: f64) -> f64 {
        1.0 / (1.0 + (-x).exp())
    }

    pub fn fit(&mut self, x: &Array2<f64>, y: &Array1<f64>) -> Result<()> {
        let n_samples = x.nrows();
        let n_features = x.ncols();
        self.n_features = n_features;

        // Base score in log-odds space
        let p = y.mean().unwrap_or(0.5).clamp(1e-7, 1.0 - 1e-7);
        self.base_score = (p / (1.0 - p)).ln();
        let mut raw_preds = Array1::from_elem(n_samples, self.base_score);

        let mut rng = match self.config.random_state {
            Some(seed) => Xoshiro256PlusPlus::seed_from_u64(seed),
            None => Xoshiro256PlusPlus::from_entropy(),
        };

        self.trees.clear();

        for _ in 0..self.config.n_estimators {
            // Logistic loss: grad = p - y, hess = p * (1 - p)
            let probs: Array1<f64> = raw_preds.mapv(Self::sigmoid);
            let grad: Array1<f64> = &probs - y;
            let hess: Array1<f64> = probs.mapv(|p| (p * (1.0 - p)).max(1e-7));

            let row_indices = subsample(&mut rng, n_samples, self.config.subsample);
            let col_indices = subsample(&mut rng, n_features, self.config.colsample_bytree);

            let tree = build_xgb_tree(x, &grad, &hess, &row_indices, &col_indices, 0, &self.config);

            // Update log-odds for ALL rows, not just the sampled ones — see the matching
            // comment in XGBoostRegressor::fit for why (keeps the running cache consistent
            // with a fresh `predict_proba()` regardless of `subsample`).
            for i in 0..n_samples {
                let row = x.row(i);
                raw_preds[i] += self.config.learning_rate * tree.predict(row.as_slice().unwrap());
            }

            self.trees.push(tree);
        }

        Ok(())
    }

    pub fn predict(&self, x: &Array2<f64>) -> Result<Array1<f64>> {
        let probs = self.predict_proba(x)?;
        Ok(probs.mapv(|p| if p >= 0.5 { 1.0 } else { 0.0 }))
    }

    pub fn predict_proba(&self, x: &Array2<f64>) -> Result<Array1<f64>> {
        let n = x.nrows();
        let mut raw = Array1::from_elem(n, self.base_score);
        for i in 0..n {
            let s = x.row(i);
            let sample = s.as_slice().unwrap();
            for tree in &self.trees {
                raw[i] += self.config.learning_rate * tree.predict(sample);
            }
        }
        Ok(raw.mapv(Self::sigmoid))
    }

    pub fn score(&self, x: &Array2<f64>, y: &Array1<f64>) -> Result<f64> {
        let preds = self.predict(x)?;
        let correct = preds.iter().zip(y.iter())
            .filter(|(p, a)| (*p - *a).abs() < 0.5)
            .count();
        Ok(correct as f64 / y.len() as f64)
    }

    /// Compute feature importances by counting splits across all trees
    pub fn feature_importances(&self) -> Option<Array1<f64>> {
        if self.n_features == 0 { return None; }
        Some(xgb_tree_importances(&self.trees, self.n_features))
    }
}

// ─── Helpers ────────────────────────────────────────────────────────────────

fn subsample(rng: &mut Xoshiro256PlusPlus, n: usize, ratio: f64) -> Vec<usize> {
    if ratio >= 1.0 {
        return (0..n).collect();
    }
    let k = ((n as f64) * ratio).ceil() as usize;
    let mut indices: Vec<usize> = (0..n).collect();
    indices.shuffle(rng);
    indices.truncate(k);
    indices.sort();
    indices
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;

    fn regression_data() -> (Array2<f64>, Array1<f64>) {
        let x = Array2::from_shape_vec((50, 2),
            (0..100).map(|i| i as f64 * 0.1).collect()
        ).unwrap();
        let y: Array1<f64> = x.rows().into_iter()
            .map(|r| r[0] * 2.0 + r[1] * 0.5 + 1.0)
            .collect();
        (x, y)
    }

    fn classification_data() -> (Array2<f64>, Array1<f64>) {
        let x = Array2::from_shape_vec((50, 2),
            (0..100).map(|i| i as f64 * 0.1).collect()
        ).unwrap();
        let y: Array1<f64> = x.rows().into_iter()
            .map(|r| if r[0] + r[1] > 5.0 { 1.0 } else { 0.0 })
            .collect();
        (x, y)
    }

    #[test]
    fn test_xgboost_regressor() {
        let (x, y) = regression_data();
        let mut model = XGBoostRegressor::new(XGBoostConfig {
            n_estimators: 50,
            max_depth: 4,
            ..Default::default()
        });
        model.fit(&x, &y).unwrap();
        let r2 = model.score(&x, &y).unwrap();
        assert!(r2 > 0.9, "XGBoost regressor R² = {}", r2);
    }

    #[test]
    fn test_xgboost_classifier() {
        let (x, y) = classification_data();
        let mut model = XGBoostClassifier::new(XGBoostConfig {
            n_estimators: 50,
            max_depth: 4,
            ..Default::default()
        });
        model.fit(&x, &y).unwrap();
        let acc = model.score(&x, &y).unwrap();
        assert!(acc >= 0.8, "XGBoost classifier accuracy = {}", acc);
    }

    #[test]
    fn test_xgboost_predict_proba() {
        let (x, y) = classification_data();
        let mut model = XGBoostClassifier::new(Default::default());
        model.fit(&x, &y).unwrap();
        let proba = model.predict_proba(&x).unwrap();
        assert_eq!(proba.len(), x.nrows());
        assert!(proba.iter().all(|&p| p >= 0.0 && p <= 1.0));
    }

    #[test]
    fn test_xgboost_regularization() {
        let (x, y) = regression_data();
        let mut model = XGBoostRegressor::new(XGBoostConfig {
            n_estimators: 30,
            reg_lambda: 10.0,
            reg_alpha: 1.0,
            gamma: 1.0,
            ..Default::default()
        });
        model.fit(&x, &y).unwrap();
        let preds = model.predict(&x).unwrap();
        assert_eq!(preds.len(), 50);
    }
}
