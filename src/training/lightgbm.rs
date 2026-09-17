//! LightGBM-style gradient boosting with leaf-wise tree growth
//!
//! Key differences from standard/XGBoost-style gradient boosting:
//! - Leaf-wise (best-first) tree growth instead of level-wise
//! - Gradient-based One-Side Sampling (GOSS): keeps top gradients, samples low gradients
//! - Typically faster training with better accuracy on large datasets

use crate::error::{AutoMLError, Result};
use ndarray::{Array1, Array2};
use rand::prelude::*;
use rand_xoshiro::Xoshiro256PlusPlus;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LightGBMConfig {
    pub n_estimators: usize,
    pub learning_rate: f64,
    pub max_leaves: usize,
    pub max_depth: Option<usize>,
    pub min_child_samples: usize,
    pub reg_lambda: f64,
    pub reg_alpha: f64,
    pub subsample: f64,
    pub colsample_bytree: f64,
    pub top_rate: f64,
    pub other_rate: f64,
    pub random_state: Option<u64>,
}

impl Default for LightGBMConfig {
    fn default() -> Self {
        Self {
            n_estimators: 100,
            learning_rate: 0.1,
            max_leaves: 31,
            max_depth: None,
            min_child_samples: 20,
            reg_lambda: 0.0,
            reg_alpha: 0.0,
            subsample: 1.0,
            colsample_bytree: 1.0,
            top_rate: 0.2,
            other_rate: 0.1,
            random_state: Some(42),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
enum LGBNode {
    Leaf { value: f64 },
    Split {
        feature: usize,
        threshold: f64,
        left: Box<LGBNode>,
        right: Box<LGBNode>,
    },
}

impl LGBNode {
    fn predict(&self, sample: &[f64]) -> f64 {
        match self {
            LGBNode::Leaf { value } => *value,
            LGBNode::Split { feature, threshold, left, right, .. } => {
                if sample[*feature] <= *threshold {
                    left.predict(sample)
                } else {
                    right.predict(sample)
                }
            }
        }
    }
}

// ---- Tree building utilities ----

fn compute_leaf_weight(g: f64, h: f64, lambda: f64, alpha: f64) -> f64 {
    let g_adj = if g.abs() <= alpha { 0.0 } else { g - alpha * g.signum() };
    -g_adj / (h + lambda)
}

fn compute_gain_single(g: f64, h: f64, lambda: f64) -> f64 {
    g * g / (h + lambda)
}

fn make_leaf(gradients: &[f64], hessians: &[f64], indices: &[usize], lambda: f64, alpha: f64) -> LGBNode {
    let g: f64 = indices.iter().map(|&i| gradients[i]).sum();
    let h: f64 = indices.iter().map(|&i| hessians[i]).sum();
    LGBNode::Leaf { value: compute_leaf_weight(g, h, lambda, alpha) }
}

#[allow(dead_code)]
struct SplitResult {
    feature: usize,
    threshold: f64,
    gain: f64,
    left_indices: Vec<usize>,
    right_indices: Vec<usize>,
}

fn find_best_split_for_feature(
    x: &Array2<f64>, gradients: &[f64], hessians: &[f64],
    indices: &[usize], feature: usize, reg_lambda: f64, min_child_samples: usize,
) -> Option<(f64, f64, Vec<usize>, Vec<usize>)> {
    let mut sorted: Vec<(usize, f64)> = indices.iter().map(|&i| (i, x[[i, feature]])).collect();
    sorted.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

    let total_g: f64 = indices.iter().map(|&i| gradients[i]).sum();
    let total_h: f64 = indices.iter().map(|&i| hessians[i]).sum();
    let base_score = compute_gain_single(total_g, total_h, reg_lambda);

    let mut left_g = 0.0;
    let mut left_h = 0.0;
    let mut best_gain = f64::NEG_INFINITY;
    let mut best_threshold = 0.0;
    let mut best_pos = 0;

    for i in 0..sorted.len() - 1 {
        left_g += gradients[sorted[i].0];
        left_h += hessians[sorted[i].0];
        let right_g = total_g - left_g;
        let right_h = total_h - left_h;

        if i + 1 < min_child_samples || sorted.len() - i - 1 < min_child_samples {
            continue;
        }
        if sorted[i].1 == sorted[i + 1].1 { continue; }

        let gain = compute_gain_single(left_g, left_h, reg_lambda)
            + compute_gain_single(right_g, right_h, reg_lambda)
            - base_score;

        if gain > best_gain {
            best_gain = gain;
            best_threshold = (sorted[i].1 + sorted[i + 1].1) / 2.0;
            best_pos = i + 1;
        }
    }

    if best_gain <= 0.0 { return None; }

    let left_indices: Vec<usize> = sorted[..best_pos].iter().map(|&(i, _)| i).collect();
    let right_indices: Vec<usize> = sorted[best_pos..].iter().map(|&(i, _)| i).collect();
    Some((best_threshold, best_gain, left_indices, right_indices))
}

/// Build tree using leaf-wise (best-first) strategy
fn build_lgb_tree(
    x: &Array2<f64>, gradients: &[f64], hessians: &[f64],
    indices: &[usize], config: &LightGBMConfig, rng: &mut Xoshiro256PlusPlus,
) -> LGBNode {
    if indices.len() < config.min_child_samples * 2 {
        return make_leaf(gradients, hessians, indices, config.reg_lambda, config.reg_alpha);
    }

    let n_features = x.ncols();
    let n_selected = ((n_features as f64 * config.colsample_bytree).ceil() as usize).max(1);
    let mut feature_indices: Vec<usize> = (0..n_features).collect();
    feature_indices.shuffle(rng);
    feature_indices.truncate(n_selected);

    // Priority queue for leaf-wise growth
    use std::collections::BinaryHeap;
    use std::cmp::Ordering;

    #[derive(Clone)]
    struct PendingSplit {
        gain: f64,
        node_id: usize,
        feature: usize,
        threshold: f64,
        left_indices: Vec<usize>,
        right_indices: Vec<usize>,
    }
    impl PartialEq for PendingSplit { fn eq(&self, other: &Self) -> bool { self.gain == other.gain } }
    impl Eq for PendingSplit {}
    impl PartialOrd for PendingSplit {
        fn partial_cmp(&self, other: &Self) -> Option<Ordering> { self.gain.partial_cmp(&other.gain) }
    }
    impl Ord for PendingSplit {
        fn cmp(&self, other: &Self) -> Ordering { self.partial_cmp(other).unwrap_or(Ordering::Equal) }
    }

    enum NodeSlot {
        Leaf(Vec<usize>),
        Split { feature: usize, threshold: f64, left: usize, right: usize },
    }

    let mut nodes: Vec<NodeSlot> = vec![NodeSlot::Leaf(indices.to_vec())];
    let mut depths: Vec<usize> = vec![0];
    let mut heap: BinaryHeap<PendingSplit> = BinaryHeap::new();
    let max_depth_limit = config.max_depth.unwrap_or(usize::MAX);

    // Find initial split for root
    let root_splits: Vec<_> = feature_indices.par_iter().filter_map(|&feat| {
        find_best_split_for_feature(x, gradients, hessians, indices, feat, config.reg_lambda, config.min_child_samples)
            .map(|(thr, gain, li, ri)| (feat, thr, gain, li, ri))
    }).collect();

    if let Some(best) = root_splits.into_iter().max_by(|a, b| a.2.partial_cmp(&b.2).unwrap_or(Ordering::Equal)) {
        heap.push(PendingSplit {
            gain: best.2, node_id: 0, feature: best.0, threshold: best.1,
            left_indices: best.3, right_indices: best.4,
        });
    }

    let mut n_leaves = 1usize;

    while n_leaves < config.max_leaves {
        let split = match heap.pop() {
            Some(s) if s.gain > 0.0 => s,
            _ => break,
        };
        if depths[split.node_id] >= max_depth_limit { continue; }

        let depth = depths[split.node_id];
        let left_id = nodes.len();
        let right_id = nodes.len() + 1;

        nodes.push(NodeSlot::Leaf(split.left_indices.clone()));
        nodes.push(NodeSlot::Leaf(split.right_indices.clone()));
        depths.push(depth + 1);
        depths.push(depth + 1);

        nodes[split.node_id] = NodeSlot::Split {
            feature: split.feature, threshold: split.threshold, left: left_id, right: right_id,
        };
        n_leaves += 1;

        // Try splitting children
        if depth + 1 < max_depth_limit {
            for (child_id, child_indices) in [(left_id, &split.left_indices), (right_id, &split.right_indices)] {
                if child_indices.len() < config.min_child_samples * 2 { continue; }
                let child_splits: Vec<_> = feature_indices.par_iter().filter_map(|&feat| {
                    find_best_split_for_feature(x, gradients, hessians, child_indices, feat, config.reg_lambda, config.min_child_samples)
                        .map(|(thr, gain, li, ri)| (feat, thr, gain, li, ri))
                }).collect();
                if let Some(best) = child_splits.into_iter().max_by(|a, b| a.2.partial_cmp(&b.2).unwrap_or(Ordering::Equal)) {
                    heap.push(PendingSplit {
                        gain: best.2, node_id: child_id, feature: best.0, threshold: best.1,
                        left_indices: best.3, right_indices: best.4,
                    });
                }
            }
        }
    }

    // Convert to LGBNode tree
    fn to_node(nodes: &[NodeSlot], idx: usize, g: &[f64], h: &[f64], lam: f64, alpha: f64) -> LGBNode {
        match &nodes[idx] {
            NodeSlot::Leaf(indices) => make_leaf(g, h, indices, lam, alpha),
            NodeSlot::Split { feature, threshold, left, right } => LGBNode::Split {
                feature: *feature, threshold: *threshold,
                left: Box::new(to_node(nodes, *left, g, h, lam, alpha)),
                right: Box::new(to_node(nodes, *right, g, h, lam, alpha)),
            },
        }
    }
    to_node(&nodes, 0, gradients, hessians, config.reg_lambda, config.reg_alpha)
}

/// GOSS (Gradient-based One-Side Sampling): keeps the top `top_rate` fraction of instances by
/// |gradient| as-is, and randomly samples an `other_rate` fraction of the remaining
/// (small-gradient) instances. Returns `(selected_indices, sampled_other_indices)` — the
/// second element (a subset of the first) is the randomly-sampled small-gradient rows, which
/// the caller must reweight by `(1 - top_rate) / other_rate` (see the `fit` methods below) to
/// keep the gradient-sum estimate that drives split-finding unbiased. Without that
/// compensation, GOSS silently biases every split toward under-counting the (much larger)
/// population of small-gradient rows that got mostly dropped.
fn goss_sample(gradients: &[f64], n: usize, top_rate: f64, other_rate: f64, rng: &mut Xoshiro256PlusPlus) -> (Vec<usize>, Vec<usize>) {
    let n_top = (n as f64 * top_rate).ceil() as usize;
    let n_other = (n as f64 * other_rate).ceil() as usize;
    let mut indices: Vec<usize> = (0..n).collect();
    if n_top < n {
        // Partial sort: partition so top n_top by absolute gradient are at the front
        indices.select_nth_unstable_by(n_top, |&a, &b| {
            gradients[b].abs().partial_cmp(&gradients[a].abs()).unwrap_or(std::cmp::Ordering::Equal)
        });
    }
    let mut selected: Vec<usize> = indices[..n_top.min(n)].to_vec();
    let mut remaining: Vec<usize> = indices[n_top.min(n)..].to_vec();
    remaining.shuffle(rng);
    let sampled_other: Vec<usize> = remaining.drain(..n_other.min(remaining.len())).collect();
    selected.extend(sampled_other.iter().copied());
    (selected, sampled_other)
}

fn sigmoid(x: f64) -> f64 { 1.0 / (1.0 + (-x).exp()) }

// ============ LightGBM Regressor ============

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LightGBMRegressor {
    pub config: LightGBMConfig,
    trees: Vec<LGBNode>,
    base_prediction: f64,
}

impl LightGBMRegressor {
    pub fn new(config: LightGBMConfig) -> Self {
        Self { config, trees: Vec::new(), base_prediction: 0.0 }
    }

    pub fn fit(&mut self, x: &Array2<f64>, y: &Array1<f64>) -> Result<()> {
        let n = x.nrows();
        if n == 0 { return Err(AutoMLError::TrainingError("Empty dataset".into())); }

        let mut rng = Xoshiro256PlusPlus::seed_from_u64(self.config.random_state.unwrap_or(42));
        self.base_prediction = y.mean().unwrap_or(0.0);
        let mut predictions = Array1::from_elem(n, self.base_prediction);

        for _ in 0..self.config.n_estimators {
            let mut gradients: Vec<f64> = predictions.iter().zip(y.iter()).map(|(&p, &yi)| p - yi).collect();
            let mut hessians: Vec<f64> = vec![1.0; n];

            let indices = if self.config.top_rate + self.config.other_rate < 1.0 {
                let (idx, sampled_other) = goss_sample(&gradients, n, self.config.top_rate, self.config.other_rate, &mut rng);
                // Compensate the randomly-sampled small-gradient rows so the gradient/hessian
                // sums split-finding relies on stay an unbiased estimate of the full dataset's.
                if !sampled_other.is_empty() && self.config.other_rate > 0.0 {
                    let factor = (1.0 - self.config.top_rate) / self.config.other_rate;
                    for &i in &sampled_other {
                        gradients[i] *= factor;
                        hessians[i] *= factor;
                    }
                }
                idx
            } else if self.config.subsample < 1.0 {
                let k = (n as f64 * self.config.subsample).ceil() as usize;
                let mut idx: Vec<usize> = (0..n).collect();
                idx.shuffle(&mut rng);
                idx.truncate(k);
                idx
            } else {
                (0..n).collect()
            };

            let tree = build_lgb_tree(x, &gradients, &hessians, &indices, &self.config, &mut rng);
            for i in 0..n {
                predictions[i] += self.config.learning_rate * tree.predict(x.row(i).as_slice().unwrap());
            }
            self.trees.push(tree);
        }
        Ok(())
    }

    pub fn predict(&self, x: &Array2<f64>) -> Result<Array1<f64>> {
        Ok(Array1::from_vec(x.rows().into_iter().map(|row| {
            let s = row.as_slice().unwrap();
            self.base_prediction + self.trees.iter().map(|t| self.config.learning_rate * t.predict(s)).sum::<f64>()
        }).collect()))
    }

    /// Compute feature importances by counting splits across all trees
    pub fn feature_importances(&self, n_features: usize) -> Option<Array1<f64>> {
        if n_features == 0 { return None; }
        Some(lgb_tree_importances(&self.trees, n_features))
    }
}

/// Shared helper: compute split-count importances from LGBNode trees
fn lgb_tree_importances(trees: &[LGBNode], n_features: usize) -> Array1<f64> {
    let mut counts = vec![0.0f64; n_features];
    for tree in trees {
        lgb_count_splits(tree, &mut counts);
    }
    let total: f64 = counts.iter().sum();
    if total > 0.0 {
        for c in counts.iter_mut() { *c /= total; }
    }
    Array1::from_vec(counts)
}

fn lgb_count_splits(node: &LGBNode, counts: &mut [f64]) {
    match node {
        LGBNode::Leaf { .. } => {}
        LGBNode::Split { feature, left, right, .. } => {
            if *feature < counts.len() { counts[*feature] += 1.0; }
            lgb_count_splits(left, counts);
            lgb_count_splits(right, counts);
        }
    }
}

// ============ LightGBM Classifier ============

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LightGBMClassifier {
    pub config: LightGBMConfig,
    trees: Vec<LGBNode>,
    base_prediction: f64,
}

impl LightGBMClassifier {
    pub fn new(config: LightGBMConfig) -> Self {
        Self { config, trees: Vec::new(), base_prediction: 0.0 }
    }

    pub fn fit(&mut self, x: &Array2<f64>, y: &Array1<f64>) -> Result<()> {
        let n = x.nrows();
        if n == 0 { return Err(AutoMLError::TrainingError("Empty dataset".into())); }

        let mut rng = Xoshiro256PlusPlus::seed_from_u64(self.config.random_state.unwrap_or(42));
        let pos = y.iter().filter(|&&v| v > 0.5).count() as f64;
        let neg = n as f64 - pos;
        self.base_prediction = (pos / neg.max(1e-10)).ln();
        let mut raw = Array1::from_elem(n, self.base_prediction);

        for _ in 0..self.config.n_estimators {
            let probs: Vec<f64> = raw.iter().map(|&r| sigmoid(r)).collect();
            let mut gradients: Vec<f64> = probs.iter().zip(y.iter()).map(|(&p, &yi)| p - yi).collect();
            let mut hessians: Vec<f64> = probs.iter().map(|&p| (p * (1.0 - p)).max(1e-16)).collect();

            let indices = if self.config.top_rate + self.config.other_rate < 1.0 {
                let (idx, sampled_other) = goss_sample(&gradients, n, self.config.top_rate, self.config.other_rate, &mut rng);
                // See the matching comment in LightGBMRegressor::fit.
                if !sampled_other.is_empty() && self.config.other_rate > 0.0 {
                    let factor = (1.0 - self.config.top_rate) / self.config.other_rate;
                    for &i in &sampled_other {
                        gradients[i] *= factor;
                        hessians[i] *= factor;
                    }
                }
                idx
            } else if self.config.subsample < 1.0 {
                let k = (n as f64 * self.config.subsample).ceil() as usize;
                let mut idx: Vec<usize> = (0..n).collect();
                idx.shuffle(&mut rng);
                idx.truncate(k);
                idx
            } else {
                (0..n).collect()
            };

            let tree = build_lgb_tree(x, &gradients, &hessians, &indices, &self.config, &mut rng);
            for i in 0..n {
                raw[i] += self.config.learning_rate * tree.predict(x.row(i).as_slice().unwrap());
            }
            self.trees.push(tree);
        }
        Ok(())
    }

    pub fn predict(&self, x: &Array2<f64>) -> Result<Array1<f64>> {
        let raw = self.predict_raw(x)?;
        Ok(raw.mapv(|r| if sigmoid(r) >= 0.5 { 1.0 } else { 0.0 }))
    }

    pub fn predict_proba(&self, x: &Array2<f64>) -> Result<Array2<f64>> {
        let raw = self.predict_raw(x)?;
        let n = x.nrows();
        let mut proba = Array2::zeros((n, 2));
        for i in 0..n {
            let p = sigmoid(raw[i]);
            proba[[i, 0]] = 1.0 - p;
            proba[[i, 1]] = p;
        }
        Ok(proba)
    }

    fn predict_raw(&self, x: &Array2<f64>) -> Result<Array1<f64>> {
        Ok(Array1::from_vec(x.rows().into_iter().map(|row| {
            let s = row.as_slice().unwrap();
            self.base_prediction + self.trees.iter().map(|t| self.config.learning_rate * t.predict(s)).sum::<f64>()
        }).collect()))
    }

    /// Compute feature importances by counting splits across all trees
    pub fn feature_importances(&self, n_features: usize) -> Option<Array1<f64>> {
        if n_features == 0 { return None; }
        Some(lgb_tree_importances(&self.trees, n_features))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_regression_data() -> (Array2<f64>, Array1<f64>) {
        let x = Array2::from_shape_vec((100, 3), (0..300).map(|i| (i as f64) / 100.0).collect()).unwrap();
        let y = Array1::from_vec((0..100).map(|i| {
            let x0 = (i * 3) as f64 / 100.0;
            2.0 * x0 + 0.1
        }).collect());
        (x, y)
    }

    fn make_classification_data() -> (Array2<f64>, Array1<f64>) {
        let x = Array2::from_shape_vec((100, 2), (0..200).map(|i| (i as f64) / 100.0).collect()).unwrap();
        let y = Array1::from_vec((0..100).map(|i| if i < 50 { 0.0 } else { 1.0 }).collect());
        (x, y)
    }

    #[test]
    fn test_lightgbm_regressor() {
        let (x, y) = make_regression_data();
        let config = LightGBMConfig { n_estimators: 20, max_leaves: 8, min_child_samples: 2, ..Default::default() };
        let mut model = LightGBMRegressor::new(config);
        model.fit(&x, &y).unwrap();
        let preds = model.predict(&x).unwrap();
        assert_eq!(preds.len(), 100);
    }

    #[test]
    fn test_lightgbm_classifier() {
        let (x, y) = make_classification_data();
        let config = LightGBMConfig { n_estimators: 30, max_leaves: 8, min_child_samples: 2, ..Default::default() };
        let mut model = LightGBMClassifier::new(config);
        model.fit(&x, &y).unwrap();
        let preds = model.predict(&x).unwrap();
        let acc = preds.iter().zip(y.iter()).filter(|(&p, &t)| p == t).count() as f64 / 100.0;
        assert!(acc > 0.7, "Accuracy too low: {}", acc);
    }

    #[test]
    fn test_lightgbm_predict_proba() {
        let (x, y) = make_classification_data();
        let config = LightGBMConfig { n_estimators: 10, max_leaves: 8, min_child_samples: 2, ..Default::default() };
        let mut model = LightGBMClassifier::new(config);
        model.fit(&x, &y).unwrap();
        let proba = model.predict_proba(&x).unwrap();
        assert_eq!(proba.ncols(), 2);
        for i in 0..proba.nrows() {
            assert!((proba[[i, 0]] + proba[[i, 1]] - 1.0).abs() < 1e-10);
        }
    }

    #[test]
    fn test_lightgbm_goss() {
        let (x, y) = make_regression_data();
        let config = LightGBMConfig {
            n_estimators: 10, max_leaves: 8, min_child_samples: 2,
            top_rate: 0.3, other_rate: 0.2, ..Default::default()
        };
        let mut model = LightGBMRegressor::new(config);
        model.fit(&x, &y).unwrap();
        assert_eq!(model.predict(&x).unwrap().len(), 100);
    }
}
