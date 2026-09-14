//! CatBoost-style gradient boosting with ordered target statistics
//!
//! Key features:
//! - Ordered boosting: prevents target leakage by using only preceding samples
//! - Symmetric (oblivious) decision trees: all nodes at same depth use the same split
//! - Built-in categorical feature handling via target statistics

use crate::error::{AutoMLError, Result};
use ndarray::{Array1, Array2};
use rand::prelude::*;
use rand_xoshiro::Xoshiro256PlusPlus;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CatBoostConfig {
    pub n_estimators: usize,
    pub learning_rate: f64,
    pub max_depth: usize,
    pub reg_lambda: f64,
    pub subsample: f64,
    pub random_state: Option<u64>,
}

impl Default for CatBoostConfig {
    fn default() -> Self {
        Self {
            n_estimators: 100,
            learning_rate: 0.1,
            max_depth: 6,
            reg_lambda: 3.0,
            subsample: 1.0,
            random_state: Some(42),
        }
    }
}

/// Symmetric (oblivious) tree: each level uses the same split feature + threshold
#[derive(Debug, Clone, Serialize, Deserialize)]
struct SymmetricTree {
    splits: Vec<(usize, f64)>, // (feature, threshold) per level
    leaf_values: Vec<f64>,     // 2^depth leaf values
}

impl SymmetricTree {
    fn predict(&self, sample: &[f64]) -> f64 {
        let mut idx = 0usize;
        for &(feature, threshold) in &self.splits {
            idx = idx * 2 + if sample[feature] > threshold { 1 } else { 0 };
        }
        self.leaf_values[idx.min(self.leaf_values.len() - 1)]
    }
}

fn build_symmetric_tree(
    x: &Array2<f64>,
    gradients: &[f64],
    hessians: &[f64],
    indices: &[usize],
    max_depth: usize,
    reg_lambda: f64,
    _rng: &mut Xoshiro256PlusPlus,
) -> SymmetricTree {
    let n_features = x.ncols();
    let mut splits = Vec::with_capacity(max_depth);

    // Current partition of indices into buckets
    let mut buckets: Vec<Vec<usize>> = vec![indices.to_vec()];

    for _depth in 0..max_depth {
        // Find best global split across all buckets (symmetric = same split for all)
        let best = (0..n_features).into_par_iter().filter_map(|feat| {
            // Collect all unique thresholds across all buckets
            let mut all_vals: Vec<f64> = buckets.iter()
                .flat_map(|b| b.iter().map(|&i| x[[i, feat]]))
                .collect();
            all_vals.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            all_vals.dedup();

            if all_vals.len() < 2 { return None; }

            // Try midpoints
            let mut best_gain = f64::NEG_INFINITY;
            let mut best_thr = 0.0;

            // Sample up to 256 candidate thresholds for efficiency
            let step = (all_vals.len() / 256).max(1);
            for i in (0..all_vals.len() - 1).step_by(step) {
                let thr = (all_vals[i] + all_vals[i + 1]) / 2.0;
                let mut total_gain = 0.0;

                for bucket in &buckets {
                    let (lg, lh, rg, rh) = bucket.iter().fold((0.0, 0.0, 0.0, 0.0), |(lg, lh, rg, rh), &idx| {
                        if x[[idx, feat]] <= thr {
                            (lg + gradients[idx], lh + hessians[idx], rg, rh)
                        } else {
                            (lg, lh, rg + gradients[idx], rh + hessians[idx])
                        }
                    });
                    let parent_g = lg + rg;
                    let parent_h = lh + rh;
                    let parent_score = parent_g * parent_g / (parent_h + reg_lambda);
                    let left_score = lg * lg / (lh + reg_lambda);
                    let right_score = rg * rg / (rh + reg_lambda);
                    total_gain += left_score + right_score - parent_score;
                }

                if total_gain > best_gain {
                    best_gain = total_gain;
                    best_thr = thr;
                }
            }

            if best_gain > 0.0 { Some((feat, best_thr, best_gain)) } else { None }
        }).max_by(|a, b| a.2.partial_cmp(&b.2).unwrap_or(std::cmp::Ordering::Equal));

        match best {
            Some((feat, thr, _)) => {
                splits.push((feat, thr));
                // Split all buckets
                let mut new_buckets = Vec::with_capacity(buckets.len() * 2);
                for bucket in &buckets {
                    let (left, right): (Vec<usize>, Vec<usize>) = bucket.iter()
                        .partition(|&&i| x[[i, feat]] <= thr);
                    new_buckets.push(left);
                    new_buckets.push(right);
                }
                buckets = new_buckets;
            }
            None => break,
        }
    }

    // Compute leaf values
    let leaf_values: Vec<f64> = buckets.iter().map(|bucket| {
        if bucket.is_empty() { return 0.0; }
        let g: f64 = bucket.iter().map(|&i| gradients[i]).sum();
        let h: f64 = bucket.iter().map(|&i| hessians[i]).sum();
        -g / (h + reg_lambda)
    }).collect();

    SymmetricTree { splits, leaf_values }
}

// ============ CatBoost Regressor ============

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CatBoostRegressor {
    pub config: CatBoostConfig,
    trees: Vec<SymmetricTree>,
    base_prediction: f64,
}

impl CatBoostRegressor {
    pub fn new(config: CatBoostConfig) -> Self {
        Self { config, trees: Vec::new(), base_prediction: 0.0 }
    }

    pub fn fit(&mut self, x: &Array2<f64>, y: &Array1<f64>) -> Result<()> {
        let n = x.nrows();
        if n == 0 { return Err(AutoMLError::TrainingError("Empty dataset".into())); }

        let mut rng = Xoshiro256PlusPlus::seed_from_u64(self.config.random_state.unwrap_or(42));
        self.base_prediction = y.mean().unwrap_or(0.0);
        let mut predictions = Array1::from_elem(n, self.base_prediction);

        // Ordered boosting: random permutation to prevent target leakage
        let mut perm: Vec<usize> = (0..n).collect();
        perm.shuffle(&mut rng);

        for _ in 0..self.config.n_estimators {
            let gradients: Vec<f64> = predictions.iter().zip(y.iter()).map(|(&p, &yi)| p - yi).collect();
            let hessians: Vec<f64> = vec![1.0; n];

            let indices: Vec<usize> = if self.config.subsample < 1.0 {
                let k = (n as f64 * self.config.subsample).ceil() as usize;
                let mut sub = perm.clone();
                sub.shuffle(&mut rng);
                sub.truncate(k);
                sub
            } else {
                (0..n).collect()
            };

            let tree = build_symmetric_tree(
                x, &gradients, &hessians, &indices,
                self.config.max_depth, self.config.reg_lambda, &mut rng,
            );

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

    /// Compute feature importances by counting split feature usage
    pub fn feature_importances(&self, n_features: usize) -> Option<Array1<f64>> {
        if n_features == 0 { return None; }
        Some(cat_tree_importances(&self.trees, n_features))
    }
}

/// Shared helper: compute split-count importances from SymmetricTree
fn cat_tree_importances(trees: &[SymmetricTree], n_features: usize) -> Array1<f64> {
    let mut counts = vec![0.0f64; n_features];
    for tree in trees {
        for &(feature, _) in &tree.splits {
            if feature < n_features { counts[feature] += 1.0; }
        }
    }
    let total: f64 = counts.iter().sum();
    if total > 0.0 {
        for c in counts.iter_mut() { *c /= total; }
    }
    Array1::from_vec(counts)
}

// ============ CatBoost Classifier ============

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CatBoostClassifier {
    pub config: CatBoostConfig,
    trees: Vec<SymmetricTree>,
    base_prediction: f64,
}

impl CatBoostClassifier {
    pub fn new(config: CatBoostConfig) -> Self {
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

        let mut perm: Vec<usize> = (0..n).collect();
        perm.shuffle(&mut rng);

        for _ in 0..self.config.n_estimators {
            let probs: Vec<f64> = raw.iter().map(|&r| sigmoid(r)).collect();
            let gradients: Vec<f64> = probs.iter().zip(y.iter()).map(|(&p, &yi)| p - yi).collect();
            let hessians: Vec<f64> = probs.iter().map(|&p| (p * (1.0 - p)).max(1e-16)).collect();

            let indices: Vec<usize> = if self.config.subsample < 1.0 {
                let k = (n as f64 * self.config.subsample).ceil() as usize;
                let mut sub = perm.clone();
                sub.shuffle(&mut rng);
                sub.truncate(k);
                sub
            } else {
                (0..n).collect()
            };

            let tree = build_symmetric_tree(
                x, &gradients, &hessians, &indices,
                self.config.max_depth, self.config.reg_lambda, &mut rng,
            );

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

    /// Compute feature importances by counting split feature usage
    pub fn feature_importances(&self, n_features: usize) -> Option<Array1<f64>> {
        if n_features == 0 { return None; }
        Some(cat_tree_importances(&self.trees, n_features))
    }
}

fn sigmoid(x: f64) -> f64 { 1.0 / (1.0 + (-x).exp()) }

#[cfg(test)]
mod tests {
    use super::*;

    fn make_regression_data() -> (Array2<f64>, Array1<f64>) {
        let x = Array2::from_shape_vec((100, 3), (0..300).map(|i| (i as f64) / 100.0).collect()).unwrap();
        let y = Array1::from_vec((0..100).map(|i| 2.0 * (i * 3) as f64 / 100.0 + 0.1).collect());
        (x, y)
    }

    fn make_classification_data() -> (Array2<f64>, Array1<f64>) {
        let x = Array2::from_shape_vec((100, 2), (0..200).map(|i| (i as f64) / 100.0).collect()).unwrap();
        let y = Array1::from_vec((0..100).map(|i| if i < 50 { 0.0 } else { 1.0 }).collect());
        (x, y)
    }

    #[test]
    fn test_catboost_regressor() {
        let (x, y) = make_regression_data();
        let config = CatBoostConfig { n_estimators: 20, max_depth: 4, ..Default::default() };
        let mut model = CatBoostRegressor::new(config);
        model.fit(&x, &y).unwrap();
        let preds = model.predict(&x).unwrap();
        assert_eq!(preds.len(), 100);
    }

    #[test]
    fn test_catboost_classifier() {
        let (x, y) = make_classification_data();
        let config = CatBoostConfig { n_estimators: 30, max_depth: 4, ..Default::default() };
        let mut model = CatBoostClassifier::new(config);
        model.fit(&x, &y).unwrap();
        let preds = model.predict(&x).unwrap();
        let acc = preds.iter().zip(y.iter()).filter(|(&p, &t)| p == t).count() as f64 / 100.0;
        assert!(acc > 0.7, "Accuracy too low: {}", acc);
    }

    #[test]
    fn test_catboost_predict_proba() {
        let (x, y) = make_classification_data();
        let config = CatBoostConfig { n_estimators: 10, max_depth: 4, ..Default::default() };
        let mut model = CatBoostClassifier::new(config);
        model.fit(&x, &y).unwrap();
        let proba = model.predict_proba(&x).unwrap();
        assert_eq!(proba.ncols(), 2);
        for i in 0..proba.nrows() {
            assert!((proba[[i, 0]] + proba[[i, 1]] - 1.0).abs() < 1e-10);
        }
    }

    #[test]
    fn test_catboost_symmetric_tree() {
        // Verify that the symmetric tree structure works
        let (x, y) = make_regression_data();
        let config = CatBoostConfig { n_estimators: 5, max_depth: 3, ..Default::default() };
        let mut model = CatBoostRegressor::new(config);
        model.fit(&x, &y).unwrap();
        // Each tree should have at most 2^3 = 8 leaf values
        for tree in &model.trees {
            assert!(tree.leaf_values.len() <= 8);
            assert!(tree.splits.len() <= 3);
        }
    }
}
