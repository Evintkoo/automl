//! Extra Trees (Extremely Randomized Trees) implementation
//!
//! Unlike Random Forest which searches for the best split among a random subset
//! of features, Extra Trees picks both the feature AND the threshold at random.
//! This further reduces variance at a small cost to bias, and is faster to train.

use crate::error::{AutoMLError, Result};
use ndarray::{Array1, Array2};
use rand::SeedableRng;
use rand::RngCore;
use rand_chacha::ChaCha8Rng;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// An extremely randomized tree node
#[derive(Debug, Clone, Serialize, Deserialize)]
enum ExtraTreeNode {
    Leaf { value: f64 },
    Split {
        feature: usize,
        threshold: f64,
        left: Box<ExtraTreeNode>,
        right: Box<ExtraTreeNode>,
    },
}

impl ExtraTreeNode {
    fn predict_sample(&self, sample: &[f64]) -> f64 {
        match self {
            ExtraTreeNode::Leaf { value } => *value,
            ExtraTreeNode::Split { feature, threshold, left, right } => {
                if sample[*feature] <= *threshold {
                    left.predict_sample(sample)
                } else {
                    right.predict_sample(sample)
                }
            }
        }
    }
}

/// Extra Trees model (Classifier + Regressor)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExtraTrees {
    trees: Vec<ExtraTreeNode>,
    pub n_estimators: usize,
    pub max_depth: Option<usize>,
    pub min_samples_split: usize,
    pub min_samples_leaf: usize,
    /// Number of features to consider per split
    pub max_features: Option<usize>,
    pub random_state: Option<u64>,
    is_classification: bool,
    n_features: usize,
    classes: Vec<f64>,
    pub is_fitted: bool,
}

impl Default for ExtraTrees {
    fn default() -> Self {
        Self::new_classifier(100)
    }
}

impl ExtraTrees {
    pub fn new_classifier(n_estimators: usize) -> Self {
        Self {
            trees: Vec::new(),
            n_estimators,
            max_depth: Some(20),
            min_samples_split: 2,
            min_samples_leaf: 1,
            max_features: None,
            random_state: None,
            is_classification: true,
            n_features: 0,
            classes: Vec::new(),
            is_fitted: false,
        }
    }

    pub fn new_regressor(n_estimators: usize) -> Self {
        Self {
            trees: Vec::new(),
            n_estimators,
            max_depth: Some(20),
            min_samples_split: 2,
            min_samples_leaf: 1,
            max_features: None,
            random_state: None,
            is_classification: false,
            n_features: 0,
            classes: Vec::new(),
            is_fitted: false,
        }
    }

    pub fn with_max_depth(mut self, depth: usize) -> Self {
        self.max_depth = Some(depth);
        self
    }

    pub fn with_random_state(mut self, seed: u64) -> Self {
        self.random_state = Some(seed);
        self
    }

    pub fn with_max_features(mut self, mf: usize) -> Self {
        self.max_features = Some(mf);
        self
    }

    fn compute_max_features(&self, n_features: usize) -> usize {
        match self.max_features {
            Some(mf) => mf.min(n_features).max(1),
            None => {
                if self.is_classification {
                    (n_features as f64).sqrt().ceil() as usize
                } else {
                    n_features // Regressors use all features by default (sklearn convention)
                }
            }
        }
    }

    /// Build a single extra tree with random splits
    fn build_tree(
        x: &Array2<f64>,
        y: &Array1<f64>,
        indices: &[usize],
        max_features: usize,
        max_depth: Option<usize>,
        min_samples_split: usize,
        min_samples_leaf: usize,
        is_classification: bool,
        depth: usize,
        rng: &mut ChaCha8Rng,
    ) -> ExtraTreeNode {
        let n = indices.len();

        // Leaf conditions
        if n < min_samples_split
            || n <= 1
            || max_depth.map_or(false, |d| depth >= d)
        {
            return ExtraTreeNode::Leaf {
                value: Self::leaf_value(y, indices, is_classification),
            };
        }

        // Check if all y values are the same
        let first_y = y[indices[0]];
        if indices.iter().all(|&i| (y[i] - first_y).abs() < 1e-15) {
            return ExtraTreeNode::Leaf { value: first_y };
        }

        let n_features = x.ncols();

        // Select random subset of features
        let feature_indices = Self::random_features(n_features, max_features, rng);

        // Find best random split across selected features
        let mut best_feature = feature_indices[0];
        let mut best_threshold = 0.0;
        let mut best_score = f64::MAX;
        let mut found_valid_split = false;

        for &f in &feature_indices {
            // Find min/max for this feature among current indices
            let mut fmin = f64::MAX;
            let mut fmax = f64::MIN;
            for &i in indices {
                let v = x[[i, f]];
                if v < fmin { fmin = v; }
                if v > fmax { fmax = v; }
            }

            if (fmax - fmin).abs() < 1e-15 {
                continue; // Feature is constant — skip
            }

            // Pick a random threshold uniformly between min and max
            let r = (rng.next_u64() as f64) / (u64::MAX as f64);
            let threshold = fmin + r * (fmax - fmin);

            // Evaluate split quality
            let (left_idx, right_idx): (Vec<usize>, Vec<usize>) =
                indices.iter().partition(|&&i| x[[i, f]] <= threshold);

            if left_idx.len() < min_samples_leaf || right_idx.len() < min_samples_leaf {
                continue;
            }

            let score = if is_classification {
                Self::gini_split(y, &left_idx, &right_idx)
            } else {
                Self::mse_split(y, &left_idx, &right_idx)
            };

            if score < best_score {
                best_score = score;
                best_feature = f;
                best_threshold = threshold;
                found_valid_split = true;
            }
        }

        if !found_valid_split {
            return ExtraTreeNode::Leaf {
                value: Self::leaf_value(y, indices, is_classification),
            };
        }

        let (left_idx, right_idx): (Vec<usize>, Vec<usize>) =
            indices.iter().partition(|&&i| x[[i, best_feature]] <= best_threshold);

        let left = Self::build_tree(x, y, &left_idx, max_features, max_depth, min_samples_split, min_samples_leaf, is_classification, depth + 1, rng);
        let right = Self::build_tree(x, y, &right_idx, max_features, max_depth, min_samples_split, min_samples_leaf, is_classification, depth + 1, rng);

        ExtraTreeNode::Split {
            feature: best_feature,
            threshold: best_threshold,
            left: Box::new(left),
            right: Box::new(right),
        }
    }

    fn random_features(n_features: usize, max_features: usize, rng: &mut ChaCha8Rng) -> Vec<usize> {
        if max_features >= n_features {
            return (0..n_features).collect();
        }
        let mut features: Vec<usize> = (0..n_features).collect();
        // Fisher-Yates partial shuffle
        for i in 0..max_features {
            let j = i + (rng.next_u64() as usize) % (n_features - i);
            features.swap(i, j);
        }
        features.truncate(max_features);
        features
    }

    fn leaf_value(y: &Array1<f64>, indices: &[usize], is_classification: bool) -> f64 {
        if is_classification {
            // Majority vote
            let mut counts: HashMap<i64, usize> = HashMap::new();
            for &i in indices {
                let key = y[i].to_bits() as i64;
                *counts.entry(key).or_insert(0) += 1;
            }
            counts
                .into_iter()
                .max_by_key(|&(_, c)| c)
                .map(|(k, _)| f64::from_bits(k as u64))
                .unwrap_or(0.0)
        } else {
            // Mean
            let sum: f64 = indices.iter().map(|&i| y[i]).sum();
            sum / indices.len().max(1) as f64
        }
    }

    fn gini_split(y: &Array1<f64>, left: &[usize], right: &[usize]) -> f64 {
        let n = (left.len() + right.len()) as f64;
        let lg = Self::gini_impurity(y, left);
        let rg = Self::gini_impurity(y, right);
        (left.len() as f64 * lg + right.len() as f64 * rg) / n
    }

    fn gini_impurity(y: &Array1<f64>, indices: &[usize]) -> f64 {
        let n = indices.len() as f64;
        if n == 0.0 { return 0.0; }
        let mut counts: HashMap<i64, usize> = HashMap::new();
        for &i in indices {
            let key = y[i].to_bits() as i64;
            *counts.entry(key).or_insert(0) += 1;
        }
        1.0 - counts.values().map(|&c| (c as f64 / n).powi(2)).sum::<f64>()
    }

    fn mse_split(y: &Array1<f64>, left: &[usize], right: &[usize]) -> f64 {
        let n = (left.len() + right.len()) as f64;
        let lm = Self::mse_impurity(y, left);
        let rm = Self::mse_impurity(y, right);
        (left.len() as f64 * lm + right.len() as f64 * rm) / n
    }

    fn mse_impurity(y: &Array1<f64>, indices: &[usize]) -> f64 {
        let n = indices.len() as f64;
        if n == 0.0 { return 0.0; }
        let mean: f64 = indices.iter().map(|&i| y[i]).sum::<f64>() / n;
        indices.iter().map(|&i| (y[i] - mean).powi(2)).sum::<f64>() / n
    }

    pub fn fit(&mut self, x: &Array2<f64>, y: &Array1<f64>) -> Result<&mut Self> {
        let n_samples = x.nrows();
        let n_features = x.ncols();
        if n_samples != y.len() {
            return Err(AutoMLError::ShapeError {
                expected: format!("y length = {}", n_samples),
                actual: format!("y length = {}", y.len()),
            });
        }

        self.n_features = n_features;
        let max_features = self.compute_max_features(n_features);
        let all_indices: Vec<usize> = (0..n_samples).collect();

        if self.is_classification {
            let mut classes: Vec<f64> = y.to_vec();
            classes.sort_by(|a, b| a.partial_cmp(b).unwrap());
            classes.dedup();
            self.classes = classes;
        }

        let base_seed = self.random_state.unwrap_or(42);
        let max_depth = self.max_depth;
        let min_split = self.min_samples_split;
        let min_leaf = self.min_samples_leaf;
        let is_cls = self.is_classification;

        // Extra Trees does NOT bootstrap — each tree uses the full dataset
        let trees: Vec<ExtraTreeNode> = (0..self.n_estimators)
            .into_par_iter()
            .map(|tree_idx| {
                let mut rng = ChaCha8Rng::seed_from_u64(base_seed.wrapping_add(tree_idx as u64));
                Self::build_tree(
                    x, y, &all_indices, max_features, max_depth,
                    min_split, min_leaf, is_cls, 0, &mut rng,
                )
            })
            .collect();

        self.trees = trees;
        self.is_fitted = true;
        Ok(self)
    }

    pub fn predict(&self, x: &Array2<f64>) -> Result<Array1<f64>> {
        if !self.is_fitted {
            return Err(AutoMLError::ModelNotFitted);
        }

        let n_samples = x.nrows();
        let predictions: Vec<f64> = (0..n_samples)
            .into_par_iter()
            .map(|i| {
                let sample = x.row(i);
                let s = sample.as_slice().unwrap();

                if self.is_classification {
                    // Vec-indexed majority vote
                    let mut votes = vec![0usize; self.classes.len().max(1)];
                    for tree in &self.trees {
                        let pred = tree.predict_sample(s);
                        if let Some(idx) = self.classes.iter().position(|&c| (c - pred).abs() < 1e-10) {
                            votes[idx] += 1;
                        }
                    }
                    let best_idx = votes.iter().enumerate()
                        .max_by_key(|(_, &c)| c)
                        .map(|(idx, _)| idx)
                        .unwrap_or(0);
                    self.classes.get(best_idx).copied().unwrap_or(0.0)
                } else {
                    let sum: f64 = self.trees.iter().map(|t| t.predict_sample(s)).sum();
                    sum / self.trees.len() as f64
                }
            })
            .collect();

        Ok(Array1::from_vec(predictions))
    }

    pub fn predict_proba(&self, x: &Array2<f64>) -> Result<Array2<f64>> {
        if !self.is_fitted || !self.is_classification {
            return Err(AutoMLError::TrainingError(
                "predict_proba requires a fitted classification model".to_string()
            ));
        }

        let n_samples = x.nrows();
        let n_classes = self.classes.len().max(2);
        let n_trees = self.trees.len() as f64;
        let classes = &self.classes;
        let trees = &self.trees;

        let proba_rows: Vec<Vec<f64>> = (0..n_samples)
            .into_par_iter()
            .map(|i| {
                let sample = x.row(i);
                let s = sample.as_slice().unwrap();
                let mut row = vec![0.0f64; n_classes];

                for tree in trees {
                    let pred = tree.predict_sample(s);
                    if let Some(idx) = classes.iter().position(|&c| (c - pred).abs() < 1e-10) {
                        row[idx] += 1.0;
                    }
                }
                for v in &mut row { *v /= n_trees; }
                row
            })
            .collect();

        let mut proba = Array2::zeros((n_samples, n_classes));
        for (i, row) in proba_rows.into_iter().enumerate() {
            for (j, v) in row.into_iter().enumerate() {
                proba[[i, j]] = v;
            }
        }

        Ok(proba)
    }

    /// Compute feature importances by counting split usage across all trees
    pub fn feature_importances(&self) -> Option<Array1<f64>> {
        if !self.is_fitted || self.n_features == 0 { return None; }
        let mut counts = vec![0.0f64; self.n_features];
        for tree in &self.trees {
            Self::count_splits(tree, &mut counts);
        }
        let total: f64 = counts.iter().sum();
        if total > 0.0 {
            for c in counts.iter_mut() { *c /= total; }
        }
        Some(Array1::from_vec(counts))
    }

    fn count_splits(node: &ExtraTreeNode, counts: &mut [f64]) {
        match node {
            ExtraTreeNode::Leaf { .. } => {}
            ExtraTreeNode::Split { feature, left, right, .. } => {
                if *feature < counts.len() { counts[*feature] += 1.0; }
                Self::count_splits(left, counts);
                Self::count_splits(right, counts);
            }
        }
    }

    pub fn score(&self, x: &Array2<f64>, y: &Array1<f64>) -> Result<f64> {
        let preds = self.predict(x)?;
        if self.is_classification {
            let correct = preds.iter().zip(y.iter())
                .filter(|(p, a)| (*p - *a).abs() < 0.5)
                .count();
            Ok(correct as f64 / y.len() as f64)
        } else {
            let ym = y.mean().unwrap_or(0.0);
            let ss_res = (&preds - y).mapv(|v| v * v).sum();
            let ss_tot = y.mapv(|v| (v - ym).powi(2)).sum();
            Ok(if ss_tot == 0.0 { 1.0 } else { 1.0 - ss_res / ss_tot })
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_extra_trees_classifier() {
        let x = array![
            [1.0, 2.0], [2.0, 1.0], [1.5, 1.5],
            [6.0, 7.0], [7.0, 6.0], [6.5, 6.5],
        ];
        let y = array![0.0, 0.0, 0.0, 1.0, 1.0, 1.0];
        let mut model = ExtraTrees::new_classifier(50).with_random_state(42);
        model.fit(&x, &y).unwrap();
        assert!(model.is_fitted);
        let acc = model.score(&x, &y).unwrap();
        assert!(acc >= 0.8, "ExtraTrees classifier accuracy = {}", acc);
    }

    #[test]
    fn test_extra_trees_regressor() {
        let x = array![[1.0], [2.0], [3.0], [4.0], [5.0]];
        let y = array![2.0, 4.0, 6.0, 8.0, 10.0];
        let mut model = ExtraTrees::new_regressor(50).with_random_state(42);
        model.fit(&x, &y).unwrap();
        assert!(model.is_fitted);
        let r2 = model.score(&x, &y).unwrap();
        assert!(r2 > 0.8, "ExtraTrees regressor R² = {}", r2);
    }

    #[test]
    fn test_extra_trees_predict_proba() {
        let x = array![[0.0, 0.0], [1.0, 1.0], [5.0, 5.0], [6.0, 6.0]];
        let y = array![0.0, 0.0, 1.0, 1.0];
        let mut model = ExtraTrees::new_classifier(30).with_random_state(42);
        model.fit(&x, &y).unwrap();
        let proba = model.predict_proba(&x).unwrap();
        assert_eq!(proba.ncols(), 2);
        assert_eq!(proba.nrows(), 4);
        // Class probabilities should sum to 1
        for i in 0..4 {
            let row_sum: f64 = (0..2).map(|j| proba[[i, j]]).sum();
            assert!((row_sum - 1.0).abs() < 1e-10, "Row {} sums to {}", i, row_sum);
        }
    }
}
