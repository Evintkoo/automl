//! Random Forest implementation

use crate::error::{AutoMLError, Result};
use super::decision_tree::{DecisionTree, Criterion};
use ndarray::{Array1, Array2};
use rand::SeedableRng;
use rand::RngCore;
use rand_chacha::ChaCha8Rng;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};


/// Random Forest model
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RandomForest {
    /// Individual trees
    trees: Vec<DecisionTree>,
    /// Number of trees
    pub n_estimators: usize,
    /// Maximum depth per tree
    pub max_depth: Option<usize>,
    /// Minimum samples to split
    pub min_samples_split: usize,
    /// Minimum samples in leaf
    pub min_samples_leaf: usize,
    /// Maximum features per tree (sqrt by default)
    pub max_features: MaxFeatures,
    /// Bootstrap sampling
    pub bootstrap: bool,
    /// Out-of-bag score
    pub oob_score: bool,
    /// Impurity criterion
    pub criterion: Criterion,
    /// Random state
    pub random_state: Option<u64>,
    /// Is classification task
    is_classification: bool,
    /// Computed OOB score
    oob_score_value: Option<f64>,
    /// Feature importances
    feature_importances: Option<Array1<f64>>,
    /// Number of features
    n_features: usize,
    /// Classes (for classification)
    classes: Vec<f64>,
}

/// Strategy for max features
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum MaxFeatures {
    /// Square root of n_features
    Sqrt,
    /// Log2 of n_features
    Log2,
    /// Fraction of n_features
    Fraction(f64),
    /// Fixed number
    Fixed(usize),
    /// All features
    All,
}

impl Default for RandomForest {
    fn default() -> Self {
        Self::new_classifier(100)
    }
}

impl RandomForest {
    /// Create a new classifier forest
    pub fn new_classifier(n_estimators: usize) -> Self {
        Self {
            trees: Vec::new(),
            n_estimators,
            max_depth: None,
            min_samples_split: 2,
            min_samples_leaf: 1,
            max_features: MaxFeatures::Sqrt,
            bootstrap: true,
            oob_score: false,
            criterion: Criterion::Gini,
            random_state: None,
            is_classification: true,
            oob_score_value: None,
            feature_importances: None,
            n_features: 0,
            classes: Vec::new(),
        }
    }

    /// Create a new regressor forest
    pub fn new_regressor(n_estimators: usize) -> Self {
        Self {
            trees: Vec::new(),
            n_estimators,
            max_depth: None,
            min_samples_split: 2,
            min_samples_leaf: 1,
            max_features: MaxFeatures::Sqrt,
            bootstrap: true,
            oob_score: false,
            criterion: Criterion::MSE,
            random_state: None,
            is_classification: false,
            oob_score_value: None,
            feature_importances: None,
            n_features: 0,
            classes: Vec::new(),
        }
    }

    /// Set maximum depth
    pub fn with_max_depth(mut self, depth: usize) -> Self {
        self.max_depth = Some(depth);
        self
    }

    /// Set minimum samples to split
    pub fn with_min_samples_split(mut self, min_samples: usize) -> Self {
        self.min_samples_split = min_samples;
        self
    }

    /// Set minimum samples in leaf
    pub fn with_min_samples_leaf(mut self, min_samples: usize) -> Self {
        self.min_samples_leaf = min_samples;
        self
    }

    /// Set max features strategy
    pub fn with_max_features(mut self, max_features: MaxFeatures) -> Self {
        self.max_features = max_features;
        self
    }

    /// Set random state
    pub fn with_random_state(mut self, seed: u64) -> Self {
        self.random_state = Some(seed);
        self
    }

    /// Enable OOB score computation
    pub fn with_oob_score(mut self, oob_score: bool) -> Self {
        self.oob_score = oob_score;
        self
    }

    fn compute_max_features(&self, n_features: usize) -> usize {
        match self.max_features {
            MaxFeatures::Sqrt => (n_features as f64).sqrt().ceil() as usize,
            MaxFeatures::Log2 => (n_features as f64).log2().ceil() as usize,
            MaxFeatures::Fraction(f) => (n_features as f64 * f).ceil() as usize,
            MaxFeatures::Fixed(n) => n.min(n_features),
            MaxFeatures::All => n_features,
        }.max(1)
    }

    /// Fit the forest to training data
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

        // Get unique classes for classification
        if self.is_classification {
            let mut classes: Vec<f64> = y.iter().copied().collect();
            classes.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            classes.dedup();
            self.classes = classes;
        }

        // Build trees in parallel
        let base_seed = self.random_state.unwrap_or(42);
        
        let trees: Vec<DecisionTree> = (0..self.n_estimators)
            .into_par_iter()
            .filter_map(|tree_idx| {
                let seed = base_seed.wrapping_add(tree_idx as u64);
                let mut rng = ChaCha8Rng::seed_from_u64(seed);

                // Bootstrap sample
                let sample_indices: Vec<usize> = if self.bootstrap {
                    (0..n_samples)
                        .map(|_| {
                            let idx = (rng.next_u64() as usize) % n_samples;
                            idx
                        })
                        .collect()
                } else {
                    (0..n_samples).collect()
                };

                // Create bootstrap dataset using ndarray select (avoids element-wise copy)
                let x_boot = x.select(ndarray::Axis(0), &sample_indices);
                let y_boot: Array1<f64> = Array1::from_vec(
                    sample_indices.iter().map(|&i| y[i]).collect()
                );

                // Build tree
                let mut tree = if self.is_classification {
                    DecisionTree::new_classifier()
                } else {
                    DecisionTree::new_regressor()
                };

                if let Some(d) = self.max_depth {
                    tree = tree.with_max_depth(d);
                }

                tree = tree
                    .with_min_samples_split(self.min_samples_split)
                    .with_min_samples_leaf(self.min_samples_leaf)
                    .with_criterion(self.criterion);

                tree.max_features = Some(max_features);
                // Give each tree its own seed so per-split random feature subsampling
                // (see DecisionTree::find_best_split) is deterministic given `random_state`
                // but independent/decorrelated across trees in the forest.
                tree.seed = seed;
                if tree.fit(&x_boot, &y_boot).is_err() {
                    return None;
                }

                Some(tree)
            })
            .collect();

        self.trees = trees;

        // Compute feature importances
        self.compute_feature_importances();

        Ok(self)
    }

    fn compute_feature_importances(&mut self) {
        if self.trees.is_empty() {
            return;
        }

        let mut total_importances = vec![0.0; self.n_features];

        for tree in &self.trees {
            if let Some(imp) = tree.feature_importances() {
                for (i, &val) in imp.iter().enumerate() {
                    if i < self.n_features {
                        total_importances[i] += val;
                    }
                }
            }
        }

        let n_trees = self.trees.len() as f64;
        for imp in &mut total_importances {
            *imp /= n_trees;
        }

        // Normalize
        let total: f64 = total_importances.iter().sum();
        if total > 0.0 {
            for imp in &mut total_importances {
                *imp /= total;
            }
        }

        self.feature_importances = Some(Array1::from_vec(total_importances));
    }

    /// Make predictions
    pub fn predict(&self, x: &Array2<f64>) -> Result<Array1<f64>> {
        if self.trees.is_empty() {
            return Err(AutoMLError::ModelNotFitted);
        }

        // Get predictions from all trees
        let all_predictions: Vec<Array1<f64>> = self.trees
            .par_iter()
            .filter_map(|tree| tree.predict(x).ok())
            .collect();

        if all_predictions.is_empty() {
            return Err(AutoMLError::ComputationError(
                "No tree could make predictions".to_string()
            ));
        }

        let n_samples = x.nrows();

        // Aggregate predictions in parallel across samples
        let predictions: Vec<f64> = if self.is_classification {
            // Majority voting parallelized over samples
            let classes = &self.classes;
            (0..n_samples)
                .into_par_iter()
                .map(|i| {
                    let mut votes = vec![0usize; classes.len()];
                    for preds in &all_predictions {
                        let class = preds[i].round() as i64;
                        if let Some(idx) = classes.iter().position(|&c| c.round() as i64 == class) {
                            votes[idx] += 1;
                        }
                    }
                    let best_idx = votes.iter().enumerate()
                        .max_by_key(|(_, &count)| count)
                        .map(|(idx, _)| idx)
                        .unwrap_or(0);
                    classes.get(best_idx).copied().unwrap_or(0.0)
                })
                .collect()
        } else {
            // Mean prediction parallelized over samples
            let n_trees = all_predictions.len() as f64;
            (0..n_samples)
                .into_par_iter()
                .map(|i| {
                    let sum: f64 = all_predictions.iter().map(|p| p[i]).sum();
                    sum / n_trees
                })
                .collect()
        };

        Ok(Array1::from_vec(predictions))
    }

    /// Predict class probabilities (classification only)
    pub fn predict_proba(&self, x: &Array2<f64>) -> Result<Array2<f64>> {
        if self.trees.is_empty() {
            return Err(AutoMLError::ModelNotFitted);
        }

        if !self.is_classification {
            return Err(AutoMLError::ValidationError(
                "predict_proba is only available for classification".to_string()
            ));
        }

        // Get predictions from all trees
        let all_predictions: Vec<Array1<f64>> = self.trees
            .par_iter()
            .filter_map(|tree| tree.predict(x).ok())
            .collect();

        let n_samples = x.nrows();
        let n_classes = self.classes.len();

        // Count votes per class in parallel across samples
        let proba_rows: Vec<Vec<f64>> = (0..n_samples)
            .into_par_iter()
            .map(|i| {
                let mut row = vec![0.0f64; n_classes];
                for preds in &all_predictions {
                    let class = preds[i].round() as i64;
                    if let Some(class_idx) = self.classes.iter().position(|&c| c.round() as i64 == class) {
                        row[class_idx] += 1.0;
                    }
                }
                let row_sum: f64 = row.iter().sum();
                if row_sum > 0.0 {
                    for v in &mut row { *v /= row_sum; }
                }
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

    /// Get feature importances
    pub fn feature_importances(&self) -> Option<&Array1<f64>> {
        self.feature_importances.as_ref()
    }

    /// Get OOB score
    pub fn oob_score_value(&self) -> Option<f64> {
        self.oob_score_value
    }

    /// Get the first tree (for visualization)
    pub fn first_tree(&self) -> Option<&DecisionTree> {
        self.trees.first()
    }

    /// Get number of trees
    pub fn n_trees(&self) -> usize {
        self.trees.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_classifier() {
        let x = array![
            [0.0, 0.0],
            [0.1, 0.1],
            [0.2, 0.2],
            [1.0, 1.0],
            [1.1, 1.1],
            [1.2, 1.2],
        ];
        let y = array![0.0, 0.0, 0.0, 1.0, 1.0, 1.0];

        let mut rf = RandomForest::new_classifier(10)
            .with_random_state(42);
        rf.fit(&x, &y).unwrap();

        let predictions = rf.predict(&x).unwrap();
        
        let accuracy = predictions.iter().zip(y.iter())
            .filter(|(p, a)| (*p - *a).abs() < 0.5)
            .count() as f64 / y.len() as f64;

        assert!(accuracy >= 0.8, "Accuracy too low: {}", accuracy);
    }

    #[test]
    fn test_regressor() {
        let x = array![
            [1.0],
            [2.0],
            [3.0],
            [4.0],
            [5.0],
        ];
        let y = array![1.0, 2.0, 3.0, 4.0, 5.0];

        let mut rf = RandomForest::new_regressor(10)
            .with_random_state(42);
        rf.fit(&x, &y).unwrap();

        let predictions = rf.predict(&x).unwrap();
        
        let mse: f64 = predictions.iter().zip(y.iter())
            .map(|(p, a)| (p - a).powi(2))
            .sum::<f64>() / y.len() as f64;

        assert!(mse < 2.0, "MSE too high: {}", mse);
    }

    #[test]
    fn test_predict_proba() {
        let x = array![
            [0.0, 0.0],
            [1.0, 1.0],
        ];
        let y = array![0.0, 1.0];

        let mut rf = RandomForest::new_classifier(10)
            .with_random_state(42);
        rf.fit(&x, &y).unwrap();

        let proba = rf.predict_proba(&x).unwrap();
        
        assert_eq!(proba.nrows(), 2);
        assert_eq!(proba.ncols(), 2);
        
        // Probabilities should sum to 1
        for i in 0..proba.nrows() {
            let row_sum: f64 = proba.row(i).sum();
            assert!((row_sum - 1.0).abs() < 1e-6, "Row {} sum: {}", i, row_sum);
        }
    }

    #[test]
    fn test_feature_importances() {
        let x = array![
            [1.0, 0.0],
            [2.0, 0.0],
            [3.0, 0.0],
            [4.0, 0.0],
        ];
        let y = array![1.0, 2.0, 3.0, 4.0];

        let mut rf = RandomForest::new_regressor(10)
            .with_random_state(42);
        rf.fit(&x, &y).unwrap();

        let importances = rf.feature_importances().unwrap();
        assert_eq!(importances.len(), 2);
        
        // First feature should be more important
        assert!(importances[0] >= importances[1]);
    }
}
