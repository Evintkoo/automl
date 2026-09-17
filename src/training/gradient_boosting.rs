//! Gradient Boosting implementation
//!
//! A native Rust implementation of gradient boosted decision trees,
//! similar to LightGBM/XGBoost but simpler.

use ndarray::{Array1, Array2};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use rand::prelude::*;
use rand_xoshiro::Xoshiro256PlusPlus;

use super::decision_tree::DecisionTree;
use crate::error::Result;

/// Gradient Boosting configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GradientBoostingConfig {
    /// Number of boosting rounds (trees)
    pub n_estimators: usize,
    /// Learning rate (shrinkage)
    pub learning_rate: f64,
    /// Maximum tree depth
    pub max_depth: usize,
    /// Minimum samples per leaf
    pub min_samples_leaf: usize,
    /// Subsample ratio for each tree
    pub subsample: f64,
    /// Column subsample ratio
    pub colsample_bytree: f64,
    /// L2 regularization
    pub reg_lambda: f64,
    /// Random seed
    pub random_state: Option<u64>,
}

impl Default for GradientBoostingConfig {
    fn default() -> Self {
        Self {
            n_estimators: 100,
            learning_rate: 0.1,
            max_depth: 6,
            min_samples_leaf: 1,
            subsample: 0.8,
            colsample_bytree: 0.8,
            reg_lambda: 1.0,
            random_state: Some(42),
        }
    }
}

/// Gradient Boosting Regressor
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GradientBoostingRegressor {
    config: GradientBoostingConfig,
    trees: Vec<DecisionTree>,
    col_indices_per_tree: Vec<Vec<usize>>,
    initial_prediction: f64,
    feature_importances: Vec<f64>,
}

impl GradientBoostingRegressor {
    pub fn new(config: GradientBoostingConfig) -> Self {
        Self {
            config,
            trees: Vec::new(),
            col_indices_per_tree: Vec::new(),
            initial_prediction: 0.0,
            feature_importances: Vec::new(),
        }
    }

    /// Fit the gradient boosting model
    pub fn fit(&mut self, x: &Array2<f64>, y: &Array1<f64>) -> Result<()> {
        let n_samples = x.nrows();
        let n_features = x.ncols();
        
        // Initialize with mean
        self.initial_prediction = y.mean().unwrap_or(0.0);
        
        // Initialize predictions
        let mut predictions = Array1::from_elem(n_samples, self.initial_prediction);
        
        // Initialize RNG
        let mut rng = match self.config.random_state {
            Some(seed) => Xoshiro256PlusPlus::seed_from_u64(seed),
            None => Xoshiro256PlusPlus::from_entropy(),
        };
        
        // Feature importance accumulator
        self.feature_importances = vec![0.0; n_features];
        
        // Boosting rounds
        for _ in 0..self.config.n_estimators {
            // Compute residuals in parallel for large datasets
            let residuals: Array1<f64> = if n_samples > 10000 {
                let preds = &predictions;
                let res: Vec<f64> = (0..n_samples)
                    .into_par_iter()
                    .map(|i| y[i] - preds[i])
                    .collect();
                Array1::from_vec(res)
            } else {
                y.iter()
                    .zip(predictions.iter())
                    .map(|(yi, pi)| yi - pi)
                    .collect()
            };
            
            // Sample rows
            let sample_indices = self.subsample_indices(n_samples, &mut rng);
            
            // Sample columns
            let col_indices = self.colsample_indices(n_features, &mut rng);
            
            // Create subsampled data
            let (x_sub, y_sub) = self.subsample_data(x, &residuals, &sample_indices, &col_indices);
            
            // Train a tree on residuals
            let mut tree = DecisionTree::new_regressor()
                .with_max_depth(self.config.max_depth)
                .with_min_samples_leaf(self.config.min_samples_leaf);
            tree.fit(&x_sub, &y_sub)?;
            
            // Update the running `predictions` cache for ALL rows, not just the ones sampled
            // to fit this round's tree — subsampling should only affect which rows are used
            // to FIT the weak learner, not which rows get this round's contribution applied.
            // Predicting only on `sample_indices` and leaving the rest stale would make
            // `predictions` diverge from what a fresh `predict()` call computes. The tree was
            // trained on the column-subsampled feature space, so we predict on all rows
            // restricted to the same `col_indices` (matching `predict()` below).
            // See lightgbm.rs/catboost.rs for the same pattern.
            let x_col_sub = x.select(ndarray::Axis(1), &col_indices);
            let tree_pred_all = tree.predict(&x_col_sub)?;
            predictions.scaled_add(self.config.learning_rate, &tree_pred_all);

            // Accumulate feature importance
            if let Some(tree_importance) = tree.feature_importances() {
                for (j, &col_idx) in col_indices.iter().enumerate() {
                    if j < tree_importance.len() {
                        self.feature_importances[col_idx] += tree_importance[j];
                    }
                }
            }

            self.trees.push(tree);
            self.col_indices_per_tree.push(col_indices);
        }

        // Normalize feature importances
        let total: f64 = self.feature_importances.iter().sum();
        if total > 0.0 {
            for imp in &mut self.feature_importances {
                *imp /= total;
            }
        }

        Ok(())
    }

    /// Make predictions
    pub fn predict(&self, x: &Array2<f64>) -> Result<Array1<f64>> {
        let n = x.nrows();
        let mut predictions = Array1::from_elem(n, self.initial_prediction);
        let lr = self.config.learning_rate;

        for (tree, col_indices) in self.trees.iter().zip(self.col_indices_per_tree.iter()) {
            let x_sub = x.select(ndarray::Axis(1), col_indices);
            let tree_pred = tree.predict(&x_sub)?;
            predictions.scaled_add(lr, &tree_pred);
        }

        Ok(predictions)
    }

    /// Get feature importances
    pub fn feature_importances(&self) -> &[f64] {
        &self.feature_importances
    }

    fn subsample_indices(&self, n: usize, rng: &mut Xoshiro256PlusPlus) -> Vec<usize> {
        let sample_size = ((n as f64) * self.config.subsample).ceil() as usize;
        let mut indices: Vec<usize> = (0..n).collect();
        // Partial Fisher-Yates: only shuffle first sample_size elements
        for i in 0..sample_size.min(n) {
            let j = i + (rng.gen_range(0..n - i));
            indices.swap(i, j);
        }
        indices.truncate(sample_size);
        indices.sort();
        indices
    }

    fn colsample_indices(&self, n: usize, rng: &mut Xoshiro256PlusPlus) -> Vec<usize> {
        let sample_size = ((n as f64) * self.config.colsample_bytree).ceil() as usize;
        let mut indices: Vec<usize> = (0..n).collect();
        // Partial Fisher-Yates: only shuffle first sample_size elements
        for i in 0..sample_size.min(n) {
            let j = i + (rng.gen_range(0..n - i));
            indices.swap(i, j);
        }
        indices.truncate(sample_size);
        indices.sort();
        indices
    }

    fn subsample_data(
        &self,
        x: &Array2<f64>,
        y: &Array1<f64>,
        row_indices: &[usize],
        col_indices: &[usize],
    ) -> (Array2<f64>, Array1<f64>) {
        let x_rows = x.select(ndarray::Axis(0), row_indices);
        let x_sub = x_rows.select(ndarray::Axis(1), col_indices);
        let y_sub: Array1<f64> = Array1::from_vec(
            row_indices.iter().map(|&i| y[i]).collect()
        );
        (x_sub, y_sub)
    }
}

/// Gradient Boosting Classifier
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GradientBoostingClassifier {
    config: GradientBoostingConfig,
    trees: Vec<DecisionTree>,
    col_indices_per_tree: Vec<Vec<usize>>,
    initial_log_odds: f64,
    classes: Vec<i64>,
    feature_importances: Vec<f64>,
}

impl GradientBoostingClassifier {
    pub fn new(config: GradientBoostingConfig) -> Self {
        Self {
            config,
            trees: Vec::new(),
            col_indices_per_tree: Vec::new(),
            initial_log_odds: 0.0,
            classes: vec![0, 1],
            feature_importances: Vec::new(),
        }
    }

    /// Fit binary classification
    pub fn fit(&mut self, x: &Array2<f64>, y: &Array1<f64>) -> Result<()> {
        let n_samples = x.nrows();
        let n_features = x.ncols();
        
        // Compute initial log odds
        let p = y.mean().unwrap_or(0.5);
        self.initial_log_odds = (p / (1.0 - p + 1e-10)).ln();
        
        // Initialize probabilities
        let mut log_odds = Array1::from_elem(n_samples, self.initial_log_odds);
        
        let mut rng = match self.config.random_state {
            Some(seed) => Xoshiro256PlusPlus::seed_from_u64(seed),
            None => Xoshiro256PlusPlus::from_entropy(),
        };
        
        self.feature_importances = vec![0.0; n_features];
        
        for _ in 0..self.config.n_estimators {
            // Compute probabilities
            let probs: Array1<f64> = log_odds.iter()
                .map(|&lo| 1.0 / (1.0 + (-lo).exp()))
                .collect();
            
            // Compute residuals (gradient of log loss)
            let residuals: Array1<f64> = y.iter()
                .zip(probs.iter())
                .map(|(yi, pi)| yi - pi)
                .collect();
            
            // Sample rows and columns
            let sample_indices = self.subsample_indices(n_samples, &mut rng);
            let col_indices = self.colsample_indices(n_features, &mut rng);
            
            let (x_sub, y_sub) = self.subsample_data(x, &residuals, &sample_indices, &col_indices);
            
            // Train tree on residuals
            let mut tree = DecisionTree::new_regressor()
                .with_max_depth(self.config.max_depth)
                .with_min_samples_leaf(self.config.min_samples_leaf);
            tree.fit(&x_sub, &y_sub)?;
            
            // Update the running `log_odds` cache for ALL rows, not just the sampled ones
            // used to fit this round's tree — see the matching comment in
            // GradientBoostingRegressor::fit for why (keeps the cache consistent with a
            // fresh `predict_proba()` call regardless of `subsample`).
            let x_col_sub = x.select(ndarray::Axis(1), &col_indices);
            let tree_pred_all = tree.predict(&x_col_sub)?;
            log_odds.scaled_add(self.config.learning_rate, &tree_pred_all);

            // Accumulate feature importance
            if let Some(tree_importance) = tree.feature_importances() {
                for (j, &col_idx) in col_indices.iter().enumerate() {
                    if j < tree_importance.len() {
                        self.feature_importances[col_idx] += tree_importance[j];
                    }
                }
            }

            self.trees.push(tree);
            self.col_indices_per_tree.push(col_indices);
        }

        // Normalize feature importances
        let total: f64 = self.feature_importances.iter().sum();
        if total > 0.0 {
            for imp in &mut self.feature_importances {
                *imp /= total;
            }
        }

        Ok(())
    }

    /// Predict class labels
    pub fn predict(&self, x: &Array2<f64>) -> Result<Array1<f64>> {
        let probs = self.predict_proba(x)?;
        Ok(probs.iter().map(|&p| if p >= 0.5 { 1.0 } else { 0.0 }).collect())
    }

    /// Predict probabilities
    pub fn predict_proba(&self, x: &Array2<f64>) -> Result<Array1<f64>> {
        let n = x.nrows();
        let mut log_odds = Array1::from_elem(n, self.initial_log_odds);
        let lr = self.config.learning_rate;

        for (tree, col_indices) in self.trees.iter().zip(self.col_indices_per_tree.iter()) {
            let x_sub = x.select(ndarray::Axis(1), col_indices);
            let tree_pred = tree.predict(&x_sub)?;
            log_odds.scaled_add(lr, &tree_pred);
        }

        Ok(log_odds.mapv(|lo| 1.0 / (1.0 + (-lo).exp())))
    }

    /// Get feature importances
    pub fn feature_importances(&self) -> &[f64] {
        &self.feature_importances
    }

    fn subsample_indices(&self, n: usize, rng: &mut Xoshiro256PlusPlus) -> Vec<usize> {
        let sample_size = ((n as f64) * self.config.subsample).ceil() as usize;
        let mut indices: Vec<usize> = (0..n).collect();
        // Partial Fisher-Yates: only shuffle first sample_size elements
        for i in 0..sample_size.min(n) {
            let j = i + (rng.gen_range(0..n - i));
            indices.swap(i, j);
        }
        indices.truncate(sample_size);
        indices.sort();
        indices
    }

    fn colsample_indices(&self, n: usize, rng: &mut Xoshiro256PlusPlus) -> Vec<usize> {
        let sample_size = ((n as f64) * self.config.colsample_bytree).ceil() as usize;
        let mut indices: Vec<usize> = (0..n).collect();
        // Partial Fisher-Yates: only shuffle first sample_size elements
        for i in 0..sample_size.min(n) {
            let j = i + (rng.gen_range(0..n - i));
            indices.swap(i, j);
        }
        indices.truncate(sample_size);
        indices.sort();
        indices
    }

    fn subsample_data(
        &self,
        x: &Array2<f64>,
        y: &Array1<f64>,
        row_indices: &[usize],
        col_indices: &[usize],
    ) -> (Array2<f64>, Array1<f64>) {
        let x_rows = x.select(ndarray::Axis(0), row_indices);
        let x_sub = x_rows.select(ndarray::Axis(1), col_indices);
        let y_sub: Array1<f64> = Array1::from_vec(
            row_indices.iter().map(|&i| y[i]).collect()
        );
        (x_sub, y_sub)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_regression_data() -> (Array2<f64>, Array1<f64>) {
        let x = Array2::from_shape_vec((100, 2), 
            (0..200).map(|i| i as f64 * 0.1).collect()
        ).unwrap();
        
        let y: Array1<f64> = x.rows().into_iter()
            .map(|row| row[0] * 2.0 + row[1] * 0.5 + 1.0)
            .collect();
        
        (x, y)
    }

    fn create_classification_data() -> (Array2<f64>, Array1<f64>) {
        let x = Array2::from_shape_vec((100, 2), 
            (0..200).map(|i| i as f64 * 0.1).collect()
        ).unwrap();
        
        let y: Array1<f64> = x.rows().into_iter()
            .map(|row| if row[0] + row[1] > 10.0 { 1.0 } else { 0.0 })
            .collect();
        
        (x, y)
    }

    #[test]
    fn test_gradient_boosting_regressor() {
        let (x, y) = create_regression_data();
        let config = GradientBoostingConfig {
            n_estimators: 10,
            max_depth: 3,
            learning_rate: 0.1,
            ..Default::default()
        };
        
        let mut model = GradientBoostingRegressor::new(config);
        model.fit(&x, &y).unwrap();
        
        let predictions = model.predict(&x).unwrap();
        assert_eq!(predictions.len(), 100);
        
        // Check that predictions are reasonable
        let mse: f64 = y.iter()
            .zip(predictions.iter())
            .map(|(yi, pi)| (yi - pi).powi(2))
            .sum::<f64>() / y.len() as f64;
        
        // MSE should be less than variance of y
        let y_var = y.var(0.0);
        assert!(mse < y_var, "MSE ({}) should be less than variance ({})", mse, y_var);
    }

    #[test]
    fn test_gradient_boosting_classifier() {
        let (x, y) = create_classification_data();
        let config = GradientBoostingConfig {
            n_estimators: 10,
            max_depth: 3,
            learning_rate: 0.1,
            ..Default::default()
        };
        
        let mut model = GradientBoostingClassifier::new(config);
        model.fit(&x, &y).unwrap();
        
        let predictions = model.predict(&x).unwrap();
        assert_eq!(predictions.len(), 100);
        
        // Check accuracy
        let correct: usize = y.iter()
            .zip(predictions.iter())
            .filter(|(&yi, &pi)| (yi - pi).abs() < 0.5)
            .count();
        
        let accuracy = correct as f64 / y.len() as f64;
        assert!(accuracy > 0.7, "Accuracy ({}) should be above 70%", accuracy);
    }

    #[test]
    fn test_feature_importances() {
        let (x, y) = create_regression_data();
        let config = GradientBoostingConfig {
            n_estimators: 10,
            ..Default::default()
        };
        
        let mut model = GradientBoostingRegressor::new(config);
        model.fit(&x, &y).unwrap();
        
        let importances = model.feature_importances();
        assert_eq!(importances.len(), 2);
        
        // Sum should be approximately 1
        let sum: f64 = importances.iter().sum();
        assert!((sum - 1.0).abs() < 0.01, "Sum of importances ({}) should be ~1", sum);
    }
}
