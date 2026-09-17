//! Voting ensemble methods

use crate::error::{AutoMLError, Result};
use crate::training::Model;
use ndarray::{Array1, Array2};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Voting strategy for classification
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
pub enum VotingStrategy {
    /// Hard voting: majority vote
    Hard,
    /// Soft voting: average probabilities
    Soft,
}

/// Voting classifier ensemble
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VotingClassifier {
    /// Voting strategy
    strategy: VotingStrategy,
    /// Weights for each model
    weights: Option<Vec<f64>>,
    /// Number of classes
    n_classes: Option<usize>,
    /// Predictions from each model during fit
    model_predictions: Option<Vec<Array1<f64>>>,
}

impl VotingClassifier {
    /// Create a new voting classifier
    pub fn new(strategy: VotingStrategy) -> Self {
        Self {
            strategy,
            weights: None,
            n_classes: None,
            model_predictions: None,
        }
    }

    /// Set model weights
    pub fn with_weights(mut self, weights: Vec<f64>) -> Self {
        self.weights = Some(weights);
        self
    }

    /// Predict using multiple models
    pub fn predict_from_models<M: Model>(
        &self,
        models: &[M],
        x: &Array2<f64>,
    ) -> Result<Array1<f64>> {
        if models.is_empty() {
            return Err(AutoMLError::ValidationError(
                "No models provided".to_string(),
            ));
        }

        let n_samples = x.nrows();
        let n_models = models.len();

        // Hard label predictions. Used directly for Hard voting, and also as the source of
        // the global class set (and the one-hot fallback) for Soft voting below.
        let predictions: Result<Vec<Array1<f64>>> =
            models.iter().map(|m| m.predict(x)).collect();
        let predictions = predictions?;

        // Get weights
        let weights: Vec<f64> = self
            .weights
            .clone()
            .unwrap_or_else(|| vec![1.0 / n_models as f64; n_models]);

        if weights.len() != n_models {
            return Err(AutoMLError::ValidationError(format!(
                "Number of weights ({}) does not match number of models ({})",
                weights.len(),
                n_models
            )));
        }

        // Normalize weights
        let weight_sum: f64 = weights.iter().sum();
        let weights: Vec<f64> = weights.iter().map(|w| w / weight_sum).collect();

        match self.strategy {
            VotingStrategy::Hard => {
                self.hard_vote(&predictions, &weights, n_samples)
            }
            VotingStrategy::Soft => {
                // Soft voting must average per-class PROBABILITIES, not raw label integers
                // (averaging labels like {0, 1, 2} is meaningless — e.g. they'd average to
                // ~1.0 regardless of which classes were actually predicted, collapsing any
                // 3+-class problem to a binary-looking result). Build a per-model
                // (n_samples x n_classes) probability matrix — using `Model::predict_proba`
                // when a model supports it, falling back to a one-hot encoding of its hard
                // label otherwise — then average those matrices and take the argmax class.
                let mut classes: Vec<i64> = predictions
                    .iter()
                    .flat_map(|p| p.iter().map(|&v| v.round() as i64))
                    .collect();
                classes.sort_unstable();
                classes.dedup();
                let n_classes = classes.len().max(1);

                let proba_matrices: Result<Vec<Array2<f64>>> = models
                    .iter()
                    .zip(predictions.iter())
                    .map(|(m, hard_pred)| -> Result<Array2<f64>> {
                        if let Some(proba) = m.predict_proba(x)? {
                            if proba.nrows() == n_samples && proba.ncols() == n_classes {
                                return Ok(proba);
                            }
                            // Shape mismatch against the global class set (e.g. a model that
                            // only ever saw a subset of classes) — fall back to one-hot below
                            // rather than risk misaligned columns.
                        }
                        Ok(Self::one_hot(hard_pred, &classes, n_samples, n_classes))
                    })
                    .collect();
                let proba_matrices = proba_matrices?;

                self.soft_vote_proba(&proba_matrices, &weights, &classes, n_samples)
            }
        }
    }

    /// One-hot encode a model's hard label predictions against the global `classes` list.
    /// Used as the soft-voting fallback for models that don't implement `predict_proba`.
    fn one_hot(
        hard_pred: &Array1<f64>,
        classes: &[i64],
        n_samples: usize,
        n_classes: usize,
    ) -> Array2<f64> {
        let mut onehot = Array2::zeros((n_samples, n_classes));
        for (i, &label) in hard_pred.iter().enumerate() {
            let li = label.round() as i64;
            if let Some(col) = classes.iter().position(|&c| c == li) {
                onehot[[i, col]] = 1.0;
            }
        }
        onehot
    }

    /// Soft-vote by averaging per-class probability matrices (weighted per model) and taking
    /// the argmax class per row. This is the actual per-class-probability averaging that
    /// "soft voting" is supposed to do — see `predict_from_models`'s `VotingStrategy::Soft`
    /// arm for how the per-model probability matrices are obtained.
    fn soft_vote_proba(
        &self,
        probas: &[Array2<f64>],
        weights: &[f64],
        classes: &[i64],
        n_samples: usize,
    ) -> Result<Array1<f64>> {
        let n_classes = classes.len().max(1);
        let mut result = Array1::zeros(n_samples);

        for i in 0..n_samples {
            let mut class_scores = vec![0.0f64; n_classes];
            for (proba, &weight) in probas.iter().zip(weights.iter()) {
                for c in 0..n_classes {
                    class_scores[c] += proba[[i, c]] * weight;
                }
            }

            let winner = class_scores
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(idx, _)| idx)
                .unwrap_or(0);

            result[i] = classes.get(winner).copied().unwrap_or(0) as f64;
        }

        Ok(result)
    }

    /// Predict from raw predictions (for use with external models)
    pub fn predict_from_predictions(
        &self,
        predictions: &[Array1<f64>],
        weights: Option<&[f64]>,
    ) -> Result<Array1<f64>> {
        if predictions.is_empty() {
            return Err(AutoMLError::ValidationError(
                "No predictions provided".to_string(),
            ));
        }

        let n_samples = predictions[0].len();
        let n_models = predictions.len();

        let weights: Vec<f64> = weights
            .map(|w| w.to_vec())
            .unwrap_or_else(|| vec![1.0 / n_models as f64; n_models]);

        if weights.len() != n_models {
            return Err(AutoMLError::ValidationError(format!(
                "Number of weights ({}) does not match number of predictions ({})",
                weights.len(),
                n_models
            )));
        }

        let weight_sum: f64 = weights.iter().sum();
        let weights: Vec<f64> = weights.iter().map(|w| w / weight_sum).collect();

        match self.strategy {
            VotingStrategy::Hard => self.hard_vote(predictions, &weights, n_samples),
            VotingStrategy::Soft => self.soft_vote(predictions, &weights, n_samples),
        }
    }

    fn hard_vote(
        &self,
        predictions: &[Array1<f64>],
        weights: &[f64],
        n_samples: usize,
    ) -> Result<Array1<f64>> {
        let mut result = Array1::zeros(n_samples);

        for i in 0..n_samples {
            let mut vote_counts: HashMap<i64, f64> = HashMap::new();

            for (pred, &weight) in predictions.iter().zip(weights.iter()) {
                let class = pred[i].round() as i64;
                *vote_counts.entry(class).or_insert(0.0) += weight;
            }

            // Find majority vote, breaking ties deterministically by lowest class label
            // instead of relying on HashMap iteration order (which is randomized per-process
            // and made the previous `into_iter().max_by(...)` non-deterministic across runs
            // whenever two or more classes tied on weight).
            let mut classes: Vec<i64> = vote_counts.keys().copied().collect();
            classes.sort_unstable();

            let mut winner = 0i64;
            let mut best_weight = f64::MIN;
            for &class in &classes {
                let w = vote_counts[&class];
                if w > best_weight {
                    best_weight = w;
                    winner = class;
                }
            }

            result[i] = winner as f64;
        }

        Ok(result)
    }

    fn soft_vote(
        &self,
        predictions: &[Array1<f64>],
        weights: &[f64],
        n_samples: usize,
    ) -> Result<Array1<f64>> {
        let mut result = Array1::zeros(n_samples);

        for i in 0..n_samples {
            let weighted_sum: f64 = predictions
                .iter()
                .zip(weights.iter())
                .map(|(pred, &weight)| pred[i] * weight)
                .sum();
            
            // For binary classification, threshold at 0.5
            result[i] = if weighted_sum >= 0.5 { 1.0 } else { 0.0 };
        }

        Ok(result)
    }

    /// Get prediction probabilities (soft voting only)
    pub fn predict_proba_from_predictions(
        &self,
        predictions: &[Array1<f64>],
        weights: Option<&[f64]>,
    ) -> Result<Array1<f64>> {
        if predictions.is_empty() {
            return Err(AutoMLError::ValidationError(
                "No predictions provided".to_string(),
            ));
        }

        let n_samples = predictions[0].len();
        let n_models = predictions.len();

        let weights: Vec<f64> = weights
            .map(|w| w.to_vec())
            .unwrap_or_else(|| vec![1.0 / n_models as f64; n_models]);

        if weights.len() != n_models {
            return Err(AutoMLError::ValidationError(format!(
                "Number of weights ({}) does not match number of predictions ({})",
                weights.len(),
                n_models
            )));
        }

        let weight_sum: f64 = weights.iter().sum();
        let weights: Vec<f64> = weights.iter().map(|w| w / weight_sum).collect();

        let mut result = Array1::zeros(n_samples);

        for i in 0..n_samples {
            let weighted_sum: f64 = predictions
                .iter()
                .zip(weights.iter())
                .map(|(pred, &weight)| pred[i] * weight)
                .sum();
            result[i] = weighted_sum;
        }

        Ok(result)
    }
}

impl Default for VotingClassifier {
    fn default() -> Self {
        Self::new(VotingStrategy::Hard)
    }
}

/// Voting regressor ensemble
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VotingRegressor {
    /// Weights for each model
    weights: Option<Vec<f64>>,
    /// Aggregation method
    aggregation: AggregationMethod,
}

/// Aggregation method for regression ensemble
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
pub enum AggregationMethod {
    /// Weighted mean
    Mean,
    /// Weighted median
    Median,
    /// Trimmed mean (removes outliers)
    TrimmedMean { trim_ratio: f64 },
}

impl VotingRegressor {
    /// Create a new voting regressor
    pub fn new() -> Self {
        Self {
            weights: None,
            aggregation: AggregationMethod::Mean,
        }
    }

    /// Set model weights
    pub fn with_weights(mut self, weights: Vec<f64>) -> Self {
        self.weights = Some(weights);
        self
    }

    /// Set aggregation method
    pub fn with_aggregation(mut self, method: AggregationMethod) -> Self {
        self.aggregation = method;
        self
    }

    /// Predict using multiple models
    pub fn predict_from_models<M: Model>(
        &self,
        models: &[M],
        x: &Array2<f64>,
    ) -> Result<Array1<f64>> {
        if models.is_empty() {
            return Err(AutoMLError::ValidationError(
                "No models provided".to_string(),
            ));
        }

        let predictions: Result<Vec<Array1<f64>>> =
            models.iter().map(|m| m.predict(x)).collect();
        let predictions = predictions?;

        self.predict_from_predictions(&predictions, self.weights.as_deref())
    }

    /// Predict from raw predictions
    pub fn predict_from_predictions(
        &self,
        predictions: &[Array1<f64>],
        weights: Option<&[f64]>,
    ) -> Result<Array1<f64>> {
        if predictions.is_empty() {
            return Err(AutoMLError::ValidationError(
                "No predictions provided".to_string(),
            ));
        }

        let n_samples = predictions[0].len();
        let n_models = predictions.len();

        let weights: Vec<f64> = weights
            .map(|w| w.to_vec())
            .unwrap_or_else(|| vec![1.0 / n_models as f64; n_models]);

        if weights.len() != n_models {
            return Err(AutoMLError::ValidationError(format!(
                "Number of weights ({}) does not match number of predictions ({})",
                weights.len(),
                n_models
            )));
        }

        let weight_sum: f64 = weights.iter().sum();
        let weights: Vec<f64> = weights.iter().map(|w| w / weight_sum).collect();

        match self.aggregation {
            AggregationMethod::Mean => {
                self.weighted_mean(predictions, &weights, n_samples)
            }
            AggregationMethod::Median => {
                self.weighted_median(predictions, &weights, n_samples)
            }
            AggregationMethod::TrimmedMean { trim_ratio } => {
                self.trimmed_mean(predictions, &weights, n_samples, trim_ratio)
            }
        }
    }

    fn weighted_mean(
        &self,
        predictions: &[Array1<f64>],
        weights: &[f64],
        n_samples: usize,
    ) -> Result<Array1<f64>> {
        let mut result = Array1::zeros(n_samples);

        for i in 0..n_samples {
            let weighted_sum: f64 = predictions
                .iter()
                .zip(weights.iter())
                .map(|(pred, &weight)| pred[i] * weight)
                .sum();
            result[i] = weighted_sum;
        }

        Ok(result)
    }

    fn weighted_median(
        &self,
        predictions: &[Array1<f64>],
        weights: &[f64],
        n_samples: usize,
    ) -> Result<Array1<f64>> {
        let mut result = Array1::zeros(n_samples);

        for i in 0..n_samples {
            let mut weighted_values: Vec<(f64, f64)> = predictions
                .iter()
                .zip(weights.iter())
                .map(|(pred, &weight)| (pred[i], weight))
                .collect();

            weighted_values.sort_by(|a, b| {
                a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal)
            });

            // Find weighted median
            let total_weight: f64 = weighted_values.iter().map(|(_, w)| w).sum();
            let mut cumsum = 0.0;

            for (value, weight) in weighted_values {
                cumsum += weight;
                if cumsum >= total_weight / 2.0 {
                    result[i] = value;
                    break;
                }
            }
        }

        Ok(result)
    }

    fn trimmed_mean(
        &self,
        predictions: &[Array1<f64>],
        weights: &[f64],
        n_samples: usize,
        trim_ratio: f64,
    ) -> Result<Array1<f64>> {
        let mut result = Array1::zeros(n_samples);
        let n_models = predictions.len();
        let n_trim = ((n_models as f64 * trim_ratio) / 2.0).floor() as usize;

        for i in 0..n_samples {
            let mut values: Vec<(f64, f64)> = predictions
                .iter()
                .zip(weights.iter())
                .map(|(pred, &weight)| (pred[i], weight))
                .collect();

            values.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

            // Trim extremes
            let trimmed: Vec<(f64, f64)> = if n_trim > 0 && values.len() > 2 * n_trim {
                values[n_trim..values.len() - n_trim].to_vec()
            } else {
                values
            };

            // Weighted mean of trimmed values
            let weight_sum: f64 = trimmed.iter().map(|(_, w)| w).sum();
            let weighted_sum: f64 = trimmed.iter().map(|(v, w)| v * w).sum();

            result[i] = if weight_sum > 0.0 {
                weighted_sum / weight_sum
            } else {
                0.0
            };
        }

        Ok(result)
    }
}

impl Default for VotingRegressor {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_hard_voting() {
        let classifier = VotingClassifier::new(VotingStrategy::Hard);

        let predictions = vec![
            array![0.0, 1.0, 1.0, 0.0, 1.0],
            array![0.0, 0.0, 1.0, 1.0, 1.0],
            array![1.0, 1.0, 1.0, 0.0, 0.0],
        ];

        let result = classifier
            .predict_from_predictions(&predictions, None)
            .unwrap();

        assert_eq!(result[0], 0.0); // 2 votes for 0
        assert_eq!(result[1], 1.0); // 2 votes for 1
        assert_eq!(result[2], 1.0); // 3 votes for 1
    }

    #[test]
    fn test_soft_voting() {
        let classifier = VotingClassifier::new(VotingStrategy::Soft);

        let predictions = vec![
            array![0.3, 0.7, 0.9],
            array![0.4, 0.6, 0.8],
            array![0.2, 0.5, 0.7],
        ];

        let result = classifier
            .predict_from_predictions(&predictions, None)
            .unwrap();

        assert_eq!(result[0], 0.0); // avg 0.3
        assert_eq!(result[1], 1.0); // avg 0.6
        assert_eq!(result[2], 1.0); // avg 0.8
    }

    #[test]
    fn test_weighted_voting() {
        let classifier = VotingClassifier::new(VotingStrategy::Soft)
            .with_weights(vec![0.5, 0.3, 0.2]);

        let predictions = vec![
            array![0.8, 0.2],
            array![0.3, 0.3],
            array![0.1, 0.1],
        ];

        let proba = classifier
            .predict_proba_from_predictions(&predictions, Some(&[0.5, 0.3, 0.2]))
            .unwrap();

        // Weighted avg = 0.8*0.5 + 0.3*0.3 + 0.1*0.2 = 0.51
        assert!(proba[0] > 0.5);
    }

    #[test]
    fn test_voting_regressor_mean() {
        let regressor = VotingRegressor::new();

        let predictions = vec![
            array![1.0, 2.0, 3.0],
            array![2.0, 3.0, 4.0],
            array![3.0, 4.0, 5.0],
        ];

        let result = regressor
            .predict_from_predictions(&predictions, None)
            .unwrap();

        assert!((result[0] - 2.0).abs() < 1e-6);
        assert!((result[1] - 3.0).abs() < 1e-6);
        assert!((result[2] - 4.0).abs() < 1e-6);
    }

    #[test]
    fn test_voting_regressor_median() {
        let regressor = VotingRegressor::new()
            .with_aggregation(AggregationMethod::Median);

        let predictions = vec![
            array![1.0, 100.0],
            array![2.0, 3.0],
            array![3.0, 4.0],
        ];

        let result = regressor
            .predict_from_predictions(&predictions, None)
            .unwrap();

        // Median is robust to outlier (100.0)
        assert!((result[1] - 4.0).abs() < 1e-6);
    }
}
