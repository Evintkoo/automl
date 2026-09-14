//! AdaBoost (Adaptive Boosting) implementation
//!
//! AdaBoost builds an ensemble of weak learners (decision stumps), weighting
//! misclassified samples more heavily in subsequent rounds.

use crate::error::{AutoMLError, Result};
use ndarray::{Array1, Array2};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// A single decision stump: splits on one feature at one threshold
#[derive(Debug, Clone, Serialize, Deserialize)]
struct Stump {
    feature_index: usize,
    threshold: f64,
    /// Prediction when feature <= threshold
    left_label: f64,
    /// Prediction when feature > threshold
    right_label: f64,
}

impl Stump {
    fn predict_sample(&self, sample: &[f64]) -> f64 {
        if sample[self.feature_index] <= self.threshold {
            self.left_label
        } else {
            self.right_label
        }
    }
}

/// AdaBoost Classifier (SAMME variant, supports multi-class)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdaBoostClassifier {
    pub n_estimators: usize,
    pub learning_rate: f64,
    stumps: Vec<Stump>,
    alphas: Vec<f64>,
    classes: Vec<f64>,
    pub is_fitted: bool,
}

impl Default for AdaBoostClassifier {
    fn default() -> Self {
        Self::new(50, 1.0)
    }
}

impl AdaBoostClassifier {
    pub fn new(n_estimators: usize, learning_rate: f64) -> Self {
        Self {
            n_estimators,
            learning_rate,
            stumps: Vec::new(),
            alphas: Vec::new(),
            classes: Vec::new(),
            is_fitted: false,
        }
    }

    pub fn with_n_estimators(mut self, n: usize) -> Self {
        self.n_estimators = n;
        self
    }

    pub fn with_learning_rate(mut self, lr: f64) -> Self {
        self.learning_rate = lr;
        self
    }

    /// Find the best decision stump given sample weights
    fn fit_stump(x: &Array2<f64>, y: &Array1<f64>, weights: &Array1<f64>, classes: &[f64]) -> Stump {
        let n_features = x.ncols();
        let n_samples = x.nrows();

        let mut best_stump = Stump {
            feature_index: 0,
            threshold: 0.0,
            left_label: classes[0],
            right_label: classes.get(1).copied().unwrap_or(classes[0]),
        };
        let mut best_error = f64::MAX;

        for f in 0..n_features {
            // Sort samples by feature value
            let mut order: Vec<usize> = (0..n_samples).collect();
            order.sort_by(|&a, &b| x[[a, f]].partial_cmp(&x[[b, f]]).unwrap_or(std::cmp::Ordering::Equal));

            // Build class-to-index mapping
            let class_idx: HashMap<u64, usize> = classes.iter()
                .enumerate()
                .map(|(i, &c)| (c.to_bits(), i))
                .collect();
            let n_classes = classes.len();

            // Initialize: all samples on the right side
            let mut left_w = vec![0.0f64; n_classes];
            let mut right_w = vec![0.0f64; n_classes];
            for &i in &order {
                if let Some(&ci) = class_idx.get(&y[i].to_bits()) {
                    right_w[ci] += weights[i];
                }
            }

            // Scan thresholds by moving samples from right to left
            for pos in 0..n_samples - 1 {
                let i = order[pos];
                if let Some(&ci) = class_idx.get(&y[i].to_bits()) {
                    left_w[ci] += weights[i];
                    right_w[ci] -= weights[i];
                }

                // Skip if next sample has same feature value
                let next_i = order[pos + 1];
                if (x[[i, f]] - x[[next_i, f]]).abs() < 1e-12 {
                    continue;
                }
                let threshold = (x[[i, f]] + x[[next_i, f]]) / 2.0;

                // For each (left_label, right_label) pair, compute weighted error from counts
                let left_total: f64 = left_w.iter().sum();
                let right_total: f64 = right_w.iter().sum();

                for (li, &left_label) in classes.iter().enumerate() {
                    for (ri, &right_label) in classes.iter().enumerate() {
                        if left_label == right_label && n_classes > 1 {
                            continue;
                        }
                        // Error = left samples not matching left_label + right samples not matching right_label
                        let error = (left_total - left_w[li]) + (right_total - right_w[ri]);
                        if error < best_error {
                            best_error = error;
                            best_stump = Stump {
                                feature_index: f,
                                threshold,
                                left_label,
                                right_label,
                            };
                        }
                    }
                }
            }
        }
        best_stump
    }

    pub fn fit(&mut self, x: &Array2<f64>, y: &Array1<f64>) -> Result<&mut Self> {
        let n_samples = x.nrows();
        if n_samples != y.len() {
            return Err(AutoMLError::ShapeError {
                expected: format!("y length = {}", n_samples),
                actual: format!("y length = {}", y.len()),
            });
        }

        // Collect unique classes
        let mut classes: Vec<f64> = y.to_vec();
        classes.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        classes.dedup();
        self.classes = classes.clone();

        let n_classes = self.classes.len();
        let mut weights = Array1::from_elem(n_samples, 1.0 / n_samples as f64);

        self.stumps.clear();
        self.alphas.clear();

        for _round in 0..self.n_estimators {
            let stump = Self::fit_stump(x, y, &weights, &self.classes);

            // Compute weighted error
            let mut error = 0.0;
            let predictions: Vec<f64> = (0..n_samples)
                .map(|i| {
                    let row = x.row(i);
                    stump.predict_sample(row.as_slice().unwrap())
                })
                .collect();

            for i in 0..n_samples {
                if (predictions[i] - y[i]).abs() > 1e-10 {
                    error += weights[i];
                }
            }

            // Clamp error to avoid division issues
            error = error.clamp(1e-15, 1.0 - 1e-15);

            // SAMME alpha for multi-class
            let alpha = self.learning_rate
                * ((1.0 - error) / error).ln()
                + (n_classes as f64 - 1.0).max(1.0).ln();

            // Update weights
            for i in 0..n_samples {
                if (predictions[i] - y[i]).abs() > 1e-10 {
                    weights[i] *= (alpha).exp();
                }
            }
            // Normalize weights
            let w_sum = weights.sum();
            if w_sum > 0.0 {
                weights /= w_sum;
            }

            self.stumps.push(stump);
            self.alphas.push(alpha);
        }

        self.is_fitted = true;
        Ok(self)
    }

    pub fn predict(&self, x: &Array2<f64>) -> Result<Array1<f64>> {
        if !self.is_fitted {
            return Err(AutoMLError::ModelNotFitted);
        }

        let n_samples = x.nrows();
        let stumps = &self.stumps;
        let alphas_ref = &self.alphas;

        let preds: Vec<f64> = (0..n_samples)
            .into_par_iter()
            .map(|i| {
                let row = x.row(i);
                let sample = row.as_slice().unwrap();

                let mut class_scores: std::collections::HashMap<u64, f64> = std::collections::HashMap::new();
                for (stump, &alpha) in stumps.iter().zip(alphas_ref.iter()) {
                    let pred = stump.predict_sample(sample);
                    let key = pred.to_bits();
                    *class_scores.entry(key).or_insert(0.0) += alpha;
                }

                let best_class_key = class_scores
                    .iter()
                    .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
                    .map(|(&k, _)| k)
                    .unwrap_or(0);
                f64::from_bits(best_class_key)
            })
            .collect();

        Ok(Array1::from_vec(preds))
    }

    pub fn predict_proba(&self, x: &Array2<f64>) -> Result<Array2<f64>> {
        if !self.is_fitted {
            return Err(AutoMLError::ModelNotFitted);
        }

        let n_samples = x.nrows();
        let n_classes = self.classes.len().max(2);
        let stumps = &self.stumps;
        let alphas_ref = &self.alphas;
        let classes = &self.classes;

        let proba_rows: Vec<Vec<f64>> = (0..n_samples)
            .into_par_iter()
            .map(|i| {
                let row = x.row(i);
                let sample = row.as_slice().unwrap();

                let mut class_scores = vec![0.0f64; n_classes];
                for (stump, &alpha) in stumps.iter().zip(alphas_ref.iter()) {
                    let pred = stump.predict_sample(sample);
                    if let Some(idx) = classes.iter().position(|&c| (c - pred).abs() < 1e-10) {
                        class_scores[idx] += alpha;
                    }
                }

                // Softmax normalization
                let max_score = class_scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
                let exp_sum: f64 = class_scores.iter().map(|&s| (s - max_score).exp()).sum();
                class_scores.iter().map(|&s| (s - max_score).exp() / exp_sum).collect()
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

    pub fn score(&self, x: &Array2<f64>, y: &Array1<f64>) -> Result<f64> {
        let preds = self.predict(x)?;
        let correct = preds.iter().zip(y.iter())
            .filter(|(p, a)| (*p - *a).abs() < 0.5)
            .count();
        Ok(correct as f64 / y.len() as f64)
    }

    /// Compute feature importances from weighted stump feature usage
    pub fn feature_importances(&self, n_features: usize) -> Option<Array1<f64>> {
        if !self.is_fitted || n_features == 0 { return None; }
        let mut importances = vec![0.0f64; n_features];
        for (stump, &alpha) in self.stumps.iter().zip(self.alphas.iter()) {
            if stump.feature_index < n_features {
                importances[stump.feature_index] += alpha.abs();
            }
        }
        let total: f64 = importances.iter().sum();
        if total > 0.0 {
            for v in importances.iter_mut() { *v /= total; }
        }
        Some(Array1::from_vec(importances))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_adaboost_binary() {
        let x = array![
            [1.0, 2.0], [2.0, 3.0], [3.0, 4.0],
            [6.0, 7.0], [7.0, 8.0], [8.0, 9.0],
        ];
        let y = array![0.0, 0.0, 0.0, 1.0, 1.0, 1.0];
        let mut model = AdaBoostClassifier::new(10, 1.0);
        model.fit(&x, &y).unwrap();
        assert!(model.is_fitted);
        let acc = model.score(&x, &y).unwrap();
        assert!(acc >= 0.8, "AdaBoost accuracy = {}", acc);
    }

    #[test]
    fn test_adaboost_predict_proba() {
        let x = array![[0.0, 0.0], [1.0, 1.0], [5.0, 5.0], [6.0, 6.0]];
        let y = array![0.0, 0.0, 1.0, 1.0];
        let mut model = AdaBoostClassifier::new(20, 1.0);
        model.fit(&x, &y).unwrap();
        let proba = model.predict_proba(&x).unwrap();
        assert_eq!(proba.ncols(), 2);
        assert_eq!(proba.nrows(), 4);
    }
}
