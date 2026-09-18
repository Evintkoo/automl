//! Permutation feature importance

use crate::error::Result;
use crate::training::Model;
use ndarray::{Array1, Array2};
use rand::prelude::*;
use serde::{Deserialize, Serialize};

/// Result of feature importance computation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImportanceResult {
    /// Feature indices
    pub feature_indices: Vec<usize>,
    /// Feature names (if provided)
    pub feature_names: Option<Vec<String>>,
    /// Mean importance scores
    pub importances_mean: Vec<f64>,
    /// Standard deviation of importance scores
    pub importances_std: Vec<f64>,
    /// Raw importance scores per repetition
    pub importances_raw: Vec<Vec<f64>>,
}

impl ImportanceResult {
    /// Get sorted feature indices by importance (descending)
    pub fn sorted_indices(&self) -> Vec<usize> {
        let mut indexed: Vec<(usize, f64)> = self
            .importances_mean
            .iter()
            .copied()
            .enumerate()
            .collect();
        indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        indexed.into_iter().map(|(i, _)| i).collect()
    }

    /// Get top k important features
    pub fn top_k(&self, k: usize) -> Vec<(usize, f64)> {
        let sorted = self.sorted_indices();
        sorted
            .into_iter()
            .take(k)
            .map(|i| (i, self.importances_mean[i]))
            .collect()
    }

    /// Get features with importance above threshold
    pub fn above_threshold(&self, threshold: f64) -> Vec<(usize, f64)> {
        self.importances_mean
            .iter()
            .copied()
            .enumerate()
            .filter(|(_, imp)| *imp > threshold)
            .collect()
    }
}

/// Permutation feature importance calculator
pub struct PermutationImportance<F>
where
    F: Fn(&Array2<f64>) -> Result<Array1<f64>>,
{
    /// Prediction function
    predict_fn: F,
    /// Number of permutation repeats
    n_repeats: usize,
    /// Random seed
    seed: Option<u64>,
    /// Feature names
    feature_names: Option<Vec<String>>,
}

impl<F> PermutationImportance<F>
where
    F: Fn(&Array2<f64>) -> Result<Array1<f64>>,
{
    /// Create new permutation importance calculator
    pub fn new(predict_fn: F) -> Self {
        Self {
            predict_fn,
            n_repeats: 5,
            seed: None,
            feature_names: None,
        }
    }

    /// Set number of permutation repeats
    pub fn with_n_repeats(mut self, n_repeats: usize) -> Self {
        self.n_repeats = n_repeats.max(1);
        self
    }

    /// Set random seed
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = Some(seed);
        self
    }

    /// Set feature names
    pub fn with_feature_names(mut self, names: Vec<String>) -> Self {
        self.feature_names = Some(names);
        self
    }

    /// Compute permutation importance using MSE as scoring function
    pub fn compute(&self, x: &Array2<f64>, y: &Array1<f64>) -> Result<ImportanceResult> {
        self.compute_with_scorer(x, y, |y_true, y_pred| {
            // Mean squared error
            y_true
                .iter()
                .zip(y_pred.iter())
                .map(|(t, p)| (t - p).powi(2))
                .sum::<f64>()
                / y_true.len() as f64
        })
    }

    /// Compute permutation importance with custom scoring function
    pub fn compute_with_scorer<S>(
        &self,
        x: &Array2<f64>,
        y: &Array1<f64>,
        scorer: S,
    ) -> Result<ImportanceResult>
    where
        S: Fn(&Array1<f64>, &Array1<f64>) -> f64,
    {
        let n_features = x.ncols();
        let mut rng = match self.seed {
            Some(seed) => StdRng::seed_from_u64(seed),
            None => StdRng::from_entropy(),
        };

        // Compute baseline score
        let baseline_pred = (self.predict_fn)(x)?;
        let baseline_score = scorer(y, &baseline_pred);

        // Compute importance for each feature
        let mut importances_raw: Vec<Vec<f64>> = vec![Vec::new(); n_features];

        // Reuse one buffer across every (repeat, feature) permutation instead
        // of cloning the full matrix n_repeats * n_features times: permute
        // one column in place, score, then restore that column's original
        // values before moving on to the next feature/repeat.
        let mut x_permuted = x.clone();
        for _ in 0..self.n_repeats {
            for feature_idx in 0..n_features {
                let mut col: Vec<f64> = x.column(feature_idx).iter().copied().collect();
                col.shuffle(&mut rng);

                for (i, val) in col.into_iter().enumerate() {
                    x_permuted[[i, feature_idx]] = val;
                }

                // Compute score with permuted feature
                let permuted_pred = (self.predict_fn)(&x_permuted)?;
                let permuted_score = scorer(y, &permuted_pred);

                // Importance = how much worse the model gets
                let importance = permuted_score - baseline_score;
                importances_raw[feature_idx].push(importance);

                // Restore the original column before permuting the next one.
                for (i, &val) in x.column(feature_idx).iter().enumerate() {
                    x_permuted[[i, feature_idx]] = val;
                }
            }
        }

        // Compute mean and std
        let importances_mean: Vec<f64> = importances_raw
            .iter()
            .map(|scores| scores.iter().sum::<f64>() / scores.len() as f64)
            .collect();

        let importances_std: Vec<f64> = importances_raw
            .iter()
            .zip(importances_mean.iter())
            .map(|(scores, mean)| {
                let variance: f64 =
                    scores.iter().map(|s| (s - mean).powi(2)).sum::<f64>() / scores.len() as f64;
                variance.sqrt()
            })
            .collect();

        Ok(ImportanceResult {
            feature_indices: (0..n_features).collect(),
            feature_names: self.feature_names.clone(),
            importances_mean,
            importances_std,
            importances_raw,
        })
    }
}

/// Trait-based permutation importance for any Model
#[allow(dead_code)]
pub struct ModelPermutationImportance {
    n_repeats: usize,
    seed: Option<u64>,
    feature_names: Option<Vec<String>>,
}

#[allow(dead_code)]
impl ModelPermutationImportance {
    /// Create new calculator
    pub fn new() -> Self {
        Self {
            n_repeats: 5,
            seed: None,
            feature_names: None,
        }
    }

    /// Set number of repeats
    pub fn with_n_repeats(mut self, n_repeats: usize) -> Self {
        self.n_repeats = n_repeats.max(1);
        self
    }

    /// Set seed
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = Some(seed);
        self
    }

    /// Set feature names
    pub fn with_feature_names(mut self, names: Vec<String>) -> Self {
        self.feature_names = Some(names);
        self
    }

    /// Compute importance for a Model implementation
    pub fn compute<M: Model>(
        &self,
        model: &M,
        x: &Array2<f64>,
        y: &Array1<f64>,
    ) -> Result<ImportanceResult> {
        let calc = PermutationImportance::new(|data| model.predict(data))
            .with_n_repeats(self.n_repeats);

        let calc = match self.seed {
            Some(seed) => calc.with_seed(seed),
            None => calc,
        };

        let calc = match &self.feature_names {
            Some(names) => calc.with_feature_names(names.clone()),
            None => calc,
        };

        calc.compute(x, y)
    }
}

impl Default for ModelPermutationImportance {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_permutation_importance_basic() {
        // Simple predict function: just return first column
        let predict_fn = |x: &Array2<f64>| -> Result<Array1<f64>> {
            Ok(x.column(0).to_owned())
        };

        let x = Array2::from_shape_vec(
            (10, 3),
            vec![
                1.0, 0.0, 0.1,
                2.0, 0.0, 0.2,
                3.0, 0.0, 0.3,
                4.0, 0.0, 0.4,
                5.0, 0.0, 0.5,
                6.0, 0.0, 0.6,
                7.0, 0.0, 0.7,
                8.0, 0.0, 0.8,
                9.0, 0.0, 0.9,
                10.0, 0.0, 1.0,
            ],
        ).unwrap();
        let y = array![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];

        let calc = PermutationImportance::new(predict_fn)
            .with_n_repeats(3)
            .with_seed(42);

        let result = calc.compute(&x, &y).unwrap();

        // Feature 0 should be most important since y = feature 0
        let sorted = result.sorted_indices();
        assert_eq!(sorted[0], 0);
    }

    #[test]
    fn test_importance_result_methods() {
        let result = ImportanceResult {
            feature_indices: vec![0, 1, 2],
            feature_names: Some(vec!["a".to_string(), "b".to_string(), "c".to_string()]),
            importances_mean: vec![0.5, 0.1, 0.3],
            importances_std: vec![0.1, 0.05, 0.08],
            importances_raw: vec![
                vec![0.5, 0.5],
                vec![0.1, 0.1],
                vec![0.3, 0.3],
            ],
        };

        assert_eq!(result.sorted_indices(), vec![0, 2, 1]);
        assert_eq!(result.top_k(2).len(), 2);
        assert_eq!(result.above_threshold(0.2).len(), 2);
    }
}
