//! Local explanations (SHAP-like feature contributions)

use crate::error::Result;
use ndarray::{Array1, Array2};
use rand::prelude::*;
use serde::{Deserialize, Serialize};

/// Feature contribution to a prediction
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeatureContribution {
    /// Feature index
    pub feature_index: usize,
    /// Feature name (if provided)
    pub feature_name: Option<String>,
    /// Feature value for this instance
    pub feature_value: f64,
    /// Contribution to prediction (SHAP value)
    pub contribution: f64,
}

/// Local explanation for a single prediction
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LocalExplanation {
    /// Instance index
    pub instance_index: usize,
    /// Base value (expected prediction)
    pub base_value: f64,
    /// Actual prediction
    pub prediction: f64,
    /// Feature contributions
    pub contributions: Vec<FeatureContribution>,
}

impl LocalExplanation {
    /// Get sum of contributions
    pub fn sum_contributions(&self) -> f64 {
        self.contributions.iter().map(|c| c.contribution).sum()
    }

    /// Get sorted contributions (by absolute value, descending)
    pub fn sorted_contributions(&self) -> Vec<&FeatureContribution> {
        let mut sorted: Vec<&FeatureContribution> = self.contributions.iter().collect();
        sorted.sort_by(|a, b| {
            b.contribution
                .abs()
                .partial_cmp(&a.contribution.abs())
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        sorted
    }

    /// Get top k contributors
    pub fn top_k_contributors(&self, k: usize) -> Vec<&FeatureContribution> {
        self.sorted_contributions().into_iter().take(k).collect()
    }

    /// Get positive contributors
    pub fn positive_contributors(&self) -> Vec<&FeatureContribution> {
        self.contributions
            .iter()
            .filter(|c| c.contribution > 0.0)
            .collect()
    }

    /// Get negative contributors
    pub fn negative_contributors(&self) -> Vec<&FeatureContribution> {
        self.contributions
            .iter()
            .filter(|c| c.contribution < 0.0)
            .collect()
    }
}

/// Local explainer using Kernel SHAP-like approach
pub struct LocalExplainer<F>
where
    F: Fn(&Array2<f64>) -> Result<Array1<f64>>,
{
    /// Prediction function
    predict_fn: F,
    /// Background dataset for computing expectations
    background: Array2<f64>,
    /// Number of samples for Monte Carlo approximation
    n_samples: usize,
    /// Random seed
    seed: Option<u64>,
    /// Feature names
    feature_names: Option<Vec<String>>,
}

impl<F> LocalExplainer<F>
where
    F: Fn(&Array2<f64>) -> Result<Array1<f64>>,
{
    /// Create new local explainer
    pub fn new(predict_fn: F, background: Array2<f64>) -> Self {
        Self {
            predict_fn,
            background,
            n_samples: 100,
            seed: None,
            feature_names: None,
        }
    }

    /// Set number of samples for approximation
    pub fn with_n_samples(mut self, n: usize) -> Self {
        self.n_samples = n.max(10);
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

    /// Explain a single instance
    pub fn explain(&self, instance: &Array1<f64>) -> Result<LocalExplanation> {
        let base_value = self.compute_base_value()?;
        self.explain_instance(instance, 0, base_value)
    }

    /// Explain multiple instances
    pub fn explain_batch(&self, instances: &Array2<f64>) -> Result<Vec<LocalExplanation>> {
        // The background dataset and predict_fn are the same for every
        // instance in the batch, so the expected prediction over the
        // background is invariant across the whole batch — compute it once
        // instead of once per instance (this used to re-run predict_fn over
        // the full background set for every single row).
        let base_value = self.compute_base_value()?;
        instances
            .rows()
            .into_iter()
            .enumerate()
            .map(|(idx, row)| self.explain_instance(&row.to_owned(), idx, base_value))
            .collect()
    }

    /// Expected prediction over the background dataset — the SHAP "base value".
    fn compute_base_value(&self) -> Result<f64> {
        let bg_preds = (self.predict_fn)(&self.background)?;
        Ok(bg_preds.mean().unwrap_or(0.0))
    }

    /// Internal method to explain a single instance
    fn explain_instance(
        &self,
        instance: &Array1<f64>,
        instance_index: usize,
        base_value: f64,
    ) -> Result<LocalExplanation> {
        let n_features = instance.len();
        let mut rng = match self.seed {
            Some(seed) => StdRng::seed_from_u64(seed + instance_index as u64),
            None => StdRng::from_entropy(),
        };

        // Compute actual prediction
        let instance_2d = instance.clone().insert_axis(ndarray::Axis(0));
        let prediction = (self.predict_fn)(&instance_2d)?[0];

        // Compute SHAP values using sampling-based approach
        let mut contributions = vec![0.0; n_features];

        for _ in 0..self.n_samples {
            // Random permutation
            let mut perm: Vec<usize> = (0..n_features).collect();
            perm.shuffle(&mut rng);

            // Random background sample
            let bg_idx = rng.gen_range(0..self.background.nrows());
            let bg_sample = self.background.row(bg_idx);

            // Iterate through permutation
            let mut x_before = bg_sample.to_owned();
            let mut pred_before = (self.predict_fn)(
                &x_before.clone().insert_axis(ndarray::Axis(0)),
            )?[0];

            for &feature_idx in &perm {
                // Add feature to coalition
                x_before[feature_idx] = instance[feature_idx];

                // Get new prediction
                let pred_after = (self.predict_fn)(
                    &x_before.clone().insert_axis(ndarray::Axis(0)),
                )?[0];

                // Marginal contribution
                contributions[feature_idx] += pred_after - pred_before;
                pred_before = pred_after;
            }
        }

        // Average contributions
        for c in &mut contributions {
            *c /= self.n_samples as f64;
        }

        // Build contribution objects
        let feature_contributions: Vec<FeatureContribution> = contributions
            .into_iter()
            .enumerate()
            .map(|(idx, contrib)| {
                let name = self
                    .feature_names
                    .as_ref()
                    .and_then(|n| n.get(idx).cloned());
                FeatureContribution {
                    feature_index: idx,
                    feature_name: name,
                    feature_value: instance[idx],
                    contribution: contrib,
                }
            })
            .collect();

        Ok(LocalExplanation {
            instance_index,
            base_value,
            prediction,
            contributions: feature_contributions,
        })
    }

    /// Compute global SHAP values (average absolute SHAP across instances)
    pub fn global_importance(&self, x: &Array2<f64>) -> Result<Vec<(usize, f64)>> {
        let explanations = self.explain_batch(x)?;
        let n_features = x.ncols();

        let mut importance = vec![0.0; n_features];

        for exp in &explanations {
            for contrib in &exp.contributions {
                importance[contrib.feature_index] += contrib.contribution.abs();
            }
        }

        // Average
        for imp in &mut importance {
            *imp /= explanations.len() as f64;
        }

        // Sort by importance
        let mut indexed: Vec<(usize, f64)> = importance.into_iter().enumerate().collect();
        indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        Ok(indexed)
    }
}

/// Summary of SHAP values across many instances
#[allow(dead_code)]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ShapSummary {
    /// Feature names
    pub feature_names: Option<Vec<String>>,
    /// Mean absolute SHAP values per feature
    pub mean_abs_shap: Vec<f64>,
    /// Mean SHAP values per feature
    pub mean_shap: Vec<f64>,
    /// Standard deviation of SHAP values per feature
    pub std_shap: Vec<f64>,
    /// Min SHAP values per feature
    pub min_shap: Vec<f64>,
    /// Max SHAP values per feature
    pub max_shap: Vec<f64>,
}

#[allow(dead_code)]
impl ShapSummary {
    /// Create from a list of explanations
    pub fn from_explanations(explanations: &[LocalExplanation]) -> Self {
        if explanations.is_empty() {
            return Self {
                feature_names: None,
                mean_abs_shap: vec![],
                mean_shap: vec![],
                std_shap: vec![],
                min_shap: vec![],
                max_shap: vec![],
            };
        }

        let n_features = explanations[0].contributions.len();
        let n_instances = explanations.len() as f64;

        let feature_names = explanations[0]
            .contributions
            .iter()
            .map(|c| c.feature_name.clone())
            .collect::<Option<Vec<_>>>();

        let mut mean_abs = vec![0.0; n_features];
        let mut mean = vec![0.0; n_features];
        let mut min_vals = vec![f64::INFINITY; n_features];
        let mut max_vals = vec![f64::NEG_INFINITY; n_features];
        let mut sum_sq = vec![0.0; n_features];

        for exp in explanations {
            for c in &exp.contributions {
                let idx = c.feature_index;
                mean_abs[idx] += c.contribution.abs();
                mean[idx] += c.contribution;
                min_vals[idx] = min_vals[idx].min(c.contribution);
                max_vals[idx] = max_vals[idx].max(c.contribution);
            }
        }

        for i in 0..n_features {
            mean_abs[i] /= n_instances;
            mean[i] /= n_instances;
        }

        // Second pass for standard deviation
        for exp in explanations {
            for c in &exp.contributions {
                let idx = c.feature_index;
                sum_sq[idx] += (c.contribution - mean[idx]).powi(2);
            }
        }

        let std: Vec<f64> = sum_sq
            .into_iter()
            .map(|s| (s / n_instances).sqrt())
            .collect();

        Self {
            feature_names,
            mean_abs_shap: mean_abs,
            mean_shap: mean,
            std_shap: std,
            min_shap: min_vals,
            max_shap: max_vals,
        }
    }

    /// Get feature ranking by mean absolute SHAP
    pub fn feature_ranking(&self) -> Vec<(usize, f64)> {
        let mut indexed: Vec<(usize, f64)> = self
            .mean_abs_shap
            .iter()
            .copied()
            .enumerate()
            .collect();
        indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        indexed
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_local_explanation_basic() {
        // Simple additive model
        let predict_fn = |x: &Array2<f64>| -> Result<Array1<f64>> {
            let preds: Vec<f64> = x
                .rows()
                .into_iter()
                .map(|row| row[0] + 2.0 * row[1] + 3.0 * row[2])
                .collect();
            Ok(Array1::from_vec(preds))
        };

        let background = Array2::from_shape_vec(
            (10, 3),
            vec![
                0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 3.0, 3.0, 3.0, 4.0, 4.0, 4.0, 5.0,
                5.0, 5.0, 6.0, 6.0, 6.0, 7.0, 7.0, 7.0, 8.0, 8.0, 8.0, 9.0, 9.0, 9.0,
            ],
        )
        .unwrap();

        let explainer = LocalExplainer::new(predict_fn, background)
            .with_n_samples(50)
            .with_seed(42);

        let instance = array![1.0, 2.0, 3.0];
        let explanation = explainer.explain(&instance).unwrap();

        assert_eq!(explanation.contributions.len(), 3);
        
        // Prediction should equal sum of contributions + base value (with some tolerance for sampling variance)
        let contrib_sum = explanation.sum_contributions();
        let expected = explanation.base_value + contrib_sum;
        assert!((expected - explanation.prediction).abs() < 5.0);
    }

    #[test]
    fn test_sorted_contributions() {
        let explanation = LocalExplanation {
            instance_index: 0,
            base_value: 0.0,
            prediction: 6.0,
            contributions: vec![
                FeatureContribution {
                    feature_index: 0,
                    feature_name: Some("a".to_string()),
                    feature_value: 1.0,
                    contribution: 1.0,
                },
                FeatureContribution {
                    feature_index: 1,
                    feature_name: Some("b".to_string()),
                    feature_value: 2.0,
                    contribution: -3.0,
                },
                FeatureContribution {
                    feature_index: 2,
                    feature_name: Some("c".to_string()),
                    feature_value: 3.0,
                    contribution: 2.0,
                },
            ],
        };

        let sorted = explanation.sorted_contributions();
        assert_eq!(sorted[0].feature_index, 1); // -3.0 has highest abs
        assert_eq!(sorted[1].feature_index, 2); // 2.0
        assert_eq!(sorted[2].feature_index, 0); // 1.0
    }

    #[test]
    fn test_shap_summary() {
        let explanations = vec![
            LocalExplanation {
                instance_index: 0,
                base_value: 0.0,
                prediction: 1.0,
                contributions: vec![
                    FeatureContribution {
                        feature_index: 0,
                        feature_name: None,
                        feature_value: 1.0,
                        contribution: 0.5,
                    },
                    FeatureContribution {
                        feature_index: 1,
                        feature_name: None,
                        feature_value: 2.0,
                        contribution: 0.5,
                    },
                ],
            },
            LocalExplanation {
                instance_index: 1,
                base_value: 0.0,
                prediction: 2.0,
                contributions: vec![
                    FeatureContribution {
                        feature_index: 0,
                        feature_name: None,
                        feature_value: 3.0,
                        contribution: 1.5,
                    },
                    FeatureContribution {
                        feature_index: 1,
                        feature_name: None,
                        feature_value: 4.0,
                        contribution: -0.5,
                    },
                ],
            },
        ];

        let summary = ShapSummary::from_explanations(&explanations);
        assert_eq!(summary.mean_abs_shap.len(), 2);
        assert_eq!(summary.mean_abs_shap[0], 1.0); // (0.5 + 1.5) / 2
        assert_eq!(summary.mean_abs_shap[1], 0.5); // (0.5 + 0.5) / 2
    }
}
