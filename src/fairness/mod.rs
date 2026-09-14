//! Fairness Metrics and Bias Detection Module
//!
//! Provides fairness evaluation, bias detection, and mitigation tools
//! for ISO TR 24027 (Bias in AI Systems) compliance.
//!
//! # ISO Standards Coverage
//! - ISO/IEC TR 24027:2021 Clauses 6-8: Bias identification, measurement, mitigation
//! - ISO/IEC 42001:2023 Annex D: AI impact assessment

use crate::error::{AutoMLError, Result};
use ndarray::Array1;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Configuration for fairness evaluation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FairnessConfig {
    /// Names of protected/sensitive attributes
    pub protected_attributes: Vec<String>,
    /// Value considered as the favorable/positive label
    pub favorable_label: f64,
    /// Thresholds for fairness violation detection
    pub thresholds: FairnessThresholds,
}

impl Default for FairnessConfig {
    fn default() -> Self {
        Self {
            protected_attributes: Vec::new(),
            favorable_label: 1.0,
            thresholds: FairnessThresholds::default(),
        }
    }
}

/// Thresholds for determining fairness violations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FairnessThresholds {
    /// Minimum acceptable disparate impact ratio (typically 0.8, the "80% rule")
    pub disparate_impact_min: f64,
    /// Maximum acceptable demographic parity difference
    pub demographic_parity_max: f64,
    /// Maximum acceptable equalized odds difference
    pub equalized_odds_max: f64,
}

impl Default for FairnessThresholds {
    fn default() -> Self {
        Self {
            disparate_impact_min: 0.8,
            demographic_parity_max: 0.1,
            equalized_odds_max: 0.1,
        }
    }
}

/// Fairness metrics for a single protected group
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GroupFairnessMetric {
    /// Name of the protected attribute
    pub protected_attribute: String,
    /// Value of this group (e.g., "male", "female")
    pub group_value: String,
    /// Number of samples in this group
    pub group_size: usize,
    /// Selection rate (proportion receiving favorable outcome)
    pub selection_rate: f64,
    /// True positive rate (sensitivity/recall)
    pub true_positive_rate: f64,
    /// False positive rate
    pub false_positive_rate: f64,
    /// Positive predictive value (precision)
    pub positive_predictive_value: f64,
    /// Base rate (proportion of actual positives in this group)
    pub base_rate: f64,
}

/// A detected fairness violation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum FairnessViolation {
    /// Disparate impact ratio below threshold
    DisparateImpact {
        attribute: String,
        privileged_group: String,
        unprivileged_group: String,
        ratio: f64,
        threshold: f64,
    },
    /// Demographic parity difference above threshold
    DemographicParity {
        attribute: String,
        group_a: String,
        group_b: String,
        difference: f64,
        threshold: f64,
    },
    /// Equalized odds difference above threshold
    EqualizedOdds {
        attribute: String,
        group_a: String,
        group_b: String,
        tpr_difference: f64,
        fpr_difference: f64,
        threshold: f64,
    },
    /// Predictive parity difference above threshold
    PredictiveParity {
        attribute: String,
        group_a: String,
        group_b: String,
        ppv_difference: f64,
    },
}

/// Complete fairness evaluation report
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FairnessReport {
    /// Per-group metrics for each protected attribute
    pub group_metrics: Vec<GroupFairnessMetric>,
    /// Whether the model passes all fairness criteria
    pub overall_fair: bool,
    /// List of detected violations
    pub violations: Vec<FairnessViolation>,
    /// Disparate impact ratios per attribute
    pub disparate_impact: HashMap<String, f64>,
    /// Demographic parity differences per attribute
    pub demographic_parity: HashMap<String, f64>,
    /// Summary statistics
    pub summary: FairnessSummary,
}

/// Summary statistics for the fairness report
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FairnessSummary {
    pub total_samples: usize,
    pub num_protected_attributes: usize,
    pub num_groups: usize,
    pub num_violations: usize,
    pub worst_disparate_impact: f64,
    pub worst_demographic_parity: f64,
}

/// Bias scan result for pre-training data analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BiasReport {
    /// Class distribution overall
    pub class_distribution: HashMap<String, usize>,
    /// Class imbalance ratio (majority / minority)
    pub class_imbalance_ratio: f64,
    /// Per-group analysis
    pub group_analysis: Vec<GroupBiasAnalysis>,
    /// Detected issues
    pub issues: Vec<BiasIssue>,
}

/// Bias analysis for a single group
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GroupBiasAnalysis {
    pub attribute: String,
    pub group_value: String,
    pub group_size: usize,
    pub group_proportion: f64,
    pub positive_rate: f64,
    pub label_skew: f64,
}

/// A detected bias issue
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum BiasIssue {
    /// Severe class imbalance
    ClassImbalance { ratio: f64, threshold: f64 },
    /// Underrepresented group
    Underrepresentation {
        attribute: String,
        group: String,
        proportion: f64,
        min_proportion: f64,
    },
    /// Label distribution skew between groups
    LabelSkew {
        attribute: String,
        group_a: String,
        group_b: String,
        skew: f64,
    },
}

/// Main fairness evaluator
pub struct FairnessEvaluator {
    config: FairnessConfig,
}

impl FairnessEvaluator {
    /// Create a new fairness evaluator
    pub fn new(config: FairnessConfig) -> Self {
        Self { config }
    }

    /// Evaluate fairness of predictions with respect to protected attributes.
    ///
    /// # Arguments
    /// * `predictions` - Model predictions (0.0 or 1.0 for binary classification)
    /// * `actuals` - Ground truth labels
    /// * `protected_attrs` - Map of attribute name to array of group membership values
    pub fn evaluate(
        &self,
        predictions: &Array1<f64>,
        actuals: &Array1<f64>,
        protected_attrs: &HashMap<String, Vec<String>>,
    ) -> Result<FairnessReport> {
        if predictions.len() != actuals.len() {
            return Err(AutoMLError::FairnessError(
                "Predictions and actuals must have the same length".to_string(),
            ));
        }

        let n = predictions.len();
        let mut group_metrics = Vec::new();
        let mut violations = Vec::new();
        let mut disparate_impact_map = HashMap::new();
        let mut demographic_parity_map = HashMap::new();

        for (attr_name, attr_values) in protected_attrs {
            if attr_values.len() != n {
                return Err(AutoMLError::FairnessError(format!(
                    "Attribute '{}' has {} values, expected {}",
                    attr_name,
                    attr_values.len(),
                    n
                )));
            }

            // Get unique groups
            let mut groups: Vec<String> = attr_values.iter().cloned().collect();
            groups.sort();
            groups.dedup();

            let mut group_data: Vec<(String, GroupFairnessMetric)> = Vec::new();

            for group_val in &groups {
                let mask: Vec<bool> = attr_values.iter().map(|v| v == group_val).collect();
                let group_size = mask.iter().filter(|&&m| m).count();

                if group_size == 0 {
                    continue;
                }

                let group_preds: Vec<f64> = mask.iter().enumerate()
                    .filter(|(_, &m)| m)
                    .map(|(i, _)| predictions[i])
                    .collect();
                let group_actuals: Vec<f64> = mask.iter().enumerate()
                    .filter(|(_, &m)| m)
                    .map(|(i, _)| actuals[i])
                    .collect();

                let favorable = self.config.favorable_label;
                let selection_rate = group_preds.iter()
                    .filter(|&&p| (p - favorable).abs() < 1e-10)
                    .count() as f64 / group_size as f64;

                let base_rate = group_actuals.iter()
                    .filter(|&&a| (a - favorable).abs() < 1e-10)
                    .count() as f64 / group_size as f64;

                // Confusion matrix components
                let mut tp = 0usize;
                let mut fp = 0usize;
                let mut tn = 0usize;
                let mut fn_ = 0usize;

                for (p, a) in group_preds.iter().zip(group_actuals.iter()) {
                    let pred_pos = (*p - favorable).abs() < 1e-10;
                    let actual_pos = (*a - favorable).abs() < 1e-10;
                    match (pred_pos, actual_pos) {
                        (true, true) => tp += 1,
                        (true, false) => fp += 1,
                        (false, false) => tn += 1,
                        (false, true) => fn_ += 1,
                    }
                }

                let tpr = if tp + fn_ > 0 { tp as f64 / (tp + fn_) as f64 } else { 0.0 };
                let fpr = if fp + tn > 0 { fp as f64 / (fp + tn) as f64 } else { 0.0 };
                let ppv = if tp + fp > 0 { tp as f64 / (tp + fp) as f64 } else { 0.0 };

                let metric = GroupFairnessMetric {
                    protected_attribute: attr_name.clone(),
                    group_value: group_val.clone(),
                    group_size,
                    selection_rate,
                    true_positive_rate: tpr,
                    false_positive_rate: fpr,
                    positive_predictive_value: ppv,
                    base_rate,
                };

                group_data.push((group_val.clone(), metric.clone()));
                group_metrics.push(metric);
            }

            // Compute pairwise metrics between groups
            for i in 0..group_data.len() {
                for j in (i + 1)..group_data.len() {
                    let (ref name_a, ref metrics_a) = group_data[i];
                    let (ref name_b, ref metrics_b) = group_data[j];

                    // Disparate impact
                    let (privileged_rate, unprivileged_rate, priv_name, unpriv_name) =
                        if metrics_a.selection_rate >= metrics_b.selection_rate {
                            (metrics_a.selection_rate, metrics_b.selection_rate, name_a, name_b)
                        } else {
                            (metrics_b.selection_rate, metrics_a.selection_rate, name_b, name_a)
                        };

                    let di_ratio = if privileged_rate > 0.0 {
                        unprivileged_rate / privileged_rate
                    } else {
                        1.0
                    };

                    let pair_key = format!("{}:{}_{}", attr_name, name_a, name_b);
                    disparate_impact_map.insert(pair_key.clone(), di_ratio);

                    if di_ratio < self.config.thresholds.disparate_impact_min {
                        violations.push(FairnessViolation::DisparateImpact {
                            attribute: attr_name.clone(),
                            privileged_group: priv_name.clone(),
                            unprivileged_group: unpriv_name.clone(),
                            ratio: di_ratio,
                            threshold: self.config.thresholds.disparate_impact_min,
                        });
                    }

                    // Demographic parity
                    let dp_diff = (metrics_a.selection_rate - metrics_b.selection_rate).abs();
                    demographic_parity_map.insert(pair_key.clone(), dp_diff);

                    if dp_diff > self.config.thresholds.demographic_parity_max {
                        violations.push(FairnessViolation::DemographicParity {
                            attribute: attr_name.clone(),
                            group_a: name_a.clone(),
                            group_b: name_b.clone(),
                            difference: dp_diff,
                            threshold: self.config.thresholds.demographic_parity_max,
                        });
                    }

                    // Equalized odds
                    let tpr_diff = (metrics_a.true_positive_rate - metrics_b.true_positive_rate).abs();
                    let fpr_diff = (metrics_a.false_positive_rate - metrics_b.false_positive_rate).abs();
                    let eo_max = tpr_diff.max(fpr_diff);

                    if eo_max > self.config.thresholds.equalized_odds_max {
                        violations.push(FairnessViolation::EqualizedOdds {
                            attribute: attr_name.clone(),
                            group_a: name_a.clone(),
                            group_b: name_b.clone(),
                            tpr_difference: tpr_diff,
                            fpr_difference: fpr_diff,
                            threshold: self.config.thresholds.equalized_odds_max,
                        });
                    }

                    // Predictive parity
                    let ppv_diff = (metrics_a.positive_predictive_value - metrics_b.positive_predictive_value).abs();
                    if ppv_diff > 0.1 {
                        violations.push(FairnessViolation::PredictiveParity {
                            attribute: attr_name.clone(),
                            group_a: name_a.clone(),
                            group_b: name_b.clone(),
                            ppv_difference: ppv_diff,
                        });
                    }
                }
            }
        }

        let worst_di = disparate_impact_map.values().copied()
            .min_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .unwrap_or(1.0);
        let worst_dp = demographic_parity_map.values().copied()
            .max_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .unwrap_or(0.0);

        let overall_fair = violations.is_empty();

        Ok(FairnessReport {
            summary: FairnessSummary {
                total_samples: n,
                num_protected_attributes: protected_attrs.len(),
                num_groups: group_metrics.len(),
                num_violations: violations.len(),
                worst_disparate_impact: worst_di,
                worst_demographic_parity: worst_dp,
            },
            group_metrics,
            overall_fair,
            violations,
            disparate_impact: disparate_impact_map,
            demographic_parity: demographic_parity_map,
        })
    }

    /// Perform pre-training bias scan on the dataset
    pub fn bias_scan(
        &self,
        labels: &Array1<f64>,
        protected_attrs: &HashMap<String, Vec<String>>,
    ) -> Result<BiasReport> {
        let n = labels.len();
        let favorable = self.config.favorable_label;

        // Overall class distribution
        let positive_count = labels.iter().filter(|&&l| (l - favorable).abs() < 1e-10).count();
        let negative_count = n - positive_count;

        let mut class_distribution = HashMap::new();
        class_distribution.insert("positive".to_string(), positive_count);
        class_distribution.insert("negative".to_string(), negative_count);

        let majority = positive_count.max(negative_count) as f64;
        let minority = positive_count.min(negative_count).max(1) as f64;
        let class_imbalance_ratio = majority / minority;

        let mut group_analysis = Vec::new();
        let mut issues = Vec::new();

        // Check class imbalance
        if class_imbalance_ratio > 3.0 {
            issues.push(BiasIssue::ClassImbalance {
                ratio: class_imbalance_ratio,
                threshold: 3.0,
            });
        }

        let overall_positive_rate = positive_count as f64 / n as f64;

        for (attr_name, attr_values) in protected_attrs {
            if attr_values.len() != n {
                continue;
            }

            let mut groups: Vec<String> = attr_values.iter().cloned().collect();
            groups.sort();
            groups.dedup();

            let mut group_rates: Vec<(String, f64)> = Vec::new();

            for group_val in &groups {
                let mask: Vec<bool> = attr_values.iter().map(|v| v == group_val).collect();
                let group_size = mask.iter().filter(|&&m| m).count();
                let group_proportion = group_size as f64 / n as f64;

                let group_positive = mask.iter().enumerate()
                    .filter(|(_, &m)| m)
                    .filter(|(i, _)| (labels[*i] - favorable).abs() < 1e-10)
                    .count();

                let positive_rate = if group_size > 0 {
                    group_positive as f64 / group_size as f64
                } else {
                    0.0
                };

                let label_skew = positive_rate - overall_positive_rate;

                group_analysis.push(GroupBiasAnalysis {
                    attribute: attr_name.clone(),
                    group_value: group_val.clone(),
                    group_size,
                    group_proportion,
                    positive_rate,
                    label_skew,
                });

                group_rates.push((group_val.clone(), positive_rate));

                // Check underrepresentation
                if group_proportion < 0.05 && group_size < 30 {
                    issues.push(BiasIssue::Underrepresentation {
                        attribute: attr_name.clone(),
                        group: group_val.clone(),
                        proportion: group_proportion,
                        min_proportion: 0.05,
                    });
                }
            }

            // Check label skew between groups
            for i in 0..group_rates.len() {
                for j in (i + 1)..group_rates.len() {
                    let skew = (group_rates[i].1 - group_rates[j].1).abs();
                    if skew > 0.15 {
                        issues.push(BiasIssue::LabelSkew {
                            attribute: attr_name.clone(),
                            group_a: group_rates[i].0.clone(),
                            group_b: group_rates[j].0.clone(),
                            skew,
                        });
                    }
                }
            }
        }

        Ok(BiasReport {
            class_distribution,
            class_imbalance_ratio,
            group_analysis,
            issues,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_config() -> FairnessConfig {
        FairnessConfig {
            protected_attributes: vec!["gender".to_string()],
            favorable_label: 1.0,
            thresholds: FairnessThresholds::default(),
        }
    }

    #[test]
    fn test_fair_predictions() {
        let evaluator = FairnessEvaluator::new(sample_config());
        let predictions = Array1::from(vec![1.0, 0.0, 1.0, 0.0, 1.0, 0.0]);
        let actuals = Array1::from(vec![1.0, 0.0, 1.0, 0.0, 1.0, 0.0]);

        let mut attrs = HashMap::new();
        attrs.insert(
            "gender".to_string(),
            vec!["M", "M", "M", "F", "F", "F"].into_iter().map(String::from).collect(),
        );

        let report = evaluator.evaluate(&predictions, &actuals, &attrs).unwrap();
        assert_eq!(report.group_metrics.len(), 2);
    }

    #[test]
    fn test_biased_predictions() {
        let evaluator = FairnessEvaluator::new(FairnessConfig {
            thresholds: FairnessThresholds {
                disparate_impact_min: 0.8,
                demographic_parity_max: 0.1,
                equalized_odds_max: 0.1,
            },
            ..sample_config()
        });

        // Group A always gets positive, Group B always gets negative
        let predictions = Array1::from(vec![1.0, 1.0, 1.0, 0.0, 0.0, 0.0]);
        let actuals = Array1::from(vec![1.0, 1.0, 0.0, 1.0, 0.0, 0.0]);

        let mut attrs = HashMap::new();
        attrs.insert(
            "gender".to_string(),
            vec!["A", "A", "A", "B", "B", "B"].into_iter().map(String::from).collect(),
        );

        let report = evaluator.evaluate(&predictions, &actuals, &attrs).unwrap();
        assert!(!report.overall_fair);
        assert!(!report.violations.is_empty());
    }

    #[test]
    fn test_bias_scan() {
        let evaluator = FairnessEvaluator::new(sample_config());
        let labels = Array1::from(vec![1.0, 1.0, 1.0, 1.0, 0.0, 0.0]);

        let mut attrs = HashMap::new();
        attrs.insert(
            "gender".to_string(),
            vec!["M", "M", "M", "F", "F", "F"].into_iter().map(String::from).collect(),
        );

        let report = evaluator.bias_scan(&labels, &attrs).unwrap();
        assert_eq!(report.group_analysis.len(), 2);
        assert!(report.class_imbalance_ratio >= 1.0);
    }

    #[test]
    fn test_class_imbalance_detection() {
        let evaluator = FairnessEvaluator::new(sample_config());
        // 9 positives, 1 negative => ratio 9.0
        let labels = Array1::from(vec![1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0]);

        let attrs = HashMap::new();
        let report = evaluator.bias_scan(&labels, &attrs).unwrap();
        assert!(report.class_imbalance_ratio > 3.0);
        assert!(report.issues.iter().any(|i| matches!(i, BiasIssue::ClassImbalance { .. })));
    }

    #[test]
    fn test_mismatched_lengths() {
        let evaluator = FairnessEvaluator::new(sample_config());
        let predictions = Array1::from(vec![1.0, 0.0]);
        let actuals = Array1::from(vec![1.0, 0.0, 1.0]);
        let attrs = HashMap::new();

        let result = evaluator.evaluate(&predictions, &actuals, &attrs);
        assert!(result.is_err());
    }

    #[test]
    fn test_mismatched_attribute_length() {
        let evaluator = FairnessEvaluator::new(sample_config());
        let predictions = Array1::from(vec![1.0, 0.0, 1.0]);
        let actuals = Array1::from(vec![1.0, 0.0, 1.0]);

        let mut attrs = HashMap::new();
        attrs.insert("gender".to_string(), vec!["M".to_string(), "F".to_string()]); // Wrong length

        let result = evaluator.evaluate(&predictions, &actuals, &attrs);
        assert!(result.is_err());
    }

    #[test]
    fn test_evaluate_no_protected_attrs() {
        let evaluator = FairnessEvaluator::new(sample_config());
        let predictions = Array1::from(vec![1.0, 0.0, 1.0]);
        let actuals = Array1::from(vec![1.0, 0.0, 1.0]);
        let attrs = HashMap::new(); // No protected attributes

        let report = evaluator.evaluate(&predictions, &actuals, &attrs).unwrap();
        assert!(report.overall_fair);
        assert!(report.violations.is_empty());
        assert_eq!(report.group_metrics.len(), 0);
    }

    #[test]
    fn test_evaluate_single_group() {
        let evaluator = FairnessEvaluator::new(sample_config());
        let predictions = Array1::from(vec![1.0, 0.0, 1.0, 0.0]);
        let actuals = Array1::from(vec![1.0, 0.0, 1.0, 0.0]);

        let mut attrs = HashMap::new();
        attrs.insert("group".to_string(), vec!["A".to_string(); 4]);

        let report = evaluator.evaluate(&predictions, &actuals, &attrs).unwrap();
        assert!(report.overall_fair); // Single group => no disparities possible
        assert_eq!(report.group_metrics.len(), 1);
    }

    #[test]
    fn test_evaluate_three_groups() {
        let evaluator = FairnessEvaluator::new(sample_config());
        let predictions = Array1::from(vec![1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0]);
        let actuals = Array1::from(vec![1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0]);

        let mut attrs = HashMap::new();
        attrs.insert(
            "race".to_string(),
            vec!["A", "A", "A", "B", "B", "B", "C", "C", "C"]
                .into_iter().map(String::from).collect(),
        );

        let report = evaluator.evaluate(&predictions, &actuals, &attrs).unwrap();
        assert_eq!(report.group_metrics.len(), 3);
        // 3 pairwise comparisons: (A,B), (A,C), (B,C)
        assert_eq!(report.summary.num_groups, 3);
    }

    #[test]
    fn test_evaluate_all_positive_predictions() {
        let evaluator = FairnessEvaluator::new(sample_config());
        let predictions = Array1::from(vec![1.0, 1.0, 1.0, 1.0]);
        let actuals = Array1::from(vec![1.0, 1.0, 0.0, 0.0]);

        let mut attrs = HashMap::new();
        attrs.insert(
            "gender".to_string(),
            vec!["M", "M", "F", "F"].into_iter().map(String::from).collect(),
        );

        let report = evaluator.evaluate(&predictions, &actuals, &attrs).unwrap();
        // All get positive prediction => selection_rate = 1.0 for both groups => fair by disparate impact
        let di = report.disparate_impact.values().next().unwrap();
        assert_eq!(*di, 1.0);
    }

    #[test]
    fn test_evaluate_all_negative_predictions() {
        let evaluator = FairnessEvaluator::new(sample_config());
        let predictions = Array1::from(vec![0.0, 0.0, 0.0, 0.0]);
        let actuals = Array1::from(vec![1.0, 0.0, 1.0, 0.0]);

        let mut attrs = HashMap::new();
        attrs.insert(
            "gender".to_string(),
            vec!["M", "M", "F", "F"].into_iter().map(String::from).collect(),
        );

        let report = evaluator.evaluate(&predictions, &actuals, &attrs).unwrap();
        // No favorable outcomes for any group => disparate impact = 1.0 (0/0 case handled as 1.0)
        let di = report.disparate_impact.values().next().unwrap();
        assert_eq!(*di, 1.0);
    }

    #[test]
    fn test_evaluate_multiple_protected_attributes() {
        let evaluator = FairnessEvaluator::new(FairnessConfig {
            protected_attributes: vec!["gender".to_string(), "age_group".to_string()],
            favorable_label: 1.0,
            thresholds: FairnessThresholds::default(),
        });

        let predictions = Array1::from(vec![1.0, 0.0, 1.0, 0.0, 1.0, 0.0]);
        let actuals = Array1::from(vec![1.0, 0.0, 1.0, 0.0, 1.0, 0.0]);

        let mut attrs = HashMap::new();
        attrs.insert("gender".to_string(),
            vec!["M", "M", "M", "F", "F", "F"].into_iter().map(String::from).collect());
        attrs.insert("age_group".to_string(),
            vec!["young", "old", "young", "old", "young", "old"].into_iter().map(String::from).collect());

        let report = evaluator.evaluate(&predictions, &actuals, &attrs).unwrap();
        // Should have metrics for both attributes
        assert!(report.group_metrics.len() >= 4); // 2 groups x 2 attrs
        assert_eq!(report.summary.num_protected_attributes, 2);
    }

    #[test]
    fn test_bias_scan_no_protected_attrs() {
        let evaluator = FairnessEvaluator::new(sample_config());
        let labels = Array1::from(vec![1.0, 0.0, 1.0, 0.0]);
        let attrs = HashMap::new();

        let report = evaluator.bias_scan(&labels, &attrs).unwrap();
        assert_eq!(report.group_analysis.len(), 0);
        assert_eq!(report.class_imbalance_ratio, 1.0);
    }

    #[test]
    fn test_bias_scan_extreme_imbalance() {
        let evaluator = FairnessEvaluator::new(sample_config());
        // 99 positive, 1 negative
        let mut labels_vec = vec![1.0; 99];
        labels_vec.push(0.0);
        let labels = Array1::from(labels_vec);

        let attrs = HashMap::new();
        let report = evaluator.bias_scan(&labels, &attrs).unwrap();
        assert_eq!(report.class_imbalance_ratio, 99.0);
        assert!(report.issues.iter().any(|i| matches!(i, BiasIssue::ClassImbalance { .. })));
    }

    #[test]
    fn test_bias_scan_label_skew() {
        let evaluator = FairnessEvaluator::new(sample_config());
        // Group A: 90% positive, Group B: 10% positive => large skew
        let labels = Array1::from(vec![
            1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0,  // A: 9/10 positive
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0,  // B: 1/10 positive
        ]);

        let mut attrs = HashMap::new();
        attrs.insert("group".to_string(),
            vec!["A"; 10].into_iter().chain(vec!["B"; 10]).map(String::from).collect());

        let report = evaluator.bias_scan(&labels, &attrs).unwrap();
        assert!(report.issues.iter().any(|i| matches!(i, BiasIssue::LabelSkew { .. })));
    }

    #[test]
    fn test_bias_scan_underrepresentation() {
        let evaluator = FairnessEvaluator::new(sample_config());
        // Group A: 97 samples, Group B: 3 samples (< 5% and < 30)
        let mut labels_vec = vec![1.0; 50];
        labels_vec.extend(vec![0.0; 50]);
        let labels = Array1::from(labels_vec);

        let mut group_vals = vec!["A"; 97];
        group_vals.extend(vec!["B"; 3]);
        let mut attrs = HashMap::new();
        attrs.insert("group".to_string(), group_vals.into_iter().map(String::from).collect());

        let report = evaluator.bias_scan(&labels, &attrs).unwrap();
        assert!(report.issues.iter().any(|i| matches!(i, BiasIssue::Underrepresentation { .. })));
    }

    #[test]
    fn test_bias_scan_balanced_classes() {
        let evaluator = FairnessEvaluator::new(sample_config());
        let labels = Array1::from(vec![1.0, 0.0, 1.0, 0.0, 1.0, 0.0]);
        let attrs = HashMap::new();

        let report = evaluator.bias_scan(&labels, &attrs).unwrap();
        assert_eq!(report.class_imbalance_ratio, 1.0);
        assert!(report.issues.is_empty() || !report.issues.iter().any(|i| matches!(i, BiasIssue::ClassImbalance { .. })));
    }

    #[test]
    fn test_fairness_summary_fields() {
        let evaluator = FairnessEvaluator::new(sample_config());
        let predictions = Array1::from(vec![1.0, 1.0, 1.0, 0.0, 0.0, 0.0]);
        let actuals = Array1::from(vec![1.0, 1.0, 0.0, 1.0, 0.0, 0.0]);

        let mut attrs = HashMap::new();
        attrs.insert("gender".to_string(),
            vec!["A", "A", "A", "B", "B", "B"].into_iter().map(String::from).collect());

        let report = evaluator.evaluate(&predictions, &actuals, &attrs).unwrap();
        assert_eq!(report.summary.total_samples, 6);
        assert_eq!(report.summary.num_protected_attributes, 1);
        assert_eq!(report.summary.num_groups, 2);
    }

    #[test]
    fn test_default_config() {
        let config = FairnessConfig::default();
        assert!(config.protected_attributes.is_empty());
        assert_eq!(config.favorable_label, 1.0);
        assert_eq!(config.thresholds.disparate_impact_min, 0.8);
        assert_eq!(config.thresholds.demographic_parity_max, 0.1);
        assert_eq!(config.thresholds.equalized_odds_max, 0.1);
    }
}
