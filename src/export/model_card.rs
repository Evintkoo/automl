//! Model Card generation for ISO TR 24028 (Trustworthiness in AI) compliance.
//!
//! Auto-generates standardized documentation for trained models including
//! model details, intended use, metrics, training data summary, and limitations.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Complete model card following the "Model Cards for Model Reporting" framework
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelCard {
    /// Core model details
    pub model_details: ModelDetails,
    /// Intended use and out-of-scope uses
    pub intended_use: IntendedUse,
    /// Performance metrics
    pub metrics: ModelMetricsCard,
    /// Summary of training data
    pub training_data_summary: TrainingDataSummary,
    /// Ethical considerations
    pub ethical_considerations: Vec<String>,
    /// Known limitations
    pub limitations: Vec<String>,
    /// When this card was generated
    pub generated_at: DateTime<Utc>,
}

/// Core model identification and technical details
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelDetails {
    pub name: String,
    pub version: String,
    pub model_type: String,
    pub task_type: String,
    pub framework: String,
    pub hyperparameters: serde_json::Value,
    pub training_date: DateTime<Utc>,
}

/// Intended and out-of-scope uses
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IntendedUse {
    pub primary_use: String,
    pub primary_users: Vec<String>,
    pub out_of_scope: Vec<String>,
}

/// Performance metrics section
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelMetricsCard {
    pub primary_metric: String,
    pub primary_value: f64,
    pub all_metrics: HashMap<String, f64>,
    pub evaluation_method: String,
    pub evaluation_dataset_size: Option<usize>,
}

/// Summary of training data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingDataSummary {
    pub num_samples: usize,
    pub num_features: usize,
    pub feature_names: Vec<String>,
    pub target_name: Option<String>,
    pub target_distribution: HashMap<String, usize>,
}

impl ModelCard {
    /// Generate a model card with sensible defaults from training results
    pub fn generate(
        model_name: &str,
        model_type: &str,
        task_type: &str,
        metrics: HashMap<String, f64>,
        num_samples: usize,
        num_features: usize,
        feature_names: Vec<String>,
        hyperparameters: serde_json::Value,
    ) -> Self {
        let primary_metric = if task_type == "classification" {
            "accuracy"
        } else {
            "r2_score"
        };
        let primary_value = metrics.get(primary_metric).copied().unwrap_or(0.0);

        Self {
            model_details: ModelDetails {
                name: model_name.to_string(),
                version: "1.0.0".to_string(),
                model_type: model_type.to_string(),
                task_type: task_type.to_string(),
                framework: format!("AutoML v{}", env!("CARGO_PKG_VERSION")),
                hyperparameters,
                training_date: Utc::now(),
            },
            intended_use: IntendedUse {
                primary_use: format!("{} using tabular data", task_type),
                primary_users: vec!["Data scientists".to_string(), "ML engineers".to_string()],
                out_of_scope: vec![
                    "Safety-critical decisions without human oversight".to_string(),
                    "Applications requiring real-time guarantees not validated".to_string(),
                ],
            },
            metrics: ModelMetricsCard {
                primary_metric: primary_metric.to_string(),
                primary_value,
                all_metrics: metrics,
                evaluation_method: "cross-validation".to_string(),
                evaluation_dataset_size: Some(num_samples),
            },
            training_data_summary: TrainingDataSummary {
                num_samples,
                num_features,
                feature_names,
                target_name: None,
                target_distribution: HashMap::new(),
            },
            ethical_considerations: vec![
                "Model performance may vary across demographic subgroups".to_string(),
                "Evaluate fairness metrics before deployment in sensitive contexts".to_string(),
                "Monitor for data drift that could degrade model behavior".to_string(),
            ],
            limitations: vec![
                "Trained on a specific dataset; may not generalize to all populations".to_string(),
                "Does not handle missing data at inference time unless preprocessed".to_string(),
                format!("Evaluated using {}; other metrics may differ", primary_metric),
            ],
            generated_at: Utc::now(),
        }
    }

    /// Render the model card as Markdown
    pub fn to_markdown(&self) -> String {
        let mut md = String::new();

        md.push_str(&format!("# Model Card: {}\n\n", self.model_details.name));
        md.push_str(&format!("*Generated: {}*\n\n", self.generated_at.format("%Y-%m-%d %H:%M UTC")));

        md.push_str("## Model Details\n\n");
        md.push_str(&format!("- **Type:** {}\n", self.model_details.model_type));
        md.push_str(&format!("- **Task:** {}\n", self.model_details.task_type));
        md.push_str(&format!("- **Version:** {}\n", self.model_details.version));
        md.push_str(&format!("- **Framework:** {}\n", self.model_details.framework));
        md.push_str(&format!("- **Training Date:** {}\n\n", self.model_details.training_date.format("%Y-%m-%d")));

        md.push_str("## Intended Use\n\n");
        md.push_str(&format!("- **Primary Use:** {}\n", self.intended_use.primary_use));
        md.push_str("- **Users:** ");
        md.push_str(&self.intended_use.primary_users.join(", "));
        md.push_str("\n- **Out of Scope:**\n");
        for item in &self.intended_use.out_of_scope {
            md.push_str(&format!("  - {}\n", item));
        }
        md.push('\n');

        md.push_str("## Metrics\n\n");
        md.push_str(&format!("- **Primary:** {} = {:.4}\n", self.metrics.primary_metric, self.metrics.primary_value));
        md.push_str(&format!("- **Evaluation:** {}\n", self.metrics.evaluation_method));
        if let Some(size) = self.metrics.evaluation_dataset_size {
            md.push_str(&format!("- **Dataset Size:** {} samples\n", size));
        }
        md.push_str("\n| Metric | Value |\n|--------|-------|\n");
        let mut sorted_metrics: Vec<_> = self.metrics.all_metrics.iter().collect();
        sorted_metrics.sort_by_key(|(k, _)| k.to_string());
        for (k, v) in sorted_metrics {
            md.push_str(&format!("| {} | {:.4} |\n", k, v));
        }
        md.push('\n');

        md.push_str("## Training Data\n\n");
        md.push_str(&format!("- **Samples:** {}\n", self.training_data_summary.num_samples));
        md.push_str(&format!("- **Features:** {}\n\n", self.training_data_summary.num_features));

        md.push_str("## Ethical Considerations\n\n");
        for item in &self.ethical_considerations {
            md.push_str(&format!("- {}\n", item));
        }
        md.push('\n');

        md.push_str("## Limitations\n\n");
        for item in &self.limitations {
            md.push_str(&format!("- {}\n", item));
        }

        md
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_model_card_generation() {
        let mut metrics = HashMap::new();
        metrics.insert("accuracy".to_string(), 0.95);
        metrics.insert("f1_score".to_string(), 0.94);

        let card = ModelCard::generate(
            "test_model",
            "RandomForest",
            "classification",
            metrics,
            1000,
            10,
            vec!["f1".to_string(), "f2".to_string()],
            serde_json::json!({"n_estimators": 100}),
        );

        assert_eq!(card.model_details.name, "test_model");
        assert_eq!(card.metrics.primary_metric, "accuracy");
        assert_eq!(card.metrics.primary_value, 0.95);
    }

    #[test]
    fn test_model_card_markdown() {
        let mut metrics = HashMap::new();
        metrics.insert("accuracy".to_string(), 0.92);

        let card = ModelCard::generate(
            "iris_classifier",
            "DecisionTree",
            "classification",
            metrics,
            150,
            4,
            vec!["sepal_length".to_string()],
            serde_json::json!({}),
        );

        let md = card.to_markdown();
        assert!(md.contains("# Model Card: iris_classifier"));
        assert!(md.contains("DecisionTree"));
        assert!(md.contains("0.9200"));
    }

    #[test]
    fn test_model_card_regression() {
        let mut metrics = HashMap::new();
        metrics.insert("r2_score".to_string(), 0.87);
        metrics.insert("rmse".to_string(), 3.14);

        let card = ModelCard::generate(
            "house_price_predictor",
            "GradientBoosting",
            "regression",
            metrics,
            5000,
            20,
            vec!["sq_ft".to_string(), "bedrooms".to_string()],
            serde_json::json!({"n_estimators": 200, "learning_rate": 0.1}),
        );

        assert_eq!(card.metrics.primary_metric, "r2_score");
        assert_eq!(card.metrics.primary_value, 0.87);
        assert_eq!(card.model_details.task_type, "regression");
        assert_eq!(card.training_data_summary.num_features, 20);

        let md = card.to_markdown();
        assert!(md.contains("r2_score"));
        assert!(md.contains("regression"));
    }

    #[test]
    fn test_model_card_empty_metrics() {
        let card = ModelCard::generate(
            "empty_model",
            "LinearRegression",
            "regression",
            HashMap::new(),
            0,
            0,
            vec![],
            serde_json::json!({}),
        );

        // Primary value should default to 0.0 when metric not found
        assert_eq!(card.metrics.primary_value, 0.0);
        assert_eq!(card.training_data_summary.num_samples, 0);
        assert!(card.training_data_summary.feature_names.is_empty());
    }

    #[test]
    fn test_model_card_markdown_has_all_sections() {
        let mut metrics = HashMap::new();
        metrics.insert("accuracy".to_string(), 0.95);

        let card = ModelCard::generate(
            "test", "RF", "classification", metrics, 100, 5, vec![], serde_json::json!({}),
        );
        let md = card.to_markdown();

        assert!(md.contains("## Model Details"));
        assert!(md.contains("## Intended Use"));
        assert!(md.contains("## Metrics"));
        assert!(md.contains("## Training Data"));
        assert!(md.contains("## Ethical Considerations"));
        assert!(md.contains("## Limitations"));
    }

    #[test]
    fn test_model_card_framework_contains_version() {
        let card = ModelCard::generate(
            "test", "RF", "classification", HashMap::new(), 0, 0, vec![], serde_json::json!({}),
        );
        assert!(card.model_details.framework.starts_with("AutoML v"));
    }

    #[test]
    fn test_model_card_serialization() {
        let mut metrics = HashMap::new();
        metrics.insert("accuracy".to_string(), 0.95);

        let card = ModelCard::generate(
            "test", "RF", "classification", metrics, 100, 5, vec![], serde_json::json!({}),
        );

        let json = serde_json::to_string(&card).unwrap();
        let deserialized: ModelCard = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized.model_details.name, "test");
        assert_eq!(deserialized.metrics.primary_value, 0.95);
    }
}
