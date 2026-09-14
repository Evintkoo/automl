//! Integration test: Training pipeline end-to-end

use automl::training::{
    TrainEngine, TrainingConfig, TaskType, ModelType, EnsembleStrategy,
};
use polars::prelude::*;

fn classification_df() -> DataFrame {
    df!(
        "f1" => &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0,
                   1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5, 10.5],
        "f2" => &[10.0, 9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0,
                   9.5, 8.5, 7.5, 6.5, 5.5, 4.5, 3.5, 2.5, 1.5, 0.5],
        "f3" => &[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0,
                   0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95, 1.05],
        "target" => &[0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                      0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    )
    .unwrap()
}

fn regression_df() -> DataFrame {
    df!(
        "x1" => &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0,
                   11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0],
        "x2" => &[2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0, 20.0,
                   22.0, 24.0, 26.0, 28.0, 30.0, 32.0, 34.0, 36.0, 38.0, 40.0],
        "target" => &[3.0, 6.0, 9.0, 12.0, 15.0, 18.0, 21.0, 24.0, 27.0, 30.0,
                      33.0, 36.0, 39.0, 42.0, 45.0, 48.0, 51.0, 54.0, 57.0, 60.0]
    )
    .unwrap()
}

#[test]
fn test_train_decision_tree_classification() {
    let df = classification_df();
    let config = TrainingConfig::new(TaskType::BinaryClassification, "target")
        .with_model(ModelType::DecisionTree);
    let mut engine = TrainEngine::new(config);
    let result = engine.fit(&df);
    assert!(result.is_ok(), "Decision tree training should succeed: {:?}", result.err());

    let metrics = engine.metrics();
    assert!(metrics.is_some(), "should have metrics after training");
}

#[test]
fn test_train_random_forest_classification() {
    let df = classification_df();
    let config = TrainingConfig::new(TaskType::BinaryClassification, "target")
        .with_model(ModelType::RandomForest);
    let mut engine = TrainEngine::new(config);
    let result = engine.fit(&df);
    assert!(result.is_ok(), "Random forest training should succeed: {:?}", result.err());
}

#[test]
fn test_train_knn_classification() {
    let df = classification_df();
    let config = TrainingConfig::new(TaskType::BinaryClassification, "target")
        .with_model(ModelType::KNN);
    let mut engine = TrainEngine::new(config);
    let result = engine.fit(&df);
    assert!(result.is_ok(), "KNN training should succeed: {:?}", result.err());
}

#[test]
fn test_train_naive_bayes_classification() {
    let df = classification_df();
    let config = TrainingConfig::new(TaskType::BinaryClassification, "target")
        .with_model(ModelType::NaiveBayes);
    let mut engine = TrainEngine::new(config);
    let result = engine.fit(&df);
    assert!(result.is_ok(), "Naive Bayes training should succeed: {:?}", result.err());
}

#[test]
fn test_train_svm_classification() {
    let df = classification_df();
    let config = TrainingConfig::new(TaskType::BinaryClassification, "target")
        .with_model(ModelType::SVM);
    let mut engine = TrainEngine::new(config);
    let result = engine.fit(&df);
    assert!(result.is_ok(), "SVM training should succeed: {:?}", result.err());
}

#[test]
fn test_train_gradient_boosting_classification() {
    let df = classification_df();
    let config = TrainingConfig::new(TaskType::BinaryClassification, "target")
        .with_model(ModelType::GradientBoosting);
    let mut engine = TrainEngine::new(config);
    let result = engine.fit(&df);
    assert!(result.is_ok(), "Gradient boosting training should succeed: {:?}", result.err());
}

#[test]
fn test_train_decision_tree_regression() {
    let df = regression_df();
    let config = TrainingConfig::new(TaskType::Regression, "target")
        .with_model(ModelType::DecisionTree);
    let mut engine = TrainEngine::new(config);
    let result = engine.fit(&df);
    assert!(result.is_ok(), "Decision tree regression should succeed: {:?}", result.err());
}

#[test]
fn test_train_random_forest_regression() {
    let df = regression_df();
    let config = TrainingConfig::new(TaskType::Regression, "target")
        .with_model(ModelType::RandomForest);
    let mut engine = TrainEngine::new(config);
    let result = engine.fit(&df);
    assert!(result.is_ok(), "Random forest regression should succeed: {:?}", result.err());
}

#[test]
fn test_prediction_after_training() {
    let df = classification_df();
    let config = TrainingConfig::new(TaskType::BinaryClassification, "target")
        .with_model(ModelType::DecisionTree);
    let mut engine = TrainEngine::new(config);
    engine.fit(&df).unwrap();

    let predictions = engine.predict(&df);
    assert!(predictions.is_ok(), "prediction should succeed");
    let preds = predictions.unwrap();
    assert_eq!(preds.len(), 20, "should have 20 predictions");
}

#[test]
fn test_generate_report() {
    let df = classification_df();
    let config = TrainingConfig::new(TaskType::BinaryClassification, "target")
        .with_model(ModelType::DecisionTree);
    let mut engine = TrainEngine::new(config);
    engine.fit(&df).unwrap();

    let report = engine.generate_report();
    assert!(!report.is_empty(), "report should not be empty");
    assert!(report.contains("AutoML"), "report should have header");
}

#[test]
fn test_feature_selection() {
    let df = classification_df();
    let config = TrainingConfig::new(TaskType::BinaryClassification, "target")
        .with_model(ModelType::DecisionTree);
    let mut engine = TrainEngine::new(config);
    engine.fit(&df).unwrap();

    let features = engine.select_features(&df, "target", 2);
    assert_eq!(features.len(), 2, "should select 2 features");
}

#[test]
fn test_performance_comparison() {
    let df = classification_df();
    let config = TrainingConfig::new(TaskType::BinaryClassification, "target")
        .with_model(ModelType::DecisionTree);
    let mut engine = TrainEngine::new(config);
    engine.fit(&df).unwrap();

    let comparisons = engine.get_performance_comparison();
    assert!(!comparisons.is_empty(), "should have at least one comparison");
}

#[test]
fn test_train_ensemble() {
    let df = classification_df();
    let config = TrainingConfig::new(TaskType::BinaryClassification, "target")
        .with_model(ModelType::DecisionTree);
    let mut engine = TrainEngine::new(config);

    let result = engine.train_ensemble(
        &df,
        "target",
        &[ModelType::DecisionTree, ModelType::KNN, ModelType::NaiveBayes],
        EnsembleStrategy::Voting,
    );
    assert!(result.is_ok(), "ensemble training should succeed: {:?}", result.err());
    let ensemble = result.unwrap();
    assert_eq!(ensemble.models.len(), 3, "should have 3 models in ensemble");
}

#[test]
fn test_epoch_history_empty_for_non_sgd() {
    let df = classification_df();
    let config = TrainingConfig::new(TaskType::BinaryClassification, "target")
        .with_model(ModelType::DecisionTree);
    let mut engine = TrainEngine::new(config);
    engine.fit(&df).unwrap();
    assert!(engine.epoch_history().is_empty(), "decision tree should have no epoch history");
}

#[test]
fn test_epoch_history_populated_for_sgd() {
    let df = regression_df();
    let config = TrainingConfig::new(TaskType::Regression, "target")
        .with_model(ModelType::SGD);
    let mut engine = TrainEngine::new(config);
    engine.fit(&df).unwrap();
    let history = engine.epoch_history();
    assert!(!history.is_empty(), "SGD should produce epoch history");
    assert_eq!(history[0].epoch, 0);
    for rec in history {
        assert!(rec.train_loss.is_finite(), "train_loss should be finite");
    }
}

#[test]
fn test_tree_nodes_decision_tree() {
    let df = classification_df();
    let config = TrainingConfig::new(TaskType::BinaryClassification, "target")
        .with_model(ModelType::DecisionTree);
    let mut engine = TrainEngine::new(config);
    engine.fit(&df).unwrap();
    let nodes = engine.tree_nodes();
    assert!(nodes.is_some(), "decision tree should serialize nodes");
    let nodes = nodes.unwrap();
    assert!(!nodes.is_empty());
    assert_eq!(nodes[0].parent_id, None, "root has no parent");
    assert!(nodes.iter().any(|n| n.is_leaf), "at least one leaf exists");
}

#[test]
fn test_tree_nodes_random_forest() {
    let df = classification_df();
    let config = TrainingConfig::new(TaskType::BinaryClassification, "target")
        .with_model(ModelType::RandomForest);
    let mut engine = TrainEngine::new(config);
    engine.fit(&df).unwrap();
    let nodes = engine.tree_nodes();
    assert!(nodes.is_some(), "random forest should serialize first tree's nodes");
    let nodes = nodes.unwrap();
    assert!(!nodes.is_empty());
    assert_eq!(nodes[0].parent_id, None, "root has no parent");
}

#[test]
fn test_tree_nodes_non_tree_returns_none() {
    let df = classification_df();
    let config = TrainingConfig::new(TaskType::BinaryClassification, "target")
        .with_model(ModelType::SGD);
    let mut engine = TrainEngine::new(config);
    engine.fit(&df).unwrap();
    assert!(engine.tree_nodes().is_none(), "SGD has no tree nodes");
}
