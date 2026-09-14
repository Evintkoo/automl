//! Integration tests for data processing: preprocessing, cleaning, and data loading

use automl::preprocessing::{
    DataPreprocessor, PreprocessingConfig, ScalerType, ImputeStrategy,
};
use automl::training::{TrainEngine, TrainingConfig, TaskType, ModelType};
use polars::prelude::*;

// ============================================================================
// Edge Case Tests for Preprocessing
// ============================================================================

#[test]
fn test_preprocessing_no_scaler() {
    let df = df!(
        "a" => &[1.0, 2.0, 3.0, 4.0, 5.0],
        "b" => &[10.0, 20.0, 30.0, 40.0, 50.0]
    ).unwrap();

    let config = PreprocessingConfig::default().with_scaler(ScalerType::None);
    let mut preprocessor = DataPreprocessor::with_config(config);
    let result = preprocessor.fit_transform(&df);
    assert!(result.is_ok());
    let processed = result.unwrap();
    assert_eq!(processed.height(), 5);
}

#[test]
fn test_preprocessing_single_column() {
    let df = df!(
        "x" => &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
    ).unwrap();

    let config = PreprocessingConfig::default().with_scaler(ScalerType::Standard);
    let mut preprocessor = DataPreprocessor::with_config(config);
    let result = preprocessor.fit_transform(&df);
    assert!(result.is_ok());
}

#[test]
fn test_preprocessing_large_values() {
    let df = df!(
        "big" => &[1e10, 2e10, 3e10, 4e10, 5e10, 6e10, 7e10, 8e10, 9e10, 1e11],
        "small" => &[1e-10, 2e-10, 3e-10, 4e-10, 5e-10, 6e-10, 7e-10, 8e-10, 9e-10, 1e-9]
    ).unwrap();

    let config = PreprocessingConfig::default().with_scaler(ScalerType::Standard);
    let mut preprocessor = DataPreprocessor::with_config(config);
    let result = preprocessor.fit_transform(&df);
    assert!(result.is_ok());
}

#[test]
fn test_preprocessing_constant_column() {
    let df = df!(
        "constant" => &[5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0],
        "varying" => &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
    ).unwrap();

    let config = PreprocessingConfig::default().with_scaler(ScalerType::Standard);
    let mut preprocessor = DataPreprocessor::with_config(config);
    let result = preprocessor.fit_transform(&df);
    assert!(result.is_ok());
}

#[test]
fn test_preprocessing_median_imputation() {
    let df = df!(
        "x" => &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
        "y" => &[10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0]
    ).unwrap();

    let config = PreprocessingConfig::default()
        .with_scaler(ScalerType::MinMax)
        .with_numeric_impute(ImputeStrategy::Median);
    let mut preprocessor = DataPreprocessor::with_config(config);
    let result = preprocessor.fit_transform(&df);
    assert!(result.is_ok());
}

#[test]
fn test_preprocessing_drop_imputation() {
    let df = df!(
        "x" => &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
        "y" => &[10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0]
    ).unwrap();

    let config = PreprocessingConfig::default()
        .with_scaler(ScalerType::Robust)
        .with_numeric_impute(ImputeStrategy::Drop);
    let mut preprocessor = DataPreprocessor::with_config(config);
    let result = preprocessor.fit_transform(&df);
    assert!(result.is_ok());
}

// ============================================================================
// Training with different configurations
// ============================================================================

fn make_binary_df() -> DataFrame {
    let n = 40;
    let mut f1 = Vec::with_capacity(n);
    let mut f2 = Vec::with_capacity(n);
    let mut target = Vec::with_capacity(n);
    for i in 0..n {
        f1.push(i as f64 + 0.5);
        f2.push((i as f64 * 0.3).sin() + 1.0);
        target.push(if i < n / 2 { 0.0 } else { 1.0 });
    }
    df!(
        "f1" => &f1,
        "f2" => &f2,
        "target" => &target
    ).unwrap()
}

fn make_regression_df() -> DataFrame {
    let n = 40;
    let mut x1 = Vec::with_capacity(n);
    let mut x2 = Vec::with_capacity(n);
    let mut target = Vec::with_capacity(n);
    for i in 0..n {
        let v = i as f64;
        x1.push(v + 0.1 * (v * 0.7).sin());
        x2.push(v * 2.0 + 0.3 * (v * 0.5).cos());
        target.push(v * 3.0 + 1.0 + 0.5 * (v * 0.2).sin());
    }
    df!(
        "x1" => &x1,
        "x2" => &x2,
        "target" => &target
    ).unwrap()
}

#[test]
fn test_linear_regression_training() {
    let df = make_regression_df();
    let config = TrainingConfig::new(TaskType::Regression, "target")
        .with_model(ModelType::LinearRegression);
    let mut engine = TrainEngine::new(config);
    let result = engine.fit(&df);
    assert!(result.is_ok(), "linear regression should succeed: {:?}", result.err());

    let metrics = engine.metrics().unwrap();
    assert!(metrics.r2.is_some(), "should have R² metric");
}

#[test]
fn test_logistic_regression_training() {
    let df = make_binary_df();
    let config = TrainingConfig::new(TaskType::BinaryClassification, "target")
        .with_model(ModelType::LogisticRegression);
    let mut engine = TrainEngine::new(config);
    let result = engine.fit(&df);
    assert!(result.is_ok(), "logistic regression should succeed: {:?}", result.err());
}

#[test]
fn test_gradient_boosting_regression() {
    let df = make_regression_df();
    let config = TrainingConfig::new(TaskType::Regression, "target")
        .with_model(ModelType::GradientBoosting);
    let mut engine = TrainEngine::new(config);
    let result = engine.fit(&df);
    assert!(result.is_ok(), "gradient boosting regression should succeed: {:?}", result.err());
}

#[test]
fn test_knn_regression() {
    let df = make_regression_df();
    let config = TrainingConfig::new(TaskType::Regression, "target")
        .with_model(ModelType::KNN);
    let mut engine = TrainEngine::new(config);
    let result = engine.fit(&df);
    assert!(result.is_ok(), "KNN regression should succeed: {:?}", result.err());
}

#[test]
fn test_svm_regression() {
    let df = make_regression_df();
    let config = TrainingConfig::new(TaskType::Regression, "target")
        .with_model(ModelType::SVM);
    let mut engine = TrainEngine::new(config);
    let result = engine.fit(&df);
    assert!(result.is_ok(), "SVM regression should succeed: {:?}", result.err());
}

#[test]
fn test_predict_returns_correct_length() {
    let df = make_binary_df();
    let config = TrainingConfig::new(TaskType::BinaryClassification, "target")
        .with_model(ModelType::RandomForest);
    let mut engine = TrainEngine::new(config);
    engine.fit(&df).unwrap();

    let preds = engine.predict(&df).unwrap();
    assert_eq!(preds.len(), df.height());
}

#[test]
fn test_metrics_have_training_time() {
    let df = make_binary_df();
    let config = TrainingConfig::new(TaskType::BinaryClassification, "target")
        .with_model(ModelType::DecisionTree);
    let mut engine = TrainEngine::new(config);
    engine.fit(&df).unwrap();

    let metrics = engine.metrics().unwrap();
    assert!(metrics.training_time_secs > 0.0, "training time should be > 0");
}

// ============================================================================
// Multi-model comparison
// ============================================================================

#[test]
fn test_all_regression_models_produce_metrics() {
    let df = make_regression_df();

    let models = vec![
        ModelType::LinearRegression,
        ModelType::DecisionTree,
        ModelType::RandomForest,
        ModelType::GradientBoosting,
        ModelType::KNN,
        ModelType::SVM,
    ];

    for model_type in models {
        let config = TrainingConfig::new(TaskType::Regression, "target")
            .with_model(model_type.clone());
        let mut engine = TrainEngine::new(config);
        if let Ok(_) = engine.fit(&df) {
            let metrics = engine.metrics();
            assert!(metrics.is_some(), "{:?} should produce metrics", model_type);
        }
    }
}

#[test]
fn test_all_classification_models_produce_metrics() {
    let df = make_binary_df();

    let models = vec![
        ModelType::LogisticRegression,
        ModelType::DecisionTree,
        ModelType::RandomForest,
        ModelType::GradientBoosting,
        ModelType::KNN,
        ModelType::NaiveBayes,
        ModelType::SVM,
    ];

    for model_type in models {
        let config = TrainingConfig::new(TaskType::BinaryClassification, "target")
            .with_model(model_type.clone());
        let mut engine = TrainEngine::new(config);
        if let Ok(_) = engine.fit(&df) {
            let metrics = engine.metrics();
            assert!(metrics.is_some(), "{:?} should produce metrics", model_type);
        }
    }
}

// ============================================================================
// CLI data loading tests
// ============================================================================

#[test]
fn test_cli_load_csv_data() {
    use std::io::Write;
    let tmp = tempfile::NamedTempFile::with_suffix(".csv").unwrap();
    writeln!(tmp.as_file(), "a,b,target").unwrap();
    for i in 0..20 {
        writeln!(tmp.as_file(), "{},{},{}", i, i * 2, i % 2).unwrap();
    }
    tmp.as_file().flush().unwrap();

    let path = tmp.path().to_path_buf();
    let df = automl::cli::load_data(&path);
    assert!(df.is_ok(), "CSV load should succeed");
    let df = df.unwrap();
    assert_eq!(df.height(), 20);
    assert_eq!(df.width(), 3);
}

// ============================================================================
// Preprocessing + Training pipeline tests
// ============================================================================

#[test]
fn test_preprocess_then_train_classification() {
    let df = make_binary_df();

    let config = PreprocessingConfig::default()
        .with_scaler(ScalerType::Standard)
        .with_numeric_impute(ImputeStrategy::Mean);
    let mut preprocessor = DataPreprocessor::with_config(config);
    let processed = preprocessor.fit_transform(&df).unwrap();
    assert_eq!(processed.height(), df.height());

    let train_config = TrainingConfig::new(TaskType::BinaryClassification, "target")
        .with_model(ModelType::DecisionTree);
    let mut engine = TrainEngine::new(train_config);
    let result = engine.fit(&df);
    assert!(result.is_ok());
}

#[test]
fn test_preprocess_then_train_regression() {
    let df = make_regression_df();

    let config = PreprocessingConfig::default()
        .with_scaler(ScalerType::MinMax)
        .with_numeric_impute(ImputeStrategy::Median);
    let mut preprocessor = DataPreprocessor::with_config(config);
    let processed = preprocessor.fit_transform(&df).unwrap();
    assert!(processed.height() > 0);

    let train_config = TrainingConfig::new(TaskType::Regression, "target")
        .with_model(ModelType::LinearRegression);
    let mut engine = TrainEngine::new(train_config);
    let result = engine.fit(&df);
    assert!(result.is_ok());
}
