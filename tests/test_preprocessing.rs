//! Integration test: Preprocessing pipeline end-to-end

use automl::preprocessing::{
    DataPreprocessor, PreprocessingConfig, ScalerType, ImputeStrategy,
};
use polars::prelude::*;

fn sample_df() -> DataFrame {
    df!(
        "age" => &[25.0, 30.0, 35.0, 40.0, 45.0, 50.0, 55.0, 60.0, 65.0, 70.0],
        "income" => &[30000.0, 45000.0, 55000.0, 70000.0, 80000.0, 90000.0, 100000.0, 110000.0, 120000.0, 130000.0],
        "score" => &[3.5, 4.0, 3.8, 4.5, 4.2, 4.8, 3.9, 4.7, 4.1, 4.6],
    )
    .unwrap()
}

#[test]
fn test_preprocessing_fit_transform() {
    let df = sample_df();
    let config = PreprocessingConfig::default().with_scaler(ScalerType::Standard);
    let mut preprocessor = DataPreprocessor::with_config(config);

    let result = preprocessor.fit_transform(&df);
    assert!(result.is_ok(), "fit_transform should succeed");

    let processed = result.unwrap();
    assert_eq!(processed.height(), 10, "row count should be preserved");
    assert!(processed.width() > 0, "should have columns");
}

#[test]
fn test_preprocessing_minmax_scaler() {
    let df = sample_df();
    let config = PreprocessingConfig::default().with_scaler(ScalerType::MinMax);
    let mut preprocessor = DataPreprocessor::with_config(config);

    let processed = preprocessor.fit_transform(&df).unwrap();
    assert_eq!(processed.height(), 10);
}

#[test]
fn test_preprocessing_robust_scaler() {
    let df = sample_df();
    let config = PreprocessingConfig::default().with_scaler(ScalerType::Robust);
    let mut preprocessor = DataPreprocessor::with_config(config);

    let processed = preprocessor.fit_transform(&df).unwrap();
    assert_eq!(processed.height(), 10);
}

#[test]
fn test_preprocessing_with_imputation() {
    let df = sample_df();
    let config = PreprocessingConfig::default()
        .with_scaler(ScalerType::Standard)
        .with_numeric_impute(ImputeStrategy::Mean);
    let mut preprocessor = DataPreprocessor::with_config(config);

    let processed = preprocessor.fit_transform(&df).unwrap();
    assert_eq!(processed.height(), 10);
}

#[test]
fn test_preprocessing_statistics() {
    let df = sample_df();
    let config = PreprocessingConfig::default().with_scaler(ScalerType::Standard);
    let mut preprocessor = DataPreprocessor::with_config(config);
    let _ = preprocessor.fit_transform(&df).unwrap();

    let stats = preprocessor.get_statistics();
    assert!(!stats.is_empty(), "statistics should not be empty after fit");
}

#[test]
fn test_preprocessing_performance_metrics() {
    let df = sample_df();
    let config = PreprocessingConfig::default().with_scaler(ScalerType::Standard);
    let mut preprocessor = DataPreprocessor::with_config(config);
    let _ = preprocessor.fit_transform(&df).unwrap();

    let metrics = preprocessor.get_performance_metrics();
    assert!(metrics.contains_key("total_samples_processed"));
}

#[test]
fn test_preprocessing_partial_fit() {
    let df = sample_df();
    let config = PreprocessingConfig::default().with_scaler(ScalerType::Standard);
    let mut preprocessor = DataPreprocessor::with_config(config);
    let _ = preprocessor.fit_transform(&df).unwrap();

    // partial_fit with new data
    let new_data = vec![
        vec![75.0, 140000.0, 4.9],
        vec![80.0, 150000.0, 5.0],
    ];
    let result = preprocessor.partial_fit(&new_data);
    assert!(result.is_ok(), "partial_fit should succeed");
}

#[test]
fn test_preprocessing_optimize_for_dataset() {
    let new_data = vec![
        vec![25.0, 30000.0, 3.5],
        vec![30.0, 45000.0, 4.0],
        vec![35.0, 55000.0, 3.8],
        vec![40.0, 70000.0, 4.5],
        vec![45.0, 80000.0, 4.2],
    ];

    let config = PreprocessingConfig::default();
    let mut preprocessor = DataPreprocessor::with_config(config);
    let result = preprocessor.optimize_for_dataset(&new_data);
    assert!(result.is_ok(), "optimize_for_dataset should succeed");
}
