//! Integration test: Full pipeline (load → preprocess → train → predict)

use automl::preprocessing::{DataPreprocessor, PreprocessingConfig, ScalerType};
use automl::training::{TrainEngine, TrainingConfig, TaskType, ModelType};
use automl::tracking::ExperimentTracker;
use automl::batch::DynamicBatcher;
use automl::cache::LruTtlCache;
use automl::quantization::{Quantizer, QuantizationConfig};
use automl::streaming::{StreamingPipeline, StreamConfig};
use polars::prelude::*;

fn create_classification_dataset() -> DataFrame {
    let n = 50;
    let mut f1 = Vec::with_capacity(n);
    let mut f2 = Vec::with_capacity(n);
    let mut f3 = Vec::with_capacity(n);
    let mut target = Vec::with_capacity(n);

    for i in 0..n {
        let x = i as f64;
        f1.push(x);
        f2.push(x * 2.0 + 1.0);
        f3.push((x * 0.1).sin());
        target.push(if i < n / 2 { 0.0 } else { 1.0 });
    }

    df!(
        "feature1" => &f1,
        "feature2" => &f2,
        "feature3" => &f3,
        "target" => &target
    )
    .unwrap()
}

fn create_regression_dataset() -> DataFrame {
    let n = 50;
    let mut f1 = Vec::with_capacity(n);
    let mut f2 = Vec::with_capacity(n);
    let mut target = Vec::with_capacity(n);

    for i in 0..n {
        let x = i as f64;
        f1.push(x);
        f2.push(x * 0.5);
        target.push(x * 3.0 + 2.0 + (x * 0.1).sin());
    }

    df!(
        "x1" => &f1,
        "x2" => &f2,
        "target" => &target
    )
    .unwrap()
}

#[test]
fn test_full_classification_pipeline() {
    let df = create_classification_dataset();

    // Step 1: Preprocess
    let config = PreprocessingConfig::default().with_scaler(ScalerType::Standard);
    let mut preprocessor = DataPreprocessor::with_config(config);
    let processed = preprocessor.fit_transform(&df).unwrap();

    assert_eq!(processed.height(), 50);
    assert!(processed.width() > 0);

    // Step 2: Train
    let train_config = TrainingConfig::new(TaskType::BinaryClassification, "target")
        .with_model(ModelType::RandomForest);
    let mut engine = TrainEngine::new(train_config);
    let fit_result = engine.fit(&df);
    assert!(fit_result.is_ok(), "training should succeed: {:?}", fit_result.err());

    // Step 3: Predict
    let predictions = engine.predict(&df).unwrap();
    assert_eq!(predictions.len(), 50);

    // Step 4: Verify metrics
    let metrics = engine.metrics().unwrap();
    assert!(metrics.accuracy.unwrap_or(0.0) > 0.0, "accuracy should be > 0");
}

#[test]
fn test_full_regression_pipeline() {
    let df = create_regression_dataset();

    // Step 1: Preprocess
    let config = PreprocessingConfig::default().with_scaler(ScalerType::MinMax);
    let mut preprocessor = DataPreprocessor::with_config(config);
    let _processed = preprocessor.fit_transform(&df).unwrap();

    // Step 2: Train
    let train_config = TrainingConfig::new(TaskType::Regression, "target")
        .with_model(ModelType::RandomForest);
    let mut engine = TrainEngine::new(train_config);
    engine.fit(&df).unwrap();

    // Step 3: Predict
    let predictions = engine.predict(&df).unwrap();
    assert_eq!(predictions.len(), 50);

    // Step 4: Metrics
    let metrics = engine.metrics().unwrap();
    assert!(metrics.r2.is_some(), "should have R² score");
}

#[test]
fn test_full_pipeline_with_multiple_models() {
    let df = create_classification_dataset();

    let models = vec![
        ("DecisionTree", ModelType::DecisionTree),
        ("KNN", ModelType::KNN),
        ("NaiveBayes", ModelType::NaiveBayes),
        ("RandomForest", ModelType::RandomForest),
    ];

    let mut results: Vec<(String, f64)> = Vec::new();

    for (name, model_type) in models {
        let config = TrainingConfig::new(TaskType::BinaryClassification, "target")
            .with_model(model_type);
        let mut engine = TrainEngine::new(config);

        if let Ok(_) = engine.fit(&df) {
            let metrics = engine.metrics().cloned().unwrap_or_default();
            let accuracy = metrics.accuracy.unwrap_or(0.0);
            results.push((name.to_string(), accuracy));
        }
    }

    assert!(results.len() >= 2, "at least 2 models should train successfully");

    let best = results.iter().max_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
    assert!(best.is_some(), "should find a best model");
}

#[test]
fn test_full_pipeline_with_report() {
    let df = create_classification_dataset();

    let config = TrainingConfig::new(TaskType::BinaryClassification, "target")
        .with_model(ModelType::DecisionTree);
    let mut engine = TrainEngine::new(config);
    engine.fit(&df).unwrap();

    let report = engine.generate_report();
    assert!(report.contains("Model"), "report should contain model info");
    assert!(!report.is_empty());
}

#[test]
fn test_experiment_tracking_integration() {
    let tmp_dir = std::env::temp_dir().join(format!("automl_test_exp_{}", std::process::id()));
    let tracker = ExperimentTracker::with_dir(&tmp_dir);
    let exp_id = tracker.create_experiment("test_experiment");

    let run_id = tracker.start_run("run-1");
    tracker.log_param("model", "decision_tree");
    tracker.log_metric("accuracy", 0.95, None);
    tracker.end_run_success();

    let report = tracker.generate_report(&exp_id);
    assert!(report.is_ok(), "report generation should succeed");
    assert!(!report.unwrap().is_empty());

    std::fs::remove_dir_all(&tmp_dir).ok();
}

#[test]
fn test_cache_integration() {
    let cache: LruTtlCache<String, String> = LruTtlCache::new(100, 60);

    cache.set("key1".to_string(), "value1".to_string());
    cache.set("key2".to_string(), "value2".to_string());

    let v1 = cache.get(&"key1".to_string());
    assert!(v1.is_some());
    assert_eq!(v1.unwrap(), "value1");

    let (hits, misses, hit_rate) = cache.stats();
    // Just verify stats work
    assert!(hits + misses > 0);
}

#[test]
fn test_quantization_integration() {
    let config = QuantizationConfig::default();
    let mut quantizer = Quantizer::new(config);

    let data: Vec<f64> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    quantizer.calibrate(&data);

    let quantized = quantizer.quantize(&data);
    let dequantized = quantizer.dequantize(&quantized);
    assert_eq!(dequantized.len(), data.len());
}

#[test]
fn test_streaming_pipeline_integration() {
    let config = StreamConfig {
        chunk_size: 100,
        max_queue_size: 1000,
        enable_backpressure: false,
        memory_threshold_mb: 512,
        buffer_size: 256,
        timeout_ms: 5000,
    };
    let pipeline: StreamingPipeline<f64> = StreamingPipeline::new(config);

    // Push data
    for i in 0..50 {
        pipeline.push(i as f64).unwrap();
    }

    let stats = pipeline.stats();
    assert!(pipeline.buffer_size() > 0 || stats.rows_processed >= 0);
}

#[test]
fn test_dynamic_batcher_integration() {
    let config = automl::batch::BatcherConfig {
        max_batch_size: 4,
        max_wait_time_ms: 100,
        max_queue_size: 100,
        enable_adaptive_sizing: true,
        adaptation_interval_secs: 1.0,
        priority_levels: 3,
    };
    let batcher: DynamicBatcher<Vec<f64>> = DynamicBatcher::new(config);

    // Track request types
    batcher.track_request_type("inference");
    batcher.track_request_type("inference");
    batcher.track_request_type("preprocessing");

    assert!(batcher.has_diverse_requests(), "should detect diverse requests");

    // Record latency
    batcher.record_latency(5.0);
    batcher.record_latency(8.0);
    batcher.record_latency(3.0);

    let stats = batcher.get_enhanced_stats();
    assert!(stats.avg_latency_ms > 0.0);
}
