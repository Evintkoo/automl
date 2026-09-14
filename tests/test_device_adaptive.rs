//! Integration test: Device optimizer and adaptive modules

use automl::device::DeviceOptimizer;
use automl::adaptive::{
    ConfigOptimizer, DatasetCharacteristics, DatasetSize, ProcessingMode,
    AdaptiveHyperparameterOptimizer, PerformanceTracker,
    SearchSpaceConfig,
};
use automl::precision::{MixedPrecisionManager, MixedPrecisionConfig};
use automl::memory::{MemoryMonitor, AdaptiveChunkProcessor};
use automl::monitoring::{AlertManager, AlertLevel, Alert, AlertCondition, SystemMonitor};
use std::collections::HashMap;

// ============================================================================
// Device Optimizer Tests
// ============================================================================

#[test]
fn test_device_optimizer_creation() {
    let optimizer = DeviceOptimizer::new();
    let hw = DeviceOptimizer::detect_hardware();
    assert!(hw.cpu.cores > 0, "should detect CPU cores");
    assert!(hw.memory.total_bytes > 0, "should detect memory");
}

#[test]
fn test_device_optimizer_system_info() {
    let optimizer = DeviceOptimizer::new();
    let info = optimizer.get_system_info();
    assert!(info.contains_key("cpu"), "system info should have cpu key");
    assert!(info.contains_key("memory"), "system info should have memory key");
}

#[test]
fn test_device_optimizer_optimal_configs() {
    let optimizer = DeviceOptimizer::new();
    let configs = optimizer.auto_tune_configs();
    assert!(configs.batch.max_batch_size > 0);
    assert!(configs.preprocess.num_workers > 0);
}

#[test]
fn test_device_optimizer_environment_detection() {
    let env = DeviceOptimizer::detect_environment();
    assert!(!env.is_empty(), "should detect environment");
}

// ============================================================================
// Adaptive Preprocessing Tests
// ============================================================================

#[test]
fn test_config_optimizer_analyze() {
    let data = vec![
        vec![1.0, 100.0, 0.5],
        vec![2.0, 200.0, 0.6],
        vec![3.0, 300.0, 0.7],
        vec![4.0, 400.0, 0.8],
        vec![5.0, 500.0, 0.9],
    ];

    let characteristics = ConfigOptimizer::analyze_dataset(&data, 5, 3);
    assert_eq!(characteristics.n_rows, 5);
    assert_eq!(characteristics.n_cols, 3);
    assert!(matches!(characteristics.dataset_size, DatasetSize::Tiny));
}

#[test]
fn test_adaptive_preprocessor_config() {
    let characteristics = DatasetCharacteristics {
        n_rows: 50000,
        n_cols: 20,
        n_numeric: 15,
        n_categorical: 5,
        n_datetime: 0,
        n_text: 0,
        sparsity: 0.1,
        skewness_avg: 0.5,
        missing_ratio: 0.05,
        outlier_ratio: 0.02,
        high_cardinality_count: 1,
        memory_mb: 50.0,
        dataset_size: DatasetSize::Medium,
    };

    use automl::adaptive::AdaptivePreprocessorConfig;
    let strategy = AdaptivePreprocessorConfig::determine_strategy(&characteristics);
    assert!(
        matches!(strategy, ProcessingMode::Speed | ProcessingMode::Balanced | ProcessingMode::Quality | ProcessingMode::Memory | ProcessingMode::LargeScale),
        "should return a valid processing mode"
    );

    let norm = AdaptivePreprocessorConfig::select_normalization(&characteristics);
    assert!(!norm.is_empty(), "should select a normalization method");

    let chunk_size = AdaptivePreprocessorConfig::suggest_chunk_size(&characteristics);
    assert!(chunk_size > 0, "chunk size should be positive");

    let workers = AdaptivePreprocessorConfig::suggest_num_workers(&characteristics);
    assert!(workers > 0, "worker count should be positive");
}

#[test]
fn test_config_optimizer_suggestions() {
    let characteristics = DatasetCharacteristics {
        n_rows: 1_000_000,
        n_cols: 100,
        n_numeric: 80,
        n_categorical: 20,
        n_datetime: 0,
        n_text: 0,
        sparsity: 0.3,
        skewness_avg: 1.5,
        missing_ratio: 0.1,
        outlier_ratio: 0.05,
        high_cardinality_count: 5,
        memory_mb: 2000.0,
        dataset_size: DatasetSize::Large,
    };

    let suggestions = ConfigOptimizer::get_memory_optimization_suggestions(&characteristics);
    assert!(!suggestions.is_empty(), "should have suggestions for large dataset");

    let time_est = ConfigOptimizer::estimate_processing_time(&characteristics);
    assert!(!time_est.is_empty(), "should estimate processing time");
}

// ============================================================================
// Adaptive Hyperopt Tests
// ============================================================================

#[test]
fn test_performance_tracker() {
    let mut tracker = PerformanceTracker::new(5);
    let mut params = HashMap::new();
    params.insert("lr".to_string(), 0.01);

    tracker.add_result(0.8, params.clone());
    tracker.add_result(0.82, params.clone());
    tracker.add_result(0.83, params.clone());

    assert!(!tracker.is_converged(), "should not be converged with few results");
}

#[test]
fn test_adaptive_hyperopt_optimizer() {
    let mut search_space = HashMap::new();
    search_space.insert(
        "param1".to_string(),
        SearchSpaceConfig {
            name: "param1".to_string(),
            param_type: "float".to_string(),
            low: Some(0.0),
            high: Some(1.0),
            choices: None,
            log_scale: false,
        },
    );

    let config = automl::adaptive::AdaptiveHyperoptConfig {
        n_trials: 10,
        timeout_secs: 30.0,
        n_startup: 3,
        adaptation_interval: 5,
    };
    let mut optimizer = AdaptiveHyperparameterOptimizer::new(config);

    let result = optimizer.optimize(
        |params| {
            let x = params.get("param1").copied().unwrap_or(0.5);
            -(x - 0.3).powi(2) // Maximize: best at x=0.3
        },
        search_space,
    );

    assert!(result.n_trials > 0, "should have run trials");
    assert!(result.best_params.contains_key("param1"));
}

// ============================================================================
// Mixed Precision Tests
// ============================================================================

#[test]
fn test_mixed_precision_manager() {
    let config = MixedPrecisionConfig::default();
    let mut manager = MixedPrecisionManager::new(config);

    let data = vec![1.0_f64, 2.0, 3.0, 4.0, 5.0];
    let optimized = manager.optimize_precision(&data).unwrap();
    assert_eq!(optimized.len(), 5);

    let restored = manager.restore_precision(&optimized);
    assert_eq!(restored.len(), 5);
}

#[test]
fn test_mixed_precision_loss_scaling() {
    let config = MixedPrecisionConfig {
        enabled: true,
        dynamic_loss_scaling: true,
        ..MixedPrecisionConfig::default()
    };
    let manager = MixedPrecisionManager::new(config);

    let scaled = manager.scale_loss(1.0);
    assert!(scaled >= 1.0, "scaled loss should be >= original");
}

#[test]
fn test_mixed_precision_overflow_detection() {
    assert!(!MixedPrecisionManager::check_overflow(&[1.0, 2.0, 3.0]));
    assert!(MixedPrecisionManager::check_overflow(&[1.0, f64::NAN, 3.0]));
    assert!(MixedPrecisionManager::check_overflow(&[1.0, f64::INFINITY, 3.0]));
}

#[test]
fn test_mixed_precision_auto_select() {
    let config = MixedPrecisionConfig::default();
    let manager = MixedPrecisionManager::new(config);

    let narrow_data = vec![0.1, 0.2, 0.3, 0.4, 0.5];
    let _mode = manager.select_precision(&narrow_data);
}

// ============================================================================
// Memory Monitor Tests
// ============================================================================

#[test]
fn test_memory_monitor() {
    let mut monitor = MemoryMonitor::new(70.0, 85.0);
    let usage = monitor.get_memory_usage();
    assert!(usage >= 0.0, "memory usage should be non-negative");

    let (total, available, _pct) = monitor.get_system_memory();
    assert!(total > 0.0, "total memory should be positive");
    assert!(available >= 0.0, "available memory should be non-negative");

    let _ = monitor.update_memory();
    let history = monitor.get_memory_history();
    assert!(!history.is_empty(), "should have history after update");
}

#[test]
fn test_adaptive_chunk_processor() {
    let mut processor = AdaptiveChunkProcessor::new(100, 10000, 70.0);
    let chunk_size = processor.get_adaptive_chunk_size(50000);
    assert!(chunk_size >= 100, "chunk size should be at least min");
    assert!(chunk_size <= 10000, "chunk size should be at most max");
}

// ============================================================================
// Alert Manager Tests
// ============================================================================

#[test]
fn test_alert_manager() {
    let mut manager = AlertManager::new();

    manager.add_alert(Alert {
        name: "high_cpu".to_string(),
        metric_name: "cpu_usage".to_string(),
        condition: AlertCondition::GreaterThan(90.0),
        level: AlertLevel::Warning,
        message: "CPU usage is high".to_string(),
        cooldown_secs: 60,
        last_triggered: None,
        triggered_count: 0,
    });

    let mut metrics = HashMap::new();
    metrics.insert("cpu_usage".to_string(), 95.0);

    let triggered = manager.check_alerts(&metrics);
    assert!(!triggered.is_empty(), "alert should trigger for high CPU");

    // Below threshold
    metrics.insert("cpu_usage".to_string(), 50.0);
    let triggered = manager.check_alerts(&metrics);
    assert!(triggered.is_empty(), "alert should not trigger for normal CPU");
}

// ============================================================================
// System Monitor Tests
// ============================================================================

#[test]
fn test_system_monitor() {
    let monitor = SystemMonitor::new(10.0);
    let metrics = SystemMonitor::collect_metrics();
    assert!(metrics.cpu_usage_pct >= 0.0);
    assert!(metrics.memory_used_mb > 0.0);
}
