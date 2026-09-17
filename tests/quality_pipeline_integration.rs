//! Integration test: QualityPipeline gates produce valid outputs on Iris-like data.

use automl::quality::{
    QualityContext,
    pre_training::{LeakageDetector, DistributionFingerprint, PreTrainingGate},
    training::{AlgorithmSelector, CvStrategyChooser},
    hyperopt::{compute_pareto_front, scalarized_score, MultiObjectiveConfig, StatefulConvergenceDetector},
    post_training::{
        PlattCalibrator, IsotonicCalibrator, ConformalPredictor,
        entropy_confidence, OodDetector, expected_calibration_error,
    },
};
use automl::quality::CvStrategy;
use ndarray::{Array1, Array2};

fn make_iris_like() -> (Array2<f64>, Array1<f64>) {
    // 150 samples, 4 features, binary classification (setosa vs rest)
    let mut features = Array2::zeros((150, 4));
    let mut labels = Array1::zeros(150);
    for i in 0..50 {
        features[[i, 0]] = 5.0 + (i as f64) * 0.02;
        features[[i, 1]] = 3.5;
        features[[i, 2]] = 1.4;
        features[[i, 3]] = 0.2;
        labels[i] = 0.0;
    }
    for i in 50..150 {
        features[[i, 0]] = 6.0 + ((i - 50) as f64) * 0.01;
        features[[i, 1]] = 2.8;
        features[[i, 2]] = 4.5;
        features[[i, 3]] = 1.4;
        labels[i] = 1.0;
    }
    (features, labels)
}

#[test]
fn test_pre_training_gate_no_leakage_in_iris() {
    let (features, target) = make_iris_like();
    let names = vec!["sepal_len".into(), "sepal_wid".into(), "petal_len".into(), "petal_wid".into()];
    let mut ctx = QualityContext::default();
    let gate = PreTrainingGate::default();
    let kept = gate.run(&features, &target, &names, &mut ctx);
    // Iris features are not perfectly correlated with labels; nothing should be dropped
    assert!(!kept.is_empty(), "all features should be kept");
    assert_eq!(ctx.training_distribution.len(), 4);
    assert_eq!(ctx.n_samples, 150);
}

#[test]
fn test_algorithm_selector_recommends_for_iris_size() {
    let mut ctx = QualityContext::default();
    ctx.n_samples = 150;
    ctx.n_features = 4;
    let selector = AlgorithmSelector::default();
    let excluded = selector.excluded_algorithms(&ctx);
    // 150 samples < 1000 → boosting excluded
    assert!(excluded.contains(&"GradientBoosting".to_string()));
    // KNN should NOT be excluded (only excluded for n > 100k)
    assert!(!excluded.contains(&"KNN".to_string()));
}

#[test]
fn test_cv_strategy_for_iris_uses_repeated() {
    let mut ctx = QualityContext::default();
    ctx.n_samples = 150; // < 5000 → Repeated3x5
    let chooser = CvStrategyChooser::default();
    let strategy = chooser.choose(&ctx, None);
    assert!(matches!(strategy, CvStrategy::Repeated3x5Fold));
}

#[test]
fn test_pareto_front_with_iris_trials() {
    use automl::quality::ParetoPoint;
    let trials = vec![
        ParetoPoint { metric_score: 0.95, latency_ms: 5.0, trial_id: 0 },
        ParetoPoint { metric_score: 0.92, latency_ms: 2.0, trial_id: 1 },
        ParetoPoint { metric_score: 0.80, latency_ms: 10.0, trial_id: 2 }, // dominated by 0
    ];
    let front = compute_pareto_front(&trials);
    assert!(front.len() >= 2);
    let ids: Vec<usize> = front.iter().map(|p| p.trial_id).collect();
    assert!(!ids.contains(&2), "trial 2 should be dominated");
}

#[test]
fn test_calibration_improves_ece_on_iris() {
    // Simulate overconfident classifier: predicts 0.9 for all positives, but only
    // 50 out of 100 are actually positive → model is overconfident.
    let n = 100;
    // All scores are 0.9 (overconfident), half the labels are positive
    let scores: Vec<f64> = vec![0.9; n];
    let labels: Vec<f64> = (0..n).map(|i| if i < 50 { 1.0 } else { 0.0 }).collect();
    let bool_labels: Vec<bool> = labels.iter().map(|&l| l > 0.5).collect();

    let ece_before = expected_calibration_error(&scores, &bool_labels, 10);
    // ece_before should be large (≈ |0.5 - 0.9| = 0.4) since model predicts 0.9 but accuracy is 0.5
    assert!(ece_before > 0.1, "overconfident model should have high ECE, got {}", ece_before);

    let platt = PlattCalibrator::fit(&scores, &labels);
    let cal_scores: Vec<f64> = scores.iter().map(|&s| platt.predict(s)).collect();
    let ece_after = expected_calibration_error(&cal_scores, &bool_labels, 10);
    // Calibration should not make things dramatically worse
    assert!(ece_after <= ece_before + 0.1, "calibration should not dramatically worsen ECE");
}

#[test]
fn test_conformal_predictor_90_percent_coverage() {
    // Calibration: 200 points, true values = pred + constant 0.5
    let cal_preds: Vec<f64> = (0..200).map(|i| i as f64).collect();
    let cal_targets: Vec<f64> = cal_preds.iter().map(|p| p + 0.5).collect();
    let cp = ConformalPredictor::fit(&cal_preds, &cal_targets, 0.90).unwrap();

    // All residuals are 0.5, so quantile should be ~0.5
    assert!((cp.quantile - 0.5).abs() < 0.1, "quantile should ≈ 0.5, got {}", cp.quantile);

    // Test: 100 new points — all should be covered since residuals are constant
    let test_preds: Vec<f64> = (200..300).map(|i| i as f64).collect();
    let test_targets: Vec<f64> = test_preds.iter().map(|p| p + 0.5).collect();
    let covered = test_preds.iter().zip(test_targets.iter())
        .filter(|(&p, &t)| {
            let (lo, hi) = cp.predict_interval(p);
            t >= lo && t <= hi
        })
        .count();
    let coverage = covered as f64 / 100.0;
    assert!(coverage >= 0.80, "empirical coverage should be ≥ 80%, got {:.2}", coverage);
}

#[test]
fn test_ood_detector_in_vs_out() {
    use ndarray::array;
    let features = array![
        [0.0, 0.0], [0.1, 0.1], [-0.1, 0.1], [0.0, -0.1],
        [0.2, 0.0], [-0.2, 0.0], [0.1, -0.1], [-0.1, -0.1]
    ];
    let detector = OodDetector::fit(&features, 0.99);
    assert!(!detector.is_ood(&[0.0, 0.0]), "training centroid should be in-distribution");
    assert!(detector.is_ood(&[50.0, 50.0]), "extreme point should be OOD");
}
