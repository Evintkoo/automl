//! Hyperparameter Optimization Example
//!
//! Demonstrates using HyperOptX for automatic hyperparameter tuning.

use polars::prelude::*;
use automl::optimizer::{HyperOptX, OptimizationConfig, SearchSpace, OptimizeDirection};
use automl::training::{TrainEngine, TrainingConfig, TaskType, ModelType};

fn main() -> anyhow::Result<()> {
    // Create sample classification data
    let n = 200;
    let x1: Vec<f64> = (0..n).map(|i| (i as f64 / 20.0).cos() + rand_simple(i)).collect();
    let x2: Vec<f64> = (0..n).map(|i| (i as f64 / 20.0).sin() + rand_simple(i + 100)).collect();
    let y: Vec<f64> = x1.iter().zip(x2.iter())
        .map(|(a, b)| if *a + *b > 1.0 { 1.0 } else { 0.0 })
        .collect();

    let df = DataFrame::new(vec![
        Series::new("x1".into(), &x1).into(),
        Series::new("x2".into(), &x2).into(),
        Series::new("target".into(), &y).into(),
    ])?;

    println!("Dataset: {} samples for classification\n", df.height());

    // Configure hyperparameter optimization
    let opt_config = OptimizationConfig {
        n_trials: 20,
        timeout_secs: Some(60.0),
        direction: OptimizeDirection::Maximize,
        random_state: Some(42),
        ..Default::default()
    };

    // Define search space
    let search_space = SearchSpace::new()
        .int("n_estimators", 5, 50)
        .int("max_depth", 2, 10);

    // Create optimizer
    let mut optimizer = HyperOptX::new(opt_config, search_space);

    println!("Running hyperparameter optimization (20 trials)...\n");

    // Define the objective function
    let study = optimizer.optimize(|params| {
        let n_est = params.get("n_estimators")
            .and_then(|v| v.as_int())
            .unwrap_or(10) as usize;
        let depth = params.get("max_depth")
            .and_then(|v| v.as_int())
            .unwrap_or(6) as usize;

        let config = TrainingConfig::new(TaskType::BinaryClassification, "target")
            .with_model(ModelType::RandomForest)
            .with_n_estimators(n_est)
            .with_max_depth(depth);

        let mut engine = TrainEngine::new(config);
        if engine.fit(&df).is_err() {
            return Ok(0.0);
        }

        let metrics = engine.metrics().cloned().unwrap_or_default();
        Ok(metrics.accuracy.unwrap_or(0.0))
    })?;

    println!("Optimization complete!");
    println!("Trials completed: {}", study.trials.len());
    if let Some(best_idx) = study.best_trial_idx {
        let best = &study.trials[best_idx];
        println!("Best score: {:.4}", best.value);
        println!("Best parameters: {:?}", best.params);
    }

    Ok(())
}

// Simple deterministic pseudo-random for reproducibility
fn rand_simple(seed: usize) -> f64 {
    ((seed * 1103515245 + 12345) % (1 << 31)) as f64 / (1 << 31) as f64 - 0.5
}
