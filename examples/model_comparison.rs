//! Model Comparison Example
//!
//! Demonstrates comparing multiple models on the same dataset.

use polars::prelude::*;
use automl::training::{TrainEngine, TrainingConfig, TaskType, ModelType};
use std::time::Instant;

fn main() -> anyhow::Result<()> {
    // Create sample regression data
    let n = 100;
    let x1: Vec<f64> = (0..n).map(|i| i as f64 / 10.0).collect();
    let x2: Vec<f64> = (0..n).map(|i| (i as f64 / 10.0).sin()).collect();
    let y: Vec<f64> = x1.iter().zip(x2.iter())
        .map(|(a, b)| 2.0 * a + 3.0 * b + 0.5)
        .collect();

    let df = DataFrame::new(vec![
        Series::new("x1".into(), &x1).into(),
        Series::new("x2".into(), &x2).into(),
        Series::new("target".into(), &y).into(),
    ])?;

    println!("Dataset: {} samples\n", df.height());
    println!("{:<25} {:>12} {:>12}", "Model", "R² Score", "Time (ms)");
    println!("{}", "-".repeat(52));

    let models = vec![
        ("Linear Regression", ModelType::LinearRegression),
        ("Decision Tree", ModelType::DecisionTree),
        ("Random Forest", ModelType::RandomForest),
        ("Gradient Boosting", ModelType::GradientBoosting),
        ("KNN", ModelType::KNN),
    ];

    for (name, model_type) in models {
        let config = TrainingConfig::new(TaskType::Regression, "target")
            .with_model(model_type);

        let mut engine = TrainEngine::new(config);
        let start = Instant::now();
        
        match engine.fit(&df) {
            Ok(_) => {
                let elapsed = start.elapsed().as_millis();
                let metrics = engine.metrics().cloned().unwrap_or_default();
                let r2 = metrics.r2.unwrap_or(0.0);
                println!("{:<25} {:>12.4} {:>12}", name, r2, elapsed);
            }
            Err(e) => {
                println!("{:<25} {:>12}", name, format!("Error: {}", e));
            }
        }
    }

    Ok(())
}
