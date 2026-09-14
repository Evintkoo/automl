//! Basic Training Example
//!
//! Demonstrates how to train a model using the AutoML framework.

use polars::prelude::*;
use automl::training::{TrainEngine, TrainingConfig, TaskType, ModelType};

fn main() -> anyhow::Result<()> {
    // Create sample data
    let df = DataFrame::new(vec![
        Series::new("feature1".into(), &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]).into(),
        Series::new("feature2".into(), &[2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0, 20.0]).into(),
        Series::new("target".into(), &[0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0]).into(),
    ])?;

    println!("Dataset: {} rows, {} columns", df.height(), df.width());

    // Configure training
    let config = TrainingConfig::new(TaskType::BinaryClassification, "target")
        .with_model(ModelType::RandomForest)
        .with_n_estimators(10)
        .with_max_depth(5);

    // Train the model
    let mut engine = TrainEngine::new(config);
    engine.fit(&df)?;

    // Check metrics
    if let Some(metrics) = engine.metrics() {
        println!("\nTraining Results:");
        println!("  Accuracy: {:.4}", metrics.accuracy.unwrap_or(0.0));
        println!("  F1 Score: {:.4}", metrics.f1_score.unwrap_or(0.0));
        println!("  Training time: {:.3}s", metrics.training_time_secs);
    }

    // Make predictions
    let predictions = engine.predict(&df)?;
    println!("\nPredictions: {:?}", predictions);

    Ok(())
}
