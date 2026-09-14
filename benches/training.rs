use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};
use automl::training::{TrainEngine, TrainingConfig, TaskType, ModelType};
use polars::prelude::*;
use rand::prelude::*;

fn create_regression_data(n_rows: usize, n_features: usize) -> DataFrame {
    let mut rng = rand::thread_rng();
    
    let mut series: Vec<Series> = (0..n_features)
        .map(|i| {
            let values: Vec<f64> = (0..n_rows).map(|_| rng.gen::<f64>() * 10.0).collect();
            Series::new(format!("feature_{}", i).into(), values)
        })
        .collect();
    
    // Create target as sum of features + noise
    let target: Vec<f64> = (0..n_rows)
        .map(|i| {
            let mut sum = 0.0;
            for s in &series {
                sum += s.f64().unwrap().get(i).unwrap_or(0.0);
            }
            sum + rng.gen::<f64>() * 0.1
        })
        .collect();
    
    series.push(Series::new("target".into(), target));
    
    DataFrame::new(series.into_iter().map(|s| s.into()).collect()).unwrap()
}

fn bench_training(c: &mut Criterion) {
    let mut group = c.benchmark_group("training");
    group.sample_size(10); // Fewer samples for training benchmarks
    
    for n_rows in [1000, 5000, 10000].iter() {
        let df = create_regression_data(*n_rows, 10);
        
        group.bench_with_input(
            BenchmarkId::new("fit", n_rows),
            &df,
            |b, df| {
                b.iter(|| {
                    let config = TrainingConfig::new(TaskType::Regression, "target");
                    let mut engine = TrainEngine::new(config);
                    engine.fit(black_box(df)).unwrap();
                    engine
                })
            },
        );
    }
    
    group.finish();
}

fn bench_prediction(c: &mut Criterion) {
    let mut group = c.benchmark_group("prediction");
    
    // Train model once
    let train_df = create_regression_data(5000, 10);
    let config = TrainingConfig::new(TaskType::Regression, "target");
    let mut engine = TrainEngine::new(config);
    engine.fit(&train_df).unwrap();
    
    for n_rows in [100, 1000, 10000].iter() {
        let test_df = create_regression_data(*n_rows, 10);
        
        group.bench_with_input(
            BenchmarkId::new("predict", n_rows),
            &test_df,
            |b, df| {
                b.iter(|| {
                    engine.predict(black_box(df)).unwrap()
                })
            },
        );
    }
    
    group.finish();
}

criterion_group!(benches, bench_training, bench_prediction);
criterion_main!(benches);
