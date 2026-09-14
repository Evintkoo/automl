use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};
use automl::preprocessing::{DataPreprocessor, PreprocessingConfig, ScalerType};
use polars::prelude::*;
use rand::prelude::*;

fn create_test_dataframe(n_rows: usize, n_cols: usize) -> DataFrame {
    let mut rng = rand::thread_rng();
    
    let series: Vec<Series> = (0..n_cols)
        .map(|i| {
            let values: Vec<f64> = (0..n_rows).map(|_| rng.gen()).collect();
            Series::new(format!("col_{}", i).into(), values)
        })
        .collect();
    
    DataFrame::new(series.into_iter().map(|s| s.into()).collect()).unwrap()
}

fn bench_preprocessing(c: &mut Criterion) {
    let mut group = c.benchmark_group("preprocessing");
    
    for n_rows in [1000, 10000, 100000].iter() {
        let df = create_test_dataframe(*n_rows, 10);
        
        group.bench_with_input(
            BenchmarkId::new("fit_transform", n_rows),
            &df,
            |b, df| {
                b.iter(|| {
                    let mut preprocessor = DataPreprocessor::with_config(
                        PreprocessingConfig::default().with_scaler(ScalerType::Standard)
                    );
                    preprocessor.fit_transform(black_box(df)).unwrap()
                })
            },
        );
    }
    
    group.finish();
}

fn bench_scaling(c: &mut Criterion) {
    let mut group = c.benchmark_group("scaling");
    
    let df = create_test_dataframe(10000, 20);
    
    for scaler in [ScalerType::Standard, ScalerType::MinMax, ScalerType::Robust].iter() {
        let name = format!("{:?}", scaler);
        
        group.bench_with_input(
            BenchmarkId::new("scaler", &name),
            scaler,
            |b, scaler| {
                b.iter(|| {
                    let mut preprocessor = DataPreprocessor::with_config(
                        PreprocessingConfig::default().with_scaler(scaler.clone())
                    );
                    preprocessor.fit_transform(black_box(&df)).unwrap()
                })
            },
        );
    }
    
    group.finish();
}

criterion_group!(benches, bench_preprocessing, bench_scaling);
criterion_main!(benches);
