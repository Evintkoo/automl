//! Preprocessing Example
//!
//! Demonstrates data preprocessing with scaling, encoding, and imputation.

use polars::prelude::*;
use automl::preprocessing::{
    DataPreprocessor, PreprocessingConfig, 
    ScalerType, EncoderType, ImputeStrategy,
};

fn main() -> anyhow::Result<()> {
    // Create sample data with missing values and categories
    let df = DataFrame::new(vec![
        Series::new("age".into(), &[Some(25.0), Some(30.0), None, Some(45.0), Some(35.0)]).into(),
        Series::new("income".into(), &[50000.0, 60000.0, 75000.0, 90000.0, 55000.0]).into(),
        Series::new("education".into(), &["high_school", "bachelor", "master", "phd", "bachelor"]).into(),
    ])?;

    println!("Original data:");
    println!("{}", df);

    // Configure preprocessing
    let config = PreprocessingConfig::default()
        .with_scaler(ScalerType::Standard)
        .with_encoder(EncoderType::OneHot)
        .with_numeric_impute(ImputeStrategy::Mean);

    // Apply preprocessing
    let mut preprocessor = DataPreprocessor::with_config(config);
    let processed = preprocessor.fit_transform(&df)?;

    println!("\nProcessed data:");
    println!("{}", processed);

    println!("\nNumeric columns: {:?}", preprocessor.numeric_columns());
    println!("Categorical columns: {:?}", preprocessor.categorical_columns());

    Ok(())
}
