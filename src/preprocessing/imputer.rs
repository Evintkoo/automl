//! Missing value imputation strategies

use crate::error::{AutoMLError, Result};
use polars::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Strategy for imputing missing values
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ImputeStrategy {
    /// Replace with mean (numeric only)
    Mean,
    /// Replace with median (numeric only)
    Median,
    /// Replace with mode / most frequent value
    MostFrequent,
    /// Replace with a constant value
    Constant(f64),
    /// Replace with a constant string (categorical)
    ConstantString(String),
    /// Forward fill
    ForwardFill,
    /// Backward fill
    BackwardFill,
    /// KNN imputation
    Knn { n_neighbors: usize },
    /// Drop rows with missing values
    Drop,
}

/// Imputer for handling missing values
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Imputer {
    strategy: ImputeStrategy,
    fill_values: HashMap<String, ImputeValue>,
    is_fitted: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
enum ImputeValue {
    Numeric(f64),
    String(String),
}

impl Imputer {
    /// Create a new imputer with the specified strategy
    pub fn new(strategy: ImputeStrategy) -> Self {
        Self {
            strategy,
            fill_values: HashMap::new(),
            is_fitted: false,
        }
    }

    /// Fit the imputer to the data
    pub fn fit(&mut self, df: &DataFrame, columns: &[&str]) -> Result<&mut Self> {
        for col_name in columns {
            let series = df
                .column(col_name)
                .map_err(|_| AutoMLError::FeatureNotFound(col_name.to_string()))?;

            let fill_value = self.compute_fill_value(series.as_materialized_series())?;
            self.fill_values.insert(col_name.to_string(), fill_value);
        }

        self.is_fitted = true;
        Ok(self)
    }

    /// Transform the data by imputing missing values
    pub fn transform(&self, df: &DataFrame) -> Result<DataFrame> {
        if !self.is_fitted {
            return Err(AutoMLError::ModelNotFitted);
        }

        let mut result = df.clone();

        for (col_name, fill_value) in &self.fill_values {
            if let Ok(col) = df.column(col_name) {
                let series = col.as_materialized_series();
                let filled = self.fill_series(series, fill_value)?;
                result = result
                    .with_column(filled)
                    .map_err(|e| AutoMLError::DataError(e.to_string()))?
                    .clone();
            }
        }

        Ok(result)
    }

    /// Fit and transform in one step
    pub fn fit_transform(&mut self, df: &DataFrame, columns: &[&str]) -> Result<DataFrame> {
        self.fit(df, columns)?;
        self.transform(df)
    }

    /// Check if dtype is numeric
    fn is_numeric_dtype(dtype: &DataType) -> bool {
        matches!(
            dtype,
            DataType::Int8
                | DataType::Int16
                | DataType::Int32
                | DataType::Int64
                | DataType::UInt8
                | DataType::UInt16
                | DataType::UInt32
                | DataType::UInt64
                | DataType::Float32
                | DataType::Float64
        )
    }

    /// Compute mode (most frequent value) for a series
    fn compute_mode_numeric(series: &Series) -> Result<f64> {
        let mut counts: HashMap<u64, usize> = HashMap::new();

        // Count occurrences by converting to integer bits for f64
        if let Ok(ca) = series.f64() {
            for val in ca.into_iter().flatten() {
                let key = val.to_bits();
                *counts.entry(key).or_insert(0) += 1;
            }
        } else if let Ok(ca) = series.i64() {
            for val in ca.into_iter().flatten() {
                let key = (val as f64).to_bits();
                *counts.entry(key).or_insert(0) += 1;
            }
        }

        let mode_key = counts
            .into_iter()
            .max_by_key(|(_, count)| *count)
            .map(|(k, _)| k)
            .unwrap_or(0);

        Ok(f64::from_bits(mode_key))
    }

    /// Compute mode for string series
    fn compute_mode_string(series: &Series) -> Result<String> {
        let mut counts: HashMap<String, usize> = HashMap::new();
        
        if let Ok(ca) = series.str() {
            for val in ca.into_iter().flatten() {
                *counts.entry(val.to_string()).or_insert(0) += 1;
            }
        }
        
        let mode = counts
            .into_iter()
            .max_by_key(|(_, count)| *count)
            .map(|(k, _)| k)
            .unwrap_or_default();
        
        Ok(mode)
    }

    fn compute_fill_value(&self, series: &Series) -> Result<ImputeValue> {
        match &self.strategy {
            ImputeStrategy::Mean => {
                let mean = series
                    .f64()
                    .map_err(|e| AutoMLError::DataError(e.to_string()))?
                    .mean()
                    .unwrap_or(0.0);
                Ok(ImputeValue::Numeric(mean))
            }
            ImputeStrategy::Median => {
                let median = series
                    .f64()
                    .map_err(|e| AutoMLError::DataError(e.to_string()))?
                    .median()
                    .unwrap_or(0.0);
                Ok(ImputeValue::Numeric(median))
            }
            ImputeStrategy::MostFrequent => {
                // Get mode (most frequent value)
                if Self::is_numeric_dtype(series.dtype()) {
                    let mode = Self::compute_mode_numeric(series)?;
                    Ok(ImputeValue::Numeric(mode))
                } else {
                    let mode = Self::compute_mode_string(series)?;
                    Ok(ImputeValue::String(mode))
                }
            }
            ImputeStrategy::Constant(val) => Ok(ImputeValue::Numeric(*val)),
            ImputeStrategy::ConstantString(val) => Ok(ImputeValue::String(val.clone())),
            _ => Ok(ImputeValue::Numeric(0.0)), // Placeholder for other strategies
        }
    }

    fn fill_series(&self, series: &Series, fill_value: &ImputeValue) -> Result<Series> {
        match fill_value {
            ImputeValue::Numeric(val) => {
                let ca = series
                    .f64()
                    .map_err(|e| AutoMLError::DataError(e.to_string()))?;
                
                // Manually fill nulls
                let filled: Float64Chunked = ca
                    .into_iter()
                    .map(|opt| Some(opt.unwrap_or(*val)))
                    .collect();
                
                Ok(filled.with_name(series.name().clone()).into_series())
            }
            ImputeValue::String(val) => {
                let ca = series
                    .str()
                    .map_err(|e| AutoMLError::DataError(e.to_string()))?;
                
                // Manually fill nulls for strings
                let filled: StringChunked = ca
                    .into_iter()
                    .map(|opt| Some(opt.unwrap_or(val.as_str()).to_string()))
                    .collect();
                
                Ok(filled.with_name(series.name().clone()).into_series())
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_imputer_creation() {
        let imputer = Imputer::new(ImputeStrategy::Mean);
        assert!(!imputer.is_fitted);
    }

    #[test]
    fn test_impute_strategy_serialize() {
        let strategy = ImputeStrategy::Knn { n_neighbors: 5 };
        let json = serde_json::to_string(&strategy).unwrap();
        assert!(json.contains("Knn"));
        assert!(json.contains("5"));
    }

    #[test]
    fn test_mean_imputation() {
        let df = DataFrame::new(vec![
            Column::new("a".into(), &[Some(1.0), None, Some(3.0), Some(4.0)]),
        ])
        .unwrap();

        let mut imputer = Imputer::new(ImputeStrategy::Mean);
        let result = imputer.fit_transform(&df, &["a"]).unwrap();

        let col = result.column("a").unwrap().f64().unwrap();
        // Mean of [1, 3, 4] = 8/3 ≈ 2.67
        assert!((col.get(1).unwrap() - 2.666666666666667).abs() < 0.001);
    }
}
