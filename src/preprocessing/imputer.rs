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

        // Drop doesn't fill a value per column; it removes rows that are missing
        // (null or NaN) in any of the fitted columns.
        if matches!(self.strategy, ImputeStrategy::Drop) {
            return self.drop_missing_rows(df);
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

    /// Drop rows that have a missing value (Polars null or float NaN) in any of
    /// the fitted columns. Used by `ImputeStrategy::Drop`.
    fn drop_missing_rows(&self, df: &DataFrame) -> Result<DataFrame> {
        let mut keep_mask: Vec<bool> = vec![true; df.height()];

        for col_name in self.fill_values.keys() {
            if let Ok(col) = df.column(col_name) {
                let series = col.as_materialized_series();
                if let Ok(ca) = series.f64() {
                    for (i, opt) in ca.into_iter().enumerate() {
                        let missing = match opt {
                            None => true,
                            Some(v) => v.is_nan(),
                        };
                        if missing {
                            keep_mask[i] = false;
                        }
                    }
                }
            }
        }

        let mask: BooleanChunked = keep_mask.into_iter().collect();
        df.filter(&mask).map_err(|e| AutoMLError::DataError(e.to_string()))
    }

    /// Carry the last (ForwardFill) or next (BackwardFill) valid observation
    /// through missing values (Polars null or float NaN), in row order.
    /// A run of missing values with no prior/next valid observation (e.g.
    /// leading values for ForwardFill) falls back to 0.0.
    fn fill_series_directional(&self, series: &Series) -> Result<Series> {
        let ca = series
            .f64()
            .map_err(|e| AutoMLError::DataError(e.to_string()))?;

        let values: Vec<Option<f64>> = ca
            .into_iter()
            .map(|opt| match opt {
                Some(v) if v.is_nan() => None,
                other => other,
            })
            .collect();

        let mut filled: Vec<Option<f64>> = Vec::with_capacity(values.len());
        match self.strategy {
            ImputeStrategy::ForwardFill => {
                let mut last: Option<f64> = None;
                for v in values {
                    match v {
                        Some(x) => {
                            last = Some(x);
                            filled.push(Some(x));
                        }
                        None => filled.push(Some(last.unwrap_or(0.0))),
                    }
                }
            }
            ImputeStrategy::BackwardFill => {
                let mut next: Option<f64> = None;
                for v in values.into_iter().rev() {
                    match v {
                        Some(x) => {
                            next = Some(x);
                            filled.push(Some(x));
                        }
                        None => filled.push(Some(next.unwrap_or(0.0))),
                    }
                }
                filled.reverse();
            }
            _ => unreachable!(
                "fill_series_directional is only called for ForwardFill/BackwardFill"
            ),
        }

        let out: Float64Chunked = filled.into_iter().collect();
        Ok(out.with_name(series.name().clone()).into_series())
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
                let ca = series
                    .f64()
                    .map_err(|e| AutoMLError::DataError(e.to_string()))?;
                // Treat float NaN as missing, same as Polars null: exclude it from
                // the aggregate so a single NaN can't poison the fitted mean.
                let clean = ca
                    .filter(&ca.is_not_nan())
                    .map_err(|e| AutoMLError::DataError(e.to_string()))?;
                let mean = clean.mean().unwrap_or(0.0);
                Ok(ImputeValue::Numeric(mean))
            }
            ImputeStrategy::Median => {
                let ca = series
                    .f64()
                    .map_err(|e| AutoMLError::DataError(e.to_string()))?;
                let clean = ca
                    .filter(&ca.is_not_nan())
                    .map_err(|e| AutoMLError::DataError(e.to_string()))?;
                let median = clean.median().unwrap_or(0.0);
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
            ImputeStrategy::ForwardFill | ImputeStrategy::BackwardFill | ImputeStrategy::Drop => {
                // These strategies don't fit a single scalar: ForwardFill/BackwardFill
                // carry observed values through the series in order, and Drop removes
                // rows instead of filling them. `fill_series`/`transform` special-case
                // `self.strategy` directly for these; this placeholder is never used.
                Ok(ImputeValue::Numeric(0.0))
            }
            ImputeStrategy::Knn { .. } => Err(AutoMLError::ValidationError(
                "ImputeStrategy::Knn is not implemented for Imputer: this imputer fits/fills \
                 one column at a time, but KNN imputation needs the full feature matrix to \
                 find neighbors. Use `crate::imputation::KNNImputer` directly on an \
                 Array2<f64> of your numeric features instead."
                    .to_string(),
            )),
        }
    }

    fn fill_series(&self, series: &Series, fill_value: &ImputeValue) -> Result<Series> {
        // Forward/backward fill are order-dependent and don't use a single fitted
        // scalar; handle them directly against the series.
        if matches!(
            self.strategy,
            ImputeStrategy::ForwardFill | ImputeStrategy::BackwardFill
        ) {
            return self.fill_series_directional(series);
        }

        match fill_value {
            ImputeValue::Numeric(val) => {
                let ca = series
                    .f64()
                    .map_err(|e| AutoMLError::DataError(e.to_string()))?;

                // Fill both Polars null and float NaN: this crate's convention
                // (see src/imputation/mod.rs::is_missing) treats NaN as missing too,
                // and `opt.unwrap_or(*val)` alone only catches None.
                let filled: Float64Chunked = ca
                    .into_iter()
                    .map(|opt| match opt {
                        None => Some(*val),
                        Some(v) if v.is_nan() => Some(*val),
                        Some(v) => Some(v),
                    })
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
