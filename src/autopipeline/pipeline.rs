//! Automated pipeline execution

use crate::error::{AutoMLError, Result};
use crate::autopipeline::{
    composer::{
        ComposerConfig, ImputeStrategy,
        ModelStep, OutlierMethod, PipelineBlueprint, PipelineComposer,
        PreprocessingStep, ScaleMethod, TransformMethod,
    },
    detector::{DataTypeDetector, DetectedSchema},
};
use crate::training::TaskType;
use crate::training::ModelMetrics;
use ndarray::{Array1, Array2};
use serde::{Deserialize, Serialize};

/// Configuration for auto pipeline
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PipelineConfig {
    /// Task type
    pub task_type: TaskType,
    /// Time budget in seconds
    pub time_budget_secs: Option<f64>,
    /// Maximum number of models to try
    pub max_models: usize,
    /// Enable hyperparameter optimization
    pub optimize_hyperparams: bool,
    /// Cross-validation folds
    pub cv_folds: usize,
    /// Metric to optimize
    pub metric: String,
    /// Verbose logging
    pub verbose: bool,
}

impl Default for PipelineConfig {
    fn default() -> Self {
        Self {
            task_type: TaskType::BinaryClassification,
            time_budget_secs: Some(300.0),
            max_models: 5,
            optimize_hyperparams: true,
            cv_folds: 5,
            metric: "accuracy".to_string(),
            verbose: true,
        }
    }
}

/// Result of pipeline execution
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PipelineResult {
    /// Best model type
    pub best_model: ModelStep,
    /// Best model metrics
    pub best_metrics: ModelMetrics,
    /// All model results
    pub all_results: Vec<(ModelStep, ModelMetrics)>,
    /// Applied preprocessing steps
    pub preprocessing_steps: Vec<PreprocessingStep>,
    /// Feature importances (if available)
    pub feature_importances: Option<Vec<(usize, f64)>>,
    /// Detected schema
    pub schema: DetectedSchema,
    /// Total time taken
    pub total_time_secs: f64,
    /// Notes and warnings
    pub notes: Vec<String>,
}

/// Fitted preprocessing state
#[derive(Debug, Clone)]
struct FittedPreprocessor {
    /// Imputation values
    impute_values: Vec<(usize, f64)>,
    /// Scaling parameters: (col, mean, std)
    scale_params: Vec<(usize, f64, f64)>,
    /// Columns to drop
    drop_columns: Vec<usize>,
    /// Original column count
    #[allow(dead_code)]
    n_cols_original: usize,
}

impl FittedPreprocessor {
    fn new(n_cols: usize) -> Self {
        Self {
            impute_values: Vec::new(),
            scale_params: Vec::new(),
            drop_columns: Vec::new(),
            n_cols_original: n_cols,
        }
    }
}

/// Auto ML Pipeline
pub struct AutoPipeline {
    #[allow(dead_code)]
    config: PipelineConfig,
    detector: DataTypeDetector,
    composer: PipelineComposer,
    fitted_preprocessor: Option<FittedPreprocessor>,
    blueprint: Option<PipelineBlueprint>,
    detected_schema: Option<DetectedSchema>,
}

impl AutoPipeline {
    /// Create new auto pipeline
    pub fn new(config: PipelineConfig) -> Self {
        let composer_config = ComposerConfig {
            task_type: config.task_type.clone(),
            ..Default::default()
        };

        Self {
            config,
            detector: DataTypeDetector::new(),
            composer: PipelineComposer::new(composer_config),
            fitted_preprocessor: None,
            blueprint: None,
            detected_schema: None,
        }
    }

    /// Create with default config
    pub fn default() -> Self {
        Self::new(PipelineConfig::default())
    }

    /// Analyze data and create pipeline blueprint
    pub fn analyze(&mut self, x: &Array2<f64>, column_names: Option<&[String]>) -> Result<&PipelineBlueprint> {
        // Detect schema
        let schema = self.detector.detect_from_array(x, column_names)?;

        // Compose pipeline
        let blueprint = self.composer.compose(&schema)?;

        self.detected_schema = Some(schema);
        self.blueprint = Some(blueprint);

        Ok(self.blueprint.as_ref().unwrap())
    }

    /// Fit preprocessing on data
    pub fn fit_preprocessing(&mut self, x: &Array2<f64>, _y: &Array1<f64>) -> Result<Array2<f64>> {
        let blueprint = self.blueprint.as_ref().ok_or_else(|| {
            AutoMLError::ValidationError("Pipeline not analyzed. Call analyze() first.".to_string())
        })?;

        let mut preprocessor = FittedPreprocessor::new(x.ncols());
        let mut x_transformed = x.clone();

        for step in &blueprint.preprocessing_steps {
            match step {
                PreprocessingStep::DropColumns { columns } => {
                    preprocessor.drop_columns.extend(columns.iter().copied());
                }
                PreprocessingStep::Impute { strategy, columns } => {
                    for &col in columns {
                        if col >= x_transformed.ncols() {
                            continue;
                        }
                        
                        let col_data = x_transformed.column(col);
                        let valid: Vec<f64> = col_data.iter().filter(|v| !v.is_nan()).copied().collect();
                        
                        let fill_value = match strategy {
                            ImputeStrategy::Mean => {
                                valid.iter().sum::<f64>() / valid.len().max(1) as f64
                            }
                            ImputeStrategy::Median => {
                                if valid.is_empty() {
                                    0.0
                                } else {
                                    let mut sorted = valid.clone();
                                    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
                                    sorted[sorted.len() / 2]
                                }
                            }
                            ImputeStrategy::Mode => {
                                // For numeric, use median as proxy
                                if valid.is_empty() {
                                    0.0
                                } else {
                                    let mut sorted = valid.clone();
                                    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
                                    sorted[sorted.len() / 2]
                                }
                            }
                            ImputeStrategy::Constant(val) => *val,
                            ImputeStrategy::KNN => {
                                // Fallback to median for simplicity
                                if valid.is_empty() {
                                    0.0
                                } else {
                                    let mut sorted = valid.clone();
                                    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
                                    sorted[sorted.len() / 2]
                                }
                            }
                        };
                        
                        preprocessor.impute_values.push((col, fill_value));
                        
                        // Apply imputation
                        for i in 0..x_transformed.nrows() {
                            if x_transformed[[i, col]].is_nan() {
                                x_transformed[[i, col]] = fill_value;
                            }
                        }
                    }
                }
                PreprocessingStep::Scale { method, columns } => {
                    for &col in columns {
                        if col >= x_transformed.ncols() || preprocessor.drop_columns.contains(&col) {
                            continue;
                        }
                        
                        let col_data = x_transformed.column(col);
                        let values: Vec<f64> = col_data.iter().copied().collect();
                        
                        /// Minimum scale value to prevent numerical instability
                        const SCALE_EPS: f64 = 1e-12;

                        /// Compute quantile using linear interpolation (matching scikit-learn)
                        fn quantile_linear(sorted: &[f64], q: f64) -> f64 {
                            if sorted.is_empty() {
                                return 0.0;
                            }
                            if sorted.len() == 1 {
                                return sorted[0];
                            }
                            let pos = q * (sorted.len() - 1) as f64;
                            let lo = pos.floor() as usize;
                            let hi = pos.ceil() as usize;
                            let frac = pos - lo as f64;
                            if lo == hi {
                                sorted[lo]
                            } else {
                                sorted[lo] * (1.0 - frac) + sorted[hi] * frac
                            }
                        }

                        let n = values.len() as f64;
                        let (center, scale) = match method {
                            ScaleMethod::Standard => {
                                let mean = values.iter().sum::<f64>() / n;
                                // Use sample std (ddof=1) consistent with scaler.rs
                                let variance = values.iter().map(|v| (v - mean).powi(2)).sum::<f64>()
                                    / (n - 1.0).max(1.0);
                                let std = variance.sqrt();
                                (mean, if std.abs() < SCALE_EPS { 1.0 } else { std })
                            }
                            ScaleMethod::MinMax => {
                                let min = values.iter().cloned().fold(f64::INFINITY, f64::min);
                                let max = values.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                                let range = max - min;
                                (min, if range.abs() < SCALE_EPS { 1.0 } else { range })
                            }
                            ScaleMethod::Robust => {
                                let mut sorted = values.clone();
                                sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
                                let q1 = quantile_linear(&sorted, 0.25);
                                let q3 = quantile_linear(&sorted, 0.75);
                                let median = quantile_linear(&sorted, 0.5);
                                let iqr = q3 - q1;
                                (median, if iqr.abs() < SCALE_EPS { 1.0 } else { iqr })
                            }
                            ScaleMethod::MaxAbs => {
                                let max_abs = values.iter().map(|v| v.abs()).fold(0.0, f64::max);
                                (0.0, if max_abs < SCALE_EPS { 1.0 } else { max_abs })
                            }
                        };
                        
                        preprocessor.scale_params.push((col, center, scale));
                        
                        // Apply scaling
                        for i in 0..x_transformed.nrows() {
                            x_transformed[[i, col]] = (x_transformed[[i, col]] - center) / scale;
                        }
                    }
                }
                PreprocessingStep::HandleOutliers { method, columns } => {
                    for &col in columns {
                        if col >= x_transformed.ncols() || preprocessor.drop_columns.contains(&col) {
                            continue;
                        }
                        
                        match method {
                            OutlierMethod::Clip => {
                                let col_data = x_transformed.column(col);
                                let values: Vec<f64> = col_data.iter().copied().collect();
                                let mean = values.iter().sum::<f64>() / values.len() as f64;
                                let std = (values.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / values.len() as f64).sqrt();
                                let lower = mean - 3.0 * std;
                                let upper = mean + 3.0 * std;
                                
                                for i in 0..x_transformed.nrows() {
                                    x_transformed[[i, col]] = x_transformed[[i, col]].clamp(lower, upper);
                                }
                            }
                            _ => {
                                // Other methods not implemented yet
                            }
                        }
                    }
                }
                PreprocessingStep::Transform { method, columns } => {
                    for &col in columns {
                        if col >= x_transformed.ncols() || preprocessor.drop_columns.contains(&col) {
                            continue;
                        }
                        
                        for i in 0..x_transformed.nrows() {
                            let val = x_transformed[[i, col]];
                            x_transformed[[i, col]] = match method {
                                TransformMethod::Log => (val.abs() + 1.0).ln() * val.signum(),
                                TransformMethod::Sqrt => val.abs().sqrt() * val.signum(),
                                TransformMethod::BoxCox | TransformMethod::YeoJohnson => {
                                    // Simplified Yeo-Johnson with lambda=0.5
                                    if val >= 0.0 {
                                        ((val + 1.0).powf(0.5) - 1.0) / 0.5
                                    } else {
                                        -((-val + 1.0).powf(0.5) - 1.0) / 0.5
                                    }
                                }
                            };
                        }
                    }
                }
                _ => {
                    // Other steps (encoding, feature selection) require more complex handling
                }
            }
        }

        // Drop columns if needed
        if !preprocessor.drop_columns.is_empty() {
            let keep_cols: Vec<usize> = (0..x_transformed.ncols())
                .filter(|c| !preprocessor.drop_columns.contains(c))
                .collect();
            
            let new_data: Vec<f64> = x_transformed
                .rows()
                .into_iter()
                .flat_map(|row| {
                    keep_cols.iter().map(|&c| row[c]).collect::<Vec<_>>()
                })
                .collect();
            
            x_transformed = Array2::from_shape_vec(
                (x_transformed.nrows(), keep_cols.len()),
                new_data,
            )?;
        }

        self.fitted_preprocessor = Some(preprocessor);
        Ok(x_transformed)
    }

    /// Transform new data using fitted preprocessing
    pub fn transform(&self, x: &Array2<f64>) -> Result<Array2<f64>> {
        let preprocessor = self.fitted_preprocessor.as_ref().ok_or_else(|| {
            AutoMLError::ValidationError("Preprocessing not fitted. Call fit_preprocessing() first.".to_string())
        })?;

        let mut x_transformed = x.clone();

        // Apply imputation
        for &(col, fill_value) in &preprocessor.impute_values {
            if col < x_transformed.ncols() {
                for i in 0..x_transformed.nrows() {
                    if x_transformed[[i, col]].is_nan() {
                        x_transformed[[i, col]] = fill_value;
                    }
                }
            }
        }

        // Apply scaling
        for &(col, center, scale) in &preprocessor.scale_params {
            if col < x_transformed.ncols() {
                for i in 0..x_transformed.nrows() {
                    x_transformed[[i, col]] = (x_transformed[[i, col]] - center) / scale;
                }
            }
        }

        // Drop columns
        if !preprocessor.drop_columns.is_empty() {
            let keep_cols: Vec<usize> = (0..x_transformed.ncols())
                .filter(|c| !preprocessor.drop_columns.contains(c))
                .collect();
            
            let new_data: Vec<f64> = x_transformed
                .rows()
                .into_iter()
                .flat_map(|row| {
                    keep_cols.iter().map(|&c| row[c]).collect::<Vec<_>>()
                })
                .collect();
            
            x_transformed = Array2::from_shape_vec(
                (x_transformed.nrows(), keep_cols.len()),
                new_data,
            )?;
        }

        Ok(x_transformed)
    }

    /// Get the detected schema
    ///
    /// Returns `None` if `analyze()` has not been called yet.
    pub fn schema(&self) -> Option<DetectedSchema> {
        self.detected_schema.clone()
    }

    /// Get the blueprint
    pub fn blueprint(&self) -> Option<&PipelineBlueprint> {
        self.blueprint.as_ref()
    }
}

impl Default for AutoPipeline {
    fn default() -> Self {
        Self::new(PipelineConfig::default())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_data() -> (Array2<f64>, Array1<f64>) {
        let x = Array2::from_shape_vec(
            (20, 4),
            vec![
                1.0, 2.0, f64::NAN, 0.0,
                2.0, 4.0, 3.0, 1.0,
                3.0, 6.0, 4.0, 0.0,
                4.0, 8.0, 5.0, 1.0,
                5.0, 10.0, 6.0, 0.0,
                6.0, 12.0, 7.0, 1.0,
                7.0, 14.0, 8.0, 0.0,
                8.0, 16.0, f64::NAN, 1.0,
                9.0, 18.0, 10.0, 0.0,
                10.0, 20.0, 11.0, 1.0,
                11.0, 22.0, 12.0, 0.0,
                12.0, 24.0, 13.0, 1.0,
                13.0, 26.0, 14.0, 0.0,
                14.0, 28.0, 15.0, 1.0,
                15.0, 30.0, 16.0, 0.0,
                16.0, 32.0, 17.0, 1.0,
                17.0, 34.0, 18.0, 0.0,
                18.0, 36.0, 19.0, 1.0,
                19.0, 38.0, 20.0, 0.0,
                20.0, 40.0, 21.0, 1.0,
            ],
        ).unwrap();
        
        let y = Array1::from_vec(vec![
            0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0,
            0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0,
        ]);
        
        (x, y)
    }

    #[test]
    fn test_analyze() {
        let (x, _) = create_test_data();
        let mut pipeline = AutoPipeline::default();
        
        let blueprint = pipeline.analyze(&x, None).unwrap();
        
        assert!(!blueprint.preprocessing_steps.is_empty());
        assert!(!blueprint.recommended_models.is_empty());
    }

    #[test]
    fn test_fit_preprocessing() {
        let (x, y) = create_test_data();
        let mut pipeline = AutoPipeline::default();
        
        pipeline.analyze(&x, None).unwrap();
        let x_transformed = pipeline.fit_preprocessing(&x, &y).unwrap();
        
        // Should have handled NaN values
        assert!(!x_transformed.iter().any(|v| v.is_nan()));
    }

    #[test]
    fn test_transform() {
        let (x, y) = create_test_data();
        let mut pipeline = AutoPipeline::default();
        
        pipeline.analyze(&x, None).unwrap();
        pipeline.fit_preprocessing(&x, &y).unwrap();
        
        // Transform new data
        let x_new = Array2::from_shape_vec(
            (2, 4),
            vec![1.5, 3.0, 4.5, 0.0, 2.5, 5.0, f64::NAN, 1.0],
        ).unwrap();
        
        let x_transformed = pipeline.transform(&x_new).unwrap();
        
        // Should have handled NaN
        assert!(!x_transformed.iter().any(|v| v.is_nan()));
    }
}
