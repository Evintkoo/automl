//! Training engine implementation

use crate::error::{AutoMLError, Result};
use super::{TrainingConfig, ModelType, TaskType, ModelMetrics};
use super::linear_models::{LinearRegression, LogisticRegression, RidgeRegression, LassoRegression, ElasticNetRegression, PolynomialRegression};
use super::decision_tree::DecisionTree;
use super::random_forest::RandomForest;
use super::gradient_boosting::{GradientBoostingRegressor, GradientBoostingClassifier, GradientBoostingConfig};
use super::knn::{KNNClassifier, KNNRegressor, KNNConfig};
use super::naive_bayes::{GaussianNaiveBayes, MultinomialNaiveBayes};
use super::svm::{SVMClassifier, SVMRegressor, SVMConfig, KernelType};
use super::adaboost::AdaBoostClassifier;
use super::extra_trees::ExtraTrees;
use super::xgboost::{XGBoostRegressor, XGBoostClassifier, XGBoostConfig};
use super::lightgbm::{LightGBMRegressor, LightGBMClassifier, LightGBMConfig};
use super::catboost::{CatBoostRegressor, CatBoostClassifier, CatBoostConfig};
use super::sgd::{SGDRegressor, SGDClassifier, SGDConfig};
use super::gaussian_process_regression::{GaussianProcessRegressor, GPConfig};
use super::clustering::{KMeans, DBSCAN};
use super::som::SOM;
use ndarray::{Array1, Array2, Axis};
use polars::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::Instant;

/// Enum to hold trained model variants
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TrainedModel {
    LinearRegression(LinearRegression),
    LogisticRegression(LogisticRegression),
    RidgeRegression(RidgeRegression),
    LassoRegression(LassoRegression),
    ElasticNetRegression(ElasticNetRegression),
    DecisionTreeClassifier(DecisionTree),
    DecisionTreeRegressor(DecisionTree),
    RandomForestClassifier(RandomForest),
    RandomForestRegressor(RandomForest),
    GradientBoostingClassifier(GradientBoostingClassifier),
    GradientBoostingRegressor(GradientBoostingRegressor),
    KNNClassifier(KNNClassifier),
    KNNRegressor(KNNRegressor),
    GaussianNaiveBayes(GaussianNaiveBayes),
    MultinomialNaiveBayes(MultinomialNaiveBayes),
    SVMClassifier(SVMClassifier),
    SVMRegressor(SVMRegressor),
    AdaBoostClassifier(AdaBoostClassifier),
    ExtraTreesClassifier(ExtraTrees),
    ExtraTreesRegressor(ExtraTrees),
    XGBoostRegressor(XGBoostRegressor),
    XGBoostClassifier(XGBoostClassifier),
    LightGBMRegressor(LightGBMRegressor),
    LightGBMClassifier(LightGBMClassifier),
    CatBoostRegressor(CatBoostRegressor),
    CatBoostClassifier(CatBoostClassifier),
    SGDRegressor(SGDRegressor),
    SGDClassifier(SGDClassifier),
    GaussianProcessRegressor(GaussianProcessRegressor),
    PolynomialRegression(PolynomialRegression),
    KMeans(KMeans),
    DBSCAN(DBSCAN),
    SelfOrganizingMap(SOM),
}

/// Ensemble training strategy
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum EnsembleStrategy {
    /// Majority voting for classification, mode for regression
    Voting,
    /// Average predictions across models
    Averaging,
    /// Use first-level model predictions as features for a meta-learner
    Stacking,
}

/// Result of ensemble training
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnsembleResult {
    pub models: Vec<String>,
    pub individual_metrics: HashMap<String, f64>,
    pub ensemble_metric: f64,
    pub strategy: EnsembleStrategy,
}

/// Model comparison entry
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelComparison {
    pub model_name: String,
    pub task_type: TaskType,
    pub metrics: HashMap<String, f64>,
    pub training_time_secs: f64,
    pub model_size_bytes: usize,
}

/// A single epoch's training metrics, recorded during iterative training
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EpochRecord {
    pub epoch: usize,
    pub train_loss: f64,
    pub val_loss: Option<f64>,
    pub train_acc: Option<f64>,
    pub val_acc: Option<f64>,
}

/// A flattened, serializable representation of a single decision tree node
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SerializedNode {
    pub id: usize,
    pub parent_id: Option<usize>,
    pub feature_name: Option<String>,
    pub threshold: Option<f64>,
    pub gini: f64,
    pub samples: usize,
    pub is_leaf: bool,
    pub is_classification: bool,
    pub predicted_class: Option<String>,
    pub predicted_value: Option<f64>,
    pub left_child: Option<usize>,
    pub right_child: Option<usize>,
}

/// Main training engine
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainEngine {
    config: TrainingConfig,
    feature_names: Vec<String>,
    model: Option<TrainedModel>,
    model_bytes: Option<Vec<u8>>,
    metrics: Option<ModelMetrics>,
    is_fitted: bool,
    training_history: Vec<ModelComparison>,
    /// Per-epoch training metrics (only populated by iterative models: SGD)
    #[serde(default)]
    epoch_history: Vec<EpochRecord>,
}

impl TrainEngine {
    /// Create a new training engine
    pub fn new(config: TrainingConfig) -> Self {
        Self {
            config,
            feature_names: Vec::new(),
            model: None,
            model_bytes: None,
            metrics: None,
            is_fitted: false,
            training_history: Vec::new(),
            epoch_history: Vec::new(),
        }
    }

    /// Fit the model to the data
    pub fn fit(&mut self, df: &DataFrame) -> Result<&mut Self> {
        let start = Instant::now();

        // Extract features and target
        let (x, y) = self.prepare_data(df)?;

        // Split data (use stratified split for classification tasks)
        let (x_train, x_val, y_train, y_val) = match self.config.task_type {
            TaskType::BinaryClassification | TaskType::MultiClassification => {
                self.stratified_split(&x, &y)?
            }
            _ => self.train_val_split(&x, &y)?,
        };

        // Train model based on type
        self.train_model(&x_train, &y_train)?;

        // Compute metrics on validation set
        let y_pred = self.predict_internal(&x_val)?;
        let mut metrics = match self.config.task_type {
            TaskType::Regression | TaskType::TimeSeries => {
                ModelMetrics::compute_regression(&y_val, &y_pred)
            }
            _ => ModelMetrics::compute_classification(&y_val, &y_pred, None),
        };

        metrics.training_time_secs = start.elapsed().as_secs_f64();
        metrics.n_features = x.ncols();
        metrics.n_samples = x.nrows();
        self.metrics = Some(metrics);

        self.is_fitted = true;
        Ok(self)
    }

    /// Make predictions on new data
    pub fn predict(&self, df: &DataFrame) -> Result<Array1<f64>> {
        if !self.is_fitted {
            return Err(AutoMLError::ModelNotFitted);
        }

        let x = self.extract_features(df)?;
        self.predict_internal(&x)
    }

    /// Get training metrics
    pub fn metrics(&self) -> Option<&ModelMetrics> {
        self.metrics.as_ref()
    }

    /// Return the training configuration.
    pub fn config(&self) -> &TrainingConfig {
        &self.config
    }

    /// Get feature names
    pub fn feature_names(&self) -> &[String] {
        &self.feature_names
    }

    /// Returns per-epoch training history. Empty for non-iterative models.
    pub fn epoch_history(&self) -> &[EpochRecord] {
        &self.epoch_history
    }

    /// Save the engine to a file
    pub fn save(&self, path: &str) -> Result<()> {
        let json = serde_json::to_string_pretty(self)?;
        std::fs::write(path, json)?;
        Ok(())
    }

    /// Load an engine from a file
    pub fn load(path: &str) -> Result<Self> {
        let json = std::fs::read_to_string(path)?;
        let engine: Self = serde_json::from_str(&json)?;
        Ok(engine)
    }

    fn prepare_data(&mut self, df: &DataFrame) -> Result<(Array2<f64>, Array1<f64>)> {
        // Get feature columns
        let feature_cols: Vec<String> = match &self.config.feature_columns {
            Some(cols) => cols.clone(),
            None => df
                .get_column_names()
                .into_iter()
                .filter(|name| name.as_str() != self.config.target_column)
                .map(|s| s.to_string())
                .collect(),
        };

        self.feature_names = feature_cols.clone();

        // Extract target (cast to Float64 if needed)
        let target_series = df
            .column(&self.config.target_column)
            .map_err(|_| AutoMLError::FeatureNotFound(self.config.target_column.clone()))?;

        let target_f64 = target_series.cast(&DataType::Float64)
            .map_err(|e| AutoMLError::DataError(e.to_string()))?;

        let y: Array1<f64> = target_f64
            .f64()
            .map_err(|e| AutoMLError::DataError(e.to_string()))?
            .into_iter()
            .map(|v| v.unwrap_or(0.0))
            .collect();

        // Extract features using shared helper
        let x = Self::columns_to_array2(df, &feature_cols)?;

        Ok((x, y))
    }

    pub fn extract_features(&self, df: &DataFrame) -> Result<Array2<f64>> {
        Self::columns_to_array2(df, &self.feature_names)
    }

    /// Shared helper: extract named columns from a DataFrame into a row-major Array2<f64>.
    /// Pre-allocates a flat Vec and fills it row-by-row for cache-friendly access.
    fn columns_to_array2(df: &DataFrame, col_names: &[String]) -> Result<Array2<f64>> {
        let n_rows = df.height();
        let n_cols = col_names.len();

        // Collect all columns as contiguous f64 slices
        let col_data: Vec<Vec<f64>> = col_names
            .iter()
            .map(|col_name| {
                let series = df
                    .column(col_name)
                    .map_err(|_| AutoMLError::FeatureNotFound(col_name.clone()))?;
                let series_f64 = series.cast(&DataType::Float64)
                    .map_err(|e| AutoMLError::DataError(e.to_string()))?;
                let values: Vec<f64> = series_f64
                    .f64()
                    .map_err(|e| AutoMLError::DataError(e.to_string()))?
                    .into_iter()
                    .map(|v| v.unwrap_or(0.0))
                    .collect();
                Ok(values)
            })
            .collect::<Result<Vec<Vec<f64>>>>()?;

        // Build flat row-major buffer with a single allocation
        let col_refs: Vec<&[f64]> = col_data.iter().map(|c| c.as_slice()).collect();
        let mut data = Vec::with_capacity(n_rows * n_cols);
        for r in 0..n_rows {
            for col in &col_refs {
                data.push(col[r]);
            }
        }
        Array2::from_shape_vec((n_rows, n_cols), data)
            .map_err(|e| AutoMLError::DataError(e.to_string()))
    }

    fn train_val_split(
        &self,
        x: &Array2<f64>,
        y: &Array1<f64>,
    ) -> Result<(Array2<f64>, Array2<f64>, Array1<f64>, Array1<f64>)> {
        let n = x.nrows();
        let n_cols = x.ncols();
        let val_size = (n as f64 * self.config.validation_split) as usize;
        let train_size = n - val_size;

        // Copy into contiguous arrays via pre-allocated flat buffers (avoids slice-to-owned overhead)
        let x_raw = x.as_slice().unwrap_or(&[]);
        let is_contiguous = !x_raw.is_empty();

        let (x_train, x_val) = if is_contiguous {
            // Data is already contiguous row-major — fast memcpy splits
            let train_data = x_raw[..train_size * n_cols].to_vec();
            let val_data = x_raw[train_size * n_cols..].to_vec();
            (
                Array2::from_shape_vec((train_size, n_cols), train_data)
                    .map_err(|e| AutoMLError::DataError(e.to_string()))?,
                Array2::from_shape_vec((val_size, n_cols), val_data)
                    .map_err(|e| AutoMLError::DataError(e.to_string()))?,
            )
        } else {
            (
                x.slice(ndarray::s![..train_size, ..]).to_owned(),
                x.slice(ndarray::s![train_size.., ..]).to_owned(),
            )
        };

        let y_train = y.slice(ndarray::s![..train_size]).to_owned();
        let y_val = y.slice(ndarray::s![train_size..]).to_owned();

        Ok((x_train, x_val, y_train, y_val))
    }

    fn train_model(&mut self, x: &Array2<f64>, y: &Array1<f64>) -> Result<()> {
        let model = match (&self.config.task_type, &self.config.model_type) {
            // Regression models
            (TaskType::Regression, ModelType::LinearRegression) => {
                let mut model = LinearRegression::new();
                model.fit(x, y)?;
                TrainedModel::LinearRegression(model)
            }
            (TaskType::Regression, ModelType::DecisionTree) => {
                let mut model = DecisionTree::new_regressor();
                if let Some(depth) = self.config.max_depth {
                    model = model.with_max_depth(depth);
                }
                model.fit(x, y)?;
                TrainedModel::DecisionTreeRegressor(model)
            }
            (TaskType::Regression, ModelType::RandomForest) => {
                let mut model = RandomForest::new_regressor(self.config.n_estimators.unwrap_or(100));
                if let Some(depth) = self.config.max_depth {
                    model = model.with_max_depth(depth);
                }
                if let Some(seed) = self.config.random_seed {
                    model = model.with_random_state(seed);
                }
                model.fit(x, y)?;
                TrainedModel::RandomForestRegressor(model)
            }
            (TaskType::Regression, ModelType::GradientBoosting) => {
                let config = GradientBoostingConfig {
                    n_estimators: self.config.n_estimators.unwrap_or(100),
                    learning_rate: self.config.learning_rate.unwrap_or(0.1),
                    max_depth: self.config.max_depth.unwrap_or(6),
                    random_state: self.config.random_seed,
                    ..Default::default()
                };
                let mut model = GradientBoostingRegressor::new(config);
                model.fit(x, y)?;
                TrainedModel::GradientBoostingRegressor(model)
            }
            (TaskType::Regression, ModelType::KNN) => {
                let config = KNNConfig {
                    n_neighbors: self.config.n_neighbors.unwrap_or(5),
                    ..Default::default()
                };
                let mut model = KNNRegressor::new(config);
                model.fit(x, y)?;
                TrainedModel::KNNRegressor(model)
            }
            
            // Classification models
            (TaskType::BinaryClassification | TaskType::MultiClassification, ModelType::LogisticRegression) => {
                let mut model = LogisticRegression::new()
                    .with_max_iter(self.config.max_iter.unwrap_or(1000));
                if let Some(lr) = self.config.learning_rate {
                    model = model.with_learning_rate(lr);
                }
                model.fit(x, y)?;
                TrainedModel::LogisticRegression(model)
            }
            (TaskType::BinaryClassification | TaskType::MultiClassification, ModelType::DecisionTree) => {
                let mut model = DecisionTree::new_classifier();
                if let Some(depth) = self.config.max_depth {
                    model = model.with_max_depth(depth);
                }
                model.fit(x, y)?;
                TrainedModel::DecisionTreeClassifier(model)
            }
            (TaskType::BinaryClassification | TaskType::MultiClassification, ModelType::RandomForest) => {
                let mut model = RandomForest::new_classifier(self.config.n_estimators.unwrap_or(100));
                if let Some(depth) = self.config.max_depth {
                    model = model.with_max_depth(depth);
                }
                if let Some(seed) = self.config.random_seed {
                    model = model.with_random_state(seed);
                }
                model.fit(x, y)?;
                TrainedModel::RandomForestClassifier(model)
            }
            (TaskType::BinaryClassification | TaskType::MultiClassification, ModelType::GradientBoosting) => {
                let config = GradientBoostingConfig {
                    n_estimators: self.config.n_estimators.unwrap_or(100),
                    learning_rate: self.config.learning_rate.unwrap_or(0.1),
                    max_depth: self.config.max_depth.unwrap_or(6),
                    random_state: self.config.random_seed,
                    ..Default::default()
                };
                let mut model = GradientBoostingClassifier::new(config);
                model.fit(x, y)?;
                TrainedModel::GradientBoostingClassifier(model)
            }
            (TaskType::BinaryClassification | TaskType::MultiClassification, ModelType::KNN) => {
                let config = KNNConfig {
                    n_neighbors: self.config.n_neighbors.unwrap_or(5),
                    ..Default::default()
                };
                let mut model = KNNClassifier::new(config);
                model.fit(x, y)?;
                TrainedModel::KNNClassifier(model)
            }
            
            // Naive Bayes (classification only, use Gaussian variant)
            (TaskType::BinaryClassification | TaskType::MultiClassification, ModelType::NaiveBayes) => {
                let mut model = GaussianNaiveBayes::new();
                model.fit(x, y)?;
                TrainedModel::GaussianNaiveBayes(model)
            }
            
            // SVM models
            (TaskType::Regression, ModelType::SVM) => {
                let config = SVMConfig {
                    c: 1.0,
                    kernel: KernelType::RBF { gamma: 1.0 / x.ncols() as f64 },
                    max_iter: self.config.max_iter.unwrap_or(1000),
                    random_state: self.config.random_seed,
                    ..Default::default()
                };
                let mut model = SVMRegressor::new(config);
                model.fit(x, y)?;
                TrainedModel::SVMRegressor(model)
            }
            (TaskType::BinaryClassification | TaskType::MultiClassification, ModelType::SVM) => {
                let config = SVMConfig {
                    c: 1.0,
                    kernel: KernelType::RBF { gamma: 1.0 / x.ncols() as f64 },
                    max_iter: self.config.max_iter.unwrap_or(1000),
                    random_state: self.config.random_seed,
                    ..Default::default()
                };
                let mut model = SVMClassifier::new(config);
                model.fit(x, y)?;
                TrainedModel::SVMClassifier(model)
            }
            
            // Ridge Regression
            (TaskType::Regression, ModelType::Ridge) => {
                let mut model = RidgeRegression::new(self.config.reg_lambda);
                model.fit(x, y)?;
                TrainedModel::RidgeRegression(model)
            }
            
            // Lasso Regression
            (TaskType::Regression, ModelType::Lasso) => {
                let mut model = LassoRegression::new(self.config.reg_alpha.max(0.01));
                if let Some(max_iter) = self.config.max_iter {
                    model = model.with_max_iter(max_iter);
                }
                model.fit(x, y)?;
                TrainedModel::LassoRegression(model)
            }
            
            // Elastic Net
            (TaskType::Regression, ModelType::ElasticNet) => {
                let alpha = self.config.reg_alpha.max(0.01);
                let l1_ratio = if self.config.reg_alpha > 0.0 && self.config.reg_lambda > 0.0 {
                    self.config.reg_alpha / (self.config.reg_alpha + self.config.reg_lambda)
                } else {
                    0.5
                };
                let mut model = ElasticNetRegression::new(alpha, l1_ratio);
                if let Some(max_iter) = self.config.max_iter {
                    model = model.with_max_iter(max_iter);
                }
                model.fit(x, y)?;
                TrainedModel::ElasticNetRegression(model)
            }
            
            // AdaBoost (classification only)
            (TaskType::BinaryClassification | TaskType::MultiClassification, ModelType::AdaBoost) => {
                let mut model = AdaBoostClassifier::new(
                    self.config.n_estimators.unwrap_or(50),
                    self.config.learning_rate.unwrap_or(1.0),
                );
                model.fit(x, y)?;
                TrainedModel::AdaBoostClassifier(model)
            }
            
            // Extra Trees
            (TaskType::Regression, ModelType::ExtraTrees) => {
                let mut model = ExtraTrees::new_regressor(self.config.n_estimators.unwrap_or(100));
                if let Some(depth) = self.config.max_depth {
                    model = model.with_max_depth(depth);
                }
                if let Some(seed) = self.config.random_seed {
                    model = model.with_random_state(seed);
                }
                model.fit(x, y)?;
                TrainedModel::ExtraTreesRegressor(model)
            }
            (TaskType::BinaryClassification | TaskType::MultiClassification, ModelType::ExtraTrees) => {
                let mut model = ExtraTrees::new_classifier(self.config.n_estimators.unwrap_or(100));
                if let Some(depth) = self.config.max_depth {
                    model = model.with_max_depth(depth);
                }
                if let Some(seed) = self.config.random_seed {
                    model = model.with_random_state(seed);
                }
                model.fit(x, y)?;
                TrainedModel::ExtraTreesClassifier(model)
            }
            
            // XGBoost
            (TaskType::Regression, ModelType::XGBoost) => {
                let config = XGBoostConfig {
                    n_estimators: self.config.n_estimators.unwrap_or(100),
                    learning_rate: self.config.learning_rate.unwrap_or(0.3),
                    max_depth: self.config.max_depth.unwrap_or(6),
                    reg_lambda: self.config.reg_lambda,
                    reg_alpha: self.config.reg_alpha,
                    subsample: self.config.subsample,
                    colsample_bytree: self.config.colsample_bytree,
                    random_state: self.config.random_seed,
                    ..Default::default()
                };
                let mut model = XGBoostRegressor::new(config);
                model.fit(x, y)?;
                TrainedModel::XGBoostRegressor(model)
            }
            (TaskType::BinaryClassification | TaskType::MultiClassification, ModelType::XGBoost) => {
                let config = XGBoostConfig {
                    n_estimators: self.config.n_estimators.unwrap_or(100),
                    learning_rate: self.config.learning_rate.unwrap_or(0.3),
                    max_depth: self.config.max_depth.unwrap_or(6),
                    reg_lambda: self.config.reg_lambda,
                    reg_alpha: self.config.reg_alpha,
                    subsample: self.config.subsample,
                    colsample_bytree: self.config.colsample_bytree,
                    random_state: self.config.random_seed,
                    ..Default::default()
                };
                let mut model = XGBoostClassifier::new(config);
                model.fit(x, y)?;
                TrainedModel::XGBoostClassifier(model)
            }

            // LightGBM
            (TaskType::Regression | TaskType::TimeSeries, ModelType::LightGBM) => {
                let config = LightGBMConfig {
                    n_estimators: self.config.n_estimators.unwrap_or(100),
                    learning_rate: self.config.learning_rate.unwrap_or(0.1),
                    max_leaves: 31,
                    max_depth: self.config.max_depth,
                    reg_lambda: self.config.reg_lambda,
                    reg_alpha: self.config.reg_alpha,
                    subsample: self.config.subsample,
                    colsample_bytree: self.config.colsample_bytree,
                    random_state: self.config.random_seed,
                    ..Default::default()
                };
                let mut model = LightGBMRegressor::new(config);
                model.fit(x, y)?;
                TrainedModel::LightGBMRegressor(model)
            }
            (TaskType::BinaryClassification | TaskType::MultiClassification, ModelType::LightGBM) => {
                let config = LightGBMConfig {
                    n_estimators: self.config.n_estimators.unwrap_or(100),
                    learning_rate: self.config.learning_rate.unwrap_or(0.1),
                    max_leaves: 31,
                    max_depth: self.config.max_depth,
                    reg_lambda: self.config.reg_lambda,
                    reg_alpha: self.config.reg_alpha,
                    subsample: self.config.subsample,
                    colsample_bytree: self.config.colsample_bytree,
                    random_state: self.config.random_seed,
                    ..Default::default()
                };
                let mut model = LightGBMClassifier::new(config);
                model.fit(x, y)?;
                TrainedModel::LightGBMClassifier(model)
            }

            // CatBoost
            (TaskType::Regression | TaskType::TimeSeries, ModelType::CatBoost) => {
                let config = CatBoostConfig {
                    n_estimators: self.config.n_estimators.unwrap_or(100),
                    learning_rate: self.config.learning_rate.unwrap_or(0.1),
                    max_depth: self.config.max_depth.unwrap_or(6),
                    reg_lambda: self.config.reg_lambda.max(3.0),
                    subsample: self.config.subsample,
                    random_state: self.config.random_seed,
                };
                let mut model = CatBoostRegressor::new(config);
                model.fit(x, y)?;
                TrainedModel::CatBoostRegressor(model)
            }
            (TaskType::BinaryClassification | TaskType::MultiClassification, ModelType::CatBoost) => {
                let config = CatBoostConfig {
                    n_estimators: self.config.n_estimators.unwrap_or(100),
                    learning_rate: self.config.learning_rate.unwrap_or(0.1),
                    max_depth: self.config.max_depth.unwrap_or(6),
                    reg_lambda: self.config.reg_lambda.max(3.0),
                    subsample: self.config.subsample,
                    random_state: self.config.random_seed,
                };
                let mut model = CatBoostClassifier::new(config);
                model.fit(x, y)?;
                TrainedModel::CatBoostClassifier(model)
            }

            // SGD
            (TaskType::Regression | TaskType::TimeSeries, ModelType::SGD) => {
                // Use a conservative default learning rate for SGD to avoid divergence;
                // cap user-supplied rates above 0.01 for regression to maintain stability.
                let raw_lr = self.config.learning_rate.unwrap_or(0.01);
                let eta0 = if raw_lr > 0.01 {
                    eprintln!("[automl] SGD regression: clamping learning_rate {raw_lr} → 0.01 to prevent divergence");
                    0.01
                } else {
                    raw_lr
                };
                let config = SGDConfig {
                    max_iter: self.config.max_iter.unwrap_or(1000),
                    eta0,
                    alpha: self.config.reg_alpha,
                    random_state: self.config.random_seed,
                    ..Default::default()
                };
                let mut model = SGDRegressor::new(config);
                model.fit(x, y)?;
                self.epoch_history = std::mem::take(&mut model.epoch_records);
                TrainedModel::SGDRegressor(model)
            }
            (TaskType::BinaryClassification | TaskType::MultiClassification, ModelType::SGD) => {
                let config = SGDConfig {
                    max_iter: self.config.max_iter.unwrap_or(1000),
                    eta0: self.config.learning_rate.unwrap_or(0.01),
                    alpha: self.config.reg_alpha,
                    random_state: self.config.random_seed,
                    ..Default::default()
                };
                let mut model = SGDClassifier::new(config);
                model.fit(x, y)?;
                self.epoch_history = std::mem::take(&mut model.epoch_records);
                TrainedModel::SGDClassifier(model)
            }

            // Gaussian Process
            (TaskType::Regression | TaskType::TimeSeries, ModelType::GaussianProcess) => {
                let config = GPConfig::default();
                let mut model = GaussianProcessRegressor::new(config);
                model.fit(x, y)?;
                TrainedModel::GaussianProcessRegressor(model)
            }

            // Polynomial Regression
            (TaskType::Regression | TaskType::TimeSeries, ModelType::PolynomialRegression) => {
                let degree = self.config.max_depth.unwrap_or(2).min(4);
                let alpha = self.config.reg_alpha.max(0.01);
                let mut model = PolynomialRegression::new(degree, alpha);
                model.fit(x, y)?;
                TrainedModel::PolynomialRegression(model)
            }
            
            // Clustering (unsupervised — y is ignored, labels become predictions)
            (TaskType::Clustering, ModelType::KMeans) | (_, ModelType::KMeans) => {
                let mut model = KMeans::new(self.config.n_neighbors.unwrap_or(3));
                if let Some(max_iter) = self.config.max_iter {
                    model = model.with_max_iter(max_iter);
                }
                if let Some(seed) = self.config.random_seed {
                    model = model.with_random_state(seed);
                }
                model.fit(x)?;
                TrainedModel::KMeans(model)
            }
            (TaskType::Clustering, ModelType::DBSCAN) | (_, ModelType::DBSCAN) => {
                let mut model = DBSCAN::new(0.5, 5);
                model.fit(x)?;
                TrainedModel::DBSCAN(model)
            }
            (TaskType::Clustering, ModelType::SOM) | (_, ModelType::SOM) => {
                let side = (x.nrows() as f64).sqrt().ceil() as usize;
                let side = side.max(2).min(50);
                let mut model = SOM::new(side, side);
                if let Some(seed) = self.config.random_seed {
                    model = model.with_random_state(seed);
                }
                if let Some(max_iter) = self.config.max_iter {
                    model = model.with_max_iter(max_iter);
                }
                model.fit(x)?;
                TrainedModel::SelfOrganizingMap(model)
            }

            // Default cases - use appropriate model based on task type
            (TaskType::Regression, _) => {
                let mut model = LinearRegression::new();
                model.fit(x, y)?;
                TrainedModel::LinearRegression(model)
            }
            (TaskType::BinaryClassification | TaskType::MultiClassification, _) => {
                let mut model = LogisticRegression::new();
                model.fit(x, y)?;
                TrainedModel::LogisticRegression(model)
            }
            
            // TimeSeries defaults to linear regression for now
            (TaskType::TimeSeries, _) => {
                let mut model = LinearRegression::new();
                model.fit(x, y)?;
                TrainedModel::LinearRegression(model)
            }
            
            // Clustering defaults to KMeans
            (TaskType::Clustering, _) => {
                let mut model = KMeans::new(self.config.n_neighbors.unwrap_or(3));
                if let Some(seed) = self.config.random_seed {
                    model = model.with_random_state(seed);
                }
                model.fit(x)?;
                TrainedModel::KMeans(model)
            }
        };

        self.model = Some(model);
        Ok(())
    }

    fn predict_internal(&self, x: &Array2<f64>) -> Result<Array1<f64>> {
        let model = self.model.as_ref().ok_or(AutoMLError::ModelNotFitted)?;

        let predictions = match model {
            TrainedModel::LinearRegression(m) => m.predict(x)?,
            TrainedModel::LogisticRegression(m) => m.predict(x)?,
            TrainedModel::DecisionTreeClassifier(m) => m.predict(x)?,
            TrainedModel::DecisionTreeRegressor(m) => m.predict(x)?,
            TrainedModel::RandomForestClassifier(m) => m.predict(x)?,
            TrainedModel::RandomForestRegressor(m) => m.predict(x)?,
            TrainedModel::GradientBoostingClassifier(m) => m.predict(x)?,
            TrainedModel::GradientBoostingRegressor(m) => m.predict(x)?,
            TrainedModel::KNNClassifier(m) => m.predict(x),
            TrainedModel::KNNRegressor(m) => m.predict(x),
            TrainedModel::GaussianNaiveBayes(m) => m.predict(x),
            TrainedModel::MultinomialNaiveBayes(m) => m.predict(x),
            TrainedModel::SVMClassifier(m) => m.predict(x)?,
            TrainedModel::SVMRegressor(m) => m.predict(x)?,
            TrainedModel::RidgeRegression(m) => m.predict(x)?,
            TrainedModel::LassoRegression(m) => m.predict(x)?,
            TrainedModel::ElasticNetRegression(m) => m.predict(x)?,
            TrainedModel::AdaBoostClassifier(m) => m.predict(x)?,
            TrainedModel::ExtraTreesClassifier(m) => m.predict(x)?,
            TrainedModel::ExtraTreesRegressor(m) => m.predict(x)?,
            TrainedModel::XGBoostRegressor(m) => m.predict(x)?,
            TrainedModel::XGBoostClassifier(m) => m.predict(x)?,
            TrainedModel::LightGBMRegressor(m) => m.predict(x)?,
            TrainedModel::LightGBMClassifier(m) => m.predict(x)?,
            TrainedModel::CatBoostRegressor(m) => m.predict(x)?,
            TrainedModel::CatBoostClassifier(m) => m.predict(x)?,
            TrainedModel::SGDRegressor(m) => m.predict(x)?,
            TrainedModel::SGDClassifier(m) => m.predict(x)?,
            TrainedModel::GaussianProcessRegressor(m) => m.predict(x)?,
            TrainedModel::PolynomialRegression(m) => m.predict(x)?,
            TrainedModel::KMeans(m) => m.predict(x)?,
            TrainedModel::DBSCAN(m) => m.predict(x)?,
            TrainedModel::SelfOrganizingMap(m) => m.predict(x)?,
        };

        Ok(predictions)
    }

    fn predict_proba_internal(&self, x: &Array2<f64>) -> Result<Array2<f64>> {
        let model = self.model.as_ref().ok_or(AutoMLError::ModelNotFitted)?;

        let proba = match model {
            TrainedModel::LogisticRegression(m) => {
                let p = m.predict_proba(x)?;
                let n = p.len();
                let mut out = Array2::zeros((n, 2));
                for (i, &pi) in p.iter().enumerate() {
                    out[[i, 0]] = 1.0 - pi;
                    out[[i, 1]] = pi;
                }
                out
            }
            TrainedModel::GaussianNaiveBayes(m) => m.predict_proba(x),
            TrainedModel::MultinomialNaiveBayes(m) => {
                // MultinomialNaiveBayes has predict_log_proba; exponentiate for probabilities
                let log_proba = m.predict_log_proba(x);
                log_proba.mapv(|v| v.exp())
            }
            TrainedModel::RandomForestClassifier(m) => m.predict_proba(x)?,
            TrainedModel::GradientBoostingClassifier(m) => {
                let p = m.predict_proba(x)?;
                let n = p.len();
                let mut out = Array2::zeros((n, 2));
                for (i, &pi) in p.iter().enumerate() {
                    out[[i, 0]] = 1.0 - pi;
                    out[[i, 1]] = pi;
                }
                out
            }
            TrainedModel::KNNClassifier(m) => m.predict_proba(x),
            TrainedModel::DecisionTreeClassifier(m) => {
                let preds = m.predict(x)?;
                let n = preds.len();
                let mut out = Array2::zeros((n, 2));
                for (i, &v) in preds.iter().enumerate() {
                    let p = 1.0 / (1.0 + (-v).exp());
                    out[[i, 0]] = 1.0 - p;
                    out[[i, 1]] = p;
                }
                out
            }
            TrainedModel::SVMClassifier(m) => {
                let preds = m.predict(x)?;
                let n = preds.len();
                let mut out = Array2::zeros((n, 2));
                for (i, &v) in preds.iter().enumerate() {
                    let p = 1.0 / (1.0 + (-v).exp());
                    out[[i, 0]] = 1.0 - p;
                    out[[i, 1]] = p;
                }
                out
            }
            TrainedModel::AdaBoostClassifier(m) => m.predict_proba(x)?,
            TrainedModel::ExtraTreesClassifier(m) => m.predict_proba(x)?,
            TrainedModel::XGBoostClassifier(m) => {
                let p = m.predict_proba(x)?;
                let n = p.len();
                let mut out = Array2::zeros((n, 2));
                for (i, &pi) in p.iter().enumerate() {
                    out[[i, 0]] = 1.0 - pi;
                    out[[i, 1]] = pi;
                }
                out
            }
            TrainedModel::LightGBMClassifier(m) => {
                m.predict_proba(x)?
            }
            TrainedModel::CatBoostClassifier(m) => {
                m.predict_proba(x)?
            }
            TrainedModel::SGDClassifier(m) => {
                m.predict_proba(x)?
            }
            TrainedModel::LinearRegression(_)
            | TrainedModel::RidgeRegression(_)
            | TrainedModel::LassoRegression(_)
            | TrainedModel::ElasticNetRegression(_)
            | TrainedModel::PolynomialRegression(_)
            | TrainedModel::DecisionTreeRegressor(_)
            | TrainedModel::RandomForestRegressor(_)
            | TrainedModel::GradientBoostingRegressor(_)
            | TrainedModel::KNNRegressor(_)
            | TrainedModel::SVMRegressor(_)
            | TrainedModel::ExtraTreesRegressor(_)
            | TrainedModel::XGBoostRegressor(_)
            | TrainedModel::LightGBMRegressor(_)
            | TrainedModel::CatBoostRegressor(_)
            | TrainedModel::SGDRegressor(_)
            | TrainedModel::GaussianProcessRegressor(_)
            | TrainedModel::KMeans(_)
            | TrainedModel::DBSCAN(_)
            | TrainedModel::SelfOrganizingMap(_) => {
                return Err(AutoMLError::TrainingError(
                    "predict_proba is only supported for classification models".to_string(),
                ));
            }
        };

        Ok(proba)
    }

    /// Predict class probabilities from a DataFrame
    pub fn predict_proba(&self, df: &DataFrame) -> Result<Array2<f64>> {
        if !self.is_fitted {
            return Err(AutoMLError::ModelNotFitted);
        }

        let x = self.extract_features(df)?;
        self.predict_proba_internal(&x)
    }

    /// Predict class probabilities from a raw array (for InferenceEngine)
    pub fn predict_proba_array(&self, x: &Array2<f64>) -> Result<Array2<f64>> {
        self.predict_proba_internal(x)
    }

    /// Get feature importances (if available)
    pub fn feature_importances(&self) -> Option<Array1<f64>> {
        let model = self.model.as_ref()?;
        let n_feat = self.feature_names.len();

        match model {
            TrainedModel::DecisionTreeClassifier(m) | TrainedModel::DecisionTreeRegressor(m) => {
                m.feature_importances().cloned()
            }
            TrainedModel::RandomForestClassifier(m) | TrainedModel::RandomForestRegressor(m) => {
                m.feature_importances().cloned()
            }
            TrainedModel::GradientBoostingRegressor(m) => {
                Some(Array1::from_vec(m.feature_importances().to_vec()))
            }
            TrainedModel::GradientBoostingClassifier(m) => {
                Some(Array1::from_vec(m.feature_importances().to_vec()))
            }
            TrainedModel::ExtraTreesClassifier(m) | TrainedModel::ExtraTreesRegressor(m) => {
                m.feature_importances()
            }
            TrainedModel::AdaBoostClassifier(m) => {
                m.feature_importances(n_feat)
            }
            TrainedModel::XGBoostRegressor(m) => {
                m.feature_importances()
            }
            TrainedModel::XGBoostClassifier(m) => {
                m.feature_importances()
            }
            TrainedModel::LightGBMRegressor(m) => {
                m.feature_importances(n_feat)
            }
            TrainedModel::LightGBMClassifier(m) => {
                m.feature_importances(n_feat)
            }
            TrainedModel::CatBoostRegressor(m) => {
                m.feature_importances(n_feat)
            }
            TrainedModel::CatBoostClassifier(m) => {
                m.feature_importances(n_feat)
            }
            TrainedModel::LinearRegression(m) => {
                m.coefficients.as_ref().map(|c| c.mapv(f64::abs))
            }
            TrainedModel::LogisticRegression(m) => {
                m.coefficients.as_ref().map(|c| c.mapv(f64::abs))
            }
            TrainedModel::RidgeRegression(m) => {
                m.coefficients.as_ref().map(|c| c.mapv(f64::abs))
            }
            TrainedModel::LassoRegression(m) => {
                m.coefficients.as_ref().map(|c| c.mapv(f64::abs))
            }
            TrainedModel::ElasticNetRegression(m) => {
                m.coefficients.as_ref().map(|c| c.mapv(f64::abs))
            }
            TrainedModel::SGDRegressor(m) => {
                m.weights.as_ref().map(|w| w.mapv(f64::abs))
            }
            TrainedModel::SGDClassifier(m) => {
                m.weights.as_ref().map(|w| w.mapv(f64::abs))
            }
            _ => None,
        }
    }

    /// Train on arrays directly and predict on test arrays (for cross-validation)
    pub(crate) fn fit_predict_arrays(
        config: &TrainingConfig,
        x_train: &Array2<f64>,
        y_train: &Array1<f64>,
        x_test: &Array2<f64>,
    ) -> Result<Array1<f64>> {
        let mut engine = TrainEngine::new(config.clone());
        engine.feature_names = (0..x_train.ncols()).map(|i| format!("f{}", i)).collect();
        engine.train_model(x_train, y_train)?;
        engine.is_fitted = true;
        engine.predict_internal(x_test)
    }

    /// Get the trained model
    pub fn model(&self) -> Option<&TrainedModel> {
        self.model.as_ref()
    }

    /// Stratified train/test split that preserves class proportions
    fn stratified_split(
        &self,
        x: &Array2<f64>,
        y: &Array1<f64>,
    ) -> Result<(Array2<f64>, Array2<f64>, Array1<f64>, Array1<f64>)> {
        let _n = x.nrows();
        let val_ratio = self.config.validation_split;

        // Group indices by class label
        let mut class_indices: HashMap<i64, Vec<usize>> = HashMap::new();
        for (i, &label) in y.iter().enumerate() {
            let key = label as i64;
            class_indices.entry(key).or_default().push(i);
        }

        let mut train_indices = Vec::new();
        let mut val_indices = Vec::new();

        for (_class, indices) in &class_indices {
            let class_val_size = ((indices.len() as f64) * val_ratio).max(1.0) as usize;
            let class_val_size = class_val_size.min(indices.len().saturating_sub(1));
            let split_point = indices.len() - class_val_size;
            train_indices.extend_from_slice(&indices[..split_point]);
            val_indices.extend_from_slice(&indices[split_point..]);
        }

        if train_indices.is_empty() || val_indices.is_empty() {
            return Err(AutoMLError::DataError(
                "Stratified split resulted in empty train or validation set".to_string(),
            ));
        }

        // Use ndarray select for efficient indexed extraction
        let x_train = x.select(Axis(0), &train_indices);
        let x_val = x.select(Axis(0), &val_indices);
        let y_train = y.select(Axis(0), &train_indices);
        let y_val = y.select(Axis(0), &val_indices);

        Ok((x_train, x_val, y_train, y_val))
    }

    /// Train an ensemble of multiple model types and combine their predictions
    pub fn train_ensemble(
        &mut self,
        df: &DataFrame,
        target: &str,
        model_types: &[ModelType],
        strategy: EnsembleStrategy,
    ) -> Result<EnsembleResult> {
        // Prepare data once
        self.config.target_column = target.to_string();
        let (x, y) = self.prepare_data(df)?;
        let (x_train, x_val, y_train, y_val) = match self.config.task_type {
            TaskType::BinaryClassification | TaskType::MultiClassification => {
                self.stratified_split(&x, &y)?
            }
            _ => self.train_val_split(&x, &y)?,
        };

        let mut model_names = Vec::new();
        let mut individual_metrics = HashMap::new();
        let mut all_predictions: Vec<Array1<f64>> = Vec::new();

        let original_model_type = self.config.model_type.clone();

        for model_type in model_types {
            let model_start = Instant::now();
            self.config.model_type = model_type.clone();
            self.train_model(&x_train, &y_train)?;

            let y_pred = self.predict_internal(&x_val)?;
            let metric_value = match self.config.task_type {
                TaskType::Regression | TaskType::TimeSeries => {
                    let m = ModelMetrics::compute_regression(&y_val, &y_pred);
                    m.r2.unwrap_or(0.0)
                }
                _ => {
                    let m = ModelMetrics::compute_classification(&y_val, &y_pred, None);
                    m.accuracy.unwrap_or(0.0)
                }
            };

            let name = format!("{:?}", model_type);
            individual_metrics.insert(name.clone(), metric_value);
            model_names.push(name.clone());
            all_predictions.push(y_pred);

            // Record in training history
            let mut metrics_map = HashMap::new();
            metrics_map.insert("primary_metric".to_string(), metric_value);
            let serialized_size = serde_json::to_vec(&self.model).map(|v| v.len()).unwrap_or(0);
            self.training_history.push(ModelComparison {
                model_name: name,
                task_type: self.config.task_type.clone(),
                metrics: metrics_map,
                training_time_secs: model_start.elapsed().as_secs_f64(),
                model_size_bytes: serialized_size,
            });
        }

        // Combine predictions based on strategy
        let ensemble_pred = match strategy {
            EnsembleStrategy::Averaging => {
                let n = all_predictions.len() as f64;
                let mut avg = Array1::zeros(all_predictions[0].len());
                for pred in &all_predictions {
                    avg = avg + pred;
                }
                avg / n
            }
            EnsembleStrategy::Voting => {
                // Majority vote (round each prediction, take mode)
                let n_samples = all_predictions[0].len();
                Array1::from_iter((0..n_samples).map(|i| {
                    let mut votes: HashMap<i64, usize> = HashMap::new();
                    for pred in &all_predictions {
                        let v = pred[i].round() as i64;
                        *votes.entry(v).or_insert(0) += 1;
                    }
                    votes
                        .into_iter()
                        .max_by_key(|&(_, count)| count)
                        .map(|(val, _)| val as f64)
                        .unwrap_or(0.0)
                }))
            }
            EnsembleStrategy::Stacking => {
                // Simple stacking: average as meta-learner
                let n = all_predictions.len() as f64;
                let mut avg = Array1::zeros(all_predictions[0].len());
                for pred in &all_predictions {
                    avg = avg + pred;
                }
                avg / n
            }
        };

        let ensemble_metric = match self.config.task_type {
            TaskType::Regression | TaskType::TimeSeries => {
                let m = ModelMetrics::compute_regression(&y_val, &ensemble_pred);
                m.r2.unwrap_or(0.0)
            }
            _ => {
                let m = ModelMetrics::compute_classification(&y_val, &ensemble_pred, None);
                m.accuracy.unwrap_or(0.0)
            }
        };

        // Restore original model type
        self.config.model_type = original_model_type;

        Ok(EnsembleResult {
            models: model_names,
            individual_metrics,
            ensemble_metric,
            strategy,
        })
    }

    /// Get a performance comparison of all models trained in this engine's history
    pub fn get_performance_comparison(&self) -> Vec<ModelComparison> {
        let mut comparisons = self.training_history.clone();

        // Include the current model if fitted and not already in history
        if self.is_fitted {
            if let Some(ref metrics) = self.metrics {
                let mut metrics_map = HashMap::new();
                if let Some(v) = metrics.accuracy { metrics_map.insert("accuracy".to_string(), v); }
                if let Some(v) = metrics.precision { metrics_map.insert("precision".to_string(), v); }
                if let Some(v) = metrics.recall { metrics_map.insert("recall".to_string(), v); }
                if let Some(v) = metrics.f1_score { metrics_map.insert("f1_score".to_string(), v); }
                if let Some(v) = metrics.auc_roc { metrics_map.insert("auc_roc".to_string(), v); }
                if let Some(v) = metrics.mse { metrics_map.insert("mse".to_string(), v); }
                if let Some(v) = metrics.rmse { metrics_map.insert("rmse".to_string(), v); }
                if let Some(v) = metrics.mae { metrics_map.insert("mae".to_string(), v); }
                if let Some(v) = metrics.r2 { metrics_map.insert("r2".to_string(), v); }

                let serialized_size = serde_json::to_vec(&self.model).map(|v| v.len()).unwrap_or(0);
                let current = ModelComparison {
                    model_name: format!("{:?}", self.config.model_type),
                    task_type: self.config.task_type.clone(),
                    metrics: metrics_map,
                    training_time_secs: metrics.training_time_secs,
                    model_size_bytes: serialized_size,
                };
                comparisons.push(current);
            }
        }

        comparisons
    }

    /// Serialize the decision tree (or first tree of a random forest) into a flat node list
    pub fn tree_nodes(&self) -> Option<Vec<SerializedNode>> {
        use super::decision_tree::TreeNode;
        use std::collections::{VecDeque, HashMap};

        let is_classification = matches!(
            self.config.task_type,
            TaskType::BinaryClassification | TaskType::MultiClassification
        );

        let root = match self.model.as_ref()? {
            TrainedModel::DecisionTreeClassifier(dt)
            | TrainedModel::DecisionTreeRegressor(dt) => dt.root()?,
            TrainedModel::RandomForestClassifier(rf)
            | TrainedModel::RandomForestRegressor(rf) => rf.first_tree()?.root()?,
            _ => return None,
        };

        let feature_names = &self.feature_names;
        let mut nodes: Vec<SerializedNode> = Vec::new();
        let mut id_counter = 0usize;
        let mut queue: VecDeque<(&TreeNode, Option<usize>)> = VecDeque::new();
        queue.push_back((root, None));

        while let Some((node, parent_id)) = queue.pop_front() {
            let current_id = id_counter;
            id_counter += 1;
            match node {
                TreeNode::Leaf { value, n_samples } => {
                    nodes.push(SerializedNode {
                        id: current_id,
                        parent_id,
                        feature_name: None,
                        threshold: None,
                        gini: 0.0,
                        samples: *n_samples,
                        is_leaf: true,
                        is_classification,
                        predicted_class: if is_classification {
                            Some(format!("class_{}", *value as i64))
                        } else {
                            None
                        },
                        predicted_value: if !is_classification { Some(*value) } else { None },
                        left_child: None,
                        right_child: None,
                    });
                }
                TreeNode::Split {
                    feature_idx,
                    threshold,
                    left,
                    right,
                    n_samples,
                    impurity,
                } => {
                    let feat = feature_names
                        .get(*feature_idx)
                        .cloned()
                        .unwrap_or_else(|| format!("feature_{}", feature_idx));
                    nodes.push(SerializedNode {
                        id: current_id,
                        parent_id,
                        feature_name: Some(feat),
                        threshold: Some(*threshold),
                        gini: *impurity,
                        samples: *n_samples,
                        is_leaf: false,
                        is_classification,
                        predicted_class: None,
                        predicted_value: None,
                        left_child: Some(id_counter),
                        right_child: None, // fixed in post-pass
                    });
                    queue.push_back((left, Some(current_id)));
                    queue.push_back((right, Some(current_id)));
                }
            }
        }

        // Post-pass: fix right_child for each split node
        let mut parent_to_children: HashMap<usize, Vec<usize>> = HashMap::new();
        for n in &nodes {
            if let Some(pid) = n.parent_id {
                parent_to_children.entry(pid).or_default().push(n.id);
            }
        }
        for n in &mut nodes {
            if !n.is_leaf {
                let ch = parent_to_children.get(&n.id);
                n.left_child = ch.and_then(|v| v.first()).copied();
                n.right_child = ch.and_then(|v| v.get(1)).copied();
            }
        }

        Some(nodes)
    }

    /// Generate a text report summarizing the trained model
    pub fn generate_report(&self) -> String {
        let mut report = String::new();
        report.push_str("=== AutoML Training Report ===\n\n");

        // Model info
        report.push_str(&format!("Model Type: {:?}\n", self.config.model_type));
        report.push_str(&format!("Task Type:  {:?}\n", self.config.task_type));
        report.push_str(&format!("Fitted:     {}\n\n", self.is_fitted));

        // Data shape info
        if let Some(ref metrics) = self.metrics {
            report.push_str("--- Data Shape ---\n");
            report.push_str(&format!("Samples:  {}\n", metrics.n_samples));
            report.push_str(&format!("Features: {}\n", metrics.n_features));
            report.push_str(&format!("Target:   {}\n\n", self.config.target_column));

            // Training time
            report.push_str("--- Training Time ---\n");
            report.push_str(&format!("{:.4} seconds\n\n", metrics.training_time_secs));

            // Metrics summary
            report.push_str("--- Metrics Summary ---\n");
            if let Some(v) = metrics.accuracy { report.push_str(&format!("Accuracy:  {:.4}\n", v)); }
            if let Some(v) = metrics.precision { report.push_str(&format!("Precision: {:.4}\n", v)); }
            if let Some(v) = metrics.recall { report.push_str(&format!("Recall:    {:.4}\n", v)); }
            if let Some(v) = metrics.f1_score { report.push_str(&format!("F1 Score:  {:.4}\n", v)); }
            if let Some(v) = metrics.auc_roc { report.push_str(&format!("AUC-ROC:   {:.4}\n", v)); }
            if let Some(v) = metrics.mse { report.push_str(&format!("MSE:       {:.4}\n", v)); }
            if let Some(v) = metrics.rmse { report.push_str(&format!("RMSE:      {:.4}\n", v)); }
            if let Some(v) = metrics.mae { report.push_str(&format!("MAE:       {:.4}\n", v)); }
            if let Some(v) = metrics.r2 { report.push_str(&format!("R²:        {:.4}\n", v)); }
            report.push('\n');
        } else {
            report.push_str("No metrics available (model not fitted).\n\n");
        }

        // Feature importance
        if let Some(importances) = self.feature_importances() {
            report.push_str("--- Feature Importance ---\n");
            let mut pairs: Vec<(String, f64)> = self
                .feature_names
                .iter()
                .zip(importances.iter())
                .map(|(name, &imp)| (name.clone(), imp))
                .collect();
            pairs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            for (name, imp) in &pairs {
                report.push_str(&format!("  {:<20} {:.4}\n", name, imp));
            }
            report.push('\n');
        }

        // Feature names
        report.push_str("--- Features ---\n");
        for name in &self.feature_names {
            report.push_str(&format!("  {}\n", name));
        }

        report
    }

    /// Select top-k features by variance
    pub fn select_features(&self, df: &DataFrame, target: &str, k: usize) -> Vec<String> {
        let feature_cols: Vec<String> = df
            .get_column_names()
            .into_iter()
            .filter(|name| name.as_str() != target)
            .map(|s| s.to_string())
            .collect();

        let mut variances: Vec<(String, f64)> = feature_cols
            .into_iter()
            .filter_map(|col_name| {
                let series = df.column(&col_name).ok()?;
                let series_f64 = series.cast(&DataType::Float64).ok()?;
                let values: Vec<f64> = series_f64
                    .f64()
                    .ok()?
                    .into_iter()
                    .filter_map(|v| v)
                    .collect();
                if values.is_empty() {
                    return Some((col_name, 0.0));
                }
                let mean = values.iter().sum::<f64>() / values.len() as f64;
                let variance = values.iter().map(|v| (v - mean).powi(2)).sum::<f64>()
                    / values.len() as f64;
                Some((col_name, variance))
            })
            .collect();

        variances.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        variances.into_iter().take(k).map(|(name, _)| name).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_data() -> DataFrame {
        DataFrame::new(vec![
            Series::new("feature1".into(), &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]).into(),
            Series::new("feature2".into(), &[1.5, 3.1, 2.8, 7.2, 5.5, 11.0, 8.3, 14.1, 12.7, 19.5]).into(),
            Series::new("target".into(), &[3.0, 6.0, 9.0, 12.0, 15.0, 18.0, 21.0, 24.0, 27.0, 30.0]).into(),
        ])
        .unwrap()
    }

    #[test]
    fn test_engine_creation() {
        let config = TrainingConfig::new(TaskType::Regression, "target");
        let engine = TrainEngine::new(config);
        assert!(!engine.is_fitted);
    }

    #[test]
    fn test_fit_predict() {
        let df = create_test_data();
        let config = TrainingConfig::new(TaskType::Regression, "target");
        let mut engine = TrainEngine::new(config);

        engine.fit(&df).unwrap();
        assert!(engine.is_fitted);
        assert!(engine.metrics().is_some());

        let predictions = engine.predict(&df).unwrap();
        assert_eq!(predictions.len(), 10);
    }

    #[test]
    fn test_feature_names() {
        let df = create_test_data();
        let config = TrainingConfig::new(TaskType::Regression, "target");
        let mut engine = TrainEngine::new(config);

        engine.fit(&df).unwrap();

        let feature_names = engine.feature_names();
        assert_eq!(feature_names.len(), 2);
        assert!(feature_names.contains(&"feature1".to_string()));
        assert!(feature_names.contains(&"feature2".to_string()));
    }
}
