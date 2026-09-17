//! Stacking ensemble method

use crate::error::{AutoMLError, Result};
use crate::training::Model;
use crate::training::cross_validation::{CrossValidator, CVStrategy};
use ndarray::{Array1, Array2};
use serde::{Deserialize, Serialize};

/// Configuration for stacking ensemble
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StackingConfig {
    /// Number of cross-validation folds
    pub n_folds: usize,
    /// Whether to use probabilities (for classification)
    pub use_probabilities: bool,
    /// Whether to include original features in meta-learner input
    pub passthrough: bool,
    /// Random seed
    pub seed: Option<u64>,
}

impl Default for StackingConfig {
    fn default() -> Self {
        Self {
            n_folds: 5,
            use_probabilities: false,
            passthrough: false,
            seed: None,
        }
    }
}

/// Stacking classifier
pub struct StackingClassifier<F>
where
    F: Fn(&Array2<f64>, &Array1<f64>) -> Result<Box<dyn Model>>,
{
    /// Configuration
    config: StackingConfig,
    /// Base model factory functions
    base_model_factories: Vec<F>,
    /// Fitted base models (one per fold per base learner)
    fitted_base_models: Option<Vec<Vec<Box<dyn Model>>>>,
    /// Meta-learner factory
    meta_learner_factory: Option<F>,
    /// Fitted meta-learner
    fitted_meta_learner: Option<Box<dyn Model>>,
    /// Class labels observed during `fit`, ascending. Only populated (and only relevant) when
    /// `config.use_probabilities` is true — it anchors the per-class probability column
    /// layout of the meta-feature matrix so `predict` can reproduce the same layout `fit`
    /// used, without needing `y` again.
    fitted_classes: Option<Vec<i64>>,
}

impl<F> StackingClassifier<F>
where
    F: Fn(&Array2<f64>, &Array1<f64>) -> Result<Box<dyn Model>>,
{
    /// Create a new stacking classifier
    pub fn new(config: StackingConfig) -> Self {
        Self {
            config,
            base_model_factories: Vec::new(),
            fitted_base_models: None,
            meta_learner_factory: None,
            fitted_meta_learner: None,
            fitted_classes: None,
        }
    }

    /// Add a base model
    pub fn add_base_model(mut self, factory: F) -> Self {
        self.base_model_factories.push(factory);
        self
    }

    /// Set the meta-learner
    pub fn with_meta_learner(mut self, factory: F) -> Self {
        self.meta_learner_factory = Some(factory);
        self
    }

    /// Fit the stacking ensemble
    pub fn fit(&mut self, x: &Array2<f64>, y: &Array1<f64>) -> Result<()> {
        if self.base_model_factories.is_empty() {
            return Err(AutoMLError::ValidationError(
                "No base models provided".to_string(),
            ));
        }

        if self.meta_learner_factory.is_none() {
            return Err(AutoMLError::ValidationError(
                "No meta-learner provided".to_string(),
            ));
        }

        let n_samples = x.nrows();
        let n_base_models = self.base_model_factories.len();
        let n_folds = self.config.n_folds;

        // Create cross-validator
        let cv = CrossValidator::new(CVStrategy::KFold {
            n_splits: n_folds,
            shuffle: true,
        })
        .with_random_state(self.config.seed.unwrap_or(42));

        let splits = cv.split(n_samples, None, None)?;

        // `use_probabilities` used to be a dead config flag — meta-features always used hard
        // labels regardless of its value. When it's set, each base model instead contributes
        // one meta-feature column per class (its predicted per-class probabilities, via
        // `Model::predict_proba`, falling back to a one-hot encoding of its hard label for
        // models that don't implement it), rather than a single hard-label column.
        let classes: Vec<i64> = if self.config.use_probabilities {
            let mut c: Vec<i64> = y.iter().map(|&v| v.round() as i64).collect();
            c.sort_unstable();
            c.dedup();
            c
        } else {
            Vec::new()
        };
        let n_classes = classes.len().max(1);
        let cols_per_model = if self.config.use_probabilities { n_classes } else { 1 };

        // Initialize meta-features matrix
        let meta_features_cols = if self.config.passthrough {
            n_base_models * cols_per_model + x.ncols()
        } else {
            n_base_models * cols_per_model
        };
        let mut meta_features = Array2::zeros((n_samples, meta_features_cols));
        let mut fitted_models: Vec<Vec<Box<dyn Model>>> = (0..n_base_models).map(|_| Vec::new()).collect();

        // Generate out-of-fold predictions for each base model
        for (base_idx, factory) in self.base_model_factories.iter().enumerate() {
            for split in &splits {
                // Train on training fold
                let x_train = split.train_indices.iter().map(|&i| x.row(i).to_owned()).collect::<Vec<_>>();
                let y_train: Vec<f64> = split.train_indices.iter().map(|&i| y[i]).collect();

                let x_train = Array2::from_shape_vec(
                    (x_train.len(), x.ncols()),
                    x_train.into_iter().flat_map(|r| r.to_vec()).collect(),
                )?;
                let y_train = Array1::from_vec(y_train);

                let model = factory(&x_train, &y_train)?;

                // Predict on validation fold
                let x_val: Vec<_> = split.test_indices.iter().map(|&i| x.row(i).to_owned()).collect();
                let x_val = Array2::from_shape_vec(
                    (x_val.len(), x.ncols()),
                    x_val.into_iter().flat_map(|r| r.to_vec()).collect(),
                )?;

                if self.config.use_probabilities {
                    let hard_pred = model.predict(&x_val)?;
                    let proba = model.predict_proba(&x_val)?;
                    let proba = match proba {
                        Some(p) if p.nrows() == x_val.nrows() && p.ncols() == n_classes => p,
                        _ => Self::one_hot(&hard_pred, &classes, x_val.nrows(), n_classes),
                    };

                    for (local_idx, &global_idx) in split.test_indices.iter().enumerate() {
                        for c in 0..n_classes {
                            meta_features[[global_idx, base_idx * cols_per_model + c]] =
                                proba[[local_idx, c]];
                        }
                    }
                } else {
                    let predictions = model.predict(&x_val)?;
                    // Store predictions as meta-features
                    for (local_idx, &global_idx) in split.test_indices.iter().enumerate() {
                        meta_features[[global_idx, base_idx]] = predictions[local_idx];
                    }
                }

                fitted_models[base_idx].push(model);
            }
        }

        // Add passthrough features if configured
        if self.config.passthrough {
            let offset = n_base_models * cols_per_model;
            for i in 0..n_samples {
                for j in 0..x.ncols() {
                    meta_features[[i, offset + j]] = x[[i, j]];
                }
            }
        }

        // Fit meta-learner on meta-features
        let meta_factory = self.meta_learner_factory.as_ref().unwrap();
        let meta_learner = meta_factory(&meta_features, y)?;

        self.fitted_base_models = Some(fitted_models);
        self.fitted_meta_learner = Some(meta_learner);
        self.fitted_classes = if self.config.use_probabilities {
            Some(classes)
        } else {
            None
        };

        Ok(())
    }

    /// One-hot encode a model's hard label predictions against the fitted `classes` list.
    /// Used as the probability fallback for base models that don't implement
    /// `Model::predict_proba` when `use_probabilities` is enabled.
    fn one_hot(
        hard_pred: &Array1<f64>,
        classes: &[i64],
        n_samples: usize,
        n_classes: usize,
    ) -> Array2<f64> {
        let mut onehot = Array2::zeros((n_samples, n_classes));
        for (i, &label) in hard_pred.iter().enumerate() {
            let li = label.round() as i64;
            if let Some(col) = classes.iter().position(|&c| c == li) {
                onehot[[i, col]] = 1.0;
            }
        }
        onehot
    }

    /// Make predictions
    pub fn predict(&self, x: &Array2<f64>) -> Result<Array1<f64>> {
        let fitted_models = self.fitted_base_models.as_ref().ok_or_else(|| {
            AutoMLError::ValidationError("Model not fitted".to_string())
        })?;

        let meta_learner = self.fitted_meta_learner.as_ref().ok_or_else(|| {
            AutoMLError::ValidationError("Model not fitted".to_string())
        })?;

        let n_samples = x.nrows();
        let n_base_models = fitted_models.len();

        let classes = self.fitted_classes.as_deref().unwrap_or(&[]);
        let n_classes = classes.len().max(1);
        let cols_per_model = if self.config.use_probabilities { n_classes } else { 1 };

        // Get predictions from each base model (average across folds), using the same
        // meta-feature column layout `fit` used (hard label vs. per-class probabilities).
        let meta_features_cols = if self.config.passthrough {
            n_base_models * cols_per_model + x.ncols()
        } else {
            n_base_models * cols_per_model
        };
        let mut meta_features = Array2::zeros((n_samples, meta_features_cols));

        for (base_idx, fold_models) in fitted_models.iter().enumerate() {
            if self.config.use_probabilities {
                // Average per-class probabilities across fold models
                let mut sum_proba = Array2::zeros((n_samples, n_classes));

                for model in fold_models {
                    let hard_pred = model.predict(x)?;
                    let proba = model.predict_proba(x)?;
                    let proba = match proba {
                        Some(p) if p.nrows() == n_samples && p.ncols() == n_classes => p,
                        _ => Self::one_hot(&hard_pred, classes, n_samples, n_classes),
                    };
                    sum_proba = sum_proba + proba;
                }

                let avg_proba = sum_proba / fold_models.len() as f64;

                for i in 0..n_samples {
                    for c in 0..n_classes {
                        meta_features[[i, base_idx * cols_per_model + c]] = avg_proba[[i, c]];
                    }
                }
            } else {
                // Average hard-label predictions across fold models
                let mut sum_preds = Array1::zeros(n_samples);

                for model in fold_models {
                    let preds = model.predict(x)?;
                    sum_preds = sum_preds + preds;
                }

                let avg_preds = sum_preds / fold_models.len() as f64;

                for i in 0..n_samples {
                    meta_features[[i, base_idx]] = avg_preds[i];
                }
            }
        }

        // Add passthrough features
        if self.config.passthrough {
            let offset = n_base_models * cols_per_model;
            for i in 0..n_samples {
                for j in 0..x.ncols() {
                    meta_features[[i, offset + j]] = x[[i, j]];
                }
            }
        }

        meta_learner.predict(&meta_features)
    }
}

/// Stacking regressor
pub struct StackingRegressor<F>
where
    F: Fn(&Array2<f64>, &Array1<f64>) -> Result<Box<dyn Model>>,
{
    /// Configuration
    config: StackingConfig,
    /// Base model factory functions
    base_model_factories: Vec<F>,
    /// Fitted base models
    fitted_base_models: Option<Vec<Vec<Box<dyn Model>>>>,
    /// Meta-learner factory
    meta_learner_factory: Option<F>,
    /// Fitted meta-learner
    fitted_meta_learner: Option<Box<dyn Model>>,
}

impl<F> StackingRegressor<F>
where
    F: Fn(&Array2<f64>, &Array1<f64>) -> Result<Box<dyn Model>>,
{
    /// Create a new stacking regressor
    pub fn new(config: StackingConfig) -> Self {
        Self {
            config,
            base_model_factories: Vec::new(),
            fitted_base_models: None,
            meta_learner_factory: None,
            fitted_meta_learner: None,
        }
    }

    /// Add a base model
    pub fn add_base_model(mut self, factory: F) -> Self {
        self.base_model_factories.push(factory);
        self
    }

    /// Set the meta-learner
    pub fn with_meta_learner(mut self, factory: F) -> Self {
        self.meta_learner_factory = Some(factory);
        self
    }

    /// Fit the stacking ensemble
    pub fn fit(&mut self, x: &Array2<f64>, y: &Array1<f64>) -> Result<()> {
        if self.base_model_factories.is_empty() {
            return Err(AutoMLError::ValidationError(
                "No base models provided".to_string(),
            ));
        }

        if self.meta_learner_factory.is_none() {
            return Err(AutoMLError::ValidationError(
                "No meta-learner provided".to_string(),
            ));
        }

        let n_samples = x.nrows();
        let n_base_models = self.base_model_factories.len();
        let n_folds = self.config.n_folds;

        let cv = CrossValidator::new(CVStrategy::KFold {
            n_splits: n_folds,
            shuffle: true,
        })
        .with_random_state(self.config.seed.unwrap_or(42));

        let splits = cv.split(n_samples, None, None)?;

        let meta_features_cols = if self.config.passthrough {
            n_base_models + x.ncols()
        } else {
            n_base_models
        };
        let mut meta_features = Array2::zeros((n_samples, meta_features_cols));
        let mut fitted_models: Vec<Vec<Box<dyn Model>>> = (0..n_base_models).map(|_| Vec::new()).collect();

        for (base_idx, factory) in self.base_model_factories.iter().enumerate() {
            for split in &splits {
                let x_train = split.train_indices.iter().map(|&i| x.row(i).to_owned()).collect::<Vec<_>>();
                let y_train: Vec<f64> = split.train_indices.iter().map(|&i| y[i]).collect();
                
                let x_train = Array2::from_shape_vec(
                    (x_train.len(), x.ncols()),
                    x_train.into_iter().flat_map(|r| r.to_vec()).collect(),
                )?;
                let y_train = Array1::from_vec(y_train);

                let model = factory(&x_train, &y_train)?;

                let x_val: Vec<_> = split.test_indices.iter().map(|&i| x.row(i).to_owned()).collect();
                let x_val = Array2::from_shape_vec(
                    (x_val.len(), x.ncols()),
                    x_val.into_iter().flat_map(|r| r.to_vec()).collect(),
                )?;

                let predictions = model.predict(&x_val)?;

                for (local_idx, &global_idx) in split.test_indices.iter().enumerate() {
                    meta_features[[global_idx, base_idx]] = predictions[local_idx];
                }

                fitted_models[base_idx].push(model);
            }
        }

        if self.config.passthrough {
            for i in 0..n_samples {
                for j in 0..x.ncols() {
                    meta_features[[i, n_base_models + j]] = x[[i, j]];
                }
            }
        }

        let meta_factory = self.meta_learner_factory.as_ref().unwrap();
        let meta_learner = meta_factory(&meta_features, y)?;

        self.fitted_base_models = Some(fitted_models);
        self.fitted_meta_learner = Some(meta_learner);

        Ok(())
    }

    /// Make predictions
    pub fn predict(&self, x: &Array2<f64>) -> Result<Array1<f64>> {
        let fitted_models = self.fitted_base_models.as_ref().ok_or_else(|| {
            AutoMLError::ValidationError("Model not fitted".to_string())
        })?;

        let meta_learner = self.fitted_meta_learner.as_ref().ok_or_else(|| {
            AutoMLError::ValidationError("Model not fitted".to_string())
        })?;

        let n_samples = x.nrows();
        let n_base_models = fitted_models.len();

        let meta_features_cols = if self.config.passthrough {
            n_base_models + x.ncols()
        } else {
            n_base_models
        };
        let mut meta_features = Array2::zeros((n_samples, meta_features_cols));

        for (base_idx, fold_models) in fitted_models.iter().enumerate() {
            let mut sum_preds = Array1::zeros(n_samples);
            
            for model in fold_models {
                let preds = model.predict(x)?;
                sum_preds = sum_preds + preds;
            }

            let avg_preds = sum_preds / fold_models.len() as f64;

            for i in 0..n_samples {
                meta_features[[i, base_idx]] = avg_preds[i];
            }
        }

        if self.config.passthrough {
            for i in 0..n_samples {
                for j in 0..x.ncols() {
                    meta_features[[i, n_base_models + j]] = x[[i, j]];
                }
            }
        }

        meta_learner.predict(&meta_features)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_stacking_config_default() {
        let config = StackingConfig::default();
        assert_eq!(config.n_folds, 5);
        assert!(!config.use_probabilities);
        assert!(!config.passthrough);
    }
}
