//! Blending ensemble method

use crate::error::{AutoMLError, Result};
use crate::training::Model;
use ndarray::{Array1, Array2};
use rand::prelude::*;
use serde::{Deserialize, Serialize};

/// Configuration for blending ensemble
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BlendingConfig {
    /// Proportion of data to use for blending (holdout set)
    pub holdout_ratio: f64,
    /// Whether to include original features in meta-learner input
    pub passthrough: bool,
    /// Random seed
    pub seed: Option<u64>,
}

impl Default for BlendingConfig {
    fn default() -> Self {
        Self {
            holdout_ratio: 0.2,
            passthrough: false,
            seed: None,
        }
    }
}

/// Blending classifier
pub struct BlendingClassifier<F>
where
    F: Fn(&Array2<f64>, &Array1<f64>) -> Result<Box<dyn Model>>,
{
    /// Configuration
    config: BlendingConfig,
    /// Base model factory functions
    base_model_factories: Vec<F>,
    /// Fitted base models
    fitted_base_models: Option<Vec<Box<dyn Model>>>,
    /// Meta-learner factory
    meta_learner_factory: Option<F>,
    /// Fitted meta-learner
    fitted_meta_learner: Option<Box<dyn Model>>,
}

impl<F> BlendingClassifier<F>
where
    F: Fn(&Array2<f64>, &Array1<f64>) -> Result<Box<dyn Model>>,
{
    /// Create a new blending classifier
    pub fn new(config: BlendingConfig) -> Self {
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

    /// Fit the blending ensemble
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
        let n_holdout = (n_samples as f64 * self.config.holdout_ratio) as usize;
        let n_train = n_samples - n_holdout;

        if n_holdout < 1 || n_train < 1 {
            return Err(AutoMLError::ValidationError(
                "Not enough samples for blending".to_string(),
            ));
        }

        // Create shuffled indices
        let mut rng = match self.config.seed {
            Some(seed) => StdRng::seed_from_u64(seed),
            None => StdRng::from_entropy(),
        };
        let mut indices: Vec<usize> = (0..n_samples).collect();
        indices.shuffle(&mut rng);

        let train_indices = &indices[..n_train];
        let holdout_indices = &indices[n_train..];

        // Split data
        let x_train = Self::select_rows(x, train_indices);
        let y_train = Self::select_indices(y, train_indices);
        let x_holdout = Self::select_rows(x, holdout_indices);
        let y_holdout = Self::select_indices(y, holdout_indices);

        // Train base models on training set
        let mut fitted_models = Vec::new();
        let n_base_models = self.base_model_factories.len();

        for factory in &self.base_model_factories {
            let model = factory(&x_train, &y_train)?;
            fitted_models.push(model);
        }

        // Get predictions on holdout set
        let meta_features_cols = if self.config.passthrough {
            n_base_models + x.ncols()
        } else {
            n_base_models
        };
        let mut meta_features = Array2::zeros((n_holdout, meta_features_cols));

        for (base_idx, model) in fitted_models.iter().enumerate() {
            let predictions = model.predict(&x_holdout)?;
            for i in 0..n_holdout {
                meta_features[[i, base_idx]] = predictions[i];
            }
        }

        // Add passthrough features
        if self.config.passthrough {
            for i in 0..n_holdout {
                for j in 0..x.ncols() {
                    meta_features[[i, n_base_models + j]] = x_holdout[[i, j]];
                }
            }
        }

        // Train meta-learner on holdout predictions
        let meta_factory = self.meta_learner_factory.as_ref().unwrap();
        let meta_learner = meta_factory(&meta_features, &y_holdout)?;

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

        for (base_idx, model) in fitted_models.iter().enumerate() {
            let predictions = model.predict(x)?;
            for i in 0..n_samples {
                meta_features[[i, base_idx]] = predictions[i];
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

    fn select_rows(x: &Array2<f64>, indices: &[usize]) -> Array2<f64> {
        let n_cols = x.ncols();
        let data: Vec<f64> = indices
            .iter()
            .flat_map(|&i| x.row(i).iter().copied().collect::<Vec<_>>())
            .collect();
        Array2::from_shape_vec((indices.len(), n_cols), data).unwrap()
    }

    fn select_indices(y: &Array1<f64>, indices: &[usize]) -> Array1<f64> {
        Array1::from_vec(indices.iter().map(|&i| y[i]).collect())
    }
}

/// Blending regressor
pub struct BlendingRegressor<F>
where
    F: Fn(&Array2<f64>, &Array1<f64>) -> Result<Box<dyn Model>>,
{
    /// Configuration
    config: BlendingConfig,
    /// Base model factory functions
    base_model_factories: Vec<F>,
    /// Fitted base models
    fitted_base_models: Option<Vec<Box<dyn Model>>>,
    /// Meta-learner factory
    meta_learner_factory: Option<F>,
    /// Fitted meta-learner
    fitted_meta_learner: Option<Box<dyn Model>>,
}

impl<F> BlendingRegressor<F>
where
    F: Fn(&Array2<f64>, &Array1<f64>) -> Result<Box<dyn Model>>,
{
    /// Create a new blending regressor
    pub fn new(config: BlendingConfig) -> Self {
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

    /// Fit the blending ensemble
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
        let n_holdout = (n_samples as f64 * self.config.holdout_ratio) as usize;
        let n_train = n_samples - n_holdout;

        if n_holdout < 1 || n_train < 1 {
            return Err(AutoMLError::ValidationError(
                "Not enough samples for blending".to_string(),
            ));
        }

        let mut rng = match self.config.seed {
            Some(seed) => StdRng::seed_from_u64(seed),
            None => StdRng::from_entropy(),
        };
        let mut indices: Vec<usize> = (0..n_samples).collect();
        indices.shuffle(&mut rng);

        let train_indices = &indices[..n_train];
        let holdout_indices = &indices[n_train..];

        let x_train = Self::select_rows(x, train_indices);
        let y_train = Self::select_indices(y, train_indices);
        let x_holdout = Self::select_rows(x, holdout_indices);
        let y_holdout = Self::select_indices(y, holdout_indices);

        let mut fitted_models = Vec::new();
        let n_base_models = self.base_model_factories.len();

        for factory in &self.base_model_factories {
            let model = factory(&x_train, &y_train)?;
            fitted_models.push(model);
        }

        let meta_features_cols = if self.config.passthrough {
            n_base_models + x.ncols()
        } else {
            n_base_models
        };
        let mut meta_features = Array2::zeros((n_holdout, meta_features_cols));

        for (base_idx, model) in fitted_models.iter().enumerate() {
            let predictions = model.predict(&x_holdout)?;
            for i in 0..n_holdout {
                meta_features[[i, base_idx]] = predictions[i];
            }
        }

        if self.config.passthrough {
            for i in 0..n_holdout {
                for j in 0..x.ncols() {
                    meta_features[[i, n_base_models + j]] = x_holdout[[i, j]];
                }
            }
        }

        let meta_factory = self.meta_learner_factory.as_ref().unwrap();
        let meta_learner = meta_factory(&meta_features, &y_holdout)?;

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

        for (base_idx, model) in fitted_models.iter().enumerate() {
            let predictions = model.predict(x)?;
            for i in 0..n_samples {
                meta_features[[i, base_idx]] = predictions[i];
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

    fn select_rows(x: &Array2<f64>, indices: &[usize]) -> Array2<f64> {
        let n_cols = x.ncols();
        let data: Vec<f64> = indices
            .iter()
            .flat_map(|&i| x.row(i).iter().copied().collect::<Vec<_>>())
            .collect();
        Array2::from_shape_vec((indices.len(), n_cols), data).unwrap()
    }

    fn select_indices(y: &Array1<f64>, indices: &[usize]) -> Array1<f64> {
        Array1::from_vec(indices.iter().map(|&i| y[i]).collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_blending_config_default() {
        let config = BlendingConfig::default();
        assert!((config.holdout_ratio - 0.2).abs() < 1e-6);
        assert!(!config.passthrough);
    }
}
