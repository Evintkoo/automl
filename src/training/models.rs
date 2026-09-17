//! Model implementations and traits

use crate::error::Result;
use ndarray::{Array1, Array2};
use serde::{Deserialize, Serialize};

/// Metrics for model evaluation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelMetrics {
    /// Accuracy (classification)
    pub accuracy: Option<f64>,
    /// Precision (classification)
    pub precision: Option<f64>,
    /// Recall (classification)
    pub recall: Option<f64>,
    /// F1 score (classification)
    pub f1_score: Option<f64>,
    /// AUC-ROC (classification)
    pub auc_roc: Option<f64>,
    /// Log loss (classification)
    pub log_loss: Option<f64>,
    /// Mean Squared Error (regression)
    pub mse: Option<f64>,
    /// Root Mean Squared Error (regression)
    pub rmse: Option<f64>,
    /// Mean Absolute Error (regression)
    pub mae: Option<f64>,
    /// R-squared (regression)
    pub r2: Option<f64>,
    /// Training time in seconds
    pub training_time_secs: f64,
    /// Number of features
    pub n_features: usize,
    /// Number of training samples
    pub n_samples: usize,
}

impl ModelMetrics {
    /// Create new empty metrics
    pub fn new() -> Self {
        Self {
            accuracy: None,
            precision: None,
            recall: None,
            f1_score: None,
            auc_roc: None,
            log_loss: None,
            mse: None,
            rmse: None,
            mae: None,
            r2: None,
            training_time_secs: 0.0,
            n_features: 0,
            n_samples: 0,
        }
    }

    /// Compute classification metrics
    pub fn compute_classification(
        y_true: &Array1<f64>,
        y_pred: &Array1<f64>,
        _y_prob: Option<&Array1<f64>>,
    ) -> Self {
        let mut metrics = Self::new();
        metrics.n_samples = y_true.len();

        // Accuracy
        let correct: usize = y_true
            .iter()
            .zip(y_pred.iter())
            .filter(|(t, p)| (*t - *p).abs() < 0.5)
            .count();
        metrics.accuracy = Some(correct as f64 / y_true.len() as f64);

        // Precision, Recall, F1.
        //
        // Determine the distinct class labels present (rounded to the nearest integer, since
        // class values are discrete labels stored as f64). For the classic binary case
        // (labels {0, 1}), keep the original positive-class-is-1 behavior unchanged. For 3+
        // classes, use macro-averaged one-vs-rest per-class precision/recall instead of
        // thresholding both true and predicted values at >0.5 — that old approach collapsed
        // every class >=1 into a single "positive" bucket (e.g. true=1/pred=2 counted as a
        // true positive), which is wrong for multiclass problems.
        let mut classes: Vec<i64> = y_true
            .iter()
            .chain(y_pred.iter())
            .map(|v| v.round() as i64)
            .collect();
        classes.sort_unstable();
        classes.dedup();

        if classes.iter().all(|&c| c == 0 || c == 1) {
            let (tp, fp, _, fn_) = Self::confusion_counts(y_true, y_pred);

            metrics.precision = if tp + fp > 0 {
                Some(tp as f64 / (tp + fp) as f64)
            } else {
                Some(0.0)
            };

            metrics.recall = if tp + fn_ > 0 {
                Some(tp as f64 / (tp + fn_) as f64)
            } else {
                Some(0.0)
            };
        } else {
            let (precision, recall) = Self::macro_confusion_metrics(y_true, y_pred, &classes);
            metrics.precision = Some(precision);
            metrics.recall = Some(recall);
        }

        if let (Some(p), Some(r)) = (metrics.precision, metrics.recall) {
            metrics.f1_score = if p + r > 0.0 {
                Some(2.0 * p * r / (p + r))
            } else {
                Some(0.0)
            };
        }

        metrics
    }

    /// Macro-averaged multiclass precision/recall: for each class `c`, compute one-vs-rest
    /// TP/FP/FN (true==c vs pred==c), derive that class's precision/recall, then average
    /// across all classes (each class weighted equally regardless of support).
    fn macro_confusion_metrics(
        y_true: &Array1<f64>,
        y_pred: &Array1<f64>,
        classes: &[i64],
    ) -> (f64, f64) {
        if classes.is_empty() {
            return (0.0, 0.0);
        }

        let mut precisions = Vec::with_capacity(classes.len());
        let mut recalls = Vec::with_capacity(classes.len());

        for &c in classes {
            let mut tp = 0usize;
            let mut fp = 0usize;
            let mut fn_ = 0usize;

            for (t, p) in y_true.iter().zip(y_pred.iter()) {
                let t_is_c = t.round() as i64 == c;
                let p_is_c = p.round() as i64 == c;

                match (t_is_c, p_is_c) {
                    (true, true) => tp += 1,
                    (false, true) => fp += 1,
                    (true, false) => fn_ += 1,
                    (false, false) => {}
                }
            }

            precisions.push(if tp + fp > 0 {
                tp as f64 / (tp + fp) as f64
            } else {
                0.0
            });
            recalls.push(if tp + fn_ > 0 {
                tp as f64 / (tp + fn_) as f64
            } else {
                0.0
            });
        }

        let n = classes.len() as f64;
        (
            precisions.iter().sum::<f64>() / n,
            recalls.iter().sum::<f64>() / n,
        )
    }

    /// Compute regression metrics
    pub fn compute_regression(y_true: &Array1<f64>, y_pred: &Array1<f64>) -> Self {
        let mut metrics = Self::new();
        metrics.n_samples = y_true.len();

        let n = y_true.len() as f64;
        let errors: Vec<f64> = y_true
            .iter()
            .zip(y_pred.iter())
            .map(|(t, p)| t - p)
            .collect();

        // MSE
        let mse: f64 = errors.iter().map(|e| e * e).sum::<f64>() / n;
        metrics.mse = Some(mse);
        metrics.rmse = Some(mse.sqrt());

        // MAE
        let mae: f64 = errors.iter().map(|e| e.abs()).sum::<f64>() / n;
        metrics.mae = Some(mae);

        // R²
        let y_mean: f64 = y_true.iter().sum::<f64>() / n;
        let ss_tot: f64 = y_true.iter().map(|y| (y - y_mean).powi(2)).sum();
        let ss_res: f64 = errors.iter().map(|e| e.powi(2)).sum();
        
        metrics.r2 = if ss_tot > 0.0 {
            Some(1.0 - ss_res / ss_tot)
        } else {
            Some(0.0)
        };

        metrics
    }

    fn confusion_counts(y_true: &Array1<f64>, y_pred: &Array1<f64>) -> (usize, usize, usize, usize) {
        let mut tp = 0;
        let mut fp = 0;
        let mut tn = 0;
        let mut fn_ = 0;

        for (t, p) in y_true.iter().zip(y_pred.iter()) {
            let t_bool = *t > 0.5;
            let p_bool = *p > 0.5;

            match (t_bool, p_bool) {
                (true, true) => tp += 1,
                (false, true) => fp += 1,
                (false, false) => tn += 1,
                (true, false) => fn_ += 1,
            }
        }

        (tp, fp, tn, fn_)
    }
}

impl Default for ModelMetrics {
    fn default() -> Self {
        Self::new()
    }
}

/// Trait for ML models
pub trait Model: Send + Sync {
    /// Fit the model to training data
    fn fit(&mut self, x: &Array2<f64>, y: &Array1<f64>) -> Result<()>;

    /// Make predictions
    fn predict(&self, x: &Array2<f64>) -> Result<Array1<f64>>;

    /// Predict per-class probability vectors, for classifiers that support it: one row per
    /// sample, one column per class, ordered by ascending class label, each row summing to 1.
    ///
    /// Returns `Ok(None)` (the default) for models that only support hard-label prediction —
    /// callers that need probabilities (e.g. soft voting, probability-based stacking) should
    /// treat `None` as "fall back to a one-hot encoding of `predict()`'s hard label" rather
    /// than as an error, which is a reasonable, documented degradation for models that
    /// genuinely can't produce calibrated probabilities.
    fn predict_proba(&self, _x: &Array2<f64>) -> Result<Option<Array2<f64>>> {
        Ok(None)
    }

    /// Get feature importances (if available)
    fn feature_importances(&self) -> Option<Array1<f64>> {
        None
    }

    /// Save model to bytes
    fn to_bytes(&self) -> Result<Vec<u8>>;

    /// Load model from bytes
    fn from_bytes(bytes: &[u8]) -> Result<Self>
    where
        Self: Sized;
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_classification_metrics() {
        let y_true = array![1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0];
        let y_pred = array![1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0];

        let metrics = ModelMetrics::compute_classification(&y_true, &y_pred, None);
        
        assert!(metrics.accuracy.is_some());
        assert!(metrics.accuracy.unwrap() > 0.5);
    }

    #[test]
    fn test_regression_metrics() {
        let y_true = array![1.0, 2.0, 3.0, 4.0, 5.0];
        let y_pred = array![1.1, 2.0, 2.9, 4.1, 5.0];

        let metrics = ModelMetrics::compute_regression(&y_true, &y_pred);
        
        assert!(metrics.mse.is_some());
        assert!(metrics.rmse.is_some());
        assert!(metrics.r2.is_some());
        assert!(metrics.r2.unwrap() > 0.9);
    }
}
