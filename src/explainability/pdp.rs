//! Partial Dependence Plots and Individual Conditional Expectation

use crate::error::{AutoMLError, Result};
use ndarray::{Array1, Array2};
use serde::{Deserialize, Serialize};

/// Result of Partial Dependence computation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PDPResult {
    /// Feature index
    pub feature_index: usize,
    /// Feature name (if provided)
    pub feature_name: Option<String>,
    /// Grid values for the feature
    pub grid_values: Vec<f64>,
    /// Average predictions at each grid point
    pub average_predictions: Vec<f64>,
    /// Standard deviation of predictions at each grid point
    pub std_predictions: Vec<f64>,
    /// Interaction strength (variance of ICE lines)
    pub interaction_strength: f64,
}

/// Individual Conditional Expectation result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ICEResult {
    /// Feature index
    pub feature_index: usize,
    /// Feature name (if provided)
    pub feature_name: Option<String>,
    /// Grid values for the feature
    pub grid_values: Vec<f64>,
    /// Individual predictions: shape (n_samples, n_grid_points)
    pub individual_predictions: Vec<Vec<f64>>,
    /// Centered ICE (c-ICE): predictions centered at first grid value
    pub centered_predictions: Option<Vec<Vec<f64>>>,
}

impl ICEResult {
    /// Compute the average (PDP) from ICE curves
    pub fn to_pdp(&self) -> PDPResult {
        let n_samples = self.individual_predictions.len();
        let n_grid = self.grid_values.len();

        let mut average = vec![0.0; n_grid];
        let mut std = vec![0.0; n_grid];

        for (grid_idx, avg) in average.iter_mut().enumerate() {
            let values: Vec<f64> = self
                .individual_predictions
                .iter()
                .map(|pred| pred[grid_idx])
                .collect();

            let mean = values.iter().sum::<f64>() / n_samples as f64;
            *avg = mean;

            let variance =
                values.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / n_samples as f64;
            std[grid_idx] = variance.sqrt();
        }

        // Interaction strength: average variance across grid points
        let interaction_strength = std.iter().sum::<f64>() / std.len() as f64;

        PDPResult {
            feature_index: self.feature_index,
            feature_name: self.feature_name.clone(),
            grid_values: self.grid_values.clone(),
            average_predictions: average,
            std_predictions: std,
            interaction_strength,
        }
    }
}

/// Partial Dependence calculator
pub struct PartialDependence<F>
where
    F: Fn(&Array2<f64>) -> Result<Array1<f64>>,
{
    /// Prediction function
    predict_fn: F,
    /// Number of grid points
    n_grid_points: usize,
    /// Percentile range for grid
    percentile_range: (f64, f64),
    /// Feature names
    feature_names: Option<Vec<String>>,
}

impl<F> PartialDependence<F>
where
    F: Fn(&Array2<f64>) -> Result<Array1<f64>>,
{
    /// Create new PDP calculator
    pub fn new(predict_fn: F) -> Self {
        Self {
            predict_fn,
            n_grid_points: 50,
            percentile_range: (5.0, 95.0),
            feature_names: None,
        }
    }

    /// Set number of grid points
    pub fn with_grid_points(mut self, n: usize) -> Self {
        self.n_grid_points = n.max(2);
        self
    }

    /// Set percentile range for grid
    pub fn with_percentile_range(mut self, low: f64, high: f64) -> Self {
        self.percentile_range = (low.clamp(0.0, 100.0), high.clamp(0.0, 100.0));
        self
    }

    /// Set feature names
    pub fn with_feature_names(mut self, names: Vec<String>) -> Self {
        self.feature_names = Some(names);
        self
    }

    /// Compute PDP for a single feature
    pub fn compute(&self, x: &Array2<f64>, feature_index: usize) -> Result<PDPResult> {
        let ice = self.compute_ice(x, feature_index)?;
        Ok(ice.to_pdp())
    }

    /// Compute ICE for a single feature
    pub fn compute_ice(&self, x: &Array2<f64>, feature_index: usize) -> Result<ICEResult> {
        if feature_index >= x.ncols() {
            return Err(AutoMLError::ValidationError(format!(
                "Feature index {} out of bounds (n_features={})",
                feature_index,
                x.ncols()
            )));
        }

        // Create grid values
        let grid_values = self.create_grid(x, feature_index);
        let n_samples = x.nrows();
        let n_grid = grid_values.len();

        // Compute individual predictions
        let mut individual_predictions = vec![vec![0.0; n_grid]; n_samples];

        // Allocate the modified matrix once and reuse it across grid points:
        // every iteration fully overwrites `feature_index` for every row
        // before predicting, so there's nothing to restore between
        // iterations and no need for a fresh full-matrix clone each time
        // (this was previously O(n_grid) clones of the whole dataset).
        let mut x_modified = x.clone();
        for (grid_idx, &grid_val) in grid_values.iter().enumerate() {
            for i in 0..n_samples {
                x_modified[[i, feature_index]] = grid_val;
            }

            // Get predictions
            let predictions = (self.predict_fn)(&x_modified)?;

            // Store individual predictions
            for (sample_idx, &pred) in predictions.iter().enumerate() {
                individual_predictions[sample_idx][grid_idx] = pred;
            }
        }

        // Compute centered ICE (c-ICE)
        let centered_predictions: Vec<Vec<f64>> = individual_predictions
            .iter()
            .map(|preds| {
                let center = preds[0];
                preds.iter().map(|p| p - center).collect()
            })
            .collect();

        let feature_name = self
            .feature_names
            .as_ref()
            .and_then(|names| names.get(feature_index).cloned());

        Ok(ICEResult {
            feature_index,
            feature_name,
            grid_values,
            individual_predictions,
            centered_predictions: Some(centered_predictions),
        })
    }

    /// Compute PDP for multiple features
    pub fn compute_batch(
        &self,
        x: &Array2<f64>,
        feature_indices: &[usize],
    ) -> Result<Vec<PDPResult>> {
        feature_indices
            .iter()
            .map(|&idx| self.compute(x, idx))
            .collect()
    }

    /// Compute 2D PDP for feature interactions
    pub fn compute_2d(
        &self,
        x: &Array2<f64>,
        feature_1: usize,
        feature_2: usize,
    ) -> Result<PDP2DResult> {
        if feature_1 >= x.ncols() || feature_2 >= x.ncols() {
            return Err(AutoMLError::ValidationError(
                "Feature index out of bounds".to_string(),
            ));
        }

        let grid_1 = self.create_grid(x, feature_1);
        let grid_2 = self.create_grid(x, feature_2);
        let n_samples = x.nrows();
        let n_grid_1 = grid_1.len();
        let n_grid_2 = grid_2.len();

        let mut predictions = vec![vec![0.0; n_grid_2]; n_grid_1];

        // As in compute_ice: reuse one buffer across the whole grid instead
        // of cloning the full matrix for every (val_1, val_2) pair (was
        // n_grid_1 * n_grid_2 full-matrix clones — 2500 by default).
        let mut x_modified = x.clone();
        for (i, &val_1) in grid_1.iter().enumerate() {
            for (j, &val_2) in grid_2.iter().enumerate() {
                // Modify both features
                for k in 0..n_samples {
                    x_modified[[k, feature_1]] = val_1;
                    x_modified[[k, feature_2]] = val_2;
                }

                // Average prediction
                let preds = (self.predict_fn)(&x_modified)?;
                predictions[i][j] = preds.mean().unwrap_or(0.0);
            }
        }

        let name_1 = self
            .feature_names
            .as_ref()
            .and_then(|n| n.get(feature_1).cloned());
        let name_2 = self
            .feature_names
            .as_ref()
            .and_then(|n| n.get(feature_2).cloned());

        Ok(PDP2DResult {
            feature_indices: (feature_1, feature_2),
            feature_names: (name_1, name_2),
            grid_values_1: grid_1,
            grid_values_2: grid_2,
            predictions,
        })
    }

    // Create grid values for a feature
    fn create_grid(&self, x: &Array2<f64>, feature_index: usize) -> Vec<f64> {
        let mut values: Vec<f64> = x.column(feature_index).iter().copied().collect();
        values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let n = values.len();
        let (low_pct, high_pct) = self.percentile_range;

        let low_idx = ((low_pct / 100.0) * (n - 1) as f64) as usize;
        let high_idx = ((high_pct / 100.0) * (n - 1) as f64) as usize;

        let low_val = values[low_idx];
        let high_val = values[high_idx];

        // Create evenly spaced grid
        let step = (high_val - low_val) / (self.n_grid_points - 1) as f64;
        (0..self.n_grid_points)
            .map(|i| low_val + i as f64 * step)
            .collect()
    }
}

/// Result of 2D PDP computation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PDP2DResult {
    /// Feature indices (feature_1, feature_2)
    pub feature_indices: (usize, usize),
    /// Feature names (if provided)
    pub feature_names: (Option<String>, Option<String>),
    /// Grid values for feature 1
    pub grid_values_1: Vec<f64>,
    /// Grid values for feature 2
    pub grid_values_2: Vec<f64>,
    /// Predictions grid: shape (n_grid_1, n_grid_2)
    pub predictions: Vec<Vec<f64>>,
}

impl PDP2DResult {
    /// Compute interaction strength between features
    pub fn interaction_strength(&self, pdp_1: &PDPResult, pdp_2: &PDPResult) -> f64 {
        // H-statistic: variance explained by interaction
        let n1 = self.grid_values_1.len();
        let n2 = self.grid_values_2.len();

        let total_mean = self
            .predictions
            .iter()
            .flat_map(|row| row.iter())
            .sum::<f64>()
            / (n1 * n2) as f64;

        // Compute residuals from additive model
        let mut ss_residual = 0.0;
        let mut ss_total = 0.0;

        for (i, row) in self.predictions.iter().enumerate() {
            for (j, &pred) in row.iter().enumerate() {
                let additive = pdp_1.average_predictions[i] + pdp_2.average_predictions[j]
                    - total_mean;
                let residual = pred - additive;
                ss_residual += residual.powi(2);
                ss_total += (pred - total_mean).powi(2);
            }
        }

        if ss_total > 0.0 {
            ss_residual / ss_total
        } else {
            0.0
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_pdp_basic() {
        // Simple linear model: y = x0 + 2*x1
        let predict_fn = |x: &Array2<f64>| -> Result<Array1<f64>> {
            let preds: Vec<f64> = x.rows().into_iter().map(|row| row[0] + 2.0 * row[1]).collect();
            Ok(Array1::from_vec(preds))
        };

        let x = Array2::from_shape_vec(
            (10, 2),
            vec![
                0.0, 0.0, 1.0, 0.5, 2.0, 1.0, 3.0, 1.5, 4.0, 2.0, 5.0, 2.5, 6.0, 3.0, 7.0, 3.5,
                8.0, 4.0, 9.0, 4.5,
            ],
        )
        .unwrap();

        let pdp = PartialDependence::new(predict_fn).with_grid_points(10);

        let result = pdp.compute(&x, 0).unwrap();

        // PDP should show linear relationship
        assert_eq!(result.feature_index, 0);
        assert!(!result.grid_values.is_empty());
        assert!(!result.average_predictions.is_empty());
    }

    #[test]
    fn test_ice_curves() {
        let predict_fn = |x: &Array2<f64>| -> Result<Array1<f64>> {
            Ok(x.column(0).to_owned())
        };

        let x = Array2::from_shape_vec((5, 2), vec![1.0, 0.0, 2.0, 0.0, 3.0, 0.0, 4.0, 0.0, 5.0, 0.0])
            .unwrap();

        let pdp = PartialDependence::new(predict_fn).with_grid_points(5);
        let ice = pdp.compute_ice(&x, 0).unwrap();

        assert_eq!(ice.individual_predictions.len(), 5);
        assert!(ice.centered_predictions.is_some());
    }

    #[test]
    fn test_pdp_2d() {
        let predict_fn = |x: &Array2<f64>| -> Result<Array1<f64>> {
            let preds: Vec<f64> = x
                .rows()
                .into_iter()
                .map(|row| row[0] * row[1])
                .collect();
            Ok(Array1::from_vec(preds))
        };

        let x = Array2::from_shape_vec(
            (6, 2),
            vec![1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 4.0, 4.0, 5.0, 5.0, 6.0, 6.0],
        )
        .unwrap();

        let pdp = PartialDependence::new(predict_fn).with_grid_points(5);
        let result = pdp.compute_2d(&x, 0, 1).unwrap();

        assert_eq!(result.predictions.len(), 5);
        assert_eq!(result.predictions[0].len(), 5);
    }
}
