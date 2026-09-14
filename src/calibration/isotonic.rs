//! Isotonic regression calibration

use crate::error::{AutoMLError, Result};
use crate::calibration::Calibrator;
use ndarray::Array1;
use serde::{Deserialize, Serialize};

/// Isotonic regression calibrator
/// 
/// Non-parametric calibration that fits a monotonically increasing function
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IsotonicRegression {
    /// Fitted x values (boundaries)
    x_values: Option<Vec<f64>>,
    /// Fitted y values (calibrated probabilities)
    y_values: Option<Vec<f64>>,
    /// Whether to clip predictions to [0, 1]
    clip: bool,
    /// Minimum value (for out of range)
    y_min: f64,
    /// Maximum value (for out of range)
    y_max: f64,
}

impl IsotonicRegression {
    /// Create new isotonic regression calibrator
    pub fn new() -> Self {
        Self {
            x_values: None,
            y_values: None,
            clip: true,
            y_min: 0.0,
            y_max: 1.0,
        }
    }

    /// Set whether to clip output to [0, 1]
    pub fn with_clip(mut self, clip: bool) -> Self {
        self.clip = clip;
        self
    }

    /// Pool Adjacent Violators Algorithm (PAVA)
    fn pava(y: &[f64], weights: &[f64]) -> Vec<f64> {
        let n = y.len();
        if n == 0 {
            return vec![];
        }

        let mut result = y.to_vec();
        let mut w = weights.to_vec();
        
        // Merge pools until isotonic
        let mut i = 0;
        while i < n - 1 {
            if result[i] > result[i + 1] {
                // Merge pools i and i+1
                let total_weight = w[i] + w[i + 1];
                let merged_value = (result[i] * w[i] + result[i + 1] * w[i + 1]) / total_weight;
                
                result[i] = merged_value;
                result[i + 1] = merged_value;
                w[i] = total_weight;
                w[i + 1] = total_weight;
                
                // Check backwards for violations
                while i > 0 && result[i - 1] > result[i] {
                    let total_weight = w[i - 1] + w[i];
                    let merged_value = (result[i - 1] * w[i - 1] + result[i] * w[i]) / total_weight;
                    
                    result[i - 1] = merged_value;
                    result[i] = merged_value;
                    w[i - 1] = total_weight;
                    w[i] = total_weight;
                    i -= 1;
                }
            }
            i += 1;
        }

        result
    }

    /// Build the stepwise function from PAVA result
    fn build_function(&mut self, x_sorted: Vec<f64>, y_isotonic: Vec<f64>) {
        if x_sorted.is_empty() {
            return;
        }

        let mut x_vals = Vec::new();
        let mut y_vals = Vec::new();

        // Deduplicate consecutive equal y values
        x_vals.push(x_sorted[0]);
        y_vals.push(y_isotonic[0]);

        for i in 1..x_sorted.len() {
            if (y_isotonic[i] - y_vals.last().unwrap()).abs() > 1e-10 {
                // Add the end of the previous block
                if x_sorted[i - 1] != *x_vals.last().unwrap() {
                    x_vals.push(x_sorted[i - 1]);
                    y_vals.push(y_isotonic[i - 1]);
                }
                // Add the start of the new block
                x_vals.push(x_sorted[i]);
                y_vals.push(y_isotonic[i]);
            }
        }

        // Add final point
        if x_sorted.last() != x_vals.last() {
            x_vals.push(*x_sorted.last().unwrap());
            y_vals.push(*y_isotonic.last().unwrap());
        }

        self.y_min = *y_vals.first().unwrap();
        self.y_max = *y_vals.last().unwrap();
        self.x_values = Some(x_vals);
        self.y_values = Some(y_vals);
    }

    /// Interpolate calibrated value
    fn interpolate(&self, x: f64) -> f64 {
        let x_vals = self.x_values.as_ref().unwrap();
        let y_vals = self.y_values.as_ref().unwrap();

        if x <= x_vals[0] {
            return self.y_min;
        }
        if x >= *x_vals.last().unwrap() {
            return self.y_max;
        }

        // Binary search for interval
        let mut lo = 0;
        let mut hi = x_vals.len() - 1;
        
        while hi - lo > 1 {
            let mid = (lo + hi) / 2;
            if x_vals[mid] <= x {
                lo = mid;
            } else {
                hi = mid;
            }
        }

        // Linear interpolation
        let x0 = x_vals[lo];
        let x1 = x_vals[hi];
        let y0 = y_vals[lo];
        let y1 = y_vals[hi];

        if (x1 - x0).abs() < 1e-10 {
            return y0;
        }

        let t = (x - x0) / (x1 - x0);
        y0 + t * (y1 - y0)
    }
}

impl Default for IsotonicRegression {
    fn default() -> Self {
        Self::new()
    }
}

impl Calibrator for IsotonicRegression {
    fn fit(&mut self, probs: &Array1<f64>, labels: &Array1<f64>) -> Result<()> {
        let n = probs.len();
        if n != labels.len() {
            return Err(AutoMLError::ValidationError(
                "Probabilities and labels must have same length".to_string(),
            ));
        }

        if n == 0 {
            return Err(AutoMLError::ValidationError(
                "Empty input".to_string(),
            ));
        }

        // Sort by probability
        let mut indices: Vec<usize> = (0..n).collect();
        indices.sort_by(|&a, &b| {
            probs[a].partial_cmp(&probs[b]).unwrap_or(std::cmp::Ordering::Equal)
        });

        let x_sorted: Vec<f64> = indices.iter().map(|&i| probs[i]).collect();
        let y_sorted: Vec<f64> = indices.iter().map(|&i| labels[i]).collect();
        let weights: Vec<f64> = vec![1.0; n];

        // Apply PAVA
        let y_isotonic = Self::pava(&y_sorted, &weights);

        // Build interpolation function
        self.build_function(x_sorted, y_isotonic);

        Ok(())
    }

    fn calibrate(&self, probs: &Array1<f64>) -> Result<Array1<f64>> {
        if self.x_values.is_none() {
            return Err(AutoMLError::ValidationError(
                "Calibrator not fitted".to_string(),
            ));
        }

        let calibrated: Vec<f64> = probs
            .iter()
            .map(|&p| {
                let mut c = self.interpolate(p);
                if self.clip {
                    c = c.clamp(0.0, 1.0);
                }
                c
            })
            .collect();

        Ok(Array1::from_vec(calibrated))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_isotonic_basic() {
        let probs = array![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9];
        let labels = array![0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0];

        let mut calibrator = IsotonicRegression::new();
        calibrator.fit(&probs, &labels).unwrap();

        let calibrated = calibrator.calibrate(&probs).unwrap();
        
        // Calibrated values should be monotonically increasing
        for i in 1..calibrated.len() {
            assert!(calibrated[i] >= calibrated[i - 1] - 1e-10);
        }
    }

    #[test]
    fn test_isotonic_non_monotonic_input() {
        // Labels that aren't monotonic with respect to probabilities
        let probs = array![0.1, 0.3, 0.5, 0.7, 0.9];
        let labels = array![0.0, 1.0, 0.0, 1.0, 1.0];

        let mut calibrator = IsotonicRegression::new();
        calibrator.fit(&probs, &labels).unwrap();

        let calibrated = calibrator.calibrate(&probs).unwrap();
        
        // Output should still be monotonic
        for i in 1..calibrated.len() {
            assert!(calibrated[i] >= calibrated[i - 1] - 1e-10);
        }
    }

    #[test]
    fn test_pava() {
        let y = vec![1.0, 0.0, 1.0, 0.0, 1.0];
        let w = vec![1.0, 1.0, 1.0, 1.0, 1.0];
        
        let result = IsotonicRegression::pava(&y, &w);
        
        // Result should be non-decreasing
        for i in 1..result.len() {
            assert!(result[i] >= result[i - 1] - 1e-10);
        }
    }
}
