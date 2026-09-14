//! Platt scaling (sigmoid calibration)

use crate::error::{AutoMLError, Result};
use crate::calibration::Calibrator;
use ndarray::Array1;
use serde::{Deserialize, Serialize};

/// Platt scaling calibrator
/// 
/// Fits a sigmoid function: P(y=1|f) = 1 / (1 + exp(A*f + B))
/// where f is the uncalibrated prediction
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlattScaling {
    /// Slope parameter A
    a: Option<f64>,
    /// Intercept parameter B  
    b: Option<f64>,
    /// Learning rate for optimization
    learning_rate: f64,
    /// Maximum iterations
    max_iter: usize,
    /// Convergence tolerance
    tol: f64,
}

impl PlattScaling {
    /// Create new Platt scaling calibrator
    pub fn new() -> Self {
        Self {
            a: None,
            b: None,
            learning_rate: 0.01,
            max_iter: 1000,
            tol: 1e-7,
        }
    }

    /// Set learning rate
    pub fn with_learning_rate(mut self, lr: f64) -> Self {
        self.learning_rate = lr;
        self
    }

    /// Set maximum iterations
    pub fn with_max_iter(mut self, max_iter: usize) -> Self {
        self.max_iter = max_iter;
        self
    }

    /// Get fitted parameters
    pub fn parameters(&self) -> Option<(f64, f64)> {
        match (self.a, self.b) {
            (Some(a), Some(b)) => Some((a, b)),
            _ => None,
        }
    }

    /// Sigmoid function
    fn sigmoid(x: f64) -> f64 {
        1.0 / (1.0 + (-x).exp())
    }

    /// Compute calibrated probability
    fn calibrate_single(&self, prob: f64, a: f64, b: f64) -> f64 {
        // Convert probability to logit, apply transformation, convert back
        let f = if prob <= 0.0 {
            -10.0
        } else if prob >= 1.0 {
            10.0
        } else {
            (prob / (1.0 - prob)).ln()
        };
        Self::sigmoid(a * f + b)
    }
}

impl Default for PlattScaling {
    fn default() -> Self {
        Self::new()
    }
}

impl Calibrator for PlattScaling {
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

        // Convert probabilities to logits for fitting
        let logits: Vec<f64> = probs
            .iter()
            .map(|&p| {
                let p_clipped = p.clamp(1e-10, 1.0 - 1e-10);
                (p_clipped / (1.0 - p_clipped)).ln()
            })
            .collect();

        // Initialize parameters
        let mut a = 1.0;
        let mut b = 0.0;

        // Target values with Platt's adjustment for small datasets
        let n_pos = labels.iter().filter(|&&y| y > 0.5).count() as f64;
        let n_neg = n as f64 - n_pos;
        
        let target_pos = (n_pos + 1.0) / (n_pos + 2.0);
        let target_neg = 1.0 / (n_neg + 2.0);

        let targets: Vec<f64> = labels
            .iter()
            .map(|&y| if y > 0.5 { target_pos } else { target_neg })
            .collect();

        // Newton's method / gradient descent optimization
        for _ in 0..self.max_iter {
            let mut grad_a = 0.0;
            let mut grad_b = 0.0;
            let mut hess_aa = 0.0;
            let mut hess_ab = 0.0;
            let mut hess_bb = 0.0;

            for i in 0..n {
                let f = logits[i];
                let p = Self::sigmoid(a * f + b);
                let t = targets[i];
                
                let d1 = p - t;
                let d2 = p * (1.0 - p);

                grad_a += f * d1;
                grad_b += d1;
                
                hess_aa += f * f * d2;
                hess_ab += f * d2;
                hess_bb += d2;
            }

            // Add regularization to Hessian diagonal
            hess_aa += 1e-6;
            hess_bb += 1e-6;

            // Solve 2x2 system using Cramer's rule
            let det = hess_aa * hess_bb - hess_ab * hess_ab;
            if det.abs() < 1e-10 {
                break;
            }

            let delta_a = (hess_bb * grad_a - hess_ab * grad_b) / det;
            let delta_b = (hess_aa * grad_b - hess_ab * grad_a) / det;

            // Update with line search
            let step = 1.0;
            a -= step * delta_a;
            b -= step * delta_b;

            // Check convergence
            if delta_a.abs() < self.tol && delta_b.abs() < self.tol {
                break;
            }
        }

        self.a = Some(a);
        self.b = Some(b);

        Ok(())
    }

    fn calibrate(&self, probs: &Array1<f64>) -> Result<Array1<f64>> {
        let (a, b) = self.parameters().ok_or_else(|| {
            AutoMLError::ValidationError("Calibrator not fitted".to_string())
        })?;

        let calibrated: Vec<f64> = probs
            .iter()
            .map(|&p| self.calibrate_single(p, a, b))
            .collect();

        Ok(Array1::from_vec(calibrated))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_platt_scaling_basic() {
        let probs = array![0.1, 0.4, 0.35, 0.8, 0.9, 0.2, 0.7, 0.6];
        let labels = array![0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0];

        let mut calibrator = PlattScaling::new();
        calibrator.fit(&probs, &labels).unwrap();

        let calibrated = calibrator.calibrate(&probs).unwrap();
        
        assert_eq!(calibrated.len(), probs.len());
        // All calibrated probabilities should be in [0, 1]
        assert!(calibrated.iter().all(|&p| p >= 0.0 && p <= 1.0));
    }

    #[test]
    fn test_platt_parameters() {
        let probs = array![0.2, 0.8, 0.3, 0.9];
        let labels = array![0.0, 1.0, 0.0, 1.0];

        let mut calibrator = PlattScaling::new();
        calibrator.fit(&probs, &labels).unwrap();

        let params = calibrator.parameters();
        assert!(params.is_some());
    }
}
