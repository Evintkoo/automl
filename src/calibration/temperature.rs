//! Temperature scaling calibration

use crate::error::{AutoMLError, Result};
use crate::calibration::Calibrator;
use ndarray::Array1;
use serde::{Deserialize, Serialize};

/// Temperature scaling calibrator
/// 
/// Scales logits by a learned temperature parameter T:
/// calibrated_prob = softmax(logit / T)
/// 
/// For binary classification: P(y=1) = sigmoid(logit / T)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TemperatureScaling {
    /// Temperature parameter (T > 1 softens, T < 1 sharpens)
    temperature: Option<f64>,
    /// Learning rate for optimization
    learning_rate: f64,
    /// Maximum iterations
    max_iter: usize,
    /// Convergence tolerance
    tol: f64,
}

impl TemperatureScaling {
    /// Create new temperature scaling calibrator
    pub fn new() -> Self {
        Self {
            temperature: None,
            learning_rate: 0.01,
            max_iter: 100,
            tol: 1e-6,
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

    /// Get fitted temperature
    pub fn temperature(&self) -> Option<f64> {
        self.temperature
    }

    /// Sigmoid function
    fn sigmoid(x: f64) -> f64 {
        if x >= 0.0 {
            1.0 / (1.0 + (-x).exp())
        } else {
            let exp_x = x.exp();
            exp_x / (1.0 + exp_x)
        }
    }

    /// Compute negative log likelihood (cross-entropy loss)
    #[allow(dead_code)]
    fn nll(logits: &[f64], labels: &[f64], temperature: f64) -> f64 {
        let mut loss = 0.0;
        for (&logit, &label) in logits.iter().zip(labels.iter()) {
            let scaled_logit = logit / temperature;
            let prob = Self::sigmoid(scaled_logit);
            let prob_clipped = prob.clamp(1e-10, 1.0 - 1e-10);
            
            if label > 0.5 {
                loss -= prob_clipped.ln();
            } else {
                loss -= (1.0 - prob_clipped).ln();
            }
        }
        loss / logits.len() as f64
    }

    /// Compute gradient of NLL with respect to temperature
    fn nll_gradient(logits: &[f64], labels: &[f64], temperature: f64) -> f64 {
        let mut grad = 0.0;
        let t_sq = temperature * temperature;
        
        for (&logit, &label) in logits.iter().zip(labels.iter()) {
            let scaled_logit = logit / temperature;
            let prob = Self::sigmoid(scaled_logit);
            
            // d/dT of NLL
            // gradient = sum((prob - label) * (-logit / T^2))
            let error = prob - label;
            grad += error * (-logit / t_sq);
        }
        grad / logits.len() as f64
    }
}

impl Default for TemperatureScaling {
    fn default() -> Self {
        Self::new()
    }
}

impl Calibrator for TemperatureScaling {
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

        // Convert probabilities to logits
        let logits: Vec<f64> = probs
            .iter()
            .map(|&p| {
                let p_clipped = p.clamp(1e-10, 1.0 - 1e-10);
                (p_clipped / (1.0 - p_clipped)).ln()
            })
            .collect();
        
        let labels_vec: Vec<f64> = labels.iter().copied().collect();

        // Initialize temperature
        let mut t = 1.0;

        // Gradient descent with line search
        for _ in 0..self.max_iter {
            let grad = Self::nll_gradient(&logits, &labels_vec, t);
            
            // Simple gradient descent with momentum
            let step = self.learning_rate * grad;
            
            // Ensure temperature stays positive
            let new_t = (t - step).max(0.01);
            
            if (new_t - t).abs() < self.tol {
                t = new_t;
                break;
            }
            
            t = new_t;
        }

        // Clamp temperature to reasonable range
        self.temperature = Some(t.clamp(0.1, 10.0));

        Ok(())
    }

    fn calibrate(&self, probs: &Array1<f64>) -> Result<Array1<f64>> {
        let t = self.temperature.ok_or_else(|| {
            AutoMLError::ValidationError("Calibrator not fitted".to_string())
        })?;

        let calibrated: Vec<f64> = probs
            .iter()
            .map(|&p| {
                let p_clipped = p.clamp(1e-10, 1.0 - 1e-10);
                let logit = (p_clipped / (1.0 - p_clipped)).ln();
                Self::sigmoid(logit / t)
            })
            .collect();

        Ok(Array1::from_vec(calibrated))
    }
}

/// Beta calibration
///
/// Fits: P(y=1|p) = 1 / (1 + 1/exp(c) * ((1-p)/p)^a * (p/(1-p))^b)
/// Simplifies to: logit(P) = c + a*log(p) - b*log(1-p)
#[allow(dead_code)]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BetaCalibration {
    /// Parameter a
    a: Option<f64>,
    /// Parameter b
    b: Option<f64>,
    /// Parameter c
    c: Option<f64>,
    /// Maximum iterations
    max_iter: usize,
    /// Learning rate
    learning_rate: f64,
}

impl BetaCalibration {
    /// Create new beta calibration
    pub fn new() -> Self {
        Self {
            a: None,
            b: None,
            c: None,
            max_iter: 1000,
            learning_rate: 0.01,
        }
    }

    /// Get fitted parameters
    pub fn parameters(&self) -> Option<(f64, f64, f64)> {
        match (self.a, self.b, self.c) {
            (Some(a), Some(b), Some(c)) => Some((a, b, c)),
            _ => None,
        }
    }

    fn sigmoid(x: f64) -> f64 {
        1.0 / (1.0 + (-x).exp())
    }
}

impl Default for BetaCalibration {
    fn default() -> Self {
        Self::new()
    }
}

impl Calibrator for BetaCalibration {
    fn fit(&mut self, probs: &Array1<f64>, labels: &Array1<f64>) -> Result<()> {
        let n = probs.len();
        if n != labels.len() {
            return Err(AutoMLError::ValidationError(
                "Probabilities and labels must have same length".to_string(),
            ));
        }

        // Transform features: log(p), log(1-p)
        let features: Vec<(f64, f64)> = probs
            .iter()
            .map(|&p| {
                let p_clipped = p.clamp(1e-10, 1.0 - 1e-10);
                (p_clipped.ln(), (1.0 - p_clipped).ln())
            })
            .collect();

        // Initialize parameters
        let mut a = 1.0;
        let mut b = 1.0;
        let mut c = 0.0;

        // Gradient descent
        for _ in 0..self.max_iter {
            let mut grad_a = 0.0;
            let mut grad_b = 0.0;
            let mut grad_c = 0.0;

            for (i, &(log_p, log_1mp)) in features.iter().enumerate() {
                let logit = c + a * log_p - b * log_1mp;
                let pred = Self::sigmoid(logit);
                let error = pred - labels[i];

                grad_a += error * log_p;
                grad_b += error * (-log_1mp);
                grad_c += error;
            }

            grad_a /= n as f64;
            grad_b /= n as f64;
            grad_c /= n as f64;

            a -= self.learning_rate * grad_a;
            b -= self.learning_rate * grad_b;
            c -= self.learning_rate * grad_c;

            // Ensure a, b are non-negative
            a = a.max(0.0);
            b = b.max(0.0);
        }

        self.a = Some(a);
        self.b = Some(b);
        self.c = Some(c);

        Ok(())
    }

    fn calibrate(&self, probs: &Array1<f64>) -> Result<Array1<f64>> {
        let (a, b, c) = self.parameters().ok_or_else(|| {
            AutoMLError::ValidationError("Calibrator not fitted".to_string())
        })?;

        let calibrated: Vec<f64> = probs
            .iter()
            .map(|&p| {
                let p_clipped = p.clamp(1e-10, 1.0 - 1e-10);
                let log_p = p_clipped.ln();
                let log_1mp = (1.0 - p_clipped).ln();
                let logit = c + a * log_p - b * log_1mp;
                Self::sigmoid(logit)
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
    fn test_temperature_scaling_basic() {
        let probs = array![0.1, 0.3, 0.5, 0.7, 0.9];
        let labels = array![0.0, 0.0, 1.0, 1.0, 1.0];

        let mut calibrator = TemperatureScaling::new();
        calibrator.fit(&probs, &labels).unwrap();

        let t = calibrator.temperature().unwrap();
        assert!(t > 0.0);

        let calibrated = calibrator.calibrate(&probs).unwrap();
        assert_eq!(calibrated.len(), probs.len());
    }

    #[test]
    fn test_beta_calibration() {
        let probs = array![0.1, 0.2, 0.6, 0.8, 0.9];
        let labels = array![0.0, 0.0, 1.0, 1.0, 1.0];

        let mut calibrator = BetaCalibration::new();
        calibrator.fit(&probs, &labels).unwrap();

        assert!(calibrator.parameters().is_some());

        let calibrated = calibrator.calibrate(&probs).unwrap();
        assert!(calibrated.iter().all(|&p| p >= 0.0 && p <= 1.0));
    }
}
