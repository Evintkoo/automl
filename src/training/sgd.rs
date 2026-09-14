//! Stochastic Gradient Descent (SGD) classifier and regressor
//!
//! Supports multiple loss functions and learning rate schedules.
//! Efficient for large-scale learning — processes one sample at a time.

use crate::error::{AutoMLError, Result};
use ndarray::{Array1, Array2};
use rand::prelude::*;
use rand_xoshiro::Xoshiro256PlusPlus;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SGDLoss {
    Hinge,          // SVM-like
    Log,            // Logistic regression
    ModifiedHuber,  // Smooth hinge
    SquaredError,   // Regression
    Huber,          // Robust regression
    Epsilon,        // Epsilon-insensitive (SVR-like)
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum LearningRateSchedule {
    Constant,
    Optimal,     // 1 / (alpha * (t + t0))
    InvScaling,  // eta0 / t^power_t
    Adaptive,    // Halve when loss stops improving
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SGDConfig {
    pub loss: SGDLoss,
    pub learning_rate: LearningRateSchedule,
    pub eta0: f64,
    pub alpha: f64,         // L2 regularization
    pub l1_ratio: f64,      // ElasticNet mixing (0 = L2, 1 = L1)
    pub max_iter: usize,
    pub tol: f64,
    pub power_t: f64,       // For InvScaling schedule
    pub epsilon: f64,       // For Huber and Epsilon losses
    pub random_state: Option<u64>,
}

impl Default for SGDConfig {
    fn default() -> Self {
        Self {
            loss: SGDLoss::SquaredError,
            learning_rate: LearningRateSchedule::InvScaling,
            eta0: 0.01,
            alpha: 0.0001,
            l1_ratio: 0.0,
            max_iter: 1000,
            tol: 1e-4,
            power_t: 0.25,
            epsilon: 0.1,
            random_state: Some(42),
        }
    }
}

fn get_lr(config: &SGDConfig, t: usize) -> f64 {
    match config.learning_rate {
        LearningRateSchedule::Constant => config.eta0,
        LearningRateSchedule::Optimal => {
            let t0 = 1.0 / (config.alpha * config.eta0);
            1.0 / (config.alpha * (t as f64 + t0))
        }
        LearningRateSchedule::InvScaling => {
            config.eta0 / (t as f64 + 1.0).powf(config.power_t)
        }
        LearningRateSchedule::Adaptive => config.eta0, // adjusted externally
    }
}

fn soft_threshold(val: f64, threshold: f64) -> f64 {
    if val > threshold { val - threshold }
    else if val < -threshold { val + threshold }
    else { 0.0 }
}

// ============ SGD Regressor ============

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SGDRegressor {
    pub config: SGDConfig,
    pub weights: Option<Array1<f64>>,
    pub bias: f64,
    #[serde(default)]
    pub epoch_records: Vec<crate::training::engine::EpochRecord>,
}

impl SGDRegressor {
    pub fn new(config: SGDConfig) -> Self {
        Self { config, weights: None, bias: 0.0, epoch_records: Vec::new() }
    }

    pub fn fit(&mut self, x: &Array2<f64>, y: &Array1<f64>) -> Result<()> {
        self.epoch_records.clear();
        let n = x.nrows();
        let p = x.ncols();
        if n == 0 { return Err(AutoMLError::TrainingError("Empty dataset".into())); }

        let mut rng = Xoshiro256PlusPlus::seed_from_u64(self.config.random_state.unwrap_or(42));
        let mut w = Array1::zeros(p);
        let mut b = 0.0;
        let mut indices: Vec<usize> = (0..n).collect();
        let mut prev_loss = f64::MAX;
        let mut current_eta = self.config.eta0;
        let mut t = 1usize;

        for epoch in 0..self.config.max_iter {
            indices.shuffle(&mut rng);
            let mut epoch_loss = 0.0;

            for &i in &indices {
                let xi = x.row(i);
                let yi = y[i];
                let pred = xi.dot(&w) + b;

                let lr = match self.config.learning_rate {
                    LearningRateSchedule::Adaptive => current_eta,
                    _ => get_lr(&self.config, t),
                };

                // Compute gradient based on loss
                let dloss = match self.config.loss {
                    SGDLoss::SquaredError => pred - yi,
                    SGDLoss::Huber => {
                        let diff = pred - yi;
                        if diff.abs() <= self.config.epsilon { diff }
                        else { self.config.epsilon * diff.signum() }
                    }
                    SGDLoss::Epsilon => {
                        let diff = pred - yi;
                        if diff.abs() <= self.config.epsilon { 0.0 }
                        else { diff.signum() }
                    }
                    _ => pred - yi, // fallback to squared error
                };

                epoch_loss += dloss * dloss;

                // Update with ElasticNet regularization
                let l2_coeff = self.config.alpha * (1.0 - self.config.l1_ratio);
                let l1_coeff = self.config.alpha * self.config.l1_ratio;

                for j in 0..p {
                    let grad = dloss * xi[j] + l2_coeff * w[j];
                    w[j] -= lr * grad;
                    w[j] = soft_threshold(w[j], lr * l1_coeff);
                }
                b -= lr * dloss;
                t += 1;
            }

            epoch_loss /= n as f64;

            self.epoch_records.push(crate::training::engine::EpochRecord {
                epoch,
                train_loss: epoch_loss,
                val_loss: None,
                train_acc: None,
                val_acc: None,
            });

            // Adaptive schedule: halve lr if loss didn't improve
            if matches!(self.config.learning_rate, LearningRateSchedule::Adaptive) {
                if epoch_loss > prev_loss - self.config.tol {
                    current_eta *= 0.5;
                    if current_eta < 1e-10 { break; }
                }
            }

            if (prev_loss - epoch_loss).abs() < self.config.tol && epoch > 0 { break; }
            prev_loss = epoch_loss;
        }

        self.weights = Some(w);
        self.bias = b;
        Ok(())
    }

    pub fn predict(&self, x: &Array2<f64>) -> Result<Array1<f64>> {
        let w = self.weights.as_ref().ok_or_else(|| AutoMLError::TrainingError("Model not fitted".into()))?;
        Ok(Array1::from_vec(x.rows().into_iter().map(|row| row.dot(w) + self.bias).collect()))
    }
}

// ============ SGD Classifier ============

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SGDClassifier {
    pub config: SGDConfig,
    pub weights: Option<Array1<f64>>,
    pub bias: f64,
    #[serde(default)]
    pub epoch_records: Vec<crate::training::engine::EpochRecord>,
}

impl SGDClassifier {
    pub fn new(config: SGDConfig) -> Self {
        let mut config = config;
        config.loss = SGDLoss::Log; // default to logistic for classification
        Self { config, weights: None, bias: 0.0, epoch_records: Vec::new() }
    }

    pub fn fit(&mut self, x: &Array2<f64>, y: &Array1<f64>) -> Result<()> {
        self.epoch_records.clear();
        let n = x.nrows();
        let p = x.ncols();
        if n == 0 { return Err(AutoMLError::TrainingError("Empty dataset".into())); }

        // Convert labels: 0/1 → -1/+1 for hinge losses
        let y_signed: Vec<f64> = y.iter().map(|&v| if v > 0.5 { 1.0 } else { -1.0 }).collect();

        let mut rng = Xoshiro256PlusPlus::seed_from_u64(self.config.random_state.unwrap_or(42));
        let mut w = Array1::zeros(p);
        let mut b = 0.0;
        let mut indices: Vec<usize> = (0..n).collect();
        let mut prev_loss = f64::MAX;
        let mut current_eta = self.config.eta0;
        let mut t = 1usize;

        for epoch in 0..self.config.max_iter {
            indices.shuffle(&mut rng);
            let mut epoch_loss = 0.0;

            for &i in &indices {
                let xi = x.row(i);
                let margin = xi.dot(&w) + b;
                let yi = y_signed[i];

                let lr = match self.config.learning_rate {
                    LearningRateSchedule::Adaptive => current_eta,
                    _ => get_lr(&self.config, t),
                };

                let (dloss_w, dloss_b) = match self.config.loss {
                    SGDLoss::Hinge => {
                        if yi * margin < 1.0 {
                            epoch_loss += 1.0 - yi * margin;
                            (-yi, -yi)
                        } else { (0.0, 0.0) }
                    }
                    SGDLoss::Log => {
                        let p = sigmoid(margin);
                        let y01 = if yi > 0.0 { 1.0 } else { 0.0 };
                        let diff = p - y01;
                        epoch_loss += -(y01 * p.max(1e-15).ln() + (1.0 - y01) * (1.0 - p).max(1e-15).ln());
                        (diff, diff)
                    }
                    SGDLoss::ModifiedHuber => {
                        let z = yi * margin;
                        if z >= 1.0 {
                            (0.0, 0.0)
                        } else if z >= -1.0 {
                            epoch_loss += (1.0 - z) * (1.0 - z);
                            (-2.0 * (1.0 - z) * yi, -2.0 * (1.0 - z) * yi)
                        } else {
                            epoch_loss += -4.0 * z;
                            (-4.0 * yi, -4.0 * yi)
                        }
                    }
                    _ => {
                        let p = sigmoid(margin);
                        let y01 = if yi > 0.0 { 1.0 } else { 0.0 };
                        let diff = p - y01;
                        (diff, diff)
                    }
                };

                let l2_coeff = self.config.alpha * (1.0 - self.config.l1_ratio);
                let l1_coeff = self.config.alpha * self.config.l1_ratio;

                for j in 0..p {
                    let grad = dloss_w * xi[j] + l2_coeff * w[j];
                    w[j] -= lr * grad;
                    w[j] = soft_threshold(w[j], lr * l1_coeff);
                }
                b -= lr * dloss_b;
                t += 1;
            }

            epoch_loss /= n as f64;

            self.epoch_records.push(crate::training::engine::EpochRecord {
                epoch,
                train_loss: epoch_loss,
                val_loss: None,
                train_acc: None,
                val_acc: None,
            });

            if matches!(self.config.learning_rate, LearningRateSchedule::Adaptive) {
                if epoch_loss > prev_loss - self.config.tol {
                    current_eta *= 0.5;
                    if current_eta < 1e-10 { break; }
                }
            }

            if (prev_loss - epoch_loss).abs() < self.config.tol && epoch > 0 { break; }
            prev_loss = epoch_loss;
        }

        self.weights = Some(w);
        self.bias = b;
        Ok(())
    }

    pub fn predict(&self, x: &Array2<f64>) -> Result<Array1<f64>> {
        let w = self.weights.as_ref().ok_or_else(|| AutoMLError::TrainingError("Model not fitted".into()))?;
        Ok(Array1::from_vec(x.rows().into_iter().map(|row| {
            let margin = row.dot(w) + self.bias;
            match self.config.loss {
                SGDLoss::Hinge | SGDLoss::ModifiedHuber => {
                    if margin >= 0.0 { 1.0 } else { 0.0 }
                }
                _ => if sigmoid(margin) >= 0.5 { 1.0 } else { 0.0 },
            }
        }).collect()))
    }

    pub fn predict_proba(&self, x: &Array2<f64>) -> Result<Array2<f64>> {
        let w = self.weights.as_ref().ok_or_else(|| AutoMLError::TrainingError("Model not fitted".into()))?;
        let n = x.nrows();
        let mut proba = Array2::zeros((n, 2));
        for (i, row) in x.rows().into_iter().enumerate() {
            let margin = row.dot(w) + self.bias;
            let p = match self.config.loss {
                SGDLoss::ModifiedHuber => {
                    let z = margin;
                    if z >= 1.0 { 1.0 }
                    else if z <= -1.0 { 0.0 }
                    else { (z + 1.0) / 2.0 }
                }
                _ => sigmoid(margin),
            };
            proba[[i, 0]] = 1.0 - p;
            proba[[i, 1]] = p;
        }
        Ok(proba)
    }
}

fn sigmoid(x: f64) -> f64 { 1.0 / (1.0 + (-x).exp()) }

#[cfg(test)]
mod tests {
    use super::*;

    fn make_regression_data() -> (Array2<f64>, Array1<f64>) {
        let x = Array2::from_shape_vec((100, 3), (0..300).map(|i| (i as f64) / 100.0).collect()).unwrap();
        let y = Array1::from_vec((0..100).map(|i| 2.0 * (i * 3) as f64 / 100.0 + 0.5).collect());
        (x, y)
    }

    fn make_classification_data() -> (Array2<f64>, Array1<f64>) {
        let x = Array2::from_shape_vec((100, 2), (0..200).map(|i| (i as f64) / 100.0).collect()).unwrap();
        let y = Array1::from_vec((0..100).map(|i| if i < 50 { 0.0 } else { 1.0 }).collect());
        (x, y)
    }

    #[test]
    fn test_sgd_regressor() {
        let (x, y) = make_regression_data();
        let config = SGDConfig { max_iter: 200, eta0: 0.001, ..Default::default() };
        let mut model = SGDRegressor::new(config);
        model.fit(&x, &y).unwrap();
        let preds = model.predict(&x).unwrap();
        assert_eq!(preds.len(), 100);
    }

    #[test]
    fn test_sgd_classifier_log() {
        let (x, y) = make_classification_data();
        let config = SGDConfig { loss: SGDLoss::Log, max_iter: 200, eta0: 0.01, ..Default::default() };
        let mut model = SGDClassifier::new(config);
        model.fit(&x, &y).unwrap();
        let preds = model.predict(&x).unwrap();
        let acc = preds.iter().zip(y.iter()).filter(|(&p, &t)| p == t).count() as f64 / 100.0;
        assert!(acc > 0.6, "Accuracy too low: {}", acc);
    }

    #[test]
    fn test_sgd_classifier_hinge() {
        let (x, y) = make_classification_data();
        let config = SGDConfig { loss: SGDLoss::Hinge, max_iter: 200, eta0: 0.01, ..Default::default() };
        let mut model = SGDClassifier::new(config);
        model.fit(&x, &y).unwrap();
        let preds = model.predict(&x).unwrap();
        assert_eq!(preds.len(), 100);
    }

    #[test]
    fn test_sgd_predict_proba() {
        let (x, y) = make_classification_data();
        let config = SGDConfig { loss: SGDLoss::Log, max_iter: 100, eta0: 0.01, ..Default::default() };
        let mut model = SGDClassifier::new(config);
        model.fit(&x, &y).unwrap();
        let proba = model.predict_proba(&x).unwrap();
        assert_eq!(proba.ncols(), 2);
        for i in 0..proba.nrows() {
            assert!((proba[[i, 0]] + proba[[i, 1]] - 1.0).abs() < 1e-10);
        }
    }
}
