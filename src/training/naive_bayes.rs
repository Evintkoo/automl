//! Naive Bayes classifiers
//!
//! Implements Gaussian Naive Bayes for continuous features.

use ndarray::{Array1, Array2};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::f64::consts::PI;

use crate::error::Result;

/// Gaussian Naive Bayes Classifier
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GaussianNaiveBayes {
    /// Mean of each feature for each class
    means: HashMap<i64, Vec<f64>>,
    /// Variance of each feature for each class
    variances: HashMap<i64, Vec<f64>>,
    /// Prior probability of each class
    priors: HashMap<i64, f64>,
    /// List of classes
    classes: Vec<i64>,
    /// Smoothing parameter for variance
    var_smoothing: f64,
}

impl Default for GaussianNaiveBayes {
    fn default() -> Self {
        Self::new()
    }
}

impl GaussianNaiveBayes {
    pub fn new() -> Self {
        Self {
            means: HashMap::new(),
            variances: HashMap::new(),
            priors: HashMap::new(),
            classes: Vec::new(),
            var_smoothing: 1e-9,
        }
    }

    /// Set variance smoothing parameter
    pub fn with_var_smoothing(mut self, smoothing: f64) -> Self {
        self.var_smoothing = smoothing;
        self
    }

    /// Fit the classifier
    pub fn fit(&mut self, x: &Array2<f64>, y: &Array1<f64>) -> Result<()> {
        let n_samples = x.nrows();
        let n_features = x.ncols();
        
        // Find unique classes
        let mut class_counts: HashMap<i64, usize> = HashMap::new();
        for &label in y.iter() {
            let class = label as i64;
            *class_counts.entry(class).or_insert(0) += 1;
        }
        
        self.classes = class_counts.keys().cloned().collect();
        self.classes.sort();
        
        // Compute priors
        for (&class, &count) in &class_counts {
            self.priors.insert(class, count as f64 / n_samples as f64);
        }
        
        // Compute mean and variance for each class
        for &class in &self.classes {
            // Filter samples for this class
            let mask: Vec<bool> = y.iter().map(|&yi| yi as i64 == class).collect();
            
            // Gather indices for this class
            let class_indices: Vec<usize> = mask.iter()
                .enumerate()
                .filter(|(_, &m)| m)
                .map(|(i, _)| i)
                .collect();
            
            let n_class = class_indices.len();
            

            // Single-pass Welford's algorithm for mean and variance
            let mut feature_means = vec![0.0; n_features];
            let mut feature_m2 = vec![0.0; n_features];
            let mut count = 0usize;
            for &idx in &class_indices {
                count += 1;
                let row = x.row(idx);
                for (j, &val) in row.iter().enumerate() {
                    let delta = val - feature_means[j];
                    feature_means[j] += delta / count as f64;
                    let delta2 = val - feature_means[j];
                    feature_m2[j] += delta * delta2;
                }
            }
            let feature_vars: Vec<f64> = feature_m2.iter()
                .map(|&m2| (m2 / n_class as f64) + self.var_smoothing)
                .collect();


            self.means.insert(class, feature_means);
            self.variances.insert(class, feature_vars);
        }
        
        Ok(())
    }

    /// Predict class labels
    pub fn predict(&self, x: &Array2<f64>) -> Array1<f64> {
        let probs = self.predict_log_proba(x);
        
        probs.rows().into_iter()
            .map(|row| {
                let max_idx = row.iter()
                    .enumerate()
                    .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                    .map(|(i, _)| i)
                    .unwrap_or(0);
                self.classes[max_idx] as f64
            })
            .collect()
    }

    /// Predict log probabilities
    pub fn predict_log_proba(&self, x: &Array2<f64>) -> Array2<f64> {
        let n_samples = x.nrows();
        let n_classes = self.classes.len();
        
        let mut log_probs = Array2::zeros((n_samples, n_classes));
        
        for (i, row) in x.rows().into_iter().enumerate() {
            for (j, &class) in self.classes.iter().enumerate() {
                let log_prior = self.priors[&class].ln();
                let log_likelihood = self.log_likelihood(&row.to_owned(), class);
                log_probs[[i, j]] = log_prior + log_likelihood;
            }
        }
        
        // Normalize (log-sum-exp trick)
        for mut row in log_probs.rows_mut() {
            let max_val = row.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let log_sum: f64 = row.iter().map(|&v| (v - max_val).exp()).sum::<f64>().ln();
            for val in row.iter_mut() {
                *val = *val - max_val - log_sum;
            }
        }
        
        log_probs
    }

    /// Predict probabilities
    pub fn predict_proba(&self, x: &Array2<f64>) -> Array2<f64> {
        let log_probs = self.predict_log_proba(x);
        log_probs.mapv(|v| v.exp())
    }

    fn log_likelihood(&self, x: &Array1<f64>, class: i64) -> f64 {
        let means = &self.means[&class];
        let vars = &self.variances[&class];
        
        x.iter()
            .zip(means.iter())
            .zip(vars.iter())
            .map(|((&xi, &mean), &var)| {
                // Log of Gaussian PDF
                -0.5 * ((xi - mean).powi(2) / var + var.ln() + (2.0 * PI).ln())
            })
            .sum()
    }

    /// Get class priors
    pub fn class_priors(&self) -> &HashMap<i64, f64> {
        &self.priors
    }

    /// Get feature means for each class
    pub fn feature_means(&self) -> &HashMap<i64, Vec<f64>> {
        &self.means
    }
}

/// Multinomial Naive Bayes (for count data)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MultinomialNaiveBayes {
    /// Log probability of each feature for each class
    feature_log_probs: HashMap<i64, Vec<f64>>,
    /// Log prior probability of each class
    class_log_priors: HashMap<i64, f64>,
    /// List of classes
    classes: Vec<i64>,
    /// Smoothing parameter (Laplace smoothing)
    alpha: f64,
}

impl MultinomialNaiveBayes {
    pub fn new(alpha: f64) -> Self {
        Self {
            feature_log_probs: HashMap::new(),
            class_log_priors: HashMap::new(),
            classes: Vec::new(),
            alpha,
        }
    }

    /// Fit the classifier
    pub fn fit(&mut self, x: &Array2<f64>, y: &Array1<f64>) -> Result<()> {
        let n_samples = x.nrows();
        let n_features = x.ncols();
        
        // Find unique classes and counts
        let mut class_counts: HashMap<i64, usize> = HashMap::new();
        for &label in y.iter() {
            let class = label as i64;
            *class_counts.entry(class).or_insert(0) += 1;
        }
        
        self.classes = class_counts.keys().cloned().collect();
        self.classes.sort();
        
        // Compute log priors
        for (&class, &count) in &class_counts {
            self.class_log_priors.insert(class, (count as f64 / n_samples as f64).ln());
        }
        
        // Compute feature counts for each class
        for &class in &self.classes {
            let mut feature_counts = vec![self.alpha; n_features]; // Laplace smoothing
            let mut total_count = self.alpha * n_features as f64;
            
            for (row, &label) in x.rows().into_iter().zip(y.iter()) {
                if label as i64 == class {
                    for (j, &val) in row.iter().enumerate() {
                        feature_counts[j] += val.max(0.0); // Counts should be non-negative
                        total_count += val.max(0.0);
                    }
                }
            }
            
            // Convert to log probabilities
            let log_probs: Vec<f64> = feature_counts.iter()
                .map(|&count| (count / total_count).ln())
                .collect();
            
            self.feature_log_probs.insert(class, log_probs);
        }
        
        Ok(())
    }

    /// Predict class labels
    pub fn predict(&self, x: &Array2<f64>) -> Array1<f64> {
        let log_probs = self.predict_log_proba(x);
        
        log_probs.rows().into_iter()
            .map(|row| {
                let max_idx = row.iter()
                    .enumerate()
                    .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                    .map(|(i, _)| i)
                    .unwrap_or(0);
                self.classes[max_idx] as f64
            })
            .collect()
    }

    /// Predict log probabilities
    pub fn predict_log_proba(&self, x: &Array2<f64>) -> Array2<f64> {
        let n_samples = x.nrows();
        let n_classes = self.classes.len();
        
        let mut log_probs = Array2::zeros((n_samples, n_classes));
        
        for (i, row) in x.rows().into_iter().enumerate() {
            for (j, &class) in self.classes.iter().enumerate() {
                let log_prior = self.class_log_priors[&class];
                let feature_probs = &self.feature_log_probs[&class];
                
                let log_likelihood: f64 = row.iter()
                    .zip(feature_probs.iter())
                    .map(|(&xi, &log_p)| xi * log_p)
                    .sum();
                
                log_probs[[i, j]] = log_prior + log_likelihood;
            }
        }

        // Normalize (log-sum-exp trick), mirroring GaussianNaiveBayes::predict_log_proba.
        // Without this, `log_probs` is just the raw (unnormalized) joint log-likelihoods —
        // fine for `predict()`'s argmax (normalization doesn't change which class is largest),
        // but wrong for anything that exponentiates these into actual probabilities (e.g.
        // engine.rs's `predict_proba_internal`, which does exactly that): exponentiated rows
        // would not sum to 1.
        for mut row in log_probs.rows_mut() {
            let max_val = row.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let log_sum: f64 = row.iter().map(|&v| (v - max_val).exp()).sum::<f64>().ln();
            for val in row.iter_mut() {
                *val = *val - max_val - log_sum;
            }
        }

        log_probs
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_classification_data() -> (Array2<f64>, Array1<f64>) {
        // Two well-separated Gaussian clusters
        let x = Array2::from_shape_vec((20, 2), vec![
            // Class 0 (centered around 0, 0)
            -1.0, -1.0, -0.5, -0.5, 0.0, 0.0, 0.5, 0.5, -1.0, 0.0,
            -0.5, 0.5, 0.0, -0.5, 0.5, -1.0, -0.2, -0.8, -0.8, -0.2,
            // Class 1 (centered around 5, 5)
            4.0, 4.0, 4.5, 4.5, 5.0, 5.0, 5.5, 5.5, 4.0, 5.0,
            4.5, 5.5, 5.0, 4.5, 5.5, 4.0, 4.2, 4.8, 4.8, 4.2,
        ]).unwrap();
        
        let y = Array1::from_vec(vec![
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
        ]);
        
        (x, y)
    }

    #[test]
    fn test_gaussian_naive_bayes() {
        let (x, y) = create_classification_data();
        
        let mut nb = GaussianNaiveBayes::new();
        nb.fit(&x, &y).unwrap();
        
        let predictions = nb.predict(&x);
        
        // Check accuracy
        let correct: usize = y.iter()
            .zip(predictions.iter())
            .filter(|(&yi, &pi)| (yi - pi).abs() < 0.5)
            .count();
        
        let accuracy = correct as f64 / y.len() as f64;
        assert!(accuracy > 0.9, "Accuracy ({}) should be above 90%", accuracy);
    }

    #[test]
    fn test_gaussian_proba() {
        let (x, y) = create_classification_data();
        
        let mut nb = GaussianNaiveBayes::new();
        nb.fit(&x, &y).unwrap();
        
        let proba = nb.predict_proba(&x);
        
        // Probabilities should sum to 1
        for row in proba.rows() {
            let sum: f64 = row.iter().sum();
            assert!((sum - 1.0).abs() < 0.01, "Probabilities should sum to 1, got {}", sum);
        }
    }

    #[test]
    fn test_class_priors() {
        let (x, y) = create_classification_data();
        
        let mut nb = GaussianNaiveBayes::new();
        nb.fit(&x, &y).unwrap();
        
        let priors = nb.class_priors();
        
        // With balanced data, priors should be 0.5 each
        assert!((priors[&0] - 0.5).abs() < 0.01);
        assert!((priors[&1] - 0.5).abs() < 0.01);
    }

    #[test]
    fn test_multinomial_naive_bayes() {
        // Count data (like word counts in text classification)
        let x = Array2::from_shape_vec((10, 4), vec![
            // Class 0 (high counts in first two features)
            5.0, 3.0, 1.0, 0.0,
            4.0, 4.0, 0.0, 1.0,
            6.0, 2.0, 1.0, 0.0,
            5.0, 5.0, 0.0, 0.0,
            4.0, 3.0, 1.0, 1.0,
            // Class 1 (high counts in last two features)
            0.0, 1.0, 5.0, 4.0,
            1.0, 0.0, 4.0, 5.0,
            0.0, 0.0, 6.0, 3.0,
            1.0, 1.0, 5.0, 5.0,
            0.0, 1.0, 4.0, 4.0,
        ]).unwrap();
        
        let y = Array1::from_vec(vec![0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0]);
        
        let mut mnb = MultinomialNaiveBayes::new(1.0);
        mnb.fit(&x, &y).unwrap();
        
        let predictions = mnb.predict(&x);
        
        // Check accuracy
        let correct: usize = y.iter()
            .zip(predictions.iter())
            .filter(|(&yi, &pi)| (yi - pi).abs() < 0.5)
            .count();
        
        let accuracy = correct as f64 / y.len() as f64;
        assert!(accuracy > 0.8, "Accuracy ({}) should be above 80%", accuracy);
    }
}
