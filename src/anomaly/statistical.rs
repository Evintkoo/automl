//! Statistical anomaly detection methods

use crate::error::{AutoMLError, Result};
use crate::anomaly::AnomalyDetector;
use ndarray::{Array1, Array2, Axis};
use serde::{Deserialize, Serialize};

/// Z-Score based anomaly detector
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ZScoreDetector {
    /// Z-score threshold
    threshold: f64,
    /// Mean values per feature
    means: Option<Array1<f64>>,
    /// Standard deviation per feature
    stds: Option<Array1<f64>>,
}

impl ZScoreDetector {
    /// Create new Z-score detector
    pub fn new(threshold: f64) -> Self {
        Self {
            threshold: threshold.abs(),
            means: None,
            stds: None,
        }
    }

    /// Default with threshold of 3.0
    pub fn default() -> Self {
        Self::new(3.0)
    }
}

impl AnomalyDetector for ZScoreDetector {
    fn fit(&mut self, x: &Array2<f64>) -> Result<()> {
        let n_features = x.ncols();
        let n_samples = x.nrows() as f64;

        let mut means = Vec::with_capacity(n_features);
        let mut stds = Vec::with_capacity(n_features);

        for col_idx in 0..n_features {
            let col = x.column(col_idx);
            let mean = col.sum() / n_samples;
            let variance = col.iter().map(|&v| (v - mean).powi(2)).sum::<f64>() / n_samples;
            let std = variance.sqrt().max(1e-10);

            means.push(mean);
            stds.push(std);
        }

        self.means = Some(Array1::from_vec(means));
        self.stds = Some(Array1::from_vec(stds));

        Ok(())
    }

    fn score_samples(&self, x: &Array2<f64>) -> Result<Array1<f64>> {
        let means = self.means.as_ref().ok_or_else(|| {
            AutoMLError::ValidationError("Model not fitted".to_string())
        })?;
        let stds = self.stds.as_ref().ok_or_else(|| {
            AutoMLError::ValidationError("Model not fitted".to_string())
        })?;

        let scores: Vec<f64> = x
            .rows()
            .into_iter()
            .map(|row| {
                // Maximum absolute z-score across features
                row.iter()
                    .enumerate()
                    .map(|(i, &v)| ((v - means[i]) / stds[i]).abs())
                    .fold(0.0, f64::max)
            })
            .collect();

        Ok(Array1::from_vec(scores))
    }

    fn predict(&self, x: &Array2<f64>) -> Result<Array1<i32>> {
        let scores = self.score_samples(x)?;

        let labels: Vec<i32> = scores
            .iter()
            .map(|&s| if s > self.threshold { -1 } else { 1 })
            .collect();

        Ok(Array1::from_vec(labels))
    }

    fn threshold(&self) -> f64 {
        self.threshold
    }
}

/// Elliptic Envelope (Gaussian-based) anomaly detector
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EllipticEnvelope {
    /// Contamination ratio
    contamination: f64,
    /// Mean vector
    mean: Option<Array1<f64>>,
    /// Covariance matrix (stored as flattened)
    covariance: Option<Array2<f64>>,
    /// Inverse covariance (precision) matrix
    precision: Option<Array2<f64>>,
    /// Decision threshold
    threshold: Option<f64>,
}

impl EllipticEnvelope {
    /// Create new Elliptic Envelope detector
    pub fn new() -> Self {
        Self {
            contamination: 0.1,
            mean: None,
            covariance: None,
            precision: None,
            threshold: None,
        }
    }

    /// Set contamination ratio
    pub fn with_contamination(mut self, c: f64) -> Self {
        self.contamination = c.clamp(0.0, 0.5);
        self
    }

    /// Compute Mahalanobis distance
    fn mahalanobis_distance(&self, x: &[f64]) -> f64 {
        let mean = self.mean.as_ref().unwrap();
        let precision = self.precision.as_ref().unwrap();

        let n = x.len();
        let diff: Vec<f64> = x.iter().zip(mean.iter()).map(|(a, b)| a - b).collect();

        // (x - mu)^T * Sigma^-1 * (x - mu)
        let mut result = 0.0;
        for i in 0..n {
            for j in 0..n {
                result += diff[i] * precision[[i, j]] * diff[j];
            }
        }

        result.sqrt()
    }

    /// Simple matrix inversion for symmetric positive definite matrices
    /// Uses Cholesky decomposition
    fn invert_covariance(&self, cov: &Array2<f64>) -> Result<Array2<f64>> {
        let n = cov.nrows();
        
        // Add small regularization for numerical stability
        let mut reg_cov = cov.clone();
        for i in 0..n {
            reg_cov[[i, i]] += 1e-6;
        }

        // Cholesky decomposition: A = L * L^T
        let mut l = Array2::zeros((n, n));
        
        for i in 0..n {
            for j in 0..=i {
                let mut sum = reg_cov[[i, j]];
                for k in 0..j {
                    sum -= l[[i, k]] * l[[j, k]];
                }
                
                if i == j {
                    if sum <= 0.0 {
                        return Err(AutoMLError::ValidationError(
                            "Covariance matrix is not positive definite".to_string()
                        ));
                    }
                    l[[i, j]] = sum.sqrt();
                } else {
                    l[[i, j]] = sum / l[[j, j]];
                }
            }
        }

        // Invert L (forward substitution)
        let mut l_inv = Array2::zeros((n, n));
        for i in 0..n {
            l_inv[[i, i]] = 1.0 / l[[i, i]];
            for j in 0..i {
                let mut sum = 0.0;
                for k in j..i {
                    sum -= l[[i, k]] * l_inv[[k, j]];
                }
                l_inv[[i, j]] = sum / l[[i, i]];
            }
        }

        // Precision = L^-T * L^-1
        let mut precision = Array2::zeros((n, n));
        for i in 0..n {
            for j in 0..n {
                let mut sum = 0.0;
                for k in i.max(j)..n {
                    sum += l_inv[[k, i]] * l_inv[[k, j]];
                }
                precision[[i, j]] = sum;
            }
        }

        Ok(precision)
    }
}

impl Default for EllipticEnvelope {
    fn default() -> Self {
        Self::new()
    }
}

impl AnomalyDetector for EllipticEnvelope {
    fn fit(&mut self, x: &Array2<f64>) -> Result<()> {
        let n_samples = x.nrows();
        let n_features = x.ncols();

        if n_samples < 2 {
            return Err(AutoMLError::ValidationError(
                "EllipticEnvelope::fit requires at least 2 samples".to_string(),
            ));
        }

        // Compute mean
        let mean: Array1<f64> = x.mean_axis(Axis(0)).ok_or_else(|| {
            AutoMLError::ValidationError("Failed to compute mean".to_string())
        })?;

        // Compute covariance matrix
        let mut cov = Array2::zeros((n_features, n_features));
        
        for i in 0..n_features {
            for j in 0..=i {
                let cov_ij: f64 = x.rows()
                    .into_iter()
                    .map(|row| (row[i] - mean[i]) * (row[j] - mean[j]))
                    .sum::<f64>() / (n_samples - 1).max(1) as f64;
                
                cov[[i, j]] = cov_ij;
                cov[[j, i]] = cov_ij;
            }
        }

        // Invert covariance matrix
        let precision = self.invert_covariance(&cov)?;

        self.mean = Some(mean);
        self.covariance = Some(cov);
        self.precision = Some(precision);

        // Compute threshold based on contamination
        let scores = self.score_samples(x)?;
        let mut sorted_scores: Vec<f64> = scores.iter().copied().collect();
        sorted_scores.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
        
        let threshold_idx = ((self.contamination * n_samples as f64) as usize).min(n_samples - 1);
        self.threshold = Some(sorted_scores[threshold_idx]);

        Ok(())
    }

    fn score_samples(&self, x: &Array2<f64>) -> Result<Array1<f64>> {
        if self.mean.is_none() {
            return Err(AutoMLError::ValidationError("Model not fitted".to_string()));
        }

        let scores: Vec<f64> = x
            .rows()
            .into_iter()
            .map(|row| {
                let point: Vec<f64> = row.iter().copied().collect();
                self.mahalanobis_distance(&point)
            })
            .collect();

        Ok(Array1::from_vec(scores))
    }

    fn predict(&self, x: &Array2<f64>) -> Result<Array1<i32>> {
        let scores = self.score_samples(x)?;
        let threshold = self.threshold.unwrap_or(3.0);

        let labels: Vec<i32> = scores
            .iter()
            .map(|&s| if s > threshold { -1 } else { 1 })
            .collect();

        Ok(Array1::from_vec(labels))
    }

    fn threshold(&self) -> f64 {
        self.threshold.unwrap_or(3.0)
    }
}

/// General statistical detector combining multiple methods
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StatisticalDetector {
    /// IQR multiplier for outlier detection
    iqr_multiplier: f64,
    /// Lower bounds per feature
    lower_bounds: Option<Array1<f64>>,
    /// Upper bounds per feature
    upper_bounds: Option<Array1<f64>>,
    /// Threshold (number of features that must be outliers)
    threshold: usize,
}

impl StatisticalDetector {
    /// Create new statistical detector
    pub fn new(iqr_multiplier: f64) -> Self {
        Self {
            iqr_multiplier: iqr_multiplier.max(1.0),
            lower_bounds: None,
            upper_bounds: None,
            threshold: 1,
        }
    }

    /// Set threshold (minimum number of outlier features)
    pub fn with_threshold(mut self, t: usize) -> Self {
        self.threshold = t.max(1);
        self
    }

    /// Compute quartiles
    fn quartiles(values: &[f64]) -> (f64, f64, f64) {
        let mut sorted = values.to_vec();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        
        let n = sorted.len();
        let q1_idx = n / 4;
        let q2_idx = n / 2;
        let q3_idx = 3 * n / 4;

        (sorted[q1_idx], sorted[q2_idx], sorted[q3_idx])
    }
}

impl Default for StatisticalDetector {
    fn default() -> Self {
        Self::new(1.5)
    }
}

impl AnomalyDetector for StatisticalDetector {
    fn fit(&mut self, x: &Array2<f64>) -> Result<()> {
        let n_features = x.ncols();
        let mut lower_bounds = Vec::with_capacity(n_features);
        let mut upper_bounds = Vec::with_capacity(n_features);

        for col_idx in 0..n_features {
            let values: Vec<f64> = x.column(col_idx).iter().copied().collect();
            let (q1, _, q3) = Self::quartiles(&values);
            let iqr = q3 - q1;

            lower_bounds.push(q1 - self.iqr_multiplier * iqr);
            upper_bounds.push(q3 + self.iqr_multiplier * iqr);
        }

        self.lower_bounds = Some(Array1::from_vec(lower_bounds));
        self.upper_bounds = Some(Array1::from_vec(upper_bounds));

        Ok(())
    }

    fn score_samples(&self, x: &Array2<f64>) -> Result<Array1<f64>> {
        let lower = self.lower_bounds.as_ref().ok_or_else(|| {
            AutoMLError::ValidationError("Model not fitted".to_string())
        })?;
        let upper = self.upper_bounds.as_ref().ok_or_else(|| {
            AutoMLError::ValidationError("Model not fitted".to_string())
        })?;

        let scores: Vec<f64> = x
            .rows()
            .into_iter()
            .map(|row| {
                // Count number of features outside bounds
                row.iter()
                    .enumerate()
                    .filter(|(i, &v)| v < lower[*i] || v > upper[*i])
                    .count() as f64
            })
            .collect();

        Ok(Array1::from_vec(scores))
    }

    fn predict(&self, x: &Array2<f64>) -> Result<Array1<i32>> {
        let scores = self.score_samples(x)?;

        let labels: Vec<i32> = scores
            .iter()
            .map(|&s| if s >= self.threshold as f64 { -1 } else { 1 })
            .collect();

        Ok(Array1::from_vec(labels))
    }

    fn threshold(&self) -> f64 {
        self.threshold as f64
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_zscore_detector() {
        // 50 normal points + 1 outlier
        let mut data = Vec::new();
        for i in 0..50 {
            data.push((i % 10) as f64);
            data.push(((i + 5) % 10) as f64);
        }
        // Add outlier
        data.extend_from_slice(&[100.0, 100.0]);

        let x = Array2::from_shape_vec((51, 2), data).unwrap();

        let mut detector = ZScoreDetector::new(3.0);
        detector.fit(&x).unwrap();

        let labels = detector.predict(&x).unwrap();
        
        // Last point should be detected as outlier
        assert_eq!(labels[50], -1);
    }

    #[test]
    fn test_elliptic_envelope() {
        // 20 points with 2 features
        let data: Vec<f64> = (0..20)
            .flat_map(|i| vec![(i % 5) as f64 + 0.1, ((i + 2) % 5) as f64 + 0.1])
            .collect();
        
        let x = Array2::from_shape_vec((20, 2), data).unwrap();
        
        let mut detector = EllipticEnvelope::new()
            .with_contamination(0.1);
        
        detector.fit(&x).unwrap();
        
        let scores = detector.score_samples(&x).unwrap();
        assert_eq!(scores.len(), 20);
    }

    #[test]
    fn test_statistical_detector() {
        // 25 normal points + 1 outlier
        let mut data = Vec::new();
        for i in 0..25 {
            data.push((i % 10) as f64);
            data.push(((i + 3) % 10) as f64);
        }
        // Add outlier
        data.extend_from_slice(&[1000.0, 1000.0]);

        let x = Array2::from_shape_vec((26, 2), data).unwrap();

        let mut detector = StatisticalDetector::new(1.5)
            .with_threshold(1);
        
        detector.fit(&x).unwrap();
        let labels = detector.predict(&x).unwrap();

        // Outlier should be detected
        assert_eq!(labels[25], -1);
    }
}
