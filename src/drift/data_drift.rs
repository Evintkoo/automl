//! Data drift detection methods

use crate::error::{AutoMLError, Result};
use crate::drift::{DriftDetector, DriftResult};
use ndarray::Array1;
use serde::{Deserialize, Serialize};
use std::cmp::Ordering;

/// Kolmogorov-Smirnov test for distribution comparison
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KolmogorovSmirnovTest {
    /// Significance level (alpha)
    alpha: f64,
}

impl KolmogorovSmirnovTest {
    /// Create new KS test
    pub fn new(alpha: f64) -> Self {
        Self {
            alpha: alpha.clamp(0.001, 0.5),
        }
    }

    /// Compute critical value for KS test
    fn critical_value(&self, n1: usize, n2: usize) -> f64 {
        // Approximation for two-sample KS test critical value
        let c_alpha = match self.alpha {
            a if a <= 0.01 => 1.63,
            a if a <= 0.05 => 1.36,
            a if a <= 0.10 => 1.22,
            _ => 1.07,
        };

        c_alpha * ((n1 + n2) as f64 / (n1 * n2) as f64).sqrt()
    }

    /// Compute empirical CDF using binary search on sorted data (O(log n))
    fn ecdf(sorted_data: &[f64], x: f64) -> f64 {
        let count = sorted_data.partition_point(|&v| v <= x);
        count as f64 / sorted_data.len() as f64
    }
}

impl Default for KolmogorovSmirnovTest {
    fn default() -> Self {
        Self::new(0.05)
    }
}

impl DriftDetector for KolmogorovSmirnovTest {
    fn detect(&self, reference: &Array1<f64>, test: &Array1<f64>) -> Result<DriftResult> {
        if reference.is_empty() || test.is_empty() {
            return Err(AutoMLError::ValidationError(
                "Empty arrays provided".to_string()
            ));
        }

        // Sort both arrays
        let mut ref_sorted: Vec<f64> = reference.iter().copied().collect();
        let mut test_sorted: Vec<f64> = test.iter().copied().collect();
        ref_sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal));
        test_sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal));

        // Compute KS statistic (maximum absolute difference between ECDFs)
        let mut combined: Vec<f64> = ref_sorted.iter().chain(test_sorted.iter()).copied().collect();
        combined.sort_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal));
        combined.dedup();

        let ks_statistic = combined
            .iter()
            .map(|&x| {
                let f1 = Self::ecdf(&ref_sorted, x);
                let f2 = Self::ecdf(&test_sorted, x);
                (f1 - f2).abs()
            })
            .fold(0.0, f64::max);

        let threshold = self.critical_value(reference.len(), test.len());

        if ks_statistic > threshold {
            let severity = if ks_statistic > threshold * 1.5 { 2 } else { 1 };
            Ok(DriftResult::drift(
                ks_statistic,
                threshold,
                severity,
                &format!("KS statistic ({:.4}) exceeds threshold ({:.4})", ks_statistic, threshold),
            ))
        } else {
            Ok(DriftResult::no_drift(ks_statistic, threshold))
        }
    }

    fn threshold(&self) -> f64 {
        self.alpha
    }

    fn reset(&mut self) {}
}

/// Population Stability Index (PSI)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PopulationStabilityIndex {
    /// Number of bins
    n_bins: usize,
    /// Warning threshold
    warning_threshold: f64,
    /// Critical threshold
    critical_threshold: f64,
}

impl PopulationStabilityIndex {
    /// Create new PSI calculator
    pub fn new(n_bins: usize) -> Self {
        Self {
            n_bins: n_bins.max(5),
            warning_threshold: 0.1,
            critical_threshold: 0.2,
        }
    }

    /// Set custom thresholds
    pub fn with_thresholds(mut self, warning: f64, critical: f64) -> Self {
        self.warning_threshold = warning.max(0.0);
        self.critical_threshold = critical.max(warning);
        self
    }

    /// Compute bin edges from reference data
    fn compute_bin_edges(&self, data: &[f64]) -> Vec<f64> {
        let mut sorted: Vec<f64> = data.to_vec();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal));

        let mut edges = Vec::with_capacity(self.n_bins + 1);
        edges.push(f64::NEG_INFINITY);

        for i in 1..self.n_bins {
            let idx = (i * sorted.len()) / self.n_bins;
            edges.push(sorted[idx]);
        }

        edges.push(f64::INFINITY);
        edges
    }

    /// Bin data using edges
    fn bin_data(&self, data: &[f64], edges: &[f64]) -> Vec<f64> {
        let n = data.len() as f64;
        let mut counts = vec![0usize; self.n_bins];

        for &value in data {
            for i in 0..self.n_bins {
                if value > edges[i] && value <= edges[i + 1] {
                    counts[i] += 1;
                    break;
                }
            }
        }

        // Convert to proportions with small epsilon to avoid division by zero
        counts.iter()
            .map(|&c| (c as f64 / n).max(0.0001))
            .collect()
    }
}

impl Default for PopulationStabilityIndex {
    fn default() -> Self {
        Self::new(10)
    }
}

impl DriftDetector for PopulationStabilityIndex {
    fn detect(&self, reference: &Array1<f64>, test: &Array1<f64>) -> Result<DriftResult> {
        if reference.is_empty() || test.is_empty() {
            return Err(AutoMLError::ValidationError(
                "Empty arrays provided".to_string()
            ));
        }

        let ref_vec: Vec<f64> = reference.iter().copied().collect();
        let test_vec: Vec<f64> = test.iter().copied().collect();

        // Compute bins from reference data
        let edges = self.compute_bin_edges(&ref_vec);

        // Get proportions for each dataset
        let ref_props = self.bin_data(&ref_vec, &edges);
        let test_props = self.bin_data(&test_vec, &edges);

        // Compute PSI
        let psi: f64 = ref_props
            .iter()
            .zip(test_props.iter())
            .map(|(&p_ref, &p_test)| (p_test - p_ref) * (p_test / p_ref).ln())
            .sum();

        if psi > self.critical_threshold {
            Ok(DriftResult::drift(
                psi,
                self.critical_threshold,
                2,
                &format!("PSI ({:.4}) exceeds critical threshold ({:.4})", psi, self.critical_threshold),
            ))
        } else if psi > self.warning_threshold {
            Ok(DriftResult::drift(
                psi,
                self.warning_threshold,
                1,
                &format!("PSI ({:.4}) exceeds warning threshold ({:.4})", psi, self.warning_threshold),
            ))
        } else {
            Ok(DriftResult::no_drift(psi, self.warning_threshold))
        }
    }

    fn threshold(&self) -> f64 {
        self.warning_threshold
    }

    fn reset(&mut self) {}
}

/// Jensen-Shannon Divergence
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JensenShannonDivergence {
    /// Number of bins for histogram
    n_bins: usize,
    /// Threshold for drift detection
    threshold: f64,
}

impl JensenShannonDivergence {
    /// Create new JS divergence calculator
    pub fn new(threshold: f64) -> Self {
        Self {
            n_bins: 20,
            threshold: threshold.clamp(0.0, 1.0),
        }
    }

    /// Set number of bins
    pub fn with_bins(mut self, n: usize) -> Self {
        self.n_bins = n.max(5);
        self
    }

    /// Compute histogram
    fn histogram(&self, data: &[f64], min_val: f64, max_val: f64) -> Vec<f64> {
        let range = max_val - min_val;
        let bin_width = range / self.n_bins as f64;
        let mut counts = vec![0usize; self.n_bins];
        let n = data.len() as f64;

        for &value in data {
            let bin = ((value - min_val) / bin_width).floor() as usize;
            let bin = bin.min(self.n_bins - 1);
            counts[bin] += 1;
        }

        // Convert to probabilities with smoothing
        let epsilon = 1e-10;
        counts.iter()
            .map(|&c| (c as f64 / n) + epsilon)
            .collect()
    }

    /// Compute KL divergence
    fn kl_divergence(p: &[f64], q: &[f64]) -> f64 {
        p.iter()
            .zip(q.iter())
            .map(|(&pi, &qi)| {
                if pi > 0.0 {
                    pi * (pi / qi).ln()
                } else {
                    0.0
                }
            })
            .sum()
    }
}

impl Default for JensenShannonDivergence {
    fn default() -> Self {
        Self::new(0.1)
    }
}

impl DriftDetector for JensenShannonDivergence {
    fn detect(&self, reference: &Array1<f64>, test: &Array1<f64>) -> Result<DriftResult> {
        if reference.is_empty() || test.is_empty() {
            return Err(AutoMLError::ValidationError(
                "Empty arrays provided".to_string()
            ));
        }

        // Find global min/max
        let ref_min = reference.iter().cloned().fold(f64::INFINITY, f64::min);
        let ref_max = reference.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let test_min = test.iter().cloned().fold(f64::INFINITY, f64::min);
        let test_max = test.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

        let min_val = ref_min.min(test_min);
        let max_val = ref_max.max(test_max);

        if (max_val - min_val).abs() < 1e-10 {
            return Ok(DriftResult::no_drift(0.0, self.threshold));
        }

        // Compute histograms
        let ref_vec: Vec<f64> = reference.iter().copied().collect();
        let test_vec: Vec<f64> = test.iter().copied().collect();

        let p = self.histogram(&ref_vec, min_val, max_val);
        let q = self.histogram(&test_vec, min_val, max_val);

        // Compute mean distribution M = (P + Q) / 2
        let m: Vec<f64> = p.iter().zip(q.iter()).map(|(&pi, &qi)| (pi + qi) / 2.0).collect();

        // JS divergence = (KL(P||M) + KL(Q||M)) / 2
        let js = (Self::kl_divergence(&p, &m) + Self::kl_divergence(&q, &m)) / 2.0;

        // JS is bounded [0, ln(2)], normalize to [0, 1]
        let js_normalized = js / 2.0_f64.ln();

        if js_normalized > self.threshold {
            let severity = if js_normalized > self.threshold * 2.0 { 2 } else { 1 };
            Ok(DriftResult::drift(
                js_normalized,
                self.threshold,
                severity,
                &format!("JS divergence ({:.4}) exceeds threshold ({:.4})", js_normalized, self.threshold),
            ))
        } else {
            Ok(DriftResult::no_drift(js_normalized, self.threshold))
        }
    }

    fn threshold(&self) -> f64 {
        self.threshold
    }

    fn reset(&mut self) {}
}

/// Combined data drift detector
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataDriftDetector {
    /// KS test
    ks_test: KolmogorovSmirnovTest,
    /// PSI calculator
    psi: PopulationStabilityIndex,
    /// JS divergence
    js_div: JensenShannonDivergence,
    /// Voting threshold (number of methods that must agree)
    voting_threshold: usize,
}

impl DataDriftDetector {
    /// Create new combined detector
    pub fn new() -> Self {
        Self {
            ks_test: KolmogorovSmirnovTest::default(),
            psi: PopulationStabilityIndex::default(),
            js_div: JensenShannonDivergence::default(),
            voting_threshold: 2,
        }
    }

    /// Detect drift using all methods
    pub fn detect_all(&self, reference: &Array1<f64>, test: &Array1<f64>) -> Result<Vec<DriftResult>> {
        let results = vec![
            self.ks_test.detect(reference, test)?,
            self.psi.detect(reference, test)?,
            self.js_div.detect(reference, test)?,
        ];
        Ok(results)
    }

    /// Detect drift with voting
    pub fn detect(&self, reference: &Array1<f64>, test: &Array1<f64>) -> Result<DriftResult> {
        let results = self.detect_all(reference, test)?;
        let drift_count = results.iter().filter(|r| r.drift_detected).count();

        if drift_count >= self.voting_threshold {
            let max_severity = results.iter().map(|r| r.severity).max().unwrap_or(1);
            let avg_score = results.iter().map(|r| r.score).sum::<f64>() / results.len() as f64;
            
            Ok(DriftResult::drift(
                avg_score,
                0.0,
                max_severity,
                &format!("{} out of {} methods detected drift", drift_count, results.len()),
            ))
        } else {
            let avg_score = results.iter().map(|r| r.score).sum::<f64>() / results.len() as f64;
            Ok(DriftResult::no_drift(avg_score, 0.0))
        }
    }
}

impl Default for DataDriftDetector {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ks_test_no_drift() {
        // Same distribution
        let ref_data = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]);
        let test_data = Array1::from_vec(vec![1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5, 10.5]);

        let detector = KolmogorovSmirnovTest::new(0.05);
        let result = detector.detect(&ref_data, &test_data).unwrap();

        assert!(!result.drift_detected);
    }

    #[test]
    fn test_ks_test_with_drift() {
        // Different distributions
        let ref_data = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]);
        let test_data = Array1::from_vec(vec![100.0, 110.0, 120.0, 130.0, 140.0, 150.0, 160.0, 170.0, 180.0, 190.0]);

        let detector = KolmogorovSmirnovTest::new(0.05);
        let result = detector.detect(&ref_data, &test_data).unwrap();

        assert!(result.drift_detected);
    }

    #[test]
    fn test_psi_no_drift() {
        // Similar distributions
        let ref_data = Array1::from_vec((0..100).map(|i| (i % 10) as f64).collect());
        let test_data = Array1::from_vec((0..100).map(|i| ((i + 1) % 10) as f64).collect());

        let detector = PopulationStabilityIndex::new(10);
        let result = detector.detect(&ref_data, &test_data).unwrap();

        assert!(result.score < 0.2); // Should have low PSI
    }

    #[test]
    fn test_js_divergence() {
        // Different distributions
        let ref_data = Array1::from_vec(vec![1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0]);
        let test_data = Array1::from_vec(vec![5.0, 5.1, 5.2, 5.3, 5.4, 5.5, 5.6, 5.7, 5.8, 5.9, 6.0]);

        let detector = JensenShannonDivergence::new(0.1);
        let result = detector.detect(&ref_data, &test_data).unwrap();

        assert!(result.drift_detected);
        assert!(result.score > 0.1);
    }
}
