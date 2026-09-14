//! Calibration metrics

use crate::error::{AutoMLError, Result};
use ndarray::Array1;
use serde::{Deserialize, Serialize};

/// Calibration metrics container
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CalibrationMetrics {
    /// Expected Calibration Error
    pub ece: f64,
    /// Maximum Calibration Error
    pub mce: f64,
    /// Brier Score
    pub brier_score: f64,
    /// Average confidence
    pub avg_confidence: f64,
    /// Accuracy
    pub accuracy: f64,
    /// Reliability diagram data
    pub reliability_data: Option<ReliabilityDiagram>,
}

/// Reliability diagram data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReliabilityDiagram {
    /// Bin edges
    pub bin_edges: Vec<f64>,
    /// Mean predicted probability in each bin
    pub bin_confidences: Vec<f64>,
    /// Fraction of positives in each bin
    pub bin_accuracies: Vec<f64>,
    /// Number of samples in each bin
    pub bin_counts: Vec<usize>,
}

/// Compute Expected Calibration Error (ECE)
/// 
/// ECE = sum_i (|B_i| / n) * |acc(B_i) - conf(B_i)|
/// 
/// where B_i is bin i, acc is accuracy, conf is average confidence
pub fn expected_calibration_error(
    probs: &Array1<f64>,
    labels: &Array1<f64>,
    n_bins: usize,
) -> Result<f64> {
    let reliability = reliability_diagram(probs, labels, n_bins)?;
    
    let n = probs.len() as f64;
    let mut ece = 0.0;

    for i in 0..reliability.bin_confidences.len() {
        let count = reliability.bin_counts[i] as f64;
        if count > 0.0 {
            let acc = reliability.bin_accuracies[i];
            let conf = reliability.bin_confidences[i];
            ece += (count / n) * (acc - conf).abs();
        }
    }

    Ok(ece)
}

/// Compute Maximum Calibration Error (MCE)
/// 
/// MCE = max_i |acc(B_i) - conf(B_i)|
pub fn maximum_calibration_error(
    probs: &Array1<f64>,
    labels: &Array1<f64>,
    n_bins: usize,
) -> Result<f64> {
    let reliability = reliability_diagram(probs, labels, n_bins)?;
    
    let mut mce: f64 = 0.0;

    for i in 0..reliability.bin_confidences.len() {
        if reliability.bin_counts[i] > 0 {
            let acc = reliability.bin_accuracies[i];
            let conf = reliability.bin_confidences[i];
            mce = mce.max((acc - conf).abs());
        }
    }

    Ok(mce)
}

/// Compute Brier Score
/// 
/// Brier = (1/n) * sum_i (p_i - y_i)^2
pub fn brier_score(probs: &Array1<f64>, labels: &Array1<f64>) -> Result<f64> {
    if probs.len() != labels.len() {
        return Err(AutoMLError::ValidationError(
            "Probabilities and labels must have same length".to_string(),
        ));
    }

    let n = probs.len() as f64;
    let score: f64 = probs
        .iter()
        .zip(labels.iter())
        .map(|(&p, &y)| (p - y).powi(2))
        .sum();

    Ok(score / n)
}

/// Compute reliability diagram data
pub fn reliability_diagram(
    probs: &Array1<f64>,
    labels: &Array1<f64>,
    n_bins: usize,
) -> Result<ReliabilityDiagram> {
    if probs.len() != labels.len() {
        return Err(AutoMLError::ValidationError(
            "Probabilities and labels must have same length".to_string(),
        ));
    }

    let n_bins = n_bins.max(1);
    let bin_width = 1.0 / n_bins as f64;

    let mut bin_edges = Vec::with_capacity(n_bins + 1);
    for i in 0..=n_bins {
        bin_edges.push(i as f64 * bin_width);
    }

    let mut bin_sums = vec![0.0; n_bins];
    let mut bin_correct = vec![0.0; n_bins];
    let mut bin_counts = vec![0usize; n_bins];

    for (&p, &y) in probs.iter().zip(labels.iter()) {
        let bin_idx = ((p / bin_width) as usize).min(n_bins - 1);
        bin_sums[bin_idx] += p;
        bin_correct[bin_idx] += y;
        bin_counts[bin_idx] += 1;
    }

    let bin_confidences: Vec<f64> = bin_sums
        .iter()
        .zip(bin_counts.iter())
        .map(|(&sum, &count)| {
            if count > 0 {
                sum / count as f64
            } else {
                0.0
            }
        })
        .collect();

    let bin_accuracies: Vec<f64> = bin_correct
        .iter()
        .zip(bin_counts.iter())
        .map(|(&correct, &count)| {
            if count > 0 {
                correct / count as f64
            } else {
                0.0
            }
        })
        .collect();

    Ok(ReliabilityDiagram {
        bin_edges,
        bin_confidences,
        bin_accuracies,
        bin_counts,
    })
}

/// Compute all calibration metrics
#[allow(dead_code)]
pub fn compute_calibration_metrics(
    probs: &Array1<f64>,
    labels: &Array1<f64>,
    n_bins: usize,
) -> Result<CalibrationMetrics> {
    let reliability = reliability_diagram(probs, labels, n_bins)?;
    
    let ece = expected_calibration_error(probs, labels, n_bins)?;
    let mce = maximum_calibration_error(probs, labels, n_bins)?;
    let brier = brier_score(probs, labels)?;

    let avg_confidence = probs.mean().unwrap_or(0.0);
    
    // Compute accuracy (using 0.5 threshold)
    let correct: usize = probs
        .iter()
        .zip(labels.iter())
        .filter(|(&p, &y)| (p >= 0.5) == (y > 0.5))
        .count();
    let accuracy = correct as f64 / probs.len() as f64;

    Ok(CalibrationMetrics {
        ece,
        mce,
        brier_score: brier,
        avg_confidence,
        accuracy,
        reliability_data: Some(reliability),
    })
}

/// Adaptive ECE with variable bin widths
#[allow(dead_code)]
pub fn adaptive_expected_calibration_error(
    probs: &Array1<f64>,
    labels: &Array1<f64>,
    n_bins: usize,
) -> Result<f64> {
    if probs.len() != labels.len() {
        return Err(AutoMLError::ValidationError(
            "Probabilities and labels must have same length".to_string(),
        ));
    }

    let n = probs.len();
    let n_bins = n_bins.max(1).min(n);
    let bin_size = (n + n_bins - 1) / n_bins;

    // Sort by probability
    let mut indexed: Vec<(usize, f64)> = probs.iter().copied().enumerate().collect();
    indexed.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

    let mut ece = 0.0;

    for bin_start in (0..n).step_by(bin_size) {
        let bin_end = (bin_start + bin_size).min(n);
        let count = bin_end - bin_start;

        if count == 0 {
            continue;
        }

        let bin_indices: Vec<usize> = indexed[bin_start..bin_end]
            .iter()
            .map(|(i, _)| *i)
            .collect();

        let bin_conf: f64 = bin_indices.iter().map(|&i| probs[i]).sum::<f64>() / count as f64;
        let bin_acc: f64 = bin_indices.iter().map(|&i| labels[i]).sum::<f64>() / count as f64;

        ece += (count as f64 / n as f64) * (bin_acc - bin_conf).abs();
    }

    Ok(ece)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_brier_score() {
        // Perfect predictions
        let probs = array![1.0, 0.0, 1.0, 0.0];
        let labels = array![1.0, 0.0, 1.0, 0.0];
        let score = brier_score(&probs, &labels).unwrap();
        assert!((score - 0.0).abs() < 1e-10);

        // Worst predictions
        let probs = array![0.0, 1.0, 0.0, 1.0];
        let labels = array![1.0, 0.0, 1.0, 0.0];
        let score = brier_score(&probs, &labels).unwrap();
        assert!((score - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_ece() {
        let probs = array![0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9];
        let labels = array![0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0];

        let ece = expected_calibration_error(&probs, &labels, 10).unwrap();
        // Well-calibrated predictions should have low ECE
        assert!(ece < 0.5);
    }

    #[test]
    fn test_reliability_diagram() {
        let probs = array![0.1, 0.3, 0.5, 0.7, 0.9];
        let labels = array![0.0, 0.0, 1.0, 1.0, 1.0];

        let diagram = reliability_diagram(&probs, &labels, 5).unwrap();
        
        assert_eq!(diagram.bin_edges.len(), 6); // n_bins + 1
        assert_eq!(diagram.bin_confidences.len(), 5);
        assert_eq!(diagram.bin_accuracies.len(), 5);
        assert_eq!(diagram.bin_counts.len(), 5);
    }

    #[test]
    fn test_calibration_metrics() {
        let probs = array![0.2, 0.4, 0.6, 0.8];
        let labels = array![0.0, 0.0, 1.0, 1.0];

        let metrics = compute_calibration_metrics(&probs, &labels, 10).unwrap();
        
        assert!(metrics.ece >= 0.0);
        assert!(metrics.mce >= 0.0);
        assert!(metrics.brier_score >= 0.0 && metrics.brier_score <= 1.0);
        assert!(metrics.accuracy >= 0.0 && metrics.accuracy <= 1.0);
    }
}
