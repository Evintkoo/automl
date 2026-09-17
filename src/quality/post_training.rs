//! Gate 4: Post-training quality — calibration, conformal intervals, OOD.

use crate::quality::{CalibrationMethod, QualityContext};

// ── ECE ───────────────────────────────────────────────────────────────────────

/// Expected Calibration Error over `n_bins` equal-width bins.
pub fn expected_calibration_error(probs: &[f64], labels: &[bool], n_bins: usize) -> f64 {
    let n = probs.len() as f64;
    let bin_width = 1.0 / n_bins as f64;
    let mut ece = 0.0_f64;
    for b in 0..n_bins {
        let lower = b as f64 * bin_width;
        let upper = lower + bin_width;
        let bin: Vec<(f64, bool)> = probs.iter().zip(labels.iter())
            .filter(|(p, _)| **p >= lower && (**p < upper || (b == n_bins - 1 && **p <= 1.0)))
            .map(|(p, l)| (*p, *l))
            .collect();
        if bin.is_empty() { continue; }
        let bin_n = bin.len() as f64;
        let acc = bin.iter().filter(|(_, l)| *l).count() as f64 / bin_n;
        let conf = bin.iter().map(|(p, _)| p).sum::<f64>() / bin_n;
        ece += (bin_n / n) * (acc - conf).abs();
    }
    ece
}

// ── PlattCalibrator ───────────────────────────────────────────────────────────

/// Logistic (Platt) calibration: P_cal = 1 / (1 + exp(a·score + b)).
/// Fit via gradient descent on the calibration set.
pub struct PlattCalibrator {
    pub a: f64,
    pub b: f64,
}

impl PlattCalibrator {
    pub fn fit(scores: &[f64], labels: &[f64]) -> Self {
        let mut a = 0.0_f64;
        let mut b = 0.0_f64;
        let lr = 0.01;
        let n = scores.len() as f64;
        for _ in 0..1_000 {
            let mut grad_a = 0.0_f64;
            let mut grad_b = 0.0_f64;
            for (&s, &y) in scores.iter().zip(labels.iter()) {
                let p = 1.0 / (1.0 + (-(a * s + b)).exp());
                grad_a += (p - y) * s;
                grad_b += p - y;
            }
            a -= lr * grad_a / n;
            b -= lr * grad_b / n;
        }
        PlattCalibrator { a, b }
    }

    pub fn predict(&self, score: f64) -> f64 {
        1.0 / (1.0 + (-(self.a * score + self.b)).exp())
    }
}

// ── IsotonicCalibrator ────────────────────────────────────────────────────────

/// Isotonic regression calibration via Pool Adjacent Violators (PAV).
pub struct IsotonicCalibrator {
    /// Sorted score thresholds (midpoints of each block).
    pub thresholds: Vec<f64>,
    /// Calibrated probability for each block.
    pub values: Vec<f64>,
}

impl IsotonicCalibrator {
    pub fn fit(scores: &[f64], labels: &[f64]) -> Result<Self, String> {
        if scores.is_empty() || labels.is_empty() {
            return Err("IsotonicCalibrator::fit requires non-empty scores and labels".to_string());
        }
        assert_eq!(scores.len(), labels.len());
        let mut pairs: Vec<(f64, f64)> = scores.iter()
            .zip(labels.iter())
            .map(|(&s, &l)| (s, l))
            .collect();
        pairs.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

        // PAV: blocks of (label_sum, score_sum, count)
        let mut blocks: Vec<(f64, f64, usize)> = Vec::new();
        for (score, label) in &pairs {
            blocks.push((*label, *score, 1));
            while blocks.len() > 1 {
                let n = blocks.len();
                let prev_mean = blocks[n - 2].0 / blocks[n - 2].2 as f64;
                let curr_mean = blocks[n - 1].0 / blocks[n - 1].2 as f64;
                if prev_mean > curr_mean {
                    let last = blocks.pop().unwrap();
                    let prev = blocks.last_mut().unwrap();
                    prev.0 += last.0;
                    prev.1 += last.1;
                    prev.2 += last.2;
                } else {
                    break;
                }
            }
        }

        let thresholds: Vec<f64> = blocks.iter().map(|(_, s, c)| s / *c as f64).collect();
        let values: Vec<f64> = blocks.iter().map(|(l, _, c)| l / *c as f64).collect();
        Ok(IsotonicCalibrator { thresholds, values })
    }

    /// Step-function interpolation.
    pub fn predict(&self, score: f64) -> f64 {
        // Defensive fallback: `fit` validates non-empty input, but guard here
        // too in case `values` is ever empty (e.g. default-constructed).
        if self.values.is_empty() {
            return 0.5;
        }
        match self.thresholds.binary_search_by(|t| t.partial_cmp(&score).unwrap_or(std::cmp::Ordering::Less)) {
            Ok(i) => self.values[i],
            Err(i) => {
                if i == 0 { self.values[0] }
                else if i >= self.values.len() { *self.values.last().unwrap_or(&0.5) }
                else { self.values[i - 1] }
            }
        }
    }
}

use ndarray::Array2;

// ── ConformalPredictor ────────────────────────────────────────────────────────

/// Split conformal prediction for regression.
/// Fits on calibration set residuals; provides coverage-guaranteed intervals.
pub struct ConformalPredictor {
    /// The quantile of absolute residuals at the requested coverage level.
    pub quantile: f64,
}

impl ConformalPredictor {
    /// Fit from calibration set predictions and true targets.
    /// `coverage` ∈ (0, 1). Default: 0.90.
    pub fn fit(predictions: &[f64], targets: &[f64], coverage: f64) -> Result<Self, String> {
        if predictions.is_empty() || targets.is_empty() {
            return Err("ConformalPredictor::fit requires non-empty predictions and targets".to_string());
        }
        assert_eq!(predictions.len(), targets.len());
        let mut residuals: Vec<f64> = predictions.iter().zip(targets.iter())
            .map(|(p, t)| (p - t).abs())
            .collect();
        residuals.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let n = residuals.len();
        let idx = (((coverage * (n + 1) as f64).ceil() as usize)).min(n).max(1) - 1;
        Ok(ConformalPredictor { quantile: residuals[idx] })
    }

    pub fn predict_interval(&self, prediction: f64) -> (f64, f64) {
        (prediction - self.quantile, prediction + self.quantile)
    }
}

// ── Entropy confidence ────────────────────────────────────────────────────────

/// Returns normalised entropy confidence: 1 - H(p) / log(n_classes).
/// = 1.0 when the model is certain, 0.0 when maximally uncertain.
pub fn entropy_confidence(probs: &[f64]) -> f64 {
    let n = probs.len();
    if n <= 1 { return 1.0; }
    let entropy: f64 = probs.iter()
        .filter(|&&p| p > 0.0)
        .map(|&p| -p * p.ln())
        .sum();
    let max_entropy = (n as f64).ln();
    if max_entropy < 1e-10 { return 1.0; }
    1.0 - entropy / max_entropy
}

// ── OodDetector ───────────────────────────────────────────────────────────────

/// Out-of-distribution detector using normalised Euclidean distance
/// (equivalent to Mahalanobis with diagonal covariance).
pub struct OodDetector {
    pub means: Vec<f64>,
    pub stds: Vec<f64>,
    /// Distance threshold at the `percentile`-th percentile of training distances.
    pub threshold: f64,
}

impl OodDetector {
    /// Fit from training feature matrix. `percentile` ∈ (0, 1); use 0.99 for the
    /// 99th percentile of training distances as the OOD threshold.
    pub fn fit(features: &Array2<f64>, percentile: f64) -> Self {
        let n = features.nrows();
        let d = features.ncols();

        if n == 0 {
            // No calibration data to derive a threshold from. Fall back to a
            // detector that never flags anything as OOD rather than
            // panicking on an empty `distances` index below.
            return OodDetector {
                means: vec![0.0; d],
                stds: vec![1.0; d],
                threshold: f64::INFINITY,
            };
        }

        let means: Vec<f64> = (0..d).map(|j| {
            features.column(j).iter().sum::<f64>() / n as f64
        }).collect();

        let stds: Vec<f64> = (0..d).map(|j| {
            let mean = means[j];
            let var = features.column(j).iter()
                .map(|x| (x - mean).powi(2))
                .sum::<f64>() / n as f64;
            var.sqrt().max(1e-8)
        }).collect();

        let mut distances: Vec<f64> = (0..n).map(|i| {
            (0..d).map(|j| ((features[[i, j]] - means[j]) / stds[j]).powi(2))
                .sum::<f64>()
                .sqrt()
        }).collect();
        distances.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let threshold_idx = ((percentile * n as f64).ceil() as usize).min(n).max(1) - 1;
        OodDetector { means, stds, threshold: distances[threshold_idx] }
    }

    pub fn distance(&self, sample: &[f64]) -> f64 {
        sample.iter().zip(self.means.iter()).zip(self.stds.iter())
            .map(|((x, m), s)| ((x - m) / s).powi(2))
            .sum::<f64>()
            .sqrt()
    }

    pub fn is_ood(&self, sample: &[f64]) -> bool {
        self.distance(sample) > self.threshold
    }
}

// ── PostTrainingGate ──────────────────────────────────────────────────────────

pub struct PostTrainingGate {
    pub ece_improvement_threshold: f64, // default 0.005
    pub conformal_coverage: f64,        // default 0.90
    pub ood_percentile: f64,            // default 0.99
    pub n_ece_bins: usize,              // default 10
}

impl Default for PostTrainingGate {
    fn default() -> Self {
        Self {
            ece_improvement_threshold: 0.005,
            conformal_coverage: 0.90,
            ood_percentile: 0.99,
            n_ece_bins: 10,
        }
    }
}

impl PostTrainingGate {
    /// Select the better calibration method by ECE. Writes result into ctx.
    pub fn select_calibration(
        &self,
        raw_scores: &[f64],
        labels: &[f64],
        ctx: &mut QualityContext,
    ) -> Option<CalibrationMethod> {
        let bool_labels: Vec<bool> = labels.iter().map(|&l| l > 0.5).collect();
        let ece_before = expected_calibration_error(raw_scores, &bool_labels, self.n_ece_bins);
        ctx.ece_before = Some(ece_before);

        let platt = PlattCalibrator::fit(raw_scores, labels);
        let platt_probs: Vec<f64> = raw_scores.iter().map(|&s| platt.predict(s)).collect();
        let ece_platt = expected_calibration_error(&platt_probs, &bool_labels, self.n_ece_bins);

        let isotonic = match IsotonicCalibrator::fit(raw_scores, labels) {
            Ok(cal) => cal,
            Err(_) => return None,
        };
        let iso_probs: Vec<f64> = raw_scores.iter().map(|&s| isotonic.predict(s)).collect();
        let ece_iso = expected_calibration_error(&iso_probs, &bool_labels, self.n_ece_bins);

        let best_ece = ece_platt.min(ece_iso);
        if best_ece < ece_before - self.ece_improvement_threshold {
            ctx.ece_after = Some(best_ece);
            if ece_platt <= ece_iso {
                ctx.calibration_method = Some(CalibrationMethod::Platt);
                Some(CalibrationMethod::Platt)
            } else {
                ctx.calibration_method = Some(CalibrationMethod::Isotonic);
                Some(CalibrationMethod::Isotonic)
            }
        } else {
            ctx.ece_after = Some(ece_before); // no improvement
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_platt_calibrator_moves_scores_toward_labels() {
        let scores = vec![0.2, 0.4, 0.6, 0.8, 0.9, 0.1, 0.3, 0.7];
        let labels = vec![0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 1.0];
        let cal = PlattCalibrator::fit(&scores, &labels);
        assert!(cal.predict(0.9) > 0.5, "high score should give high probability");
        assert!(cal.predict(0.1) < 0.5, "low score should give low probability");
    }

    #[test]
    fn test_isotonic_calibrator_is_monotone() {
        let scores = vec![0.1, 0.3, 0.5, 0.7, 0.9];
        let labels = vec![0.0, 0.0, 1.0, 1.0, 1.0];
        let cal = IsotonicCalibrator::fit(&scores, &labels).unwrap();
        let p1 = cal.predict(0.2);
        let p2 = cal.predict(0.6);
        let p3 = cal.predict(0.8);
        assert!(p1 <= p2, "isotonic must be non-decreasing: {} <= {}", p1, p2);
        assert!(p2 <= p3, "isotonic must be non-decreasing: {} <= {}", p2, p3);
    }

    #[test]
    fn test_ece_perfect_calibration_is_zero() {
        // Use midpoint probabilities (0.05, 0.95) so each falls in a single bin
        // with accuracy matching confidence: acc=0/50=0 ≈ conf=0.05, acc=50/50=1 ≈ conf=0.95
        let probs: Vec<f64> = (0..50).map(|_| 0.05).chain((0..50).map(|_| 0.95)).collect();
        let labels: Vec<bool> = (0..50).map(|_| false).chain((0..50).map(|_| true)).collect();
        let ece = expected_calibration_error(&probs, &labels, 10);
        assert!(ece < 0.05, "ECE should be near 0 for well-calibrated model, got {}", ece);
    }

    #[test]
    fn test_ece_overconfident_model_has_high_ece() {
        let probs: Vec<f64> = vec![0.99; 100];
        let labels: Vec<bool> = (0..100).map(|i| i % 2 == 0).collect();
        let ece = expected_calibration_error(&probs, &labels, 10);
        assert!(ece > 0.4, "Overconfident model should have high ECE, got {}", ece);
    }

    #[test]
    fn test_conformal_predictor_coverage() {
        let preds: Vec<f64> = (0..100).map(|i| i as f64).collect();
        let targets: Vec<f64> = (0..100).map(|i| i as f64 + 0.5).collect(); // residuals = 0.5
        let cp = ConformalPredictor::fit(&preds, &targets, 0.90).unwrap();
        // All residuals are 0.5, so quantile should be 0.5
        assert!((cp.quantile - 0.5).abs() < 0.1, "quantile should ≈ 0.5, got {}", cp.quantile);
        let (lo, hi) = cp.predict_interval(10.0);
        assert!(lo < 10.0 && hi > 10.0, "interval should straddle prediction");
    }

    #[test]
    fn test_entropy_confidence_certain() {
        // P = [1.0, 0.0] → H = 0 → confidence = 1.0
        let probs = vec![1.0_f64, 0.0];
        let conf = entropy_confidence(&probs);
        assert!((conf - 1.0).abs() < 1e-10, "got {}", conf);
    }

    #[test]
    fn test_entropy_confidence_uniform() {
        // P = [0.5, 0.5] → H = log(2) → confidence = 0.0
        let probs = vec![0.5_f64, 0.5];
        let conf = entropy_confidence(&probs);
        assert!(conf < 0.01, "uniform distribution should give near-zero confidence, got {}", conf);
    }

    #[test]
    fn test_ood_detector_flags_out_of_distribution() {
        use ndarray::array;
        // Training: all samples near (0, 0)
        let features = array![[0.1, 0.1], [-0.1, 0.1], [0.0, -0.1], [0.1, 0.0]];
        let detector = OodDetector::fit(&features, 0.99);
        // In-distribution sample
        assert!(!detector.is_ood(&[0.05, 0.05]));
        // Far out-of-distribution sample
        assert!(detector.is_ood(&[1000.0, 1000.0]));
    }
}
