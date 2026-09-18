//! Concept drift detection (online/streaming methods)

use crate::drift::DriftResult;
use serde::{Deserialize, Serialize};

/// Drift Detection Method (DDM)
/// Based on monitoring prediction error rates
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DDM {
    /// Minimum number of samples before detection
    min_samples: usize,
    /// Warning level multiplier
    warning_level: f64,
    /// Drift level multiplier  
    drift_level: f64,
    /// Number of samples seen
    n: usize,
    /// Sum of errors
    sum_errors: f64,
    /// Minimum error rate seen
    min_p: f64,
    /// Standard deviation at minimum
    min_s: f64,
    /// Current state (0=normal, 1=warning, 2=drift)
    state: u8,
}

impl DDM {
    /// Create new DDM detector
    pub fn new() -> Self {
        Self {
            min_samples: 30,
            warning_level: 2.0,
            drift_level: 3.0,
            n: 0,
            sum_errors: 0.0,
            min_p: f64::MAX,
            min_s: f64::MAX,
            state: 0,
        }
    }

    /// Set minimum samples
    pub fn with_min_samples(mut self, n: usize) -> Self {
        self.min_samples = n.max(10);
        self
    }

    /// Set warning and drift levels
    pub fn with_levels(mut self, warning: f64, drift: f64) -> Self {
        self.warning_level = warning.max(1.0);
        self.drift_level = drift.max(warning);
        self
    }

    /// Add a single prediction result (0 = correct, 1 = error)
    pub fn add_element(&mut self, error: bool) -> DriftResult {
        self.n += 1;
        if error {
            self.sum_errors += 1.0;
        }

        if self.n < self.min_samples {
            return DriftResult::no_drift(0.0, self.drift_level);
        }

        // Current error rate
        let p = self.sum_errors / self.n as f64;
        // Standard deviation
        let s = (p * (1.0 - p) / self.n as f64).sqrt();

        // Update minimums
        if p + s < self.min_p + self.min_s {
            self.min_p = p;
            self.min_s = s;
        }

        // Check for drift
        if p + s >= self.min_p + self.drift_level * self.min_s {
            self.state = 2;
            DriftResult::drift(
                p,
                self.min_p + self.drift_level * self.min_s,
                2,
                "Concept drift detected (DDM)",
            )
        } else if p + s >= self.min_p + self.warning_level * self.min_s {
            self.state = 1;
            DriftResult::drift(
                p,
                self.min_p + self.warning_level * self.min_s,
                1,
                "Warning: possible drift (DDM)",
            )
        } else {
            self.state = 0;
            DriftResult::no_drift(p, self.min_p + self.warning_level * self.min_s)
        }
    }

    /// Process a batch of predictions
    pub fn detect_batch(&mut self, errors: &[bool]) -> Vec<DriftResult> {
        errors.iter().map(|&e| self.add_element(e)).collect()
    }

    /// Reset detector state
    pub fn reset(&mut self) {
        self.n = 0;
        self.sum_errors = 0.0;
        self.min_p = f64::MAX;
        self.min_s = f64::MAX;
        self.state = 0;
    }

    /// Get current state
    pub fn state(&self) -> u8 {
        self.state
    }
}

impl Default for DDM {
    fn default() -> Self {
        Self::new()
    }
}

/// Early Drift Detection Method (EDDM)
/// More sensitive to gradual drift than DDM
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EDDM {
    /// Minimum number of errors before detection
    min_errors: usize,
    /// Warning level
    warning_level: f64,
    /// Drift level
    drift_level: f64,
    /// Number of samples
    n: usize,
    /// Number of errors
    n_errors: usize,
    /// Last error position
    last_error_pos: usize,
    /// Sum of distances between errors
    sum_distances: f64,
    /// Sum of squared distances
    sum_sq_distances: f64,
    /// Maximum distance mean + 2*std
    max_p: f64,
    /// State
    state: u8,
}

impl EDDM {
    /// Create new EDDM detector
    pub fn new() -> Self {
        Self {
            min_errors: 30,
            warning_level: 0.95,
            drift_level: 0.90,
            n: 0,
            n_errors: 0,
            last_error_pos: 0,
            sum_distances: 0.0,
            sum_sq_distances: 0.0,
            max_p: 0.0,
            state: 0,
        }
    }

    /// Add a single prediction result
    pub fn add_element(&mut self, error: bool) -> DriftResult {
        self.n += 1;

        if error {
            if self.n_errors > 0 {
                let distance = (self.n - self.last_error_pos) as f64;
                self.sum_distances += distance;
                self.sum_sq_distances += distance * distance;
            }
            self.n_errors += 1;
            self.last_error_pos = self.n;
        }

        if self.n_errors < self.min_errors {
            return DriftResult::no_drift(0.0, self.drift_level);
        }

        // Mean and std of distances between errors
        let mean = self.sum_distances / self.n_errors as f64;
        let std = (self.sum_sq_distances / self.n_errors as f64 - mean * mean).max(0.0).sqrt();
        let p = mean + 2.0 * std;

        // Update maximum
        if p > self.max_p {
            self.max_p = p;
        }

        let ratio = p / self.max_p;

        if ratio < self.drift_level {
            self.state = 2;
            DriftResult::drift(ratio, self.drift_level, 2, "Concept drift detected (EDDM)")
        } else if ratio < self.warning_level {
            self.state = 1;
            DriftResult::drift(ratio, self.warning_level, 1, "Warning: possible drift (EDDM)")
        } else {
            self.state = 0;
            DriftResult::no_drift(ratio, self.warning_level)
        }
    }

    /// Reset detector
    pub fn reset(&mut self) {
        self.n = 0;
        self.n_errors = 0;
        self.last_error_pos = 0;
        self.sum_distances = 0.0;
        self.sum_sq_distances = 0.0;
        self.max_p = 0.0;
        self.state = 0;
    }
}

impl Default for EDDM {
    fn default() -> Self {
        Self::new()
    }
}

/// ADWIN (Adaptive Windowing) detector
/// Automatically adjusts window size based on data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ADWIN {
    /// Confidence parameter (delta)
    delta: f64,
    /// Window of values
    window: std::collections::VecDeque<f64>,
    /// Sum of window values
    total: f64,
    /// Variance estimate
    variance: f64,
    /// Maximum window size
    max_window: usize,
}

impl ADWIN {
    /// Create new ADWIN detector
    pub fn new(delta: f64) -> Self {
        Self {
            delta: delta.clamp(0.0001, 0.5),
            window: std::collections::VecDeque::new(),
            total: 0.0,
            variance: 0.0,
            max_window: 10000,
        }
    }

    /// Set maximum window size
    pub fn with_max_window(mut self, max: usize) -> Self {
        self.max_window = max.max(100);
        self
    }

    /// Add element and detect drift
    pub fn add_element(&mut self, value: f64) -> DriftResult {
        // Add to window
        self.window.push_back(value);
        self.total += value;

        // Limit window size. `pop_front` on a VecDeque is O(1); the previous
        // `Vec::remove(0)` shifted every remaining element down by one on
        // every push once the window was at capacity (O(max_window) per
        // call, forever, in steady state).
        while self.window.len() > self.max_window {
            self.total -= self.window.pop_front().unwrap();
        }

        let n = self.window.len();
        if n < 10 {
            return DriftResult::no_drift(0.0, self.delta);
        }

        // Try different split points. `make_contiguous` needs mutable access
        // but is O(1) here in the common case (front/back-only push/pop keeps
        // a VecDeque's live range contiguous within its ring buffer until it
        // wraps past the allocation boundary).
        let window_slice = self.window.make_contiguous();

        let mut drift_detected = false;
        let mut max_diff = 0.0;
        let mut best_split = 0;

        for split in 1..n {
            let n0 = split as f64;
            let n1 = (n - split) as f64;

            // Mean of each half
            let sum0: f64 = window_slice[..split].iter().sum();
            let sum1: f64 = window_slice[split..].iter().sum();
            let mu0 = sum0 / n0;
            let mu1 = sum1 / n1;

            // Epsilon cut
            let m = 1.0 / (1.0 / n0 + 1.0 / n1);
            let eps_cut = ((2.0 / m * (n as f64 / self.delta).ln()).sqrt()) / 2.0;

            let diff = (mu0 - mu1).abs();
            if diff > eps_cut {
                drift_detected = true;
                if diff > max_diff {
                    max_diff = diff;
                    best_split = split;
                }
            }
        }

        if drift_detected {
            // Remove old data up to split point. `drain` on the front of a
            // VecDeque moves only the retained elements, unlike the previous
            // `self.window[best_split..].to_vec()`, which allocated a brand
            // new buffer and copied the retained tail into it every time.
            self.window.drain(0..best_split);
            self.total = self.window.iter().sum();

            DriftResult::drift(max_diff, 0.0, 2, "Distribution change detected (ADWIN)")
        } else {
            DriftResult::no_drift(0.0, self.delta)
        }
    }

    /// Get current window mean
    pub fn mean(&self) -> f64 {
        if self.window.is_empty() {
            0.0
        } else {
            self.total / self.window.len() as f64
        }
    }

    /// Reset detector
    pub fn reset(&mut self) {
        self.window.clear();
        self.total = 0.0;
        self.variance = 0.0;
    }
}

impl Default for ADWIN {
    fn default() -> Self {
        Self::new(0.002)
    }
}

/// Page-Hinkley test for change detection
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PageHinkley {
    /// Threshold for detection
    threshold: f64,
    /// Minimum number of samples
    min_samples: usize,
    /// Delta parameter (minimum change magnitude to detect)
    delta: f64,
    /// Alpha for EWMA of mean
    alpha: f64,
    /// Cumulative sum
    sum: f64,
    /// Running mean
    mean: f64,
    /// Minimum sum seen
    min_sum: f64,
    /// Number of samples
    n: usize,
}

impl PageHinkley {
    /// Create new Page-Hinkley detector
    pub fn new(threshold: f64) -> Self {
        Self {
            threshold: threshold.max(1.0),
            min_samples: 30,
            delta: 0.005,
            alpha: 0.9999,
            sum: 0.0,
            mean: 0.0,
            min_sum: f64::MAX,
            n: 0,
        }
    }

    /// Set delta parameter
    pub fn with_delta(mut self, delta: f64) -> Self {
        self.delta = delta.max(0.0001);
        self
    }

    /// Add element and check for drift
    pub fn add_element(&mut self, value: f64) -> DriftResult {
        self.n += 1;

        // Update running mean
        if self.n == 1 {
            self.mean = value;
        } else {
            self.mean = self.alpha * self.mean + (1.0 - self.alpha) * value;
        }

        // Update cumulative sum
        self.sum += value - self.mean - self.delta;

        // Update minimum
        if self.sum < self.min_sum {
            self.min_sum = self.sum;
        }

        if self.n < self.min_samples {
            return DriftResult::no_drift(0.0, self.threshold);
        }

        // Page-Hinkley statistic
        let ph = self.sum - self.min_sum;

        if ph > self.threshold {
            DriftResult::drift(ph, self.threshold, 2, "Change detected (Page-Hinkley)")
        } else {
            DriftResult::no_drift(ph, self.threshold)
        }
    }

    /// Reset detector
    pub fn reset(&mut self) {
        self.sum = 0.0;
        self.mean = 0.0;
        self.min_sum = f64::MAX;
        self.n = 0;
    }
}

impl Default for PageHinkley {
    fn default() -> Self {
        Self::new(50.0)
    }
}

/// Combined concept drift detector
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConceptDriftDetector {
    /// DDM detector
    ddm: DDM,
    /// EDDM detector
    eddm: EDDM,
    /// Use ensemble voting
    use_voting: bool,
}

impl ConceptDriftDetector {
    /// Create new combined detector
    pub fn new() -> Self {
        Self {
            ddm: DDM::new(),
            eddm: EDDM::new(),
            use_voting: true,
        }
    }

    /// Add prediction error and detect drift
    pub fn add_element(&mut self, error: bool) -> DriftResult {
        let ddm_result = self.ddm.add_element(error);
        let eddm_result = self.eddm.add_element(error);

        if self.use_voting {
            // Both must agree on drift
            if ddm_result.drift_detected && eddm_result.drift_detected {
                DriftResult::drift(
                    (ddm_result.score + eddm_result.score) / 2.0,
                    0.0,
                    ddm_result.severity.max(eddm_result.severity),
                    "Drift confirmed by both DDM and EDDM",
                )
            } else if ddm_result.drift_detected || eddm_result.drift_detected {
                DriftResult::drift(
                    (ddm_result.score + eddm_result.score) / 2.0,
                    0.0,
                    1, // Warning only
                    "Possible drift (single detector)",
                )
            } else {
                DriftResult::no_drift(
                    (ddm_result.score + eddm_result.score) / 2.0,
                    0.0,
                )
            }
        } else {
            // Use DDM as primary
            ddm_result
        }
    }

    /// Reset both detectors
    pub fn reset(&mut self) {
        self.ddm.reset();
        self.eddm.reset();
    }
}

impl Default for ConceptDriftDetector {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ddm_no_drift() {
        let mut ddm = DDM::new().with_min_samples(20);
        
        // Mixed error rate that stays stable (10% errors)
        for i in 0..100 {
            let error = i % 10 == 0; // 10% error rate
            ddm.add_element(error);
        }
        
        // State should be 0 (normal) or 1 (warning) - not drift
        assert!(ddm.state() < 2, "Should not detect drift with stable error rate");
    }

    #[test]
    fn test_ddm_with_drift() {
        let mut ddm = DDM::new().with_min_samples(20);
        
        // Initial low error rate
        for _ in 0..50 {
            ddm.add_element(false);
        }
        
        // Sudden high error rate
        let mut drift_detected = false;
        for _ in 0..50 {
            let result = ddm.add_element(true);
            if result.drift_detected && result.severity == 2 {
                drift_detected = true;
                break;
            }
        }
        
        assert!(drift_detected || ddm.state() >= 1);
    }

    #[test]
    fn test_adwin() {
        let mut adwin = ADWIN::new(0.01);
        
        // Stable period
        for i in 0..50 {
            adwin.add_element((i % 5) as f64);
        }
        
        // Change point
        let mut drift_detected = false;
        for i in 0..50 {
            let result = adwin.add_element(100.0 + (i % 5) as f64);
            if result.drift_detected {
                drift_detected = true;
                break;
            }
        }
        
        assert!(drift_detected);
    }

    #[test]
    fn test_page_hinkley() {
        let mut ph = PageHinkley::new(10.0);
        
        // Stable period
        for i in 0..30 {
            ph.add_element((i % 3) as f64);
        }
        
        // Change - should eventually detect
        let mut drift_detected = false;
        for _ in 0..100 {
            let result = ph.add_element(50.0);
            if result.drift_detected {
                drift_detected = true;
                break;
            }
        }
        
        assert!(drift_detected);
    }
}
