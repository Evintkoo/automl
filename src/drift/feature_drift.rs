//! Feature drift monitoring

use crate::error::{AutoMLError, Result};
use crate::drift::DriftDetector;
use crate::drift::data_drift::{KolmogorovSmirnovTest, PopulationStabilityIndex, JensenShannonDivergence};
use ndarray::{Array1, Array2};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Result for a single feature's drift analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeatureDriftResult {
    /// Feature name/index
    pub feature_name: String,
    /// Whether drift was detected
    pub drift_detected: bool,
    /// Drift scores from different methods
    pub scores: HashMap<String, f64>,
    /// Overall severity
    pub severity: u8,
    /// Reference statistics
    pub ref_stats: FeatureStats,
    /// Test statistics
    pub test_stats: FeatureStats,
}

/// Basic statistics for a feature
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeatureStats {
    pub mean: f64,
    pub std: f64,
    pub min: f64,
    pub max: f64,
    pub median: f64,
    pub q25: f64,
    pub q75: f64,
    pub n_unique: usize,
    pub missing_rate: f64,
}

impl FeatureStats {
    /// Compute statistics from data
    pub fn from_data(data: &Array1<f64>) -> Self {
        let n = data.len() as f64;
        let mut sorted: Vec<f64> = data.iter().copied().collect();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let mean = data.sum() / n;
        let variance = data.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / n;
        let std = variance.sqrt();

        let min = sorted.first().copied().unwrap_or(0.0);
        let max = sorted.last().copied().unwrap_or(0.0);
        let median = if sorted.len() % 2 == 0 && sorted.len() >= 2 {
            (sorted[sorted.len() / 2 - 1] + sorted[sorted.len() / 2]) / 2.0
        } else {
            sorted[sorted.len() / 2]
        };
        let q25 = sorted[sorted.len() / 4];
        let q75 = sorted[(3 * sorted.len()) / 4];

        // Count unique values (approximately)
        let mut unique: Vec<f64> = sorted.clone();
        unique.dedup();
        let n_unique = unique.len();

        // Count NaN/missing
        let n_missing = data.iter().filter(|x| x.is_nan()).count();
        let missing_rate = n_missing as f64 / n;

        Self {
            mean,
            std,
            min,
            max,
            median,
            q25,
            q75,
            n_unique,
            missing_rate,
        }
    }
}

/// Complete drift report for all features
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DriftReport {
    /// Feature-level results
    pub features: Vec<FeatureDriftResult>,
    /// Number of features with drift
    pub n_drifted: usize,
    /// Overall drift detected
    pub overall_drift: bool,
    /// Timestamp
    pub timestamp: String,
    /// Reference data size
    pub ref_size: usize,
    /// Test data size
    pub test_size: usize,
}

impl DriftReport {
    /// Get drifted feature names
    pub fn drifted_features(&self) -> Vec<&str> {
        self.features
            .iter()
            .filter(|f| f.drift_detected)
            .map(|f| f.feature_name.as_str())
            .collect()
    }

    /// Get critical features (severity == 2)
    pub fn critical_features(&self) -> Vec<&str> {
        self.features
            .iter()
            .filter(|f| f.severity == 2)
            .map(|f| f.feature_name.as_str())
            .collect()
    }

    /// Generate summary string
    pub fn summary(&self) -> String {
        let mut s = String::new();
        s.push_str(&format!("Drift Report\n"));
        s.push_str(&format!("============\n"));
        s.push_str(&format!("Reference size: {}\n", self.ref_size));
        s.push_str(&format!("Test size: {}\n", self.test_size));
        s.push_str(&format!("Total features: {}\n", self.features.len()));
        s.push_str(&format!("Drifted features: {}\n", self.n_drifted));
        s.push_str(&format!("Overall drift: {}\n\n", self.overall_drift));

        if self.n_drifted > 0 {
            s.push_str("Drifted Features:\n");
            for f in self.features.iter().filter(|f| f.drift_detected) {
                let severity_str = match f.severity {
                    2 => "CRITICAL",
                    1 => "WARNING",
                    _ => "INFO",
                };
                s.push_str(&format!("  - {} [{}]\n", f.feature_name, severity_str));
            }
        }

        s
    }
}

/// Feature drift monitor for production ML systems
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeatureDriftMonitor {
    /// Feature names
    feature_names: Vec<String>,
    /// Reference data statistics
    ref_stats: Option<Vec<FeatureStats>>,
    /// Reference data (stored for comparison)
    ref_data: Option<Array2<f64>>,
    /// Drift threshold
    drift_threshold: f64,
    /// Use multiple methods
    use_ensemble: bool,
}

impl FeatureDriftMonitor {
    /// Create new monitor
    pub fn new(feature_names: Vec<String>) -> Self {
        Self {
            feature_names,
            ref_stats: None,
            ref_data: None,
            drift_threshold: 0.1,
            use_ensemble: true,
        }
    }

    /// Create with default feature names
    pub fn with_n_features(n: usize) -> Self {
        let names: Vec<String> = (0..n).map(|i| format!("feature_{}", i)).collect();
        Self::new(names)
    }

    /// Set drift threshold
    pub fn with_threshold(mut self, threshold: f64) -> Self {
        self.drift_threshold = threshold.clamp(0.01, 0.5);
        self
    }

    /// Fit on reference data
    pub fn fit(&mut self, x_ref: &Array2<f64>) -> Result<()> {
        if x_ref.ncols() != self.feature_names.len() {
            return Err(AutoMLError::ValidationError(format!(
                "Expected {} features, got {}",
                self.feature_names.len(),
                x_ref.ncols()
            )));
        }

        // Compute statistics for each feature
        let mut stats = Vec::with_capacity(x_ref.ncols());
        for col_idx in 0..x_ref.ncols() {
            let col = x_ref.column(col_idx).to_owned();
            stats.push(FeatureStats::from_data(&col));
        }

        self.ref_stats = Some(stats);
        self.ref_data = Some(x_ref.clone());

        Ok(())
    }

    /// Detect drift between reference and test data
    pub fn detect(&self, x_test: &Array2<f64>) -> Result<DriftReport> {
        let ref_data = self.ref_data.as_ref().ok_or_else(|| {
            AutoMLError::ValidationError("Monitor not fitted".to_string())
        })?;
        let ref_stats = self.ref_stats.as_ref().unwrap();

        if x_test.ncols() != self.feature_names.len() {
            return Err(AutoMLError::ValidationError(format!(
                "Expected {} features, got {}",
                self.feature_names.len(),
                x_test.ncols()
            )));
        }

        let mut features = Vec::with_capacity(self.feature_names.len());
        let mut n_drifted = 0;

        // Drift detectors
        let ks_test = KolmogorovSmirnovTest::new(0.05);
        let psi = PopulationStabilityIndex::new(10);
        let js = JensenShannonDivergence::new(self.drift_threshold);

        for (col_idx, name) in self.feature_names.iter().enumerate() {
            let ref_col = ref_data.column(col_idx).to_owned();
            let test_col = x_test.column(col_idx).to_owned();

            // Compute test statistics
            let test_stats = FeatureStats::from_data(&test_col);

            // Run drift tests
            let mut scores = HashMap::new();
            let mut drift_votes = 0;
            let mut max_severity = 0u8;

            // KS test
            if let Ok(result) = ks_test.detect(&ref_col, &test_col) {
                scores.insert("ks".to_string(), result.score);
                if result.drift_detected {
                    drift_votes += 1;
                    max_severity = max_severity.max(result.severity);
                }
            }

            // PSI
            if let Ok(result) = psi.detect(&ref_col, &test_col) {
                scores.insert("psi".to_string(), result.score);
                if result.drift_detected {
                    drift_votes += 1;
                    max_severity = max_severity.max(result.severity);
                }
            }

            // JS divergence
            if let Ok(result) = js.detect(&ref_col, &test_col) {
                scores.insert("js".to_string(), result.score);
                if result.drift_detected {
                    drift_votes += 1;
                    max_severity = max_severity.max(result.severity);
                }
            }

            // Determine if drift detected (majority voting if ensemble)
            let drift_detected = if self.use_ensemble {
                drift_votes >= 2
            } else {
                drift_votes >= 1
            };

            if drift_detected {
                n_drifted += 1;
            }

            features.push(FeatureDriftResult {
                feature_name: name.clone(),
                drift_detected,
                scores,
                severity: if drift_detected { max_severity } else { 0 },
                ref_stats: ref_stats[col_idx].clone(),
                test_stats,
            });
        }

        Ok(DriftReport {
            features,
            n_drifted,
            overall_drift: n_drifted > 0,
            timestamp: "".to_string(), // Timestamp should be set by caller
            ref_size: ref_data.nrows(),
            test_size: x_test.nrows(),
        })
    }

    /// Quick check - returns true if any drift detected
    pub fn has_drift(&self, x_test: &Array2<f64>) -> Result<bool> {
        let report = self.detect(x_test)?;
        Ok(report.overall_drift)
    }

    /// Get feature names that have drifted
    pub fn get_drifted_features(&self, x_test: &Array2<f64>) -> Result<Vec<String>> {
        let report = self.detect(x_test)?;
        Ok(report.drifted_features().iter().map(|s| s.to_string()).collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_feature_stats() {
        let data = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]);
        let stats = FeatureStats::from_data(&data);

        assert!((stats.mean - 5.5).abs() < 0.01);
        assert_eq!(stats.min, 1.0);
        assert_eq!(stats.max, 10.0);
    }

    #[test]
    fn test_drift_monitor_no_drift() {
        let names = vec!["f1".to_string(), "f2".to_string()];
        let mut monitor = FeatureDriftMonitor::new(names);

        // Reference data: 50 rows, 2 columns
        let ref_data: Vec<f64> = (0..50)
            .flat_map(|i| vec![(i % 10) as f64, ((i + 3) % 10) as f64])
            .collect();
        let x_ref = Array2::from_shape_vec((50, 2), ref_data).unwrap();

        // Similar test data
        let test_data: Vec<f64> = (0..50)
            .flat_map(|i| vec![((i + 1) % 10) as f64, ((i + 4) % 10) as f64])
            .collect();
        let x_test = Array2::from_shape_vec((50, 2), test_data).unwrap();

        monitor.fit(&x_ref).unwrap();
        let report = monitor.detect(&x_test).unwrap();

        // Similar distributions should not trigger drift
        assert!(report.n_drifted <= 1); // May detect minor drift due to small sample
    }

    #[test]
    fn test_drift_monitor_with_drift() {
        let names = vec!["f1".to_string(), "f2".to_string()];
        let mut monitor = FeatureDriftMonitor::new(names);

        // Reference data - values around 0-10
        let ref_data: Vec<f64> = (0..50)
            .flat_map(|i| vec![(i % 10) as f64, ((i + 3) % 10) as f64])
            .collect();
        let x_ref = Array2::from_shape_vec((50, 2), ref_data).unwrap();

        // Completely different test data - values around 100-110
        let test_data: Vec<f64> = (0..50)
            .flat_map(|i| vec![100.0 + (i % 10) as f64, 100.0 + ((i + 3) % 10) as f64])
            .collect();
        let x_test = Array2::from_shape_vec((50, 2), test_data).unwrap();

        monitor.fit(&x_ref).unwrap();
        let report = monitor.detect(&x_test).unwrap();

        assert!(report.overall_drift);
        assert!(report.n_drifted >= 1);
    }

    #[test]
    fn test_drift_report_summary() {
        let report = DriftReport {
            features: vec![
                FeatureDriftResult {
                    feature_name: "feature_0".to_string(),
                    drift_detected: true,
                    scores: HashMap::new(),
                    severity: 2,
                    ref_stats: FeatureStats::from_data(&Array1::from_vec(vec![1.0, 2.0, 3.0])),
                    test_stats: FeatureStats::from_data(&Array1::from_vec(vec![10.0, 20.0, 30.0])),
                },
            ],
            n_drifted: 1,
            overall_drift: true,
            timestamp: "2024-01-01".to_string(),
            ref_size: 100,
            test_size: 50,
        };

        let summary = report.summary();
        assert!(summary.contains("Drifted features: 1"));
        assert!(summary.contains("feature_0"));
        assert!(summary.contains("CRITICAL"));
    }
}
