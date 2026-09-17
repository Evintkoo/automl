//! Quantization Calibration
//!
//! Methods for calibrating quantization parameters.

use super::{QuantizationType, QuantizationMode};

/// Calibration method for determining quantization parameters
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CalibrationMethod {
    /// Use min/max of calibration data
    MinMax,
    /// Use histogram-based calibration
    Histogram,
    /// Use entropy-based calibration (KL divergence)
    Entropy,
    /// Use percentile-based calibration
    Percentile,
    /// Use moving average for streaming data
    MovingAverage,
}

impl Default for CalibrationMethod {
    fn default() -> Self {
        CalibrationMethod::MinMax
    }
}

/// Calibrator for quantization
pub struct QuantizationCalibrator {
    method: CalibrationMethod,
    num_bins: usize,
    percentile: f64,
    moving_average_alpha: f64,
    
    // Collected statistics
    min_val: f64,
    max_val: f64,
    histogram: Vec<u64>,
    histogram_min: f64,
    histogram_max: f64,
    sample_count: u64,
    
    // Moving average state
    ema_min: f64,
    ema_max: f64,
}

impl QuantizationCalibrator {
    /// Create a new calibrator with the given method
    pub fn new(method: CalibrationMethod) -> Self {
        Self {
            method,
            num_bins: 2048,
            percentile: 99.99,
            moving_average_alpha: 0.1,
            min_val: f64::INFINITY,
            max_val: f64::NEG_INFINITY,
            histogram: vec![0; 2048],
            histogram_min: 0.0,
            histogram_max: 1.0,
            sample_count: 0,
            ema_min: 0.0,
            ema_max: 0.0,
        }
    }
    
    /// Create with MinMax method
    pub fn min_max() -> Self {
        Self::new(CalibrationMethod::MinMax)
    }
    
    /// Create with Histogram method
    pub fn histogram(num_bins: usize) -> Self {
        let mut cal = Self::new(CalibrationMethod::Histogram);
        cal.num_bins = num_bins;
        cal.histogram = vec![0; num_bins];
        cal
    }
    
    /// Create with Percentile method
    pub fn percentile(percentile: f64) -> Self {
        let mut cal = Self::new(CalibrationMethod::Percentile);
        cal.percentile = percentile;
        cal
    }
    
    /// Collect calibration data
    pub fn collect(&mut self, data: &[f64]) {
        let had_samples = self.sample_count > 0;
        let mut batch_min = f64::INFINITY;
        let mut batch_max = f64::NEG_INFINITY;
        let mut batch_has_finite = false;

        for &value in data {
            if value.is_finite() {
                self.min_val = self.min_val.min(value);
                self.max_val = self.max_val.max(value);
                self.sample_count += 1;
                batch_min = batch_min.min(value);
                batch_max = batch_max.max(value);
                batch_has_finite = true;
            }
        }

        // Update moving average. The EMA update is applied once per batch, using
        // the batch's own min/max as the new observation - this is the standard
        // "moving average min/max observer" pattern (as used e.g. by TensorRT /
        // PyTorch calibrators) and, unlike blending `value.min(ema_min)` into
        // `ema_min`, it lets the tracked bound move in *either* direction as the
        // underlying data distribution drifts (relaxing back inward when the
        // distribution narrows), instead of being a one-way ratchet that can only
        // ever expand outward.
        if batch_has_finite {
            if !had_samples {
                self.ema_min = batch_min;
                self.ema_max = batch_max;
            } else {
                self.ema_min = self.moving_average_alpha * batch_min
                    + (1.0 - self.moving_average_alpha) * self.ema_min;
                self.ema_max = self.moving_average_alpha * batch_max
                    + (1.0 - self.moving_average_alpha) * self.ema_max;
            }
        }

        // Update histogram if needed
        if matches!(self.method, CalibrationMethod::Histogram | CalibrationMethod::Entropy) {
            self.update_histogram(data);
        }
    }
    
    fn update_histogram(&mut self, data: &[f64]) {
        if self.min_val >= self.max_val {
            return;
        }
        
        // Resize histogram range if needed
        if self.sample_count == data.len() as u64 {
            // First batch, set histogram range
            self.histogram_min = self.min_val;
            self.histogram_max = self.max_val;
        }
        
        let range = self.histogram_max - self.histogram_min;
        if range <= 0.0 {
            return;
        }
        
        for &value in data {
            if value.is_finite() && value >= self.histogram_min && value <= self.histogram_max {
                let bin = ((value - self.histogram_min) / range * (self.num_bins - 1) as f64) as usize;
                let bin = bin.min(self.num_bins - 1);
                self.histogram[bin] += 1;
            }
        }
    }
    
    /// Compute the calibrated range
    pub fn compute_range(&self) -> (f64, f64) {
        match self.method {
            CalibrationMethod::MinMax => {
                (self.min_val, self.max_val)
            }
            CalibrationMethod::Histogram => {
                self.compute_histogram_range()
            }
            CalibrationMethod::Entropy => {
                self.compute_entropy_range()
            }
            CalibrationMethod::Percentile => {
                self.compute_percentile_range()
            }
            CalibrationMethod::MovingAverage => {
                (self.ema_min, self.ema_max)
            }
        }
    }
    
    fn compute_histogram_range(&self) -> (f64, f64) {
        // Find the range that contains most of the data
        let total: u64 = self.histogram.iter().sum();
        if total == 0 {
            return (self.min_val, self.max_val);
        }
        
        // Split the excluded mass evenly across both tails: excluding
        // `tail_fraction` from the low tail AND `tail_fraction` from the high
        // tail totals `(100 - percentile)/100` excluded, matching the intended
        // "keep the central `percentile`%" semantics (mirrors
        // `quantizer.rs::compute_range`'s `tail_fraction` calculation).
        let tail_fraction = (100.0 - self.percentile) / 200.0;
        let threshold = (total as f64 * tail_fraction) as u64;
        
        // Find min bin
        let mut cumsum = 0u64;
        let mut min_bin = 0;
        for (i, &count) in self.histogram.iter().enumerate() {
            cumsum += count;
            if cumsum >= threshold {
                min_bin = i;
                break;
            }
        }
        
        // Find max bin
        cumsum = 0;
        let mut max_bin = self.num_bins - 1;
        for (i, &count) in self.histogram.iter().enumerate().rev() {
            cumsum += count;
            if cumsum >= threshold {
                max_bin = i;
                break;
            }
        }
        
        let range = self.histogram_max - self.histogram_min;
        let bin_width = range / self.num_bins as f64;
        
        let min_val = self.histogram_min + min_bin as f64 * bin_width;
        let max_val = self.histogram_min + (max_bin + 1) as f64 * bin_width;
        
        (min_val, max_val)
    }
    
    fn compute_entropy_range(&self) -> (f64, f64) {
        // Simplified entropy-based range finding
        // Full implementation would use KL divergence
        self.compute_histogram_range()
    }
    
    fn compute_percentile_range(&self) -> (f64, f64) {
        // Use histogram to estimate percentiles
        self.compute_histogram_range()
    }
    
    /// Compute scale and zero point for given quantization parameters
    pub fn compute_parameters(
        &self,
        quantization_type: QuantizationType,
        quantization_mode: QuantizationMode,
    ) -> (f64, f64) {
        let (min_val, max_val) = self.compute_range();
        
        let (qmin, qmax) = match quantization_type {
            QuantizationType::Int8 => (-128i32, 127i32),
            QuantizationType::UInt8 => (0i32, 255i32),
            QuantizationType::Int16 => (-32768i32, 32767i32),
            QuantizationType::Float16 => (-65504i32, 65504i32),
        };
        
        match quantization_mode {
            QuantizationMode::Symmetric => {
                let abs_max = min_val.abs().max(max_val.abs());
                let scale = if abs_max == 0.0 { 1.0 } else { abs_max / qmax as f64 };
                (scale, 0.0)
            }
            QuantizationMode::Asymmetric => {
                let scale = if max_val == min_val {
                    1.0
                } else {
                    (max_val - min_val) / (qmax - qmin) as f64
                };
                let zero_point = qmin as f64 - min_val / scale;
                (scale, zero_point)
            }
            _ => {
                let abs_max = min_val.abs().max(max_val.abs());
                let scale = if abs_max == 0.0 { 1.0 } else { abs_max / qmax as f64 };
                (scale, 0.0)
            }
        }
    }
    
    /// Reset the calibrator
    pub fn reset(&mut self) {
        self.min_val = f64::INFINITY;
        self.max_val = f64::NEG_INFINITY;
        self.histogram.fill(0);
        self.sample_count = 0;
        self.ema_min = 0.0;
        self.ema_max = 0.0;
    }
    
    /// Get the number of samples collected
    pub fn sample_count(&self) -> u64 {
        self.sample_count
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_minmax_calibration() {
        let mut cal = QuantizationCalibrator::min_max();
        
        cal.collect(&[-10.0, -5.0, 0.0, 5.0, 10.0]);
        
        let (min, max) = cal.compute_range();
        assert_eq!(min, -10.0);
        assert_eq!(max, 10.0);
    }
    
    #[test]
    fn test_symmetric_parameters() {
        let mut cal = QuantizationCalibrator::min_max();
        cal.collect(&[-10.0, 10.0]);
        
        let (scale, zero_point) = cal.compute_parameters(
            QuantizationType::Int8,
            QuantizationMode::Symmetric,
        );
        
        assert!((zero_point - 0.0).abs() < 1e-10);
        assert!((scale - 10.0 / 127.0).abs() < 1e-10);
    }
    
    #[test]
    fn test_asymmetric_parameters() {
        let mut cal = QuantizationCalibrator::min_max();
        cal.collect(&[0.0, 10.0]);
        
        let (scale, zero_point) = cal.compute_parameters(
            QuantizationType::UInt8,
            QuantizationMode::Asymmetric,
        );
        
        assert!(scale > 0.0);
        assert!(zero_point >= 0.0);
    }
}
