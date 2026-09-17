//! Quantizer Implementation
//!
//! High-performance quantization with multiple types and modes.

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::RwLock;

use crate::error::{AutoMLError, Result};

/// Quantization data type
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum QuantizationType {
    /// 8-bit signed integer
    Int8,
    /// 8-bit unsigned integer
    UInt8,
    /// 16-bit signed integer
    Int16,
    /// 16-bit floating point
    Float16,
}

impl Default for QuantizationType {
    fn default() -> Self {
        QuantizationType::Int8
    }
}

/// Quantization mode
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QuantizationMode {
    /// Symmetric quantization (zero point = 0)
    Symmetric,
    /// Asymmetric quantization
    Asymmetric,
    /// Per-channel quantization
    PerChannel,
    /// Dynamic quantization
    Dynamic,
}

impl Default for QuantizationMode {
    fn default() -> Self {
        QuantizationMode::Symmetric
    }
}

/// Configuration for quantization
#[derive(Debug, Clone)]
pub struct QuantizationConfig {
    /// Type of quantization
    pub quantization_type: QuantizationType,
    /// Quantization mode
    pub quantization_mode: QuantizationMode,
    /// Number of bits for quantization
    pub num_bits: u8,
    /// Enable caching of quantized values
    pub enable_cache: bool,
    /// Cache size
    pub cache_size: usize,
    /// Symmetric range (use max abs value)
    pub symmetric_range: bool,
    /// Clipping percentile (for outlier handling)
    pub clip_percentile: Option<f64>,
}

impl Default for QuantizationConfig {
    fn default() -> Self {
        Self {
            quantization_type: QuantizationType::Int8,
            quantization_mode: QuantizationMode::Symmetric,
            num_bits: 8,
            enable_cache: true,
            cache_size: 1000,
            symmetric_range: true,
            clip_percentile: None,
        }
    }
}

impl QuantizationConfig {
    /// Validate the configuration, returning a clear error for out-of-range values
    /// instead of letting them reach unchecked arithmetic (e.g. `num_bits - 1`, which
    /// panics with "attempt to subtract with overflow" if `num_bits == 0`).
    pub fn validate(&self) -> Result<()> {
        if self.num_bits == 0 || self.num_bits > 16 {
            return Err(AutoMLError::ConfigError(format!(
                "QuantizationConfig.num_bits must be in 1..=16, got {}",
                self.num_bits
            )));
        }
        if let Some(p) = self.clip_percentile {
            if !(0.0..=100.0).contains(&p) {
                return Err(AutoMLError::ConfigError(format!(
                    "QuantizationConfig.clip_percentile must be in [0.0, 100.0], got {}",
                    p
                )));
            }
        }
        Ok(())
    }
}

/// Quantized data container
#[derive(Debug, Clone)]
pub struct QuantizedData {
    /// Quantized values (stored as i32 for flexibility)
    pub data: Vec<i32>,
    /// Scale factor
    pub scale: f64,
    /// Zero point
    pub zero_point: f64,
    /// Original shape
    pub shape: Vec<usize>,
    /// Quantization type used
    pub quantization_type: QuantizationType,
}

impl QuantizedData {
    /// Dequantize back to f64
    pub fn dequantize(&self) -> Vec<f64> {
        self.data.iter()
            .map(|&q| (q as f64 - self.zero_point) * self.scale)
            .collect()
    }
    
    /// Get the size in bytes
    pub fn size_bytes(&self) -> usize {
        match self.quantization_type {
            QuantizationType::Int8 | QuantizationType::UInt8 => self.data.len(),
            QuantizationType::Int16 | QuantizationType::Float16 => self.data.len() * 2,
        }
    }
    
    /// Get compression ratio compared to f64
    pub fn compression_ratio(&self) -> f64 {
        let original_bytes = self.data.len() * 8; // f64 = 8 bytes
        let quantized_bytes = self.size_bytes();
        original_bytes as f64 / quantized_bytes as f64
    }
}

/// High-performance quantizer
pub struct Quantizer {
    config: QuantizationConfig,
    
    // Quantization parameters
    global_scale: f64,
    global_zero_point: f64,
    per_channel_scales: Vec<f64>,
    per_channel_zero_points: Vec<f64>,
    
    // Range values
    qmin: i32,
    qmax: i32,
    
    // Calibration state
    is_calibrated: bool,
    calibration_data: Vec<f64>,
    
    // Statistics
    quantize_calls: AtomicU64,
    dequantize_calls: AtomicU64,
    clipped_values: AtomicU64,
    total_values: AtomicU64,
    
    // Cache
    #[allow(dead_code)]
    cache: Option<RwLock<HashMap<u64, i32>>>,
}

impl Quantizer {
    /// Create a new quantizer with the given configuration
    pub fn new(config: QuantizationConfig) -> Self {
        let (qmin, qmax) = Self::get_quantization_range(&config);
        
        let cache = if config.enable_cache {
            Some(RwLock::new(HashMap::with_capacity(config.cache_size)))
        } else {
            None
        };
        
        Self {
            config,
            global_scale: 1.0,
            global_zero_point: 0.0,
            per_channel_scales: Vec::new(),
            per_channel_zero_points: Vec::new(),
            qmin,
            qmax,
            is_calibrated: false,
            calibration_data: Vec::new(),
            quantize_calls: AtomicU64::new(0),
            dequantize_calls: AtomicU64::new(0),
            clipped_values: AtomicU64::new(0),
            total_values: AtomicU64::new(0),
            cache,
        }
    }
    
    /// Get quantization range based on config
    fn get_quantization_range(config: &QuantizationConfig) -> (i32, i32) {
        // Clamp to a safe range to avoid "attempt to subtract with overflow" panics
        // below when `num_bits` is 0, and to keep the bit-shifts well within i32 range.
        // `QuantizationConfig.num_bits` has no validation before reaching this point
        // (it can come from an unchecked config), so this is a defensive floor/ceiling.
        let num_bits = config.num_bits.clamp(1, 16);

        match config.quantization_type {
            QuantizationType::Int8 => {
                let half = 1i32 << (num_bits - 1);
                (-half, half - 1)
            }
            QuantizationType::UInt8 => {
                (0, (1i32 << num_bits) - 1)
            }
            QuantizationType::Int16 => {
                let half = 1i32 << (num_bits.min(15) - 1);
                (-half, half - 1)
            }
            QuantizationType::Float16 => {
                (-65504, 65504)
            }
        }
    }
    
    /// Calibrate the quantizer with sample data
    pub fn calibrate(&mut self, data: &[f64]) {
        self.calibration_data.extend_from_slice(data);
        
        if self.calibration_data.is_empty() {
            return;
        }
        
        let (min_val, max_val) = self.compute_range(&self.calibration_data);
        
        // Compute scale and zero point
        match self.config.quantization_mode {
            QuantizationMode::Symmetric => {
                let abs_max = min_val.abs().max(max_val.abs());
                self.global_scale = abs_max / (self.qmax as f64);
                self.global_zero_point = 0.0;
            }
            QuantizationMode::Asymmetric => {
                self.global_scale = (max_val - min_val) / ((self.qmax - self.qmin) as f64);
                self.global_zero_point = self.qmin as f64 - min_val / self.global_scale;
            }
            _ => {
                // Per-channel and dynamic modes use different calibration
                let abs_max = min_val.abs().max(max_val.abs());
                self.global_scale = abs_max / (self.qmax as f64);
                self.global_zero_point = 0.0;
            }
        }
        
        self.is_calibrated = true;
    }
    
    /// Compute min/max range, optionally with clipping
    fn compute_range(&self, data: &[f64]) -> (f64, f64) {
        if data.is_empty() {
            return (0.0, 1.0);
        }
        
        if let Some(percentile) = self.config.clip_percentile {
            // Use percentile-based clipping. `percentile` is the amount of the
            // *central* mass to keep (e.g. 99 => keep the central 99%, clipping
            // (100-99)/2 = 0.5% off each tail). Clamp defensively in case an
            // unvalidated config carries a value outside [0, 100].
            let percentile = percentile.clamp(0.0, 100.0);
            let mut sorted: Vec<f64> = data.iter().copied().collect();
            sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

            let max_idx = sorted.len().saturating_sub(1);
            let tail_fraction = (100.0 - percentile) / 200.0;
            let low_idx = ((tail_fraction * max_idx as f64).round() as usize).min(max_idx);
            let high_idx = (((1.0 - tail_fraction) * max_idx as f64).round() as usize).min(max_idx);

            let (mut min_val, mut max_val) = (sorted[low_idx], sorted[high_idx]);
            if min_val > max_val {
                std::mem::swap(&mut min_val, &mut max_val);
            }
            debug_assert!(min_val <= max_val, "compute_range: min_val must be <= max_val");

            (min_val, max_val)
        } else {
            let min_val = data.iter().copied().fold(f64::INFINITY, f64::min);
            let max_val = data.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            (min_val, max_val)
        }
    }
    
    /// Quantize a single value
    pub fn quantize_value(&self, value: f64) -> i32 {
        let scale = if self.global_scale == 0.0 { 1.0 } else { self.global_scale };
        let quantized = (value / scale + self.global_zero_point).round() as i32;
        quantized.clamp(self.qmin, self.qmax)
    }
    
    /// Dequantize a single value
    pub fn dequantize_value(&self, quantized: i32) -> f64 {
        (quantized as f64 - self.global_zero_point) * self.global_scale
    }
    
    /// Quantize an array of values
    pub fn quantize(&self, data: &[f64]) -> QuantizedData {
        self.quantize_calls.fetch_add(1, Ordering::Relaxed);
        self.total_values.fetch_add(data.len() as u64, Ordering::Relaxed);
        
        // Use calibrated parameters if available, otherwise compute on-the-fly
        let (scale, zero_point) = if self.is_calibrated {
            (self.global_scale, self.global_zero_point)
        } else {
            // Dynamic quantization
            let (min_val, max_val) = self.compute_range(data);
            let abs_max = min_val.abs().max(max_val.abs());
            let scale = if abs_max == 0.0 { 1.0 } else { abs_max / (self.qmax as f64) };
            (scale, 0.0)
        };
        
        let scale = if scale == 0.0 { 1.0 } else { scale };
        
        let mut clipped = 0u64;
        let quantized: Vec<i32> = data.iter().map(|&v| {
            let q = (v / scale + zero_point).round() as i32;
            let clamped = q.clamp(self.qmin, self.qmax);
            if q != clamped {
                clipped += 1;
            }
            clamped
        }).collect();
        
        self.clipped_values.fetch_add(clipped, Ordering::Relaxed);
        
        QuantizedData {
            data: quantized,
            scale,
            zero_point,
            shape: vec![data.len()],
            quantization_type: self.config.quantization_type,
        }
    }
    
    /// Quantize with shape information
    pub fn quantize_with_shape(&self, data: &[f64], shape: Vec<usize>) -> QuantizedData {
        let mut result = self.quantize(data);
        result.shape = shape;
        result
    }
    
    /// Dequantize data back to f64
    pub fn dequantize(&self, quantized: &QuantizedData) -> Vec<f64> {
        self.dequantize_calls.fetch_add(1, Ordering::Relaxed);
        quantized.dequantize()
    }
    
    /// Get quantization statistics
    pub fn stats(&self) -> QuantizationStats {
        let total = self.total_values.load(Ordering::Relaxed);
        let clipped = self.clipped_values.load(Ordering::Relaxed);
        
        QuantizationStats {
            quantize_calls: self.quantize_calls.load(Ordering::Relaxed),
            dequantize_calls: self.dequantize_calls.load(Ordering::Relaxed),
            total_values: total,
            clipped_values: clipped,
            clip_rate: if total > 0 { clipped as f64 / total as f64 } else { 0.0 },
            scale: self.global_scale,
            zero_point: self.global_zero_point,
            is_calibrated: self.is_calibrated,
        }
    }
    
    /// Reset calibration
    pub fn reset(&mut self) {
        self.is_calibrated = false;
        self.calibration_data.clear();
        self.global_scale = 1.0;
        self.global_zero_point = 0.0;
        self.per_channel_scales.clear();
        self.per_channel_zero_points.clear();
    }
    
    /// Get the quantization config
    pub fn config(&self) -> &QuantizationConfig {
        &self.config
    }
    
    /// Check if calibrated
    pub fn is_calibrated(&self) -> bool {
        self.is_calibrated
    }
}

/// Quantization statistics
#[derive(Debug, Clone)]
pub struct QuantizationStats {
    /// Number of quantize calls
    pub quantize_calls: u64,
    /// Number of dequantize calls
    pub dequantize_calls: u64,
    /// Total values quantized
    pub total_values: u64,
    /// Number of clipped values
    pub clipped_values: u64,
    /// Clip rate
    pub clip_rate: f64,
    /// Current scale
    pub scale: f64,
    /// Current zero point
    pub zero_point: f64,
    /// Whether calibrated
    pub is_calibrated: bool,
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_quantizer_basic() {
        let config = QuantizationConfig::default();
        let quantizer = Quantizer::new(config);
        
        let data = vec![0.0, 0.5, 1.0, -0.5, -1.0];
        let quantized = quantizer.quantize(&data);
        let dequantized = quantizer.dequantize(&quantized);
        
        // Should be close to original (with some quantization error)
        for (orig, deq) in data.iter().zip(dequantized.iter()) {
            assert!((orig - deq).abs() < 0.1);
        }
    }
    
    #[test]
    fn test_quantizer_calibration() {
        let config = QuantizationConfig::default();
        let mut quantizer = Quantizer::new(config);
        
        // Calibrate with sample data
        let calibration_data: Vec<f64> = (-100..=100).map(|x| x as f64 / 10.0).collect();
        quantizer.calibrate(&calibration_data);
        
        assert!(quantizer.is_calibrated());
        
        let data = vec![0.0, 5.0, 10.0, -5.0, -10.0];
        let quantized = quantizer.quantize(&data);
        let dequantized = quantizer.dequantize(&quantized);
        
        for (orig, deq) in data.iter().zip(dequantized.iter()) {
            assert!((orig - deq).abs() < 0.2);
        }
    }
    
    #[test]
    fn test_quantized_data_compression() {
        let config = QuantizationConfig::default();
        let quantizer = Quantizer::new(config);
        
        let data = vec![1.0; 1000];
        let quantized = quantizer.quantize(&data);
        
        // INT8 should give 8x compression ratio
        assert!(quantized.compression_ratio() >= 7.0);
    }
    
    #[test]
    fn test_uint8_quantization() {
        let config = QuantizationConfig {
            quantization_type: QuantizationType::UInt8,
            quantization_mode: QuantizationMode::Asymmetric,
            ..Default::default()
        };
        
        let mut quantizer = Quantizer::new(config);
        quantizer.calibrate(&[0.0, 1.0]);
        
        let data = vec![0.0, 0.25, 0.5, 0.75, 1.0];
        let quantized = quantizer.quantize(&data);
        
        // All values should be non-negative for UINT8
        assert!(quantized.data.iter().all(|&v| v >= 0));
    }
}
