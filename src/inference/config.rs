//! Inference configuration

use serde::{Deserialize, Serialize};

use crate::error::{AutoMLError, Result};

/// Quantization type for model optimization
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum QuantizationType {
    /// No quantization
    None,
    /// 8-bit integer quantization
    Int8,
    /// 16-bit float quantization
    Float16,
    /// Dynamic quantization
    Dynamic,
}

/// Configuration for model inference
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InferenceConfig {
    /// Batch size for inference
    pub batch_size: usize,
    
    /// Number of parallel workers
    pub n_workers: Option<usize>,
    
    /// Quantization type
    pub quantization: QuantizationType,
    
    /// Whether to use GPU acceleration (future)
    pub use_gpu: bool,
    
    /// GPU device ID (if using GPU)
    pub device_id: usize,
    
    /// Maximum memory usage in bytes
    pub max_memory_bytes: Option<usize>,
    
    /// Whether to enable streaming mode for large datasets
    pub streaming: bool,
    
    /// Chunk size for streaming mode
    pub stream_chunk_size: usize,
    
    /// Whether to cache preprocessed data and prediction results
    pub cache_preprocessing: bool,

    /// Maximum number of entries in the prediction cache
    pub prediction_cache_size: usize,

    /// Time-to-live in seconds for prediction cache entries
    pub prediction_cache_ttl_secs: u64,
    
    /// Output probability scores (for classification)
    pub output_probabilities: bool,
    
    /// Threshold for binary classification
    pub classification_threshold: f64,
}

impl Default for InferenceConfig {
    fn default() -> Self {
        Self {
            batch_size: 1000,
            n_workers: None,
            quantization: QuantizationType::None,
            use_gpu: false,
            device_id: 0,
            max_memory_bytes: None,
            streaming: false,
            stream_chunk_size: 10000,
            cache_preprocessing: true,
            prediction_cache_size: 1000,
            prediction_cache_ttl_secs: 300,
            output_probabilities: false,
            classification_threshold: 0.5,
        }
    }
}

impl InferenceConfig {
    /// Create a new inference configuration
    pub fn new() -> Self {
        Self::default()
    }

    /// Builder method to set batch size
    pub fn with_batch_size(mut self, size: usize) -> Self {
        self.batch_size = size;
        self
    }

    /// Builder method to set number of workers
    pub fn with_n_workers(mut self, n: usize) -> Self {
        self.n_workers = Some(n);
        self
    }

    /// Builder method to enable streaming
    pub fn with_streaming(mut self, chunk_size: usize) -> Self {
        self.streaming = true;
        self.stream_chunk_size = chunk_size;
        self
    }

    /// Builder method to set quantization
    pub fn with_quantization(mut self, quant: QuantizationType) -> Self {
        self.quantization = quant;
        self
    }

    /// Builder method to set maximum memory budget in bytes
    pub fn with_max_memory(mut self, bytes: usize) -> Self {
        self.max_memory_bytes = Some(bytes);
        self
    }

    /// Builder method to set classification threshold
    pub fn with_threshold(mut self, threshold: f64) -> Self {
        self.classification_threshold = threshold;
        self
    }

    /// Builder method to configure prediction cache
    pub fn with_prediction_cache(mut self, size: usize, ttl_secs: u64) -> Self {
        self.prediction_cache_size = size;
        self.prediction_cache_ttl_secs = ttl_secs;
        self
    }

    /// Builder method to enable probability output
    pub fn with_probabilities(mut self) -> Self {
        self.output_probabilities = true;
        self
    }

    /// Validate the configuration, returning a clear error for values that would
    /// otherwise cause a division-by-zero panic downstream (e.g. `batch_size` or
    /// `stream_chunk_size` of 0). This is important because `InferenceConfig`
    /// derives `Deserialize`, so a config loaded from disk or over the wire could
    /// otherwise carry invalid values straight into the inference engine.
    pub fn validate(&self) -> Result<()> {
        if self.batch_size == 0 {
            return Err(AutoMLError::ConfigError(
                "InferenceConfig.batch_size must be greater than 0".to_string(),
            ));
        }
        if self.stream_chunk_size == 0 {
            return Err(AutoMLError::ConfigError(
                "InferenceConfig.stream_chunk_size must be greater than 0".to_string(),
            ));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = InferenceConfig::default();
        assert_eq!(config.batch_size, 1000);
        assert!(!config.streaming);
    }

    #[test]
    fn test_builder_pattern() {
        let config = InferenceConfig::new()
            .with_batch_size(500)
            .with_streaming(5000)
            .with_n_workers(4);
        
        assert_eq!(config.batch_size, 500);
        assert!(config.streaming);
        assert_eq!(config.stream_chunk_size, 5000);
        assert_eq!(config.n_workers, Some(4));
    }
}
