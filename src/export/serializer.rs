//! Model serialization utilities
//!
//! Provides serialization and deserialization for trained models.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs::File;
use std::io::{Read, Write, BufReader, BufWriter};
use std::path::Path;

use crate::error::{AutoMLError, Result};

/// Serialization format
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SerializationFormat {
    /// Binary format using bincode (efficient)
    Binary,
    /// JSON format (portable, human-readable)
    Json,
}

impl Default for SerializationFormat {
    fn default() -> Self {
        SerializationFormat::Binary
    }
}

/// Model metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelMetadata {
    /// Model name
    pub name: String,
    /// Model version
    pub version: String,
    /// Training timestamp (ISO 8601)
    pub trained_at: String,
    /// Feature names
    pub feature_names: Vec<String>,
    /// Target name
    pub target_name: String,
    /// Model type
    pub model_type: String,
    /// Hyperparameters
    pub hyperparameters: HashMap<String, String>,
    /// Training metrics
    pub metrics: HashMap<String, f64>,
    /// Additional metadata
    pub extra: HashMap<String, String>,
}

impl Default for ModelMetadata {
    fn default() -> Self {
        Self {
            name: "model".to_string(),
            version: "1.0.0".to_string(),
            trained_at: String::new(),
            feature_names: Vec::new(),
            target_name: "target".to_string(),
            model_type: "unknown".to_string(),
            hyperparameters: HashMap::new(),
            metrics: HashMap::new(),
            extra: HashMap::new(),
        }
    }
}

impl ModelMetadata {
    /// Create new metadata with name
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            ..Default::default()
        }
    }

    /// Set version
    pub fn with_version(mut self, version: impl Into<String>) -> Self {
        self.version = version.into();
        self
    }

    /// Set model type
    pub fn with_model_type(mut self, model_type: impl Into<String>) -> Self {
        self.model_type = model_type.into();
        self
    }

    /// Set feature names
    pub fn with_features(mut self, features: Vec<String>) -> Self {
        self.feature_names = features;
        self
    }

    /// Set target name
    pub fn with_target(mut self, target: impl Into<String>) -> Self {
        self.target_name = target.into();
        self
    }

    /// Add hyperparameter
    pub fn add_hyperparameter(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.hyperparameters.insert(key.into(), value.into());
        self
    }

    /// Add metric
    pub fn add_metric(mut self, key: impl Into<String>, value: f64) -> Self {
        self.metrics.insert(key.into(), value);
        self
    }
}

/// Serializable model wrapper
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SerializedModel {
    /// Magic bytes for format detection
    pub magic: [u8; 4],
    /// Format version
    pub format_version: u32,
    /// Model metadata
    pub metadata: ModelMetadata,
    /// Serialized model data
    pub model_data: Vec<u8>,
    /// Checksum for integrity verification
    pub checksum: u64,
}

impl SerializedModel {
    /// Magic bytes for AutoML model files
    const MAGIC: [u8; 4] = [b'K', b'O', b'L', b'M'];
    /// Current format version
    const VERSION: u32 = 1;

    /// Create new serialized model
    pub fn new(metadata: ModelMetadata, model_data: Vec<u8>) -> Self {
        let checksum = Self::compute_checksum(&model_data);
        Self {
            magic: Self::MAGIC,
            format_version: Self::VERSION,
            metadata,
            model_data,
            checksum,
        }
    }

    /// Compute checksum using FNV-1a hash
    fn compute_checksum(data: &[u8]) -> u64 {
        const FNV_OFFSET: u64 = 14695981039346656037;
        const FNV_PRIME: u64 = 1099511628211;
        
        let mut hash = FNV_OFFSET;
        for byte in data {
            hash ^= *byte as u64;
            hash = hash.wrapping_mul(FNV_PRIME);
        }
        hash
    }

    /// Verify checksum
    pub fn verify_checksum(&self) -> bool {
        Self::compute_checksum(&self.model_data) == self.checksum
    }
}

/// Model serializer trait
pub trait ModelSerializer: Serialize + for<'de> Deserialize<'de> + Sized {
    /// Get model metadata
    fn metadata(&self) -> ModelMetadata;

    /// Serialize to bytes using binary format
    fn to_bytes(&self) -> Result<Vec<u8>> {
        bincode::serialize(self).map_err(|e| {
            AutoMLError::SerializationError(format!("Failed to serialize: {}", e))
        })
    }

    /// Deserialize from bytes
    fn from_bytes(data: &[u8]) -> Result<Self> {
        bincode::deserialize(data).map_err(|e| {
            AutoMLError::SerializationError(format!("Failed to deserialize: {}", e))
        })
    }

    /// Serialize to JSON string
    fn to_json(&self) -> Result<String> {
        serde_json::to_string_pretty(self).map_err(|e| {
            AutoMLError::SerializationError(format!("Failed to serialize to JSON: {}", e))
        })
    }

    /// Deserialize from JSON string
    fn from_json(json: &str) -> Result<Self> {
        serde_json::from_str(json).map_err(|e| {
            AutoMLError::SerializationError(format!("Failed to deserialize from JSON: {}", e))
        })
    }

    /// Save to file
    fn save(&self, path: impl AsRef<Path>, format: SerializationFormat) -> Result<()> {
        let model_data = self.to_bytes()?;
        let serialized = SerializedModel::new(self.metadata(), model_data);
        
        let file = File::create(path.as_ref()).map_err(|e| {
            AutoMLError::DataError(format!("Failed to create file: {}", e))
        })?;
        let mut writer = BufWriter::new(file);

        match format {
            SerializationFormat::Binary => {
                let bytes = bincode::serialize(&serialized).map_err(|e| {
                    AutoMLError::SerializationError(format!("Failed to serialize: {}", e))
                })?;
                writer.write_all(&bytes).map_err(|e| {
                    AutoMLError::DataError(format!("Failed to write: {}", e))
                })?;
            }
            SerializationFormat::Json => {
                serde_json::to_writer_pretty(&mut writer, &serialized).map_err(|e| {
                    AutoMLError::SerializationError(format!("Failed to write JSON: {}", e))
                })?;
            }
        }

        Ok(())
    }

    /// Load from file
    fn load(path: impl AsRef<Path>, format: SerializationFormat) -> Result<Self> {
        let file = File::open(path.as_ref()).map_err(|e| {
            AutoMLError::DataError(format!("Failed to open file: {}", e))
        })?;
        let mut reader = BufReader::new(file);

        let serialized: SerializedModel = match format {
            SerializationFormat::Binary => {
                let mut bytes = Vec::new();
                reader.read_to_end(&mut bytes).map_err(|e| {
                    AutoMLError::DataError(format!("Failed to read: {}", e))
                })?;
                bincode::deserialize(&bytes).map_err(|e| {
                    AutoMLError::SerializationError(format!("Failed to deserialize: {}", e))
                })?
            }
            SerializationFormat::Json => {
                serde_json::from_reader(&mut reader).map_err(|e| {
                    AutoMLError::SerializationError(format!("Failed to read JSON: {}", e))
                })?
            }
        };

        // Verify checksum
        if !serialized.verify_checksum() {
            return Err(AutoMLError::SerializationError(
                "Checksum verification failed - file may be corrupted".to_string()
            ));
        }

        Self::from_bytes(&serialized.model_data)
    }
}

/// Save a serializable model to file
pub fn save_model<M: Serialize>(
    model: &M,
    path: impl AsRef<Path>,
    metadata: ModelMetadata,
) -> Result<()> {
    let model_data = bincode::serialize(model).map_err(|e| {
        AutoMLError::SerializationError(format!("Failed to serialize: {}", e))
    })?;
    
    let serialized = SerializedModel::new(metadata, model_data);
    
    let file = File::create(path.as_ref()).map_err(|e| {
        AutoMLError::DataError(format!("Failed to create file: {}", e))
    })?;
    let writer = BufWriter::new(file);

    bincode::serialize_into(writer, &serialized).map_err(|e| {
        AutoMLError::SerializationError(format!("Failed to write: {}", e))
    })?;

    Ok(())
}

/// Load a model from file
pub fn load_model<M: for<'de> Deserialize<'de>>(path: impl AsRef<Path>) -> Result<(M, ModelMetadata)> {
    let file = File::open(path.as_ref()).map_err(|e| {
        AutoMLError::DataError(format!("Failed to open file: {}", e))
    })?;
    let reader = BufReader::new(file);

    let serialized: SerializedModel = bincode::deserialize_from(reader).map_err(|e| {
        AutoMLError::SerializationError(format!("Failed to deserialize: {}", e))
    })?;

    if !serialized.verify_checksum() {
        return Err(AutoMLError::SerializationError(
            "Checksum verification failed".to_string()
        ));
    }

    let model: M = bincode::deserialize(&serialized.model_data).map_err(|e| {
        AutoMLError::SerializationError(format!("Failed to deserialize model: {}", e))
    })?;

    Ok((model, serialized.metadata))
}

/// Save model to JSON file
pub fn save_model_json<M: Serialize>(
    model: &M,
    path: impl AsRef<Path>,
    metadata: ModelMetadata,
) -> Result<()> {
    #[derive(Serialize)]
    struct JsonModel<'a, M: Serialize> {
        metadata: &'a ModelMetadata,
        model: &'a M,
    }

    let json_model = JsonModel {
        metadata: &metadata,
        model,
    };

    let file = File::create(path.as_ref()).map_err(|e| {
        AutoMLError::DataError(format!("Failed to create file: {}", e))
    })?;
    let writer = BufWriter::new(file);

    serde_json::to_writer_pretty(writer, &json_model).map_err(|e| {
        AutoMLError::SerializationError(format!("Failed to write JSON: {}", e))
    })?;

    Ok(())
}

/// Load model from JSON file
pub fn load_model_json<M: for<'de> Deserialize<'de>>(
    path: impl AsRef<Path>,
) -> Result<(M, ModelMetadata)> {
    #[derive(Deserialize)]
    struct JsonModel<M> {
        metadata: ModelMetadata,
        model: M,
    }

    let file = File::open(path.as_ref()).map_err(|e| {
        AutoMLError::DataError(format!("Failed to open file: {}", e))
    })?;
    let reader = BufReader::new(file);

    let json_model: JsonModel<M> = serde_json::from_reader(reader).map_err(|e| {
        AutoMLError::SerializationError(format!("Failed to read JSON: {}", e))
    })?;

    Ok((json_model.model, json_model.metadata))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    #[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
    struct TestModel {
        weights: Vec<f64>,
        bias: f64,
    }

    #[test]
    fn test_serialized_model_checksum() {
        let data = vec![1, 2, 3, 4, 5];
        let metadata = ModelMetadata::new("test");
        let serialized = SerializedModel::new(metadata, data);
        
        assert!(serialized.verify_checksum());
    }

    #[test]
    fn test_serialized_model_checksum_failure() {
        let data = vec![1, 2, 3, 4, 5];
        let metadata = ModelMetadata::new("test");
        let mut serialized = SerializedModel::new(metadata, data);
        
        // Corrupt the data
        serialized.model_data[0] = 99;
        
        assert!(!serialized.verify_checksum());
    }

    #[test]
    fn test_metadata_builder() {
        let metadata = ModelMetadata::new("my_model")
            .with_version("2.0.0")
            .with_model_type("linear_regression")
            .with_features(vec!["x1".to_string(), "x2".to_string()])
            .with_target("y")
            .add_hyperparameter("learning_rate", "0.01")
            .add_metric("rmse", 0.123);
        
        assert_eq!(metadata.name, "my_model");
        assert_eq!(metadata.version, "2.0.0");
        assert_eq!(metadata.model_type, "linear_regression");
        assert_eq!(metadata.feature_names.len(), 2);
        assert_eq!(metadata.hyperparameters.get("learning_rate"), Some(&"0.01".to_string()));
        assert_eq!(metadata.metrics.get("rmse"), Some(&0.123));
    }

    #[test]
    fn test_model_serialization() {
        let model = TestModel {
            weights: vec![1.0, 2.0, 3.0],
            bias: 0.5,
        };
        
        let bytes = bincode::serialize(&model).unwrap();
        let restored: TestModel = bincode::deserialize(&bytes).unwrap();
        
        assert_eq!(model, restored);
    }

    #[test]
    fn test_json_serialization() {
        let model = TestModel {
            weights: vec![1.0, 2.0, 3.0],
            bias: 0.5,
        };
        
        let json = serde_json::to_string(&model).unwrap();
        let restored: TestModel = serde_json::from_str(&json).unwrap();
        
        assert_eq!(model, restored);
    }
}
