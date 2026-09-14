//! Error types for the AutoML framework

use thiserror::Error;

/// Result type alias for AutoML operations
pub type Result<T> = std::result::Result<T, AutoMLError>;

/// Main error type for the AutoML framework
#[derive(Error, Debug)]
pub enum AutoMLError {
    #[error("Data error: {0}")]
    DataError(String),

    #[error("Preprocessing error: {0}")]
    PreprocessingError(String),

    #[error("Training error: {0}")]
    TrainingError(String),

    #[error("Inference error: {0}")]
    InferenceError(String),

    #[error("Optimization error: {0}")]
    OptimizationError(String),

    #[error("Configuration error: {0}")]
    ConfigError(String),

    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),

    #[error("Serialization error: {0}")]
    SerializationError(String),

    #[error("Invalid shape: expected {expected}, got {actual}")]
    ShapeError { expected: String, actual: String },

    #[error("Feature not found: {0}")]
    FeatureNotFound(String),

    #[error("Model not fitted")]
    ModelNotFitted,

    #[error("Invalid parameter: {name} = {value}, {reason}")]
    InvalidParameter {
        name: String,
        value: String,
        reason: String,
    },

    #[error("Convergence failed after {iterations} iterations")]
    ConvergenceError { iterations: usize },

    #[error("Memory allocation error: {0}")]
    MemoryError(String),

    #[error("Thread pool error: {0}")]
    ThreadPoolError(String),

    #[error("Validation error: {0}")]
    ValidationError(String),

    #[error("Computation error: {0}")]
    ComputationError(String),

    #[error("Invalid input: {0}")]
    InvalidInput(String),

    #[error("Processing error: {0}")]
    ProcessingError(String),

    #[error("Security error: {0}")]
    SecurityError(String),

    #[error("Privacy error: {0}")]
    PrivacyError(String),

    #[error("Fairness error: {0}")]
    FairnessError(String),

    #[error("Compliance error: {0}")]
    ComplianceError(String),

    #[error("Provenance error: {0}")]
    ProvenanceError(String),
}

impl From<polars::error::PolarsError> for AutoMLError {
    fn from(err: polars::error::PolarsError) -> Self {
        AutoMLError::DataError(err.to_string())
    }
}

impl From<serde_json::Error> for AutoMLError {
    fn from(err: serde_json::Error) -> Self {
        AutoMLError::SerializationError(err.to_string())
    }
}

impl From<ndarray::ShapeError> for AutoMLError {
    fn from(err: ndarray::ShapeError) -> Self {
        AutoMLError::ShapeError {
            expected: format!("compatible shape (kind: {:?})", err.kind()),
            actual: err.to_string(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_display() {
        let err = AutoMLError::DataError("test error".to_string());
        assert_eq!(err.to_string(), "Data error: test error");
    }

    #[test]
    fn test_error_from_io() {
        let io_err = std::io::Error::new(std::io::ErrorKind::NotFound, "file not found");
        let err: AutoMLError = io_err.into();
        assert!(matches!(err, AutoMLError::IoError(_)));
    }
}
