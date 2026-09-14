//! AutoML - High-performance AutoML engine
//!
//! This crate provides a complete AutoML framework including:
//! - Data preprocessing, scaling, encoding
//! - Model training with multiple algorithms
//! - High-performance inference engine
//! - Hyperparameter optimization
//! - Web server and CLI interfaces
//!
//! # Modules
//!
//! ## Core ML Modules
//! - [`preprocessing`] - Data preprocessing, scaling, encoding
//! - [`training`] - Model training with multiple algorithms
//! - [`inference`] - High-performance inference engine
//! - [`optimizer`] - Hyperparameter optimization (Bayesian, ASHT)
//!
//! ## Advanced ML Features
//! - [`explainability`] - Model interpretability (SHAP-like, PDP)
//! - [`ensemble`] - Voting and stacking ensembles
//! - [`calibration`] - Probability calibration
//! - [`anomaly`] - Anomaly detection (Isolation Forest, LOF)
//! - [`drift`] - Data and concept drift detection
//!
//! ## Data Processing
//! - [`feature_engineering`] - Feature generation and transformation
//! - [`imputation`] - Missing value imputation
//! - [`synthetic`] - Synthetic data generation (SMOTE, ADASYN)
//! - [`timeseries`] - Time series features and validation
//!
//! ## Infrastructure
//! - [`batch`] - Batch processing with priority queues
//! - [`cache`] - Multi-level caching with LRU and TTL
//! - [`memory`] - Memory pooling for buffer management
//! - [`streaming`] - Streaming pipelines with backpressure
//! - [`quantization`] - Data quantization for efficiency
//! - [`monitoring`] - Performance metrics and statistics
//! - [`tracking`] - Experiment tracking
//!
//! ## Services
//! - [`server`] - HTTP server with REST API
//! - [`cli`] - Command-line interface
//!
//! ## Utilities
//! - [`autopipeline`] - Automatic pipeline construction
//! - [`export`] - Model serialization (ONNX, PMML), model cards, datasheets
//! - [`nas`] - Neural architecture search
//!
//! ## ISO Compliance
//! - [`provenance`] - Data lineage and provenance tracking (ISO 5259, 5338)
//! - [`fairness`] - Fairness metrics and bias detection (ISO TR 24027)
//! - [`privacy`] - PII detection, anonymization, retention (ISO 27701)
//! - [`compliance`] - Compliance reporting and audit (ISO 42001, 27001)

// Core error handling
pub mod error;

// Core ML modules
pub mod preprocessing;
pub mod training;
pub mod inference;
pub mod optimizer;

// Advanced ML features
pub mod explainability;
pub mod ensemble;
pub mod calibration;
pub mod anomaly;
pub mod drift;

// Data processing
pub mod feature_engineering;
pub mod imputation;
pub mod synthetic;
pub mod timeseries;

// Infrastructure - Performance & Optimization
pub mod batch;
pub mod cache;
pub mod memory;
pub mod streaming;
pub mod quantization;
pub mod precision;
pub mod monitoring;
pub mod tracking;

// Utilities
pub mod autopipeline;
pub mod export;
pub mod nas;
pub mod utils;
pub mod architectures;

// Services
pub mod server;
pub mod cli;

// Device optimization
pub mod device;

// Pipeline quality gates
pub mod quality;

// Adaptive optimization
pub mod adaptive;

// Security
pub mod security;

// Visualization
pub mod visualization;

// ISO Compliance Modules
pub mod provenance;
pub mod fairness;
pub mod privacy;
pub mod compliance;

pub use error::{AutoMLError, Result};

/// Re-export commonly used types
pub mod prelude {
    // Error handling
    pub use crate::error::{AutoMLError, Result};
    
    // Preprocessing
    pub use crate::preprocessing::{DataPreprocessor, PreprocessingConfig, FeatureSelector, SelectionMethod};
    
    // Training
    pub use crate::training::{TrainEngine, TrainingConfig, ModelType};
    
    // Inference
    pub use crate::inference::{InferenceEngine, InferenceConfig, InferenceStats};
    
    // Optimization
    pub use crate::optimizer::{HyperOptX, OptimizationConfig, SearchSpace, BayesianOptimizer, GaussianProcess};
    
    // Explainability
    pub use crate::explainability::{PermutationImportance, PartialDependence, LocalExplainer};
    
    // Ensemble
    pub use crate::ensemble::{VotingClassifier, VotingRegressor, StackingConfig};
    
    // Time series
    pub use crate::timeseries::{TimeSeriesFeatures, TimeSeriesCV, Differencer};
    
    // Auto pipeline
    pub use crate::autopipeline::{AutoPipeline, PipelineConfig, DataTypeDetector};
    
    // Calibration
    pub use crate::calibration::{PlattScaling, IsotonicRegression, TemperatureScaling, Calibrator};
    
    // Anomaly detection
    pub use crate::anomaly::{IsolationForest, LocalOutlierFactor, AnomalyDetector};
    
    // Drift detection
    pub use crate::drift::{DataDriftDetector, ConceptDriftDetector, FeatureDriftMonitor, DriftResult};
    
    // Feature engineering
    pub use crate::feature_engineering::{PolynomialFeatures, FeatureInteractions, TfidfVectorizer, FeatureTransformer};
    
    // Imputation
    pub use crate::imputation::{MICEImputer, KNNImputer, IterativeImputer, Imputer};
    
    // Synthetic data
    pub use crate::synthetic::{SMOTE, ADASYN, RandomOverSampler, Sampler};
    
    // Export
    pub use crate::export::{ModelSerializer, ModelMetadata, SerializationFormat, ONNXExporter, PMMLExporter, ModelRegistry};
    
    // Batch processing
    pub use crate::batch::{BatchProcessor, BatchProcessorConfig, DynamicBatcher, Priority, PriorityQueue};
    
    // Caching
    pub use crate::cache::{LruTtlCache, MultiLevelCache, CacheLevel};
    
    // Memory management
    pub use crate::memory::{MemoryPool, PooledBuffer};
    
    // Streaming
    pub use crate::streaming::{StreamingPipeline, StreamConfig, BackpressureController};
    
    // Quantization
    pub use crate::quantization::{Quantizer, QuantizationConfig, QuantizationType, QuantizationMode};
    
    // Monitoring
    pub use crate::monitoring::{PerformanceMetrics, BatchStats};
    
    // Experiment tracking
    pub use crate::tracking::{ExperimentTracker, ExperimentConfig, Experiment, Run};
    
    // Device optimization
    pub use crate::device::{DeviceOptimizer, HardwareInfo, CpuCapabilities, OptimalConfigs};
    
    // Adaptive processing
    pub use crate::adaptive::{
        AdaptivePreprocessorConfig, DatasetCharacteristics, DatasetSize, ProcessingMode,
        AdaptiveHyperparameterOptimizer, AdaptiveHyperoptConfig, AdaptiveSearchSpace,
    };
    
    // Mixed precision
    pub use crate::precision::{MixedPrecisionManager, MixedPrecisionConfig, PrecisionMode};
    
    // Security
    pub use crate::security::{SecurityManager, SecurityConfig, RateLimiter, RateLimitConfig};
    
    // Neural architecture search
    pub use crate::nas::{NASSearchSpace, NASController, DARTSSearch, DARTSConfig, ArchitectureEvaluator};

    // Architecture layers
    pub use crate::architectures::{TabNet, TabNetConfig, FTTransformer, FTTransformerConfig};

    // ISO Compliance - Provenance
    pub use crate::provenance::{ProvenanceTracker, DataLineage, DataSource, TransformationRecord};

    // ISO Compliance - Fairness
    pub use crate::fairness::{FairnessEvaluator, FairnessConfig, FairnessReport, BiasReport};

    // ISO Compliance - Privacy
    pub use crate::privacy::{PiiScanner, PiiScanResult, Anonymizer, AnonymizationMethod, RetentionManager, DataClassification};

    // ISO Compliance - Compliance Reporting
    pub use crate::compliance::{ComplianceChecker, ComplianceReport, ComplianceCheckConfig};

    // ISO Compliance - Security Extensions
    pub use crate::security::{RbacManager, Role, AuditTrail, AuditEventType};

    // ISO Compliance - Export Extensions
    pub use crate::export::{ModelCard, Datasheet};
}
