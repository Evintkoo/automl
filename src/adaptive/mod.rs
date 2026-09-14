//! Adaptive optimization modules — auto-tuned preprocessing and hyperparameter search.
//!
//! This module provides two key capabilities:
//!
//! - **Adaptive Preprocessing** ([`AdaptivePreprocessorConfig`]): Profiles dataset characteristics
//!   (size, sparsity, skewness, missing ratio) and automatically selects optimal preprocessing
//!   strategies (normalization, chunk sizes, processing modes).
//!
//! - **Adaptive Hyperparameter Optimization** ([`AdaptiveHyperparameterOptimizer`]): Narrows the
//!   search space over time based on top-performing trials, with convergence detection and
//!   promising region identification.
//!
//! # Example
//! ```
//! use automl::adaptive::{AdaptivePreprocessorConfig, DatasetCharacteristics, ProcessingMode};
//!
//! let chars = DatasetCharacteristics {
//!     n_rows: 100_000,
//!     n_cols: 50,
//!     n_numeric: 40,
//!     n_categorical: 10,
//!     n_datetime: 0,
//!     n_text: 0,
//!     sparsity: 0.1,
//!     skewness_avg: 0.5,
//!     missing_ratio: 0.05,
//!     outlier_ratio: 0.02,
//!     high_cardinality_count: 0,
//!     memory_mb: 500.0,
//!     dataset_size: automl::adaptive::DatasetSize::Large,
//! };
//! let mode = AdaptivePreprocessorConfig::determine_strategy(&chars);
//! ```
pub mod preprocessing;
pub mod hyperopt;

pub use preprocessing::{
    AdaptivePreprocessorConfig, ConfigOptimizer, DatasetCharacteristics, DatasetSize, ProcessingMode,
};
pub use hyperopt::{
    AdaptiveHyperparameterOptimizer, AdaptiveHyperoptConfig, PerformanceTracker, AdaptiveSearchSpace, SearchSpaceConfig,
};
