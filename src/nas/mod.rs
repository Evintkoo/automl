//! Neural Architecture Search (NAS) Module
//!
//! Provides automated neural architecture search capabilities:
//! - Search space definition for neural networks
//! - DARTS-style differentiable architecture search
//! - Controller-based search (ENAS-style)
//! - Random search baseline
//! - Aging Evolution

mod search_space;
mod controller;
mod darts;
mod evaluator;

pub use search_space::{
    NASSearchSpace, Operation, OperationType, Cell, CellType,
    NetworkArchitecture, SearchSpaceConfig,
};
pub use controller::{NASController, ControllerConfig, ControllerState, Decision, DecisionHead};
pub use darts::{DARTSSearch, DARTSConfig, ArchitectureWeights};
pub use evaluator::{ArchitectureEvaluator, EvaluationResult, EvaluationConfig};
