//! NAS Search Space Definition
//!
//! Defines the space of possible neural network architectures.

use rand::prelude::SliceRandom;
use serde::{Deserialize, Serialize};

/// Types of operations in the search space
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum OperationType {
    /// No operation (skip)
    None,
    /// Skip connection (identity)
    Skip,
    /// Fully connected layer
    Dense,
    /// 1D Convolution
    Conv1D,
    /// Separable convolution
    SeparableConv,
    /// Dilated convolution
    DilatedConv,
    /// Max pooling
    MaxPool,
    /// Average pooling
    AvgPool,
    /// Batch normalization
    BatchNorm,
    /// Layer normalization
    LayerNorm,
    /// Dropout
    Dropout,
    /// ReLU activation
    ReLU,
    /// GELU activation
    GELU,
    /// Swish activation
    Swish,
    /// Attention layer
    Attention,
    /// Multi-head attention
    MultiHeadAttention,
    /// Feed-forward network
    FeedForward,
}

impl OperationType {
    /// Get all standard operations for tabular data
    pub fn tabular_ops() -> Vec<Self> {
        vec![
            Self::None,
            Self::Skip,
            Self::Dense,
            Self::BatchNorm,
            Self::LayerNorm,
            Self::Dropout,
            Self::ReLU,
            Self::GELU,
            Self::Attention,
        ]
    }

    /// Get operations for sequence data
    pub fn sequence_ops() -> Vec<Self> {
        vec![
            Self::None,
            Self::Skip,
            Self::Dense,
            Self::Conv1D,
            Self::SeparableConv,
            Self::DilatedConv,
            Self::MaxPool,
            Self::AvgPool,
            Self::LayerNorm,
            Self::MultiHeadAttention,
        ]
    }
}

/// A single operation with parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Operation {
    /// Operation type
    pub op_type: OperationType,
    /// Hidden dimension (for Dense, Attention, etc.)
    pub hidden_dim: Option<usize>,
    /// Kernel size (for convolutions)
    pub kernel_size: Option<usize>,
    /// Number of heads (for multi-head attention)
    pub num_heads: Option<usize>,
    /// Dropout rate
    pub dropout_rate: Option<f64>,
    /// Dilation rate (for dilated conv)
    pub dilation: Option<usize>,
}

impl Operation {
    /// Create a new operation
    pub fn new(op_type: OperationType) -> Self {
        Self {
            op_type,
            hidden_dim: None,
            kernel_size: None,
            num_heads: None,
            dropout_rate: None,
            dilation: None,
        }
    }

    /// Set hidden dimension
    pub fn with_hidden_dim(mut self, dim: usize) -> Self {
        self.hidden_dim = Some(dim);
        self
    }

    /// Set kernel size
    pub fn with_kernel_size(mut self, size: usize) -> Self {
        self.kernel_size = Some(size);
        self
    }

    /// Set number of attention heads
    pub fn with_num_heads(mut self, heads: usize) -> Self {
        self.num_heads = Some(heads);
        self
    }

    /// Set dropout rate
    pub fn with_dropout(mut self, rate: f64) -> Self {
        self.dropout_rate = Some(rate);
        self
    }

    /// Create skip connection
    pub fn skip() -> Self {
        Self::new(OperationType::Skip)
    }

    /// Create dense layer
    pub fn dense(hidden_dim: usize) -> Self {
        Self::new(OperationType::Dense).with_hidden_dim(hidden_dim)
    }

    /// Create attention layer
    pub fn attention(hidden_dim: usize, num_heads: usize) -> Self {
        Self::new(OperationType::MultiHeadAttention)
            .with_hidden_dim(hidden_dim)
            .with_num_heads(num_heads)
    }
}

/// Cell type in the architecture
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CellType {
    /// Normal cell (preserves dimensions)
    Normal,
    /// Reduction cell (reduces dimensions)
    Reduction,
}

/// A cell in the neural network
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Cell {
    /// Cell type
    pub cell_type: CellType,
    /// Operations in this cell
    pub operations: Vec<Operation>,
    /// Input indices for each operation
    pub input_indices: Vec<Vec<usize>>,
    /// Output aggregation method
    pub output_aggregation: AggregationType,
}

/// How to aggregate outputs
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AggregationType {
    /// Sum outputs
    Sum,
    /// Concatenate outputs
    Concat,
    /// Average outputs
    Mean,
    /// Use last output only
    Last,
}

impl Cell {
    /// Create a new cell
    pub fn new(cell_type: CellType) -> Self {
        Self {
            cell_type,
            operations: Vec::new(),
            input_indices: Vec::new(),
            output_aggregation: AggregationType::Sum,
        }
    }

    /// Add operation
    pub fn add_operation(mut self, op: Operation, inputs: Vec<usize>) -> Self {
        self.operations.push(op);
        self.input_indices.push(inputs);
        self
    }

    /// Set output aggregation
    pub fn with_aggregation(mut self, agg: AggregationType) -> Self {
        self.output_aggregation = agg;
        self
    }

    /// Number of operations
    pub fn num_ops(&self) -> usize {
        self.operations.len()
    }
}

/// Complete network architecture
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NetworkArchitecture {
    /// Input dimension
    pub input_dim: usize,
    /// Output dimension
    pub output_dim: usize,
    /// Cells in the network
    pub cells: Vec<Cell>,
    /// Global hidden dimension
    pub hidden_dim: usize,
    /// Number of layers/cells to stack
    pub num_layers: usize,
    /// Global dropout rate
    pub dropout_rate: f64,
    /// Architecture encoding (for comparison)
    pub encoding: Vec<usize>,
}

impl NetworkArchitecture {
    /// Create new architecture
    pub fn new(input_dim: usize, output_dim: usize) -> Self {
        Self {
            input_dim,
            output_dim,
            cells: Vec::new(),
            hidden_dim: 64,
            num_layers: 3,
            dropout_rate: 0.1,
            encoding: Vec::new(),
        }
    }

    /// Set hidden dimension
    pub fn with_hidden_dim(mut self, dim: usize) -> Self {
        self.hidden_dim = dim;
        self
    }

    /// Set number of layers
    pub fn with_num_layers(mut self, n: usize) -> Self {
        self.num_layers = n;
        self
    }

    /// Add a cell
    pub fn add_cell(mut self, cell: Cell) -> Self {
        self.cells.push(cell);
        self
    }

    /// Compute architecture hash for comparison
    pub fn compute_hash(&self) -> u64 {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let mut hasher = DefaultHasher::new();
        for idx in &self.encoding {
            idx.hash(&mut hasher);
        }
        hasher.finish()
    }
}

/// Configuration for the search space
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchSpaceConfig {
    /// Available operations
    pub operations: Vec<OperationType>,
    /// Minimum hidden dimension
    pub min_hidden_dim: usize,
    /// Maximum hidden dimension
    pub max_hidden_dim: usize,
    /// Hidden dimension step
    pub hidden_dim_step: usize,
    /// Minimum number of layers
    pub min_layers: usize,
    /// Maximum number of layers
    pub max_layers: usize,
    /// Number of nodes per cell
    pub nodes_per_cell: usize,
    /// Maximum number of input connections per node
    pub max_inputs_per_node: usize,
    /// Dropout rate options
    pub dropout_rates: Vec<f64>,
}

impl Default for SearchSpaceConfig {
    fn default() -> Self {
        Self {
            operations: OperationType::tabular_ops(),
            min_hidden_dim: 32,
            max_hidden_dim: 256,
            hidden_dim_step: 32,
            min_layers: 2,
            max_layers: 6,
            nodes_per_cell: 4,
            max_inputs_per_node: 2,
            dropout_rates: vec![0.0, 0.1, 0.2, 0.3, 0.5],
        }
    }
}

/// NAS Search Space
#[derive(Debug, Clone)]
pub struct NASSearchSpace {
    /// Configuration
    pub config: SearchSpaceConfig,
    /// Cached hidden dim options
    hidden_dims: Vec<usize>,
}

impl NASSearchSpace {
    /// Create new search space with config
    ///
    /// # Errors
    /// Returns an error if `min_hidden_dim > max_hidden_dim`, `hidden_dim_step == 0`
    /// (which would panic inside `step_by`), or `min_layers > max_layers`. Any of
    /// these leave `hidden_dims` empty or make `num_layer_choices` underflow, which
    /// later panics when sampling (`gen_range` on an empty range) or attempts a
    /// huge allocation in `NASController::new`.
    pub fn new(config: SearchSpaceConfig) -> crate::error::Result<Self> {
        if config.min_hidden_dim > config.max_hidden_dim {
            return Err(crate::error::AutoMLError::InvalidParameter {
                name: "min_hidden_dim".to_string(),
                value: config.min_hidden_dim.to_string(),
                reason: format!(
                    "must be <= max_hidden_dim ({})",
                    config.max_hidden_dim
                ),
            });
        }
        if config.hidden_dim_step == 0 {
            return Err(crate::error::AutoMLError::InvalidParameter {
                name: "hidden_dim_step".to_string(),
                value: config.hidden_dim_step.to_string(),
                reason: "must be >= 1".to_string(),
            });
        }
        if config.min_layers > config.max_layers {
            return Err(crate::error::AutoMLError::InvalidParameter {
                name: "min_layers".to_string(),
                value: config.min_layers.to_string(),
                reason: format!("must be <= max_layers ({})", config.max_layers),
            });
        }

        let hidden_dims: Vec<usize> = (config.min_hidden_dim..=config.max_hidden_dim)
            .step_by(config.hidden_dim_step)
            .collect();

        Ok(Self { config, hidden_dims })
    }

    /// Create default search space for tabular data
    pub fn tabular() -> Self {
        Self::new(SearchSpaceConfig::default())
            .expect("SearchSpaceConfig::default() must be a valid configuration")
    }

    /// Get number of operation choices
    pub fn num_operations(&self) -> usize {
        self.config.operations.len()
    }

    /// Get hidden dimension choices
    pub fn hidden_dim_choices(&self) -> &[usize] {
        &self.hidden_dims
    }

    /// Get number of layer choices
    ///
    /// Defense in depth: `NASSearchSpace::new` already validates
    /// `min_layers <= max_layers`, but guard the subtraction with
    /// `saturating_sub` here too so this can never underflow/panic (debug)
    /// or wrap to a huge value (release) if that invariant is ever bypassed.
    pub fn num_layer_choices(&self) -> usize {
        self.config.max_layers.saturating_sub(self.config.min_layers) + 1
    }

    /// Sample a random architecture
    pub fn sample_random(&self, rng: &mut impl rand::Rng) -> NetworkArchitecture {
        let num_layers = rng.gen_range(self.config.min_layers..=self.config.max_layers);
        let hidden_dim = self.hidden_dims[rng.gen_range(0..self.hidden_dims.len())];
        let dropout_rate = self.config.dropout_rates[rng.gen_range(0..self.config.dropout_rates.len())];

        let mut arch = NetworkArchitecture::new(0, 0)
            .with_hidden_dim(hidden_dim)
            .with_num_layers(num_layers);
        arch.dropout_rate = dropout_rate;

        // Generate cells
        let mut encoding = Vec::new();
        for layer_idx in 0..num_layers {
            let cell_type = if layer_idx % 2 == 0 {
                CellType::Normal
            } else {
                CellType::Reduction
            };

            let mut cell = Cell::new(cell_type);

            // Add operations to cell
            for node_idx in 0..self.config.nodes_per_cell {
                let op_idx = rng.gen_range(0..self.config.operations.len());
                let op_type = self.config.operations[op_idx];
                
                let mut op = Operation::new(op_type);
                if matches!(op_type, OperationType::Dense | OperationType::Attention | OperationType::MultiHeadAttention) {
                    op = op.with_hidden_dim(hidden_dim);
                }
                if matches!(op_type, OperationType::MultiHeadAttention) {
                    op = op.with_num_heads(4);
                }
                if matches!(op_type, OperationType::Dropout) {
                    op = op.with_dropout(dropout_rate);
                }

                // Random input connections
                let max_inputs = (node_idx + 2).min(self.config.max_inputs_per_node);
                let num_inputs = rng.gen_range(1..=max_inputs);
                let mut inputs: Vec<usize> = (0..node_idx + 2).collect();
                inputs.shuffle(rng);
                inputs.truncate(num_inputs);

                encoding.push(op_idx);
                for &inp in &inputs {
                    encoding.push(inp);
                }

                cell = cell.add_operation(op, inputs);
            }

            arch = arch.add_cell(cell);
        }

        arch.encoding = encoding;
        arch
    }

    /// Mutate an architecture
    pub fn mutate(&self, arch: &NetworkArchitecture, rng: &mut impl rand::Rng) -> NetworkArchitecture {
        let mut new_arch = arch.clone();

        // Choose mutation type
        let mutation_type = rng.gen_range(0..4);

        match mutation_type {
            0 => {
                // Change hidden dimension
                new_arch.hidden_dim = self.hidden_dims[rng.gen_range(0..self.hidden_dims.len())];
            }
            1 => {
                // Change dropout
                new_arch.dropout_rate = self.config.dropout_rates[rng.gen_range(0..self.config.dropout_rates.len())];
            }
            2 => {
                // Change an operation
                if !new_arch.cells.is_empty() {
                    let cell_idx = rng.gen_range(0..new_arch.cells.len());
                    let cell = &mut new_arch.cells[cell_idx];
                    if !cell.operations.is_empty() {
                        let op_idx = rng.gen_range(0..cell.operations.len());
                        let new_op_type = self.config.operations[rng.gen_range(0..self.config.operations.len())];
                        cell.operations[op_idx].op_type = new_op_type;
                    }
                }
            }
            _ => {
                // Change number of layers
                let new_layers = rng.gen_range(self.config.min_layers..=self.config.max_layers);
                if new_layers > new_arch.num_layers && new_arch.cells.len() < new_layers {
                    // Add a cell
                    let cell = Cell::new(CellType::Normal)
                        .add_operation(Operation::dense(new_arch.hidden_dim), vec![0]);
                    new_arch.cells.push(cell);
                } else if new_layers < new_arch.num_layers && !new_arch.cells.is_empty() {
                    // Remove a cell
                    new_arch.cells.pop();
                }
                new_arch.num_layers = new_layers;
            }
        }

        new_arch
    }

    /// Crossover two architectures
    pub fn crossover(
        &self,
        parent1: &NetworkArchitecture,
        parent2: &NetworkArchitecture,
        rng: &mut impl rand::Rng,
    ) -> NetworkArchitecture {
        let mut child = NetworkArchitecture::new(parent1.input_dim, parent1.output_dim);

        // Inherit from parents
        child.hidden_dim = if rng.gen_bool(0.5) {
            parent1.hidden_dim
        } else {
            parent2.hidden_dim
        };

        child.dropout_rate = if rng.gen_bool(0.5) {
            parent1.dropout_rate
        } else {
            parent2.dropout_rate
        };

        child.num_layers = if rng.gen_bool(0.5) {
            parent1.num_layers
        } else {
            parent2.num_layers
        };

        // Mix cells from both parents
        let max_cells = parent1.cells.len().max(parent2.cells.len());
        for i in 0..child.num_layers.min(max_cells) {
            let cell = if rng.gen_bool(0.5) {
                if i < parent1.cells.len() {
                    parent1.cells[i].clone()
                } else if i < parent2.cells.len() {
                    parent2.cells[i].clone()
                } else {
                    Cell::new(CellType::Normal)
                }
            } else {
                if i < parent2.cells.len() {
                    parent2.cells[i].clone()
                } else if i < parent1.cells.len() {
                    parent1.cells[i].clone()
                } else {
                    Cell::new(CellType::Normal)
                }
            };
            child.cells.push(cell);
        }

        child
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;
    use rand_xoshiro::Xoshiro256PlusPlus;

    #[test]
    fn test_search_space_creation() {
        let space = NASSearchSpace::tabular();
        assert!(space.num_operations() > 0);
        assert!(!space.hidden_dim_choices().is_empty());
    }

    #[test]
    fn test_sample_random_architecture() {
        let space = NASSearchSpace::tabular();
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(42);
        
        let arch = space.sample_random(&mut rng);
        
        assert!(arch.num_layers >= 2);
        assert!(arch.hidden_dim >= 32);
    }

    #[test]
    fn test_mutate_architecture() {
        let space = NASSearchSpace::tabular();
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(42);
        
        let arch = space.sample_random(&mut rng);
        let mutated = space.mutate(&arch, &mut rng);
        
        // Should produce a valid architecture
        assert!(mutated.num_layers >= space.config.min_layers);
    }

    #[test]
    fn test_crossover() {
        let space = NASSearchSpace::tabular();
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(42);
        
        let parent1 = space.sample_random(&mut rng);
        let parent2 = space.sample_random(&mut rng);
        let child = space.crossover(&parent1, &parent2, &mut rng);
        
        assert!(child.num_layers >= space.config.min_layers);
    }

    #[test]
    fn test_operation_builder() {
        let op = Operation::dense(128).with_dropout(0.2);
        assert_eq!(op.hidden_dim, Some(128));
        assert_eq!(op.dropout_rate, Some(0.2));
    }

    #[test]
    fn test_cell_builder() {
        let cell = Cell::new(CellType::Normal)
            .add_operation(Operation::dense(64), vec![0])
            .add_operation(Operation::skip(), vec![0, 1])
            .with_aggregation(AggregationType::Concat);
        
        assert_eq!(cell.num_ops(), 2);
        assert_eq!(cell.output_aggregation, AggregationType::Concat);
    }
}
