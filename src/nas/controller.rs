//! NAS Controller (ENAS-style)
//!
//! Implements a controller network that learns to generate good architectures.

use ndarray::{Array1, Array2};
use rand::prelude::*;
use rand_xoshiro::Xoshiro256PlusPlus;
use serde::{Deserialize, Serialize};

use super::search_space::{NASSearchSpace, NetworkArchitecture, Operation, OperationType, Cell, CellType};

/// Controller configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ControllerConfig {
    /// Hidden state dimension
    pub hidden_dim: usize,
    /// Learning rate
    pub learning_rate: f64,
    /// Entropy weight for exploration
    pub entropy_weight: f64,
    /// Baseline decay for REINFORCE
    pub baseline_decay: f64,
    /// Temperature for sampling
    pub temperature: f64,
    /// Number of samples per architecture search step
    pub num_samples: usize,
}

impl Default for ControllerConfig {
    fn default() -> Self {
        Self {
            hidden_dim: 64,
            learning_rate: 0.001,
            entropy_weight: 0.001,
            baseline_decay: 0.99,
            temperature: 1.0,
            num_samples: 10,
        }
    }
}

/// Which weight matrix a given categorical decision's logits came from.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DecisionHead {
    /// Number-of-layers choice (`hidden_to_layers`)
    Layers,
    /// Hidden-dimension choice (`hidden_to_dim`)
    Dim,
    /// Operation choice for a cell node (`hidden_to_op`)
    Op,
}

/// Everything needed to compute the exact score-function (REINFORCE)
/// gradient for one categorical decision made during `sample()`: which head
/// produced it, the controller hidden state used as input to that head, the
/// resulting softmax probabilities, and the action actually sampled.
#[derive(Debug, Clone)]
pub struct Decision {
    pub head: DecisionHead,
    pub hidden: Array1<f64>,
    pub probs: Array1<f64>,
    pub action: usize,
}

/// Controller state for architecture generation
#[derive(Debug, Clone)]
pub struct ControllerState {
    /// Hidden state
    pub hidden: Array1<f64>,
    /// Log probabilities of sampled actions
    pub log_probs: Vec<f64>,
    /// Entropies of distributions
    pub entropies: Vec<f64>,
    /// Per-decision info needed for the analytic REINFORCE gradient (see
    /// `NASController::update`).
    pub decisions: Vec<Decision>,
}

impl ControllerState {
    /// Create new state
    pub fn new(hidden_dim: usize) -> Self {
        Self {
            hidden: Array1::zeros(hidden_dim),
            log_probs: Vec::new(),
            entropies: Vec::new(),
            decisions: Vec::new(),
        }
    }

    /// Total log probability
    pub fn total_log_prob(&self) -> f64 {
        self.log_probs.iter().sum()
    }

    /// Total entropy
    pub fn total_entropy(&self) -> f64 {
        self.entropies.iter().sum()
    }
}

/// NAS Controller using REINFORCE
#[derive(Debug)]
pub struct NASController {
    /// Configuration
    config: ControllerConfig,
    /// Search space
    search_space: NASSearchSpace,
    /// Operation embedding weights
    op_embeddings: Array2<f64>,
    /// Hidden to operation logits
    hidden_to_op: Array2<f64>,
    /// Hidden to hidden dimension logits
    hidden_to_dim: Array2<f64>,
    /// Hidden to layer count logits
    hidden_to_layers: Array2<f64>,
    /// Recurrent weights
    recurrent_weights: Array2<f64>,
    /// Baseline for variance reduction
    baseline: f64,
    /// Random number generator
    rng: Xoshiro256PlusPlus,
    /// Training history
    history: Vec<(NetworkArchitecture, f64)>,
}

impl NASController {
    /// Create new controller
    pub fn new(search_space: NASSearchSpace, config: ControllerConfig, seed: Option<u64>) -> Self {
        let rng = match seed {
            Some(s) => Xoshiro256PlusPlus::seed_from_u64(s),
            None => Xoshiro256PlusPlus::from_entropy(),
        };

        let num_ops = search_space.num_operations();
        let num_dims = search_space.hidden_dim_choices().len();
        let num_layers = search_space.num_layer_choices();
        let hidden_dim = config.hidden_dim;

        // Initialize weights with small random values
        let mut controller = Self {
            config,
            search_space,
            op_embeddings: Array2::zeros((num_ops, hidden_dim)),
            hidden_to_op: Array2::zeros((hidden_dim, num_ops)),
            hidden_to_dim: Array2::zeros((hidden_dim, num_dims)),
            hidden_to_layers: Array2::zeros((hidden_dim, num_layers)),
            recurrent_weights: Array2::zeros((hidden_dim, hidden_dim)),
            baseline: 0.0,
            rng,
            history: Vec::new(),
        };

        controller.initialize_weights();
        controller
    }

    /// Initialize weights randomly
    fn initialize_weights(&mut self) {
        let scale = 0.1;
        
        for val in self.op_embeddings.iter_mut() {
            *val = (self.rng.gen::<f64>() - 0.5) * scale;
        }
        for val in self.hidden_to_op.iter_mut() {
            *val = (self.rng.gen::<f64>() - 0.5) * scale;
        }
        for val in self.hidden_to_dim.iter_mut() {
            *val = (self.rng.gen::<f64>() - 0.5) * scale;
        }
        for val in self.hidden_to_layers.iter_mut() {
            *val = (self.rng.gen::<f64>() - 0.5) * scale;
        }
        for val in self.recurrent_weights.iter_mut() {
            *val = (self.rng.gen::<f64>() - 0.5) * scale;
        }
    }

    /// Sample an architecture from the controller
    pub fn sample(&mut self) -> (NetworkArchitecture, ControllerState) {
        let hidden_dim = self.config.hidden_dim;
        let mut state = ControllerState::new(hidden_dim);

        // Sample number of layers
        let layer_logits = self.hidden_to_layers.t().dot(&state.hidden);
        let layer_probs = softmax(&layer_logits, self.config.temperature);
        let layer_idx = sample_categorical(&layer_probs, &mut self.rng);
        state.log_probs.push(layer_probs[layer_idx].ln());
        state.entropies.push(entropy(&layer_probs));
        state.decisions.push(Decision {
            head: DecisionHead::Layers,
            hidden: state.hidden.clone(),
            probs: layer_probs.clone(),
            action: layer_idx,
        });

        let num_layers = self.search_space.config.min_layers + layer_idx;

        // Sample hidden dimension
        let dim_logits = self.hidden_to_dim.t().dot(&state.hidden);
        let dim_probs = softmax(&dim_logits, self.config.temperature);
        let dim_idx = sample_categorical(&dim_probs, &mut self.rng);
        state.log_probs.push(dim_probs[dim_idx].ln());
        state.entropies.push(entropy(&dim_probs));
        state.decisions.push(Decision {
            head: DecisionHead::Dim,
            hidden: state.hidden.clone(),
            probs: dim_probs.clone(),
            action: dim_idx,
        });

        let hidden_dim_choice = self.search_space.hidden_dim_choices()[dim_idx];

        // Sample dropout
        let dropout_idx = self.rng.gen_range(0..self.search_space.config.dropout_rates.len());
        let dropout = self.search_space.config.dropout_rates[dropout_idx];

        let mut arch = NetworkArchitecture::new(0, 0)
            .with_hidden_dim(hidden_dim_choice)
            .with_num_layers(num_layers);
        arch.dropout_rate = dropout;

        // Sample cells
        let mut encoding = Vec::new();
        for layer_idx in 0..num_layers {
            let cell_type = if layer_idx % 2 == 0 {
                CellType::Normal
            } else {
                CellType::Reduction
            };

            let mut cell = Cell::new(cell_type);

            // Sample operations for each node
            for node_idx in 0..self.search_space.config.nodes_per_cell {
                // Update hidden state
                state.hidden = self.recurrent_weights.dot(&state.hidden).mapv(|x| x.tanh());

                // Sample operation
                let op_logits = self.hidden_to_op.t().dot(&state.hidden);
                let op_probs = softmax(&op_logits, self.config.temperature);
                let op_idx = sample_categorical(&op_probs, &mut self.rng);
                state.log_probs.push(op_probs[op_idx].ln());
                state.entropies.push(entropy(&op_probs));
                state.decisions.push(Decision {
                    head: DecisionHead::Op,
                    hidden: state.hidden.clone(),
                    probs: op_probs.clone(),
                    action: op_idx,
                });

                let op_type = self.search_space.config.operations[op_idx];
                let mut op = Operation::new(op_type);
                
                if matches!(op_type, OperationType::Dense | OperationType::Attention | OperationType::MultiHeadAttention) {
                    op = op.with_hidden_dim(hidden_dim_choice);
                }
                if matches!(op_type, OperationType::MultiHeadAttention) {
                    op = op.with_num_heads(4);
                }
                if matches!(op_type, OperationType::Dropout) {
                    op = op.with_dropout(dropout);
                }

                // Sample input connections
                let num_possible = node_idx + 2;
                let inputs = if num_possible <= 2 {
                    vec![0]
                } else {
                    let mut inputs = Vec::new();
                    for i in 0..num_possible.min(2) {
                        if self.rng.gen_bool(0.5) || inputs.is_empty() {
                            inputs.push(i);
                        }
                    }
                    inputs
                };

                encoding.push(op_idx);
                cell = cell.add_operation(op, inputs);

                // Update hidden with operation embedding
                if op_idx < self.op_embeddings.nrows() {
                    let emb = self.op_embeddings.row(op_idx);
                    state.hidden = &state.hidden + &emb;
                }
            }

            arch.cells.push(cell);
        }

        arch.encoding = encoding;
        (arch, state)
    }

    /// Update controller based on reward using a manual REINFORCE
    /// (score-function) gradient estimate.
    ///
    /// This controller has no autodiff engine, so instead of backpropagating
    /// through the sampling process we compute the policy-gradient update
    /// analytically for the parameterization actually used here: every
    /// action is drawn from a softmax categorical distribution whose logits
    /// are `logits = W^T . hidden` for the relevant head (`hidden_to_layers`,
    /// `hidden_to_dim`, or `hidden_to_op`). For a softmax categorical, the
    /// score function has the closed form
    /// `d log p(a) / d logits = one_hot(a) - p`, so
    /// `d log p(a) / d W[h, j] = hidden[h] * (one_hot(a)[j] - p[j])`.
    /// We scale that by the baseline-subtracted reward (the REINFORCE
    /// advantage) and apply it as a gradient-*ascent* step on the relevant
    /// weight matrix for every decision recorded in `state.decisions`.
    ///
    /// Known limitations vs. full REINFORCE: `recurrent_weights` and
    /// `op_embeddings` are not updated here, since their effect on the
    /// sampled actions runs through the hidden-state recurrence across
    /// timesteps and would require backpropagation-through-time to get an
    /// exact gradient for; only the direct logit-producing heads get the
    /// exact analytic gradient. The entropy term is applied as a simple
    /// heuristic push toward the uniform distribution (`entropy_weight *
    /// (uniform - probs)`) rather than the exact analytic gradient of the
    /// softmax entropy, since the closed form for that gradient is
    /// significantly more involved and the heuristic already produces the
    /// intended qualitative effect (more exploration when `entropy_weight`
    /// is larger).
    pub fn update(&mut self, state: &ControllerState, reward: f64) {
        // Update baseline
        self.baseline = self.config.baseline_decay * self.baseline
            + (1.0 - self.config.baseline_decay) * reward;

        let advantage = reward - self.baseline;
        let lr = self.config.learning_rate;
        let entropy_weight = self.config.entropy_weight;

        for decision in &state.decisions {
            let n = decision.probs.len();
            if n == 0 {
                continue;
            }
            let uniform = 1.0 / n as f64;

            // d log p(a) / d logits = one_hot(a) - p, scaled by the
            // baseline-subtracted reward (the REINFORCE advantage), plus a
            // heuristic entropy-exploration nudge (see doc comment above).
            let mut grad_logits: Array1<f64> = decision.probs.mapv(|p| -advantage * p);
            grad_logits[decision.action] += advantage;
            for j in 0..n {
                grad_logits[j] += entropy_weight * (uniform - decision.probs[j]);
            }

            let target = match decision.head {
                DecisionHead::Layers => &mut self.hidden_to_layers,
                DecisionHead::Dim => &mut self.hidden_to_dim,
                DecisionHead::Op => &mut self.hidden_to_op,
            };

            // Gradient ascent on expected reward: logits = W^T . hidden, so
            // d logits_j / d W[h, j] = hidden[h].
            for (h, &hv) in decision.hidden.iter().enumerate() {
                if hv == 0.0 {
                    continue;
                }
                for (j, &g) in grad_logits.iter().enumerate() {
                    target[[h, j]] += lr * hv * g;
                }
            }
        }
    }

    /// Get best architecture from history
    pub fn best_architecture(&self) -> Option<&NetworkArchitecture> {
        self.history
            .iter()
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(arch, _)| arch)
    }

    /// Add to history
    pub fn record(&mut self, arch: NetworkArchitecture, reward: f64) {
        self.history.push((arch, reward));
    }

    /// Get history
    pub fn history(&self) -> &[(NetworkArchitecture, f64)] {
        &self.history
    }

    /// Sample multiple architectures
    pub fn sample_batch(&mut self, n: usize) -> Vec<(NetworkArchitecture, ControllerState)> {
        (0..n).map(|_| self.sample()).collect()
    }
}

/// Softmax with temperature
fn softmax(logits: &Array1<f64>, temperature: f64) -> Array1<f64> {
    let scaled = logits / temperature;
    let max_val = scaled.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let exp: Array1<f64> = scaled.mapv(|x| (x - max_val).exp());
    let sum: f64 = exp.sum();
    exp / sum
}

/// Sample from categorical distribution
fn sample_categorical(probs: &Array1<f64>, rng: &mut impl Rng) -> usize {
    let r: f64 = rng.gen();
    let mut cumsum = 0.0;
    for (i, &p) in probs.iter().enumerate() {
        cumsum += p;
        if r < cumsum {
            return i;
        }
    }
    probs.len() - 1
}

/// Compute entropy of distribution
fn entropy(probs: &Array1<f64>) -> f64 {
    -probs.iter()
        .filter(|&&p| p > 1e-10)
        .map(|&p| p * p.ln())
        .sum::<f64>()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_controller_creation() {
        let space = NASSearchSpace::tabular();
        let config = ControllerConfig::default();
        let controller = NASController::new(space, config, Some(42));
        
        assert!(controller.baseline == 0.0);
    }

    #[test]
    fn test_controller_sample() {
        let space = NASSearchSpace::tabular();
        let config = ControllerConfig::default();
        let mut controller = NASController::new(space, config, Some(42));
        
        let (arch, state) = controller.sample();
        
        assert!(arch.num_layers >= 2);
        assert!(!state.log_probs.is_empty());
    }

    #[test]
    fn test_controller_update() {
        let space = NASSearchSpace::tabular();
        let config = ControllerConfig::default();
        let mut controller = NASController::new(space, config, Some(42));
        
        let (arch, state) = controller.sample();
        controller.update(&state, 0.9);
        controller.record(arch, 0.9);
        
        assert!(controller.baseline > 0.0);
        assert_eq!(controller.history().len(), 1);
    }

    #[test]
    fn test_controller_batch_sample() {
        let space = NASSearchSpace::tabular();
        let config = ControllerConfig::default();
        let mut controller = NASController::new(space, config, Some(42));
        
        let batch = controller.sample_batch(5);
        assert_eq!(batch.len(), 5);
    }

    #[test]
    fn test_softmax() {
        let logits = Array1::from_vec(vec![1.0, 2.0, 3.0]);
        let probs = softmax(&logits, 1.0);
        
        assert!((probs.sum() - 1.0).abs() < 1e-6);
        assert!(probs[2] > probs[1]);
        assert!(probs[1] > probs[0]);
    }

    #[test]
    fn test_entropy() {
        let uniform = Array1::from_vec(vec![0.25, 0.25, 0.25, 0.25]);
        let peaked = Array1::from_vec(vec![0.97, 0.01, 0.01, 0.01]);
        
        assert!(entropy(&uniform) > entropy(&peaked));
    }
}
