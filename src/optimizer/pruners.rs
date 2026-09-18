//! Pruners for early stopping of trials

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Pruner trait for deciding when to stop unpromising trials early
pub trait Pruner: Send + Sync {
    /// Decide whether to prune a trial at the current step
    fn should_prune(&self, trial_id: usize, step: usize, value: f64, history: &TrialHistory) -> bool;
    
    /// Report a value for a trial at a step
    fn report(&mut self, trial_id: usize, step: usize, value: f64);
}

/// History of trial intermediate values
#[derive(Debug, Clone, Default)]
pub struct TrialHistory {
    /// Values per trial: trial_id -> (step -> value)
    values: HashMap<usize, HashMap<usize, f64>>,
    /// Same data, transposed: step -> (trial_id -> value). Kept in sync with
    /// `values` on every `report()` call so `get_step_values` — called once
    /// per step per trial from `should_prune` — doesn't need to scan every
    /// trial ever reported (O(n_trials) per call, O(n_trials^2 * n_steps)
    /// over an optimization run) to find the handful reporting this step.
    by_step: HashMap<usize, HashMap<usize, f64>>,
}

impl TrialHistory {
    /// Create new history
    pub fn new() -> Self {
        Self::default()
    }

    /// Report a value
    pub fn report(&mut self, trial_id: usize, step: usize, value: f64) {
        self.values
            .entry(trial_id)
            .or_insert_with(HashMap::new)
            .insert(step, value);
        self.by_step
            .entry(step)
            .or_insert_with(HashMap::new)
            .insert(trial_id, value);
    }

    /// Get all values for a trial
    pub fn get_trial(&self, trial_id: usize) -> Option<&HashMap<usize, f64>> {
        self.values.get(&trial_id)
    }

    /// Get value at specific step for trial
    pub fn get_value(&self, trial_id: usize, step: usize) -> Option<f64> {
        self.values.get(&trial_id)?.get(&step).copied()
    }

    /// Get all values at a specific step across trials
    pub fn get_step_values(&self, step: usize) -> Vec<f64> {
        self.by_step
            .get(&step)
            .map(|trials| trials.values().copied().collect())
            .unwrap_or_default()
    }

    /// Get number of trials
    pub fn n_trials(&self) -> usize {
        self.values.len()
    }

    /// Get all trial IDs
    pub fn trial_ids(&self) -> Vec<usize> {
        self.values.keys().copied().collect()
    }
}

/// Median pruner - prunes if value is worse than median of previous trials
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MedianPruner {
    /// Number of startup trials to skip pruning
    n_startup_trials: usize,
    /// Number of warmup steps to skip pruning
    n_warmup_steps: usize,
    /// Minimum number of trials at step before pruning
    n_min_trials: usize,
    /// Whether to minimize (true) or maximize (false)
    minimize: bool,
    /// Internal history
    #[serde(skip)]
    history: TrialHistory,
}

impl MedianPruner {
    /// Create a new median pruner
    pub fn new(minimize: bool) -> Self {
        Self {
            n_startup_trials: 5,
            n_warmup_steps: 0,
            n_min_trials: 1,
            minimize,
            history: TrialHistory::new(),
        }
    }

    /// Set number of startup trials
    pub fn with_n_startup_trials(mut self, n: usize) -> Self {
        self.n_startup_trials = n;
        self
    }

    /// Set number of warmup steps
    pub fn with_n_warmup_steps(mut self, n: usize) -> Self {
        self.n_warmup_steps = n;
        self
    }
}

impl Pruner for MedianPruner {
    fn should_prune(&self, trial_id: usize, step: usize, value: f64, _history: &TrialHistory) -> bool {
        // Skip startup trials
        if trial_id < self.n_startup_trials {
            return false;
        }

        // Skip warmup steps
        if step < self.n_warmup_steps {
            return false;
        }

        // Get all values at this step
        let values = self.history.get_step_values(step);
        if values.len() < self.n_min_trials {
            return false;
        }

        // Compute median
        let mut sorted = values.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let median = sorted[sorted.len() / 2];

        // Prune if worse than median
        if self.minimize {
            value > median
        } else {
            value < median
        }
    }

    fn report(&mut self, trial_id: usize, step: usize, value: f64) {
        self.history.report(trial_id, step, value);
    }
}

/// Percentile pruner - prunes if value is in the worst percentile
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PercentilePruner {
    /// Percentile threshold (0-100)
    percentile: f64,
    /// Number of startup trials
    n_startup_trials: usize,
    /// Number of warmup steps
    n_warmup_steps: usize,
    /// Minimum trials at step
    n_min_trials: usize,
    /// Whether to minimize
    minimize: bool,
    /// Internal history
    #[serde(skip)]
    history: TrialHistory,
}

impl PercentilePruner {
    /// Create a new percentile pruner
    pub fn new(percentile: f64, minimize: bool) -> Self {
        Self {
            percentile: percentile.clamp(0.0, 100.0),
            n_startup_trials: 5,
            n_warmup_steps: 0,
            n_min_trials: 1,
            minimize,
            history: TrialHistory::new(),
        }
    }
}

impl Pruner for PercentilePruner {
    fn should_prune(&self, trial_id: usize, step: usize, value: f64, _history: &TrialHistory) -> bool {
        if trial_id < self.n_startup_trials || step < self.n_warmup_steps {
            return false;
        }

        let values = self.history.get_step_values(step);
        if values.len() < self.n_min_trials {
            return false;
        }

        let mut sorted = values.clone();
        if self.minimize {
            sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        } else {
            sorted.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
        }

        let threshold_idx = ((sorted.len() as f64) * (self.percentile / 100.0)) as usize;
        let threshold_idx = threshold_idx.min(sorted.len() - 1);
        let threshold = sorted[threshold_idx];

        if self.minimize {
            value > threshold
        } else {
            value < threshold
        }
    }

    fn report(&mut self, trial_id: usize, step: usize, value: f64) {
        self.history.report(trial_id, step, value);
    }
}

/// Hyperband pruner - aggressive early stopping based on successive halving
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HyperbandPruner {
    /// Minimum resource (e.g., epochs)
    min_resource: usize,
    /// Maximum resource
    max_resource: usize,
    /// Reduction factor (eta)
    reduction_factor: f64,
    /// Whether to minimize
    minimize: bool,
    /// Internal history
    #[serde(skip)]
    history: TrialHistory,
    /// Budget assignments per trial
    #[serde(skip)]
    #[allow(dead_code)]
    budgets: HashMap<usize, usize>,
}

impl HyperbandPruner {
    /// Create a new Hyperband pruner
    ///
    /// # Panics
    /// Panics if `min_resource < 1`, since a resource of 0 combined with the
    /// default `reduction_factor` would never advance `compute_rung`'s loop.
    pub fn new(min_resource: usize, max_resource: usize, minimize: bool) -> Self {
        assert!(
            min_resource >= 1,
            "HyperbandPruner::new: min_resource must be >= 1, got {}",
            min_resource
        );
        Self {
            min_resource,
            max_resource,
            reduction_factor: 3.0,
            minimize,
            history: TrialHistory::new(),
            budgets: HashMap::new(),
        }
    }

    /// Set reduction factor
    ///
    /// # Panics
    /// Panics if `eta <= 1.0`, since `compute_rung`'s loop multiplies the
    /// resource by `eta` each iteration and would never terminate otherwise.
    pub fn with_reduction_factor(mut self, eta: f64) -> Self {
        assert!(
            eta > 1.0,
            "HyperbandPruner::with_reduction_factor: reduction_factor must be > 1.0, got {}",
            eta
        );
        self.reduction_factor = eta;
        self
    }

    fn compute_rung(&self, step: usize) -> usize {
        // Find which rung this step corresponds to
        let mut rung = 0;
        let mut r = self.min_resource as f64;
        while r <= self.max_resource as f64 {
            if step <= r as usize {
                break;
            }
            r *= self.reduction_factor;
            rung += 1;
        }
        rung
    }
}

impl Pruner for HyperbandPruner {
    fn should_prune(&self, _trial_id: usize, step: usize, value: f64, _history: &TrialHistory) -> bool {
        let rung = self.compute_rung(step);
        
        // Get all values at this rung
        let rung_step = (self.min_resource as f64 * self.reduction_factor.powi(rung as i32)) as usize;
        let values = self.history.get_step_values(rung_step);
        
        if values.is_empty() {
            return false;
        }

        // Keep top 1/eta trials
        let n_keep = (values.len() as f64 / self.reduction_factor).ceil() as usize;
        if n_keep == 0 {
            return false;
        }

        let mut sorted: Vec<(usize, f64)> = values.iter().enumerate().map(|(i, &v)| (i, v)).collect();
        if self.minimize {
            sorted.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        } else {
            sorted.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        }

        // Check if this trial is in the top trials
        let _top_trials: Vec<usize> = sorted.iter().take(n_keep).map(|(i, _)| *i).collect();
        
        // Get rank of current value
        let rank = if self.minimize {
            sorted.iter().position(|(_, v)| *v >= value).unwrap_or(sorted.len())
        } else {
            sorted.iter().position(|(_, v)| *v <= value).unwrap_or(sorted.len())
        };

        rank >= n_keep
    }

    fn report(&mut self, trial_id: usize, step: usize, value: f64) {
        self.history.report(trial_id, step, value);
    }
}

/// No pruning - always continue
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct NoPruner;

impl Pruner for NoPruner {
    fn should_prune(&self, _trial_id: usize, _step: usize, _value: f64, _history: &TrialHistory) -> bool {
        false
    }

    fn report(&mut self, _trial_id: usize, _step: usize, _value: f64) {}
}

/// Create a pruner from configuration
pub fn create_pruner(config: PrunerConfig, minimize: bool) -> Box<dyn Pruner> {
    match config {
        PrunerConfig::Median { n_startup_trials, n_warmup_steps } => {
            Box::new(
                MedianPruner::new(minimize)
                    .with_n_startup_trials(n_startup_trials)
                    .with_n_warmup_steps(n_warmup_steps)
            )
        }
        PrunerConfig::Percentile { percentile, n_startup_trials } => {
            Box::new(PercentilePruner {
                percentile,
                n_startup_trials,
                n_warmup_steps: 0,
                n_min_trials: 1,
                minimize,
                history: TrialHistory::new(),
            })
        }
        PrunerConfig::Hyperband { min_resource, max_resource, reduction_factor } => {
            Box::new(
                HyperbandPruner::new(min_resource, max_resource, minimize)
                    .with_reduction_factor(reduction_factor)
            )
        }
        PrunerConfig::None => {
            Box::new(NoPruner)
        }
    }
}

/// Pruner configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PrunerConfig {
    Median {
        n_startup_trials: usize,
        n_warmup_steps: usize,
    },
    Percentile {
        percentile: f64,
        n_startup_trials: usize,
    },
    Hyperband {
        min_resource: usize,
        max_resource: usize,
        reduction_factor: f64,
    },
    None,
}

impl Default for PrunerConfig {
    fn default() -> Self {
        PrunerConfig::Median {
            n_startup_trials: 5,
            n_warmup_steps: 0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_median_pruner() {
        let mut pruner = MedianPruner::new(true)
            .with_n_startup_trials(2)
            .with_n_warmup_steps(0);

        // Report some values
        pruner.report(0, 1, 1.0);
        pruner.report(1, 1, 2.0);
        pruner.report(2, 1, 3.0);

        let history = TrialHistory::new();

        // First two trials shouldn't be pruned (startup)
        assert!(!pruner.should_prune(0, 1, 1.0, &history));
        assert!(!pruner.should_prune(1, 1, 2.0, &history));

        // Trial with value > median should be pruned
        assert!(pruner.should_prune(3, 1, 3.5, &history));
        
        // Trial with value < median should not be pruned
        assert!(!pruner.should_prune(4, 1, 0.5, &history));
    }

    #[test]
    fn test_hyperband_pruner() {
        let mut pruner = HyperbandPruner::new(1, 27, true)
            .with_reduction_factor(3.0);

        // Report values at rung 0 (step 1)
        for i in 0..9 {
            pruner.report(i, 1, i as f64);
        }

        // Best 3 should survive
        let history = TrialHistory::new();
        assert!(!pruner.should_prune(0, 1, 0.0, &history)); // Best
        assert!(pruner.should_prune(8, 1, 8.0, &history));  // Worst
    }

    #[test]
    fn test_no_pruner() {
        let pruner = NoPruner;
        let history = TrialHistory::new();
        
        assert!(!pruner.should_prune(0, 100, f64::INFINITY, &history));
    }
}
