//! HyperOptX - Main hyperparameter optimizer

use crate::error::Result;
use super::{
    config::{OptimizationConfig, OptimizeDirection},
    search_space::{SearchSpace, TrialParams},
    samplers::{Sampler, create_sampler},
};
use serde::{Deserialize, Serialize};
use std::time::Instant;

/// Result of a single trial
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrialResult {
    /// Trial number
    pub trial_id: usize,
    /// Parameters used
    pub params: TrialParams,
    /// Objective value
    pub value: f64,
    /// Trial duration in seconds
    pub duration_secs: f64,
    /// Whether trial was pruned
    pub pruned: bool,
    /// Additional metrics
    pub metrics: std::collections::HashMap<String, f64>,
}

/// Study containing all trials
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Study {
    /// All trial results
    pub trials: Vec<TrialResult>,
    /// Best trial index
    pub best_trial_idx: Option<usize>,
    /// Total duration
    pub total_duration_secs: f64,
    /// Optimization direction
    pub direction: OptimizeDirection,
}

impl Study {
    /// Create a new study
    pub fn new(direction: OptimizeDirection) -> Self {
        Self {
            trials: Vec::new(),
            best_trial_idx: None,
            total_duration_secs: 0.0,
            direction,
        }
    }

    /// Get the best trial
    pub fn best_trial(&self) -> Option<&TrialResult> {
        self.best_trial_idx.map(|idx| &self.trials[idx])
    }

    /// Get the best value
    pub fn best_value(&self) -> Option<f64> {
        self.best_trial().map(|t| t.value)
    }

    /// Get the best parameters
    pub fn best_params(&self) -> Option<&TrialParams> {
        self.best_trial().map(|t| &t.params)
    }

    /// Add a trial result
    pub fn add_trial(&mut self, result: TrialResult) {
        let idx = self.trials.len();
        
        // Update best trial
        let is_better = match self.best_trial_idx {
            None => true,
            Some(best_idx) => {
                let best_val = self.trials[best_idx].value;
                match self.direction {
                    OptimizeDirection::Minimize => result.value < best_val,
                    OptimizeDirection::Maximize => result.value > best_val,
                }
            }
        };

        if is_better && !result.pruned {
            self.best_trial_idx = Some(idx);
        }

        self.trials.push(result);
    }
}

/// Main hyperparameter optimizer
pub struct HyperOptX {
    config: OptimizationConfig,
    search_space: SearchSpace,
    sampler: Box<dyn Sampler>,
    study: Study,
}

impl HyperOptX {
    /// Create a new optimizer
    pub fn new(config: OptimizationConfig, search_space: SearchSpace) -> Self {
        let minimize = matches!(config.direction, OptimizeDirection::Minimize);
        let sampler = create_sampler(config.sampler.clone(), config.random_state, minimize);
        let study = Study::new(config.direction.clone());

        Self {
            config,
            search_space,
            sampler,
            study,
        }
    }

    /// Run optimization with an objective function
    pub fn optimize<F>(&mut self, objective: F) -> Result<&Study>
    where
        F: Fn(&TrialParams) -> Result<f64> + Sync,
    {
        let start = Instant::now();
        let timeout = self.config.timeout_secs;
        let n_trials = self.config.n_trials;
        let patience = self.config.early_stopping_patience;
        
        let mut trials_without_improvement = 0;
        let mut history: Vec<(TrialParams, f64)> = Vec::new();

        for trial_id in 0..n_trials {
            // Check timeout
            if let Some(t) = timeout {
                if start.elapsed().as_secs_f64() > t {
                    if self.config.verbose {
                        println!("Timeout reached after {} trials", trial_id);
                    }
                    break;
                }
            }

            // Check early stopping
            if let Some(p) = patience {
                if trials_without_improvement >= p {
                    if self.config.verbose {
                        println!("Early stopping after {} trials without improvement", p);
                    }
                    break;
                }
            }

            let trial_start = Instant::now();

            // Sample parameters
            let params = self.sampler.sample(&self.search_space, &history);

            // Evaluate objective
            let result = match objective(&params) {
                Ok(value) => {
                    history.push((params.clone(), value));

                    let trial = TrialResult {
                        trial_id,
                        params,
                        value,
                        duration_secs: trial_start.elapsed().as_secs_f64(),
                        pruned: false,
                        metrics: std::collections::HashMap::new(),
                    };

                    // Check if this is an improvement
                    let is_improvement = match self.study.best_value() {
                        None => true,
                        Some(best) => match self.config.direction {
                            OptimizeDirection::Minimize => value < best - self.config.min_improvement,
                            OptimizeDirection::Maximize => value > best + self.config.min_improvement,
                        },
                    };

                    if is_improvement {
                        trials_without_improvement = 0;
                    } else {
                        trials_without_improvement += 1;
                    }

                    trial
                }
                Err(_e) => {
                    // Trial failed - mark as pruned with worst value
                    let worst_val = match self.config.direction {
                        OptimizeDirection::Minimize => f64::INFINITY,
                        OptimizeDirection::Maximize => f64::NEG_INFINITY,
                    };

                    // An erroring trial is treated the same as a non-improving
                    // trial for early-stopping patience, otherwise a run whose
                    // objective repeatedly errors never triggers early stopping.
                    trials_without_improvement += 1;

                    TrialResult {
                        trial_id,
                        params,
                        value: worst_val,
                        duration_secs: trial_start.elapsed().as_secs_f64(),
                        pruned: true,
                        metrics: std::collections::HashMap::new(),
                    }
                }
            };

            if self.config.verbose {
                let status = if result.pruned { "PRUNED" } else { "" };
                println!(
                    "Trial {}: value={:.6} {} (best={:.6})",
                    trial_id,
                    result.value,
                    status,
                    self.study.best_value().unwrap_or(result.value)
                );
            }

            self.study.add_trial(result);
        }

        self.study.total_duration_secs = start.elapsed().as_secs_f64();

        Ok(&self.study)
    }

    /// Get the study results
    pub fn study(&self) -> &Study {
        &self.study
    }

    /// Save study to file
    pub fn save_study(&self, path: &str) -> Result<()> {
        let json = serde_json::to_string_pretty(&self.study)?;
        std::fs::write(path, json)?;
        Ok(())
    }

    /// Load study from file
    pub fn load_study(path: &str) -> Result<Study> {
        let json = std::fs::read_to_string(path)?;
        let study: Study = serde_json::from_str(&json)?;
        Ok(study)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn quadratic_objective(params: &TrialParams) -> Result<f64> {
        let x = params.get("x").and_then(|p| p.as_float()).unwrap_or(0.0);
        let y = params.get("y").and_then(|p| p.as_float()).unwrap_or(0.0);
        Ok(x * x + y * y) // Minimum at (0, 0)
    }

    #[test]
    fn test_optimizer_creation() {
        let config = OptimizationConfig::new();
        let space = SearchSpace::new()
            .float("x", -10.0, 10.0)
            .float("y", -10.0, 10.0);

        let optimizer = HyperOptX::new(config, space);
        assert!(optimizer.study().trials.is_empty());
    }

    #[test]
    fn test_optimization() {
        let config = OptimizationConfig::new()
            .with_n_trials(20)
            .with_direction(OptimizeDirection::Minimize);

        let space = SearchSpace::new()
            .float("x", -5.0, 5.0)
            .float("y", -5.0, 5.0);

        let mut optimizer = HyperOptX::new(config, space);
        let study = optimizer.optimize(quadratic_objective).unwrap();

        assert_eq!(study.trials.len(), 20);
        assert!(study.best_value().is_some());
        
        // Best value should be reasonably close to 0 (minimum)
        assert!(study.best_value().unwrap() < 25.0); // x^2 + y^2 < 25
    }

    #[test]
    fn test_early_stopping() {
        let config = OptimizationConfig::new()
            .with_n_trials(100)
            .with_direction(OptimizeDirection::Minimize);

        let space = SearchSpace::new().float("x", 0.0, 1.0);

        // Objective that always returns same value
        let constant_objective = |_: &TrialParams| -> Result<f64> { Ok(1.0) };

        let mut optimizer = HyperOptX::new(config, space);
        let study = optimizer.optimize(constant_objective).unwrap();

        // Should stop early due to no improvement
        assert!(study.trials.len() < 100);
    }
}
