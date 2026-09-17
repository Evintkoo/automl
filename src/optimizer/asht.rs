//! ASHT - Adaptive Successive Halving with Time budget

use crate::error::Result;
use super::search_space::{SearchSpace, TrialParams};
use super::samplers::{Sampler, create_sampler_from_config, SamplerConfig};
use serde::{Deserialize, Serialize};
use std::time::Instant;

/// ASHT configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ASHTConfig {
    /// Minimum resource per trial (e.g., epochs)
    pub min_resource: usize,
    /// Maximum resource per trial
    pub max_resource: usize,
    /// Reduction factor (eta) - typically 3
    pub reduction_factor: f64,
    /// Maximum number of trials
    pub max_trials: usize,
    /// Time budget in seconds
    pub time_budget_secs: Option<f64>,
    /// Whether to minimize objective
    pub minimize: bool,
    /// Number of parallel trials
    pub n_parallel: usize,
    /// Sampler configuration
    pub sampler: SamplerConfig,
    /// Random state
    pub random_state: Option<u64>,
    /// Verbose output
    pub verbose: bool,
}

impl Default for ASHTConfig {
    fn default() -> Self {
        Self {
            min_resource: 1,
            max_resource: 81,
            reduction_factor: 3.0,
            max_trials: 100,
            time_budget_secs: None,
            minimize: true,
            n_parallel: 1,
            sampler: SamplerConfig::TPE { n_startup_trials: 10 },
            random_state: None,
            verbose: false,
        }
    }
}

/// A single bracket in successive halving
#[derive(Debug, Clone)]
struct Bracket {
    /// Bracket index
    #[allow(dead_code)]
    s: usize,
    /// Number of configurations
    #[allow(dead_code)]
    n: usize,
    /// Starting resource
    r: usize,
    /// Trials in this bracket
    trials: Vec<BracketTrial>,
    /// Current rung (promotion level)
    rung: usize,
}

#[derive(Debug, Clone)]
struct BracketTrial {
    trial_id: usize,
    params: TrialParams,
    best_value: f64,
    current_resource: usize,
    is_complete: bool,
    is_promoted: bool,
}

/// Result of ASHT optimization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ASHTResult {
    /// Best parameters found
    pub best_params: TrialParams,
    /// Best objective value
    pub best_value: f64,
    /// Total trials run
    pub n_trials: usize,
    /// Total time in seconds
    pub total_time_secs: f64,
    /// All trial results
    pub all_trials: Vec<TrialResult>,
}

/// Individual trial result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrialResult {
    pub trial_id: usize,
    pub params: TrialParams,
    pub value: f64,
    pub resource: usize,
    pub bracket: usize,
    pub rung: usize,
}

/// ASHT - Adaptive Successive Halving with Time budget
pub struct ASHT {
    config: ASHTConfig,
    search_space: SearchSpace,
    sampler: Box<dyn Sampler>,
    brackets: Vec<Bracket>,
    trial_counter: usize,
    best_params: Option<TrialParams>,
    best_value: f64,
    all_results: Vec<TrialResult>,
    history: Vec<(TrialParams, f64)>,
}

impl ASHT {
    /// Create a new ASHT optimizer
    ///
    /// # Panics
    /// Panics if `config.min_resource < 1` or `config.reduction_factor <= 1.0`.
    /// Either condition makes `compute_s_max`'s `ln` ratio diverge (or the
    /// subsequent bracket sizing attempt a near-unbounded allocation).
    pub fn new(config: ASHTConfig, search_space: SearchSpace) -> Self {
        assert!(
            config.min_resource >= 1,
            "ASHT::new: config.min_resource must be >= 1, got {}",
            config.min_resource
        );
        assert!(
            config.reduction_factor > 1.0,
            "ASHT::new: config.reduction_factor must be > 1.0, got {}",
            config.reduction_factor
        );

        let sampler =
            create_sampler_from_config(config.sampler.clone(), config.random_state, config.minimize);
        let best_value = if config.minimize { f64::INFINITY } else { f64::NEG_INFINITY };

        Self {
            config,
            search_space,
            sampler,
            brackets: Vec::new(),
            trial_counter: 0,
            best_params: None,
            best_value,
            all_results: Vec::new(),
            history: Vec::new(),
        }
    }

    /// Compute the number of brackets (s_max + 1)
    fn compute_s_max(&self) -> usize {
        let eta = self.config.reduction_factor;
        let max_r = self.config.max_resource as f64;
        let min_r = self.config.min_resource as f64;
        
        ((max_r / min_r).ln() / eta.ln()).floor() as usize
    }

    /// Initialize brackets for a round of successive halving
    fn init_brackets(&mut self) {
        let s_max = self.compute_s_max();
        let eta = self.config.reduction_factor;
        let max_r = self.config.max_resource;
        
        self.brackets.clear();

        for s in (0..=s_max).rev() {
            // Number of configurations
            let n = ((s_max + 1) as f64 * eta.powi(s as i32) / (s + 1) as f64).ceil() as usize;
            // Starting resource.
            //
            // NOTE: `eta.powi(s as i32) as usize` truncates the f64 divisor to
            // an integer *before* the division happens (in Rust, `as` binds
            // tighter than `/`), which silently loses precision whenever
            // `eta` is non-integer (e.g. eta=2.5 at s=1 truncates the divisor
            // to 2 instead of dividing by 2.5). Compute entirely in f64 and
            // only round to an integer resource count at the very end.
            let r = ((max_r as f64) / eta.powi(s as i32)).round() as usize;
            let r = r.max(self.config.min_resource);

            // Sample configurations
            let trials: Vec<BracketTrial> = (0..n)
                .map(|_| {
                    let params = self.sampler.sample(&self.search_space, &self.history);
                    let trial_id = self.trial_counter;
                    self.trial_counter += 1;
                    
                    BracketTrial {
                        trial_id,
                        params,
                        best_value: if self.config.minimize { f64::INFINITY } else { f64::NEG_INFINITY },
                        current_resource: 0,
                        is_complete: false,
                        is_promoted: false,
                    }
                })
                .collect();

            self.brackets.push(Bracket {
                s,
                n,
                r,
                trials,
                rung: 0,
            });
        }
    }

    /// Run optimization with objective function
    pub fn optimize<F>(&mut self, objective: F) -> Result<ASHTResult>
    where
        F: Fn(&TrialParams, usize) -> Result<f64> + Sync,
    {
        let start = Instant::now();
        let time_budget = self.config.time_budget_secs;
        let max_trials = self.config.max_trials;

        // Initialize brackets
        self.init_brackets();

        // Main optimization loop
        while self.trial_counter < max_trials {
            // Check time budget
            if let Some(budget) = time_budget {
                if start.elapsed().as_secs_f64() > budget {
                    if self.config.verbose {
                        println!("Time budget exhausted");
                    }
                    break;
                }
            }

            // Check if all brackets are complete
            let all_complete = self.brackets.iter().all(|b| {
                b.trials.iter().all(|t| t.is_complete)
            });

            if all_complete {
                // Re-initialize brackets for another round
                self.init_brackets();
            }

            // Process each bracket
            for bracket_idx in 0..self.brackets.len() {
                self.process_bracket(bracket_idx, &objective)?;
            }
        }

        // Collect results
        let result = ASHTResult {
            best_params: self.best_params.clone().unwrap_or_default(),
            best_value: self.best_value,
            n_trials: self.trial_counter,
            total_time_secs: start.elapsed().as_secs_f64(),
            all_trials: self.all_results.clone(),
        };

        Ok(result)
    }

    fn process_bracket<F>(&mut self, bracket_idx: usize, objective: &F) -> Result<()>
    where
        F: Fn(&TrialParams, usize) -> Result<f64> + Sync,
    {
        let eta = self.config.reduction_factor;
        
        // Extract values we need without holding borrow
        let (rung, resource, trials_to_run) = {
            let bracket = &self.brackets[bracket_idx];
            let rung = bracket.rung;
            let resource = (bracket.r as f64 * eta.powi(rung as i32)) as usize;
            let resource = resource.min(self.config.max_resource);

            // Get trials that need evaluation at this rung
            let trials_to_run: Vec<usize> = bracket.trials
                .iter()
                .enumerate()
                .filter(|(_, t)| !t.is_complete && t.current_resource < resource)
                .map(|(i, _)| i)
                .collect();
            
            (rung, resource, trials_to_run)
        };

        if trials_to_run.is_empty() {
            // Promote best trials to next rung
            self.promote_bracket(bracket_idx);
            return Ok(());
        }

        // Evaluate trials
        for trial_idx in trials_to_run {
            // Get params and trial_id first (only read)
            let (params, trial_id) = {
                let trial = &self.brackets[bracket_idx].trials[trial_idx];
                (trial.params.clone(), trial.trial_id)
            };

            match objective(&params, resource) {
                Ok(value) => {
                    // Update trial - get mutable reference now
                    {
                        let trial = &mut self.brackets[bracket_idx].trials[trial_idx];
                        trial.current_resource = resource;
                        
                        let is_better = if self.config.minimize {
                            value < trial.best_value
                        } else {
                            value > trial.best_value
                        };
                        
                        if is_better {
                            trial.best_value = value;
                        }
                    }

                    // Update global best
                    let global_better = if self.config.minimize {
                        value < self.best_value
                    } else {
                        value > self.best_value
                    };

                    if global_better {
                        self.best_value = value;
                        self.best_params = Some(params.clone());
                    }

                    // Record result
                    self.all_results.push(TrialResult {
                        trial_id,
                        params: params.clone(),
                        value,
                        resource,
                        bracket: bracket_idx,
                        rung,
                    });

                    // Add to history for sampler
                    self.history.push((params, value));

                    if self.config.verbose {
                        println!(
                            "Bracket {} Rung {} Trial {}: value={:.6} resource={}",
                            bracket_idx, rung, trial_id, value, resource
                        );
                    }
                }
                Err(e) => {
                    // Mark trial as failed/complete
                    let trial = &mut self.brackets[bracket_idx].trials[trial_idx];
                    trial.is_complete = true;
                    
                    if self.config.verbose {
                        println!("Trial {} failed: {:?}", trial_id, e);
                    }
                }
            }
        }

        Ok(())
    }

    fn promote_bracket(&mut self, bracket_idx: usize) {
        let eta = self.config.reduction_factor;
        let minimize = self.config.minimize;
        let max_rung = self.compute_s_max();
        
        let bracket = &mut self.brackets[bracket_idx];
        
        // Sort trials by value
        let mut indices: Vec<usize> = (0..bracket.trials.len())
            .filter(|&i| !bracket.trials[i].is_complete)
            .collect();

        if indices.is_empty() {
            return;
        }

        if minimize {
            indices.sort_by(|&a, &b| {
                bracket.trials[a].best_value
                    .partial_cmp(&bracket.trials[b].best_value)
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
        } else {
            indices.sort_by(|&a, &b| {
                bracket.trials[b].best_value
                    .partial_cmp(&bracket.trials[a].best_value)
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
        }

        // Keep top 1/eta trials
        let n_keep = (indices.len() as f64 / eta).floor() as usize;
        let n_keep = n_keep.max(1);

        // Mark non-promoted trials as complete
        for (i, &idx) in indices.iter().enumerate() {
            if i < n_keep {
                bracket.trials[idx].is_promoted = true;
            } else {
                bracket.trials[idx].is_complete = true;
            }
        }

        // Check if bracket is complete
        if bracket.rung >= max_rung || n_keep == 1 {
            // Bracket is complete
            for trial in &mut bracket.trials {
                trial.is_complete = true;
            }
        } else {
            bracket.rung += 1;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn simple_objective(params: &TrialParams, resource: usize) -> Result<f64> {
        let x = params.get("x").and_then(|p| p.as_float()).unwrap_or(0.0);
        // Objective improves with more resource
        Ok(x * x / (resource as f64 + 1.0))
    }

    #[test]
    fn test_asht_creation() {
        let config = ASHTConfig::default();
        let space = SearchSpace::new().float("x", -5.0, 5.0);
        let asht = ASHT::new(config, space);
        
        assert_eq!(asht.trial_counter, 0);
    }

    #[test]
    fn test_asht_optimization() {
        let config = ASHTConfig {
            min_resource: 1,
            max_resource: 9,
            reduction_factor: 3.0,
            max_trials: 20,
            minimize: true,
            verbose: false,
            ..Default::default()
        };

        let space = SearchSpace::new().float("x", -5.0, 5.0);
        let mut asht = ASHT::new(config, space);

        let result = asht.optimize(simple_objective).unwrap();
        
        assert!(result.n_trials > 0);
        // Best value should be reasonably small
        assert!(result.best_value < 25.0);
    }

    #[test]
    fn test_s_max_computation() {
        let config = ASHTConfig {
            min_resource: 1,
            max_resource: 81,
            reduction_factor: 3.0,
            ..Default::default()
        };
        let space = SearchSpace::new().float("x", -1.0, 1.0);
        let asht = ASHT::new(config, space);
        
        // s_max = floor(log_3(81/1)) = floor(4) = 4
        assert_eq!(asht.compute_s_max(), 4);
    }
}
