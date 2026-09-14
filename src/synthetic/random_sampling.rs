//! Random sampling methods

use crate::error::{AutoMLError, Result};
use crate::synthetic::{Sampler, ResampleResult, class_counts, class_indices};
use ndarray::{Array1, Array2};
use rand::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Random oversampler (duplicates minority samples)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RandomOverSampler {
    /// Sampling strategy (ratio)
    sampling_strategy: f64,
    /// Random seed
    seed: Option<u64>,
    /// Target counts
    target_counts: Option<HashMap<i64, usize>>,
}

impl RandomOverSampler {
    /// Create new random oversampler
    pub fn new() -> Self {
        Self {
            sampling_strategy: 1.0,
            seed: None,
            target_counts: None,
        }
    }

    /// Set sampling strategy
    pub fn with_sampling_strategy(mut self, ratio: f64) -> Self {
        self.sampling_strategy = ratio.clamp(0.1, 10.0);
        self
    }

    /// Set random seed
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = Some(seed);
        self
    }
}

impl Default for RandomOverSampler {
    fn default() -> Self {
        Self::new()
    }
}

impl Sampler for RandomOverSampler {
    fn fit(&mut self, _x: &Array2<f64>, y: &Array1<i64>) -> Result<()> {
        let counts = class_counts(y);
        let max_count = *counts.values().max().unwrap();

        let mut targets = HashMap::new();
        for (&class, &count) in &counts {
            let target = (max_count as f64 * self.sampling_strategy) as usize;
            targets.insert(class, target.max(count));
        }

        self.target_counts = Some(targets);
        Ok(())
    }

    fn resample(&self, x: &Array2<f64>, y: &Array1<i64>) -> Result<ResampleResult> {
        let targets = self.target_counts.as_ref().ok_or_else(|| {
            AutoMLError::ValidationError("Sampler not fitted".to_string())
        })?;

        let mut rng = match self.seed {
            Some(seed) => StdRng::seed_from_u64(seed),
            None => StdRng::from_entropy(),
        };

        let indices = class_indices(y);
        let counts = class_counts(y);
        let n_features = x.ncols();

        let mut all_x: Vec<Vec<f64>> = x.rows()
            .into_iter()
            .map(|row| row.iter().copied().collect())
            .collect();
        let mut all_y: Vec<i64> = y.iter().copied().collect();
        let mut n_synthetic = Vec::new();

        for (&class, &target_count) in targets {
            let current_count = counts.get(&class).copied().unwrap_or(0);
            let n_to_add = target_count.saturating_sub(current_count);

            if n_to_add == 0 {
                n_synthetic.push(0);
                continue;
            }

            let class_idx = indices.get(&class).unwrap();

            for _ in 0..n_to_add {
                let idx = class_idx[rng.gen_range(0..class_idx.len())];
                let sample: Vec<f64> = x.row(idx).iter().copied().collect();
                all_x.push(sample);
                all_y.push(class);
            }

            n_synthetic.push(n_to_add);
        }

        let n_samples = all_x.len();
        let mut result_x = Array2::zeros((n_samples, n_features));
        for (i, sample) in all_x.iter().enumerate() {
            for (j, &val) in sample.iter().enumerate() {
                result_x[[i, j]] = val;
            }
        }

        Ok(ResampleResult {
            x: result_x,
            y: Array1::from_vec(all_y),
            n_synthetic,
        })
    }
}

/// Random undersampler (removes majority samples)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RandomUnderSampler {
    /// Sampling strategy (ratio of minority to majority after sampling)
    sampling_strategy: f64,
    /// Random seed
    seed: Option<u64>,
    /// Replacement (sample with replacement)
    replacement: bool,
}

impl RandomUnderSampler {
    /// Create new random undersampler
    pub fn new() -> Self {
        Self {
            sampling_strategy: 1.0,
            seed: None,
            replacement: false,
        }
    }

    /// Set sampling strategy
    pub fn with_sampling_strategy(mut self, ratio: f64) -> Self {
        self.sampling_strategy = ratio.clamp(0.1, 10.0);
        self
    }

    /// Set random seed
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = Some(seed);
        self
    }

    /// Set replacement
    pub fn with_replacement(mut self, replacement: bool) -> Self {
        self.replacement = replacement;
        self
    }
}

impl Default for RandomUnderSampler {
    fn default() -> Self {
        Self::new()
    }
}

impl Sampler for RandomUnderSampler {
    fn fit(&mut self, _x: &Array2<f64>, _y: &Array1<i64>) -> Result<()> {
        Ok(())
    }

    fn resample(&self, x: &Array2<f64>, y: &Array1<i64>) -> Result<ResampleResult> {
        let mut rng = match self.seed {
            Some(seed) => StdRng::seed_from_u64(seed),
            None => StdRng::from_entropy(),
        };

        let counts = class_counts(y);
        let indices = class_indices(y);
        let n_features = x.ncols();

        // Find target count (minimum class count)
        let min_count = *counts.values().min().unwrap();
        let target_count = (min_count as f64 / self.sampling_strategy) as usize;

        let mut selected_indices: Vec<usize> = Vec::new();

        for (&class, class_idx) in &indices {
            let count = counts.get(&class).copied().unwrap_or(0);
            let n_to_keep = target_count.min(count);

            if self.replacement {
                // Sample with replacement
                for _ in 0..n_to_keep {
                    let idx = class_idx[rng.gen_range(0..class_idx.len())];
                    selected_indices.push(idx);
                }
            } else {
                // Sample without replacement
                let mut shuffled = class_idx.clone();
                shuffled.shuffle(&mut rng);
                selected_indices.extend(shuffled.into_iter().take(n_to_keep));
            }
        }

        // Sort to maintain some order
        selected_indices.sort();

        let n_samples = selected_indices.len();
        let mut result_x = Array2::zeros((n_samples, n_features));
        let mut result_y = Vec::with_capacity(n_samples);

        for (i, &idx) in selected_indices.iter().enumerate() {
            for j in 0..n_features {
                result_x[[i, j]] = x[[idx, j]];
            }
            result_y.push(y[idx]);
        }

        Ok(ResampleResult {
            x: result_x,
            y: Array1::from_vec(result_y),
            n_synthetic: vec![0; counts.len()],
        })
    }
}

/// NearMiss undersampling (keeps samples nearest to minority class)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NearMiss {
    /// Version (1, 2, or 3)
    version: usize,
    /// Number of neighbors
    n_neighbors: usize,
    /// Random seed
    seed: Option<u64>,
}

impl NearMiss {
    /// Create new NearMiss undersampler
    pub fn new(version: usize) -> Self {
        Self {
            version: version.clamp(1, 3),
            n_neighbors: 3,
            seed: None,
        }
    }

    /// Set number of neighbors
    pub fn with_n_neighbors(mut self, n: usize) -> Self {
        self.n_neighbors = n.max(1);
        self
    }

    /// Set random seed
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = Some(seed);
        self
    }

    /// Euclidean distance
    fn distance(a: &[f64], b: &[f64]) -> f64 {
        a.iter()
            .zip(b.iter())
            .map(|(ai, bi)| (ai - bi).powi(2))
            .sum::<f64>()
            .sqrt()
    }
}

impl Default for NearMiss {
    fn default() -> Self {
        Self::new(1)
    }
}

impl Sampler for NearMiss {
    fn fit(&mut self, _x: &Array2<f64>, _y: &Array1<i64>) -> Result<()> {
        Ok(())
    }

    fn resample(&self, x: &Array2<f64>, y: &Array1<i64>) -> Result<ResampleResult> {
        let counts = class_counts(y);
        let indices = class_indices(y);
        let n_features = x.ncols();

        // Find majority and minority classes
        let (&majority_class, _) = counts.iter()
            .max_by_key(|(_, &c)| c)
            .unwrap();

        let min_count = *counts.values().min().unwrap();
        let majority_idx = indices.get(&majority_class).unwrap();

        // Get all samples as vectors
        let all_samples: Vec<Vec<f64>> = x.rows()
            .into_iter()
            .map(|row| row.iter().copied().collect())
            .collect();

        // Get minority samples
        let minority_samples: Vec<&Vec<f64>> = indices.iter()
            .filter(|(&c, _)| c != majority_class)
            .flat_map(|(_, idx)| idx.iter().map(|&i| &all_samples[i]))
            .collect();

        // Score majority samples based on version
        let mut majority_scores: Vec<(usize, f64)> = majority_idx.iter()
            .map(|&idx| {
                let sample = &all_samples[idx];
                
                // Calculate distances to all minority samples
                let mut distances: Vec<f64> = minority_samples.iter()
                    .map(|minority| Self::distance(sample, minority))
                    .collect();
                distances.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

                let score = match self.version {
                    1 => {
                        // Average distance to k nearest minority samples
                        let k = self.n_neighbors.min(distances.len());
                        distances[..k].iter().sum::<f64>() / k as f64
                    }
                    2 => {
                        // Average distance to k farthest minority samples
                        let k = self.n_neighbors.min(distances.len());
                        let n = distances.len();
                        distances[n-k..].iter().sum::<f64>() / k as f64
                    }
                    3 => {
                        // Average distance to k nearest minority samples
                        // but only considering minority samples with majority neighbors
                        let k = self.n_neighbors.min(distances.len());
                        distances[..k].iter().sum::<f64>() / k as f64
                    }
                    _ => distances.iter().sum::<f64>() / distances.len() as f64,
                };

                (idx, score)
            })
            .collect();

        // Sort by score (ascending for version 1, descending for version 2)
        match self.version {
            2 => majority_scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal)),
            _ => majority_scores.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal)),
        }

        // Select top samples
        let selected_majority: Vec<usize> = majority_scores.into_iter()
            .take(min_count)
            .map(|(idx, _)| idx)
            .collect();

        // Collect all selected indices
        let mut selected_indices: Vec<usize> = indices.iter()
            .filter(|(&c, _)| c != majority_class)
            .flat_map(|(_, idx)| idx.iter().copied())
            .collect();
        selected_indices.extend(selected_majority);
        selected_indices.sort();

        let n_samples = selected_indices.len();
        let mut result_x = Array2::zeros((n_samples, n_features));
        let mut result_y = Vec::with_capacity(n_samples);

        for (i, &idx) in selected_indices.iter().enumerate() {
            for j in 0..n_features {
                result_x[[i, j]] = x[[idx, j]];
            }
            result_y.push(y[idx]);
        }

        Ok(ResampleResult {
            x: result_x,
            y: Array1::from_vec(result_y),
            n_synthetic: vec![0; counts.len()],
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_imbalanced_data() -> (Array2<f64>, Array1<i64>) {
        let mut data = Vec::new();
        let mut labels = Vec::new();

        for i in 0..30 {
            data.push((i % 6) as f64);
            data.push((i / 6) as f64);
            labels.push(0i64);
        }

        for i in 0..5 {
            data.push(10.0 + (i % 3) as f64);
            data.push(10.0 + (i / 3) as f64);
            labels.push(1i64);
        }

        let x = Array2::from_shape_vec((35, 2), data).unwrap();
        let y = Array1::from_vec(labels);

        (x, y)
    }

    #[test]
    fn test_random_oversampler() {
        let (x, y) = create_imbalanced_data();
        
        let mut sampler = RandomOverSampler::new().with_seed(42);
        let result = sampler.fit_resample(&x, &y).unwrap();

        assert!(result.x.nrows() > x.nrows());
        
        let new_counts = class_counts(&result.y);
        let count_0 = new_counts.get(&0).copied().unwrap_or(0);
        let count_1 = new_counts.get(&1).copied().unwrap_or(0);
        
        // Should be balanced
        assert_eq!(count_0, count_1);
    }

    #[test]
    fn test_random_undersampler() {
        let (x, y) = create_imbalanced_data();
        
        let mut sampler = RandomUnderSampler::new().with_seed(42);
        let result = sampler.fit_resample(&x, &y).unwrap();

        assert!(result.x.nrows() < x.nrows());
        
        let new_counts = class_counts(&result.y);
        let count_0 = new_counts.get(&0).copied().unwrap_or(0);
        let count_1 = new_counts.get(&1).copied().unwrap_or(0);
        
        // Should be balanced (both equal to minority count)
        assert_eq!(count_0, count_1);
        assert_eq!(count_1, 5);
    }

    #[test]
    fn test_nearmiss() {
        let (x, y) = create_imbalanced_data();
        
        let mut sampler = NearMiss::new(1).with_n_neighbors(3);
        let result = sampler.fit_resample(&x, &y).unwrap();

        assert!(result.x.nrows() < x.nrows());
        
        let new_counts = class_counts(&result.y);
        let count_0 = new_counts.get(&0).copied().unwrap_or(0);
        let count_1 = new_counts.get(&1).copied().unwrap_or(0);
        
        assert_eq!(count_0, count_1);
    }
}
