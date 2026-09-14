//! ADASYN (Adaptive Synthetic Sampling)

use crate::error::{AutoMLError, Result};
use crate::synthetic::{Sampler, ResampleResult, class_counts, class_indices};
use ndarray::{Array1, Array2};
use rand::prelude::*;
use serde::{Deserialize, Serialize};

/// ADASYN adaptive synthetic sampling
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ADASYN {
    /// Number of nearest neighbors
    k_neighbors: usize,
    /// Sampling strategy (ratio)
    sampling_strategy: f64,
    /// Random seed
    seed: Option<u64>,
    /// Imbalance threshold
    imbalance_threshold: f64,
}

impl ADASYN {
    /// Create new ADASYN sampler
    pub fn new() -> Self {
        Self {
            k_neighbors: 5,
            sampling_strategy: 1.0,
            seed: None,
            imbalance_threshold: 0.5,
        }
    }

    /// Set number of neighbors
    pub fn with_k_neighbors(mut self, k: usize) -> Self {
        self.k_neighbors = k.max(1);
        self
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

    /// Set imbalance threshold
    pub fn with_imbalance_threshold(mut self, threshold: f64) -> Self {
        self.imbalance_threshold = threshold.clamp(0.0, 1.0);
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

    /// Find k nearest neighbors from all data
    fn find_neighbors_all(&self, point: &[f64], data: &[Vec<f64>], k: usize) -> Vec<usize> {
        let mut distances: Vec<(usize, f64)> = data.iter()
            .enumerate()
            .map(|(i, d)| (i, Self::distance(point, d)))
            .filter(|&(_, dist)| dist > 0.0)
            .collect();

        distances.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        distances.into_iter().take(k).map(|(i, _)| i).collect()
    }

    /// Find k nearest neighbors from class data
    fn find_neighbors_class(&self, point: &[f64], data: &[Vec<f64>], k: usize) -> Vec<usize> {
        let mut distances: Vec<(usize, f64)> = data.iter()
            .enumerate()
            .map(|(i, d)| (i, Self::distance(point, d)))
            .filter(|&(_, dist)| dist > 0.0)
            .collect();

        distances.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        distances.into_iter().take(k).map(|(i, _)| i).collect()
    }

    /// Generate synthetic sample
    fn generate_sample(&self, point: &[f64], neighbor: &[f64], rng: &mut StdRng) -> Vec<f64> {
        let gap: f64 = rng.gen();
        point.iter()
            .zip(neighbor.iter())
            .map(|(&p, &n)| p + gap * (n - p))
            .collect()
    }
}

impl Default for ADASYN {
    fn default() -> Self {
        Self::new()
    }
}

impl Sampler for ADASYN {
    fn fit(&mut self, _x: &Array2<f64>, y: &Array1<i64>) -> Result<()> {
        let counts = class_counts(y);
        
        if counts.len() < 2 {
            return Err(AutoMLError::ValidationError(
                "Need at least 2 classes for ADASYN".to_string()
            ));
        }

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

        // Find majority and minority class
        let (&majority_class, &majority_count) = counts.iter()
            .max_by_key(|(_, &c)| c)
            .unwrap();

        let minority_classes: Vec<(i64, usize)> = counts.iter()
            .filter(|(&c, _)| c != majority_class)
            .map(|(&c, &n)| (c, n))
            .collect();

        // Check imbalance ratio
        let min_count = counts.values().copied().min().unwrap_or(0);
        let imbalance_ratio = min_count as f64 / majority_count as f64;

        if imbalance_ratio > self.imbalance_threshold {
            // Data is not imbalanced enough, return original
            return Ok(ResampleResult {
                x: x.clone(),
                y: y.clone(),
                n_synthetic: vec![0; counts.len()],
            });
        }

        // All samples for neighbor search
        let all_samples: Vec<Vec<f64>> = x.rows()
            .into_iter()
            .map(|row| row.iter().copied().collect())
            .collect();

        // Collect original samples
        let mut all_x: Vec<Vec<f64>> = all_samples.clone();
        let mut all_y: Vec<i64> = y.iter().copied().collect();
        let mut n_synthetic = Vec::new();

        // Process each minority class
        for (minority_class, minority_count) in minority_classes {
            // Number of synthetic samples to generate
            let g = ((majority_count - minority_count) as f64 * self.sampling_strategy) as usize;

            if g == 0 {
                n_synthetic.push(0);
                continue;
            }

            let minority_idx = indices.get(&minority_class).unwrap();
            let minority_samples: Vec<Vec<f64>> = minority_idx.iter()
                .map(|&i| all_samples[i].clone())
                .collect();

            // Calculate ratio (r_i) for each minority sample
            let mut ratios = Vec::with_capacity(minority_idx.len());
            let k = self.k_neighbors.min(all_samples.len() - 1);

            for &idx in minority_idx {
                let sample = &all_samples[idx];
                let neighbors = self.find_neighbors_all(sample, &all_samples, k);

                // Count majority class neighbors
                let n_majority = neighbors.iter()
                    .filter(|&&ni| y[ni] == majority_class)
                    .count();

                let ratio = n_majority as f64 / k as f64;
                ratios.push(ratio);
            }

            // Normalize ratios
            let sum_ratios: f64 = ratios.iter().sum();
            let normalized: Vec<f64> = if sum_ratios > 0.0 {
                ratios.iter().map(|&r| r / sum_ratios).collect()
            } else {
                vec![1.0 / ratios.len() as f64; ratios.len()]
            };

            // Calculate number of samples to generate for each minority sample
            let samples_per_point: Vec<usize> = normalized.iter()
                .map(|&r| (r * g as f64).round() as usize)
                .collect();

            let k_class = self.k_neighbors.min(minority_samples.len() - 1).max(1);

            // Generate synthetic samples
            let mut generated = 0;
            for (i, &n_samples) in samples_per_point.iter().enumerate() {
                let sample = &minority_samples[i];

                for _ in 0..n_samples {
                    let neighbors = self.find_neighbors_class(sample, &minority_samples, k_class);
                    
                    if neighbors.is_empty() {
                        continue;
                    }

                    let neighbor_idx = neighbors[rng.gen_range(0..neighbors.len())];
                    let neighbor = &minority_samples[neighbor_idx];

                    let synthetic = self.generate_sample(sample, neighbor, &mut rng);
                    
                    all_x.push(synthetic);
                    all_y.push(minority_class);
                    generated += 1;
                }
            }

            n_synthetic.push(generated);
        }

        // Convert to arrays
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

#[cfg(test)]
mod tests {
    use super::*;

    fn create_imbalanced_data() -> (Array2<f64>, Array1<i64>) {
        let mut data = Vec::new();
        let mut labels = Vec::new();

        // Majority class
        for i in 0..30 {
            data.push((i % 6) as f64);
            data.push((i / 6) as f64);
            labels.push(0i64);
        }

        // Minority class
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
    fn test_adasyn_basic() {
        let (x, y) = create_imbalanced_data();
        
        let mut adasyn = ADASYN::new()
            .with_k_neighbors(3)
            .with_seed(42);

        let result = adasyn.fit_resample(&x, &y).unwrap();

        // Should have more samples
        assert!(result.x.nrows() > x.nrows());

        // Check class balance improved
        let new_counts = class_counts(&result.y);
        let count_1 = new_counts.get(&1).copied().unwrap_or(0);
        
        assert!(count_1 > 5);
    }

    #[test]
    fn test_adasyn_adaptive() {
        let (x, y) = create_imbalanced_data();
        
        let mut adasyn = ADASYN::new()
            .with_k_neighbors(5)
            .with_seed(42)
            .with_imbalance_threshold(0.3);

        let result = adasyn.fit_resample(&x, &y).unwrap();

        // ADASYN should generate more samples near decision boundary
        assert!(result.n_synthetic.iter().sum::<usize>() > 0);
    }
}
