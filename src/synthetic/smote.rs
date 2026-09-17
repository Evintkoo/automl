//! SMOTE and variants

use crate::error::{AutoMLError, Result};
use crate::synthetic::{Sampler, ResampleResult, class_counts, class_indices};
use ndarray::{Array1, Array2};
use rand::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// SMOTE variant
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SMOTEVariant {
    /// Standard SMOTE
    Regular,
    /// Borderline SMOTE (type 1)
    Borderline1,
    /// Borderline SMOTE (type 2)  
    Borderline2,
}

/// SMOTE (Synthetic Minority Over-sampling Technique)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SMOTE {
    /// Number of nearest neighbors
    k_neighbors: usize,
    /// Sampling strategy (ratio of minority to majority)
    sampling_strategy: f64,
    /// Random seed
    seed: Option<u64>,
    /// Target samples per class
    target_counts: Option<HashMap<i64, usize>>,
}

impl SMOTE {
    /// Create new SMOTE sampler
    pub fn new() -> Self {
        Self {
            k_neighbors: 5,
            sampling_strategy: 1.0, // Balance classes
            seed: None,
            target_counts: None,
        }
    }

    /// Set number of neighbors
    pub fn with_k_neighbors(mut self, k: usize) -> Self {
        self.k_neighbors = k.max(1);
        self
    }

    /// Set sampling strategy (ratio)
    pub fn with_sampling_strategy(mut self, ratio: f64) -> Self {
        self.sampling_strategy = ratio.clamp(0.1, 10.0);
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

    /// Find k nearest neighbors
    fn find_neighbors(&self, point: &[f64], data: &[Vec<f64>], k: usize) -> Vec<usize> {
        let mut distances: Vec<(usize, f64)> = data.iter()
            .enumerate()
            .map(|(i, d)| (i, Self::distance(point, d)))
            .filter(|&(_, dist)| dist > 0.0) // Exclude self
            .collect();

        distances.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        distances.into_iter().take(k).map(|(i, _)| i).collect()
    }

    /// Generate synthetic sample between two points
    fn generate_sample(&self, point: &[f64], neighbor: &[f64], rng: &mut StdRng) -> Vec<f64> {
        let gap: f64 = rng.gen();
        point.iter()
            .zip(neighbor.iter())
            .map(|(&p, &n)| p + gap * (n - p))
            .collect()
    }

    /// Generate synthetic sample between two points, restricting the
    /// interpolation gap to `[0, max_gap)` instead of the full `[0, 1)`. Used
    /// by Borderline2 when interpolating toward majority-class neighbors, so
    /// the synthetic point stays closer to the minority sample and doesn't
    /// cross fully into majority territory.
    fn generate_sample_with_max_gap(
        &self,
        point: &[f64],
        neighbor: &[f64],
        max_gap: f64,
        rng: &mut StdRng,
    ) -> Vec<f64> {
        let gap: f64 = rng.gen::<f64>() * max_gap;
        point.iter()
            .zip(neighbor.iter())
            .map(|(&p, &n)| p + gap * (n - p))
            .collect()
    }
}

impl Default for SMOTE {
    fn default() -> Self {
        Self::new()
    }
}

impl Sampler for SMOTE {
    fn fit(&mut self, _x: &Array2<f64>, y: &Array1<i64>) -> Result<()> {
        let counts = class_counts(y);
        
        if counts.len() < 2 {
            return Err(AutoMLError::ValidationError(
                "Need at least 2 classes for SMOTE".to_string()
            ));
        }

        // Find majority class count
        let max_count = *counts.values().max().unwrap();

        // Calculate target counts
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
            AutoMLError::ValidationError("SMOTE not fitted".to_string())
        })?;

        let mut rng = match self.seed {
            Some(seed) => StdRng::seed_from_u64(seed),
            None => StdRng::from_entropy(),
        };

        let indices = class_indices(y);
        let counts = class_counts(y);
        let n_features = x.ncols();

        // Collect all samples and labels
        let mut all_x: Vec<Vec<f64>> = x.rows()
            .into_iter()
            .map(|row| row.iter().copied().collect())
            .collect();
        let mut all_y: Vec<i64> = y.iter().copied().collect();
        let mut n_synthetic = Vec::new();

        for (&class, &target_count) in targets {
            let current_count = counts.get(&class).copied().unwrap_or(0);
            let n_to_generate = target_count.saturating_sub(current_count);

            if n_to_generate == 0 {
                n_synthetic.push(0);
                continue;
            }

            // Get samples for this class
            let class_idx = indices.get(&class).unwrap();
            let class_samples: Vec<Vec<f64>> = class_idx.iter()
                .map(|&i| x.row(i).iter().copied().collect())
                .collect();

            let k = self.k_neighbors.min(class_samples.len() - 1).max(1);

            // Determine which samples in this class actually have a neighbor
            // (a singleton class, or a sample whose only neighbors are exact
            // duplicates of itself, would otherwise cause an infinite loop below).
            let viable_indices: Vec<usize> = (0..class_samples.len())
                .filter(|&i| !self.find_neighbors(&class_samples[i], &class_samples, k).is_empty())
                .collect();

            if viable_indices.is_empty() {
                // No sample in this class has any neighbor to pair with - the class
                // is too small/degenerate to resample. Skip it rather than looping forever.
                n_synthetic.push(0);
                continue;
            }

            // Generate synthetic samples
            let mut generated = 0;
            while generated < n_to_generate {
                // Pick random sample from the samples known to have neighbors
                let idx = viable_indices[rng.gen_range(0..viable_indices.len())];
                let sample = &class_samples[idx];

                // Find neighbors
                let neighbors = self.find_neighbors(sample, &class_samples, k);

                // Pick random neighbor
                let neighbor_idx = neighbors[rng.gen_range(0..neighbors.len())];
                let neighbor = &class_samples[neighbor_idx];

                // Generate synthetic sample
                let synthetic = self.generate_sample(sample, neighbor, &mut rng);

                all_x.push(synthetic);
                all_y.push(class);
                generated += 1;
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

        let result_y = Array1::from_vec(all_y);

        Ok(ResampleResult {
            x: result_x,
            y: result_y,
            n_synthetic,
        })
    }
}

/// Borderline SMOTE
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BorderlineSMOTE {
    /// Base SMOTE
    smote: SMOTE,
    /// Variant type
    variant: SMOTEVariant,
    /// Number of neighbors for borderline detection
    m_neighbors: usize,
}

impl BorderlineSMOTE {
    /// Create new Borderline SMOTE
    pub fn new(variant: SMOTEVariant) -> Self {
        Self {
            smote: SMOTE::new(),
            variant,
            m_neighbors: 10,
        }
    }

    /// Set k neighbors for SMOTE
    pub fn with_k_neighbors(mut self, k: usize) -> Self {
        self.smote = self.smote.with_k_neighbors(k);
        self
    }

    /// Set m neighbors for borderline detection
    pub fn with_m_neighbors(mut self, m: usize) -> Self {
        self.m_neighbors = m.max(1);
        self
    }

    /// Set random seed
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.smote = self.smote.with_seed(seed);
        self
    }

    /// Check if a point is borderline
    fn is_borderline(&self, point_idx: usize, x: &Array2<f64>, y: &Array1<i64>) -> bool {
        let point: Vec<f64> = x.row(point_idx).iter().copied().collect();
        let point_class = y[point_idx];

        // Find m nearest neighbors (including other classes)
        let all_samples: Vec<Vec<f64>> = x.rows()
            .into_iter()
            .map(|row| row.iter().copied().collect())
            .collect();

        let mut distances: Vec<(usize, f64)> = all_samples.iter()
            .enumerate()
            .filter(|&(i, _)| i != point_idx)
            .map(|(i, d)| (i, SMOTE::distance(&point, d)))
            .collect();

        distances.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

        let neighbors: Vec<usize> = distances.into_iter()
            .take(self.m_neighbors)
            .map(|(i, _)| i)
            .collect();

        // Count neighbors from different class
        let n_different = neighbors.iter()
            .filter(|&&i| y[i] != point_class)
            .count();

        // Borderline (danger) region per Han et al.: the majority of neighbors
        // (but not all of them) belong to the other class, i.e. a ratio of
        // majority-class neighbors in [0.5, 1.0). ratio == 1.0 means the point is
        // fully surrounded by the other class ("noise") and should NOT be
        // oversampled; ratio < 0.5 means the point is "safe" and should not be
        // oversampled either.
        let ratio = n_different as f64 / self.m_neighbors as f64;
        ratio >= 0.5 && ratio < 1.0
    }

    /// Find the nearest neighbors of `point_idx` that belong to a different
    /// class (used by Borderline2 to interpolate toward the majority class).
    fn majority_neighbors(
        &self,
        point_idx: usize,
        x: &Array2<f64>,
        y: &Array1<i64>,
        k: usize,
    ) -> Vec<usize> {
        let point: Vec<f64> = x.row(point_idx).iter().copied().collect();
        let point_class = y[point_idx];

        let mut distances: Vec<(usize, f64)> = x
            .rows()
            .into_iter()
            .enumerate()
            .filter(|&(i, _)| i != point_idx && y[i] != point_class)
            .map(|(i, row)| {
                let other: Vec<f64> = row.iter().copied().collect();
                (i, SMOTE::distance(&point, &other))
            })
            .collect();

        distances.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        distances.into_iter().take(k).map(|(i, _)| i).collect()
    }
}

impl Sampler for BorderlineSMOTE {
    fn fit(&mut self, x: &Array2<f64>, y: &Array1<i64>) -> Result<()> {
        self.smote.fit(x, y)
    }

    fn resample(&self, x: &Array2<f64>, y: &Array1<i64>) -> Result<ResampleResult> {
        let targets = self.smote.target_counts.as_ref().ok_or_else(|| {
            AutoMLError::ValidationError("Borderline SMOTE not fitted".to_string())
        })?;

        let mut rng = match self.smote.seed {
            Some(seed) => StdRng::seed_from_u64(seed),
            None => StdRng::from_entropy(),
        };

        let indices = class_indices(y);
        let counts = class_counts(y);
        let n_features = x.ncols();

        // Find minority classes
        let max_count = *counts.values().max().unwrap();
        let minority_classes: Vec<i64> = counts.iter()
            .filter(|(_, &c)| c < max_count)
            .map(|(&k, _)| k)
            .collect();

        // Find borderline samples for minority classes
        let mut borderline_indices: HashMap<i64, Vec<usize>> = HashMap::new();
        for &class in &minority_classes {
            let class_idx = indices.get(&class).unwrap();
            let borderline: Vec<usize> = class_idx.iter()
                .filter(|&&i| self.is_borderline(i, x, y))
                .copied()
                .collect();
            borderline_indices.insert(class, borderline);
        }

        // Collect original samples
        let mut all_x: Vec<Vec<f64>> = x.rows()
            .into_iter()
            .map(|row| row.iter().copied().collect())
            .collect();
        let mut all_y: Vec<i64> = y.iter().copied().collect();
        let mut n_synthetic = Vec::new();

        // Generate synthetic samples only from borderline samples
        for (&class, &target_count) in targets {
            let current_count = counts.get(&class).copied().unwrap_or(0);
            let n_to_generate = target_count.saturating_sub(current_count);

            if n_to_generate == 0 {
                n_synthetic.push(0);
                continue;
            }

            let class_idx = indices.get(&class).unwrap();
            let borderline = borderline_indices.get(&class)
                .map(|v| v.as_slice())
                .unwrap_or(&[]);

            // Use borderline samples if available, otherwise all samples
            let source_indices = if borderline.is_empty() {
                class_idx.as_slice()
            } else {
                borderline
            };

            let class_samples: Vec<Vec<f64>> = class_idx.iter()
                .map(|&i| x.row(i).iter().copied().collect())
                .collect();

            let k = self.smote.k_neighbors.min(class_samples.len() - 1).max(1);

            // Determine which source samples actually have a neighbor within the
            // class (a singleton class, or an all-duplicate class, would otherwise
            // cause an infinite loop below).
            let viable_source_indices: Vec<usize> = source_indices.iter()
                .copied()
                .filter(|&idx| {
                    let sample: Vec<f64> = x.row(idx).iter().copied().collect();
                    !self.smote.find_neighbors(&sample, &class_samples, k).is_empty()
                })
                .collect();

            if viable_source_indices.is_empty() {
                // No source sample in this class has any neighbor to pair with -
                // the class is too small/degenerate to resample. Skip it rather
                // than looping forever.
                n_synthetic.push(0);
                continue;
            }

            // Borderline2 additionally interpolates borderline minority samples
            // toward their nearest *majority*-class neighbors (with a smaller
            // interpolation gap), in addition to interpolating toward minority
            // neighbors like Borderline1. Split the generation budget between
            // the two directions.
            let is_borderline2 = self.variant == SMOTEVariant::Borderline2;
            let n_to_generate_majority = if is_borderline2 { n_to_generate / 2 } else { 0 };
            let n_to_generate_minority = n_to_generate - n_to_generate_majority;

            // Determine which source samples have at least one majority-class
            // (different-class) neighbor, needed only for Borderline2.
            let viable_majority_source_indices: Vec<usize> = if is_borderline2 {
                source_indices
                    .iter()
                    .copied()
                    .filter(|&idx| !self.majority_neighbors(idx, x, y, k).is_empty())
                    .collect()
            } else {
                Vec::new()
            };

            let mut generated = 0;
            while generated < n_to_generate_minority {
                let idx = viable_source_indices[rng.gen_range(0..viable_source_indices.len())];
                let sample: Vec<f64> = x.row(idx).iter().copied().collect();

                let neighbors = self.smote.find_neighbors(&sample, &class_samples, k);

                let neighbor_idx = neighbors[rng.gen_range(0..neighbors.len())];
                let neighbor = &class_samples[neighbor_idx];

                let synthetic = self.smote.generate_sample(&sample, neighbor, &mut rng);

                all_x.push(synthetic);
                all_y.push(class);
                generated += 1;
            }

            if is_borderline2 && !viable_majority_source_indices.is_empty() {
                let mut generated_majority = 0;
                while generated_majority < n_to_generate_majority {
                    let idx = viable_majority_source_indices
                        [rng.gen_range(0..viable_majority_source_indices.len())];
                    let sample: Vec<f64> = x.row(idx).iter().copied().collect();

                    let maj_neighbors = self.majority_neighbors(idx, x, y, k);
                    let neighbor_idx = maj_neighbors[rng.gen_range(0..maj_neighbors.len())];
                    let neighbor: Vec<f64> = x.row(neighbor_idx).iter().copied().collect();

                    // Smaller gap range [0, 0.5) keeps the synthetic point closer
                    // to the minority sample rather than crossing fully into
                    // majority territory.
                    let synthetic = self
                        .smote
                        .generate_sample_with_max_gap(&sample, &neighbor, 0.5, &mut rng);

                    all_x.push(synthetic);
                    all_y.push(class);
                    generated += 1;
                    generated_majority += 1;
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
        // Create imbalanced dataset: 20 majority, 5 minority
        let mut data = Vec::new();
        let mut labels = Vec::new();

        // Majority class (0) around (0, 0)
        for i in 0..20 {
            data.push((i % 5) as f64);
            data.push((i / 5) as f64);
            labels.push(0i64);
        }

        // Minority class (1) around (10, 10)
        for i in 0..5 {
            data.push(10.0 + (i % 3) as f64);
            data.push(10.0 + (i / 3) as f64);
            labels.push(1i64);
        }

        let x = Array2::from_shape_vec((25, 2), data).unwrap();
        let y = Array1::from_vec(labels);

        (x, y)
    }

    #[test]
    fn test_smote_basic() {
        let (x, y) = create_imbalanced_data();
        
        let mut smote = SMOTE::new()
            .with_k_neighbors(3)
            .with_seed(42);

        let result = smote.fit_resample(&x, &y).unwrap();

        // Check that we have more samples
        assert!(result.x.nrows() > x.nrows());
        assert!(result.y.len() > y.len());

        // Check class balance improved
        let new_counts = class_counts(&result.y);
        let count_0 = new_counts.get(&0).copied().unwrap_or(0);
        let count_1 = new_counts.get(&1).copied().unwrap_or(0);
        
        // Classes should be more balanced
        assert!(count_1 > 5); // More minority samples
    }

    #[test]
    fn test_smote_preserves_original() {
        let (x, y) = create_imbalanced_data();
        let original_rows = x.nrows();
        
        let mut smote = SMOTE::new().with_seed(42);
        let result = smote.fit_resample(&x, &y).unwrap();

        // First rows should be original data
        for i in 0..original_rows {
            for j in 0..x.ncols() {
                assert_eq!(result.x[[i, j]], x[[i, j]]);
            }
        }
    }

    #[test]
    fn test_borderline_smote() {
        let (x, y) = create_imbalanced_data();
        
        let mut bsmote = BorderlineSMOTE::new(SMOTEVariant::Borderline1)
            .with_k_neighbors(3)
            .with_m_neighbors(5)
            .with_seed(42);

        let result = bsmote.fit_resample(&x, &y).unwrap();

        // Should have generated samples
        assert!(result.x.nrows() >= x.nrows());
    }
}
