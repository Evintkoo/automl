//! Isolation Forest anomaly detection

use crate::error::{AutoMLError, Result};
use crate::anomaly::AnomalyDetector;
use ndarray::{Array1, Array2};
use rand::prelude::*;
use serde::{Deserialize, Serialize};

/// Isolation Tree node
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum IsolationTree {
    /// Internal node with split
    Internal {
        /// Feature index for split
        feature: usize,
        /// Split threshold
        threshold: f64,
        /// Left subtree (values < threshold)
        left: Box<IsolationTree>,
        /// Right subtree (values >= threshold)
        right: Box<IsolationTree>,
    },
    /// External (leaf) node
    External {
        /// Number of samples in this node
        size: usize,
    },
}

impl IsolationTree {
    /// Build an isolation tree
    pub fn build(
        x: &Array2<f64>,
        indices: &[usize],
        height: usize,
        max_height: usize,
        rng: &mut impl Rng,
    ) -> Self {
        let n_samples = indices.len();

        // Stop if max height reached or too few samples
        if height >= max_height || n_samples <= 1 {
            return IsolationTree::External { size: n_samples };
        }

        let n_features = x.ncols();

        // Randomly select a feature
        let feature = rng.gen_range(0..n_features);

        // Get min and max values for this feature in current subset
        let values: Vec<f64> = indices.iter().map(|&i| x[[i, feature]]).collect();
        let min_val = values.iter().cloned().fold(f64::INFINITY, f64::min);
        let max_val = values.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

        // If all values are the same, create a leaf
        if (max_val - min_val).abs() < 1e-10 {
            return IsolationTree::External { size: n_samples };
        }

        // Random threshold between min and max
        let threshold = rng.gen_range(min_val..max_val);

        // Split indices
        let (left_indices, right_indices): (Vec<usize>, Vec<usize>) = indices
            .iter()
            .partition(|&&i| x[[i, feature]] < threshold);

        // If split is degenerate, create a leaf
        if left_indices.is_empty() || right_indices.is_empty() {
            return IsolationTree::External { size: n_samples };
        }

        // Recursively build subtrees
        let left = Box::new(Self::build(x, &left_indices, height + 1, max_height, rng));
        let right = Box::new(Self::build(x, &right_indices, height + 1, max_height, rng));

        IsolationTree::Internal {
            feature,
            threshold,
            left,
            right,
        }
    }

    /// Compute path length for a sample
    pub fn path_length(&self, sample: &[f64], current_height: usize) -> f64 {
        match self {
            IsolationTree::External { size } => {
                current_height as f64 + Self::c(*size)
            }
            IsolationTree::Internal {
                feature,
                threshold,
                left,
                right,
            } => {
                if sample[*feature] < *threshold {
                    left.path_length(sample, current_height + 1)
                } else {
                    right.path_length(sample, current_height + 1)
                }
            }
        }
    }

    /// Average path length of unsuccessful search in BST
    /// c(n) = 2 * H(n-1) - 2(n-1)/n for n > 2
    /// where H(i) is the harmonic number
    fn c(n: usize) -> f64 {
        if n <= 1 {
            0.0
        } else if n == 2 {
            1.0
        } else {
            let n_f = n as f64;
            2.0 * (n_f - 1.0).ln() + 0.5772156649 - 2.0 * (n_f - 1.0) / n_f
        }
    }
}

/// Isolation Forest anomaly detector
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IsolationForest {
    /// Number of trees
    n_estimators: usize,
    /// Maximum samples per tree
    max_samples: usize,
    /// Contamination ratio (expected proportion of outliers)
    contamination: f64,
    /// Random seed
    seed: Option<u64>,
    /// Fitted trees
    trees: Option<Vec<IsolationTree>>,
    /// Decision threshold
    threshold: Option<f64>,
    /// Number of samples used for fitting
    n_samples: Option<usize>,
}

impl IsolationForest {
    /// Create new Isolation Forest
    pub fn new() -> Self {
        Self {
            n_estimators: 100,
            max_samples: 256,
            contamination: 0.1,
            seed: None,
            trees: None,
            threshold: None,
            n_samples: None,
        }
    }

    /// Set number of trees
    pub fn with_n_estimators(mut self, n: usize) -> Self {
        self.n_estimators = n.max(1);
        self
    }

    /// Set maximum samples per tree
    pub fn with_max_samples(mut self, n: usize) -> Self {
        self.max_samples = n.max(1);
        self
    }

    /// Set contamination ratio
    pub fn with_contamination(mut self, c: f64) -> Self {
        self.contamination = c.clamp(0.0, 0.5);
        self
    }

    /// Set random seed
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = Some(seed);
        self
    }

    /// Compute anomaly score for samples
    /// Score is between 0 and 1, higher means more anomalous
    fn compute_scores(&self, x: &Array2<f64>) -> Result<Array1<f64>> {
        let trees = self.trees.as_ref().ok_or_else(|| {
            AutoMLError::ValidationError("Model not fitted".to_string())
        })?;

        let n_samples_fit = self.n_samples.unwrap_or(256);
        let c_n = IsolationTree::c(n_samples_fit);

        let scores: Vec<f64> = x
            .rows()
            .into_iter()
            .map(|row| {
                let sample: Vec<f64> = row.iter().copied().collect();
                
                // Average path length across all trees
                let avg_path_length: f64 = trees
                    .iter()
                    .map(|tree| tree.path_length(&sample, 0))
                    .sum::<f64>()
                    / trees.len() as f64;

                // Convert to anomaly score: s(x, n) = 2^(-E[h(x)] / c(n))
                2.0_f64.powf(-avg_path_length / c_n)
            })
            .collect();

        Ok(Array1::from_vec(scores))
    }
}

impl Default for IsolationForest {
    fn default() -> Self {
        Self::new()
    }
}

impl AnomalyDetector for IsolationForest {
    fn fit(&mut self, x: &Array2<f64>) -> Result<()> {
        let n_samples = x.nrows();
        let samples_per_tree = self.max_samples.min(n_samples);

        let mut rng = match self.seed {
            Some(seed) => StdRng::seed_from_u64(seed),
            None => StdRng::from_entropy(),
        };

        // Maximum tree height
        let max_height = (samples_per_tree as f64).log2().ceil() as usize;

        // Build trees
        let mut trees = Vec::with_capacity(self.n_estimators);

        for _ in 0..self.n_estimators {
            // Sample indices with replacement
            let indices: Vec<usize> = (0..samples_per_tree)
                .map(|_| rng.gen_range(0..n_samples))
                .collect();

            let tree = IsolationTree::build(x, &indices, 0, max_height, &mut rng);
            trees.push(tree);
        }

        self.trees = Some(trees);
        self.n_samples = Some(samples_per_tree);

        // Compute threshold based on contamination
        let scores = self.compute_scores(x)?;
        let mut sorted_scores: Vec<f64> = scores.iter().copied().collect();
        sorted_scores.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));

        let threshold_idx = ((self.contamination * n_samples as f64) as usize).min(n_samples - 1);
        self.threshold = Some(sorted_scores[threshold_idx]);

        Ok(())
    }

    fn score_samples(&self, x: &Array2<f64>) -> Result<Array1<f64>> {
        self.compute_scores(x)
    }

    fn predict(&self, x: &Array2<f64>) -> Result<Array1<i32>> {
        let scores = self.score_samples(x)?;
        let threshold = self.threshold.unwrap_or(0.5);

        let labels: Vec<i32> = scores
            .iter()
            .map(|&s| if s >= threshold { -1 } else { 1 })
            .collect();

        Ok(Array1::from_vec(labels))
    }

    fn threshold(&self) -> f64 {
        self.threshold.unwrap_or(0.5)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_isolation_forest_basic() {
        // Normal data cluster: 50 points with 2 features each
        let mut data = Vec::new();
        for i in 0..50 {
            data.push((i % 10) as f64);
            data.push(((i % 10) + 1) as f64);
        }
        // Add 2 outliers
        data.extend_from_slice(&[100.0, 100.0]);
        data.extend_from_slice(&[-50.0, -50.0]);

        let x = Array2::from_shape_vec((52, 2), data).unwrap();

        let mut iforest = IsolationForest::new()
            .with_n_estimators(50)
            .with_contamination(0.05)
            .with_seed(42);

        iforest.fit(&x).unwrap();

        let scores = iforest.score_samples(&x).unwrap();
        let labels = iforest.predict(&x).unwrap();

        // Outliers should have higher scores
        assert!(scores[50] > scores[0]); // [100, 100] should be more anomalous
        assert!(scores[51] > scores[0]); // [-50, -50] should be more anomalous

        // Should have some anomalies
        let n_anomalies = labels.iter().filter(|&&l| l == -1).count();
        assert!(n_anomalies > 0);
    }

    #[test]
    fn test_isolation_tree_path_length() {
        let x = Array2::from_shape_vec(
            (10, 2),
            vec![
                1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 4.0, 4.0, 5.0, 5.0,
                6.0, 6.0, 7.0, 7.0, 8.0, 8.0, 9.0, 9.0, 10.0, 10.0,
            ],
        ).unwrap();

        let indices: Vec<usize> = (0..10).collect();
        let mut rng = StdRng::seed_from_u64(42);
        let tree = IsolationTree::build(&x, &indices, 0, 10, &mut rng);

        // Path length should be positive
        let path = tree.path_length(&[5.0, 5.0], 0);
        assert!(path > 0.0);
    }
}
