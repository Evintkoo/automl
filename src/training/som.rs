//! Self-Organizing Map (Kohonen SOM) implementation
//!
//! Unsupervised neural-network-based clustering that preserves topological structure.

use ndarray::{Array1, Array2};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

use crate::error::{AutoMLError, Result};

/// Self-Organizing Map with rectangular grid topology
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SOM {
    grid_rows: usize,
    grid_cols: usize,
    max_iter: usize,
    initial_learning_rate: f64,
    initial_radius: f64,
    random_state: Option<u64>,
    weights: Option<Array2<f64>>,
    is_fitted: bool,
}

impl SOM {
    /// Create a new SOM with the given grid dimensions
    pub fn new(grid_rows: usize, grid_cols: usize) -> Self {
        let initial_radius = grid_rows.max(grid_cols) as f64 / 2.0;
        Self {
            grid_rows,
            grid_cols,
            max_iter: 1000,
            initial_learning_rate: 0.5,
            initial_radius,
            random_state: None,
            weights: None,
            is_fitted: false,
        }
    }

    pub fn with_max_iter(mut self, max_iter: usize) -> Self {
        self.max_iter = max_iter;
        self
    }

    pub fn with_learning_rate(mut self, lr: f64) -> Self {
        self.initial_learning_rate = lr;
        self
    }

    pub fn with_random_state(mut self, seed: u64) -> Self {
        self.random_state = Some(seed);
        self
    }

    /// Fit the SOM to the data
    pub fn fit(&mut self, x: &Array2<f64>) -> Result<()> {
        let n_samples = x.nrows();
        let n_features = x.ncols();
        let n_neurons = self.grid_rows * self.grid_cols;

        if n_samples == 0 {
            return Err(AutoMLError::TrainingError("Empty dataset".to_string()));
        }

        // Compute per-feature min/max for weight initialization
        let mut feat_min = vec![f64::INFINITY; n_features];
        let mut feat_max = vec![f64::NEG_INFINITY; n_features];
        for row in x.rows() {
            for (j, &v) in row.iter().enumerate() {
                if v < feat_min[j] { feat_min[j] = v; }
                if v > feat_max[j] { feat_max[j] = v; }
            }
        }

        // Initialize weights uniformly between min and max per feature
        let mut rng_state = self.random_state.unwrap_or(42);
        let mut next_f64 = || -> f64 {
            rng_state ^= rng_state << 13;
            rng_state ^= rng_state >> 7;
            rng_state ^= rng_state << 17;
            (rng_state as f64) / (u64::MAX as f64)
        };

        let mut weights = Array2::zeros((n_neurons, n_features));
        for i in 0..n_neurons {
            for j in 0..n_features {
                let range = feat_max[j] - feat_min[j];
                weights[[i, j]] = feat_min[j] + next_f64() * range;
            }
        }

        // Time constant for radius decay
        let time_constant = if self.initial_radius > 1.0 {
            self.max_iter as f64 / self.initial_radius.ln()
        } else {
            self.max_iter as f64
        };

        // Training loop
        for t in 0..self.max_iter {
            let lr = self.initial_learning_rate * (-(t as f64) / self.max_iter as f64).exp();
            let radius = self.initial_radius * (-(t as f64) / time_constant).exp();
            let radius_sq = radius * radius;

            // Pick a random sample
            let sample_idx = (next_f64() * n_samples as f64) as usize % n_samples;
            let sample = x.row(sample_idx);

            // Find BMU (Best Matching Unit)
            let mut bmu_idx = 0;
            let mut bmu_dist = f64::INFINITY;
            for n in 0..n_neurons {
                let dist: f64 = weights.row(n).iter()
                    .zip(sample.iter())
                    .map(|(&w, &s)| { let d = w - s; d * d })
                    .sum();
                if dist < bmu_dist {
                    bmu_dist = dist;
                    bmu_idx = n;
                }
            }

            let bmu_row = bmu_idx / self.grid_cols;
            let bmu_col = bmu_idx % self.grid_cols;

            // Update neurons within radius
            for n in 0..n_neurons {
                let n_row = n / self.grid_cols;
                let n_col = n % self.grid_cols;

                let dr = n_row as f64 - bmu_row as f64;
                let dc = n_col as f64 - bmu_col as f64;
                let grid_dist_sq = dr * dr + dc * dc;

                if grid_dist_sq <= radius_sq * 4.0 {
                    // Neighborhood function: h(dist) = exp(-grid_dist² / (2 * radius²))
                    let h = (-grid_dist_sq / (2.0 * radius_sq.max(1e-10))).exp();
                    let influence = lr * h;

                    for j in 0..n_features {
                        weights[[n, j]] += influence * (sample[j] - weights[[n, j]]);
                    }
                }
            }
        }

        self.weights = Some(weights);
        self.is_fitted = true;
        Ok(())
    }

    /// Predict BMU index (flattened row * grid_cols + col) for each sample
    pub fn predict(&self, x: &Array2<f64>) -> Result<Array1<f64>> {
        let weights = self.weights.as_ref()
            .ok_or_else(|| AutoMLError::ModelNotFitted)?;

        let n_neurons = self.grid_rows * self.grid_cols;
        let predictions: Vec<f64> = (0..x.nrows())
            .into_par_iter()
            .map(|i| {
                let sample = x.row(i);
                let mut bmu_idx = 0;
                let mut bmu_dist = f64::INFINITY;
                for n in 0..n_neurons {
                    let dist: f64 = weights.row(n).iter()
                        .zip(sample.iter())
                        .map(|(&w, &s)| { let d = w - s; d * d })
                        .sum();
                    if dist < bmu_dist {
                        bmu_dist = dist;
                        bmu_idx = n;
                    }
                }
                bmu_idx as f64
            })
            .collect();

        Ok(Array1::from_vec(predictions))
    }

    /// Compute U-Matrix (average Euclidean distance to grid-adjacent neurons)
    pub fn get_umatrix(&self) -> Option<Array2<f64>> {
        let weights = self.weights.as_ref()?;
        let mut umatrix = Array2::zeros((self.grid_rows, self.grid_cols));

        for r in 0..self.grid_rows {
            for c in 0..self.grid_cols {
                let idx = r * self.grid_cols + c;
                let w = weights.row(idx);
                let mut total_dist = 0.0;
                let mut n_neighbors = 0;

                // Check 4-connected neighbors
                let neighbors: [(i32, i32); 4] = [(-1, 0), (1, 0), (0, -1), (0, 1)];
                for (dr, dc) in &neighbors {
                    let nr = r as i32 + dr;
                    let nc = c as i32 + dc;
                    if nr >= 0 && nr < self.grid_rows as i32 && nc >= 0 && nc < self.grid_cols as i32 {
                        let nidx = nr as usize * self.grid_cols + nc as usize;
                        let nw = weights.row(nidx);
                        let dist: f64 = w.iter().zip(nw.iter())
                            .map(|(&a, &b)| { let d = a - b; d * d })
                            .sum::<f64>()
                            .sqrt();
                        total_dist += dist;
                        n_neighbors += 1;
                    }
                }

                umatrix[[r, c]] = if n_neighbors > 0 { total_dist / n_neighbors as f64 } else { 0.0 };
            }
        }

        Some(umatrix)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_som_fit_and_predict() {
        // Two well-separated clusters
        let mut data = Vec::new();
        for i in 0..20 {
            // Cluster A near (0, 0)
            data.push(0.0 + (i as f64) * 0.05);
            data.push(0.0 + (i as f64) * 0.03);
        }
        for i in 0..20 {
            // Cluster B near (10, 10)
            data.push(10.0 + (i as f64) * 0.05);
            data.push(10.0 + (i as f64) * 0.03);
        }
        let x = Array2::from_shape_vec((40, 2), data).unwrap();

        let mut som = SOM::new(3, 3)
            .with_max_iter(500)
            .with_random_state(42);
        som.fit(&x).unwrap();

        assert!(som.is_fitted);

        let predictions = som.predict(&x).unwrap();
        assert_eq!(predictions.len(), 40);

        // Verify the two clusters map to different BMUs
        let cluster_a_bmus: Vec<f64> = predictions.iter().take(20).cloned().collect();
        let cluster_b_bmus: Vec<f64> = predictions.iter().skip(20).cloned().collect();

        // Most samples in cluster A should share a BMU, same for B, and they should differ
        let mode_a = mode(&cluster_a_bmus);
        let mode_b = mode(&cluster_b_bmus);
        assert_ne!(mode_a, mode_b, "Two clusters should map to different BMUs");
    }

    #[test]
    fn test_som_umatrix_shape() {
        let x = Array2::from_shape_vec((10, 3), (0..30).map(|i| i as f64).collect()).unwrap();
        let mut som = SOM::new(4, 5).with_max_iter(100).with_random_state(7);
        som.fit(&x).unwrap();

        let umatrix = som.get_umatrix().unwrap();
        assert_eq!(umatrix.shape(), &[4, 5]);
    }

    #[test]
    fn test_som_is_fitted() {
        let x = Array2::from_shape_vec((10, 2), (0..20).map(|i| i as f64).collect()).unwrap();
        let mut som = SOM::new(2, 2).with_max_iter(50).with_random_state(1);
        assert!(!som.is_fitted);
        som.fit(&x).unwrap();
        assert!(som.is_fitted);
    }

    fn mode(values: &[f64]) -> i64 {
        let mut counts = std::collections::HashMap::new();
        for &v in values {
            *counts.entry(v as i64).or_insert(0) += 1;
        }
        counts.into_iter().max_by_key(|&(_, c)| c).map(|(v, _)| v).unwrap_or(0)
    }
}
