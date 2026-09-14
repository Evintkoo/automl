//! Clustering algorithms: KMeans and DBSCAN
//!
//! These are unsupervised models — they take X only (no y labels).
//! They implement `fit()` and `predict()` to assign cluster labels.

use crate::error::{AutoMLError, Result};
use ndarray::{Array1, Array2};
use rand::SeedableRng;
use rand::RngCore;
use rand_chacha::ChaCha8Rng;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

// ═══════════════════════════════════════════════════════════════════════════
//  K-Means Clustering
// ═══════════════════════════════════════════════════════════════════════════

/// K-Means clustering with k-means++ initialization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KMeans {
    pub n_clusters: usize,
    pub max_iter: usize,
    pub tol: f64,
    pub random_state: Option<u64>,
    /// Fitted cluster centroids (n_clusters × n_features)
    centroids: Option<Array2<f64>>,
    /// Cluster labels assigned during fit
    pub labels: Option<Array1<f64>>,
    /// Sum of squared distances to nearest centroid (inertia)
    pub inertia: Option<f64>,
    pub is_fitted: bool,
}

impl Default for KMeans {
    fn default() -> Self {
        Self::new(3)
    }
}

impl KMeans {
    pub fn new(n_clusters: usize) -> Self {
        Self {
            n_clusters,
            max_iter: 300,
            tol: 1e-4,
            random_state: Some(42),
            centroids: None,
            labels: None,
            inertia: None,
            is_fitted: false,
        }
    }

    pub fn with_max_iter(mut self, max_iter: usize) -> Self {
        self.max_iter = max_iter;
        self
    }

    pub fn with_random_state(mut self, seed: u64) -> Self {
        self.random_state = Some(seed);
        self
    }

    /// K-means++ initialization: pick centroids spread apart
    fn kmeans_pp_init(x: &Array2<f64>, k: usize, rng: &mut ChaCha8Rng) -> Array2<f64> {
        let n_samples = x.nrows();
        let n_features = x.ncols();
        let mut centroids = Array2::zeros((k, n_features));

        // Pick first centroid uniformly at random
        let first = (rng.next_u64() as usize) % n_samples;
        centroids.row_mut(0).assign(&x.row(first));

        // Maintain running minimum distances for O(k*n) instead of O(k²*n)
        let mut min_dists = vec![f64::MAX; n_samples];

        for c in 1..k {
            // Update min_dists against only the newly added centroid (c-1)
            let new_centroid = centroids.row(c - 1);
            for i in 0..n_samples {
                let row = x.row(i);
                let diff = &row - &new_centroid;
                let d = diff.mapv(|v| v * v).sum();
                if d < min_dists[i] {
                    min_dists[i] = d;
                }
            }

            // Weighted random selection proportional to D²
            let total: f64 = min_dists.iter().sum();
            if total <= 0.0 {
                let idx = (rng.next_u64() as usize) % n_samples;
                centroids.row_mut(c).assign(&x.row(idx));
                continue;
            }

            let r = (rng.next_u64() as f64 / u64::MAX as f64) * total;
            let mut cumulative = 0.0;
            let mut chosen = 0;
            for (i, &d) in min_dists.iter().enumerate() {
                cumulative += d;
                if cumulative >= r {
                    chosen = i;
                    break;
                }
            }
            centroids.row_mut(c).assign(&x.row(chosen));
        }

        centroids
    }

    fn euclidean_sq(a: &ndarray::ArrayView1<f64>, b: &ndarray::ArrayView1<f64>) -> f64 {
        a.iter().zip(b.iter()).map(|(x, y)| (x - y).powi(2)).sum()
    }

    /// Fit the model (unsupervised — no y needed)
    pub fn fit(&mut self, x: &Array2<f64>) -> Result<&mut Self> {
        let n_samples = x.nrows();
        if n_samples < self.n_clusters {
            return Err(AutoMLError::TrainingError(format!(
                "n_samples ({}) < n_clusters ({})", n_samples, self.n_clusters
            )));
        }

        let mut rng = ChaCha8Rng::seed_from_u64(self.random_state.unwrap_or(42));
        let mut centroids = Self::kmeans_pp_init(x, self.n_clusters, &mut rng);
        let mut labels = Array1::zeros(n_samples);
        let mut last_distances = vec![0.0f64; n_samples];

        for _iter in 0..self.max_iter {
            // Assignment step: assign each point to nearest centroid, reuse distances
            let new_assignments: Vec<(usize, f64)> = (0..n_samples)
                .into_par_iter()
                .map(|i| {
                    let row = x.row(i);
                    let mut best_c = 0;
                    let mut best_dist = f64::MAX;
                    for c in 0..self.n_clusters {
                        let d = Self::euclidean_sq(&row, &centroids.row(c));
                        if d < best_dist {
                            best_dist = d;
                            best_c = c;
                        }
                    }
                    (best_c, best_dist)
                })
                .collect();

            let new_labels_arr = Array1::from_vec(new_assignments.iter().map(|&(c, _)| c as f64).collect());
            for (i, &(_, d)) in new_assignments.iter().enumerate() {
                last_distances[i] = d;
            }

            // Check convergence
            let changed: usize = new_labels_arr.iter().zip(labels.iter())
                .filter(|(a, b): &(&f64, &f64)| (**a - **b).abs() > 0.5)
                .count();

            labels = new_labels_arr;

            // Update step: recompute centroids
            let mut new_centroids = Array2::zeros(centroids.dim());
            let mut counts = vec![0usize; self.n_clusters];

            for i in 0..n_samples {
                let c = labels[i] as usize;
                counts[c] += 1;
                for j in 0..x.ncols() {
                    new_centroids[[c, j]] += x[[i, j]];
                }
            }

            for c in 0..self.n_clusters {
                if counts[c] > 0 {
                    for j in 0..x.ncols() {
                        new_centroids[[c, j]] /= counts[c] as f64;
                    }
                } else {
                    // Empty cluster — reinitialize randomly
                    let idx = (rng.next_u64() as usize) % n_samples;
                    new_centroids.row_mut(c).assign(&x.row(idx));
                }
            }

            // Check centroid movement convergence
            let shift: f64 = centroids.iter().zip(new_centroids.iter())
                .map(|(a, b)| (a - b).powi(2))
                .sum::<f64>()
                .sqrt();

            centroids = new_centroids;

            if changed == 0 || shift < self.tol {
                break;
            }
        }

        // Inertia from last assignment distances (avoids recomputation)
        let inertia: f64 = last_distances.iter().sum();

        self.centroids = Some(centroids);
        self.labels = Some(labels);
        self.inertia = Some(inertia);
        self.is_fitted = true;
        Ok(self)
    }

    /// Predict cluster labels for new data
    pub fn predict(&self, x: &Array2<f64>) -> Result<Array1<f64>> {
        let centroids = self.centroids.as_ref()
            .ok_or(AutoMLError::ModelNotFitted)?;

        let labels: Vec<f64> = (0..x.nrows())
            .into_par_iter()
            .map(|i| {
                let row = x.row(i);
                let mut best_c = 0;
                let mut best_dist = f64::MAX;
                for c in 0..self.n_clusters {
                    let d = Self::euclidean_sq(&row, &centroids.row(c));
                    if d < best_dist {
                        best_dist = d;
                        best_c = c;
                    }
                }
                best_c as f64
            })
            .collect();

        Ok(Array1::from_vec(labels))
    }

    /// Get cluster centroids
    pub fn centroids(&self) -> Option<&Array2<f64>> {
        self.centroids.as_ref()
    }
}

// ═══════════════════════════════════════════════════════════════════════════
//  DBSCAN Clustering
// ═══════════════════════════════════════════════════════════════════════════

/// DBSCAN (Density-Based Spatial Clustering of Applications with Noise)
///
/// Points are classified as core, border, or noise:
/// - Core: has ≥ min_samples neighbors within eps radius
/// - Border: within eps of a core point but not core itself
/// - Noise: neither core nor border (label = -1)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DBSCAN {
    /// Maximum distance between neighbors
    pub eps: f64,
    /// Minimum points to form a dense region
    pub min_samples: usize,
    /// Assigned cluster labels (-1 = noise)
    pub labels: Option<Array1<f64>>,
    /// Number of clusters found (excluding noise)
    pub n_clusters_found: usize,
    /// Number of noise points
    pub n_noise: usize,
    pub is_fitted: bool,
    /// Training data (needed for predict on new data)
    train_x: Option<Array2<f64>>,
}

impl Default for DBSCAN {
    fn default() -> Self {
        Self::new(0.5, 5)
    }
}

impl DBSCAN {
    pub fn new(eps: f64, min_samples: usize) -> Self {
        Self {
            eps,
            min_samples,
            labels: None,
            n_clusters_found: 0,
            n_noise: 0,
            is_fitted: false,
            train_x: None,
        }
    }

    pub fn with_eps(mut self, eps: f64) -> Self {
        self.eps = eps;
        self
    }

    pub fn with_min_samples(mut self, min_samples: usize) -> Self {
        self.min_samples = min_samples;
        self
    }

    fn euclidean_dist(a: &ndarray::ArrayView1<f64>, b: &ndarray::ArrayView1<f64>) -> f64 {
        a.iter().zip(b.iter()).map(|(x, y)| (x - y).powi(2)).sum::<f64>().sqrt()
    }

    /// Find all neighbors within eps distance
    fn region_query(x: &Array2<f64>, point_idx: usize, eps: f64) -> Vec<usize> {
        let row = x.row(point_idx);
        (0..x.nrows())
            .filter(|&i| Self::euclidean_dist(&row, &x.row(i)) <= eps)
            .collect()
    }

    /// Fit the model (unsupervised)
    pub fn fit(&mut self, x: &Array2<f64>) -> Result<&mut Self> {
        let n_samples = x.nrows();
        let eps = self.eps;
        let min_samples = self.min_samples;

        // Pre-compute neighbor lists for all points (parallelized)
        let neighbors: Vec<Vec<usize>> = (0..n_samples)
            .into_par_iter()
            .map(|i| Self::region_query(x, i, eps))
            .collect();

        // Identify core points
        let is_core: Vec<bool> = neighbors.iter()
            .map(|n| n.len() >= min_samples)
            .collect();

        let mut labels = vec![-1i64; n_samples];
        let mut cluster_id: i64 = 0;

        for i in 0..n_samples {
            if labels[i] != -1 || !is_core[i] {
                continue;
            }

            // Expand cluster from core point i
            labels[i] = cluster_id;
            let mut queue: Vec<usize> = neighbors[i].clone();
            let mut head = 0;

            while head < queue.len() {
                let q = queue[head];
                head += 1;

                if labels[q] == -1 {
                    labels[q] = cluster_id;
                }
                if !is_core[q] {
                    continue;
                }
                // Expand from this core point
                for &neighbor in &neighbors[q] {
                    if labels[neighbor] == -1 {
                        labels[neighbor] = cluster_id;
                        queue.push(neighbor);
                    }
                }
            }

            cluster_id += 1;
        }

        let n_noise = labels.iter().filter(|&&l| l == -1).count();
        self.labels = Some(Array1::from_vec(labels.iter().map(|&l| l as f64).collect()));
        self.n_clusters_found = cluster_id as usize;
        self.n_noise = n_noise;
        self.train_x = Some(x.clone());
        self.is_fitted = true;
        Ok(self)
    }

    /// Predict cluster labels for new data by nearest core-point assignment
    pub fn predict(&self, x: &Array2<f64>) -> Result<Array1<f64>> {
        let train_x = self.train_x.as_ref()
            .ok_or(AutoMLError::ModelNotFitted)?;
        let train_labels = self.labels.as_ref()
            .ok_or(AutoMLError::ModelNotFitted)?;

        let labels: Vec<f64> = (0..x.nrows())
            .into_par_iter()
            .map(|i| {
                let row = x.row(i);
                let mut best_label = -1.0;
                let mut best_dist = f64::MAX;
                for j in 0..train_x.nrows() {
                    let d = Self::euclidean_dist(&row, &train_x.row(j));
                    if d < best_dist && d <= self.eps {
                        best_dist = d;
                        best_label = train_labels[j];
                    }
                }
                best_label
            })
            .collect();

        Ok(Array1::from_vec(labels))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_kmeans_basic() {
        // Two clear clusters
        let x = array![
            [1.0, 1.0], [1.5, 1.5], [1.2, 1.3],
            [8.0, 8.0], [8.5, 8.5], [8.2, 8.3],
        ];
        let mut model = KMeans::new(2);
        model.fit(&x).unwrap();
        assert!(model.is_fitted);
        assert_eq!(model.labels.as_ref().unwrap().len(), 6);
        // First 3 should be in same cluster, last 3 in different cluster
        let labels = model.labels.as_ref().unwrap();
        assert_eq!(labels[0], labels[1]);
        assert_eq!(labels[0], labels[2]);
        assert_eq!(labels[3], labels[4]);
        assert_eq!(labels[3], labels[5]);
        assert_ne!(labels[0], labels[3]);
    }

    #[test]
    fn test_kmeans_predict() {
        let x = array![
            [0.0, 0.0], [0.5, 0.5],
            [10.0, 10.0], [10.5, 10.5],
        ];
        let mut model = KMeans::new(2);
        model.fit(&x).unwrap();

        let new_x = array![[0.1, 0.1], [10.1, 10.1]];
        let labels = model.predict(&new_x).unwrap();
        assert_ne!(labels[0], labels[1]);
    }

    #[test]
    fn test_kmeans_inertia() {
        let x = array![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [10.0, 10.0]];
        let mut model = KMeans::new(2);
        model.fit(&x).unwrap();
        assert!(model.inertia.unwrap() > 0.0);
    }

    #[test]
    fn test_dbscan_basic() {
        let x = array![
            [1.0, 1.0], [1.1, 1.1], [1.2, 1.0], [1.0, 1.2],
            [8.0, 8.0], [8.1, 8.1], [8.2, 8.0], [8.0, 8.2],
            [50.0, 50.0], // noise point
        ];
        let mut model = DBSCAN::new(0.5, 3);
        model.fit(&x).unwrap();
        assert!(model.is_fitted);
        assert_eq!(model.n_clusters_found, 2);
        let labels = model.labels.as_ref().unwrap();
        // First cluster
        assert_eq!(labels[0], labels[1]);
        // Second cluster
        assert_eq!(labels[4], labels[5]);
        // Different clusters
        assert_ne!(labels[0], labels[4]);
        // Noise point
        assert_eq!(labels[8], -1.0);
    }

    #[test]
    fn test_dbscan_predict() {
        let x = array![
            [0.0, 0.0], [0.1, 0.1], [0.2, 0.0],
            [5.0, 5.0], [5.1, 5.1], [5.2, 5.0],
        ];
        let mut model = DBSCAN::new(0.5, 2);
        model.fit(&x).unwrap();

        let new_x = array![[0.05, 0.05], [5.05, 5.05], [100.0, 100.0]];
        let labels = model.predict(&new_x).unwrap();
        assert_ne!(labels[0], labels[1]); // different clusters
        assert_eq!(labels[2], -1.0); // too far = noise
    }
}
