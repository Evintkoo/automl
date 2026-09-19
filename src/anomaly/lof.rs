//! Local Outlier Factor (LOF) anomaly detection

use crate::error::{AutoMLError, Result};
use crate::anomaly::AnomalyDetector;
use crate::training::knn::{DistanceMetric, KDTree};
use ndarray::{Array1, Array2};
use serde::{Deserialize, Serialize};
use std::cmp::Ordering;

/// LOF result for a single point
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LOFResult {
    /// LOF scores for each point
    pub lof_scores: Array1<f64>,
    /// k-distances for each point
    pub k_distances: Array1<f64>,
    /// Local reachability densities
    pub lrd: Array1<f64>,
}

/// Local Outlier Factor anomaly detector
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LocalOutlierFactor {
    /// Number of neighbors
    n_neighbors: usize,
    /// Contamination ratio
    contamination: f64,
    /// Training data
    x_train: Option<Array2<f64>>,
    /// Precomputed k-distances
    k_distances: Option<Array1<f64>>,
    /// Precomputed LRD values
    lrd: Option<Array1<f64>>,
    /// Decision threshold
    threshold: Option<f64>,
}

impl LocalOutlierFactor {
    /// Create new LOF detector
    pub fn new(n_neighbors: usize) -> Self {
        Self {
            n_neighbors: n_neighbors.max(1),
            contamination: 0.1,
            x_train: None,
            k_distances: None,
            lrd: None,
            threshold: None,
        }
    }

    /// Set contamination ratio
    pub fn with_contamination(mut self, c: f64) -> Self {
        self.contamination = c.clamp(0.0, 0.5);
        self
    }

    /// Find k nearest neighbors and their distances, using a KD-tree built
    /// once by the caller (per `fit`/`compute_lof_details` call, not once
    /// per query point) instead of a brute-force O(n) distance scan per
    /// query — O(n) per `fit` / O(n_train) per scored point, respectively,
    /// down from O(n^2) / O(n_query * n_train) total.
    ///
    /// When `exclude_self` is set, this queries k+1 nearest and drops the
    /// specific excluded index if present (matching the original scan's
    /// "skip exactly this index" semantics, not "skip whichever is
    /// closest") — a genuine duplicate point at a *different* index is a
    /// legitimate neighbor and is kept. Note: unlike the brute-force scan,
    /// which self-excludes deterministically by index, tie-breaking among
    /// multiple points at the exact same distance can differ from the KD-
    /// tree's traversal order; this only matters for datasets containing
    /// exact coordinate duplicates, in which case swapping which duplicate
    /// is treated as "the" neighbor is a distance-preserving substitution.
    fn k_nearest_neighbors(
        &self,
        tree: &KDTree,
        point: &[f64],
        data: &Array2<f64>,
        k: usize,
        exclude_self: Option<usize>,
    ) -> Vec<(usize, f64)> {
        let query_k = if exclude_self.is_some() { k + 1 } else { k };
        let mut results = tree.query_k_nearest(point, data, query_k, DistanceMetric::Euclidean);
        if let Some(exclude_idx) = exclude_self {
            results.retain(|&(_, idx)| idx != exclude_idx);
        }
        results.truncate(k);
        results.into_iter().map(|(d, i)| (i, d)).collect()
    }

    /// Compute k-distance for a point (distance to k-th nearest neighbor)
    fn k_distance(&self, neighbors: &[(usize, f64)]) -> f64 {
        neighbors
            .iter()
            .map(|(_, d)| *d)
            .fold(0.0, f64::max)
    }

    /// Compute reachability distance
    fn reachability_distance(&self, _point_idx: usize, neighbor_idx: usize, dist: f64, k_distances: &Array1<f64>) -> f64 {
        k_distances[neighbor_idx].max(dist)
    }

    /// Compute Local Reachability Density (LRD)
    fn compute_lrd(
        &self,
        _point: &[f64],
        neighbors: &[(usize, f64)],
        k_distances: &Array1<f64>,
    ) -> f64 {
        if neighbors.is_empty() {
            return 0.0;
        }

        let sum_reach_dist: f64 = neighbors
            .iter()
            .map(|&(idx, dist)| self.reachability_distance(0, idx, dist, k_distances))
            .sum();

        if sum_reach_dist == 0.0 {
            f64::INFINITY
        } else {
            neighbors.len() as f64 / sum_reach_dist
        }
    }

    /// Compute LOF for a single point
    fn compute_lof_single(
        &self,
        lrd_point: f64,
        neighbors: &[(usize, f64)],
        lrd_values: &Array1<f64>,
    ) -> f64 {
        if neighbors.is_empty() || lrd_point == 0.0 {
            return 1.0;
        }

        let sum_lrd_ratio: f64 = neighbors
            .iter()
            .map(|&(idx, _)| lrd_values[idx] / lrd_point)
            .sum();

        sum_lrd_ratio / neighbors.len() as f64
    }

    /// Get detailed LOF results
    pub fn compute_lof_details(&self, x: &Array2<f64>) -> Result<LOFResult> {
        let x_train = self.x_train.as_ref().ok_or_else(|| {
            AutoMLError::ValidationError("Model not fitted".to_string())
        })?;

        let train_k_distances = self.k_distances.as_ref().unwrap();
        let train_lrd = self.lrd.as_ref().unwrap();

        let n = x.nrows();
        let mut lof_scores = Vec::with_capacity(n);
        let mut k_distances = Vec::with_capacity(n);
        let mut lrd_values = Vec::with_capacity(n);

        // Built once for this call and reused for every query point below,
        // instead of each `k_nearest_neighbors` call scanning all of
        // x_train from scratch (was O(n_query * n_train), now O(n_train log
        // n_train) to build plus O(log n_train) per query).
        let tree = KDTree::build(x_train);

        for row in x.rows() {
            let point: Vec<f64> = row.iter().copied().collect();
            let mut neighbors =
                self.k_nearest_neighbors(&tree, &point, x_train, self.n_neighbors, None);

            // If the query point coincides with a training point (distance 0 to its
            // nearest neighbor), that neighbor is almost certainly the point's own
            // row in the training set (the common case of scoring the training data
            // itself, e.g. via score_samples/predict/fit_predict on x_train). Recompute
            // the k-NN set excluding that row, matching the `exclude_self` behavior
            // already used during fit().
            if let Some(&(zero_idx, _)) = neighbors.iter().find(|&&(_, d)| d == 0.0) {
                neighbors = self.k_nearest_neighbors(
                    &tree,
                    &point,
                    x_train,
                    self.n_neighbors,
                    Some(zero_idx),
                );
            }

            let k_dist = self.k_distance(&neighbors);
            k_distances.push(k_dist);

            let lrd = self.compute_lrd(&point, &neighbors, train_k_distances);
            lrd_values.push(lrd);

            let lof = self.compute_lof_single(lrd, &neighbors, train_lrd);
            lof_scores.push(lof);
        }

        Ok(LOFResult {
            lof_scores: Array1::from_vec(lof_scores),
            k_distances: Array1::from_vec(k_distances),
            lrd: Array1::from_vec(lrd_values),
        })
    }
}

impl Default for LocalOutlierFactor {
    fn default() -> Self {
        Self::new(20)
    }
}

impl AnomalyDetector for LocalOutlierFactor {
    fn fit(&mut self, x: &Array2<f64>) -> Result<()> {
        let n = x.nrows();

        if n < 2 {
            return Err(AutoMLError::ValidationError(
                "LocalOutlierFactor::fit requires at least 2 samples".to_string(),
            ));
        }

        let k = self.n_neighbors.min(n - 1).max(1);

        // Built once and reused for every training point's k-NN query below
        // — was an O(n) brute-force scan per point (O(n^2) total for fit).
        let tree = KDTree::build(x);

        // Compute k-distances for all training points
        let mut k_distances = Vec::with_capacity(n);
        let mut all_neighbors: Vec<Vec<(usize, f64)>> = Vec::with_capacity(n);

        for (i, row) in x.rows().into_iter().enumerate() {
            let point: Vec<f64> = row.iter().copied().collect();
            let neighbors = self.k_nearest_neighbors(&tree, &point, x, k, Some(i));
            k_distances.push(self.k_distance(&neighbors));
            all_neighbors.push(neighbors);
        }

        let k_distances = Array1::from_vec(k_distances);

        // Compute LRD for all training points
        let mut lrd_values = Vec::with_capacity(n);
        for (i, row) in x.rows().into_iter().enumerate() {
            let point: Vec<f64> = row.iter().copied().collect();
            let lrd = self.compute_lrd(&point, &all_neighbors[i], &k_distances);
            lrd_values.push(lrd);
        }

        let lrd = Array1::from_vec(lrd_values);

        // Compute LOF scores for training data
        let mut lof_scores = Vec::with_capacity(n);
        for i in 0..n {
            let lof = self.compute_lof_single(lrd[i], &all_neighbors[i], &lrd);
            lof_scores.push(lof);
        }

        // Set threshold based on contamination
        let mut sorted_scores = lof_scores.clone();
        sorted_scores.sort_by(|a, b| b.partial_cmp(a).unwrap_or(Ordering::Equal));
        let threshold_idx = ((self.contamination * n as f64) as usize).min(n - 1);
        self.threshold = Some(sorted_scores[threshold_idx]);

        self.x_train = Some(x.clone());
        self.k_distances = Some(k_distances);
        self.lrd = Some(lrd);

        Ok(())
    }

    fn score_samples(&self, x: &Array2<f64>) -> Result<Array1<f64>> {
        let result = self.compute_lof_details(x)?;
        Ok(result.lof_scores)
    }

    fn predict(&self, x: &Array2<f64>) -> Result<Array1<i32>> {
        let scores = self.score_samples(x)?;
        let threshold = self.threshold.unwrap_or(1.5);

        let labels: Vec<i32> = scores
            .iter()
            .map(|&s| if s > threshold { -1 } else { 1 })
            .collect();

        Ok(Array1::from_vec(labels))
    }

    fn threshold(&self) -> f64 {
        self.threshold.unwrap_or(1.5)
    }
}

#[cfg(test)]
mod lof_differential {
    // Verifies the KD-tree-backed k_nearest_neighbors produces identical
    // fit()/compute_lof_details() output to the original brute-force,
    // max-heap-based scan, on realistic (duplicate-free) continuous data —
    // see the caveat on `k_nearest_neighbors` about exact-distance ties.
    use super::*;

    fn ref_euclidean_distance(a: &[f64], b: &[f64]) -> f64 {
        a.iter().zip(b.iter()).map(|(x, y)| (x - y).powi(2)).sum::<f64>().sqrt()
    }

    fn ref_k_nearest_neighbors(
        point: &[f64],
        data: &Array2<f64>,
        k: usize,
        exclude_self: Option<usize>,
    ) -> Vec<(usize, f64)> {
        use std::collections::BinaryHeap;

        #[derive(Clone, Copy)]
        struct OrderedFloat(f64, usize);
        impl PartialEq for OrderedFloat {
            fn eq(&self, other: &Self) -> bool { self.0 == other.0 }
        }
        impl Eq for OrderedFloat {}
        impl PartialOrd for OrderedFloat {
            fn partial_cmp(&self, other: &Self) -> Option<Ordering> { Some(self.cmp(other)) }
        }
        impl Ord for OrderedFloat {
            fn cmp(&self, other: &Self) -> Ordering {
                self.0.partial_cmp(&other.0).unwrap_or(Ordering::Equal)
            }
        }

        let mut heap: BinaryHeap<OrderedFloat> = BinaryHeap::new();
        for (i, row) in data.rows().into_iter().enumerate() {
            if Some(i) == exclude_self {
                continue;
            }
            let row_vec: Vec<f64> = row.iter().copied().collect();
            let dist = ref_euclidean_distance(point, &row_vec);
            if heap.len() < k {
                heap.push(OrderedFloat(dist, i));
            } else if let Some(&OrderedFloat(max_dist, _)) = heap.peek() {
                if dist < max_dist {
                    heap.pop();
                    heap.push(OrderedFloat(dist, i));
                }
            }
        }
        heap.into_iter().map(|OrderedFloat(d, i)| (i, d)).collect()
    }

    fn ref_k_distance(neighbors: &[(usize, f64)]) -> f64 {
        neighbors.iter().map(|(_, d)| *d).fold(0.0, f64::max)
    }

    fn ref_lrd(neighbors: &[(usize, f64)], k_distances: &Array1<f64>) -> f64 {
        if neighbors.is_empty() {
            return 0.0;
        }
        let sum_reach_dist: f64 = neighbors
            .iter()
            .map(|&(idx, dist)| k_distances[idx].max(dist))
            .sum();
        if sum_reach_dist == 0.0 {
            f64::INFINITY
        } else {
            neighbors.len() as f64 / sum_reach_dist
        }
    }

    fn ref_lof_single(lrd_point: f64, neighbors: &[(usize, f64)], lrd_values: &Array1<f64>) -> f64 {
        if neighbors.is_empty() || lrd_point == 0.0 {
            return 1.0;
        }
        let sum_lrd_ratio: f64 = neighbors.iter().map(|&(idx, _)| lrd_values[idx] / lrd_point).sum();
        sum_lrd_ratio / neighbors.len() as f64
    }

    /// Full brute-force reference: fit on `x`, then score `x` against itself
    /// (the common "score the training data" path, which also exercises the
    /// zero-distance self-exclusion retry in compute_lof_details).
    fn reference_fit_and_score(x: &Array2<f64>, k: usize) -> (Array1<f64>, Array1<f64>, Array1<f64>) {
        let n = x.nrows();
        let k = k.min(n - 1).max(1);

        let mut k_distances = Vec::with_capacity(n);
        let mut all_neighbors: Vec<Vec<(usize, f64)>> = Vec::with_capacity(n);
        for (i, row) in x.rows().into_iter().enumerate() {
            let point: Vec<f64> = row.iter().copied().collect();
            let neighbors = ref_k_nearest_neighbors(&point, x, k, Some(i));
            k_distances.push(ref_k_distance(&neighbors));
            all_neighbors.push(neighbors);
        }
        let k_distances = Array1::from_vec(k_distances);

        let mut lrd_values = Vec::with_capacity(n);
        for i in 0..n {
            lrd_values.push(ref_lrd(&all_neighbors[i], &k_distances));
        }
        let lrd = Array1::from_vec(lrd_values);

        let mut lof_scores = Vec::with_capacity(n);
        for i in 0..n {
            lof_scores.push(ref_lof_single(lrd[i], &all_neighbors[i], &lrd));
        }

        (Array1::from_vec(lof_scores), k_distances, lrd)
    }

    fn make_data(n: usize, seed: u64) -> Array2<f64> {
        // Continuous, non-duplicate points (a few well-separated blobs) —
        // exact distance ties are a measure-zero event for this data, so
        // the KD-tree and brute-force scan should select identical
        // neighbor sets.
        let mut state = seed;
        let mut next = move || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            (state as f64 / u64::MAX as f64) - 0.5
        };
        let centers = [(0.0, 0.0), (10.0, 0.0), (0.0, 10.0)];
        let mut data = Vec::with_capacity(n * 2);
        for i in 0..n {
            let (cx, cy) = centers[i % centers.len()];
            data.push(cx + next() * 3.0);
            data.push(cy + next() * 3.0);
        }
        // A couple of clear outliers
        data.push(100.0);
        data.push(100.0);
        data.push(-100.0);
        data.push(-50.0);
        Array2::from_shape_vec((n + 2, 2), data).unwrap()
    }

    #[test]
    fn optimized_matches_reference() {
        for &k in &[3usize, 5, 10] {
            let x = make_data(80, 0x1234_5678_9abc_def1);

            let (ref_scores, ref_kd, ref_lrd) = reference_fit_and_score(&x, k);

            let mut lof = LocalOutlierFactor::new(k);
            lof.fit(&x).unwrap();
            let got = lof.compute_lof_details(&x).unwrap();

            for i in 0..x.nrows() {
                assert!(
                    (got.k_distances[i] - ref_kd[i]).abs() < 1e-9,
                    "k_distance mismatch at k={k}, i={i}: got {} want {}",
                    got.k_distances[i], ref_kd[i]
                );
                assert!(
                    (got.lrd[i] - ref_lrd[i]).abs() < 1e-9,
                    "lrd mismatch at k={k}, i={i}: got {} want {}",
                    got.lrd[i], ref_lrd[i]
                );
                assert!(
                    (got.lof_scores[i] - ref_scores[i]).abs() < 1e-9,
                    "lof_score mismatch at k={k}, i={i}: got {} want {}",
                    got.lof_scores[i], ref_scores[i]
                );
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lof_basic() {
        // Create a cluster of normal points: 10 points with 2 features
        let mut data = Vec::new();
        for i in 0..10 {
            data.push((i % 5) as f64);
            data.push(((i % 5) + 1) as f64);
        }
        // Add an outlier
        data.extend_from_slice(&[50.0, 50.0]);

        let x = Array2::from_shape_vec((11, 2), data).unwrap();

        let mut lof = LocalOutlierFactor::new(3)
            .with_contamination(0.1);

        lof.fit(&x).unwrap();

        let scores = lof.score_samples(&x).unwrap();
        
        // The outlier (last point) should have higher LOF score
        let outlier_score = scores[10];
        let normal_avg: f64 = scores.iter().take(10).sum::<f64>() / 10.0;
        
        assert!(outlier_score > normal_avg);
    }

    #[test]
    fn test_lof_prediction() {
        // 15 points with 2 features each
        let mut data = Vec::new();
        for i in 0..15 {
            data.push((i % 6) as f64);
            data.push(((i + 1) % 6) as f64);
        }
        data.extend_from_slice(&[100.0, 100.0]);
        data.extend_from_slice(&[-100.0, -100.0]);

        let x = Array2::from_shape_vec((17, 2), data).unwrap();

        let mut lof = LocalOutlierFactor::new(5)
            .with_contamination(0.15);

        let labels = lof.fit_predict(&x).unwrap();

        // Should detect at least the obvious outliers
        let n_anomalies = labels.iter().filter(|&&l| l == -1).count();
        assert!(n_anomalies >= 1);
    }
}
