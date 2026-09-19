//! UMAP — Uniform Manifold Approximation and Projection
//!
//! High-performance dimensionality reduction for 2D visualization.
//! Implements the UMAP algorithm (McInnes et al., 2018) with:
//! - Parallel KNN graph construction via rayon
//! - SIMD-accelerated distance computation
//! - Fuzzy simplicial set with binary-search sigma
//! - SGD layout optimization with negative sampling

use crate::training::knn::{DistanceMetric, KDTree};
use crate::utils::simd::SimdOps;
use ndarray::Array2;
use rand::Rng;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::cmp::Ordering;

/// UMAP configuration parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UmapConfig {
    /// Number of nearest neighbors (controls local vs global structure)
    pub n_neighbors: usize,
    /// Minimum distance between points in the embedding
    pub min_dist: f64,
    /// Number of output dimensions
    pub n_components: usize,
    /// Number of optimization epochs
    pub n_epochs: usize,
    /// SGD learning rate
    pub learning_rate: f64,
    /// Number of negative samples per positive edge
    pub negative_sample_rate: usize,
    /// Spread of the embedding
    pub spread: f64,
    /// Random seed for reproducibility
    pub random_state: u64,
    /// Maximum samples (subsample if dataset is larger)
    pub max_samples: usize,
}

impl Default for UmapConfig {
    fn default() -> Self {
        Self {
            n_neighbors: 15,
            min_dist: 0.1,
            n_components: 2,
            n_epochs: 200,
            learning_rate: 1.0,
            negative_sample_rate: 5,
            spread: 1.0,
            random_state: 42,
            max_samples: 10_000,
        }
    }
}

/// An edge in the fuzzy simplicial set
struct Edge {
    i: usize,
    j: usize,
    weight: f64,
}

/// How many SGD epochs between callback emissions during streaming.
const UMAP_STREAM_EMIT_EVERY: usize = 20;

/// UMAP dimensionality reduction
pub struct Umap {
    config: UmapConfig,
}

impl Umap {
    /// Create a new UMAP instance
    pub fn new(config: UmapConfig) -> Self {
        Self { config }
    }

    /// Run UMAP on dense data. Returns n_samples x 2 embedding.
    pub fn fit_transform(&self, data: &[Vec<f64>]) -> crate::error::Result<Vec<[f64; 2]>> {
        self.fit_transform_with_cb(data, |_, _| {})
    }

    /// Run UMAP and emit partial embeddings to `on_epoch` every
    /// `UMAP_STREAM_EMIT_EVERY` epochs and once after the final epoch.
    /// The final return value is the fully-converged embedding.
    pub fn fit_transform_with_cb<F>(
        &self,
        data: &[Vec<f64>],
        mut on_epoch: F,
    ) -> crate::error::Result<Vec<[f64; 2]>>
    where
        F: FnMut(usize, &[[f64; 2]]),
    {
        let n = data.len();
        if n < 3 {
            return Err(crate::error::AutoMLError::DataError(
                "UMAP requires at least 3 samples".to_string(),
            ));
        }

        let k = self.config.n_neighbors.min(n - 1);

        let (work_data, sample_indices) = if n > self.config.max_samples {
            let mut rng = ChaCha8Rng::seed_from_u64(self.config.random_state);
            let mut indices: Vec<usize> = (0..n).collect();
            for i in 0..self.config.max_samples {
                let j = rng.gen_range(i..n);
                indices.swap(i, j);
            }
            indices.truncate(self.config.max_samples);
            indices.sort_unstable();
            let sampled: Vec<Vec<f64>> = indices.iter().map(|&i| data[i].clone()).collect();
            (sampled, Some(indices))
        } else {
            (data.to_vec(), None)
        };

        let n_work = work_data.len();
        let (knn_indices, knn_distances) = self.compute_knn(&work_data, k);
        let edges = self.compute_fuzzy_set(&knn_indices, &knn_distances, k);

        let mut embedding = self.optimize_layout_with_cb(n_work, &edges, |epoch, emb| {
            on_epoch(epoch, emb);
        });

        if let Some(indices) = sample_indices {
            let mut full_embedding = vec![[0.0f64; 2]; n];
            for (sub_idx, &orig_idx) in indices.iter().enumerate() {
                full_embedding[orig_idx] = embedding[sub_idx];
            }
            for i in 0..n {
                if !indices.contains(&i) {
                    let mut best_dist = f64::INFINITY;
                    let mut best_idx = 0;
                    for &si in &indices {
                        let d = SimdOps::squared_euclidean_distance(&data[i], &data[si]);
                        if d < best_dist {
                            best_dist = d;
                            best_idx = si;
                        }
                    }
                    let emb = full_embedding[best_idx];
                    full_embedding[i] = [
                        emb[0] + (i as f64 * 0.001).sin() * 0.1,
                        emb[1] + (i as f64 * 0.001).cos() * 0.1,
                    ];
                }
            }
            embedding = full_embedding;
        }

        // Final callback: fires only if not already emitted by the loop
        if self.config.n_epochs % UMAP_STREAM_EMIT_EVERY != 0 {
            on_epoch(self.config.n_epochs, &embedding);
        }

        Ok(embedding)
    }

    /// Phase 1: Compute k-nearest neighbors via a KD-tree (falls back to a brute-force
    /// SIMD scan for degenerate inputs). Parallelized over samples with rayon.
    ///
    /// The original brute-force scan computed a distance to every other point for
    /// every query point (O(n^2)); a KD-tree brings the average case down to
    /// O(n log n) for the low/moderate dimensionality this is normally run on.
    /// Each query point is one of the tree's own rows, so it queries k+1 nearest
    /// and filters its own index out (mirroring the `exclude_self` pattern used by
    /// LOF/DBSCAN's KD-tree lookups) rather than never considering itself a
    /// candidate in the first place, as the brute-force loop did. In the
    /// vanishingly unlikely case of more than k exact-duplicate points at distance
    /// zero, this could in principle drop a point's own index from its neighbor
    /// set instead of another zero-distance duplicate — the same documented,
    /// negligible-probability caveat already accepted for LOF/DBSCAN.
    fn compute_knn(
        &self,
        data: &[Vec<f64>],
        k: usize,
    ) -> (Vec<Vec<usize>>, Vec<Vec<f64>>) {
        let n = data.len();
        let n_dims = data.first().map(|row| row.len()).unwrap_or(0);

        if n == 0 || n_dims == 0 {
            return (vec![Vec::new(); n], vec![Vec::new(); n]);
        }

        let flat: Vec<f64> = data.iter().flat_map(|row| row.iter().copied()).collect();
        let array_data = Array2::from_shape_vec((n, n_dims), flat)
            .expect("flattened row-major data always matches (n, n_dims)");
        let tree = KDTree::build(&array_data);

        let results: Vec<(Vec<usize>, Vec<f64>)> = (0..n)
            .into_par_iter()
            .map(|i| {
                let mut kd_results =
                    tree.query_k_nearest(&data[i], &array_data, k + 1, DistanceMetric::Euclidean);
                kd_results.retain(|&(_, idx)| idx != i);
                kd_results.truncate(k);
                kd_results.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(Ordering::Equal));

                let indices: Vec<usize> = kd_results.iter().map(|&(_, idx)| idx).collect();
                let distances: Vec<f64> = kd_results.iter().map(|&(d, _)| d).collect();
                (indices, distances)
            })
            .collect();

        let mut knn_indices = Vec::with_capacity(n);
        let mut knn_distances = Vec::with_capacity(n);
        for (idx, dist) in results {
            knn_indices.push(idx);
            knn_distances.push(dist);
        }
        (knn_indices, knn_distances)
    }

    /// Phase 2: Compute fuzzy simplicial set (edge weights).
    /// For each point, find rho (nearest neighbor distance) and sigma
    /// (smooth normalization via binary search), then symmetrize.
    fn compute_fuzzy_set(
        &self,
        knn_indices: &[Vec<usize>],
        knn_distances: &[Vec<f64>],
        k: usize,
    ) -> Vec<Edge> {
        let n = knn_indices.len();
        let target = (k as f64).ln() / std::f64::consts::LN_2; // log2(k)

        // Compute rho and sigma per point (parallelized)
        let params: Vec<(f64, f64)> = (0..n)
            .into_par_iter()
            .map(|i| {
                let dists = &knn_distances[i];
                let rho = if dists.is_empty() {
                    0.0
                } else {
                    dists[0].max(1e-12)
                };

                // Binary search for sigma
                let mut lo = 1e-8_f64;
                let mut hi = 1000.0_f64;
                let mut sigma = 1.0;

                for _ in 0..64 {
                    sigma = (lo + hi) / 2.0;
                    let sum: f64 = dists.iter()
                        .map(|&d| (-(d - rho).max(0.0) / sigma).exp())
                        .sum();

                    if (sum - target).abs() < 1e-5 {
                        break;
                    }
                    if sum > target {
                        hi = sigma;
                    } else {
                        lo = sigma;
                    }
                }

                (rho, sigma)
            })
            .collect();

        // Build directed edge weights
        use std::collections::HashMap;
        let mut edge_map: HashMap<(usize, usize), f64> = HashMap::with_capacity(n * k * 2);

        for i in 0..n {
            let (rho, sigma) = params[i];
            let dists = &knn_distances[i];
            let indices = &knn_indices[i];

            for (idx, (&j, &d)) in indices.iter().zip(dists.iter()).enumerate() {
                let w = if idx == 0 {
                    1.0 // nearest neighbor always has weight 1
                } else {
                    (-(d - rho).max(0.0) / sigma.max(1e-12)).exp()
                };
                edge_map.insert((i, j), w);
            }
        }

        // Symmetrize: w_sym(i,j) = w(i,j) + w(j,i) - w(i,j) * w(j,i)
        let mut symmetric_edges: HashMap<(usize, usize), f64> = HashMap::with_capacity(edge_map.len());

        for (&(i, j), &w_ij) in &edge_map {
            let key = if i < j { (i, j) } else { (j, i) };
            let w_ji = edge_map.get(&(j, i)).copied().unwrap_or(0.0);
            let w_sym = w_ij + w_ji - w_ij * w_ji;

            symmetric_edges
                .entry(key)
                .and_modify(|w| *w = w.max(w_sym))
                .or_insert(w_sym);
        }

        symmetric_edges
            .into_iter()
            .filter(|(_, w)| *w > 1e-8)
            .map(|((i, j), weight)| Edge { i, j, weight })
            .collect()
    }

    /// SGD layout with epoch callbacks — fires on_epoch every UMAP_STREAM_EMIT_EVERY epochs.
    fn optimize_layout_with_cb<F>(
        &self,
        n_samples: usize,
        edges: &[Edge],
        mut on_epoch: F,
    ) -> Vec<[f64; 2]>
    where
        F: FnMut(usize, &[[f64; 2]]),
    {
        let (a, b) = self.find_ab_params(self.config.spread, self.config.min_dist);
        let mut rng = ChaCha8Rng::seed_from_u64(self.config.random_state);
        let mut embedding: Vec<[f64; 2]> = (0..n_samples)
            .map(|_| [rng.gen_range(-10.0..10.0) * 0.01, rng.gen_range(-10.0..10.0) * 0.01])
            .collect();

        let n_epochs = self.config.n_epochs;
        let neg_rate = self.config.negative_sample_rate;
        let max_weight = edges.iter().map(|e| e.weight).fold(0.0_f64, f64::max);

        for epoch in 0..n_epochs {
            let alpha = self.config.learning_rate * (1.0 - epoch as f64 / n_epochs as f64);
            if alpha < 1e-8 { break; }

            for edge in edges {
                let epochs_per_sample = if edge.weight > 0.0 {
                    max_weight / edge.weight
                } else {
                    f64::INFINITY
                };
                if epoch as f64 % epochs_per_sample.max(1.0) >= 1.0 { continue; }

                let i = edge.i;
                let j = edge.j;
                let dy = [embedding[i][0] - embedding[j][0], embedding[i][1] - embedding[j][1]];
                let dist_sq = dy[0] * dy[0] + dy[1] * dy[1] + 1e-8;
                let grad_coeff = -2.0 * a * b * dist_sq.powf(b - 1.0) / (1.0 + a * dist_sq.powf(b));
                let gd0 = grad_coeff * dy[0];
                let gd1 = grad_coeff * dy[1];
                embedding[i][0] += alpha * gd0;
                embedding[i][1] += alpha * gd1;
                embedding[j][0] -= alpha * gd0;
                embedding[j][1] -= alpha * gd1;

                for _ in 0..neg_rate {
                    let k = rng.gen_range(0..n_samples);
                    if k == i { continue; }
                    let dy_neg = [embedding[i][0] - embedding[k][0], embedding[i][1] - embedding[k][1]];
                    let dsq_neg = dy_neg[0] * dy_neg[0] + dy_neg[1] * dy_neg[1] + 1e-8;
                    let gc_neg = 2.0 * b / ((0.001 + dsq_neg) * (1.0 + a * dsq_neg.powf(b)));
                    embedding[i][0] += alpha * gc_neg * dy_neg[0];
                    embedding[i][1] += alpha * gc_neg * dy_neg[1];
                }

                embedding[i][0] = embedding[i][0].clamp(-10.0, 10.0);
                embedding[i][1] = embedding[i][1].clamp(-10.0, 10.0);
                embedding[j][0] = embedding[j][0].clamp(-10.0, 10.0);
                embedding[j][1] = embedding[j][1].clamp(-10.0, 10.0);
            }

            // Emit intermediate snapshot every UMAP_STREAM_EMIT_EVERY epochs
            if (epoch + 1) % UMAP_STREAM_EMIT_EVERY == 0 {
                on_epoch(epoch + 1, &embedding);
            }
        }

        embedding
    }

    /// Find a, b parameters for the UMAP curve: 1 / (1 + a * d^(2b))
    /// that approximates a smooth step function at min_dist.
    fn find_ab_params(&self, spread: f64, min_dist: f64) -> (f64, f64) {
        // Use the curve fitting approach from the original UMAP paper.
        // We want: for d < min_dist, f(d) ~ 1; for d > min_dist, f(d) decays.
        // The parametric form is f(d) = 1 / (1 + a * d^(2b))
        //
        // A good approximation for typical spread=1.0:
        // b ~ 1, a ~ found by solving 1/(1 + a * min_dist^(2b)) = 0.5
        // when spread = 1.0.

        let mut b = 1.0;
        let mut a;

        // Simple iterative solver for a, b
        // For the standard case (spread=1.0), a closed-form approximation works well:
        if (spread - 1.0).abs() < 1e-6 {
            b = 1.0;
            // From 1/(1 + a * min_dist^2) ≈ exp(-min_dist + 1), solve for a:
            a = if min_dist > 0.0 {
                (2.0_f64.powf(2.0 * b) - 1.0) / min_dist.powf(2.0 * b)
            } else {
                1.0
            };
        } else {
            // General case: binary search for b, then solve a
            let mut lo = 0.1_f64;
            let mut hi = 5.0_f64;
            for _ in 0..64 {
                b = (lo + hi) / 2.0;
                let target_d = spread;
                // At d = spread, we want f(d) ≈ 0.5
                a = (2.0_f64.powf(2.0 * b) - 1.0) / target_d.powf(2.0 * b);
                let val = 1.0 / (1.0 + a * min_dist.powf(2.0 * b));
                if val > 0.99 {
                    hi = b;
                } else {
                    lo = b;
                }
            }
            a = (2.0_f64.powf(2.0 * b) - 1.0) / spread.powf(2.0 * b);
        }

        (a.max(1e-8), b.max(0.1))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_umap_basic() {
        // Two well-separated clusters in 5D
        let data = vec![
            vec![0.0, 0.0, 0.0, 0.0, 0.0],
            vec![0.1, 0.1, 0.0, 0.0, 0.0],
            vec![0.0, 0.1, 0.1, 0.0, 0.0],
            vec![0.1, 0.0, 0.1, 0.0, 0.0],
            vec![10.0, 10.0, 10.0, 10.0, 10.0],
            vec![10.1, 10.0, 10.0, 10.0, 10.0],
            vec![10.0, 10.1, 10.0, 10.0, 10.0],
            vec![10.1, 10.1, 10.0, 10.0, 10.0],
        ];

        let config = UmapConfig {
            n_neighbors: 3,
            n_epochs: 100,
            ..Default::default()
        };
        let umap = Umap::new(config);
        let result = umap.fit_transform(&data).unwrap();

        assert_eq!(result.len(), 8);

        // Each point should have 2 coordinates
        for point in &result {
            assert!(point[0].is_finite());
            assert!(point[1].is_finite());
        }
    }

    #[test]
    fn test_umap_separation() {
        // Two clusters should be separated in the embedding
        let mut data = Vec::new();
        for i in 0..20 {
            data.push(vec![i as f64 * 0.01, i as f64 * 0.01, 0.0]);
        }
        for i in 0..20 {
            data.push(vec![10.0 + i as f64 * 0.01, 10.0 + i as f64 * 0.01, 10.0]);
        }

        let config = UmapConfig {
            n_neighbors: 5,
            n_epochs: 200,
            ..Default::default()
        };
        let umap = Umap::new(config);
        let result = umap.fit_transform(&data).unwrap();

        // Compute mean of each cluster in embedding
        let cluster_a: Vec<[f64; 2]> = result[..20].to_vec();
        let cluster_b: Vec<[f64; 2]> = result[20..].to_vec();

        let mean_a = [
            cluster_a.iter().map(|p| p[0]).sum::<f64>() / 20.0,
            cluster_a.iter().map(|p| p[1]).sum::<f64>() / 20.0,
        ];
        let mean_b = [
            cluster_b.iter().map(|p| p[0]).sum::<f64>() / 20.0,
            cluster_b.iter().map(|p| p[1]).sum::<f64>() / 20.0,
        ];

        let inter_dist = ((mean_a[0] - mean_b[0]).powi(2) + (mean_a[1] - mean_b[1]).powi(2)).sqrt();
        assert!(inter_dist > 0.5, "Clusters should be separated, got distance: {}", inter_dist);
    }

    #[test]
    fn test_umap_config_defaults() {
        let config = UmapConfig::default();
        assert_eq!(config.n_neighbors, 15);
        assert!((config.min_dist - 0.1).abs() < 1e-10);
        assert_eq!(config.n_components, 2);
        assert_eq!(config.n_epochs, 200);
    }

    #[test]
    fn test_umap_too_few_samples() {
        let data = vec![vec![1.0, 2.0], vec![3.0, 4.0]];
        let umap = Umap::new(UmapConfig::default());
        assert!(umap.fit_transform(&data).is_err());
    }

    #[test]
    fn test_fit_transform_with_cb_fires_callbacks() {
        let data: Vec<Vec<f64>> = (0..20)
            .map(|i| vec![i as f64, (i % 2) as f64 * 10.0])
            .collect();
        let config = UmapConfig { n_neighbors: 3, n_epochs: 41, ..Default::default() };
        let umap = Umap::new(config);

        let mut callback_count = 0usize;
        let mut last_points_len = 0usize;
        let result = umap.fit_transform_with_cb(&data, |_epoch, pts| {
            callback_count += 1;
            last_points_len = pts.len();
        }).unwrap();

        // 41 epochs / 20 per emit = 2 mid-run callbacks + 1 final (41 % 20 != 0) = at least 3
        assert!(callback_count >= 3, "expected >= 3 callbacks, got {}", callback_count);
        assert_eq!(last_points_len, data.len());
        assert_eq!(result.len(), data.len());
    }

    #[test]
    fn test_fit_transform_delegates_to_with_cb() {
        let data: Vec<Vec<f64>> = (0..10)
            .map(|i| vec![i as f64, 0.0, 1.0])
            .collect();
        let config = UmapConfig { n_neighbors: 3, n_epochs: 20, ..Default::default() };
        let umap = Umap::new(config);
        // fit_transform must still work (delegates to with_cb with no-op)
        let result = umap.fit_transform(&data).unwrap();
        assert_eq!(result.len(), data.len());
    }
}

/// Verifies the KD-tree-backed `compute_knn` returns the same neighbor sets/distances
/// as the original brute-force scan, since this touches both float-accumulation order
/// (SIMD vs. the KD-tree's scalar `compute_distance`) and traversal-order tie-breaking.
#[cfg(test)]
mod knn_differential {
    use super::*;
    use std::collections::BinaryHeap;

    #[derive(Clone)]
    struct RefNeighbor {
        index: usize,
        distance: f64,
    }
    impl PartialEq for RefNeighbor {
        fn eq(&self, other: &Self) -> bool {
            self.distance == other.distance
        }
    }
    impl Eq for RefNeighbor {}
    impl PartialOrd for RefNeighbor {
        fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
            Some(self.cmp(other))
        }
    }
    impl Ord for RefNeighbor {
        fn cmp(&self, other: &Self) -> Ordering {
            self.distance.partial_cmp(&other.distance).unwrap_or(Ordering::Equal)
        }
    }

    // Verbatim copy of the pre-optimization brute-force compute_knn, kept only as a
    // reference to check the KD-tree version against.
    fn reference_compute_knn(data: &[Vec<f64>], k: usize) -> (Vec<Vec<usize>>, Vec<Vec<f64>>) {
        let n = data.len();
        let mut knn_indices = Vec::with_capacity(n);
        let mut knn_distances = Vec::with_capacity(n);

        for i in 0..n {
            let mut heap: BinaryHeap<RefNeighbor> = BinaryHeap::with_capacity(k + 1);
            for j in 0..n {
                if i == j {
                    continue;
                }
                let dist = SimdOps::squared_euclidean_distance(&data[i], &data[j]).sqrt();
                if heap.len() < k {
                    heap.push(RefNeighbor { index: j, distance: dist });
                } else if let Some(top) = heap.peek() {
                    if dist < top.distance {
                        heap.pop();
                        heap.push(RefNeighbor { index: j, distance: dist });
                    }
                }
            }
            let mut neighbors: Vec<RefNeighbor> = heap.into_vec();
            neighbors.sort_by(|a, b| a.distance.partial_cmp(&b.distance).unwrap_or(Ordering::Equal));
            knn_indices.push(neighbors.iter().map(|n| n.index).collect());
            knn_distances.push(neighbors.iter().map(|n| n.distance).collect());
        }
        (knn_indices, knn_distances)
    }

    // Small dependency-free xorshift64 PRNG for deterministic stress data.
    struct Xorshift64(u64);
    impl Xorshift64 {
        fn next_u64(&mut self) -> u64 {
            let mut x = self.0;
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            self.0 = x;
            x
        }
        fn next_f64(&mut self) -> f64 {
            (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
        }
    }

    fn make_data(n: usize, dims: usize, seed: u64) -> Vec<Vec<f64>> {
        let mut rng = Xorshift64(seed | 1);
        (0..n)
            .map(|_| (0..dims).map(|_| rng.next_f64() * 100.0).collect())
            .collect()
    }

    #[test]
    fn kd_tree_matches_brute_force_reference() {
        let umap = Umap::new(UmapConfig::default());

        for &(n, dims, k) in &[(30usize, 2usize, 5usize), (80, 4, 10), (15, 3, 3), (50, 8, 15)] {
            // Continuous random coordinates make exact-distance ties (and hence any
            // traversal-order-dependent tie-break ambiguity) vanishingly unlikely.
            let seed = 1000 * n as u64 + 100 * dims as u64 + k as u64 + 42;
            let data = make_data(n, dims, seed);

            let (ref_idx, ref_dist) = reference_compute_knn(&data, k);
            let (got_idx, got_dist) = umap.compute_knn(&data, k);

            for i in 0..n {
                assert_eq!(
                    ref_idx[i].len(),
                    got_idx[i].len(),
                    "neighbor count mismatch at n={n} dims={dims} k={k} i={i}"
                );
                for j in 0..ref_idx[i].len() {
                    assert_eq!(
                        ref_idx[i][j], got_idx[i][j],
                        "neighbor index mismatch at n={n} dims={dims} k={k} i={i} j={j}"
                    );
                    assert!(
                        (ref_dist[i][j] - got_dist[i][j]).abs() < 1e-9,
                        "distance mismatch at n={n} dims={dims} k={k} i={i} j={j}: ref={} got={}",
                        ref_dist[i][j],
                        got_dist[i][j]
                    );
                }
            }
        }
    }
}
