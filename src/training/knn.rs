//! K-Nearest Neighbors implementation
//!
//! KNN classifier and regressor with distance metrics.

use ndarray::{Array1, Array2};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::BinaryHeap;
use std::collections::HashMap;
use std::cmp::Ordering;

use crate::error::Result;

/// Dimensionality threshold above which random projection is applied
const HIGH_DIM_THRESHOLD: usize = 16;

// ============================================================================
// Random Projection (Johnson-Lindenstrauss) for high-dimensional KNN
// ============================================================================

/// Random Gaussian projection for dimensionality reduction (JL lemma)
#[derive(Debug, Clone, Serialize, Deserialize)]
struct RandomProjection {
    projection_matrix: Array2<f64>,
    target_dim: usize,
}

impl RandomProjection {
    /// Build a random projection matrix with entries ~ N(0, 1/target_dim).
    /// `target_dim = min(n_features, max(k * 4, ceil(5 * ln(n_samples))))` capped at n_features.
    fn build(n_features: usize, n_samples: usize, k_neighbors: usize, seed: u64) -> Self {
        let jl_dim = (5.0 * (n_samples as f64).ln()).ceil() as usize;
        let target_dim = (k_neighbors * 4).max(jl_dim).min(n_features);

        // Simple seeded PRNG (xorshift64) for Gaussian generation via Box-Muller
        let mut rng_state = seed;
        let mut next_u64 = || -> u64 {
            rng_state ^= rng_state << 13;
            rng_state ^= rng_state >> 7;
            rng_state ^= rng_state << 17;
            rng_state
        };

        let scale = 1.0 / (target_dim as f64).sqrt();
        let mut data = Vec::with_capacity(n_features * target_dim);
        for _ in 0..(n_features * target_dim / 2 + 1) {
            // Box-Muller transform
            let u1 = (next_u64() as f64) / (u64::MAX as f64);
            let u2 = (next_u64() as f64) / (u64::MAX as f64);
            let u1 = u1.max(1e-15); // avoid log(0)
            let r = (-2.0 * u1.ln()).sqrt();
            let theta = 2.0 * std::f64::consts::PI * u2;
            data.push(r * theta.cos() * scale);
            data.push(r * theta.sin() * scale);
        }
        data.truncate(n_features * target_dim);

        let projection_matrix =
            Array2::from_shape_vec((n_features, target_dim), data).unwrap();

        RandomProjection {
            projection_matrix,
            target_dim,
        }
    }

    /// Project data: x.dot(&projection_matrix)
    fn transform(&self, x: &Array2<f64>) -> Array2<f64> {
        x.dot(&self.projection_matrix)
    }
}

/// Distance metric for KNN
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum DistanceMetric {
    /// Euclidean distance (L2)
    Euclidean,
    /// Manhattan distance (L1)
    Manhattan,
    /// Minkowski distance with parameter p
    Minkowski(f64),
    /// Cosine similarity (converted to distance)
    Cosine,
}

impl Default for DistanceMetric {
    fn default() -> Self {
        Self::Euclidean
    }
}

/// KNN configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KNNConfig {
    /// Number of neighbors
    pub n_neighbors: usize,
    /// Distance metric
    pub metric: DistanceMetric,
    /// Weighting scheme
    pub weights: WeightScheme,
}

/// Weighting scheme for neighbors
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum WeightScheme {
    /// All neighbors have equal weight
    Uniform,
    /// Closer neighbors have more weight (inverse distance)
    Distance,
}

impl Default for WeightScheme {
    fn default() -> Self {
        Self::Uniform
    }
}

impl Default for KNNConfig {
    fn default() -> Self {
        Self {
            n_neighbors: 5,
            metric: DistanceMetric::Euclidean,
            weights: WeightScheme::Uniform,
        }
    }
}

/// K-Nearest Neighbors Classifier
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KNNClassifier {
    config: KNNConfig,
    x_train: Option<Array2<f64>>,
    y_train: Option<Array1<f64>>,
    classes: Vec<i64>,
    kd_tree: Option<KDTree>,
    projection: Option<RandomProjection>,
    x_train_projected: Option<Array2<f64>>,
}

impl KNNClassifier {
    pub fn new(config: KNNConfig) -> Self {
        Self {
            config,
            x_train: None,
            y_train: None,
            classes: Vec::new(),
            kd_tree: None,
            projection: None,
            x_train_projected: None,
        }
    }

    /// Create with default config and specified k
    pub fn with_k(k: usize) -> Self {
        Self::new(KNNConfig {
            n_neighbors: k,
            ..Default::default()
        })
    }

    /// Fit the classifier (stores training data and builds KD-tree if applicable)
    pub fn fit(&mut self, x: &Array2<f64>, y: &Array1<f64>) -> Result<()> {
        self.x_train = Some(x.clone());
        self.y_train = Some(y.clone());

        // Find unique classes
        let mut class_set: Vec<i64> = y.iter()
            .map(|&v| v as i64)
            .collect();
        class_set.sort();
        class_set.dedup();
        self.classes = class_set;

        // Build KD-tree for compatible metrics
        if kd_tree_compatible(self.config.metric) {
            if x.ncols() > HIGH_DIM_THRESHOLD {
                // High-dimensional: apply random projection before building KD-tree
                let seed = 42u64;
                let proj = RandomProjection::build(x.ncols(), x.nrows(), self.config.n_neighbors, seed);
                let x_proj = proj.transform(x);
                self.kd_tree = Some(KDTree::build(&x_proj));
                self.x_train_projected = Some(x_proj);
                self.projection = Some(proj);
            } else {
                self.kd_tree = Some(KDTree::build(x));
                self.projection = None;
                self.x_train_projected = None;
            }
        }

        Ok(())
    }

    /// Predict class labels (parallelized over test samples)
    pub fn predict(&self, x: &Array2<f64>) -> Array1<f64> {
        let x_train = self.x_train.as_ref().expect("Model not fitted");
        let y_train = self.y_train.as_ref().expect("Model not fitted");
        let k = self.config.n_neighbors;
        let metric = self.config.metric;
        let weights = self.config.weights;
        let classes = &self.classes;
        let kd_tree = &self.kd_tree;

        // If projection is active, project query points and use projected training data for KD-tree
        let x_query;
        let kd_data;
        if let Some(ref proj) = self.projection {
            x_query = proj.transform(x);
            kd_data = self.x_train_projected.as_ref().unwrap();
        } else {
            x_query = x.to_owned();
            kd_data = x_train;
        }

        let predictions: Vec<f64> = (0..x_query.nrows())
            .into_par_iter()
            .map(|i| {
                let row = x_query.row(i);
                let point = row.as_slice().unwrap();
                let neighbors = if let Some(tree) = kd_tree {
                    let kd_results = tree.query_k_nearest(point, kd_data, k, metric);
                    kd_results.into_iter().map(|(d, idx)| (d, y_train[idx])).collect()
                } else {
                    find_k_nearest(point, x_train, y_train, k, metric)
                };
                vote_classify(&neighbors, classes, weights)
            })
            .collect();

        Array1::from_vec(predictions)
    }

    /// Predict class probabilities (parallelized)
    pub fn predict_proba(&self, x: &Array2<f64>) -> Array2<f64> {
        let x_train = self.x_train.as_ref().expect("Model not fitted");
        let y_train = self.y_train.as_ref().expect("Model not fitted");
        let n_classes = self.classes.len();
        let k = self.config.n_neighbors;
        let metric = self.config.metric;
        let weights = self.config.weights;
        let classes = &self.classes;
        let kd_tree = &self.kd_tree;

        // If projection is active, project query points and use projected training data for KD-tree
        let x_query;
        let kd_data;
        if let Some(ref proj) = self.projection {
            x_query = proj.transform(x);
            kd_data = self.x_train_projected.as_ref().unwrap();
        } else {
            x_query = x.to_owned();
            kd_data = x_train;
        }

        let probs: Vec<Vec<f64>> = (0..x_query.nrows())
            .into_par_iter()
            .map(|i| {
                let row = x_query.row(i);
                let point = row.as_slice().unwrap();
                let neighbors = if let Some(tree) = kd_tree {
                    let kd_results = tree.query_k_nearest(point, kd_data, k, metric);
                    kd_results.into_iter().map(|(d, idx)| (d, y_train[idx])).collect()
                } else {
                    find_k_nearest(point, x_train, y_train, k, metric)
                };
                class_probs_from(&neighbors, classes, n_classes, weights)
            })
            .collect();

        let flat: Vec<f64> = probs.into_iter().flatten().collect();
        Array2::from_shape_vec((x.nrows(), n_classes), flat).unwrap()
    }

    #[allow(dead_code)]
    fn distance(&self, a: &Array1<f64>, b: &Array1<f64>) -> f64 {
        compute_distance(a.as_slice().unwrap(), b.as_slice().unwrap(), self.config.metric)
    }
}

/// K-Nearest Neighbors Regressor
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KNNRegressor {
    config: KNNConfig,
    x_train: Option<Array2<f64>>,
    y_train: Option<Array1<f64>>,
    kd_tree: Option<KDTree>,
    projection: Option<RandomProjection>,
    x_train_projected: Option<Array2<f64>>,
}

impl KNNRegressor {
    pub fn new(config: KNNConfig) -> Self {
        Self {
            config,
            x_train: None,
            y_train: None,
            kd_tree: None,
            projection: None,
            x_train_projected: None,
        }
    }

    /// Create with default config and specified k
    pub fn with_k(k: usize) -> Self {
        Self::new(KNNConfig {
            n_neighbors: k,
            ..Default::default()
        })
    }

    /// Fit the regressor (stores training data and builds KD-tree if applicable)
    pub fn fit(&mut self, x: &Array2<f64>, y: &Array1<f64>) -> Result<()> {
        self.x_train = Some(x.clone());
        self.y_train = Some(y.clone());

        if kd_tree_compatible(self.config.metric) {
            if x.ncols() > HIGH_DIM_THRESHOLD {
                let seed = 42u64;
                let proj = RandomProjection::build(x.ncols(), x.nrows(), self.config.n_neighbors, seed);
                let x_proj = proj.transform(x);
                self.kd_tree = Some(KDTree::build(&x_proj));
                self.x_train_projected = Some(x_proj);
                self.projection = Some(proj);
            } else {
                self.kd_tree = Some(KDTree::build(x));
                self.projection = None;
                self.x_train_projected = None;
            }
        }

        Ok(())
    }

    /// Predict target values (parallelized over test samples)
    pub fn predict(&self, x: &Array2<f64>) -> Array1<f64> {
        let x_train = self.x_train.as_ref().expect("Model not fitted");
        let y_train = self.y_train.as_ref().expect("Model not fitted");
        let k = self.config.n_neighbors;
        let metric = self.config.metric;
        let weights = self.config.weights;
        let kd_tree = &self.kd_tree;

        let x_query;
        let kd_data;
        if let Some(ref proj) = self.projection {
            x_query = proj.transform(x);
            kd_data = self.x_train_projected.as_ref().unwrap();
        } else {
            x_query = x.to_owned();
            kd_data = x_train;
        }

        let predictions: Vec<f64> = (0..x_query.nrows())
            .into_par_iter()
            .map(|i| {
                let row = x_query.row(i);
                let point = row.as_slice().unwrap();
                let neighbors = if let Some(tree) = kd_tree {
                    let kd_results = tree.query_k_nearest(point, kd_data, k, metric);
                    kd_results.into_iter().map(|(d, idx)| (d, y_train[idx])).collect()
                } else {
                    find_k_nearest(point, x_train, y_train, k, metric)
                };
                weighted_mean_from(&neighbors, weights)
            })
            .collect();

        Array1::from_vec(predictions)
    }
}

// ============================================================================
// KD-Tree spatial index for O(log N) nearest-neighbor queries
// ============================================================================

/// KD-tree node
#[derive(Debug, Clone, Serialize, Deserialize)]
enum KDNode {
    Leaf {
        indices: Vec<usize>,
    },
    Split {
        dimension: usize,
        threshold: f64,
        left: Box<KDNode>,
        right: Box<KDNode>,
    },
}

/// KD-tree for spatial indexing.
///
/// `pub(crate)` so other modules with the same "find nearby points fast"
/// problem (e.g. DBSCAN in `clustering.rs`) can reuse this index instead of
/// falling back to an O(n) brute-force scan per query.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct KDTree {
    root: KDNode,
    n_dims: usize,
}

/// Max-heap entry for KD-tree k-nearest query
#[derive(PartialEq)]
struct HeapEntry {
    dist: f64,
    idx: usize,
}

impl Eq for HeapEntry {}
impl PartialOrd for HeapEntry {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        self.dist.partial_cmp(&other.dist)
    }
}
impl Ord for HeapEntry {
    fn cmp(&self, other: &Self) -> Ordering {
        self.partial_cmp(other).unwrap_or(Ordering::Equal)
    }
}

const KD_LEAF_SIZE: usize = 16;

impl KDTree {
    /// Build a KD-tree from a dataset
    pub(crate) fn build(data: &Array2<f64>) -> Self {
        let n_dims = data.ncols();
        let indices: Vec<usize> = (0..data.nrows()).collect();
        let root = Self::build_node(data, indices, 0, n_dims);
        KDTree { root, n_dims }
    }

    fn build_node(data: &Array2<f64>, mut indices: Vec<usize>, depth: usize, n_dims: usize) -> KDNode {
        if indices.len() <= KD_LEAF_SIZE {
            return KDNode::Leaf { indices };
        }

        let dim = depth % n_dims;

        // Sort by the splitting dimension
        indices.sort_by(|&a, &b| {
            data[[a, dim]].partial_cmp(&data[[b, dim]]).unwrap_or(Ordering::Equal)
        });

        let mid = indices.len() / 2;
        let threshold = data[[indices[mid], dim]];

        let right_indices = indices.split_off(mid);
        let left_indices = indices;

        KDNode::Split {
            dimension: dim,
            threshold,
            left: Box::new(Self::build_node(data, left_indices, depth + 1, n_dims)),
            right: Box::new(Self::build_node(data, right_indices, depth + 1, n_dims)),
        }
    }

    /// Query k nearest neighbors, returns Vec<(distance, index)>
    pub(crate) fn query_k_nearest(
        &self,
        point: &[f64],
        data: &Array2<f64>,
        k: usize,
        metric: DistanceMetric,
    ) -> Vec<(f64, usize)> {
        let mut heap = BinaryHeap::with_capacity(k + 1);
        self.search_node(&self.root, point, data, k, metric, &mut heap);
        heap.into_iter().map(|e| (e.dist, e.idx)).collect()
    }

    /// Query all point indices within `eps` distance of `point` (inclusive).
    /// This is the range-query counterpart to `query_k_nearest`: instead of
    /// keeping the k closest candidates in a heap, it prunes any subtree
    /// whose splitting plane is farther than `eps` away, so a query costs
    /// roughly O(log n + m) (m = number of points returned) rather than the
    /// O(n) brute-force distance scan a naive "for every point, check
    /// distance" loop would need.
    pub(crate) fn query_radius(
        &self,
        point: &[f64],
        data: &Array2<f64>,
        eps: f64,
        metric: DistanceMetric,
    ) -> Vec<usize> {
        let mut out = Vec::new();
        self.search_radius(&self.root, point, data, eps, metric, &mut out);
        out
    }

    fn search_radius(
        &self,
        node: &KDNode,
        point: &[f64],
        data: &Array2<f64>,
        eps: f64,
        metric: DistanceMetric,
        out: &mut Vec<usize>,
    ) {
        match node {
            KDNode::Leaf { indices } => {
                for &idx in indices {
                    let dist = compute_distance(point, data.row(idx).as_slice().unwrap(), metric);
                    if dist <= eps {
                        out.push(idx);
                    }
                }
            }
            KDNode::Split { dimension, threshold, left, right } => {
                let diff = point[*dimension] - threshold;

                // The side containing the query point must always be visited.
                let (near, far) = if diff <= 0.0 {
                    (left.as_ref(), right.as_ref())
                } else {
                    (right.as_ref(), left.as_ref())
                };
                self.search_radius(near, point, data, eps, metric, out);

                // The far side can only contain points within `eps` if the
                // splitting plane itself is within `eps` of the query point.
                if diff.abs() <= eps {
                    self.search_radius(far, point, data, eps, metric, out);
                }
            }
        }
    }

    fn search_node(
        &self,
        node: &KDNode,
        point: &[f64],
        data: &Array2<f64>,
        k: usize,
        metric: DistanceMetric,
        heap: &mut BinaryHeap<HeapEntry>,
    ) {
        match node {
            KDNode::Leaf { indices } => {
                for &idx in indices {
                    let dist = compute_distance(point, data.row(idx).as_slice().unwrap(), metric);
                    if heap.len() < k {
                        heap.push(HeapEntry { dist, idx });
                    } else if let Some(top) = heap.peek() {
                        if dist < top.dist {
                            heap.pop();
                            heap.push(HeapEntry { dist, idx });
                        }
                    }
                }
            }
            KDNode::Split { dimension, threshold, left, right } => {
                let val = point[*dimension];
                let diff = val - threshold;

                // Visit closer child first
                let (first, second) = if diff <= 0.0 {
                    (left.as_ref(), right.as_ref())
                } else {
                    (right.as_ref(), left.as_ref())
                };

                self.search_node(first, point, data, k, metric, heap);

                // Check if we need to visit the far child:
                // Only prune if heap is full and splitting plane distance >= k-th nearest
                let plane_dist = diff.abs();
                let should_visit_far = heap.len() < k
                    || plane_dist < heap.peek().map(|e| e.dist).unwrap_or(f64::INFINITY);

                if should_visit_far {
                    self.search_node(second, point, data, k, metric, heap);
                }
            }
        }
    }
}

/// Returns true if the metric is compatible with KD-tree pruning
fn kd_tree_compatible(metric: DistanceMetric) -> bool {
    matches!(metric, DistanceMetric::Euclidean | DistanceMetric::Manhattan)
}

// ============================================================================
// Shared optimized helpers (used by both Classifier and Regressor)
// ============================================================================

/// Max-heap entry for partial sort (keeps k smallest distances)
#[derive(PartialEq)]
struct DistLabel(f64, f64);

impl Eq for DistLabel {}
impl PartialOrd for DistLabel {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        self.0.partial_cmp(&other.0)
    }
}
impl Ord for DistLabel {
    fn cmp(&self, other: &Self) -> Ordering {
        self.partial_cmp(other).unwrap_or(Ordering::Equal)
    }
}

/// Find k nearest neighbors using a max-heap — O(n log k) instead of O(n log n)
fn find_k_nearest(
    point: &[f64],
    x_train: &Array2<f64>,
    y_train: &Array1<f64>,
    k: usize,
    metric: DistanceMetric,
) -> Vec<(f64, f64)> {
    let mut heap = BinaryHeap::with_capacity(k + 1);

    for (i, row) in x_train.rows().into_iter().enumerate() {
        let dist = compute_distance(point, row.as_slice().unwrap(), metric);
        if heap.len() < k {
            heap.push(DistLabel(dist, y_train[i]));
        } else if let Some(top) = heap.peek() {
            if dist < top.0 {
                heap.pop();
                heap.push(DistLabel(dist, y_train[i]));
            }
        }
    }

    heap.into_iter().map(|dl| (dl.0, dl.1)).collect()
}

/// Compute distance between two points using the specified metric
fn compute_distance(a: &[f64], b: &[f64], metric: DistanceMetric) -> f64 {
    match metric {
        DistanceMetric::Euclidean => {
            a.iter()
                .zip(b.iter())
                .map(|(ai, bi)| {
                    let d = ai - bi;
                    d * d
                })
                .sum::<f64>()
                .sqrt()
        }
        DistanceMetric::Manhattan => {
            a.iter()
                .zip(b.iter())
                .map(|(ai, bi)| (ai - bi).abs())
                .sum()
        }
        DistanceMetric::Minkowski(p) => {
            a.iter()
                .zip(b.iter())
                .map(|(ai, bi)| (ai - bi).abs().powf(p))
                .sum::<f64>()
                .powf(1.0 / p)
        }
        DistanceMetric::Cosine => {
            let mut dot = 0.0;
            let mut norm_a = 0.0;
            let mut norm_b = 0.0;
            for (ai, bi) in a.iter().zip(b.iter()) {
                dot += ai * bi;
                norm_a += ai * ai;
                norm_b += bi * bi;
            }
            let denom = norm_a.sqrt() * norm_b.sqrt();
            if denom > 0.0 { 1.0 - (dot / denom) } else { 1.0 }
        }
    }
}

/// Classify by weighted majority vote
fn vote_classify(neighbors: &[(f64, f64)], _classes: &[i64], weights: WeightScheme) -> f64 {
    let mut votes: HashMap<i64, f64> = HashMap::new();
    for &(dist, label) in neighbors {
        let weight = match weights {
            WeightScheme::Uniform => 1.0,
            WeightScheme::Distance => 1.0 / (dist + 1e-10),
        };
        *votes.entry(label as i64).or_insert(0.0) += weight;
    }
    votes.into_iter()
        .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal))
        .map(|(label, _)| label as f64)
        .unwrap_or(0.0)
}

/// Compute class probabilities
fn class_probs_from(neighbors: &[(f64, f64)], classes: &[i64], n_classes: usize, weights: WeightScheme) -> Vec<f64> {
    let mut counts = vec![0.0; n_classes];
    let mut total = 0.0;
    for &(dist, label) in neighbors {
        let weight = match weights {
            WeightScheme::Uniform => 1.0,
            WeightScheme::Distance => 1.0 / (dist + 1e-10),
        };
        let class_idx = classes.iter().position(|&c| c == label as i64).unwrap_or(0);
        counts[class_idx] += weight;
        total += weight;
    }
    if total > 0.0 {
        counts.iter_mut().for_each(|c| *c /= total);
    }
    counts
}

/// Compute weighted mean for regression
fn weighted_mean_from(neighbors: &[(f64, f64)], weights: WeightScheme) -> f64 {
    match weights {
        WeightScheme::Uniform => {
            neighbors.iter().map(|(_, y)| y).sum::<f64>() / neighbors.len() as f64
        }
        WeightScheme::Distance => {
            let mut weighted_sum = 0.0;
            let mut weight_total = 0.0;
            for &(dist, y) in neighbors {
                let w = 1.0 / (dist + 1e-10);
                weighted_sum += w * y;
                weight_total += w;
            }
            if weight_total > 0.0 { weighted_sum / weight_total }
            else { neighbors.iter().map(|(_, y)| y).sum::<f64>() / neighbors.len() as f64 }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_classification_data() -> (Array2<f64>, Array1<f64>) {
        // Create linearly separable data
        let x = Array2::from_shape_vec((20, 2), vec![
            // Class 0 (low values)
            1.0, 1.0, 1.5, 1.5, 2.0, 2.0, 2.5, 2.5, 1.0, 2.0,
            1.5, 2.5, 2.0, 1.5, 2.5, 1.0, 1.2, 1.8, 1.8, 1.2,
            // Class 1 (high values)
            8.0, 8.0, 8.5, 8.5, 9.0, 9.0, 9.5, 9.5, 8.0, 9.0,
            8.5, 9.5, 9.0, 8.5, 9.5, 8.0, 8.2, 8.8, 8.8, 8.2,
        ]).unwrap();
        
        let y = Array1::from_vec(vec![
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
        ]);
        
        (x, y)
    }

    fn create_regression_data() -> (Array2<f64>, Array1<f64>) {
        let x = Array2::from_shape_vec((10, 2), 
            (0..20).map(|i| i as f64).collect()
        ).unwrap();
        
        let y: Array1<f64> = x.rows().into_iter()
            .map(|row| row[0] + row[1])
            .collect();
        
        (x, y)
    }

    #[test]
    fn test_knn_classifier() {
        let (x, y) = create_classification_data();
        
        let mut knn = KNNClassifier::with_k(3);
        knn.fit(&x, &y).unwrap();
        
        let predictions = knn.predict(&x);
        
        // Check accuracy (should be perfect for this separable data)
        let correct: usize = y.iter()
            .zip(predictions.iter())
            .filter(|(&yi, &pi)| (yi - pi).abs() < 0.5)
            .count();
        
        let accuracy = correct as f64 / y.len() as f64;
        assert!(accuracy > 0.9, "Accuracy ({}) should be above 90%", accuracy);
    }

    #[test]
    fn test_knn_regressor() {
        let (x, y) = create_regression_data();
        
        let mut knn = KNNRegressor::with_k(3);
        knn.fit(&x, &y).unwrap();
        
        let predictions = knn.predict(&x);
        
        // Check that predictions are reasonable
        let mse: f64 = y.iter()
            .zip(predictions.iter())
            .map(|(yi, pi)| (yi - pi).powi(2))
            .sum::<f64>() / y.len() as f64;
        
        // MSE should be low for this simple data
        assert!(mse < 10.0, "MSE ({}) should be low", mse);
    }

    #[test]
    fn test_distance_metrics() {
        let a = Array1::from_vec(vec![0.0, 0.0]);
        let b = Array1::from_vec(vec![3.0, 4.0]);
        
        let knn = KNNClassifier::new(KNNConfig {
            metric: DistanceMetric::Euclidean,
            ..Default::default()
        });
        
        let dist = knn.distance(&a, &b);
        assert!((dist - 5.0).abs() < 0.001, "Euclidean distance should be 5.0");
    }

    #[test]
    fn test_weighted_knn() {
        let (x, y) = create_classification_data();

        let mut knn = KNNClassifier::new(KNNConfig {
            n_neighbors: 5,
            weights: WeightScheme::Distance,
            ..Default::default()
        });
        knn.fit(&x, &y).unwrap();

        let predictions = knn.predict(&x);
        assert_eq!(predictions.len(), 20);
    }

    #[test]
    fn test_knn_high_dimensional_projection() {
        // Create high-dimensional data (30 features > HIGH_DIM_THRESHOLD=16)
        let n_samples = 40;
        let n_features = 30;
        let mut data = vec![0.0; n_samples * n_features];
        for i in 0..n_samples {
            for j in 0..n_features {
                // Class 0: features centered around 0, Class 1: centered around 5
                let base = if i < n_samples / 2 { 0.0 } else { 5.0 };
                data[i * n_features + j] = base + (((i * 7 + j * 13) % 100) as f64) * 0.01;
            }
        }
        let x = Array2::from_shape_vec((n_samples, n_features), data).unwrap();
        let mut labels = vec![0.0; n_samples / 2];
        labels.extend(vec![1.0; n_samples / 2]);
        let y = Array1::from_vec(labels);

        let mut knn = KNNClassifier::with_k(3);
        knn.fit(&x, &y).unwrap();

        // Projection should be active
        assert!(knn.projection.is_some());
        assert!(knn.x_train_projected.is_some());

        let predictions = knn.predict(&x);
        let correct: usize = y.iter()
            .zip(predictions.iter())
            .filter(|(&yi, &pi)| (yi - pi).abs() < 0.5)
            .count();
        let accuracy = correct as f64 / y.len() as f64;
        assert!(accuracy > 0.8, "High-dim KNN accuracy ({}) should be above 80%", accuracy);
    }

    #[test]
    fn test_knn_low_dim_no_projection() {
        let (x, y) = create_classification_data();
        let mut knn = KNNClassifier::with_k(3);
        knn.fit(&x, &y).unwrap();
        // 2D data should NOT trigger projection
        assert!(knn.projection.is_none());
        assert!(knn.x_train_projected.is_none());
    }
}
