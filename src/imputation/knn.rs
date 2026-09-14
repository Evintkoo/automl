//! KNN-based imputation

use crate::error::{AutoMLError, Result};
use crate::imputation::{Imputer, is_missing};
use ndarray::{Array1, Array2};
use serde::{Deserialize, Serialize};
use std::collections::BinaryHeap;
use std::cmp::Ordering;

/// Ordered float for priority queue
#[derive(Debug, Clone, Copy)]
struct DistanceIdx(f64, usize);

impl PartialEq for DistanceIdx {
    fn eq(&self, other: &Self) -> bool {
        self.0 == other.0
    }
}

impl Eq for DistanceIdx {}

impl PartialOrd for DistanceIdx {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for DistanceIdx {
    fn cmp(&self, other: &Self) -> Ordering {
        // Max heap by distance (we want to pop largest distances)
        self.0.partial_cmp(&other.0).unwrap_or(Ordering::Equal)
    }
}

/// KNN-based imputer
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KNNImputer {
    /// Number of neighbors
    n_neighbors: usize,
    /// Distance metric ("euclidean" or "manhattan")
    metric: String,
    /// Weights for averaging ("uniform" or "distance")
    weights: String,
    /// Training data (complete rows only)
    complete_data: Option<Array2<f64>>,
    /// Feature means for fallback
    feature_means: Option<Array1<f64>>,
}

impl KNNImputer {
    /// Create new KNN imputer
    pub fn new(n_neighbors: usize) -> Self {
        Self {
            n_neighbors: n_neighbors.max(1),
            metric: "euclidean".to_string(),
            weights: "uniform".to_string(),
            complete_data: None,
            feature_means: None,
        }
    }

    /// Set distance metric
    pub fn with_metric(mut self, metric: &str) -> Self {
        self.metric = metric.to_lowercase();
        self
    }

    /// Set weighting scheme
    pub fn with_weights(mut self, weights: &str) -> Self {
        self.weights = weights.to_lowercase();
        self
    }

    /// Compute distance between two samples (ignoring NaN positions)
    fn distance(&self, a: &[f64], b: &[f64]) -> f64 {
        let pairs: Vec<(f64, f64)> = a.iter()
            .zip(b.iter())
            .filter(|(&ai, &bi)| !is_missing(ai) && !is_missing(bi))
            .map(|(&ai, &bi)| (ai, bi))
            .collect();

        if pairs.is_empty() {
            return f64::INFINITY;
        }

        match self.metric.as_str() {
            "manhattan" => {
                let sum: f64 = pairs.iter().map(|(ai, bi)| (ai - bi).abs()).sum();
                sum / pairs.len() as f64 // Normalize by number of features
            }
            _ => {
                // Euclidean (default)
                let sum_sq: f64 = pairs.iter().map(|(ai, bi)| (ai - bi).powi(2)).sum();
                (sum_sq / pairs.len() as f64).sqrt()
            }
        }
    }

    /// Find k nearest neighbors in complete data
    fn find_neighbors(&self, sample: &[f64], k: usize) -> Vec<(usize, f64)> {
        let data = self.complete_data.as_ref().unwrap();
        let mut heap: BinaryHeap<DistanceIdx> = BinaryHeap::new();

        for (i, row) in data.rows().into_iter().enumerate() {
            let row_vec: Vec<f64> = row.iter().copied().collect();
            let dist = self.distance(sample, &row_vec);

            if dist.is_finite() {
                if heap.len() < k {
                    heap.push(DistanceIdx(dist, i));
                } else if let Some(&DistanceIdx(max_dist, _)) = heap.peek() {
                    if dist < max_dist {
                        heap.pop();
                        heap.push(DistanceIdx(dist, i));
                    }
                }
            }
        }

        heap.into_iter()
            .map(|DistanceIdx(d, i)| (i, d))
            .collect()
    }

    /// Impute missing value using neighbors
    fn impute_value(&self, neighbors: &[(usize, f64)], feature_idx: usize) -> f64 {
        let data = self.complete_data.as_ref().unwrap();

        if neighbors.is_empty() {
            // Fall back to mean
            return self.feature_means.as_ref()
                .map(|m| m[feature_idx])
                .unwrap_or(0.0);
        }

        match self.weights.as_str() {
            "distance" => {
                // Inverse distance weighting
                let mut weighted_sum = 0.0;
                let mut weight_sum = 0.0;

                for &(idx, dist) in neighbors {
                    let weight = if dist < 1e-10 { 1e10 } else { 1.0 / dist };
                    weighted_sum += data[[idx, feature_idx]] * weight;
                    weight_sum += weight;
                }

                if weight_sum > 0.0 {
                    weighted_sum / weight_sum
                } else {
                    self.feature_means.as_ref()
                        .map(|m| m[feature_idx])
                        .unwrap_or(0.0)
                }
            }
            _ => {
                // Uniform weighting (simple average)
                let sum: f64 = neighbors.iter()
                    .map(|&(idx, _)| data[[idx, feature_idx]])
                    .sum();
                sum / neighbors.len() as f64
            }
        }
    }
}

impl Default for KNNImputer {
    fn default() -> Self {
        Self::new(5)
    }
}

impl Imputer for KNNImputer {
    fn fit(&mut self, x: &Array2<f64>) -> Result<()> {
        // Store complete rows only
        let complete_rows: Vec<usize> = x.rows()
            .into_iter()
            .enumerate()
            .filter(|(_, row)| !row.iter().any(|&v| is_missing(v)))
            .map(|(i, _)| i)
            .collect();

        if complete_rows.is_empty() {
            return Err(AutoMLError::ValidationError(
                "No complete rows found for KNN imputation".to_string()
            ));
        }

        let n_features = x.ncols();
        let mut complete_data = Array2::zeros((complete_rows.len(), n_features));

        for (i, &row_idx) in complete_rows.iter().enumerate() {
            for j in 0..n_features {
                complete_data[[i, j]] = x[[row_idx, j]];
            }
        }

        // Compute feature means from complete data
        let feature_means = complete_data.mean_axis(ndarray::Axis(0))
            .ok_or_else(|| AutoMLError::ValidationError("Failed to compute means".to_string()))?;

        self.complete_data = Some(complete_data);
        self.feature_means = Some(feature_means);

        Ok(())
    }

    fn transform(&self, x: &Array2<f64>) -> Result<Array2<f64>> {
        if self.complete_data.is_none() {
            return Err(AutoMLError::ValidationError(
                "Imputer not fitted".to_string()
            ));
        }

        let mut result = x.clone();
        let _n_features = x.ncols();

        for (row_idx, row) in x.rows().into_iter().enumerate() {
            let row_vec: Vec<f64> = row.iter().copied().collect();
            
            // Find missing features in this row
            let missing_features: Vec<usize> = row_vec.iter()
                .enumerate()
                .filter(|(_, &v)| is_missing(v))
                .map(|(i, _)| i)
                .collect();

            if missing_features.is_empty() {
                continue;
            }

            // Find neighbors
            let neighbors = self.find_neighbors(&row_vec, self.n_neighbors);

            // Impute each missing feature
            for &feat_idx in &missing_features {
                let value = self.impute_value(&neighbors, feat_idx);
                result[[row_idx, feat_idx]] = value;
            }
        }

        Ok(result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_knn_imputer_basic() {
        let data = Array2::from_shape_vec(
            (6, 2),
            vec![
                1.0, 10.0,
                2.0, 20.0,
                3.0, 30.0,
                4.0, 40.0,
                f64::NAN, 25.0, // Missing first feature
                2.5, f64::NAN, // Missing second feature
            ],
        ).unwrap();

        let mut imputer = KNNImputer::new(3);
        let result = imputer.fit_transform(&data).unwrap();

        // Check no NaN values remain
        assert!(!result.iter().any(|&v| v.is_nan()));

        // Imputed values should be reasonable (between 1-4 for first col, 10-40 for second)
        assert!(result[[4, 0]] >= 1.0 && result[[4, 0]] <= 4.0);
        assert!(result[[5, 1]] >= 10.0 && result[[5, 1]] <= 40.0);
    }

    #[test]
    fn test_knn_imputer_distance_weights() {
        let data = Array2::from_shape_vec(
            (5, 2),
            vec![
                0.0, 0.0,
                1.0, 1.0,
                2.0, 2.0,
                3.0, 3.0,
                0.1, f64::NAN, // Very close to first row
            ],
        ).unwrap();

        let mut imputer = KNNImputer::new(3)
            .with_weights("distance");

        let result = imputer.fit_transform(&data).unwrap();

        // Should be close to 0.0 (first row's value) due to distance weighting
        assert!(result[[4, 1]].abs() < 1.0);
    }

    #[test]
    fn test_knn_imputer_manhattan() {
        let data = Array2::from_shape_vec(
            (4, 2),
            vec![
                1.0, 1.0,
                2.0, 2.0,
                3.0, 3.0,
                1.5, f64::NAN,
            ],
        ).unwrap();

        let mut imputer = KNNImputer::new(2)
            .with_metric("manhattan");

        let result = imputer.fit_transform(&data).unwrap();
        assert!(!result.iter().any(|&v| v.is_nan()));
    }
}
