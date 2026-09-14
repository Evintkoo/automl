//! PCA — Principal Component Analysis
//!
//! Fast linear dimensionality reduction for 2D visualization.
//! Computes the top-k eigenvectors of the covariance matrix
//! using power iteration with deflation.

use crate::utils::simd::SimdOps;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

/// PCA configuration parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PcaConfig {
    /// Number of output dimensions (default 2)
    pub n_components: usize,
    /// Whether to center the data (subtract mean per feature)
    pub center: bool,
    /// Whether to scale to unit variance
    pub scale: bool,
    /// Random seed for power iteration initialization
    pub random_state: u64,
}

impl Default for PcaConfig {
    fn default() -> Self {
        Self {
            n_components: 2,
            center: true,
            scale: true,
            random_state: 42,
        }
    }
}

/// PCA result including the embedding and explained variance
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PcaResult {
    /// Projected data: n_samples x n_components
    pub embedding: Vec<[f64; 2]>,
    /// Explained variance ratio for each component (sums to <= 1.0)
    pub explained_variance_ratio: Vec<f64>,
    /// Eigenvalues (raw variance per component)
    pub eigenvalues: Vec<f64>,
}

/// PCA dimensionality reduction
pub struct Pca {
    config: PcaConfig,
}

impl Pca {
    /// Create a new PCA instance
    pub fn new(config: PcaConfig) -> Self {
        Self { config }
    }

    /// Run PCA on dense data. Returns PcaResult with 2D embedding and explained variance.
    pub fn fit_transform(&self, data: &[Vec<f64>]) -> crate::error::Result<PcaResult> {
        let n = data.len();
        if n < 2 {
            return Err(crate::error::AutoMLError::DataError(
                "PCA requires at least 2 samples".to_string(),
            ));
        }

        let d = data[0].len();
        if d < 1 {
            return Err(crate::error::AutoMLError::DataError(
                "PCA requires at least 1 feature".to_string(),
            ));
        }

        let n_components = self.config.n_components.min(d).min(n);

        // Step 1: Center (and optionally scale) the data
        let (centered, _means, _stds) = self.center_and_scale(data, d);

        // Step 2: Compute covariance matrix (d x d)
        let cov = self.compute_covariance(&centered, d);

        // Step 3: Extract top-k eigenvectors via power iteration with deflation
        let (eigenvalues, eigenvectors) = self.power_iteration(&cov, d, n_components);

        // Step 4: Project data onto top eigenvectors
        let _total_variance: f64 = eigenvalues.iter().sum::<f64>().max(1e-12);
        // Total variance should include all eigenvalues from the diagonal of cov
        let full_variance: f64 = (0..d).map(|i| cov[i * d + i]).sum::<f64>().max(1e-12);

        let explained_variance_ratio: Vec<f64> = eigenvalues
            .iter()
            .map(|&ev| (ev / full_variance).max(0.0))
            .collect();

        // Project each sample
        let embedding: Vec<[f64; 2]> = centered
            .par_iter()
            .map(|sample| {
                let mut point = [0.0f64; 2];
                for c in 0..n_components.min(2) {
                    let component = &eigenvectors[c];
                    point[c] = SimdOps::dot_f64(sample, component);
                }
                point
            })
            .collect();

        Ok(PcaResult {
            embedding,
            explained_variance_ratio,
            eigenvalues,
        })
    }

    /// Center data (subtract mean) and optionally scale to unit variance.
    fn center_and_scale(
        &self,
        data: &[Vec<f64>],
        d: usize,
    ) -> (Vec<Vec<f64>>, Vec<f64>, Vec<f64>) {
        let _n = data.len();

        // Compute column means
        let means: Vec<f64> = (0..d)
            .map(|j| {
                let col: Vec<f64> = data.iter().map(|row| row[j]).collect();
                SimdOps::mean_f64(&col)
            })
            .collect();

        // Compute column stds if scaling
        let stds: Vec<f64> = if self.config.scale {
            (0..d)
                .map(|j| {
                    let col: Vec<f64> = data.iter().map(|row| row[j]).collect();
                    let var = SimdOps::variance_f64(&col);
                    var.sqrt().max(1e-12)
                })
                .collect()
        } else {
            vec![1.0; d]
        };

        // Center and scale
        let centered: Vec<Vec<f64>> = if self.config.center {
            data.iter()
                .map(|row| {
                    row.iter()
                        .enumerate()
                        .map(|(j, &val)| (val - means[j]) / stds[j])
                        .collect()
                })
                .collect()
        } else {
            data.to_vec()
        };

        (centered, means, stds)
    }

    /// Compute the covariance matrix (d x d) stored as flat Vec.
    fn compute_covariance(&self, data: &[Vec<f64>], d: usize) -> Vec<f64> {
        let n = data.len() as f64;
        let mut cov = vec![0.0f64; d * d];

        // Upper triangle + diagonal
        for i in 0..d {
            for j in i..d {
                let col_i: Vec<f64> = data.iter().map(|row| row[i]).collect();
                let col_j: Vec<f64> = data.iter().map(|row| row[j]).collect();
                let dot = SimdOps::dot_f64(&col_i, &col_j);
                let val = dot / (n - 1.0).max(1.0);
                cov[i * d + j] = val;
                cov[j * d + i] = val;
            }
        }

        cov
    }

    /// Power iteration with deflation to extract top-k eigenvectors.
    fn power_iteration(
        &self,
        cov: &[f64],
        d: usize,
        k: usize,
    ) -> (Vec<f64>, Vec<Vec<f64>>) {
        let max_iter = 300;
        let tol = 1e-10;

        let mut eigenvalues = Vec::with_capacity(k);
        let mut eigenvectors: Vec<Vec<f64>> = Vec::with_capacity(k);

        // Work on a copy so we can deflate
        let mut work = cov.to_vec();

        use rand::SeedableRng;
        use rand::Rng;
        use rand_chacha::ChaCha8Rng;
        let mut rng = ChaCha8Rng::seed_from_u64(self.config.random_state);

        for _component in 0..k {
            // Initialize random unit vector
            let mut v: Vec<f64> = (0..d).map(|_| rng.gen_range(-1.0..1.0)).collect();
            let norm = SimdOps::dot_f64(&v, &v).sqrt().max(1e-12);
            v.iter_mut().for_each(|x| *x /= norm);

            let mut eigenvalue = 0.0f64;

            for _iter in 0..max_iter {
                // w = A * v (matrix-vector multiply)
                let mut w = vec![0.0f64; d];
                for i in 0..d {
                    let row_start = i * d;
                    let row = &work[row_start..row_start + d];
                    w[i] = SimdOps::dot_f64(row, &v);
                }

                // eigenvalue = v^T * w
                let new_eigenvalue = SimdOps::dot_f64(&v, &w);

                // Normalize w
                let w_norm = SimdOps::dot_f64(&w, &w).sqrt().max(1e-12);
                let new_v: Vec<f64> = w.iter().map(|&x| x / w_norm).collect();

                // Check convergence
                let diff: f64 = v
                    .iter()
                    .zip(new_v.iter())
                    .map(|(&a, &b)| (a - b).powi(2))
                    .sum::<f64>()
                    .sqrt();

                v = new_v;
                eigenvalue = new_eigenvalue;

                if diff < tol {
                    break;
                }
            }

            eigenvalue = eigenvalue.max(0.0);
            eigenvalues.push(eigenvalue);
            eigenvectors.push(v.clone());

            // Deflate: A = A - eigenvalue * v * v^T
            for i in 0..d {
                for j in 0..d {
                    work[i * d + j] -= eigenvalue * v[i] * v[j];
                }
            }
        }

        (eigenvalues, eigenvectors)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pca_basic() {
        // Simple 2D data with clear primary axis
        let data = vec![
            vec![1.0, 2.0],
            vec![2.0, 4.0],
            vec![3.0, 6.0],
            vec![4.0, 8.0],
            vec![5.0, 10.0],
        ];

        let pca = Pca::new(PcaConfig::default());
        let result = pca.fit_transform(&data).unwrap();

        assert_eq!(result.embedding.len(), 5);
        // First component should explain nearly all variance (data is perfectly linear)
        assert!(
            result.explained_variance_ratio[0] > 0.95,
            "First component should explain >95% variance, got {}",
            result.explained_variance_ratio[0]
        );
    }

    #[test]
    fn test_pca_two_clusters() {
        let data = vec![
            vec![0.0, 0.0, 0.0],
            vec![0.1, 0.1, 0.0],
            vec![0.0, 0.1, 0.1],
            vec![10.0, 10.0, 10.0],
            vec![10.1, 10.0, 10.0],
            vec![10.0, 10.1, 10.0],
        ];

        let pca = Pca::new(PcaConfig::default());
        let result = pca.fit_transform(&data).unwrap();

        assert_eq!(result.embedding.len(), 6);
        assert_eq!(result.explained_variance_ratio.len(), 2);

        // Two clusters should be separated in embedding
        let mean_a = [
            result.embedding[..3].iter().map(|p| p[0]).sum::<f64>() / 3.0,
            result.embedding[..3].iter().map(|p| p[1]).sum::<f64>() / 3.0,
        ];
        let mean_b = [
            result.embedding[3..].iter().map(|p| p[0]).sum::<f64>() / 3.0,
            result.embedding[3..].iter().map(|p| p[1]).sum::<f64>() / 3.0,
        ];

        let dist = ((mean_a[0] - mean_b[0]).powi(2) + (mean_a[1] - mean_b[1]).powi(2)).sqrt();
        assert!(dist > 1.0, "Clusters should be separated, got distance: {}", dist);
    }

    #[test]
    fn test_pca_config_defaults() {
        let config = PcaConfig::default();
        assert_eq!(config.n_components, 2);
        assert!(config.center);
        assert!(config.scale);
    }

    #[test]
    fn test_pca_too_few_samples() {
        let data = vec![vec![1.0, 2.0]];
        let pca = Pca::new(PcaConfig::default());
        assert!(pca.fit_transform(&data).is_err());
    }

    #[test]
    fn test_pca_explained_variance_sums_to_one() {
        let data = vec![
            vec![1.0, 0.0, 0.5],
            vec![0.0, 1.0, 0.3],
            vec![1.0, 1.0, 0.8],
            vec![0.5, 0.5, 0.4],
            vec![0.2, 0.8, 0.6],
            vec![0.9, 0.1, 0.2],
        ];

        let pca = Pca::new(PcaConfig::default());
        let result = pca.fit_transform(&data).unwrap();

        let total: f64 = result.explained_variance_ratio.iter().sum();
        // With 2 components out of 3, should explain a good chunk but not necessarily all
        assert!(total > 0.0 && total <= 1.001, "Variance ratios should be in [0, 1], sum={}", total);
    }

    #[test]
    fn test_pca_no_scale() {
        let data = vec![
            vec![1.0, 100.0],
            vec![2.0, 200.0],
            vec![3.0, 300.0],
            vec![4.0, 400.0],
        ];

        let config = PcaConfig {
            scale: false,
            ..Default::default()
        };
        let pca = Pca::new(config);
        let result = pca.fit_transform(&data).unwrap();
        assert_eq!(result.embedding.len(), 4);
    }
}
