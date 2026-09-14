//! Polynomial feature generation

use crate::error::{AutoMLError, Result};
use crate::feature_engineering::FeatureTransformer;
use ndarray::Array2;
use serde::{Deserialize, Serialize};

/// Configuration for polynomial features
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PolynomialConfig {
    /// Maximum polynomial degree
    pub degree: usize,
    /// Include bias term (constant 1)
    pub include_bias: bool,
    /// Include interaction terms only (no powers > 1 of same feature)
    pub interaction_only: bool,
}

impl Default for PolynomialConfig {
    fn default() -> Self {
        Self {
            degree: 2,
            include_bias: true,
            interaction_only: false,
        }
    }
}

/// Polynomial feature generator
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PolynomialFeatures {
    /// Configuration
    config: PolynomialConfig,
    /// Number of input features
    n_features_in: Option<usize>,
    /// Feature names
    feature_names: Option<Vec<String>>,
    /// Output feature combinations
    combinations: Option<Vec<Vec<usize>>>,
}

impl PolynomialFeatures {
    /// Create new polynomial feature generator
    pub fn new(degree: usize) -> Self {
        Self {
            config: PolynomialConfig {
                degree: degree.max(1),
                ..Default::default()
            },
            n_features_in: None,
            feature_names: None,
            combinations: None,
        }
    }

    /// Set configuration
    pub fn with_config(mut self, config: PolynomialConfig) -> Self {
        self.config = config;
        self
    }

    /// Include bias term
    pub fn with_bias(mut self, include: bool) -> Self {
        self.config.include_bias = include;
        self
    }

    /// Set interaction only mode
    pub fn interaction_only(mut self, only: bool) -> Self {
        self.config.interaction_only = only;
        self
    }

    /// Set input feature names
    pub fn with_feature_names(mut self, names: Vec<String>) -> Self {
        self.feature_names = Some(names);
        self
    }

    /// Generate all combinations of features for polynomial expansion
    fn generate_combinations(&self, n_features: usize) -> Vec<Vec<usize>> {
        let mut combinations = Vec::new();

        // Add bias (empty combination = constant 1)
        if self.config.include_bias {
            combinations.push(Vec::new());
        }

        // Generate combinations for each degree
        for d in 1..=self.config.degree {
            self.generate_combinations_recursive(
                n_features,
                d,
                0,
                &mut Vec::new(),
                &mut combinations,
            );
        }

        combinations
    }

    /// Recursive helper for generating combinations
    fn generate_combinations_recursive(
        &self,
        n_features: usize,
        remaining_degree: usize,
        start_idx: usize,
        current: &mut Vec<usize>,
        result: &mut Vec<Vec<usize>>,
    ) {
        if remaining_degree == 0 {
            // Check interaction_only constraint
            if self.config.interaction_only {
                let mut counts = vec![0; n_features];
                for &idx in current.iter() {
                    counts[idx] += 1;
                    if counts[idx] > 1 {
                        return; // Skip combinations with repeated features
                    }
                }
            }
            result.push(current.clone());
            return;
        }

        for i in start_idx..n_features {
            current.push(i);
            self.generate_combinations_recursive(
                n_features,
                remaining_degree - 1,
                i, // Allow same feature again for powers
                current,
                result,
            );
            current.pop();
        }
    }

    /// Compute feature value for a combination
    fn compute_combination(&self, row: &[f64], combination: &[usize]) -> f64 {
        if combination.is_empty() {
            return 1.0; // Bias term
        }
        combination.iter().map(|&i| row[i]).product()
    }

    /// Generate feature name for a combination
    fn combination_name(&self, combination: &[usize]) -> String {
        if combination.is_empty() {
            return "1".to_string();
        }

        let names = self.feature_names.as_ref();
        
        // Count occurrences of each feature
        let mut counts: std::collections::HashMap<usize, usize> = std::collections::HashMap::new();
        for &idx in combination {
            *counts.entry(idx).or_insert(0) += 1;
        }

        let mut sorted_indices: Vec<usize> = counts.keys().copied().collect();
        sorted_indices.sort();

        let parts: Vec<String> = sorted_indices
            .iter()
            .map(|&idx| {
                let count = counts[&idx];
                let name = names
                    .map(|n| n[idx].clone())
                    .unwrap_or_else(|| format!("x{}", idx));
                
                if count == 1 {
                    name
                } else {
                    format!("{}^{}", name, count)
                }
            })
            .collect();

        parts.join(" * ")
    }
}

impl Default for PolynomialFeatures {
    fn default() -> Self {
        Self::new(2)
    }
}

impl FeatureTransformer for PolynomialFeatures {
    fn fit(&mut self, x: &Array2<f64>) -> Result<()> {
        self.n_features_in = Some(x.ncols());
        self.combinations = Some(self.generate_combinations(x.ncols()));
        Ok(())
    }

    fn transform(&self, x: &Array2<f64>) -> Result<Array2<f64>> {
        let combinations = self.combinations.as_ref().ok_or_else(|| {
            AutoMLError::ValidationError("Transformer not fitted".to_string())
        })?;

        let n_samples = x.nrows();
        let n_output = combinations.len();

        let mut result = Array2::zeros((n_samples, n_output));

        for (i, row) in x.rows().into_iter().enumerate() {
            let row_vec: Vec<f64> = row.iter().copied().collect();
            for (j, combo) in combinations.iter().enumerate() {
                result[[i, j]] = self.compute_combination(&row_vec, combo);
            }
        }

        Ok(result)
    }

    fn get_feature_names(&self) -> Vec<String> {
        self.combinations
            .as_ref()
            .map(|combos| combos.iter().map(|c| self.combination_name(c)).collect())
            .unwrap_or_default()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_polynomial_degree_2() {
        let x = Array2::from_shape_vec(
            (3, 2),
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        ).unwrap();

        let mut poly = PolynomialFeatures::new(2)
            .with_bias(true);

        let result = poly.fit_transform(&x).unwrap();

        // For 2 features, degree 2: 1, x0, x1, x0^2, x0*x1, x1^2 = 6 features
        assert_eq!(result.ncols(), 6);

        // Check first row: [1.0, 2.0]
        // Expected: [1, 1, 2, 1, 2, 4] = [bias, x0, x1, x0^2, x0*x1, x1^2]
        assert_eq!(result[[0, 0]], 1.0); // bias
        assert_eq!(result[[0, 1]], 1.0); // x0
        assert_eq!(result[[0, 2]], 2.0); // x1
    }

    #[test]
    fn test_polynomial_interaction_only() {
        let x = Array2::from_shape_vec(
            (2, 3),
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        ).unwrap();

        let mut poly = PolynomialFeatures::new(2)
            .with_bias(false)
            .interaction_only(true);

        let result = poly.fit_transform(&x).unwrap();

        // For 3 features, degree 2, interaction only:
        // x0, x1, x2, x0*x1, x0*x2, x1*x2 = 6 features (no x0^2, x1^2, x2^2)
        assert_eq!(result.ncols(), 6);
    }

    #[test]
    fn test_feature_names() {
        let x = Array2::from_shape_vec(
            (2, 2),
            vec![1.0, 2.0, 3.0, 4.0],
        ).unwrap();

        let mut poly = PolynomialFeatures::new(2)
            .with_bias(true)
            .with_feature_names(vec!["a".to_string(), "b".to_string()]);

        poly.fit(&x).unwrap();
        let names = poly.get_feature_names();

        assert!(names.contains(&"1".to_string())); // bias
        assert!(names.contains(&"a".to_string()));
        assert!(names.contains(&"b".to_string()));
        assert!(names.contains(&"a^2".to_string()));
        assert!(names.contains(&"a * b".to_string()));
        assert!(names.contains(&"b^2".to_string()));
    }
}
