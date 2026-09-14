//! Feature interaction generation

use crate::error::{AutoMLError, Result};
use crate::feature_engineering::FeatureTransformer;
use ndarray::Array2;
use serde::{Deserialize, Serialize};

/// Type of interaction to generate
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum InteractionType {
    /// Multiplication: x1 * x2
    Multiply,
    /// Addition: x1 + x2
    Add,
    /// Subtraction: x1 - x2
    Subtract,
    /// Division: x1 / x2 (with epsilon for safety)
    Divide,
    /// Absolute difference: |x1 - x2|
    AbsDiff,
    /// Maximum: max(x1, x2)
    Max,
    /// Minimum: min(x1, x2)
    Min,
    /// Mean: (x1 + x2) / 2
    Mean,
}

impl InteractionType {
    /// Apply interaction operation
    pub fn apply(&self, a: f64, b: f64) -> f64 {
        match self {
            InteractionType::Multiply => a * b,
            InteractionType::Add => a + b,
            InteractionType::Subtract => a - b,
            InteractionType::Divide => a / (b + 1e-10),
            InteractionType::AbsDiff => (a - b).abs(),
            InteractionType::Max => a.max(b),
            InteractionType::Min => a.min(b),
            InteractionType::Mean => (a + b) / 2.0,
        }
    }

    /// Get operation symbol for naming
    pub fn symbol(&self) -> &'static str {
        match self {
            InteractionType::Multiply => "*",
            InteractionType::Add => "+",
            InteractionType::Subtract => "-",
            InteractionType::Divide => "/",
            InteractionType::AbsDiff => "absdiff",
            InteractionType::Max => "max",
            InteractionType::Min => "min",
            InteractionType::Mean => "mean",
        }
    }
}

/// Feature interaction pair
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeatureCrossing {
    /// First feature index
    pub feature_a: usize,
    /// Second feature index
    pub feature_b: usize,
    /// Interaction type
    pub interaction: InteractionType,
}

/// Feature interaction generator
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeatureInteractions {
    /// Types of interactions to generate
    interaction_types: Vec<InteractionType>,
    /// Only generate specified crossings (if None, all pairs)
    specified_crossings: Option<Vec<(usize, usize)>>,
    /// Include original features in output
    include_original: bool,
    /// Number of input features
    n_features_in: Option<usize>,
    /// Generated feature names
    feature_names_in: Option<Vec<String>>,
    /// Output feature crossings
    crossings: Option<Vec<FeatureCrossing>>,
}

impl FeatureInteractions {
    /// Create new feature interaction generator
    pub fn new(interaction_types: Vec<InteractionType>) -> Self {
        Self {
            interaction_types,
            specified_crossings: None,
            include_original: true,
            n_features_in: None,
            feature_names_in: None,
            crossings: None,
        }
    }

    /// Create with default multiplication interactions
    pub fn multiplicative() -> Self {
        Self::new(vec![InteractionType::Multiply])
    }

    /// Create with all arithmetic interactions
    pub fn arithmetic() -> Self {
        Self::new(vec![
            InteractionType::Add,
            InteractionType::Subtract,
            InteractionType::Multiply,
            InteractionType::Divide,
        ])
    }

    /// Set specific feature pairs to cross
    pub fn with_crossings(mut self, pairs: Vec<(usize, usize)>) -> Self {
        self.specified_crossings = Some(pairs);
        self
    }

    /// Include original features in output
    pub fn with_original(mut self, include: bool) -> Self {
        self.include_original = include;
        self
    }

    /// Set input feature names
    pub fn with_feature_names(mut self, names: Vec<String>) -> Self {
        self.feature_names_in = Some(names);
        self
    }

    /// Generate all feature pairs
    fn generate_pairs(&self, n_features: usize) -> Vec<(usize, usize)> {
        if let Some(ref pairs) = self.specified_crossings {
            pairs
                .iter()
                .filter(|&&(a, b)| a < n_features && b < n_features && a != b)
                .copied()
                .collect()
        } else {
            let mut pairs = Vec::new();
            for i in 0..n_features {
                for j in (i + 1)..n_features {
                    pairs.push((i, j));
                }
            }
            pairs
        }
    }
}

impl Default for FeatureInteractions {
    fn default() -> Self {
        Self::multiplicative()
    }
}

impl FeatureTransformer for FeatureInteractions {
    fn fit(&mut self, x: &Array2<f64>) -> Result<()> {
        let n_features = x.ncols();
        self.n_features_in = Some(n_features);

        let pairs = self.generate_pairs(n_features);

        let mut crossings = Vec::new();
        for (a, b) in pairs {
            for &interaction in &self.interaction_types {
                crossings.push(FeatureCrossing {
                    feature_a: a,
                    feature_b: b,
                    interaction,
                });
            }
        }

        self.crossings = Some(crossings);
        Ok(())
    }

    fn transform(&self, x: &Array2<f64>) -> Result<Array2<f64>> {
        use ndarray::{s, Array1};
        use rayon::prelude::*;

        let crossings = self.crossings.as_ref().ok_or_else(|| {
            AutoMLError::ValidationError("Transformer not fitted".to_string())
        })?;

        let n_original = if self.include_original { x.ncols() } else { 0 };
        let n_output = n_original + crossings.len();
        let mut result = Array2::zeros((x.nrows(), n_output));

        // Copy original columns with column-slice assignment (cache-friendly)
        if self.include_original {
            for j in 0..x.ncols() {
                result.slice_mut(s![.., j]).assign(&x.slice(s![.., j]));
            }
        }

        // Compute interaction columns in parallel, collecting one Vec<f64> per crossing
        let interaction_cols: Vec<Array1<f64>> = crossings
            .par_iter()
            .map(|crossing| {
                let col_a = x.slice(s![.., crossing.feature_a]);
                let col_b = x.slice(s![.., crossing.feature_b]);
                Array1::from_iter(
                    col_a.iter().zip(col_b.iter()).map(|(&a, &b)| crossing.interaction.apply(a, b))
                )
            })
            .collect();

        // Assign each computed column into the result (sequential, but just memcpy-level work)
        for (idx, col) in interaction_cols.iter().enumerate() {
            result.slice_mut(s![.., n_original + idx]).assign(col);
        }

        Ok(result)
    }

    fn get_feature_names(&self) -> Vec<String> {
        let mut names = Vec::new();
        let n_features = self.n_features_in.unwrap_or(0);

        if self.include_original {
            if let Some(ref input_names) = self.feature_names_in {
                names.extend_from_slice(input_names);
            } else {
                for i in 0..n_features {
                    names.push(format!("x{}", i));
                }
            }
        }

        if let Some(ref crossings) = self.crossings {
            for crossing in crossings {
                let name_a = self.feature_names_in
                    .as_ref()
                    .map(|n| n[crossing.feature_a].clone())
                    .unwrap_or_else(|| format!("x{}", crossing.feature_a));
                let name_b = self.feature_names_in
                    .as_ref()
                    .map(|n| n[crossing.feature_b].clone())
                    .unwrap_or_else(|| format!("x{}", crossing.feature_b));

                let sym = crossing.interaction.symbol();
                let name = if sym.len() > 1 {
                    format!("{}({},{})", sym, name_a, name_b)
                } else {
                    format!("{}{}{}", name_a, sym, name_b)
                };
                names.push(name);
            }
        }

        names
    }
}

/// Automatic feature crossing based on correlation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AutoCrossing {
    /// Maximum number of crossings
    max_crossings: usize,
    /// Correlation threshold
    correlation_threshold: f64,
    /// Selected pairs
    selected_pairs: Option<Vec<(usize, usize)>>,
    /// Generator
    generator: Option<FeatureInteractions>,
}

impl AutoCrossing {
    /// Create new auto crossing generator
    pub fn new(max_crossings: usize) -> Self {
        Self {
            max_crossings,
            correlation_threshold: 0.3,
            selected_pairs: None,
            generator: None,
        }
    }

    /// Set correlation threshold
    pub fn with_correlation_threshold(mut self, threshold: f64) -> Self {
        self.correlation_threshold = threshold.clamp(0.0, 1.0);
        self
    }

    fn correlation_from_stats(
        x: &Array2<f64>,
        col_a: usize, mean_a: f64, std_a: f64,
        col_b: usize, mean_b: f64, std_b: f64,
    ) -> f64 {
        if std_a < 1e-10 || std_b < 1e-10 { return 0.0; }
        let n = x.nrows() as f64;
        let cov: f64 = x.column(col_a).iter()
            .zip(x.column(col_b).iter())
            .map(|(&a, &b)| (a - mean_a) * (b - mean_b))
            .sum::<f64>() / n;
        cov / (std_a * std_b)
    }
}

impl FeatureTransformer for AutoCrossing {
    fn fit(&mut self, x: &Array2<f64>) -> Result<()> {
        let n_features = x.ncols();
        let n = x.nrows() as f64;

        // Precompute means and stds once for all columns
        let means: Vec<f64> = (0..n_features)
            .map(|i| x.column(i).iter().sum::<f64>() / n)
            .collect();
        let stds: Vec<f64> = (0..n_features)
            .map(|i| {
                let m = means[i];
                (x.column(i).iter().map(|&v| (v - m).powi(2)).sum::<f64>() / n).sqrt()
            })
            .collect();

        let mut pair_correlations: Vec<((usize, usize), f64)> = Vec::new();

        for i in 0..n_features {
            for j in (i + 1)..n_features {
                let corr = Self::correlation_from_stats(x, i, means[i], stds[i], j, means[j], stds[j]);
                if corr.abs() >= self.correlation_threshold {
                    pair_correlations.push(((i, j), corr.abs()));
                }
            }
        }

        pair_correlations.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        let selected: Vec<(usize, usize)> = pair_correlations
            .into_iter()
            .take(self.max_crossings)
            .map(|(pair, _)| pair)
            .collect();

        self.selected_pairs = Some(selected.clone());

        let mut generator = FeatureInteractions::multiplicative()
            .with_crossings(selected)
            .with_original(true);

        generator.fit(x)?;
        self.generator = Some(generator);

        Ok(())
    }

    fn transform(&self, x: &Array2<f64>) -> Result<Array2<f64>> {
        self.generator
            .as_ref()
            .ok_or_else(|| AutoMLError::ValidationError("Transformer not fitted".to_string()))?
            .transform(x)
    }

    fn get_feature_names(&self) -> Vec<String> {
        self.generator
            .as_ref()
            .map(|g| g.get_feature_names())
            .unwrap_or_default()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_feature_interactions_multiply() {
        let x = Array2::from_shape_vec(
            (3, 3),
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
        ).unwrap();

        let mut fi = FeatureInteractions::multiplicative()
            .with_original(false);

        let result = fi.fit_transform(&x).unwrap();
        assert_eq!(result.ncols(), 3);
        assert_eq!(result[[0, 0]], 2.0);
        assert_eq!(result[[0, 1]], 3.0);
        assert_eq!(result[[0, 2]], 6.0);
    }

    #[test]
    fn test_feature_interactions_with_original() {
        let x = Array2::from_shape_vec(
            (2, 2),
            vec![1.0, 2.0, 3.0, 4.0],
        ).unwrap();

        let mut fi = FeatureInteractions::multiplicative()
            .with_original(true);

        let result = fi.fit_transform(&x).unwrap();
        assert_eq!(result.ncols(), 3);
        assert_eq!(result[[0, 0]], 1.0);
        assert_eq!(result[[0, 1]], 2.0);
        assert_eq!(result[[0, 2]], 2.0);
    }
}
