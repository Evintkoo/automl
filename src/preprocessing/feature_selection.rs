//! Feature selection algorithms
//!
//! Provides various feature selection methods including:
//! - Variance threshold selection
//! - Mutual information based selection
//! - Recursive feature elimination
//! - Correlation-based selection
//! - L1-based selection (Lasso)

use crate::error::{AutoMLError, Result};
use ndarray::{Array1, Array2};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};

/// Feature selection method
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SelectionMethod {
    /// Remove features with variance below threshold
    VarianceThreshold { threshold: f64 },
    /// Select k best features based on mutual information
    MutualInformation { k: usize },
    /// Select features by correlation threshold
    CorrelationThreshold { threshold: f64 },
    /// Recursive Feature Elimination
    RFE { n_features_to_select: usize, step: usize },
    /// Select by percentile of scores
    Percentile { percentile: f64 },
    /// Select features with importance above threshold
    ImportanceThreshold { threshold: f64 },
}

/// Feature selector for dimensionality reduction
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeatureSelector {
    method: SelectionMethod,
    selected_features: Option<Vec<usize>>,
    feature_scores: Option<Vec<f64>>,
    feature_names: Option<Vec<String>>,
    n_features_in: Option<usize>,
}

impl FeatureSelector {
    /// Create a new feature selector with the given method
    pub fn new(method: SelectionMethod) -> Self {
        Self {
            method,
            selected_features: None,
            feature_scores: None,
            feature_names: None,
            n_features_in: None,
        }
    }

    /// Create variance threshold selector
    pub fn variance_threshold(threshold: f64) -> Self {
        Self::new(SelectionMethod::VarianceThreshold { threshold })
    }

    /// Create mutual information selector
    pub fn mutual_information(k: usize) -> Self {
        Self::new(SelectionMethod::MutualInformation { k })
    }

    /// Create correlation threshold selector
    pub fn correlation_threshold(threshold: f64) -> Self {
        Self::new(SelectionMethod::CorrelationThreshold { threshold })
    }

    /// Create RFE selector
    pub fn rfe(n_features_to_select: usize, step: usize) -> Self {
        Self::new(SelectionMethod::RFE {
            n_features_to_select,
            step: step.max(1),
        })
    }

    /// Create percentile selector
    pub fn percentile(percentile: f64) -> Self {
        Self::new(SelectionMethod::Percentile {
            percentile: percentile.clamp(0.0, 100.0),
        })
    }

    /// Set feature names
    pub fn with_feature_names(mut self, names: Vec<String>) -> Self {
        self.feature_names = Some(names);
        self
    }

    /// Fit the selector to data
    pub fn fit(&mut self, x: &Array2<f64>, y: &Array1<f64>) -> Result<()> {
        let n_features = x.ncols();
        self.n_features_in = Some(n_features);

        match &self.method {
            SelectionMethod::VarianceThreshold { threshold } => {
                self.fit_variance_threshold(x, *threshold)?;
            }
            SelectionMethod::MutualInformation { k } => {
                self.fit_mutual_information(x, y, *k)?;
            }
            SelectionMethod::CorrelationThreshold { threshold } => {
                self.fit_correlation_threshold(x, y, *threshold)?;
            }
            SelectionMethod::RFE { n_features_to_select, step } => {
                self.fit_rfe(x, y, *n_features_to_select, *step)?;
            }
            SelectionMethod::Percentile { percentile } => {
                self.fit_percentile(x, y, *percentile)?;
            }
            SelectionMethod::ImportanceThreshold { threshold } => {
                self.fit_importance_threshold(x, y, *threshold)?;
            }
        }

        Ok(())
    }

    /// Transform data by selecting features
    pub fn transform(&self, x: &Array2<f64>) -> Result<Array2<f64>> {
        let selected = self.selected_features.as_ref().ok_or_else(|| {
            AutoMLError::ValidationError("Selector not fitted".to_string())
        })?;

        if selected.is_empty() {
            return Err(AutoMLError::ValidationError(
                "No features selected".to_string(),
            ));
        }

        let n_samples = x.nrows();
        let n_selected = selected.len();
        let mut result = Array2::zeros((n_samples, n_selected));

        for (new_idx, &old_idx) in selected.iter().enumerate() {
            result.column_mut(new_idx).assign(&x.column(old_idx));
        }

        Ok(result)
    }

    /// Fit and transform in one step
    pub fn fit_transform(&mut self, x: &Array2<f64>, y: &Array1<f64>) -> Result<Array2<f64>> {
        self.fit(x, y)?;
        self.transform(x)
    }

    /// Get selected feature indices
    pub fn selected_indices(&self) -> Option<&[usize]> {
        self.selected_features.as_deref()
    }

    /// Get feature scores
    pub fn scores(&self) -> Option<&[f64]> {
        self.feature_scores.as_deref()
    }

    /// Get selected feature names
    pub fn selected_names(&self) -> Option<Vec<String>> {
        let indices = self.selected_features.as_ref()?;
        let names = self.feature_names.as_ref()?;
        
        Some(
            indices
                .iter()
                .filter_map(|&i| names.get(i).cloned())
                .collect(),
        )
    }

    /// Get feature ranking (1 = best)
    pub fn ranking(&self) -> Option<Vec<usize>> {
        let scores = self.feature_scores.as_ref()?;
        
        let mut indexed: Vec<(usize, f64)> = scores.iter().copied().enumerate().collect();
        indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        
        let mut ranking = vec![0; scores.len()];
        for (rank, (idx, _)) in indexed.into_iter().enumerate() {
            ranking[idx] = rank + 1;
        }
        
        Some(ranking)
    }

    // Fit variance threshold
    fn fit_variance_threshold(&mut self, x: &Array2<f64>, threshold: f64) -> Result<()> {
        let n_features = x.ncols();
        let mut variances = Vec::with_capacity(n_features);
        let mut selected = Vec::new();

        for col_idx in 0..n_features {
            let col = x.column(col_idx);
            let mean = col.mean().unwrap_or(0.0);
            let variance = col.iter().map(|&v| (v - mean).powi(2)).sum::<f64>() / col.len() as f64;
            
            variances.push(variance);
            if variance > threshold {
                selected.push(col_idx);
            }
        }

        self.feature_scores = Some(variances);
        self.selected_features = Some(selected);
        Ok(())
    }

    // Fit mutual information
    fn fit_mutual_information(
        &mut self,
        x: &Array2<f64>,
        y: &Array1<f64>,
        k: usize,
    ) -> Result<()> {
        let n_features = x.ncols();
        let mut mi_scores = Vec::with_capacity(n_features);

        for col_idx in 0..n_features {
            let col = x.column(col_idx);
            let mi = Self::compute_mutual_information(&col.to_owned(), y);
            mi_scores.push(mi);
        }

        // Select top k features
        let mut indexed: Vec<(usize, f64)> = mi_scores.iter().copied().enumerate().collect();
        indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        
        let selected: Vec<usize> = indexed.into_iter().take(k.min(n_features)).map(|(i, _)| i).collect();

        self.feature_scores = Some(mi_scores);
        self.selected_features = Some(selected);
        Ok(())
    }

    // Compute mutual information between two variables
    fn compute_mutual_information(x: &Array1<f64>, y: &Array1<f64>) -> f64 {
        let n = x.len() as f64;
        if n < 2.0 {
            return 0.0;
        }

        // Discretize continuous variables into bins
        let n_bins = ((n.sqrt()) as usize).max(2).min(20);
        
        let x_bins = Self::discretize(x, n_bins);
        let y_bins = Self::discretize(y, n_bins);

        // Compute joint and marginal distributions
        let mut joint_counts: HashMap<(usize, usize), usize> = HashMap::new();
        let mut x_counts: HashMap<usize, usize> = HashMap::new();
        let mut y_counts: HashMap<usize, usize> = HashMap::new();

        for (&xb, &yb) in x_bins.iter().zip(y_bins.iter()) {
            *joint_counts.entry((xb, yb)).or_insert(0) += 1;
            *x_counts.entry(xb).or_insert(0) += 1;
            *y_counts.entry(yb).or_insert(0) += 1;
        }

        // Compute mutual information
        let mut mi = 0.0;
        let n_total = n;

        for (&(xb, yb), &count) in &joint_counts {
            let p_xy = count as f64 / n_total;
            let p_x = *x_counts.get(&xb).unwrap() as f64 / n_total;
            let p_y = *y_counts.get(&yb).unwrap() as f64 / n_total;

            if p_xy > 0.0 && p_x > 0.0 && p_y > 0.0 {
                mi += p_xy * (p_xy / (p_x * p_y)).ln();
            }
        }

        mi.max(0.0)
    }

    // Discretize continuous variable into bins
    fn discretize(x: &Array1<f64>, n_bins: usize) -> Vec<usize> {
        // Single pass: compute min, max, and bins together
        let mut min_val = f64::INFINITY;
        let mut max_val = f64::NEG_INFINITY;
        for &v in x.iter() {
            if v < min_val { min_val = v; }
            if v > max_val { max_val = v; }
        }
        let range = max_val - min_val;
        if range <= 0.0 {
            return vec![0; x.len()];
        }
        let bin_width = range / n_bins as f64;
        x.iter()
            .map(|&v| (((v - min_val) / bin_width) as usize).min(n_bins - 1))
            .collect()
    }

    // Fit correlation threshold
    fn fit_correlation_threshold(
        &mut self,
        x: &Array2<f64>,
        y: &Array1<f64>,
        threshold: f64,
    ) -> Result<()> {
        let n_features = x.ncols();
        let mut correlations = Vec::with_capacity(n_features);
        let mut selected = Vec::new();

        let y_mean = y.mean().unwrap_or(0.0);
        let y_std = (y.iter().map(|&v| (v - y_mean).powi(2)).sum::<f64>() / y.len() as f64).sqrt();

        for col_idx in 0..n_features {
            let col = x.column(col_idx);
            let x_mean = col.mean().unwrap_or(0.0);
            let x_std = (col.iter().map(|&v| (v - x_mean).powi(2)).sum::<f64>() / col.len() as f64).sqrt();

            let correlation = if x_std > 0.0 && y_std > 0.0 {
                let covariance: f64 = col
                    .iter()
                    .zip(y.iter())
                    .map(|(&x, &y)| (x - x_mean) * (y - y_mean))
                    .sum::<f64>()
                    / col.len() as f64;
                (covariance / (x_std * y_std)).abs()
            } else {
                0.0
            };

            correlations.push(correlation);
            if correlation >= threshold {
                selected.push(col_idx);
            }
        }

        self.feature_scores = Some(correlations);
        self.selected_features = Some(selected);
        Ok(())
    }

    // Fit RFE (simplified version using correlation as feature importance proxy)
    fn fit_rfe(
        &mut self,
        x: &Array2<f64>,
        y: &Array1<f64>,
        n_features_to_select: usize,
        step: usize,
    ) -> Result<()> {
        let n_features = x.ncols();
        let n_select = n_features_to_select.min(n_features);
        
        let mut remaining: HashSet<usize> = (0..n_features).collect();
        let mut ranking = vec![0usize; n_features];
        let mut current_rank = n_features;

        // Iteratively remove features
        while remaining.len() > n_select {
            // Compute importance scores for remaining features
            use rayon::prelude::*;
            let mut scores: Vec<(usize, f64)> = remaining
                .par_iter()
                .map(|&idx| {
                    let col = x.column(idx);
                    let importance = Self::compute_correlation(&col.to_owned(), y).abs();
                    (idx, importance)
                })
                .collect();

            // Sort by importance (ascending, to remove worst first)
            scores.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

            // Remove step features
            let n_to_remove = step.min(remaining.len() - n_select);
            for (idx, _) in scores.into_iter().take(n_to_remove) {
                remaining.remove(&idx);
                ranking[idx] = current_rank;
                current_rank = current_rank.saturating_sub(1);
            }
        }

        // Assign rank 1 to remaining features
        for &idx in &remaining {
            ranking[idx] = 1;
        }

        // Convert ranking to scores (inverse so higher is better)
        let scores: Vec<f64> = ranking
            .iter()
            .map(|&r| 1.0 / r as f64)
            .collect();

        self.feature_scores = Some(scores);
        self.selected_features = Some(remaining.into_iter().collect());
        Ok(())
    }

    // Fit percentile selection
    fn fit_percentile(&mut self, x: &Array2<f64>, y: &Array1<f64>, percentile: f64) -> Result<()> {
        let n_features = x.ncols();
        if n_features == 0 {
            // Guard against the `(n_features - 1) as f64` subtraction below
            // underflowing (usize) when there are no columns to score.
            self.feature_scores = Some(Vec::new());
            self.selected_features = Some(Vec::new());
            return Ok(());
        }
        let mut scores = Vec::with_capacity(n_features);

        // Compute correlation-based scores
        for col_idx in 0..n_features {
            let col = x.column(col_idx);
            let score = Self::compute_correlation(&col.to_owned(), y).abs();
            scores.push(score);
        }

        // Select by percentile
        let mut sorted_scores: Vec<f64> = scores.clone();
        sorted_scores.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        
        let percentile_idx = ((percentile / 100.0) * (n_features - 1) as f64) as usize;
        let threshold = sorted_scores.get(percentile_idx).copied().unwrap_or(0.0);

        let selected: Vec<usize> = scores
            .iter()
            .enumerate()
            .filter(|(_, &s)| s >= threshold)
            .map(|(i, _)| i)
            .collect();

        self.feature_scores = Some(scores);
        self.selected_features = Some(selected);
        Ok(())
    }

    // Fit importance threshold
    fn fit_importance_threshold(
        &mut self,
        x: &Array2<f64>,
        y: &Array1<f64>,
        threshold: f64,
    ) -> Result<()> {
        let n_features = x.ncols();
        let mut scores = Vec::with_capacity(n_features);
        let mut selected = Vec::new();

        for col_idx in 0..n_features {
            let col = x.column(col_idx);
            let score = Self::compute_correlation(&col.to_owned(), y).abs();
            scores.push(score);
            if score >= threshold {
                selected.push(col_idx);
            }
        }

        self.feature_scores = Some(scores);
        self.selected_features = Some(selected);
        Ok(())
    }

    // Helper to compute Pearson correlation
    fn compute_correlation(x: &Array1<f64>, y: &Array1<f64>) -> f64 {
        let n = x.len() as f64;
        if n < 2.0 {
            return 0.0;
        }

        let x_mean = x.mean().unwrap_or(0.0);
        let y_mean = y.mean().unwrap_or(0.0);

        let x_std = (x.iter().map(|&v| (v - x_mean).powi(2)).sum::<f64>() / n).sqrt();
        let y_std = (y.iter().map(|&v| (v - y_mean).powi(2)).sum::<f64>() / n).sqrt();

        if x_std <= 0.0 || y_std <= 0.0 {
            return 0.0;
        }

        let covariance: f64 = x
            .iter()
            .zip(y.iter())
            .map(|(&x, &y)| (x - x_mean) * (y - y_mean))
            .sum::<f64>()
            / n;

        covariance / (x_std * y_std)
    }
}

/// Remove highly correlated features
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CorrelationFilter {
    threshold: f64,
    selected_features: Option<Vec<usize>>,
    removed_features: Option<Vec<(usize, usize, f64)>>,
}

impl CorrelationFilter {
    /// Create a new correlation filter
    pub fn new(threshold: f64) -> Self {
        Self {
            threshold: threshold.abs(),
            selected_features: None,
            removed_features: None,
        }
    }

    /// Fit the filter to remove highly correlated features
    pub fn fit(&mut self, x: &Array2<f64>) -> Result<()> {
        let n_features = x.ncols();
        let mut to_remove: HashSet<usize> = HashSet::new();
        let mut removed_pairs = Vec::new();

        // Precompute full correlation matrix once — O(n²) instead of O(n³)
        let corr_matrix = Self::compute_correlation_matrix(x);

        for i in 0..n_features {
            if to_remove.contains(&i) {
                continue;
            }
            for j in (i + 1)..n_features {
                if to_remove.contains(&j) {
                    continue;
                }
                let corr = corr_matrix[[i, j]];
                if corr.abs() > self.threshold {
                    // Row-mean from precomputed matrix (O(n) read, no recomputation)
                    let n_active = n_features - to_remove.len();
                    let denom = (n_active.saturating_sub(1)).max(1) as f64;
                    let mean_i: f64 = (0..n_features)
                        .filter(|&k| k != i && !to_remove.contains(&k))
                        .map(|k| corr_matrix[[i, k]].abs())
                        .sum::<f64>() / denom;
                    let mean_j: f64 = (0..n_features)
                        .filter(|&k| k != j && !to_remove.contains(&k))
                        .map(|k| corr_matrix[[j, k]].abs())
                        .sum::<f64>() / denom;

                    let remove_idx = if mean_i > mean_j { i } else { j };
                    to_remove.insert(remove_idx);
                    removed_pairs.push((i, j, corr));
                }
            }
        }

        let selected: Vec<usize> = (0..n_features)
            .filter(|i| !to_remove.contains(i))
            .collect();

        self.selected_features = Some(selected);
        self.removed_features = Some(removed_pairs);
        Ok(())
    }

    /// Transform data
    pub fn transform(&self, x: &Array2<f64>) -> Result<Array2<f64>> {
        let selected = self.selected_features.as_ref().ok_or_else(|| {
            AutoMLError::ValidationError("Filter not fitted".to_string())
        })?;

        let n_samples = x.nrows();
        let n_selected = selected.len();
        let mut result = Array2::zeros((n_samples, n_selected));

        for (new_idx, &old_idx) in selected.iter().enumerate() {
            result.column_mut(new_idx).assign(&x.column(old_idx));
        }

        Ok(result)
    }

    /// Get selected feature indices
    pub fn selected_indices(&self) -> Option<&[usize]> {
        self.selected_features.as_deref()
    }

    fn pearson_correlation(x: &Array1<f64>, y: &Array1<f64>) -> f64 {
        let n = x.len() as f64;
        if n < 2.0 {
            return 0.0;
        }

        let x_mean = x.mean().unwrap_or(0.0);
        let y_mean = y.mean().unwrap_or(0.0);

        let mut sum_xy = 0.0;
        let mut sum_x2 = 0.0;
        let mut sum_y2 = 0.0;

        for (&xi, &yi) in x.iter().zip(y.iter()) {
            let dx = xi - x_mean;
            let dy = yi - y_mean;
            sum_xy += dx * dy;
            sum_x2 += dx * dx;
            sum_y2 += dy * dy;
        }

        let denom = (sum_x2 * sum_y2).sqrt();
        if denom == 0.0 {
            0.0
        } else {
            sum_xy / denom
        }
    }

    /// Compute the full NxN Pearson correlation matrix for the given data
    pub fn compute_correlation_matrix(x: &Array2<f64>) -> Array2<f64> {
        use ndarray::Axis;
        use rayon::prelude::*;

        let n = x.ncols();
        // Precompute owned columns to avoid repeated borrows in parallel closure
        let cols: Vec<ndarray::Array1<f64>> = (0..n).map(|i| x.column(i).to_owned()).collect();

        let mut corr = Array2::<f64>::eye(n);
        // Split by rows — each row slice is an independent mutable region
        corr.axis_iter_mut(Axis(0))
            .into_par_iter()
            .enumerate()
            .for_each(|(i, mut row)| {
                for j in 0..n {
                    if i != j {
                        row[j] = Self::pearson_correlation(&cols[i], &cols[j]);
                    }
                }
            });
        corr
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_variance_threshold() {
        let x = Array2::from_shape_vec(
            (5, 3),
            vec![
                1.0, 0.0, 1.0,
                2.0, 0.0, 2.0,
                3.0, 0.0, 3.0,
                4.0, 0.0, 4.0,
                5.0, 0.0, 5.0,
            ],
        ).unwrap();
        let y = array![1.0, 2.0, 3.0, 4.0, 5.0];

        let mut selector = FeatureSelector::variance_threshold(0.1);
        selector.fit(&x, &y).unwrap();

        // Column 1 (all zeros) should be removed
        let selected = selector.selected_indices().unwrap();
        assert!(!selected.contains(&1));
        assert!(selected.contains(&0));
        assert!(selected.contains(&2));
    }

    #[test]
    fn test_mutual_information() {
        let x = Array2::from_shape_vec(
            (6, 3),
            vec![
                1.0, 5.0, 0.1,
                2.0, 4.0, 0.2,
                3.0, 3.0, 0.3,
                4.0, 2.0, 0.4,
                5.0, 1.0, 0.5,
                6.0, 0.0, 0.6,
            ],
        ).unwrap();
        let y = array![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];

        let mut selector = FeatureSelector::mutual_information(2);
        selector.fit(&x, &y).unwrap();

        let selected = selector.selected_indices().unwrap();
        assert_eq!(selected.len(), 2);
    }

    #[test]
    fn test_correlation_filter() {
        let x = Array2::from_shape_vec(
            (5, 3),
            vec![
                1.0, 1.1, 5.0,
                2.0, 2.1, 4.0,
                3.0, 3.1, 3.0,
                4.0, 4.1, 2.0,
                5.0, 5.1, 1.0,
            ],
        ).unwrap();

        let mut filter = CorrelationFilter::new(0.9);
        filter.fit(&x).unwrap();

        // Columns 0 and 1 are highly correlated, one should be removed
        let selected = filter.selected_indices().unwrap();
        assert!(selected.len() < 3);
    }

    #[test]
    fn test_transform() {
        let x = Array2::from_shape_vec(
            (3, 4),
            vec![
                1.0, 2.0, 3.0, 4.0,
                5.0, 6.0, 7.0, 8.0,
                9.0, 10.0, 11.0, 12.0,
            ],
        ).unwrap();
        let y = array![1.0, 2.0, 3.0];

        let mut selector = FeatureSelector::mutual_information(2);
        let transformed = selector.fit_transform(&x, &y).unwrap();

        assert_eq!(transformed.ncols(), 2);
        assert_eq!(transformed.nrows(), 3);
    }
}
