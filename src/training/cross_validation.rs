//! Cross-validation implementations

use crate::error::{AutoMLError, Result};
use ndarray::{Array1, Array2, Axis};
use rand::seq::SliceRandom;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

/// Cross-validation strategy
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CVStrategy {
    /// K-Fold cross-validation
    KFold { n_splits: usize, shuffle: bool },
    /// Stratified K-Fold (maintains class distribution)
    StratifiedKFold { n_splits: usize, shuffle: bool },
    /// Time series split (no shuffling, respects temporal order)
    TimeSeriesSplit { n_splits: usize, max_train_size: Option<usize> },
    /// Leave-one-out cross-validation
    LeaveOneOut,
    /// Group K-Fold (keeps groups together)
    GroupKFold { n_splits: usize },
    /// Repeated K-Fold
    RepeatedKFold { n_splits: usize, n_repeats: usize },
}

impl Default for CVStrategy {
    fn default() -> Self {
        CVStrategy::KFold { n_splits: 5, shuffle: true }
    }
}

/// A single train/test split
#[derive(Debug, Clone)]
pub struct CVSplit {
    pub train_indices: Vec<usize>,
    pub test_indices: Vec<usize>,
    pub fold_idx: usize,
}

/// Cross-validation splitter
pub struct CrossValidator {
    strategy: CVStrategy,
    random_state: Option<u64>,
}

impl CrossValidator {
    /// Create a new cross-validator
    pub fn new(strategy: CVStrategy) -> Self {
        Self {
            strategy,
            random_state: None,
        }
    }

    /// Set random state for reproducibility
    pub fn with_random_state(mut self, seed: u64) -> Self {
        self.random_state = Some(seed);
        self
    }

    /// Generate train/test splits
    pub fn split(&self, n_samples: usize, y: Option<&Array1<f64>>, groups: Option<&Array1<i64>>) -> Result<Vec<CVSplit>> {
        match &self.strategy {
            CVStrategy::KFold { n_splits, shuffle } => {
                self.k_fold_split(n_samples, *n_splits, *shuffle)
            }
            CVStrategy::StratifiedKFold { n_splits, shuffle } => {
                let y = y.ok_or_else(|| AutoMLError::ValidationError(
                    "StratifiedKFold requires target array".to_string()
                ))?;
                self.stratified_k_fold_split(n_samples, y, *n_splits, *shuffle)
            }
            CVStrategy::TimeSeriesSplit { n_splits, max_train_size } => {
                self.time_series_split(n_samples, *n_splits, *max_train_size)
            }
            CVStrategy::LeaveOneOut => {
                self.leave_one_out_split(n_samples)
            }
            CVStrategy::GroupKFold { n_splits } => {
                let groups = groups.ok_or_else(|| AutoMLError::ValidationError(
                    "GroupKFold requires groups array".to_string()
                ))?;
                self.group_k_fold_split(n_samples, groups, *n_splits)
            }
            CVStrategy::RepeatedKFold { n_splits, n_repeats } => {
                self.repeated_k_fold_split(n_samples, *n_splits, *n_repeats)
            }
        }
    }

    fn k_fold_split(&self, n_samples: usize, n_splits: usize, shuffle: bool) -> Result<Vec<CVSplit>> {
        if n_splits < 2 {
            return Err(AutoMLError::ValidationError(
                "n_splits must be at least 2".to_string()
            ));
        }
        if n_samples < n_splits {
            return Err(AutoMLError::ValidationError(
                format!("n_samples ({}) must be >= n_splits ({})", n_samples, n_splits)
            ));
        }

        let mut indices: Vec<usize> = (0..n_samples).collect();
        
        if shuffle {
            let mut rng = match self.random_state {
                Some(seed) => ChaCha8Rng::seed_from_u64(seed),
                None => ChaCha8Rng::from_entropy(),
            };
            indices.shuffle(&mut rng);
        }

        let fold_sizes: Vec<usize> = (0..n_splits)
            .map(|i| {
                let base = n_samples / n_splits;
                let remainder = n_samples % n_splits;
                if i < remainder { base + 1 } else { base }
            })
            .collect();

        let mut splits = Vec::with_capacity(n_splits);
        let mut current = 0;

        for fold_idx in 0..n_splits {
            let fold_size = fold_sizes[fold_idx];
            let test_indices: Vec<usize> = indices[current..current + fold_size].to_vec();
            let train_indices: Vec<usize> = indices[..current]
                .iter()
                .chain(indices[current + fold_size..].iter())
                .copied()
                .collect();

            splits.push(CVSplit {
                train_indices,
                test_indices,
                fold_idx,
            });

            current += fold_size;
        }

        Ok(splits)
    }

    fn stratified_k_fold_split(
        &self,
        _n_samples: usize,
        y: &Array1<f64>,
        n_splits: usize,
        shuffle: bool,
    ) -> Result<Vec<CVSplit>> {
        if n_splits < 2 {
            return Err(AutoMLError::ValidationError(
                "n_splits must be at least 2".to_string()
            ));
        }

        // Group samples by class. Use a BTreeMap (not a HashMap) so class iteration order is
        // deterministic (ascending class label) — a HashMap's iteration order depends on its
        // randomized per-process hasher seed, which previously made `indices.shuffle(&mut
        // rng)` below consume the shared `rng`'s draws in a different order per class from
        // run to run, so results were non-reproducible even with a fixed `random_state`.
        let mut class_indices: std::collections::BTreeMap<i64, Vec<usize>> = std::collections::BTreeMap::new();

        for (idx, &val) in y.iter().enumerate() {
            let class = val.round() as i64;
            class_indices.entry(class).or_default().push(idx);
        }

        let mut rng = match self.random_state {
            Some(seed) => ChaCha8Rng::seed_from_u64(seed),
            None => ChaCha8Rng::from_entropy(),
        };

        // Shuffle within each class if needed
        if shuffle {
            for indices in class_indices.values_mut() {
                indices.shuffle(&mut rng);
            }
        }

        // Initialize folds
        let mut folds: Vec<Vec<usize>> = vec![Vec::new(); n_splits];

        // Distribute samples from each class to folds
        for indices in class_indices.values() {
            for (i, &idx) in indices.iter().enumerate() {
                folds[i % n_splits].push(idx);
            }
        }

        // Create splits
        let mut splits = Vec::with_capacity(n_splits);
        for fold_idx in 0..n_splits {
            let test_indices = folds[fold_idx].clone();
            let train_indices: Vec<usize> = folds
                .iter()
                .enumerate()
                .filter(|(i, _)| *i != fold_idx)
                .flat_map(|(_, f)| f.iter().copied())
                .collect();

            splits.push(CVSplit {
                train_indices,
                test_indices,
                fold_idx,
            });
        }

        Ok(splits)
    }

    fn time_series_split(
        &self,
        n_samples: usize,
        n_splits: usize,
        max_train_size: Option<usize>,
    ) -> Result<Vec<CVSplit>> {
        if n_splits < 2 {
            return Err(AutoMLError::ValidationError(
                "n_splits must be at least 2".to_string()
            ));
        }

        let test_size = n_samples / (n_splits + 1);
        if test_size == 0 {
            return Err(AutoMLError::ValidationError(format!(
                "not enough samples ({}) for requested n_splits ({}): each fold would be empty",
                n_samples, n_splits
            )));
        }
        let mut splits = Vec::with_capacity(n_splits);

        for fold_idx in 0..n_splits {
            let test_start = (fold_idx + 1) * test_size;
            let test_end = std::cmp::min(test_start + test_size, n_samples);

            let train_start = match max_train_size {
                Some(max) => test_start.saturating_sub(max),
                None => 0,
            };

            let train_indices: Vec<usize> = (train_start..test_start).collect();
            let test_indices: Vec<usize> = (test_start..test_end).collect();

            splits.push(CVSplit {
                train_indices,
                test_indices,
                fold_idx,
            });
        }

        Ok(splits)
    }

    fn leave_one_out_split(&self, n_samples: usize) -> Result<Vec<CVSplit>> {
        let splits: Vec<CVSplit> = (0..n_samples)
            .map(|i| {
                let train_indices: Vec<usize> = (0..n_samples).filter(|&j| j != i).collect();
                CVSplit {
                    train_indices,
                    test_indices: vec![i],
                    fold_idx: i,
                }
            })
            .collect();

        Ok(splits)
    }

    fn group_k_fold_split(
        &self,
        _n_samples: usize,
        groups: &Array1<i64>,
        n_splits: usize,
    ) -> Result<Vec<CVSplit>> {
        if n_splits < 2 {
            return Err(AutoMLError::ValidationError(
                "n_splits must be at least 2".to_string()
            ));
        }

        // Get unique groups
        let mut unique_groups: Vec<i64> = groups.iter().copied().collect();
        unique_groups.sort_unstable();
        unique_groups.dedup();

        if unique_groups.len() < n_splits {
            return Err(AutoMLError::ValidationError(
                format!("Number of groups ({}) must be >= n_splits ({})", unique_groups.len(), n_splits)
            ));
        }

        // Assign groups to folds
        let mut group_to_fold: std::collections::HashMap<i64, usize> = std::collections::HashMap::new();
        for (i, &group) in unique_groups.iter().enumerate() {
            group_to_fold.insert(group, i % n_splits);
        }

        // Create splits
        let mut splits = Vec::with_capacity(n_splits);
        for fold_idx in 0..n_splits {
            let test_indices: Vec<usize> = groups
                .iter()
                .enumerate()
                .filter(|(_, &g)| group_to_fold.get(&g) == Some(&fold_idx))
                .map(|(i, _)| i)
                .collect();

            let train_indices: Vec<usize> = groups
                .iter()
                .enumerate()
                .filter(|(_, &g)| group_to_fold.get(&g) != Some(&fold_idx))
                .map(|(i, _)| i)
                .collect();

            splits.push(CVSplit {
                train_indices,
                test_indices,
                fold_idx,
            });
        }

        Ok(splits)
    }

    fn repeated_k_fold_split(&self, n_samples: usize, n_splits: usize, n_repeats: usize) -> Result<Vec<CVSplit>> {
        let mut all_splits = Vec::with_capacity(n_splits * n_repeats);

        for repeat in 0..n_repeats {
            let seed = self.random_state.map(|s| s + repeat as u64);
            let cv = CrossValidator {
                strategy: CVStrategy::KFold { n_splits, shuffle: true },
                random_state: seed,
            };

            let mut splits = cv.k_fold_split(n_samples, n_splits, true)?;
            
            // Update fold indices to be unique across repeats
            for split in &mut splits {
                split.fold_idx = repeat * n_splits + split.fold_idx;
            }
            
            all_splits.extend(splits);
        }

        Ok(all_splits)
    }
}

/// Cross-validation results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CVResults {
    /// Scores for each fold
    pub scores: Vec<f64>,
    /// Mean score across folds
    pub mean_score: f64,
    /// Standard deviation of scores
    pub std_score: f64,
    /// Number of folds
    pub n_folds: usize,
}

impl CVResults {
    /// Create CV results from fold scores
    pub fn from_scores(scores: Vec<f64>) -> Result<Self> {
        if scores.is_empty() {
            return Err(AutoMLError::ValidationError(
                "cannot compute CV results from an empty list of fold scores".to_string()
            ));
        }
        let n_folds = scores.len();
        let mean_score = scores.iter().sum::<f64>() / n_folds as f64;
        let variance = scores.iter().map(|s| (s - mean_score).powi(2)).sum::<f64>() / n_folds as f64;
        let std_score = variance.sqrt();

        Ok(Self {
            scores,
            mean_score,
            std_score,
            n_folds,
        })
    }
}

/// Run cross-validation with parallel fold training.
///
/// Trains all folds in parallel using rayon, computes per-fold scores, and
/// returns aggregated `CVResults`.
pub fn cross_val_score(
    config: &super::TrainingConfig,
    x: &Array2<f64>,
    y: &Array1<f64>,
    cv: &CVStrategy,
    random_state: Option<u64>,
) -> Result<CVResults> {
    use super::engine::TrainEngine;

    let validator = match random_state {
        Some(seed) => CrossValidator::new(cv.clone()).with_random_state(seed),
        None => CrossValidator::new(cv.clone()),
    };
    let splits = validator.split(x.nrows(), Some(y), None)?;

    let is_regression = matches!(
        config.task_type,
        super::TaskType::Regression | super::TaskType::TimeSeries
    );

    // Train all folds in parallel
    let fold_scores: Vec<Result<f64>> = splits
        .par_iter()
        .map(|split| {
            let x_train = x.select(Axis(0), &split.train_indices);
            let y_train = Array1::from_iter(split.train_indices.iter().map(|&i| y[i]));

            let x_test = x.select(Axis(0), &split.test_indices);
            let y_test: Array1<f64> = Array1::from_iter(split.test_indices.iter().map(|&i| y[i]));

            let y_pred = TrainEngine::fit_predict_arrays(config, &x_train, &y_train, &x_test)?;

            let score = if is_regression {
                // R² score
                let mean = y_test.iter().sum::<f64>() / y_test.len() as f64;
                let ss_tot: f64 = y_test.iter().map(|&v| (v - mean).powi(2)).sum();
                let ss_res: f64 = y_test.iter().zip(y_pred.iter()).map(|(&a, &p)| (a - p).powi(2)).sum();
                if ss_tot > 0.0 { 1.0 - ss_res / ss_tot } else { 0.0 }
            } else {
                // Accuracy
                let correct = y_test.iter().zip(y_pred.iter())
                    .filter(|(&a, &p)| (a - p).abs() < 0.5)
                    .count();
                correct as f64 / y_test.len() as f64
            };

            Ok(score)
        })
        .collect();

    let scores: Vec<f64> = fold_scores.into_iter().collect::<Result<Vec<f64>>>()?;
    CVResults::from_scores(scores)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_k_fold() {
        let cv = CrossValidator::new(CVStrategy::KFold { n_splits: 5, shuffle: false });
        let splits = cv.split(100, None, None).unwrap();
        
        assert_eq!(splits.len(), 5);
        
        // Each test set should have 20 samples
        for split in &splits {
            assert_eq!(split.test_indices.len(), 20);
            assert_eq!(split.train_indices.len(), 80);
        }

        // All indices should be covered exactly once in test sets
        let mut all_test: Vec<usize> = splits.iter().flat_map(|s| s.test_indices.clone()).collect();
        all_test.sort();
        assert_eq!(all_test, (0..100).collect::<Vec<_>>());
    }

    #[test]
    fn test_stratified_k_fold() {
        let y = Array1::from_vec(vec![
            0.0, 0.0, 0.0, 0.0, 0.0,  // 5 samples of class 0
            1.0, 1.0, 1.0, 1.0, 1.0,  // 5 samples of class 1
        ]);
        
        let cv = CrossValidator::new(CVStrategy::StratifiedKFold { n_splits: 5, shuffle: false });
        let splits = cv.split(10, Some(&y), None).unwrap();
        
        assert_eq!(splits.len(), 5);
        
        // Each fold should have 1 sample from each class
        for split in &splits {
            assert_eq!(split.test_indices.len(), 2);
        }
    }

    #[test]
    fn test_time_series_split() {
        let cv = CrossValidator::new(CVStrategy::TimeSeriesSplit { n_splits: 3, max_train_size: None });
        let splits = cv.split(100, None, None).unwrap();
        
        assert_eq!(splits.len(), 3);
        
        // Each subsequent split should have more training data
        for i in 1..splits.len() {
            assert!(splits[i].train_indices.len() >= splits[i-1].train_indices.len());
        }
        
        // Test indices should not overlap with train indices
        for split in &splits {
            for &test_idx in &split.test_indices {
                assert!(!split.train_indices.contains(&test_idx));
            }
        }
    }

    #[test]
    fn test_leave_one_out() {
        let cv = CrossValidator::new(CVStrategy::LeaveOneOut);
        let splits = cv.split(10, None, None).unwrap();
        
        assert_eq!(splits.len(), 10);
        
        for split in &splits {
            assert_eq!(split.test_indices.len(), 1);
            assert_eq!(split.train_indices.len(), 9);
        }
    }

    #[test]
    fn test_repeated_k_fold() {
        let cv = CrossValidator::new(CVStrategy::RepeatedKFold { n_splits: 5, n_repeats: 3 })
            .with_random_state(42);
        let splits = cv.split(100, None, None).unwrap();
        
        assert_eq!(splits.len(), 15); // 5 * 3
    }
}
