//! Time series cross-validation

use serde::{Deserialize, Serialize};

/// Time series split for cross-validation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimeSeriesSplit {
    /// Training indices
    pub train_indices: Vec<usize>,
    /// Test indices
    pub test_indices: Vec<usize>,
    /// Fold number
    pub fold: usize,
}

/// Time series cross-validator
#[derive(Debug, Clone)]
pub struct TimeSeriesCV {
    /// Number of splits
    n_splits: usize,
    /// Maximum training size (None = use all available)
    max_train_size: Option<usize>,
    /// Minimum training size
    min_train_size: usize,
    /// Gap between train and test
    gap: usize,
    /// Test size
    test_size: usize,
}

impl TimeSeriesCV {
    /// Create new time series CV
    pub fn new(n_splits: usize) -> Self {
        Self {
            n_splits: n_splits.max(2),
            max_train_size: None,
            min_train_size: 1,
            gap: 0,
            test_size: 1,
        }
    }

    /// Set maximum training size
    pub fn with_max_train_size(mut self, size: usize) -> Self {
        self.max_train_size = Some(size);
        self
    }

    /// Set minimum training size
    pub fn with_min_train_size(mut self, size: usize) -> Self {
        self.min_train_size = size.max(1);
        self
    }

    /// Set gap between train and test
    pub fn with_gap(mut self, gap: usize) -> Self {
        self.gap = gap;
        self
    }

    /// Set test size
    pub fn with_test_size(mut self, size: usize) -> Self {
        self.test_size = size.max(1);
        self
    }

    /// Generate splits
    pub fn split(&self, n_samples: usize) -> Vec<TimeSeriesSplit> {
        let mut splits = Vec::new();
        
        // Calculate test fold size
        let total_test_size = self.n_splits * self.test_size;
        let total_gap = self.n_splits * self.gap;
        
        if n_samples <= total_test_size + total_gap + self.min_train_size {
            return splits;
        }

        let _available_for_test = n_samples - self.min_train_size;
        let test_fold_size = self.test_size;

        for fold in 0..self.n_splits {
            let test_end = n_samples - fold * test_fold_size;
            let test_start = test_end - test_fold_size;
            
            if test_start <= self.min_train_size + self.gap {
                break;
            }

            let train_end = test_start - self.gap;
            let train_start = match self.max_train_size {
                Some(max) => train_end.saturating_sub(max),
                None => 0,
            };

            if train_end - train_start < self.min_train_size {
                continue;
            }

            splits.push(TimeSeriesSplit {
                train_indices: (train_start..train_end).collect(),
                test_indices: (test_start..test_end).collect(),
                fold: self.n_splits - 1 - fold,
            });
        }

        splits.reverse();
        splits
    }
}

/// Walk-forward cross-validation (expanding window)
#[derive(Debug, Clone)]
pub struct WalkForwardCV {
    /// Initial training size
    initial_train_size: usize,
    /// Step size for each iteration
    step_size: usize,
    /// Test size
    test_size: usize,
    /// Gap between train and test
    gap: usize,
    /// Whether to use expanding or sliding window
    expanding: bool,
}

impl WalkForwardCV {
    /// Create new walk-forward CV with expanding window
    pub fn expanding(initial_train_size: usize) -> Self {
        Self {
            initial_train_size: initial_train_size.max(1),
            step_size: 1,
            test_size: 1,
            gap: 0,
            expanding: true,
        }
    }

    /// Create new walk-forward CV with sliding window
    pub fn sliding(window_size: usize) -> Self {
        Self {
            initial_train_size: window_size.max(1),
            step_size: 1,
            test_size: 1,
            gap: 0,
            expanding: false,
        }
    }

    /// Set step size
    pub fn with_step_size(mut self, step: usize) -> Self {
        self.step_size = step.max(1);
        self
    }

    /// Set test size
    pub fn with_test_size(mut self, size: usize) -> Self {
        self.test_size = size.max(1);
        self
    }

    /// Set gap
    pub fn with_gap(mut self, gap: usize) -> Self {
        self.gap = gap;
        self
    }

    /// Generate splits
    pub fn split(&self, n_samples: usize) -> Vec<TimeSeriesSplit> {
        let mut splits = Vec::new();
        let mut fold = 0;

        let mut train_end = self.initial_train_size;

        while train_end + self.gap + self.test_size <= n_samples {
            let test_start = train_end + self.gap;
            let test_end = test_start + self.test_size;

            let train_start = if self.expanding {
                0
            } else {
                train_end.saturating_sub(self.initial_train_size)
            };

            splits.push(TimeSeriesSplit {
                train_indices: (train_start..train_end).collect(),
                test_indices: (test_start..test_end).collect(),
                fold,
            });

            train_end += self.step_size;
            fold += 1;
        }

        splits
    }
}

/// Blocked time series cross-validation
#[allow(dead_code)]
#[derive(Debug, Clone)]
pub struct BlockedTimeSeriesCV {
    /// Number of blocks
    n_blocks: usize,
    /// Gap between blocks
    gap: usize,
}

#[allow(dead_code)]
impl BlockedTimeSeriesCV {
    /// Create new blocked CV
    pub fn new(n_blocks: usize) -> Self {
        Self {
            n_blocks: n_blocks.max(2),
            gap: 0,
        }
    }

    /// Set gap between blocks
    pub fn with_gap(mut self, gap: usize) -> Self {
        self.gap = gap;
        self
    }

    /// Generate splits
    pub fn split(&self, n_samples: usize) -> Vec<TimeSeriesSplit> {
        let mut splits = Vec::new();
        let block_size = n_samples / self.n_blocks;

        if block_size < 1 {
            return splits;
        }

        for fold in 0..self.n_blocks {
            let test_start = fold * block_size;
            let test_end = if fold == self.n_blocks - 1 {
                n_samples
            } else {
                (fold + 1) * block_size
            };

            // Training is everything except the test block (respecting temporal order)
            let mut train_indices = Vec::new();
            
            // Include data before test block (minus gap)
            if test_start > self.gap {
                train_indices.extend(0..test_start.saturating_sub(self.gap));
            }

            if train_indices.is_empty() {
                continue;
            }

            splits.push(TimeSeriesSplit {
                train_indices,
                test_indices: (test_start..test_end).collect(),
                fold,
            });
        }

        splits
    }
}

/// Purged time series cross-validation
/// Ensures no data leakage by purging samples close to the test set
#[allow(dead_code)]
#[derive(Debug, Clone)]
pub struct PurgedTimeSeriesCV {
    /// Number of splits
    n_splits: usize,
    /// Purge size before test set
    purge_before: usize,
    /// Embargo size after test set
    embargo_after: usize,
}

#[allow(dead_code)]
impl PurgedTimeSeriesCV {
    /// Create new purged CV
    pub fn new(n_splits: usize) -> Self {
        Self {
            n_splits: n_splits.max(2),
            purge_before: 0,
            embargo_after: 0,
        }
    }

    /// Set purge size
    pub fn with_purge(mut self, size: usize) -> Self {
        self.purge_before = size;
        self
    }

    /// Set embargo size
    pub fn with_embargo(mut self, size: usize) -> Self {
        self.embargo_after = size;
        self
    }

    /// Generate splits
    pub fn split(&self, n_samples: usize) -> Vec<TimeSeriesSplit> {
        let mut splits = Vec::new();
        let test_size = n_samples / (self.n_splits + 1);

        if test_size < 1 {
            return splits;
        }

        // Embargo zones left behind by every fold's test window (test_end,
        // test_end + embargo_after). Any later fold whose training range would
        // otherwise include one of these zones must exclude it, so that samples
        // immediately following a test window are never used for training.
        let mut embargo_zones: Vec<(usize, usize)> = Vec::new();

        for fold in 0..self.n_splits {
            let test_start = (fold + 1) * test_size;
            let test_end = if fold == self.n_splits - 1 {
                n_samples
            } else {
                (fold + 2) * test_size
            };

            // Training includes all data before (test_start - purge), minus any
            // embargoed region left behind by earlier folds' test windows.
            let train_end = test_start.saturating_sub(self.purge_before);

            if train_end < 1 {
                embargo_zones.push((test_end, (test_end + self.embargo_after).min(n_samples)));
                continue;
            }

            let train_indices: Vec<usize> = (0..train_end)
                .filter(|&i| !embargo_zones.iter().any(|&(start, end)| i >= start && i < end))
                .collect();
            let test_indices: Vec<usize> = (test_start..test_end).collect();

            embargo_zones.push((test_end, (test_end + self.embargo_after).min(n_samples)));

            splits.push(TimeSeriesSplit {
                train_indices,
                test_indices,
                fold,
            });
        }

        splits
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_time_series_cv() {
        let cv = TimeSeriesCV::new(3).with_test_size(2);
        let splits = cv.split(20);

        assert!(!splits.is_empty());
        
        // Each split should have test_size test samples
        for split in &splits {
            assert_eq!(split.test_indices.len(), 2);
            // Train should come before test
            assert!(split.train_indices.last().unwrap() < split.test_indices.first().unwrap());
        }
    }

    #[test]
    fn test_walk_forward_expanding() {
        let cv = WalkForwardCV::expanding(5)
            .with_step_size(2)
            .with_test_size(1);
        
        let splits = cv.split(15);

        // First split: train [0-5), test [5-6)
        // Second split: train [0-7), test [7-8)
        // etc.
        
        assert!(!splits.is_empty());
        
        // Training size should increase
        if splits.len() >= 2 {
            assert!(splits[1].train_indices.len() > splits[0].train_indices.len());
        }
    }

    #[test]
    fn test_walk_forward_sliding() {
        let cv = WalkForwardCV::sliding(5)
            .with_step_size(1)
            .with_test_size(1);
        
        let splits = cv.split(10);

        // Training size should remain constant
        for split in &splits {
            assert_eq!(split.train_indices.len(), 5);
        }
    }

    #[test]
    fn test_blocked_cv() {
        let cv = BlockedTimeSeriesCV::new(4);
        let splits = cv.split(20);

        assert!(!splits.is_empty());
        
        // Test blocks should be non-overlapping
        for (i, split) in splits.iter().enumerate() {
            for (j, other) in splits.iter().enumerate() {
                if i != j {
                    let overlap: Vec<_> = split
                        .test_indices
                        .iter()
                        .filter(|x| other.test_indices.contains(x))
                        .collect();
                    assert!(overlap.is_empty());
                }
            }
        }
    }

    #[test]
    fn test_purged_cv() {
        let cv = PurgedTimeSeriesCV::new(3)
            .with_purge(2)
            .with_embargo(1);
        
        let splits = cv.split(30);

        assert!(!splits.is_empty());
        
        // Training end should be at least purge_before before test start
        for split in &splits {
            let train_end = *split.train_indices.last().unwrap();
            let test_start = *split.test_indices.first().unwrap();
            assert!(train_end + 2 <= test_start);
        }
    }

    #[test]
    fn test_purged_cv_embargo_respected() {
        let cv = PurgedTimeSeriesCV::new(3)
            .with_purge(0)
            .with_embargo(2);

        let splits = cv.split(30);
        assert_eq!(splits.len(), 3);

        // test_size = 30 / (3 + 1) = 7
        // Fold 0's test window is [7, 14) -> embargo zone [14, 16)
        // Fold 2's training range (train_end = test_start = 21, purge = 0) would
        // otherwise include indices 14 and 15, which must be excluded.
        let fold2 = &splits[2];
        assert!(!fold2.train_indices.contains(&14));
        assert!(!fold2.train_indices.contains(&15));
        // Indices outside the embargo zone should still be present.
        assert!(fold2.train_indices.contains(&13));
        assert!(fold2.train_indices.contains(&16));
    }
}
