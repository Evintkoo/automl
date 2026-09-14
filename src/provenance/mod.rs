//! Data Provenance and Lineage Tracking Module
//!
//! Provides data lineage tracking for ISO 5259 (Data Quality for AI) and
//! ISO 5338 (AI System Lifecycle) compliance. Records the origin, transformations,
//! and schema history of datasets throughout the ML pipeline.
//!
//! # ISO Standards Coverage
//! - ISO/IEC 5259-1 Clause 6: Data quality management
//! - ISO/IEC 5338 Clause 6.3.3: Data management processes

use crate::error::{AutoMLError, Result};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::HashMap;
use std::sync::Arc;
use parking_lot::RwLock;

/// Source of a dataset
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DataSource {
    /// File uploaded via API
    FileUpload {
        filename: String,
        mime_type: String,
        size_bytes: u64,
    },
    /// Imported from Kaggle
    Kaggle { dataset_ref: String },
    /// Imported from URL
    Url { url: String },
    /// Built-in sample dataset
    Sample { name: String },
    /// Derived from another dataset
    Derived { parent_id: String },
    /// Unknown/unspecified source
    Unknown,
}

/// Record of a single transformation applied to data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TransformationRecord {
    /// Name of the transformation step (e.g., "StandardScaler", "OneHotEncoder")
    pub step: String,
    /// Parameters used for this transformation
    pub parameters: serde_json::Value,
    /// When the transformation was applied
    pub applied_at: DateTime<Utc>,
    /// Number of rows before transformation
    pub rows_before: usize,
    /// Number of rows after transformation
    pub rows_after: usize,
    /// Number of columns before transformation
    pub cols_before: usize,
    /// Number of columns after transformation
    pub cols_after: usize,
    /// Columns affected by this transformation
    pub columns_affected: Vec<String>,
    /// Duration of the transformation in milliseconds
    pub duration_ms: u64,
}

/// Snapshot of dataset schema at a point in time
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SchemaSnapshot {
    /// Columns in the schema
    pub columns: Vec<ColumnSchema>,
    /// When this snapshot was taken
    pub captured_at: DateTime<Utc>,
}

/// Schema information for a single column
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ColumnSchema {
    /// Column name
    pub name: String,
    /// Data type (e.g., "Float64", "Utf8", "Int64")
    pub dtype: String,
    /// Whether null values are present
    pub nullable: bool,
    /// Basic statistics if available
    pub stats: Option<ColumnStats>,
}

/// Basic column-level statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ColumnStats {
    pub count: usize,
    pub null_count: usize,
    pub mean: Option<f64>,
    pub std: Option<f64>,
    pub min: Option<f64>,
    pub max: Option<f64>,
    pub unique_count: Option<usize>,
}

/// Complete data lineage record for a dataset
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataLineage {
    /// Unique identifier for this dataset
    pub dataset_id: String,
    /// Human-readable name
    pub name: String,
    /// Source of the data
    pub source: DataSource,
    /// When the data was first ingested
    pub ingested_at: DateTime<Utc>,
    /// Ordered list of transformations applied
    pub transformations: Vec<TransformationRecord>,
    /// Initial schema snapshot (at ingestion time)
    pub initial_schema: SchemaSnapshot,
    /// Current schema snapshot (after all transformations)
    pub current_schema: Option<SchemaSnapshot>,
    /// Number of rows
    pub row_count: usize,
    /// SHA-256 hash of the raw data (at ingestion)
    pub data_hash: String,
    /// Arbitrary metadata tags
    pub tags: HashMap<String, String>,
}

impl DataLineage {
    /// Create a new lineage record
    pub fn new(
        dataset_id: impl Into<String>,
        name: impl Into<String>,
        source: DataSource,
        row_count: usize,
        schema: SchemaSnapshot,
        data_hash: String,
    ) -> Self {
        Self {
            dataset_id: dataset_id.into(),
            name: name.into(),
            source,
            ingested_at: Utc::now(),
            transformations: Vec::new(),
            initial_schema: schema.clone(),
            current_schema: Some(schema),
            row_count,
            data_hash,
            tags: HashMap::new(),
        }
    }

    /// Record a transformation step
    pub fn record_transformation(&mut self, record: TransformationRecord) {
        self.row_count = record.rows_after;
        self.transformations.push(record);
    }

    /// Update the current schema
    pub fn update_schema(&mut self, schema: SchemaSnapshot) {
        self.current_schema = Some(schema);
    }

    /// Add a metadata tag
    pub fn add_tag(&mut self, key: impl Into<String>, value: impl Into<String>) {
        self.tags.insert(key.into(), value.into());
    }

    /// Get total number of transformations applied
    pub fn transformation_count(&self) -> usize {
        self.transformations.len()
    }

    /// Get total processing time across all transformations
    pub fn total_processing_time_ms(&self) -> u64 {
        self.transformations.iter().map(|t| t.duration_ms).sum()
    }

    /// Generate a summary of the lineage
    pub fn summary(&self) -> LineageSummary {
        LineageSummary {
            dataset_id: self.dataset_id.clone(),
            name: self.name.clone(),
            source_type: match &self.source {
                DataSource::FileUpload { .. } => "file_upload".to_string(),
                DataSource::Kaggle { .. } => "kaggle".to_string(),
                DataSource::Url { .. } => "url".to_string(),
                DataSource::Sample { name } => format!("sample:{}", name),
                DataSource::Derived { parent_id } => format!("derived:{}", parent_id),
                DataSource::Unknown => "unknown".to_string(),
            },
            ingested_at: self.ingested_at,
            transformation_count: self.transformations.len(),
            row_count: self.row_count,
            column_count: self.current_schema.as_ref()
                .map(|s| s.columns.len())
                .unwrap_or(0),
            data_hash: self.data_hash.clone(),
        }
    }
}

/// Compact summary of lineage information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LineageSummary {
    pub dataset_id: String,
    pub name: String,
    pub source_type: String,
    pub ingested_at: DateTime<Utc>,
    pub transformation_count: usize,
    pub row_count: usize,
    pub column_count: usize,
    pub data_hash: String,
}

/// Provenance tracker that manages lineage records for all datasets
#[derive(Debug, Clone)]
pub struct ProvenanceTracker {
    /// All lineage records by dataset ID
    records: Arc<RwLock<HashMap<String, DataLineage>>>,
    /// Maximum number of records to keep
    max_records: usize,
}

impl ProvenanceTracker {
    /// Create a new provenance tracker
    pub fn new(max_records: usize) -> Self {
        Self {
            records: Arc::new(RwLock::new(HashMap::new())),
            max_records,
        }
    }

    /// Register a new dataset and return its lineage record
    pub fn register_dataset(
        &self,
        dataset_id: &str,
        name: &str,
        source: DataSource,
        row_count: usize,
        columns: Vec<ColumnSchema>,
        raw_data_sample: &[u8],
    ) -> DataLineage {
        let schema = SchemaSnapshot {
            columns,
            captured_at: Utc::now(),
        };

        let data_hash = compute_sha256(raw_data_sample);

        let lineage = DataLineage::new(
            dataset_id,
            name,
            source,
            row_count,
            schema,
            data_hash,
        );

        let mut records = self.records.write();
        // Evict oldest if at capacity
        if records.len() >= self.max_records {
            if let Some(oldest_key) = records.values()
                .min_by_key(|v| v.ingested_at)
                .map(|v| v.dataset_id.clone())
            {
                records.remove(&oldest_key);
            }
        }
        // Warn if overwriting an existing record (transformation history will be lost)
        if records.contains_key(dataset_id) {
            tracing::warn!(
                dataset_id = %dataset_id,
                "Overwriting existing provenance record — previous transformation history lost"
            );
        }
        records.insert(dataset_id.to_string(), lineage.clone());

        lineage
    }

    /// Record a transformation for a dataset
    pub fn record_transformation(
        &self,
        dataset_id: &str,
        record: TransformationRecord,
    ) -> Result<()> {
        let mut records = self.records.write();
        let lineage = records.get_mut(dataset_id).ok_or_else(|| {
            AutoMLError::ProvenanceError(format!("Dataset not found: {}", dataset_id))
        })?;
        lineage.record_transformation(record);
        Ok(())
    }

    /// Update schema for a dataset
    pub fn update_schema(
        &self,
        dataset_id: &str,
        schema: SchemaSnapshot,
    ) -> Result<()> {
        let mut records = self.records.write();
        let lineage = records.get_mut(dataset_id).ok_or_else(|| {
            AutoMLError::ProvenanceError(format!("Dataset not found: {}", dataset_id))
        })?;
        lineage.update_schema(schema);
        Ok(())
    }

    /// Get lineage for a specific dataset
    pub fn get_lineage(&self, dataset_id: &str) -> Option<DataLineage> {
        self.records.read().get(dataset_id).cloned()
    }

    /// Get all lineage summaries
    pub fn list_all(&self) -> Vec<LineageSummary> {
        self.records.read().values().map(|l| l.summary()).collect()
    }

    /// Delete lineage record for a dataset
    pub fn delete(&self, dataset_id: &str) -> bool {
        self.records.write().remove(dataset_id).is_some()
    }

    /// Get the number of tracked datasets
    pub fn count(&self) -> usize {
        self.records.read().len()
    }
}

impl Default for ProvenanceTracker {
    fn default() -> Self {
        Self::new(1000)
    }
}

/// Compute SHA-256 hash of data
pub fn compute_sha256(data: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(data);
    format!("{:x}", hasher.finalize())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_columns() -> Vec<ColumnSchema> {
        vec![
            ColumnSchema {
                name: "age".to_string(),
                dtype: "Float64".to_string(),
                nullable: false,
                stats: None,
            },
            ColumnSchema {
                name: "income".to_string(),
                dtype: "Float64".to_string(),
                nullable: true,
                stats: None,
            },
        ]
    }

    #[test]
    fn test_register_dataset() {
        let tracker = ProvenanceTracker::new(100);
        let lineage = tracker.register_dataset(
            "ds-001",
            "test_data",
            DataSource::Sample { name: "iris".to_string() },
            150,
            sample_columns(),
            b"sample data bytes",
        );

        assert_eq!(lineage.dataset_id, "ds-001");
        assert_eq!(lineage.name, "test_data");
        assert_eq!(lineage.row_count, 150);
        assert_eq!(lineage.transformation_count(), 0);
    }

    #[test]
    fn test_record_transformation() {
        let tracker = ProvenanceTracker::new(100);
        tracker.register_dataset(
            "ds-001",
            "test",
            DataSource::Unknown,
            100,
            sample_columns(),
            b"data",
        );

        let record = TransformationRecord {
            step: "StandardScaler".to_string(),
            parameters: serde_json::json!({"with_mean": true}),
            applied_at: Utc::now(),
            rows_before: 100,
            rows_after: 100,
            cols_before: 2,
            cols_after: 2,
            columns_affected: vec!["age".to_string(), "income".to_string()],
            duration_ms: 15,
        };

        tracker.record_transformation("ds-001", record).unwrap();

        let lineage = tracker.get_lineage("ds-001").unwrap();
        assert_eq!(lineage.transformation_count(), 1);
        assert_eq!(lineage.total_processing_time_ms(), 15);
    }

    #[test]
    fn test_lineage_summary() {
        let tracker = ProvenanceTracker::new(100);
        tracker.register_dataset(
            "ds-002",
            "sample_data",
            DataSource::FileUpload {
                filename: "data.csv".to_string(),
                mime_type: "text/csv".to_string(),
                size_bytes: 1024,
            },
            50,
            sample_columns(),
            b"csv data",
        );

        let summaries = tracker.list_all();
        assert_eq!(summaries.len(), 1);
        assert_eq!(summaries[0].source_type, "file_upload");
    }

    #[test]
    fn test_compute_sha256() {
        let hash = compute_sha256(b"hello world");
        assert_eq!(hash.len(), 64); // SHA-256 hex = 64 chars
    }

    #[test]
    fn test_max_records_eviction() {
        let tracker = ProvenanceTracker::new(2);
        tracker.register_dataset("ds-1", "a", DataSource::Unknown, 10, vec![], b"1");
        tracker.register_dataset("ds-2", "b", DataSource::Unknown, 20, vec![], b"2");
        tracker.register_dataset("ds-3", "c", DataSource::Unknown, 30, vec![], b"3");

        assert_eq!(tracker.count(), 2);
    }

    #[test]
    fn test_add_tag() {
        let mut lineage = DataLineage::new(
            "ds-t", "tagged", DataSource::Unknown, 10,
            SchemaSnapshot { columns: vec![], captured_at: Utc::now() },
            "hash".to_string(),
        );
        lineage.add_tag("owner", "data_team");
        lineage.add_tag("env", "production");
        lineage.add_tag("unicode", "日本語テスト");

        assert_eq!(lineage.tags.len(), 3);
        assert_eq!(lineage.tags["owner"], "data_team");
        assert_eq!(lineage.tags["unicode"], "日本語テスト");

        // Overwrite existing tag
        lineage.add_tag("owner", "ml_team");
        assert_eq!(lineage.tags["owner"], "ml_team");
        assert_eq!(lineage.tags.len(), 3);
    }

    #[test]
    fn test_update_schema() {
        let mut lineage = DataLineage::new(
            "ds-s", "schema_test", DataSource::Unknown, 10,
            SchemaSnapshot { columns: sample_columns(), captured_at: Utc::now() },
            "hash".to_string(),
        );
        assert_eq!(lineage.current_schema.as_ref().unwrap().columns.len(), 2);

        let new_schema = SchemaSnapshot {
            columns: vec![
                ColumnSchema { name: "age".to_string(), dtype: "Float64".to_string(), nullable: false, stats: None },
                ColumnSchema { name: "income".to_string(), dtype: "Float64".to_string(), nullable: true, stats: None },
                ColumnSchema { name: "age_binned".to_string(), dtype: "Utf8".to_string(), nullable: false, stats: None },
            ],
            captured_at: Utc::now(),
        };
        lineage.update_schema(new_schema);

        assert_eq!(lineage.current_schema.as_ref().unwrap().columns.len(), 3);
        // Initial schema should remain unchanged
        assert_eq!(lineage.initial_schema.columns.len(), 2);
    }

    #[test]
    fn test_tracker_update_schema() {
        let tracker = ProvenanceTracker::new(100);
        tracker.register_dataset("ds-us", "test", DataSource::Unknown, 50, sample_columns(), b"data");

        let new_schema = SchemaSnapshot {
            columns: vec![ColumnSchema {
                name: "combined".to_string(),
                dtype: "Float64".to_string(),
                nullable: false,
                stats: None,
            }],
            captured_at: Utc::now(),
        };

        tracker.update_schema("ds-us", new_schema).unwrap();
        let lineage = tracker.get_lineage("ds-us").unwrap();
        assert_eq!(lineage.current_schema.as_ref().unwrap().columns.len(), 1);
    }

    #[test]
    fn test_tracker_update_schema_missing_dataset() {
        let tracker = ProvenanceTracker::new(100);
        let schema = SchemaSnapshot { columns: vec![], captured_at: Utc::now() };
        let result = tracker.update_schema("nonexistent", schema);
        assert!(result.is_err());
    }

    #[test]
    fn test_get_lineage_existing_and_missing() {
        let tracker = ProvenanceTracker::new(100);
        tracker.register_dataset("ds-gl", "test", DataSource::Unknown, 10, vec![], b"data");

        assert!(tracker.get_lineage("ds-gl").is_some());
        assert!(tracker.get_lineage("nonexistent").is_none());
    }

    #[test]
    fn test_delete_existing_and_missing() {
        let tracker = ProvenanceTracker::new(100);
        tracker.register_dataset("ds-del", "test", DataSource::Unknown, 10, vec![], b"data");

        assert_eq!(tracker.count(), 1);
        assert!(tracker.delete("ds-del"));
        assert_eq!(tracker.count(), 0);

        // Deleting non-existent returns false
        assert!(!tracker.delete("ds-del"));
        assert!(!tracker.delete("never-existed"));
    }

    #[test]
    fn test_count_tracking() {
        let tracker = ProvenanceTracker::new(100);
        assert_eq!(tracker.count(), 0);

        tracker.register_dataset("ds-1", "a", DataSource::Unknown, 10, vec![], b"1");
        assert_eq!(tracker.count(), 1);

        tracker.register_dataset("ds-2", "b", DataSource::Unknown, 20, vec![], b"2");
        assert_eq!(tracker.count(), 2);

        tracker.delete("ds-1");
        assert_eq!(tracker.count(), 1);
    }

    #[test]
    fn test_list_all_empty() {
        let tracker = ProvenanceTracker::new(100);
        assert!(tracker.list_all().is_empty());
    }

    #[test]
    fn test_list_all_multiple_sources() {
        let tracker = ProvenanceTracker::new(100);
        tracker.register_dataset("ds-f", "file_ds", DataSource::FileUpload {
            filename: "a.csv".to_string(), mime_type: "text/csv".to_string(), size_bytes: 500,
        }, 10, vec![], b"1");
        tracker.register_dataset("ds-k", "kaggle_ds", DataSource::Kaggle {
            dataset_ref: "user/dataset".to_string(),
        }, 20, vec![], b"2");
        tracker.register_dataset("ds-u", "url_ds", DataSource::Url {
            url: "https://example.com/data.csv".to_string(),
        }, 30, vec![], b"3");
        tracker.register_dataset("ds-d", "derived_ds", DataSource::Derived {
            parent_id: "ds-f".to_string(),
        }, 10, vec![], b"4");

        let summaries = tracker.list_all();
        assert_eq!(summaries.len(), 4);

        let source_types: Vec<&str> = summaries.iter()
            .map(|s| s.source_type.as_str())
            .collect();
        assert!(source_types.contains(&"file_upload"));
        assert!(source_types.contains(&"kaggle"));
        assert!(source_types.iter().any(|s| s.starts_with("derived:")));
    }

    #[test]
    fn test_record_transformation_missing_dataset() {
        let tracker = ProvenanceTracker::new(100);
        let record = TransformationRecord {
            step: "Scaler".to_string(),
            parameters: serde_json::json!({}),
            applied_at: Utc::now(),
            rows_before: 100, rows_after: 100,
            cols_before: 2, cols_after: 2,
            columns_affected: vec![],
            duration_ms: 5,
        };
        let result = tracker.record_transformation("nonexistent", record);
        assert!(result.is_err());
    }

    #[test]
    fn test_total_processing_time() {
        let tracker = ProvenanceTracker::new(100);
        tracker.register_dataset("ds-pt", "test", DataSource::Unknown, 100, vec![], b"d");

        for dur in [10, 20, 30] {
            let record = TransformationRecord {
                step: "step".to_string(),
                parameters: serde_json::json!({}),
                applied_at: Utc::now(),
                rows_before: 100, rows_after: 100,
                cols_before: 2, cols_after: 2,
                columns_affected: vec![],
                duration_ms: dur,
            };
            tracker.record_transformation("ds-pt", record).unwrap();
        }

        let lineage = tracker.get_lineage("ds-pt").unwrap();
        assert_eq!(lineage.total_processing_time_ms(), 60);
        assert_eq!(lineage.transformation_count(), 3);
    }

    #[test]
    fn test_summary_source_types() {
        let sources = vec![
            (DataSource::FileUpload { filename: "f".into(), mime_type: "t".into(), size_bytes: 0 }, "file_upload"),
            (DataSource::Kaggle { dataset_ref: "r".into() }, "kaggle"),
            (DataSource::Url { url: "u".into() }, "url"),
            (DataSource::Sample { name: "iris".into() }, "sample:iris"),
            (DataSource::Derived { parent_id: "p".into() }, "derived:p"),
            (DataSource::Unknown, "unknown"),
        ];

        for (source, expected_type) in sources {
            let lineage = DataLineage::new(
                "id", "name", source, 10,
                SchemaSnapshot { columns: vec![], captured_at: Utc::now() },
                "hash".to_string(),
            );
            assert_eq!(lineage.summary().source_type, expected_type);
        }
    }

    #[test]
    fn test_compute_sha256_deterministic() {
        let h1 = compute_sha256(b"hello");
        let h2 = compute_sha256(b"hello");
        let h3 = compute_sha256(b"world");
        assert_eq!(h1, h2);
        assert_ne!(h1, h3);
    }

    #[test]
    fn test_compute_sha256_empty() {
        let h = compute_sha256(b"");
        assert_eq!(h.len(), 64);
    }

    #[test]
    fn test_default_tracker() {
        let tracker = ProvenanceTracker::default();
        assert_eq!(tracker.count(), 0);
        // Default max is 1000, can store many
        for i in 0..5 {
            tracker.register_dataset(&format!("ds-{}", i), "test", DataSource::Unknown, 1, vec![], b"d");
        }
        assert_eq!(tracker.count(), 5);
    }

    #[test]
    fn test_column_stats_in_schema() {
        let cols = vec![ColumnSchema {
            name: "val".to_string(),
            dtype: "Float64".to_string(),
            nullable: true,
            stats: Some(ColumnStats {
                count: 100,
                null_count: 5,
                mean: Some(42.0),
                std: Some(10.0),
                min: Some(0.0),
                max: Some(100.0),
                unique_count: Some(95),
            }),
        }];
        let tracker = ProvenanceTracker::new(100);
        let lineage = tracker.register_dataset("ds-cs", "stats_test", DataSource::Unknown, 100, cols, b"data");

        let schema = lineage.current_schema.unwrap();
        let stats = schema.columns[0].stats.as_ref().unwrap();
        assert_eq!(stats.count, 100);
        assert_eq!(stats.mean, Some(42.0));
    }
}
