//! Tamper-evident audit trail for ISO 27001 and ISO TR 24028 compliance.
//!
//! Provides hash-chained audit logging that can detect tampering,
//! with support for structured event types and queryable history.
//!
//! # ISO Standards Coverage
//! - ISO/IEC 27001:2022 Annex A.8.15: Logging
//! - ISO/IEC TR 24028:2020 Clause 8: Accountability

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use parking_lot::RwLock;

/// Types of auditable events
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AuditEventType {
    /// Authentication attempt
    Authentication {
        method: String,
        success: bool,
        client_id: String,
    },
    /// Data uploaded or imported
    DataUpload {
        filename: String,
        rows: usize,
        columns: usize,
        dataset_id: String,
    },
    /// Training job started
    TrainingStarted {
        model_type: String,
        job_id: String,
    },
    /// Training job completed
    TrainingCompleted {
        model_id: String,
        model_type: String,
        metrics: serde_json::Value,
    },
    /// Prediction made
    Prediction {
        model_id: String,
        input_hash: String,
        batch_size: usize,
    },
    /// Model exported
    ModelExported {
        model_id: String,
        format: String,
    },
    /// Model deleted
    ModelDeleted {
        model_id: String,
    },
    /// Drift detected
    DriftDetected {
        severity: u8,
        details: String,
    },
    /// Fairness violation detected
    FairnessViolation {
        attribute: String,
        metric: String,
        value: f64,
    },
    /// Data deletion request
    DataDeletion {
        dataset_id: String,
        reason: String,
    },
    /// System configuration change
    ConfigChange {
        setting: String,
        old_value: String,
        new_value: String,
    },
    /// Generic event
    Custom {
        category: String,
        details: serde_json::Value,
    },
}

/// A single entry in the tamper-evident audit trail
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuditTrailEntry {
    /// Sequential entry ID
    pub id: u64,
    /// Event timestamp
    pub timestamp: DateTime<Utc>,
    /// Hash of the previous entry (creates the chain)
    pub prev_hash: String,
    /// The event data
    pub event: AuditEventType,
    /// IP address of the requestor
    pub ip_address: String,
    /// User/client identifier
    pub client_id: String,
    /// SHA-256 hash of this entry (id + prev_hash + event + timestamp)
    pub hash: String,
}

/// Tamper-evident audit trail with hash chain
#[derive(Debug)]
pub struct AuditTrail {
    entries: Arc<RwLock<Vec<AuditTrailEntry>>>,
    max_entries: usize,
    /// The real `prev_hash` of the current first (oldest surviving) entry, as recorded
    /// at the moment it became the head due to eviction. `None` means no eviction has
    /// happened yet and the chain still starts at the genesis hash. This is the
    /// tamper-evidence anchor for `verify_integrity()`: it lets us confirm the current
    /// head's `prev_hash` still matches what we ourselves recorded when the truncation
    /// happened, instead of trusting an unauthenticated sentinel value.
    truncated_prev_hash: Arc<RwLock<Option<String>>>,
    /// Cumulative number of entries evicted from the front of the log over its lifetime.
    truncated_count: Arc<RwLock<usize>>,
    /// Monotonically increasing counter used to assign entry IDs. Unlike deriving
    /// the id from `entries.len()`, this never goes backwards after an eviction,
    /// so IDs stay unique (and sequential) across the entire lifetime of the trail,
    /// not just across the entries currently held in the buffer.
    next_id: Arc<AtomicU64>,
}

impl AuditTrail {
    /// Create a new audit trail
    pub fn new(max_entries: usize) -> Self {
        Self {
            entries: Arc::new(RwLock::new(Vec::new())),
            max_entries,
            truncated_prev_hash: Arc::new(RwLock::new(None)),
            truncated_count: Arc::new(RwLock::new(0)),
            next_id: Arc::new(AtomicU64::new(0)),
        }
    }

    /// Record a new audit event
    pub fn record(
        &self,
        event: AuditEventType,
        ip_address: &str,
        client_id: &str,
    ) -> AuditTrailEntry {
        let mut entries = self.entries.write();

        // Derived from a monotonic counter, not `entries.len()`: after the first
        // eviction below, `entries.len()` is permanently smaller than the true
        // number of entries ever written, so deriving the id from it would
        // eventually collide with a surviving entry's id. The counter only ever
        // increments, so ids stay unique across evictions.
        let id = self.next_id.fetch_add(1, Ordering::SeqCst);
        let prev_hash = entries.last()
            .map(|e| e.hash.clone())
            .unwrap_or_else(|| "0".repeat(64));

        let timestamp = Utc::now();

        // Compute hash of this entry
        let event_json = serde_json::to_string(&event).unwrap_or_default();
        let hash_input = format!("{}|{}|{}|{}|{}|{}",
            id, prev_hash, timestamp.to_rfc3339(), event_json, ip_address, client_id
        );
        let hash = compute_hash(&hash_input);

        let entry = AuditTrailEntry {
            id,
            timestamp,
            prev_hash,
            event,
            ip_address: ip_address.to_string(),
            client_id: client_id.to_string(),
            hash,
        };

        // Evict oldest entries if at capacity (keep the most recent).
        // We deliberately do NOT overwrite the new first entry's prev_hash with a
        // sentinel - that would discard real hash-chain linkage and let anyone with
        // write access to the stored entries relabel an arbitrarily-truncated head as
        // valid. Instead we leave prev_hash untouched (it already holds the real hash
        // of the entry that is being evicted) and separately record that same value as
        // metadata at the moment of eviction, so verify_integrity() can confirm the
        // head's prev_hash still matches what was legitimately recorded here rather
        // than accepting any arbitrary value.
        if entries.len() >= self.max_entries {
            let drain_count = self.max_entries / 10;
            entries.drain(..drain_count);
            *self.truncated_count.write() += drain_count;
            if let Some(first) = entries.first() {
                *self.truncated_prev_hash.write() = Some(first.prev_hash.clone());
            }
        }

        entries.push(entry.clone());
        entry
    }

    /// Verify the integrity of the audit chain
    pub fn verify_integrity(&self) -> AuditIntegrityResult {
        let entries = self.entries.read();

        if entries.is_empty() {
            return AuditIntegrityResult {
                valid: true,
                total_entries: 0,
                verified_entries: 0,
                first_invalid_id: None,
                message: "Empty audit trail".to_string(),
            };
        }

        let mut verified = 0;

        for (i, entry) in entries.iter().enumerate() {
            // Verify hash chain linkage
            if i > 0 {
                let prev = &entries[i - 1];
                if entry.prev_hash != prev.hash {
                    return AuditIntegrityResult {
                        valid: false,
                        total_entries: entries.len(),
                        verified_entries: verified,
                        first_invalid_id: Some(entry.id),
                        message: format!("Chain broken at entry {}: prev_hash mismatch", entry.id),
                    };
                }
            } else {
                // The first (oldest surviving) entry must either be the genesis entry
                // (prev_hash of all zeros, when no eviction has ever happened) or, if
                // eviction has occurred, its prev_hash must match the value we recorded
                // ourselves at the moment of truncation. This rejects any unauthenticated
                // sentinel or hand-edited prev_hash on the head entry.
                let truncated_count = *self.truncated_count.read();
                let expected_prev_hash = if truncated_count == 0 {
                    "0".repeat(64)
                } else {
                    self.truncated_prev_hash.read().clone().unwrap_or_default()
                };

                if entry.prev_hash != expected_prev_hash {
                    return AuditIntegrityResult {
                        valid: false,
                        total_entries: entries.len(),
                        verified_entries: verified,
                        first_invalid_id: Some(entry.id),
                        message: format!("First entry {} has unexpected prev_hash (possible tampering)", entry.id),
                    };
                }
            }

            // Verify entry's own hash
            let event_json = serde_json::to_string(&entry.event).unwrap_or_default();
            let hash_input = format!("{}|{}|{}|{}|{}|{}",
                entry.id, entry.prev_hash, entry.timestamp.to_rfc3339(),
                event_json, entry.ip_address, entry.client_id
            );
            let expected_hash = compute_hash(&hash_input);

            if entry.hash != expected_hash {
                return AuditIntegrityResult {
                    valid: false,
                    total_entries: entries.len(),
                    verified_entries: verified,
                    first_invalid_id: Some(entry.id),
                    message: format!("Hash mismatch at entry {}: data may have been tampered", entry.id),
                };
            }

            verified += 1;
        }

        AuditIntegrityResult {
            valid: true,
            total_entries: entries.len(),
            verified_entries: verified,
            first_invalid_id: None,
            message: "Audit trail integrity verified".to_string(),
        }
    }

    /// Query entries by event type
    pub fn query_by_type(&self, type_filter: &str) -> Vec<AuditTrailEntry> {
        let entries = self.entries.read();
        entries.iter()
            .filter(|e| {
                let type_name = match &e.event {
                    AuditEventType::Authentication { .. } => "authentication",
                    AuditEventType::DataUpload { .. } => "data_upload",
                    AuditEventType::TrainingStarted { .. } => "training_started",
                    AuditEventType::TrainingCompleted { .. } => "training_completed",
                    AuditEventType::Prediction { .. } => "prediction",
                    AuditEventType::ModelExported { .. } => "model_exported",
                    AuditEventType::ModelDeleted { .. } => "model_deleted",
                    AuditEventType::DriftDetected { .. } => "drift_detected",
                    AuditEventType::FairnessViolation { .. } => "fairness_violation",
                    AuditEventType::DataDeletion { .. } => "data_deletion",
                    AuditEventType::ConfigChange { .. } => "config_change",
                    AuditEventType::Custom { category, .. } => category.as_str(),
                };
                type_name == type_filter
            })
            .cloned()
            .collect()
    }

    /// Query entries within a time range
    pub fn query_by_time_range(
        &self,
        from: DateTime<Utc>,
        to: DateTime<Utc>,
    ) -> Vec<AuditTrailEntry> {
        let entries = self.entries.read();
        entries.iter()
            .filter(|e| e.timestamp >= from && e.timestamp <= to)
            .cloned()
            .collect()
    }

    /// Get the most recent entries
    pub fn get_recent(&self, limit: usize) -> Vec<AuditTrailEntry> {
        let entries = self.entries.read();
        entries.iter().rev().take(limit).cloned().collect()
    }

    /// Get total entry count
    pub fn count(&self) -> usize {
        self.entries.read().len()
    }
}

impl Default for AuditTrail {
    fn default() -> Self {
        Self::new(100_000)
    }
}

impl Clone for AuditTrail {
    fn clone(&self) -> Self {
        Self {
            entries: Arc::clone(&self.entries),
            max_entries: self.max_entries,
            truncated_prev_hash: Arc::clone(&self.truncated_prev_hash),
            truncated_count: Arc::clone(&self.truncated_count),
            next_id: Arc::clone(&self.next_id),
        }
    }
}

/// Result of an audit trail integrity verification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuditIntegrityResult {
    pub valid: bool,
    pub total_entries: usize,
    pub verified_entries: usize,
    pub first_invalid_id: Option<u64>,
    pub message: String,
}

fn compute_hash(input: &str) -> String {
    let mut hasher = Sha256::new();
    hasher.update(input.as_bytes());
    format!("{:x}", hasher.finalize())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_record_and_verify() {
        let trail = AuditTrail::new(1000);

        trail.record(
            AuditEventType::Authentication {
                method: "api_key".to_string(),
                success: true,
                client_id: "user1".to_string(),
            },
            "127.0.0.1",
            "user1",
        );

        trail.record(
            AuditEventType::DataUpload {
                filename: "data.csv".to_string(),
                rows: 100,
                columns: 5,
                dataset_id: "ds-001".to_string(),
            },
            "127.0.0.1",
            "user1",
        );

        let result = trail.verify_integrity();
        assert!(result.valid);
        assert_eq!(result.verified_entries, 2);
    }

    #[test]
    fn test_query_by_type() {
        let trail = AuditTrail::new(1000);

        trail.record(
            AuditEventType::Prediction {
                model_id: "m-001".to_string(),
                input_hash: "abc123".to_string(),
                batch_size: 1,
            },
            "127.0.0.1",
            "user1",
        );

        trail.record(
            AuditEventType::Authentication {
                method: "jwt".to_string(),
                success: true,
                client_id: "user2".to_string(),
            },
            "127.0.0.1",
            "user2",
        );

        let predictions = trail.query_by_type("prediction");
        assert_eq!(predictions.len(), 1);

        let auths = trail.query_by_type("authentication");
        assert_eq!(auths.len(), 1);
    }

    #[test]
    fn test_hash_chain_integrity() {
        let trail = AuditTrail::new(1000);

        for i in 0..5 {
            trail.record(
                AuditEventType::Custom {
                    category: "test".to_string(),
                    details: serde_json::json!({"iteration": i}),
                },
                "127.0.0.1",
                "test",
            );
        }

        let result = trail.verify_integrity();
        assert!(result.valid);
        assert_eq!(result.verified_entries, 5);
    }

    #[test]
    fn test_get_recent() {
        let trail = AuditTrail::new(1000);

        for _ in 0..10 {
            trail.record(
                AuditEventType::Custom {
                    category: "test".to_string(),
                    details: serde_json::json!({}),
                },
                "127.0.0.1",
                "test",
            );
        }

        let recent = trail.get_recent(3);
        assert_eq!(recent.len(), 3);
    }

    #[test]
    fn test_get_recent_limit_exceeds_total() {
        let trail = AuditTrail::new(1000);
        trail.record(
            AuditEventType::Custom { category: "t".to_string(), details: serde_json::json!({}) },
            "127.0.0.1", "test",
        );
        let recent = trail.get_recent(100);
        assert_eq!(recent.len(), 1);
    }

    #[test]
    fn test_get_recent_zero() {
        let trail = AuditTrail::new(1000);
        trail.record(
            AuditEventType::Custom { category: "t".to_string(), details: serde_json::json!({}) },
            "127.0.0.1", "test",
        );
        let recent = trail.get_recent(0);
        assert_eq!(recent.len(), 0);
    }

    #[test]
    fn test_count() {
        let trail = AuditTrail::new(1000);
        assert_eq!(trail.count(), 0);

        trail.record(
            AuditEventType::Custom { category: "t".to_string(), details: serde_json::json!({}) },
            "127.0.0.1", "test",
        );
        assert_eq!(trail.count(), 1);

        for _ in 0..4 {
            trail.record(
                AuditEventType::Custom { category: "t".to_string(), details: serde_json::json!({}) },
                "127.0.0.1", "test",
            );
        }
        assert_eq!(trail.count(), 5);
    }

    #[test]
    fn test_query_by_time_range() {
        let trail = AuditTrail::new(1000);

        let before = Utc::now();
        trail.record(
            AuditEventType::DataUpload {
                filename: "a.csv".to_string(), rows: 10, columns: 2, dataset_id: "d1".to_string(),
            },
            "127.0.0.1", "user1",
        );
        let after = Utc::now();

        let results = trail.query_by_time_range(before, after);
        assert_eq!(results.len(), 1);

        // Query range before any entries
        let far_past = Utc::now() - chrono::Duration::hours(1);
        let still_past = Utc::now() - chrono::Duration::minutes(30);
        let results = trail.query_by_time_range(far_past, still_past);
        assert_eq!(results.len(), 0);
    }

    #[test]
    fn test_verify_empty_trail() {
        let trail = AuditTrail::new(1000);
        let result = trail.verify_integrity();
        assert!(result.valid);
        assert_eq!(result.total_entries, 0);
        assert_eq!(result.verified_entries, 0);
        assert!(result.message.contains("Empty"));
    }

    #[test]
    fn test_all_event_types() {
        let trail = AuditTrail::new(1000);

        // Record one of each event type
        trail.record(AuditEventType::Authentication {
            method: "api_key".to_string(), success: true, client_id: "c1".to_string(),
        }, "127.0.0.1", "c1");

        trail.record(AuditEventType::DataUpload {
            filename: "f.csv".to_string(), rows: 10, columns: 2, dataset_id: "d1".to_string(),
        }, "127.0.0.1", "c1");

        trail.record(AuditEventType::TrainingStarted {
            model_type: "RandomForest".to_string(), job_id: "j1".to_string(),
        }, "127.0.0.1", "c1");

        trail.record(AuditEventType::TrainingCompleted {
            model_id: "m1".to_string(), model_type: "RandomForest".to_string(),
            metrics: serde_json::json!({"accuracy": 0.95}),
        }, "127.0.0.1", "c1");

        trail.record(AuditEventType::Prediction {
            model_id: "m1".to_string(), input_hash: "abc".to_string(), batch_size: 1,
        }, "127.0.0.1", "c1");

        trail.record(AuditEventType::ModelExported {
            model_id: "m1".to_string(), format: "onnx".to_string(),
        }, "127.0.0.1", "c1");

        trail.record(AuditEventType::ModelDeleted {
            model_id: "m1".to_string(),
        }, "127.0.0.1", "c1");

        trail.record(AuditEventType::DriftDetected {
            severity: 2, details: "Feature drift".to_string(),
        }, "127.0.0.1", "system");

        trail.record(AuditEventType::FairnessViolation {
            attribute: "gender".to_string(), metric: "disparate_impact".to_string(), value: 0.6,
        }, "127.0.0.1", "system");

        trail.record(AuditEventType::DataDeletion {
            dataset_id: "d1".to_string(), reason: "retention policy".to_string(),
        }, "127.0.0.1", "system");

        trail.record(AuditEventType::ConfigChange {
            setting: "rate_limit".to_string(), old_value: "100".to_string(), new_value: "200".to_string(),
        }, "127.0.0.1", "admin");

        trail.record(AuditEventType::Custom {
            category: "custom_cat".to_string(), details: serde_json::json!({"key": "val"}),
        }, "127.0.0.1", "c1");

        assert_eq!(trail.count(), 12);

        // Verify all events maintain chain integrity
        let result = trail.verify_integrity();
        assert!(result.valid);
        assert_eq!(result.verified_entries, 12);

        // Query each type
        assert_eq!(trail.query_by_type("authentication").len(), 1);
        assert_eq!(trail.query_by_type("data_upload").len(), 1);
        assert_eq!(trail.query_by_type("training_started").len(), 1);
        assert_eq!(trail.query_by_type("training_completed").len(), 1);
        assert_eq!(trail.query_by_type("prediction").len(), 1);
        assert_eq!(trail.query_by_type("model_exported").len(), 1);
        assert_eq!(trail.query_by_type("model_deleted").len(), 1);
        assert_eq!(trail.query_by_type("drift_detected").len(), 1);
        assert_eq!(trail.query_by_type("fairness_violation").len(), 1);
        assert_eq!(trail.query_by_type("data_deletion").len(), 1);
        assert_eq!(trail.query_by_type("config_change").len(), 1);
        assert_eq!(trail.query_by_type("custom_cat").len(), 1);
    }

    #[test]
    fn test_query_by_type_no_matches() {
        let trail = AuditTrail::new(1000);
        trail.record(
            AuditEventType::Prediction {
                model_id: "m".to_string(), input_hash: "h".to_string(), batch_size: 1,
            },
            "127.0.0.1", "c1",
        );
        assert_eq!(trail.query_by_type("authentication").len(), 0);
    }

    #[test]
    fn test_eviction_at_capacity() {
        let trail = AuditTrail::new(20);

        for i in 0..25 {
            trail.record(
                AuditEventType::Custom {
                    category: "test".to_string(),
                    details: serde_json::json!({"i": i}),
                },
                "127.0.0.1",
                "test",
            );
        }

        // After eviction, should have fewer than 25 entries
        assert!(trail.count() < 25);
        assert!(trail.count() > 0);
    }

    #[test]
    fn test_clone_shares_entries() {
        let trail = AuditTrail::new(1000);
        trail.record(
            AuditEventType::Custom { category: "t".to_string(), details: serde_json::json!({}) },
            "127.0.0.1", "test",
        );

        let cloned = trail.clone();
        assert_eq!(cloned.count(), 1);

        // Recording on original is visible through clone (shared Arc)
        trail.record(
            AuditEventType::Custom { category: "t".to_string(), details: serde_json::json!({}) },
            "127.0.0.1", "test",
        );
        assert_eq!(cloned.count(), 2);
    }

    #[test]
    fn test_default_trail() {
        let trail = AuditTrail::default();
        assert_eq!(trail.count(), 0);
        // Default max is 100_000
        trail.record(
            AuditEventType::Custom { category: "t".to_string(), details: serde_json::json!({}) },
            "127.0.0.1", "test",
        );
        assert_eq!(trail.count(), 1);
    }

    #[test]
    fn test_entry_fields_populated() {
        let trail = AuditTrail::new(1000);
        let entry = trail.record(
            AuditEventType::Authentication {
                method: "jwt".to_string(), success: false, client_id: "attacker".to_string(),
            },
            "192.168.1.100",
            "attacker",
        );

        assert_eq!(entry.id, 0);
        assert_eq!(entry.ip_address, "192.168.1.100");
        assert_eq!(entry.client_id, "attacker");
        assert!(!entry.hash.is_empty());
        assert_eq!(entry.prev_hash, "0".repeat(64)); // First entry has zero prev_hash
    }
}
