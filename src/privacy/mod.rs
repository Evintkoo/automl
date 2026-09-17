//! Privacy Module — PII Detection, Anonymization, and Data Retention
//!
//! Provides tools for detecting personally identifiable information (PII),
//! anonymizing sensitive data, and enforcing data retention policies.
//!
//! # ISO Standards Coverage
//! - ISO/IEC 27701:2019 Clauses 7.2-7.5: PII handling
//! - ISO/IEC 27001:2022 Annex A.5.12-5.13: Data classification

use chrono::{DateTime, Duration, Utc};
use regex::Regex;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use parking_lot::RwLock;

/// Types of PII that can be detected
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum PiiType {
    Email,
    Phone,
    SocialSecurityNumber,
    CreditCard,
    IpAddress,
    DateOfBirth,
    PostalCode,
    Name,
    Address,
    Custom(String),
}

impl std::fmt::Display for PiiType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            PiiType::Email => write!(f, "email"),
            PiiType::Phone => write!(f, "phone"),
            PiiType::SocialSecurityNumber => write!(f, "ssn"),
            PiiType::CreditCard => write!(f, "credit_card"),
            PiiType::IpAddress => write!(f, "ip_address"),
            PiiType::DateOfBirth => write!(f, "date_of_birth"),
            PiiType::PostalCode => write!(f, "postal_code"),
            PiiType::Name => write!(f, "name"),
            PiiType::Address => write!(f, "address"),
            PiiType::Custom(s) => write!(f, "custom:{}", s),
        }
    }
}

/// Result of scanning a single column for PII
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ColumnPiiResult {
    /// Column name
    pub column: String,
    /// Detected PII types
    pub detected_types: Vec<PiiType>,
    /// Confidence score for each detection (0.0 - 1.0)
    pub confidence: HashMap<String, f64>,
    /// Number of values that matched PII patterns
    pub match_count: usize,
    /// Total values scanned
    pub total_count: usize,
    /// Recommendation
    pub recommendation: PiiRecommendation,
}

/// Recommendation for handling detected PII
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PiiRecommendation {
    /// No PII detected, safe to use
    Safe,
    /// PII detected, should anonymize before use
    Anonymize,
    /// PII detected, should remove column
    Remove,
    /// PII detected, needs manual review
    Review,
}

/// Result of a full PII scan on a dataset
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PiiScanResult {
    /// Per-column results
    pub columns: Vec<ColumnPiiResult>,
    /// Total PII columns detected
    pub pii_column_count: usize,
    /// Overall risk level
    pub risk_level: RiskLevel,
    /// Summary recommendations
    pub recommendations: Vec<String>,
}

/// Overall risk level of a dataset
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum RiskLevel {
    Low,
    Medium,
    High,
    Critical,
}

/// PII scanner for detecting sensitive data in datasets
pub struct PiiScanner {
    /// Compiled regex patterns for PII detection
    patterns: Vec<(PiiType, Regex)>,
    /// Column name heuristics
    name_patterns: Vec<(PiiType, Regex)>,
}

impl PiiScanner {
    /// Create a new PII scanner with default patterns
    pub fn new() -> Self {
        let patterns = vec![
            (PiiType::Email, Regex::new(r"^[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}$").unwrap()),
            (PiiType::Phone, Regex::new(r"^(\+?\d{1,3}[\s\-]?)?\(?\d{2,4}\)?[\s\-]?\d{3,4}[\s\-]?\d{3,4}$").unwrap()),
            (PiiType::SocialSecurityNumber, Regex::new(r"^\d{3}-?\d{2}-?\d{4}$").unwrap()),
            (PiiType::CreditCard, Regex::new(r"^\d{4}[\s\-]?\d{4}[\s\-]?\d{4}[\s\-]?\d{4}$").unwrap()),
            (PiiType::IpAddress, Regex::new(r"^(?:(?:25[0-5]|2[0-4]\d|[01]?\d\d?)\.){3}(?:25[0-5]|2[0-4]\d|[01]?\d\d?)$").unwrap()),
            (PiiType::PostalCode, Regex::new(r"^\d{5}(-\d{4})?$").unwrap()),
        ];

        let name_patterns = vec![
            (PiiType::Email, Regex::new(r"(?i)(e[\-_]?mail|email_addr)").unwrap()),
            (PiiType::Phone, Regex::new(r"(?i)(phone|tel|mobile|cell)").unwrap()),
            (PiiType::SocialSecurityNumber, Regex::new(r"(?i)(ssn|social_sec|national_id)").unwrap()),
            (PiiType::CreditCard, Regex::new(r"(?i)(credit_card|card_num|cc_num)").unwrap()),
            (PiiType::Name, Regex::new(r"(?i)(first_?name|last_?name|full_?name|surname|given_name)").unwrap()),
            (PiiType::Address, Regex::new(r"(?i)(address|street|city|state|zip|postal)").unwrap()),
            (PiiType::DateOfBirth, Regex::new(r"(?i)(dob|date_of_birth|birth_?date|birthday)").unwrap()),
            (PiiType::IpAddress, Regex::new(r"(?i)(ip_addr|ip_address|source_ip|client_ip)").unwrap()),
        ];

        Self { patterns, name_patterns }
    }

    /// Scan a column of string values for PII
    pub fn scan_column(&self, column_name: &str, values: &[String]) -> ColumnPiiResult {
        let total_count = values.len();
        let mut detected_types = Vec::new();
        let mut confidence = HashMap::new();

        // Check column name heuristics
        for (pii_type, pattern) in &self.name_patterns {
            if pattern.is_match(column_name) {
                detected_types.push(pii_type.clone());
                confidence.insert(pii_type.to_string(), 0.7);
            }
        }

        // Check value patterns (sample up to 1000 values)
        let sample_size = total_count.min(1000);
        let sample: Vec<&String> = values.iter().take(sample_size).collect();

        let mut match_counts: HashMap<String, usize> = HashMap::new();

        for value in &sample {
            let trimmed = value.trim();
            if trimmed.is_empty() {
                continue;
            }
            for (pii_type, pattern) in &self.patterns {
                if pattern.is_match(trimmed) {
                    *match_counts.entry(pii_type.to_string()).or_insert(0) += 1;
                }
            }
        }

        let mut match_count = 0;
        for (pii_type_str, count) in &match_counts {
            let ratio = *count as f64 / sample_size.max(1) as f64;
            if ratio > 0.1 {
                // More than 10% of sampled values match
                match_count = match_count.max(*count);
                let pii_type = self.patterns.iter()
                    .find(|(pt, _)| pt.to_string() == *pii_type_str)
                    .map(|(pt, _)| pt.clone());
                if let Some(pt) = pii_type {
                    if !detected_types.contains(&pt) {
                        detected_types.push(pt);
                    }
                }
                confidence.insert(pii_type_str.clone(), ratio.min(1.0));
            }
        }

        // Credit card: validate with Luhn algorithm for high-confidence matches
        if detected_types.contains(&PiiType::CreditCard) {
            let luhn_valid = sample.iter()
                .filter(|v| luhn_check(&v.replace(['-', ' '], "")))
                .count();
            let luhn_ratio = luhn_valid as f64 / sample_size.max(1) as f64;
            confidence.insert("credit_card".to_string(), luhn_ratio);
            if luhn_ratio < 0.05 {
                detected_types.retain(|t| t != &PiiType::CreditCard);
            }
        }

        let recommendation = if detected_types.is_empty() {
            PiiRecommendation::Safe
        } else if detected_types.iter().any(|t| matches!(t,
            PiiType::SocialSecurityNumber | PiiType::CreditCard))
        {
            PiiRecommendation::Remove
        } else if detected_types.iter().any(|t| matches!(t,
            PiiType::Email | PiiType::Phone | PiiType::Name))
        {
            PiiRecommendation::Anonymize
        } else {
            PiiRecommendation::Review
        };

        ColumnPiiResult {
            column: column_name.to_string(),
            detected_types,
            confidence,
            match_count,
            total_count,
            recommendation,
        }
    }

    /// Scan multiple columns and produce a full dataset PII report
    pub fn scan_dataset(
        &self,
        columns: &HashMap<String, Vec<String>>,
    ) -> PiiScanResult {
        let mut column_results = Vec::new();
        let mut pii_count = 0;
        let mut recommendations = Vec::new();

        for (col_name, values) in columns {
            let result = self.scan_column(col_name, values);
            if !result.detected_types.is_empty() {
                pii_count += 1;
                match &result.recommendation {
                    PiiRecommendation::Remove => {
                        recommendations.push(format!(
                            "REMOVE column '{}': contains {}",
                            col_name,
                            result.detected_types.iter()
                                .map(|t| t.to_string())
                                .collect::<Vec<_>>().join(", ")
                        ));
                    }
                    PiiRecommendation::Anonymize => {
                        recommendations.push(format!(
                            "ANONYMIZE column '{}': contains {}",
                            col_name,
                            result.detected_types.iter()
                                .map(|t| t.to_string())
                                .collect::<Vec<_>>().join(", ")
                        ));
                    }
                    PiiRecommendation::Review => {
                        recommendations.push(format!(
                            "REVIEW column '{}': may contain PII",
                            col_name
                        ));
                    }
                    PiiRecommendation::Safe => {}
                }
            }
            column_results.push(result);
        }

        let risk_level = match pii_count {
            0 => RiskLevel::Low,
            1..=2 => RiskLevel::Medium,
            3..=5 => RiskLevel::High,
            _ => RiskLevel::Critical,
        };

        PiiScanResult {
            columns: column_results,
            pii_column_count: pii_count,
            risk_level,
            recommendations,
        }
    }
}

impl Default for PiiScanner {
    fn default() -> Self {
        Self::new()
    }
}

/// Anonymization strategy
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AnonymizationMethod {
    /// Remove the column entirely
    Suppress,
    /// Replace with an HMAC-SHA256-based pseudonym.
    ///
    /// `secret_key` is used as the HMAC key and **must be treated as a
    /// secret** (like a signing key), not a public salt — anyone who learns
    /// it can brute-force low-cardinality inputs (SSNs, phone numbers,
    /// common names) and can re-link pseudonyms across any dataset that
    /// reuses the same key.
    Pseudonymize { secret_key: String },
    /// Generalize values (e.g., age -> age range)
    Generalize { bins: Vec<f64> },
    /// Mask part of the value (e.g., email -> j***@example.com)
    Mask { visible_chars: usize },
    /// Replace with random value from same distribution (Fisher-Yates shuffle).
    ///
    /// `seed`: when `None` (the recommended default), the shuffle RNG is
    /// seeded from OS entropy so the resulting permutation cannot be
    /// reconstructed or reversed. Pass `Some(seed)` only to get a
    /// reproducible/deterministic shuffle for testing — never as the
    /// default for data that needs a real privacy guarantee.
    Randomize { seed: Option<u64> },
}

/// Anonymizer for applying privacy transformations
pub struct Anonymizer;

impl Anonymizer {
    /// Anonymize a column of string values
    pub fn anonymize_column(
        values: &[String],
        method: &AnonymizationMethod,
    ) -> Vec<String> {
        match method {
            AnonymizationMethod::Suppress => {
                vec!["[REDACTED]".to_string(); values.len()]
            }
            AnonymizationMethod::Pseudonymize { secret_key } => {
                use hmac::{Hmac, Mac};
                use sha2::Sha256;
                type HmacSha256 = Hmac<Sha256>;
                values.iter().map(|v| {
                    // Full HMAC-SHA256 output (256 bits), not a truncated hash of a
                    // public salt: this resists precomputation/brute-force attacks
                    // as long as `secret_key` stays secret.
                    let mut mac = HmacSha256::new_from_slice(secret_key.as_bytes())
                        .expect("HMAC-SHA256 accepts a key of any length");
                    mac.update(v.as_bytes());
                    let digest = mac.finalize().into_bytes();
                    format!("pseudo_{:x}", digest)
                }).collect()
            }
            AnonymizationMethod::Mask { visible_chars } => {
                values.iter().map(|v| {
                    if v.len() <= *visible_chars {
                        "*".repeat(v.len())
                    } else {
                        let visible: String = v.chars().take(*visible_chars).collect();
                        format!("{}***", visible)
                    }
                }).collect()
            }
            AnonymizationMethod::Generalize { bins } => {
                values.iter().map(|v| {
                    if let Ok(val) = v.parse::<f64>() {
                        for i in 0..bins.len().saturating_sub(1) {
                            if val >= bins[i] && val < bins[i + 1] {
                                return format!("{}-{}", bins[i], bins[i + 1]);
                            }
                        }
                        if let Some(&last) = bins.last() {
                            if val >= last {
                                return format!("{}+", last);
                            }
                        }
                        v.clone()
                    } else {
                        v.clone()
                    }
                }).collect()
            }
            AnonymizationMethod::Randomize { seed } => {
                // Fisher-Yates shuffle. Default path (seed == None) draws a fresh
                // seed from OS entropy on every call, so the permutation cannot be
                // precomputed or reversed by anyone who knows this code — unlike a
                // hardcoded seed, which would make the shuffle a pure function of
                // `values.len()` and fully reconstructible.
                use rand::prelude::*;
                use rand_xoshiro::Xoshiro256PlusPlus;
                let mut shuffled = values.to_vec();
                let mut rng = match seed {
                    Some(s) => Xoshiro256PlusPlus::seed_from_u64(*s),
                    None => Xoshiro256PlusPlus::seed_from_u64(rand::random::<u64>()),
                };
                shuffled.shuffle(&mut rng);
                shuffled
            }
        }
    }

    /// Check k-anonymity: returns the minimum group size when grouping by quasi-identifiers
    pub fn check_k_anonymity(
        quasi_identifiers: &[Vec<String>],
    ) -> usize {
        if quasi_identifiers.is_empty() {
            return 0;
        }

        let n = quasi_identifiers[0].len();
        let mut groups: HashMap<Vec<&str>, usize> = HashMap::new();

        for i in 0..n {
            let key: Vec<&str> = quasi_identifiers.iter()
                .map(|col| col.get(i).map(|s| s.as_str()).unwrap_or(""))
                .collect();
            *groups.entry(key).or_insert(0) += 1;
        }

        groups.values().copied().min().unwrap_or(0)
    }
}

/// Data classification level (ISO 27001 Annex A.5.12)
#[derive(Debug, Clone, Hash, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum DataClassification {
    Public,
    Internal,
    Confidential,
    Restricted,
}

impl std::fmt::Display for DataClassification {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DataClassification::Public => write!(f, "Public"),
            DataClassification::Internal => write!(f, "Internal"),
            DataClassification::Confidential => write!(f, "Confidential"),
            DataClassification::Restricted => write!(f, "Restricted"),
        }
    }
}

/// Retention policy for data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RetentionPolicy {
    /// Data classification level
    pub classification: DataClassification,
    /// Maximum retention period in days (None = indefinite)
    pub retention_days: Option<u64>,
    /// Whether data can be deleted on user request
    pub allows_deletion_request: bool,
}

impl RetentionPolicy {
    /// Default policies per classification level
    pub fn default_for(classification: DataClassification) -> Self {
        match classification {
            DataClassification::Public => Self {
                classification: DataClassification::Public,
                retention_days: None,
                allows_deletion_request: false,
            },
            DataClassification::Internal => Self {
                classification: DataClassification::Internal,
                retention_days: Some(365),
                allows_deletion_request: true,
            },
            DataClassification::Confidential => Self {
                classification: DataClassification::Confidential,
                retention_days: Some(90),
                allows_deletion_request: true,
            },
            DataClassification::Restricted => Self {
                classification: DataClassification::Restricted,
                retention_days: Some(30),
                allows_deletion_request: true,
            },
        }
    }

    /// Check if data has exceeded its retention period
    pub fn is_expired(&self, created_at: DateTime<Utc>) -> bool {
        if let Some(days) = self.retention_days {
            let expiry = created_at + Duration::days(days as i64);
            Utc::now() > expiry
        } else {
            false
        }
    }
}

/// Retention manager for tracking data lifecycle
#[derive(Debug, Clone)]
pub struct RetentionManager {
    policies: HashMap<DataClassification, RetentionPolicy>,
    records: Arc<RwLock<Vec<RetentionRecord>>>,
}

/// Record of a data item's retention status
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RetentionRecord {
    pub dataset_id: String,
    pub classification: DataClassification,
    pub created_at: DateTime<Utc>,
    pub expires_at: Option<DateTime<Utc>>,
    pub deleted: bool,
    pub deletion_reason: Option<String>,
}

impl RetentionManager {
    /// Create a new retention manager with default policies
    pub fn new() -> Self {
        let mut policies = HashMap::new();
        policies.insert(
            DataClassification::Public,
            RetentionPolicy::default_for(DataClassification::Public),
        );
        policies.insert(
            DataClassification::Internal,
            RetentionPolicy::default_for(DataClassification::Internal),
        );
        policies.insert(
            DataClassification::Confidential,
            RetentionPolicy::default_for(DataClassification::Confidential),
        );
        policies.insert(
            DataClassification::Restricted,
            RetentionPolicy::default_for(DataClassification::Restricted),
        );

        Self {
            policies,
            records: Arc::new(RwLock::new(Vec::new())),
        }
    }

    /// Register a data item for retention tracking
    pub fn register(&self, dataset_id: &str, classification: DataClassification) {
        let now = Utc::now();
        let expires_at = self.policies.get(&classification)
            .and_then(|p| p.retention_days)
            .map(|days| now + Duration::days(days as i64));

        self.records.write().push(RetentionRecord {
            dataset_id: dataset_id.to_string(),
            classification,
            created_at: now,
            expires_at,
            deleted: false,
            deletion_reason: None,
        });
    }

    /// Get all expired records that should be purged
    pub fn get_expired(&self) -> Vec<RetentionRecord> {
        let now = Utc::now();
        self.records.read().iter()
            .filter(|r| !r.deleted)
            .filter(|r| r.expires_at.map(|e| now > e).unwrap_or(false))
            .cloned()
            .collect()
    }

    /// Mark a dataset as deleted
    pub fn mark_deleted(&self, dataset_id: &str, reason: &str) -> bool {
        let mut records = self.records.write();
        if let Some(record) = records.iter_mut().find(|r| r.dataset_id == dataset_id) {
            record.deleted = true;
            record.deletion_reason = Some(reason.to_string());
            true
        } else {
            false
        }
    }

    /// Get retention status for a dataset
    pub fn get_status(&self, dataset_id: &str) -> Option<RetentionRecord> {
        self.records.read().iter()
            .find(|r| r.dataset_id == dataset_id)
            .cloned()
    }

    /// Get all active (non-deleted) records
    pub fn get_active(&self) -> Vec<RetentionRecord> {
        self.records.read().iter()
            .filter(|r| !r.deleted)
            .cloned()
            .collect()
    }
}

impl Default for RetentionManager {
    fn default() -> Self {
        Self::new()
    }
}

/// Luhn algorithm check for credit card validation
fn luhn_check(number: &str) -> bool {
    let digits: Vec<u32> = number.chars()
        .filter(|c| c.is_ascii_digit())
        .filter_map(|c| c.to_digit(10))
        .collect();

    if digits.len() < 13 || digits.len() > 19 {
        return false;
    }

    let sum: u32 = digits.iter().rev().enumerate().map(|(i, &d)| {
        if i % 2 == 1 {
            let doubled = d * 2;
            if doubled > 9 { doubled - 9 } else { doubled }
        } else {
            d
        }
    }).sum();

    sum % 10 == 0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pii_scanner_email() {
        let scanner = PiiScanner::new();
        let values = vec![
            "user@example.com".to_string(),
            "admin@test.org".to_string(),
            "hello@world.net".to_string(),
        ];
        let result = scanner.scan_column("email_addr", &values);
        assert!(!result.detected_types.is_empty());
    }

    #[test]
    fn test_pii_scanner_clean() {
        let scanner = PiiScanner::new();
        let values = vec![
            "cat".to_string(),
            "dog".to_string(),
            "bird".to_string(),
        ];
        let result = scanner.scan_column("animal", &values);
        assert!(result.detected_types.is_empty());
        assert!(matches!(result.recommendation, PiiRecommendation::Safe));
    }

    #[test]
    fn test_anonymize_suppress() {
        let values = vec!["secret1".to_string(), "secret2".to_string()];
        let result = Anonymizer::anonymize_column(&values, &AnonymizationMethod::Suppress);
        assert!(result.iter().all(|v| v == "[REDACTED]"));
    }

    #[test]
    fn test_anonymize_pseudonymize() {
        let values = vec!["alice".to_string(), "bob".to_string()];
        let result = Anonymizer::anonymize_column(
            &values,
            &AnonymizationMethod::Pseudonymize { secret_key: "test_secret".to_string() },
        );
        assert!(result.iter().all(|v| v.starts_with("pseudo_")));
        assert_ne!(result[0], result[1]);
    }

    #[test]
    fn test_anonymize_mask() {
        let values = vec!["john.doe@email.com".to_string()];
        let result = Anonymizer::anonymize_column(
            &values,
            &AnonymizationMethod::Mask { visible_chars: 3 },
        );
        assert_eq!(result[0], "joh***");
    }

    #[test]
    fn test_k_anonymity() {
        let col1 = vec!["A".to_string(), "A".to_string(), "B".to_string(), "B".to_string()];
        let col2 = vec!["X".to_string(), "X".to_string(), "Y".to_string(), "Y".to_string()];
        let k = Anonymizer::check_k_anonymity(&[col1, col2]);
        assert_eq!(k, 2);
    }

    #[test]
    fn test_retention_policy() {
        let policy = RetentionPolicy::default_for(DataClassification::Restricted);
        assert_eq!(policy.retention_days, Some(30));
        assert!(policy.allows_deletion_request);
    }

    #[test]
    fn test_retention_manager() {
        let manager = RetentionManager::new();
        manager.register("ds-001", DataClassification::Confidential);
        let status = manager.get_status("ds-001").unwrap();
        assert!(!status.deleted);
        assert!(status.expires_at.is_some());
    }

    #[test]
    fn test_luhn_check() {
        assert!(luhn_check("4111111111111111")); // Valid Visa test number
        assert!(!luhn_check("1234567890123456")); // Invalid
    }

    #[test]
    fn test_data_classification_ordering() {
        assert!(DataClassification::Public < DataClassification::Internal);
        assert!(DataClassification::Internal < DataClassification::Confidential);
        assert!(DataClassification::Confidential < DataClassification::Restricted);
    }

    // --- PII Scanner extended tests ---

    #[test]
    fn test_pii_scanner_phone() {
        let scanner = PiiScanner::new();
        let values = vec![
            "+1-555-123-4567".to_string(),
            "(555) 987-6543".to_string(),
            "555-111-2222".to_string(),
        ];
        let result = scanner.scan_column("phone_number", &values);
        assert!(!result.detected_types.is_empty());
    }

    #[test]
    fn test_pii_scanner_ssn() {
        let scanner = PiiScanner::new();
        let values = vec![
            "123-45-6789".to_string(),
            "987-65-4321".to_string(),
            "111-22-3333".to_string(),
        ];
        let result = scanner.scan_column("ssn", &values);
        assert!(result.detected_types.contains(&PiiType::SocialSecurityNumber));
        assert!(matches!(result.recommendation, PiiRecommendation::Remove));
    }

    #[test]
    fn test_pii_scanner_ip_address() {
        let scanner = PiiScanner::new();
        let values = vec![
            "192.168.1.1".to_string(),
            "10.0.0.1".to_string(),
            "172.16.0.100".to_string(),
        ];
        let result = scanner.scan_column("source_ip", &values);
        assert!(!result.detected_types.is_empty());
    }

    #[test]
    fn test_pii_scanner_column_name_heuristics() {
        let scanner = PiiScanner::new();
        // Column name alone triggers detection even with non-matching values
        let values = vec!["John".to_string(), "Jane".to_string(), "Bob".to_string()];
        let result = scanner.scan_column("first_name", &values);
        assert!(result.detected_types.contains(&PiiType::Name));
    }

    #[test]
    fn test_pii_scanner_empty_values() {
        let scanner = PiiScanner::new();
        let values: Vec<String> = vec![];
        let result = scanner.scan_column("data", &values);
        assert_eq!(result.total_count, 0);
        assert!(result.detected_types.is_empty());
    }

    #[test]
    fn test_pii_scanner_empty_strings() {
        let scanner = PiiScanner::new();
        let values = vec!["".to_string(), "".to_string(), "".to_string()];
        let result = scanner.scan_column("notes", &values);
        assert!(result.detected_types.is_empty());
    }

    #[test]
    fn test_scan_dataset_multiple_columns() {
        let scanner = PiiScanner::new();
        let mut columns = HashMap::new();
        columns.insert("email".to_string(), vec![
            "a@b.com".to_string(), "c@d.org".to_string(), "e@f.net".to_string(),
        ]);
        columns.insert("age".to_string(), vec![
            "25".to_string(), "30".to_string(), "45".to_string(),
        ]);
        columns.insert("ssn".to_string(), vec![
            "123-45-6789".to_string(), "987-65-4321".to_string(), "111-22-3333".to_string(),
        ]);

        let result = scanner.scan_dataset(&columns);
        assert!(result.pii_column_count >= 2);
        assert!(result.risk_level == RiskLevel::Medium || result.risk_level == RiskLevel::High);
        assert!(!result.recommendations.is_empty());
    }

    #[test]
    fn test_scan_dataset_clean() {
        let scanner = PiiScanner::new();
        let mut columns = HashMap::new();
        columns.insert("color".to_string(), vec!["red".to_string(), "blue".to_string()]);
        columns.insert("size".to_string(), vec!["small".to_string(), "large".to_string()]);

        let result = scanner.scan_dataset(&columns);
        assert_eq!(result.pii_column_count, 0);
        assert_eq!(result.risk_level, RiskLevel::Low);
    }

    #[test]
    fn test_scan_dataset_critical_risk() {
        let scanner = PiiScanner::new();
        let mut columns = HashMap::new();
        // 6+ PII columns => Critical
        for name in &["email_addr", "phone", "ssn", "first_name", "last_name", "address"] {
            columns.insert(name.to_string(), vec!["dummy".to_string()]);
        }
        let result = scanner.scan_dataset(&columns);
        assert_eq!(result.risk_level, RiskLevel::Critical);
    }

    // --- Anonymization extended tests ---

    #[test]
    fn test_anonymize_generalize() {
        let values = vec![
            "25".to_string(), "32".to_string(), "47".to_string(), "65".to_string(), "18".to_string(),
        ];
        let result = Anonymizer::anonymize_column(
            &values,
            &AnonymizationMethod::Generalize { bins: vec![0.0, 30.0, 60.0, 90.0] },
        );
        assert_eq!(result[0], "0-30");   // 25
        assert_eq!(result[1], "30-60");  // 32
        assert_eq!(result[2], "30-60");  // 47
        assert_eq!(result[3], "60-90");  // 65
        assert_eq!(result[4], "0-30");   // 18
    }

    #[test]
    fn test_anonymize_generalize_above_max_bin() {
        let values = vec!["100".to_string()];
        let result = Anonymizer::anonymize_column(
            &values,
            &AnonymizationMethod::Generalize { bins: vec![0.0, 50.0] },
        );
        assert_eq!(result[0], "50+");
    }

    #[test]
    fn test_anonymize_generalize_non_numeric() {
        let values = vec!["hello".to_string(), "world".to_string()];
        let result = Anonymizer::anonymize_column(
            &values,
            &AnonymizationMethod::Generalize { bins: vec![0.0, 50.0] },
        );
        // Non-numeric values pass through unchanged
        assert_eq!(result[0], "hello");
        assert_eq!(result[1], "world");
    }

    #[test]
    fn test_anonymize_randomize() {
        let values = vec![
            "alice".to_string(), "bob".to_string(), "charlie".to_string(),
            "dave".to_string(), "eve".to_string(),
        ];
        let result = Anonymizer::anonymize_column(&values, &AnonymizationMethod::Randomize { seed: Some(42) });
        // Same length, same elements (just shuffled)
        assert_eq!(result.len(), values.len());
        let mut sorted_orig: Vec<_> = values.clone();
        let mut sorted_result: Vec<_> = result.clone();
        sorted_orig.sort();
        sorted_result.sort();
        assert_eq!(sorted_orig, sorted_result);
    }

    #[test]
    fn test_anonymize_mask_short_string() {
        let values = vec!["ab".to_string()];
        let result = Anonymizer::anonymize_column(
            &values,
            &AnonymizationMethod::Mask { visible_chars: 5 },
        );
        // String shorter than visible_chars => all masked
        assert_eq!(result[0], "**");
    }

    #[test]
    fn test_anonymize_pseudonymize_deterministic() {
        let values = vec!["alice".to_string()];
        let method = AnonymizationMethod::Pseudonymize { secret_key: "s1".to_string() };
        let r1 = Anonymizer::anonymize_column(&values, &method);
        let r2 = Anonymizer::anonymize_column(&values, &method);
        assert_eq!(r1, r2); // Same input + key => same output

        let method2 = AnonymizationMethod::Pseudonymize { secret_key: "s2".to_string() };
        let r3 = Anonymizer::anonymize_column(&values, &method2);
        assert_ne!(r1, r3); // Different key => different output
    }

    #[test]
    fn test_anonymize_pseudonymize_full_hmac_output() {
        // Full HMAC-SHA256 hex output (32 bytes = 64 hex chars), not a
        // truncated 48-bit hash, so it isn't feasible to brute-force over
        // the full output space.
        let values = vec!["bob".to_string()];
        let result = Anonymizer::anonymize_column(
            &values,
            &AnonymizationMethod::Pseudonymize { secret_key: "secret".to_string() },
        );
        let hex_part = result[0].strip_prefix("pseudo_").unwrap();
        assert_eq!(hex_part.len(), 64);
    }

    #[test]
    fn test_anonymize_randomize_default_seed_is_not_hardcoded() {
        // Two default-seeded (entropy-seeded) shuffles of a large-enough
        // input should not reliably produce the same permutation, unlike
        // the old hardcoded-seed implementation.
        let values: Vec<String> = (0..20).map(|i| i.to_string()).collect();
        let r1 = Anonymizer::anonymize_column(&values, &AnonymizationMethod::Randomize { seed: None });
        let r2 = Anonymizer::anonymize_column(&values, &AnonymizationMethod::Randomize { seed: None });
        assert_ne!(r1, r2, "entropy-seeded shuffles should not be deterministic across calls");
    }

    #[test]
    fn test_anonymize_randomize_explicit_seed_is_reproducible() {
        let values: Vec<String> = (0..10).map(|i| i.to_string()).collect();
        let r1 = Anonymizer::anonymize_column(&values, &AnonymizationMethod::Randomize { seed: Some(7) });
        let r2 = Anonymizer::anonymize_column(&values, &AnonymizationMethod::Randomize { seed: Some(7) });
        assert_eq!(r1, r2, "an explicit seed must still be reproducible for testing");
    }

    // --- k-anonymity extended tests ---

    #[test]
    fn test_k_anonymity_empty_quasi_identifiers() {
        let k = Anonymizer::check_k_anonymity(&[]);
        assert_eq!(k, 0);
    }

    #[test]
    fn test_k_anonymity_single_row() {
        let col = vec!["A".to_string()];
        let k = Anonymizer::check_k_anonymity(&[col]);
        assert_eq!(k, 1);
    }

    #[test]
    fn test_k_anonymity_all_same() {
        let col = vec!["A".to_string(), "A".to_string(), "A".to_string()];
        let k = Anonymizer::check_k_anonymity(&[col]);
        assert_eq!(k, 3);
    }

    #[test]
    fn test_k_anonymity_all_unique() {
        let col = vec!["A".to_string(), "B".to_string(), "C".to_string()];
        let k = Anonymizer::check_k_anonymity(&[col]);
        assert_eq!(k, 1);
    }

    // --- RetentionManager extended tests ---

    #[test]
    fn test_retention_manager_get_expired() {
        let manager = RetentionManager::new();
        // Register with restricted classification (30-day retention)
        // We can't easily time-travel, but we can verify the method runs
        manager.register("ds-exp", DataClassification::Restricted);
        let expired = manager.get_expired();
        // Just registered, so shouldn't be expired
        assert!(expired.is_empty());
    }

    #[test]
    fn test_retention_manager_mark_deleted() {
        let manager = RetentionManager::new();
        manager.register("ds-md", DataClassification::Internal);

        assert!(manager.mark_deleted("ds-md", "User request"));

        let status = manager.get_status("ds-md").unwrap();
        assert!(status.deleted);
        assert_eq!(status.deletion_reason.as_deref(), Some("User request"));

        // Can't delete non-existent
        assert!(!manager.mark_deleted("nonexistent", "reason"));
    }

    #[test]
    fn test_retention_manager_get_status_missing() {
        let manager = RetentionManager::new();
        assert!(manager.get_status("nonexistent").is_none());
    }

    #[test]
    fn test_retention_manager_get_active() {
        let manager = RetentionManager::new();
        manager.register("ds-a1", DataClassification::Public);
        manager.register("ds-a2", DataClassification::Internal);
        manager.register("ds-a3", DataClassification::Confidential);

        assert_eq!(manager.get_active().len(), 3);

        manager.mark_deleted("ds-a2", "cleanup");
        assert_eq!(manager.get_active().len(), 2);
    }

    #[test]
    fn test_retention_policy_public_no_expiry() {
        let policy = RetentionPolicy::default_for(DataClassification::Public);
        assert_eq!(policy.retention_days, None);
        assert!(!policy.allows_deletion_request);
        // Public data never expires
        assert!(!policy.is_expired(Utc::now() - Duration::days(10000)));
    }

    #[test]
    fn test_retention_policy_all_classifications() {
        let public = RetentionPolicy::default_for(DataClassification::Public);
        let internal = RetentionPolicy::default_for(DataClassification::Internal);
        let confidential = RetentionPolicy::default_for(DataClassification::Confidential);
        let restricted = RetentionPolicy::default_for(DataClassification::Restricted);

        assert_eq!(public.retention_days, None);
        assert_eq!(internal.retention_days, Some(365));
        assert_eq!(confidential.retention_days, Some(90));
        assert_eq!(restricted.retention_days, Some(30));
    }

    #[test]
    fn test_retention_manager_public_no_expires_at() {
        let manager = RetentionManager::new();
        manager.register("ds-pub", DataClassification::Public);
        let status = manager.get_status("ds-pub").unwrap();
        assert!(status.expires_at.is_none());
    }

    #[test]
    fn test_retention_manager_restricted_has_expires_at() {
        let manager = RetentionManager::new();
        manager.register("ds-rst", DataClassification::Restricted);
        let status = manager.get_status("ds-rst").unwrap();
        assert!(status.expires_at.is_some());
    }

    // --- Display trait tests ---

    #[test]
    fn test_pii_type_display() {
        assert_eq!(PiiType::Email.to_string(), "email");
        assert_eq!(PiiType::Phone.to_string(), "phone");
        assert_eq!(PiiType::SocialSecurityNumber.to_string(), "ssn");
        assert_eq!(PiiType::CreditCard.to_string(), "credit_card");
        assert_eq!(PiiType::IpAddress.to_string(), "ip_address");
        assert_eq!(PiiType::DateOfBirth.to_string(), "date_of_birth");
        assert_eq!(PiiType::PostalCode.to_string(), "postal_code");
        assert_eq!(PiiType::Name.to_string(), "name");
        assert_eq!(PiiType::Address.to_string(), "address");
        assert_eq!(PiiType::Custom("foo".to_string()).to_string(), "custom:foo");
    }

    #[test]
    fn test_data_classification_display() {
        assert_eq!(DataClassification::Public.to_string(), "Public");
        assert_eq!(DataClassification::Internal.to_string(), "Internal");
        assert_eq!(DataClassification::Confidential.to_string(), "Confidential");
        assert_eq!(DataClassification::Restricted.to_string(), "Restricted");
    }

    // --- Luhn extended tests ---

    #[test]
    fn test_luhn_valid_cards() {
        assert!(luhn_check("4111111111111111")); // Visa test
        assert!(luhn_check("5500000000000004")); // Mastercard test
        assert!(luhn_check("340000000000009"));  // Amex test (15 digits)
    }

    #[test]
    fn test_luhn_too_short() {
        assert!(!luhn_check("1234"));
        assert!(!luhn_check("123456789012")); // 12 digits
    }

    #[test]
    fn test_luhn_too_long() {
        assert!(!luhn_check("12345678901234567890")); // 20 digits
    }

    #[test]
    fn test_luhn_non_digit_filtering() {
        // Dashes and spaces should be filtered
        assert!(luhn_check("4111-1111-1111-1111"));
    }
}
