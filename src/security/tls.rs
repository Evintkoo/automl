use serde::{Serialize, Deserialize};
use std::collections::HashMap;
use std::fs;

/// Certificate expiry/validity status. Since this crate does not depend on an X.509
/// parsing library (e.g. `x509-parser`), we cannot cryptographically determine real
/// expiry from the certificate bytes. `Unknown` is the fail-closed result for that case:
/// callers MUST treat `Unknown` the same as `Expired`/invalid (never as healthy), rather
/// than assuming a certificate is fine just because it couldn't be checked.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ExpiryStatus {
    Valid,
    Expired,
    /// Could not be determined - no X.509 parser available. Treat as invalid/expired.
    Unknown,
}

impl ExpiryStatus {
    /// True unless the status is definitively `Valid`. Use this instead of matching only
    /// on `Expired`, so callers fail closed on `Unknown` too.
    pub fn should_treat_as_invalid(&self) -> bool {
        !matches!(self, ExpiryStatus::Valid)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CertificateInfo {
    pub subject: String,
    pub issuer: String,
    pub not_before: String,
    pub not_after: String,
    pub serial_number: String,
    /// Fail-closed status - see [`ExpiryStatus`]. `Unknown` must be treated as invalid.
    pub expiry_status: ExpiryStatus,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TlsConfig {
    pub cert_path: String,
    pub key_path: String,
    pub ca_path: Option<String>,
    pub min_tls_version: String,
    pub cipher_suites: Vec<String>,
    pub enable_hsts: bool,
}

impl Default for TlsConfig {
    fn default() -> Self {
        Self {
            cert_path: String::new(),
            key_path: String::new(),
            ca_path: None,
            min_tls_version: "1.2".to_string(),
            cipher_suites: Self::default_cipher_suites(),
            enable_hsts: true,
        }
    }
}

impl TlsConfig {
    fn default_cipher_suites() -> Vec<String> {
        vec![
            "TLS_AES_256_GCM_SHA384".to_string(),
            "TLS_AES_128_GCM_SHA256".to_string(),
            "TLS_CHACHA20_POLY1305_SHA256".to_string(),
        ]
    }
}

#[derive(Debug, Clone)]
pub struct TlsManager {
    config: TlsConfig,
}

impl TlsManager {
    pub fn new(config: TlsConfig) -> Self {
        Self { config }
    }

    /// Validates that a certificate file exists and looks like PEM-encoded data.
    ///
    /// NOTE: This crate has no X.509 parsing dependency (e.g. `x509-parser`), so this
    /// cannot actually parse the certificate's subject/issuer/validity window. Those
    /// fields are therefore reported as unknown and `expiry_status` is always
    /// `ExpiryStatus::Unknown` on success, which callers MUST treat as
    /// invalid/unverified (fail closed) - never as "certificate is healthy". A real
    /// implementation needs a proper X.509 parsing crate as a follow-up.
    pub fn validate_certificate(&self, cert_path: &str) -> Result<CertificateInfo, String> {
        let content = fs::read_to_string(cert_path)
            .map_err(|e| format!("Failed to read certificate: {}", e))?;

        if !content.contains("-----BEGIN CERTIFICATE-----") {
            return Err("Invalid certificate format: missing PEM header".to_string());
        }

        // Basic PEM presence check only — no real X.509 parsing is performed, so we
        // cannot claim to know the certificate's actual validity. Fail closed.
        Ok(CertificateInfo {
            subject: "N/A (requires x509-parser for full parsing)".to_string(),
            issuer: "N/A (requires x509-parser for full parsing)".to_string(),
            not_before: "N/A (requires x509-parser for full parsing)".to_string(),
            not_after: "N/A (requires x509-parser for full parsing)".to_string(),
            serial_number: "N/A".to_string(),
            expiry_status: ExpiryStatus::Unknown,
        })
    }

    /// Returns days until certificate expiry. Requires a real X.509 parser to compute;
    /// since we don't have one, this fails closed with an explicit error rather than
    /// fabricating a "365 days remaining" answer. Callers MUST treat an `Err` here as
    /// "cannot verify -> treat certificate as invalid/expired", not as healthy.
    pub fn check_expiry(&self, cert_path: &str) -> Result<i64, String> {
        let info = self.validate_certificate(cert_path)?;
        match info.expiry_status {
            ExpiryStatus::Valid => Ok(365),
            ExpiryStatus::Expired | ExpiryStatus::Unknown => Err(
                "Cannot verify certificate expiry: no X.509 parser available (requires \
                 x509-parser crate). Treating certificate as invalid/expired (fail closed)."
                    .to_string(),
            ),
        }
    }

    pub fn get_security_headers(&self) -> HashMap<String, String> {
        let mut headers = HashMap::new();
        if self.config.enable_hsts {
            headers.insert(
                "Strict-Transport-Security".to_string(),
                "max-age=31536000; includeSubDomains; preload".to_string(),
            );
        }
        headers.insert(
            "X-Content-Type-Options".to_string(),
            "nosniff".to_string(),
        );
        headers.insert(
            "X-Frame-Options".to_string(),
            "DENY".to_string(),
        );
        headers
    }

    pub fn get_recommended_cipher_suites(&self) -> Vec<String> {
        vec![
            "TLS_AES_256_GCM_SHA384".to_string(),
            "TLS_AES_128_GCM_SHA256".to_string(),
            "TLS_CHACHA20_POLY1305_SHA256".to_string(),
            "ECDHE-ECDSA-AES256-GCM-SHA384".to_string(),
            "ECDHE-RSA-AES256-GCM-SHA384".to_string(),
            "ECDHE-ECDSA-AES128-GCM-SHA256".to_string(),
        ]
    }

    pub fn get_secure_tls_config(&self) -> HashMap<String, String> {
        let mut config = HashMap::new();
        config.insert("min_version".to_string(), self.config.min_tls_version.clone());
        config.insert(
            "cipher_suites".to_string(),
            self.config.cipher_suites.join(","),
        );
        config.insert("session_tickets".to_string(), "false".to_string());
        config.insert("compression".to_string(), "false".to_string());
        config.insert("renegotiation".to_string(), "false".to_string());
        config
    }
}
