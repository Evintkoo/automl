use serde::{Serialize, Deserialize};
use std::collections::HashMap;
use std::sync::Arc;
use parking_lot::RwLock;
use chrono::{Utc, DateTime};
use jsonwebtoken::{encode, decode, Header, Validation, EncodingKey, DecodingKey};
use rand::Rng;

/// Constant-time byte comparison to prevent timing attacks
fn constant_time_eq(a: &[u8], b: &[u8]) -> bool {
    if a.len() != b.len() {
        return false;
    }
    let mut result = 0u8;
    for (x, y) in a.iter().zip(b.iter()) {
        result |= x ^ y;
    }
    result == 0
}

const MAX_AUDIT_LOG_ENTRIES: usize = 10_000;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SecurityConfig {
    pub api_keys: Vec<String>,
    pub jwt_secret: Option<String>,
    pub jwt_expiry_secs: u64,
    pub enable_api_key_auth: bool,
    pub enable_jwt_auth: bool,
    pub blocked_ips: Vec<String>,
    pub audit_log_enabled: bool,
}

impl Default for SecurityConfig {
    fn default() -> Self {
        Self {
            api_keys: Vec::new(),
            jwt_secret: None,
            jwt_expiry_secs: 3600,
            enable_api_key_auth: true,
            enable_jwt_auth: false,
            blocked_ips: Vec::new(),
            audit_log_enabled: true,
        }
    }
}

#[derive(Debug, Clone)]
pub struct ApiKeyVerifier {
    keys: Arc<RwLock<Vec<String>>>,
}

impl ApiKeyVerifier {
    pub fn new(keys: Vec<String>) -> Self {
        Self {
            keys: Arc::new(RwLock::new(keys)),
        }
    }

    pub fn verify(&self, key: &str) -> bool {
        let keys = self.keys.read();
        keys.iter().any(|k| constant_time_eq(k.as_bytes(), key.as_bytes()))
    }

    pub fn add_key(&self, key: &str) {
        self.keys.write().push(key.to_string());
    }

    pub fn remove_key(&self, key: &str) {
        self.keys.write().retain(|k| k != key);
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct JwtClaims {
    sub: String,
    exp: usize,
    iat: usize,
    #[serde(flatten)]
    extra: HashMap<String, String>,
}

#[derive(Debug, Clone)]
pub struct JwtVerifier {
    secret: String,
    expiry_secs: u64,
}

impl JwtVerifier {
    pub fn new(secret: String, expiry_secs: u64) -> Self {
        Self { secret, expiry_secs }
    }

    pub fn create_token(&self, claims: HashMap<String, String>) -> Result<String, String> {
        let now = Utc::now().timestamp() as usize;
        let jwt_claims = JwtClaims {
            sub: claims.get("sub").cloned().unwrap_or_default(),
            exp: now + self.expiry_secs as usize,
            iat: now,
            extra: claims,
        };
        encode(
            &Header::default(),
            &jwt_claims,
            &EncodingKey::from_secret(self.secret.as_bytes()),
        )
        .map_err(|e| format!("Failed to create token: {}", e))
    }

    pub fn verify_token(&self, token: &str) -> Result<HashMap<String, String>, String> {
        let token_data = decode::<JwtClaims>(
            token,
            &DecodingKey::from_secret(self.secret.as_bytes()),
            &Validation::default(),
        )
        .map_err(|e| format!("Failed to verify token: {}", e))?;

        let mut result = token_data.claims.extra;
        result.insert("sub".to_string(), token_data.claims.sub);
        Ok(result)
    }
}

/// Security status for compliance reporting
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SecurityStatus {
    pub api_key_auth_enabled: bool,
    pub jwt_auth_enabled: bool,
    pub audit_log_enabled: bool,
    pub blocked_ips_count: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuditEntry {
    pub timestamp: DateTime<Utc>,
    pub client_id: String,
    pub action: String,
    pub resource: String,
    pub status_code: u16,
    pub ip_address: String,
}

#[derive(Debug, Clone)]
pub struct SecurityManager {
    config: SecurityConfig,
    api_key_verifier: ApiKeyVerifier,
    jwt_verifier: Option<JwtVerifier>,
    audit_log: Arc<RwLock<Vec<AuditEntry>>>,
}

impl SecurityManager {
    pub fn new(config: SecurityConfig) -> Self {
        let api_key_verifier = ApiKeyVerifier::new(config.api_keys.clone());
        let jwt_verifier = config.jwt_secret.as_ref().map(|secret| {
            JwtVerifier::new(secret.clone(), config.jwt_expiry_secs)
        });
        Self {
            config,
            api_key_verifier,
            jwt_verifier,
            audit_log: Arc::new(RwLock::new(Vec::new())),
        }
    }

    pub fn verify_api_key(&self, key: &str) -> bool {
        if !self.config.enable_api_key_auth {
            return true;
        }
        self.api_key_verifier.verify(key)
    }

    pub fn verify_jwt(&self, token: &str) -> Result<HashMap<String, String>, String> {
        if !self.config.enable_jwt_auth {
            return Ok(HashMap::new());
        }
        match &self.jwt_verifier {
            Some(verifier) => verifier.verify_token(token),
            None => Err("JWT authentication not configured".to_string()),
        }
    }

    pub fn check_ip_blocked(&self, ip: &str) -> bool {
        self.config.blocked_ips.iter().any(|blocked| blocked == ip)
    }

    pub fn sanitize_input(input: &str) -> String {
        input
            .chars()
            .filter(|c| {
                c.is_alphanumeric()
                    || c.is_whitespace()
                    || matches!(c, '.' | '-' | '_' | '/' | ':' | '@' | ',')
            })
            .collect()
    }

    pub fn validate_input(input: &str) -> Result<(), String> {
        let patterns = [
            "SELECT ", "INSERT ", "UPDATE ", "DELETE ", "DROP ",
            "<script", "javascript:", "onclick=", "onerror=",
            "../", "..\\", "; rm ", "| rm ", "&& rm ",
        ];
        let lower = input.to_lowercase();
        for pattern in &patterns {
            if lower.contains(&pattern.to_lowercase()) {
                return Err(format!("Potentially dangerous pattern detected: {}", pattern));
            }
        }
        Ok(())
    }

    pub fn log_request(&self, entry: AuditEntry) {
        if self.config.audit_log_enabled {
            let mut log = self.audit_log.write();
            if log.len() >= MAX_AUDIT_LOG_ENTRIES {
                let drain_count = MAX_AUDIT_LOG_ENTRIES / 10;
                log.drain(..drain_count);
            }
            log.push(entry);
        }
    }

    pub fn get_audit_log(&self, limit: usize) -> Vec<AuditEntry> {
        let log = self.audit_log.read();
        log.iter().rev().take(limit).cloned().collect()
    }

    /// Get the most recent N audit entries (oldest first)
    pub fn get_recent_entries(&self, limit: usize) -> Vec<AuditEntry> {
        let log = self.audit_log.read();
        let start = log.len().saturating_sub(limit);
        log[start..].to_vec()
    }

    /// Get security status summary for compliance reporting
    pub fn get_status(&self) -> SecurityStatus {
        SecurityStatus {
            api_key_auth_enabled: self.config.enable_api_key_auth,
            jwt_auth_enabled: self.config.enable_jwt_auth,
            audit_log_enabled: self.config.audit_log_enabled,
            blocked_ips_count: self.config.blocked_ips.len(),
        }
    }

    pub fn generate_api_key() -> String {
        let mut rng = rand::thread_rng();
        let chars: Vec<char> = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
            .chars()
            .collect();
        (0..32).map(|_| chars[rng.gen_range(0..chars.len())]).collect()
    }

    pub fn get_security_headers() -> HashMap<String, String> {
        let mut headers = HashMap::new();
        headers.insert(
            "Strict-Transport-Security".to_string(),
            "max-age=31536000; includeSubDomains".to_string(),
        );
        headers.insert(
            "X-Content-Type-Options".to_string(),
            "nosniff".to_string(),
        );
        headers.insert(
            "X-Frame-Options".to_string(),
            "DENY".to_string(),
        );
        headers.insert(
            "X-XSS-Protection".to_string(),
            "1; mode=block".to_string(),
        );
        headers.insert(
            "Content-Security-Policy".to_string(),
            "default-src 'self'; style-src 'self' 'unsafe-inline'; script-src 'self' 'unsafe-inline' https://cdn.plot.ly; img-src 'self' data:; font-src 'self' data:".to_string(),
        );
        headers.insert(
            "Referrer-Policy".to_string(),
            "strict-origin-when-cross-origin".to_string(),
        );
        headers
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_manager() -> SecurityManager {
        SecurityManager::new(SecurityConfig::default())
    }

    #[test]
    fn test_csp_allows_inline_styles() {
        let headers = SecurityManager::get_security_headers();
        let csp = headers.get("Content-Security-Policy").expect("CSP header missing");
        assert!(csp.contains("style-src 'self' 'unsafe-inline'"),
            "CSP must allow inline styles for the web UI, got: {}", csp);
    }

    #[test]
    fn test_csp_allows_inline_scripts() {
        let headers = SecurityManager::get_security_headers();
        let csp = headers.get("Content-Security-Policy").expect("CSP header missing");
        assert!(csp.contains("script-src 'self' 'unsafe-inline'"),
            "CSP must allow inline scripts for the web UI, got: {}", csp);
    }

    #[test]
    fn test_csp_allows_data_uris_for_fonts() {
        let headers = SecurityManager::get_security_headers();
        let csp = headers.get("Content-Security-Policy").expect("CSP header missing");
        assert!(csp.contains("font-src 'self' data:"),
            "CSP must allow data: URIs for fonts, got: {}", csp);
    }

    #[test]
    fn test_csp_allows_data_uris_for_images() {
        let headers = SecurityManager::get_security_headers();
        let csp = headers.get("Content-Security-Policy").expect("CSP header missing");
        assert!(csp.contains("img-src 'self' data:"),
            "CSP must allow data: URIs for images, got: {}", csp);
    }

    #[test]
    fn test_all_required_security_headers_present() {
        let headers = SecurityManager::get_security_headers();
        let required = [
            "Strict-Transport-Security",
            "X-Content-Type-Options",
            "X-Frame-Options",
            "X-XSS-Protection",
            "Content-Security-Policy",
            "Referrer-Policy",
        ];
        for name in &required {
            assert!(headers.contains_key(*name), "Missing security header: {}", name);
        }
    }

    #[test]
    fn test_security_manager_create_and_verify_key() {
        let key = SecurityManager::generate_api_key();
        let mut config = SecurityConfig::default();
        config.api_keys.push(key.clone());
        let mgr = SecurityManager::new(config);
        assert!(mgr.verify_api_key(&key));
        assert!(!mgr.verify_api_key("wrong-key"));
    }

    #[test]
    fn test_input_sanitization_removes_script_tags() {
        let clean = SecurityManager::sanitize_input("<script>alert('xss')</script>hello");
        assert!(!clean.contains("<script>"));
        assert!(clean.contains("hello"));
    }

    #[test]
    fn test_input_validation_blocks_sql_injection() {
        let result = SecurityManager::validate_input("'; DROP TABLE users; --");
        assert!(result.is_err());
    }

    #[test]
    fn test_input_validation_allows_normal_input() {
        let result = SecurityManager::validate_input("normal data input 123");
        assert!(result.is_ok());
    }
}
