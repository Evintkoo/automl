//! Integration test: Security module

use automl::security::{
    SecurityManager, SecurityConfig, ApiKeyVerifier, RateLimiter, RateLimitConfig,
    RateLimitAlgorithm, SecretsManager, SecretType, TlsManager, TlsConfig,
};

#[test]
fn test_security_manager_api_key() {
    let config = SecurityConfig {
        api_keys: vec!["test-key-123".to_string(), "test-key-456".to_string()],
        jwt_secret: Some("supersecretjwt".to_string()),
        jwt_expiry_secs: 3600,
        enable_api_key_auth: true,
        enable_jwt_auth: true,
        blocked_ips: vec!["192.168.1.100".to_string()],
        audit_log_enabled: true,
    };
    let manager = SecurityManager::new(config);

    assert!(manager.verify_api_key("test-key-123"));
    assert!(manager.verify_api_key("test-key-456"));
    assert!(!manager.verify_api_key("invalid-key"));
}

#[test]
fn test_security_manager_ip_blocking() {
    let config = SecurityConfig {
        api_keys: vec![],
        jwt_secret: None,
        jwt_expiry_secs: 3600,
        enable_api_key_auth: false,
        enable_jwt_auth: false,
        blocked_ips: vec!["10.0.0.1".to_string(), "192.168.1.100".to_string()],
        audit_log_enabled: false,
    };
    let manager = SecurityManager::new(config);

    assert!(manager.check_ip_blocked("10.0.0.1"));
    assert!(manager.check_ip_blocked("192.168.1.100"));
    assert!(!manager.check_ip_blocked("10.0.0.2"));
}

#[test]
fn test_security_manager_input_sanitization() {
    let sanitized = SecurityManager::sanitize_input("<script>alert('xss')</script>");
    assert!(!sanitized.contains("<script>"), "should strip script tags");
}

#[test]
fn test_security_manager_input_validation() {
    assert!(SecurityManager::validate_input("normal text").is_ok());
}

#[test]
fn test_security_headers() {
    let headers = SecurityManager::get_security_headers();
    assert!(headers.contains_key("X-Content-Type-Options"));
    assert!(headers.contains_key("X-Frame-Options"));
}

#[test]
fn test_generate_api_key() {
    let key = SecurityManager::generate_api_key();
    assert!(key.len() >= 32, "API key should be at least 32 chars");
}

#[test]
fn test_api_key_verifier() {
    let mut verifier = ApiKeyVerifier::new(vec!["key1".to_string()]);
    assert!(verifier.verify("key1"));
    assert!(!verifier.verify("key2"));

    verifier.add_key("key2");
    assert!(verifier.verify("key2"));

    verifier.remove_key("key1");
    assert!(!verifier.verify("key1"));
}

#[test]
fn test_rate_limiter_fixed_window() {
    let config = RateLimitConfig {
        requests_per_window: 3,
        window_seconds: 60,
        algorithm: RateLimitAlgorithm::FixedWindow,
        burst_size: 3,
    };
    let limiter = RateLimiter::new(config);

    assert!(limiter.is_allowed("client1"));
    assert!(limiter.is_allowed("client1"));
    assert!(limiter.is_allowed("client1"));
    assert!(!limiter.is_allowed("client1"), "4th request should be blocked");

    // Different client should still be allowed
    assert!(limiter.is_allowed("client2"));
}

#[test]
fn test_rate_limiter_remaining() {
    let config = RateLimitConfig {
        requests_per_window: 5,
        window_seconds: 60,
        algorithm: RateLimitAlgorithm::FixedWindow,
        burst_size: 5,
    };
    let limiter = RateLimiter::new(config);

    limiter.is_allowed("client1");
    limiter.is_allowed("client1");

    let remaining = limiter.get_remaining("client1");
    assert_eq!(remaining, 3);
}

#[test]
fn test_rate_limiter_reset() {
    let config = RateLimitConfig {
        requests_per_window: 2,
        window_seconds: 60,
        algorithm: RateLimitAlgorithm::FixedWindow,
        burst_size: 2,
    };
    let limiter = RateLimiter::new(config);

    limiter.is_allowed("client1");
    limiter.is_allowed("client1");
    assert!(!limiter.is_allowed("client1"));

    limiter.reset_client("client1");
    assert!(limiter.is_allowed("client1"), "should be allowed after reset");
}

#[test]
fn test_secrets_manager() {
    let tmp_dir = std::env::temp_dir().join(format!("automl_test_secrets_{}", std::process::id()));
    std::fs::create_dir_all(&tmp_dir).ok();
    let path = tmp_dir.join(".secrets");
    let manager = SecretsManager::new(path.to_str().unwrap());

    // Store a secret
    let result = manager.store_secret("api-key-1", "my-secret-key", SecretType::ApiKey);
    assert!(result.is_ok(), "storing secret should succeed");

    // Retrieve it
    let secret = manager.get_secret("api-key-1");
    assert!(secret.is_ok());
    assert_eq!(secret.unwrap(), "my-secret-key");

    // List secrets
    let list = manager.list_secrets();
    assert_eq!(list.len(), 1);

    // Delete it
    let deleted = manager.delete_secret("api-key-1");
    assert!(deleted.is_ok());

    let list = manager.list_secrets();
    assert_eq!(list.len(), 0);

    // Cleanup
    std::fs::remove_dir_all(&tmp_dir).ok();
}

#[test]
fn test_secrets_generate() {
    let tmp_dir = std::env::temp_dir().join(format!("automl_test_gen_{}", std::process::id()));
    std::fs::create_dir_all(&tmp_dir).ok();
    let path = tmp_dir.join(".secrets");
    let manager = SecretsManager::new(path.to_str().unwrap());

    let key = manager.generate_secret(SecretType::ApiKey, 32);
    assert_eq!(key.len(), 32);

    let password = manager.generate_secret(SecretType::Password, 16);
    assert_eq!(password.len(), 16);

    std::fs::remove_dir_all(&tmp_dir).ok();
}

#[test]
fn test_secrets_strength_assessment() {
    let tmp_dir = std::env::temp_dir().join(format!("automl_test_str_{}", std::process::id()));
    std::fs::create_dir_all(&tmp_dir).ok();
    let path = tmp_dir.join(".secrets");
    let manager = SecretsManager::new(path.to_str().unwrap());

    let weak = manager.assess_strength("abc");
    let strong = manager.assess_strength("Xy9$kL2m#Pw!qR5nTz@8");
    assert!(strong > weak, "strong password should score higher than weak");

    std::fs::remove_dir_all(&tmp_dir).ok();
}

#[test]
fn test_tls_security_headers() {
    let config = TlsConfig {
        cert_path: String::new(),
        key_path: String::new(),
        ca_path: None,
        min_tls_version: "1.2".to_string(),
        cipher_suites: vec![],
        enable_hsts: true,
    };
    let manager = TlsManager::new(config);

    let headers = manager.get_security_headers();
    assert!(headers.contains_key("Strict-Transport-Security"));
}

#[test]
fn test_tls_cipher_suites() {
    let config = TlsConfig {
        cert_path: String::new(),
        key_path: String::new(),
        ca_path: None,
        min_tls_version: "1.2".to_string(),
        cipher_suites: vec![],
        enable_hsts: true,
    };
    let manager = TlsManager::new(config);

    let suites = manager.get_recommended_cipher_suites();
    assert!(!suites.is_empty(), "should recommend cipher suites");
}
