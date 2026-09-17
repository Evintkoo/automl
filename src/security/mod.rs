// Security module - authentication, rate limiting, TLS, secrets management, RBAC, audit
pub mod auth;
pub mod rate_limiter;
pub mod tls;
pub mod secrets;
pub mod middleware;
pub mod rbac;
pub mod audit_trail;

pub use auth::{SecurityManager, SecurityConfig, SecurityStatus, ApiKeyVerifier, JwtVerifier};
pub use rate_limiter::{RateLimiter, RateLimitConfig, RateLimitAlgorithm};
pub use tls::{TlsManager, TlsConfig, CertificateInfo, ExpiryStatus};
pub use secrets::{SecretsManager, SecretType, SecretMetadata};
pub use middleware::SecurityMiddleware;
pub use rbac::{RbacManager, Role, Resource, Action, Permission};
pub use audit_trail::{AuditTrail, AuditTrailEntry, AuditEventType, AuditIntegrityResult};
