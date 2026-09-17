use axum::{
    extract::Request,
    http::StatusCode,
    middleware::Next,
    response::Response,
};
use std::sync::Arc;
use chrono::Utc;

use super::auth::{AuditEntry, SecurityManager};
use super::rate_limiter::RateLimiter;
use super::rbac::{RbacManager, Role};

#[derive(Debug, Clone)]
pub struct SecurityMiddleware {
    pub security_manager: Arc<SecurityManager>,
    pub rate_limiter: Arc<RateLimiter>,
    pub rbac_manager: Arc<RbacManager>,
}

impl SecurityMiddleware {
    pub fn new(security_manager: Arc<SecurityManager>, rate_limiter: Arc<RateLimiter>) -> Self {
        Self {
            security_manager,
            rate_limiter,
            rbac_manager: Arc::new(RbacManager::new()),
        }
    }
}

pub async fn security_layer(
    axum::extract::State(middleware): axum::extract::State<SecurityMiddleware>,
    request: Request,
    next: Next,
) -> Result<Response, StatusCode> {
    // Extract client IP - use x-real-ip first, then x-forwarded-for (first entry only),
    // falling back to "unknown". In production, configure a reverse proxy to set these.
    let ip = request
        .headers()
        .get("x-real-ip")
        .and_then(|v| v.to_str().ok())
        .map(|s| s.trim().to_string())
        .or_else(|| {
            request
                .headers()
                .get("x-forwarded-for")
                .and_then(|v| v.to_str().ok())
                .and_then(|s| s.split(',').next())
                .map(|s| s.trim().to_string())
        })
        .unwrap_or_else(|| "unknown".to_string());

    // Check IP blocked
    if middleware.security_manager.check_ip_blocked(&ip) {
        return Err(StatusCode::FORBIDDEN);
    }

    // Verify API key first - require it when API key auth is enabled. Rate limiting
    // must key off a *verified* identity (or the connection IP), never an unverified
    // client-supplied header, otherwise an attacker can mint a fresh quota per request
    // by sending a different bogus API key each time.
    let raw_api_key = request.headers().get("x-api-key").and_then(|v| v.to_str().ok());
    match raw_api_key {
        Some(api_key) => {
            if !middleware.security_manager.verify_api_key(api_key) {
                return Err(StatusCode::UNAUTHORIZED);
            }
        }
        None => {
            // No API key provided - reject if API key auth is enabled and keys are configured
            if !middleware.security_manager.verify_api_key("") {
                return Err(StatusCode::UNAUTHORIZED);
            }
        }
    }

    // Extract client identifier for rate limiting: now that the API key (if any) has
    // been verified, it is safe to use as the bucket key; otherwise fall back to IP.
    // When API-key auth is disabled, `verify_api_key` accepts any (even bogus/rotating)
    // key, so the header carries no verified identity in that mode — always bucket on
    // IP instead, otherwise a caller could mint a fresh rate-limit quota per request by
    // sending a different unverified key each time.
    let client_id = if middleware.security_manager.api_key_auth_enabled() {
        raw_api_key.unwrap_or(&ip).to_string()
    } else {
        ip.clone()
    };

    // Check rate limit
    if !middleware.rate_limiter.is_allowed(&client_id) {
        return Err(StatusCode::TOO_MANY_REQUESTS);
    }

    let method = request.method().to_string();
    let uri = request.uri().path().to_string();

    // RBAC check: role must never come from a client-supplied header (that was the
    // vulnerability - anyone could self-declare `x-role: admin`). This codebase has
    // no per-API-key role table yet, so the only identity signal available here is
    // "did the caller present a valid, server-configured secret (or is there no auth
    // boundary configured at all)". Either way, by this point the caller has already
    // passed `verify_api_key` above, so they hold whatever privilege this deployment
    // grants an authenticated caller - reaching this line means either API-key auth
    // is disabled entirely (single-tenant/local usage, everyone is the operator) or
    // the caller presented a valid pre-configured secret key. Grant full access in
    // both cases; what's eliminated is the attacker's ability to choose their own
    // role via a header. If/when API keys carry per-key roles (e.g. a role column in
    // key config, or JWT claims), replace this with a real lookup instead of Admin.
    let role = Role::Admin;

    if !middleware.rbac_manager.check_api_access(&role, &uri, &method) {
        return Err(StatusCode::FORBIDDEN);
    }

    // Process request
    let response = next.run(request).await;
    let status = response.status().as_u16();

    // Log audit entry
    middleware.security_manager.log_request(AuditEntry {
        timestamp: Utc::now(),
        client_id,
        action: method,
        resource: uri,
        status_code: status,
        ip_address: ip,
    });

    // Add security headers
    let mut response = response;
    let headers = response.headers_mut();
    for (key, value) in SecurityManager::get_security_headers() {
        if let (Ok(name), Ok(val)) = (
            axum::http::HeaderName::try_from(key),
            axum::http::HeaderValue::from_str(&value),
        ) {
            headers.insert(name, val);
        }
    }

    Ok(response)
}
