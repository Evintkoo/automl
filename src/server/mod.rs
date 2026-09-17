//! AutoML Server Module
//!
//! High-performance web server for the AutoML platform.
//! Provides REST API and web UI for data preprocessing, model training,
//! and inference.

mod api;
mod error;
pub mod import;
pub mod state;
mod handlers;

pub use api::create_router;
pub use error::ServerError;
pub use state::AppState;

use std::net::SocketAddr;
use std::sync::Arc;
use tracing::{info, warn};

/// Server configuration
#[derive(Debug, Clone)]
pub struct ServerConfig {
    pub host: String,
    pub port: u16,
    pub static_dir: Option<String>,
    pub data_dir: String,
    pub models_dir: String,
    pub max_upload_size: usize,
}

impl Default for ServerConfig {
    fn default() -> Self {
        Self {
            host: std::env::var("API_HOST").unwrap_or_else(|_| "127.0.0.1".to_string()),
            port: std::env::var("API_PORT")
                .ok()
                .and_then(|p| p.parse().ok())
                .unwrap_or(8080),
            static_dir: Some(Self::resolve_static_dir()),
            data_dir: std::env::var("DATA_DIR").unwrap_or_else(|_| "./data".to_string()),
            models_dir: std::env::var("MODELS_DIR").unwrap_or_else(|_| "./models".to_string()),
            max_upload_size: std::env::var("MAX_UPLOAD_SIZE")
                .ok()
                .and_then(|s| s.parse().ok())
                .unwrap_or(100 * 1024 * 1024), // 100MB
        }
    }
}

impl ServerConfig {
    /// Resolve the static directory by checking multiple candidate paths.
    fn resolve_static_dir() -> String {
        if let Ok(dir) = std::env::var("STATIC_DIR") {
            if std::path::Path::new(&dir).exists() {
                return dir;
            }
        }

        let candidates = [
            "automl-web/static".to_string(),
            format!("{}/automl-web/static", env!("CARGO_MANIFEST_DIR")),
        ];

        for candidate in &candidates {
            if std::path::Path::new(candidate).exists() {
                return candidate.clone();
            }
        }

        // Return default relative path as last resort
        "automl-web/static".to_string()
    }
}

/// Start the server with the given configuration
pub async fn run_server(config: ServerConfig) -> anyhow::Result<()> {
    let start_time = chrono::Utc::now();
    info!(
        data_dir = %config.data_dir,
        models_dir = %config.models_dir,
        started_at = %start_time.to_rfc3339(),
        "Initializing server directories"
    );

    std::fs::create_dir_all(&config.data_dir)?;
    std::fs::create_dir_all(&config.models_dir)?;

    if let Some(ref static_dir) = config.static_dir {
        if !std::path::Path::new(static_dir).exists() {
            warn!(static_dir = %static_dir, "Static directory not found, web UI will be unavailable");
        }
    }

    let state = Arc::new(AppState::new(config.clone()));

    // Restore previously-trained models from disk so they survive server restarts
    crate::server::state::restore_models_from_disk(&config.models_dir, &state.models);

    // Periodically evict stale jobs/train-engines/insights/studies so long-running
    // servers don't grow memory unbounded. `evict_stale_entries()` previously existed
    // but was never invoked anywhere; wire it up as a background task for the lifetime
    // of the server.
    {
        let eviction_state = Arc::clone(&state);
        tokio::spawn(async move {
            let mut interval = tokio::time::interval(std::time::Duration::from_secs(5 * 60));
            // The first tick fires immediately; skip it so we don't evict right at startup.
            interval.tick().await;
            loop {
                interval.tick().await;
                eviction_state.evict_stale_entries().await;
            }
        });
    }

    let app = create_router(state, &config);

    let addr: SocketAddr = format!("{}:{}", config.host, config.port).parse()?;
    info!(
        host = %config.host,
        port = config.port,
        address = %addr,
        max_upload_size_mb = config.max_upload_size / 1024 / 1024,
        started_at = %start_time.to_rfc3339(),
        "AutoML Server starting"
    );
    info!(url = %format!("http://{}", addr), "Web UI available");
    info!(url = %format!("http://{}/api", addr), "REST API available");
    info!(url = %format!("http://{}/api/health", addr), "Health endpoint available");

    let listener = tokio::net::TcpListener::bind(addr).await?;
    info!(address = %addr, pid = std::process::id(), "Server listening and ready to accept connections");

    // Graceful shutdown on ctrl+c
    let start_time_for_shutdown = start_time;
    let shutdown_signal = async move {
        tokio::signal::ctrl_c()
            .await
            .expect("Failed to install CTRL+C signal handler");
        let stop_time = chrono::Utc::now();
        let uptime = stop_time.signed_duration_since(start_time_for_shutdown);
        info!(
            stopped_at = %stop_time.to_rfc3339(),
            uptime_secs = uptime.num_seconds(),
            "Shutdown signal received, stopping server gracefully"
        );

        // Handle second ctrl+c (force exit) or timeout
        tokio::spawn(async {
            tokio::select! {
                _ = tokio::signal::ctrl_c() => {
                    eprintln!("\nForce shutdown requested, exiting immediately");
                    std::process::exit(1);
                }
                _ = tokio::time::sleep(std::time::Duration::from_secs(10)) => {
                    eprintln!("Graceful shutdown timed out after 10s, forcing exit");
                    std::process::exit(1);
                }
            }
        });
    };

    info!("Server started successfully (press ctrl+c to stop)");
    axum::serve(listener, app)
        .with_graceful_shutdown(shutdown_signal)
        .await?;

    info!("Server shut down cleanly");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = ServerConfig::default();
        assert_eq!(config.port, 8080);
        assert_eq!(config.max_upload_size, 100 * 1024 * 1024);
    }
}
