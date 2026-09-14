//! API route definitions

use std::sync::Arc;
use axum::{
    extract::DefaultBodyLimit,
    http::StatusCode,
    middleware as axum_middleware,
    response::IntoResponse,
    routing::{get, post},
    Json, Router,
};
use serde_json::json;
use tower_http::{
    compression::CompressionLayer,
    cors::{Any, CorsLayer},
    services::ServeDir,
    trace::TraceLayer,
};

use crate::security::middleware::{SecurityMiddleware, security_layer};

use super::{handlers, state::AppState, ServerConfig};

async fn handle_404() -> impl IntoResponse {
    (
        StatusCode::NOT_FOUND,
        Json(json!({
            "error": true,
            "message": "Not found. Visit / for the web UI or /api/health to check API status.",
        })),
    )
}

async fn handle_405() -> impl IntoResponse {
    (
        StatusCode::METHOD_NOT_ALLOWED,
        Json(json!({
            "error": true,
            "message": "Method not allowed. Check the API documentation for supported methods.",
        })),
    )
}

/// Create the main application router
pub fn create_router(state: Arc<AppState>, config: &ServerConfig) -> Router {
    // API routes
    let api_routes = Router::new()
        // Data endpoints
        .route("/data/upload", post(handlers::upload_data))
        .route("/data/preview", get(handlers::get_data_preview))
        .route("/data/info", get(handlers::get_data_info))
        .route("/data/sample/:name", get(handlers::load_sample_data))
        .route("/data/clean", post(handlers::auto_clean_data))
        .route("/data/clear", axum::routing::delete(handlers::clear_data))
        .route("/data/analyze", get(handlers::analyze_data))
        .route("/data/import/kaggle", post(handlers::import_kaggle))
        // Preprocessing
        .route("/preprocess", post(handlers::run_preprocessing))
        .route("/preprocess/config", get(handlers::get_preprocessing_config))
        // Training
        .route("/train", post(handlers::start_training))
        .route("/train/status/:job_id", get(handlers::get_training_status))
        .route("/train/compare", post(handlers::compare_models))
        // Models
        .route("/models", get(handlers::list_models))
        .route("/models/:model_id", get(handlers::get_model).delete(handlers::delete_model))
        .route("/models/:model_id/download", get(handlers::download_model))
        // Inference
        .route("/predict", post(handlers::predict))
        .route("/predict/batch", post(handlers::predict_batch))
        .route("/predict/proba", post(handlers::predict_proba))
        .route("/predict/stats/:model_id", get(handlers::get_inference_stats))
        .route("/predict/cache/:model_id", axum::routing::delete(handlers::evict_model_cache))
        // Cache management
        .route("/cache/stats", get(handlers::get_cache_stats))
        .route("/cache", axum::routing::delete(handlers::clear_prediction_cache))
        // System
        .route("/system/status", get(handlers::get_system_status))
        .route("/health", get(handlers::health_check))
        // Batch processor
        .route("/batch/config", post(handlers::configure_batch_processor))
        .route("/batch/stats", get(handlers::get_batch_stats))
        // Quantizer
        .route("/quantize", post(handlers::quantize_data))
        .route("/dequantize", post(handlers::dequantize_data))
        .route("/quantize/calibrate", post(handlers::calibrate_quantizer))
        // Device optimizer
        .route("/device/info", get(handlers::get_device_info))
        .route("/device/optimal-configs", get(handlers::get_optimal_configs))
        // Model management
        .route("/models/:model_id/metrics", get(handlers::get_model_metrics))
        // Monitoring
        .route("/monitoring/dashboard", get(handlers::get_monitoring_dashboard))
        .route("/monitoring/alerts", get(handlers::get_alerts))
        .route("/monitoring/performance", get(handlers::get_performance_analysis))
        // Hyperparameter optimization
        .route("/hyperopt", post(handlers::run_hyperopt))
        .route("/hyperopt/config", get(handlers::get_hyperopt_config))
        // Explainability
        .route("/explain/importance", post(handlers::get_feature_importance))
        .route("/explain/local", post(handlers::get_local_explanations))
        // Ensemble
        .route("/ensemble/train", post(handlers::train_ensemble))
        // Drift detection
        .route("/drift/detect", post(handlers::detect_drift))
        // URL import
        .route("/data/import/url", post(handlers::import_from_url))
        // Auto-tune
        .route("/autotune", post(handlers::auto_tune))
        // Enhanced hyperopt
        .route("/hyperopt/search-space/:model", get(handlers::get_search_space))
        .route("/hyperopt/history", get(handlers::get_hyperopt_history))
        .route("/hyperopt/apply", post(handlers::apply_best_params))
        // Model export
        .route("/export/model", post(handlers::export_model))
        // Anomaly detection
        .route("/anomaly/detect", post(handlers::detect_anomalies))
        // Clustering
        .route("/clustering/run", post(handlers::run_clustering))
        // Visualization
        .route("/visualization/umap", post(handlers::generate_umap))
        .route("/visualization/pca", post(handlers::generate_pca))
        .route("/visualization/umap/stream", get(handlers::umap_stream_handler))
        // AutoML pipeline
        .route("/automl/run", post(handlers::run_automl_pipeline))
        .route("/automl/status/:job_id", get(handlers::get_automl_status))
        // Security management
        .route("/security/status", get(handlers::get_security_status))
        .route("/security/audit-log", get(handlers::get_audit_log))
        .route("/security/rate-limit/stats", get(handlers::get_rate_limit_stats))
        // === ISO Compliance Endpoints ===
        // Data provenance (ISO 5259, 5338)
        .route("/data/lineage", get(handlers::get_data_lineage))
        .route("/data/quality", get(handlers::get_data_quality))
        .route("/data/datasheet", get(handlers::get_data_datasheet))
        // Fairness (ISO TR 24027)
        .route("/fairness/evaluate", post(handlers::evaluate_fairness))
        .route("/fairness/bias-scan", post(handlers::bias_scan))
        // Model cards (ISO TR 24028)
        .route("/models/:model_id/card", get(handlers::get_model_card))
        // Audit trail (ISO 27001, TR 24028)
        .route("/audit/events", get(handlers::get_audit_events))
        .route("/audit/integrity", get(handlers::verify_audit_integrity))
        // Privacy (ISO 27701)
        .route("/privacy/scan", post(handlers::scan_pii))
        .route("/privacy/retention", get(handlers::get_retention_status))
        // Compliance (ISO 42001)
        .route("/compliance/report", get(handlers::get_compliance_report))
        // SLO monitoring (ISO 25010)
        .route("/monitoring/slo", get(handlers::get_slo_status))
        // Feedback loop (ISO 42001, 5338)
        .route("/predict/feedback", post(handlers::submit_prediction_feedback))
        // Insights
        .route("/insights/model-structure", get(handlers::get_insights_model_structure))
        .route("/insights/evaluation", get(handlers::get_insights_evaluation))
        .route("/quality/report/:model_id", get(handlers::get_quality_report))
        .fallback(handle_404)
        .method_not_allowed_fallback(handle_405);

    // Build security middleware
    let security_middleware = SecurityMiddleware::new(
        Arc::clone(&state.security_manager),
        Arc::clone(&state.rate_limiter),
    );

    // Build main router
    let mut app = Router::new()
        .nest("/api", api_routes)
        .route("/", get(handlers::serve_index))
        .fallback(handle_404)
        .method_not_allowed_fallback(handle_405)
        .with_state(state)
        .layer(axum_middleware::from_fn_with_state(
            security_middleware,
            security_layer,
        ));

    // Serve static files if directory exists
    if let Some(ref static_dir) = config.static_dir {
        let static_path = std::path::Path::new(static_dir);
        if static_path.exists() {
            app = app.nest_service("/static", ServeDir::new(static_path));
        } else {
            // Fallback: try CARGO_MANIFEST_DIR-relative path
            let manifest_path = format!("{}/automl-web/static", env!("CARGO_MANIFEST_DIR"));
            let fallback_path = std::path::Path::new(&manifest_path);
            if fallback_path.exists() {
                app = app.nest_service("/static", ServeDir::new(fallback_path));
            }
        }
    }

    // Add middleware — CORS configured via CORS_ORIGIN env var.
    // Default: allow only localhost origins (safe for local-first use).
    // Set CORS_ORIGIN=* to allow all origins in development.
    let cors = match std::env::var("CORS_ORIGIN") {
        Ok(origin) if origin == "*" => {
            CorsLayer::new()
                .allow_origin(Any)
                .allow_methods(Any)
                .allow_headers(Any)
        }
        Ok(origin) if !origin.is_empty() => {
            let header_val = origin.parse::<axum::http::HeaderValue>().unwrap_or_else(|e| {
                tracing::warn!(origin = %origin, error = %e, "Invalid CORS_ORIGIN, falling back to localhost");
                axum::http::HeaderValue::from_static("http://localhost:8080")
            });
            CorsLayer::new()
                .allow_origin(header_val)
                .allow_methods(Any)
                .allow_headers(Any)
        }
        _ => {
            // Default: only allow same-origin (localhost)
            CorsLayer::new()
                .allow_origin(
                    [
                        "http://localhost:8080".parse().unwrap(),
                        "http://127.0.0.1:8080".parse().unwrap(),
                    ]
                )
                .allow_methods(Any)
                .allow_headers(Any)
        }
    };

    app.layer(DefaultBodyLimit::max(config.max_upload_size))
        .layer(CompressionLayer::new())
        .layer(cors)
        .layer(TraceLayer::new_for_http())
}
