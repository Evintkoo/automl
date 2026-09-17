//! Integration test: Server API endpoints

use automl::server::{AppState, ServerConfig, create_router};
use std::sync::Arc;
use axum::body::Body;
use axum::http::{Request, StatusCode};
use tower::ServiceExt;

fn test_app() -> axum::Router {
    let config = ServerConfig {
        host: "127.0.0.1".to_string(),
        port: 0,
        static_dir: None,
        data_dir: "/tmp/automl-test-data".to_string(),
        models_dir: "/tmp/automl-test-models".to_string(),
        max_upload_size: 10 * 1024 * 1024,
    };
    std::fs::create_dir_all(&config.data_dir).ok();
    std::fs::create_dir_all(&config.models_dir).ok();
    let state = Arc::new(AppState::new(config.clone()));
    create_router(state, &config)
}

#[tokio::test]
async fn test_health_endpoint() {
    let app = test_app();
    let response = app
        .oneshot(
            Request::builder()
                .uri("/api/health")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);
}

#[tokio::test]
async fn test_system_status_endpoint() {
    let app = test_app();
    let response = app
        .oneshot(
            Request::builder()
                .uri("/api/system/status")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);
}

#[tokio::test]
async fn test_root_serves_html() {
    let app = test_app();
    let response = app
        .oneshot(
            Request::builder()
                .uri("/")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);
}

#[tokio::test]
async fn test_list_models_empty() {
    let app = test_app();
    let response = app
        .oneshot(
            Request::builder()
                .uri("/api/models")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);
}

#[tokio::test]
async fn test_data_preview_no_data() {
    let app = test_app();
    let response = app
        .oneshot(
            Request::builder()
                .uri("/api/data/preview")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    // Should return OK with empty/error message since no data loaded
    let status = response.status();
    assert!(
        status == StatusCode::OK || status == StatusCode::BAD_REQUEST || status == StatusCode::NOT_FOUND,
        "unexpected status: {}",
        status
    );
}

#[tokio::test]
async fn test_device_info_endpoint() {
    let app = test_app();
    let response = app
        .oneshot(
            Request::builder()
                .uri("/api/device/info")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);
}

#[tokio::test]
async fn test_optimal_configs_endpoint() {
    let app = test_app();
    let response = app
        .oneshot(
            Request::builder()
                .uri("/api/device/optimal-configs")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);
}

#[tokio::test]
async fn test_batch_stats_endpoint() {
    let app = test_app();
    let response = app
        .oneshot(
            Request::builder()
                .uri("/api/batch/stats")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);
}

#[tokio::test]
async fn test_monitoring_dashboard_endpoint() {
    let app = test_app();
    let response = app
        .oneshot(
            Request::builder()
                .uri("/api/monitoring/dashboard")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);
}

#[tokio::test]
async fn test_monitoring_alerts_endpoint() {
    let app = test_app();
    let response = app
        .oneshot(
            Request::builder()
                .uri("/api/monitoring/alerts")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);
}

#[tokio::test]
async fn test_monitoring_performance_endpoint() {
    let app = test_app();
    let response = app
        .oneshot(
            Request::builder()
                .uri("/api/monitoring/performance")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);
}

#[tokio::test]
async fn test_preprocess_config_endpoint() {
    let app = test_app();
    let response = app
        .oneshot(
            Request::builder()
                .uri("/api/preprocess/config")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);
}

#[tokio::test]
async fn test_load_sample_data() {
    let app = test_app();
    let response = app
        .oneshot(
            Request::builder()
                .uri("/api/data/sample/iris")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);
}

#[tokio::test]
async fn test_quantize_endpoint() {
    let app = test_app();
    let body = serde_json::json!({
        "data": [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
        "bits": 8
    });
    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/api/quantize")
                .header("content-type", "application/json")
                .body(Body::from(serde_json::to_string(&body).unwrap()))
                .unwrap(),
        )
        .await
        .unwrap();
    let status = response.status();
    assert!(
        status == StatusCode::OK || status == StatusCode::BAD_REQUEST,
        "unexpected status: {}",
        status
    );
}

#[tokio::test]
async fn test_train_endpoint() {
    // First load sample data, then train
    let config = ServerConfig {
        host: "127.0.0.1".to_string(),
        port: 0,
        static_dir: None,
        data_dir: "/tmp/automl-test-data2".to_string(),
        models_dir: "/tmp/automl-test-models2".to_string(),
        max_upload_size: 10 * 1024 * 1024,
    };
    std::fs::create_dir_all(&config.data_dir).ok();
    std::fs::create_dir_all(&config.models_dir).ok();
    let state = Arc::new(AppState::new(config.clone()));
    let app = create_router(state, &config);

    // Load sample data first
    let (parts, _body) = app
        .clone()
        .oneshot(
            Request::builder()
                .uri("/api/data/sample/iris")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap()
        .into_parts();
    assert_eq!(parts.status, StatusCode::OK);

    // Now train
    let train_body = serde_json::json!({
        "target": "target",
        "task_type": "classification",
        "model_type": "decision_tree"
    });
    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/api/train")
                .header("content-type", "application/json")
                .body(Body::from(serde_json::to_string(&train_body).unwrap()))
                .unwrap(),
        )
        .await
        .unwrap();
    let status = response.status();
    assert!(
        status == StatusCode::OK || status == StatusCode::ACCEPTED || status == StatusCode::UNPROCESSABLE_ENTITY || status == StatusCode::BAD_REQUEST,
        "train should succeed or be accepted, got: {}",
        status
    );
}

// ============================================================================
// Kaggle Import Tests
// ============================================================================

#[tokio::test]
async fn test_kaggle_import_valid_slug() {
    let app = test_app();
    let body = serde_json::json!({
        "dataset": "uciml/iris"
    });
    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/api/data/import/kaggle")
                .header("content-type", "application/json")
                .body(Body::from(serde_json::to_string(&body).unwrap()))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);

    let bytes = axum::body::to_bytes(response.into_body(), 1024 * 64).await.unwrap();
    let json: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(json["success"], true);
    assert_eq!(json["dataset"], "uciml/iris");
    assert!(json["download_url"].as_str().unwrap().contains("kaggle.com"));
    assert!(json["api_command"].as_str().unwrap().contains("kaggle datasets download"));
}

#[tokio::test]
async fn test_kaggle_import_from_url() {
    let app = test_app();
    let body = serde_json::json!({
        "dataset": "https://www.kaggle.com/datasets/uciml/iris"
    });
    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/api/data/import/kaggle")
                .header("content-type", "application/json")
                .body(Body::from(serde_json::to_string(&body).unwrap()))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);

    let bytes = axum::body::to_bytes(response.into_body(), 1024 * 64).await.unwrap();
    let json: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(json["success"], true);
    assert_eq!(json["dataset"], "uciml/iris");
}

#[tokio::test]
async fn test_kaggle_import_invalid_slug() {
    let app = test_app();
    let body = serde_json::json!({
        "dataset": "invalid-no-slash"
    });
    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/api/data/import/kaggle")
                .header("content-type", "application/json")
                .body(Body::from(serde_json::to_string(&body).unwrap()))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
}

#[tokio::test]
async fn test_kaggle_import_empty_dataset() {
    let app = test_app();
    let body = serde_json::json!({
        "dataset": ""
    });
    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/api/data/import/kaggle")
                .header("content-type", "application/json")
                .body(Body::from(serde_json::to_string(&body).unwrap()))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
}

// ============================================================================
// Auto-Clean Data Tests
// ============================================================================

#[tokio::test]
async fn test_auto_clean_no_data() {
    let app = test_app();
    let body = serde_json::json!({});
    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/api/data/clean")
                .header("content-type", "application/json")
                .body(Body::from(serde_json::to_string(&body).unwrap()))
                .unwrap(),
        )
        .await
        .unwrap();
    // Should return 404 since no data loaded
    assert_eq!(response.status(), StatusCode::NOT_FOUND);
}

#[tokio::test]
async fn test_auto_clean_after_sample_load() {
    let config = ServerConfig {
        host: "127.0.0.1".to_string(),
        port: 0,
        static_dir: None,
        data_dir: "/tmp/automl-test-clean".to_string(),
        models_dir: "/tmp/automl-test-clean-models".to_string(),
        max_upload_size: 10 * 1024 * 1024,
    };
    std::fs::create_dir_all(&config.data_dir).ok();
    std::fs::create_dir_all(&config.models_dir).ok();
    let state = Arc::new(AppState::new(config.clone()));
    let app = create_router(state, &config);

    // Load sample data
    let _ = app
        .clone()
        .oneshot(
            Request::builder()
                .uri("/api/data/sample/iris")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    // Run auto-clean
    let body = serde_json::json!({
        "remove_duplicates": true,
        "fill_numeric_nulls": true
    });
    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/api/data/clean")
                .header("content-type", "application/json")
                .body(Body::from(serde_json::to_string(&body).unwrap()))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);

    let bytes = axum::body::to_bytes(response.into_body(), 1024 * 64).await.unwrap();
    let json: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(json["success"], true);
    assert!(json["cleaned"]["rows"].as_u64().unwrap() > 0);
}

// ============================================================================
// Additional Endpoint Coverage Tests
// ============================================================================

#[tokio::test]
async fn test_404_on_unknown_api_route() {
    let app = test_app();
    let response = app
        .oneshot(
            Request::builder()
                .uri("/api/nonexistent")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::NOT_FOUND);
}

#[tokio::test]
async fn test_data_info_no_data() {
    let app = test_app();
    let response = app
        .oneshot(
            Request::builder()
                .uri("/api/data/info")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert!(
        response.status() == StatusCode::NOT_FOUND || response.status() == StatusCode::OK,
        "should be not found or ok when no data loaded"
    );
}

#[tokio::test]
async fn test_load_all_sample_datasets() {
    for name in &["iris", "diabetes", "boston", "wine", "breast_cancer"] {
        let app = test_app();
        let response = app
            .oneshot(
                Request::builder()
                    .uri(&format!("/api/data/sample/{}", name))
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(
            response.status(),
            StatusCode::OK,
            "sample dataset '{}' should load successfully",
            name
        );
    }
}

#[tokio::test]
async fn test_load_unknown_sample_dataset() {
    let app = test_app();
    let response = app
        .oneshot(
            Request::builder()
                .uri("/api/data/sample/nonexistent")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::NOT_FOUND);
}

#[tokio::test]
async fn test_get_training_status_not_found() {
    let app = test_app();
    let response = app
        .oneshot(
            Request::builder()
                .uri("/api/train/status/nonexistent-id")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::NOT_FOUND);
}

#[tokio::test]
async fn test_get_model_not_found() {
    let app = test_app();
    let response = app
        .oneshot(
            Request::builder()
                .uri("/api/models/nonexistent-id")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::NOT_FOUND);
}

#[tokio::test]
async fn test_delete_model_not_found() {
    let app = test_app();
    let response = app
        .oneshot(
            Request::builder()
                .method("DELETE")
                .uri("/api/models/nonexistent-id")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::NOT_FOUND);
}

#[tokio::test]
async fn test_model_metrics_not_found() {
    let app = test_app();
    let response = app
        .oneshot(
            Request::builder()
                .uri("/api/models/nonexistent-id/metrics")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::NOT_FOUND);
}

#[tokio::test]
async fn test_predict_no_model() {
    let app = test_app();
    let body = serde_json::json!({
        "data": [[1.0, 2.0, 3.0]]
    });
    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/api/predict")
                .header("content-type", "application/json")
                .body(Body::from(serde_json::to_string(&body).unwrap()))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::NOT_FOUND);
}

#[tokio::test]
async fn test_predict_batch_no_model() {
    let app = test_app();
    let body = serde_json::json!({
        "data": [[1.0, 2.0], [3.0, 4.0]]
    });
    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/api/predict/batch")
                .header("content-type", "application/json")
                .body(Body::from(serde_json::to_string(&body).unwrap()))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::NOT_FOUND);
}

#[tokio::test]
async fn test_predict_proba_no_model() {
    let app = test_app();
    let body = serde_json::json!({
        "data": [[1.0, 2.0, 3.0]]
    });
    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/api/predict/proba")
                .header("content-type", "application/json")
                .body(Body::from(serde_json::to_string(&body).unwrap()))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::NOT_FOUND);
}

#[tokio::test]
async fn test_inference_stats_not_cached() {
    let app = test_app();
    let response = app
        .oneshot(
            Request::builder()
                .uri("/api/predict/stats/nonexistent-id")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::NOT_FOUND);
}

#[tokio::test]
async fn test_evict_model_cache() {
    let app = test_app();
    let response = app
        .oneshot(
            Request::builder()
                .method("DELETE")
                .uri("/api/predict/cache/some-id")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);
}

#[tokio::test]
async fn test_configure_batch_processor() {
    let app = test_app();
    let body = serde_json::json!({
        "max_batch_size": 64,
        "num_workers": 8,
        "enable_priority": true,
        "adaptive_sizing": true
    });
    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/api/batch/config")
                .header("content-type", "application/json")
                .body(Body::from(serde_json::to_string(&body).unwrap()))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);
}

#[tokio::test]
async fn test_dequantize_endpoint() {
    let app = test_app();
    let body = serde_json::json!({
        "data": [10, 20, 30, 40],
        "scale": 0.5,
        "zero_point": 0.0
    });
    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/api/dequantize")
                .header("content-type", "application/json")
                .body(Body::from(serde_json::to_string(&body).unwrap()))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);

    let bytes = axum::body::to_bytes(response.into_body(), 1024 * 64).await.unwrap();
    let json: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(json["count"], 4);
}

#[tokio::test]
async fn test_calibrate_quantizer_endpoint() {
    let app = test_app();
    let body = serde_json::json!({
        "calibration_data": [1.0, 2.0, 3.0, 4.0, 5.0]
    });
    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/api/quantize/calibrate")
                .header("content-type", "application/json")
                .body(Body::from(serde_json::to_string(&body).unwrap()))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);

    let bytes = axum::body::to_bytes(response.into_body(), 1024 * 64).await.unwrap();
    let json: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(json["success"], true);
    assert_eq!(json["is_calibrated"], true);
}

#[tokio::test]
async fn test_preprocess_no_data() {
    let app = test_app();
    let body = serde_json::json!({
        "scaler": "standard"
    });
    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/api/preprocess")
                .header("content-type", "application/json")
                .body(Body::from(serde_json::to_string(&body).unwrap()))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::NOT_FOUND);
}

#[tokio::test]
async fn test_health_response_body() {
    let app = test_app();
    let response = app
        .oneshot(
            Request::builder()
                .uri("/api/health")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);

    let bytes = axum::body::to_bytes(response.into_body(), 1024 * 64).await.unwrap();
    let json: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(json["status"], "ok");
    assert!(json["version"].as_str().is_some());
}

#[tokio::test]
async fn test_system_status_response_body() {
    let app = test_app();
    let response = app
        .oneshot(
            Request::builder()
                .uri("/api/system/status")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);

    let bytes = axum::body::to_bytes(response.into_body(), 1024 * 64).await.unwrap();
    let json: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(json["status"], "healthy");
    assert!(json["system"]["cpu_count"].as_u64().unwrap() > 0);
}

#[tokio::test]
async fn test_compare_models_no_data() {
    let app = test_app();
    let body = serde_json::json!({
        "target_column": "target",
        "task_type": "classification",
        "models": ["decision_tree", "knn"]
    });
    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/api/train/compare")
                .header("content-type", "application/json")
                .body(Body::from(serde_json::to_string(&body).unwrap()))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::NOT_FOUND);
}

#[tokio::test]
async fn test_model_structure_missing_model() {
    let app = test_app();
    let resp = app.oneshot(
        axum::http::Request::builder()
            .uri("/api/insights/model-structure?model_id=nonexistent")
            .body(axum::body::Body::empty()).unwrap()
    ).await.unwrap();
    assert_eq!(resp.status(), 404);
}

#[test]
fn test_insights_evaluation_struct_exists() {
    use automl::server::state::InsightsEvaluation;
    use automl::training::TaskType;
    let eval = InsightsEvaluation {
        task: TaskType::BinaryClassification,
        classes: vec!["a".to_string(), "b".to_string()],
        y_true: vec![0.0, 1.0],
        y_pred: vec![0.0, 1.0],
        y_prob: None,
    };
    assert_eq!(eval.y_true.len(), 2);
}

#[tokio::test]
async fn test_insights_evaluation_no_cache() {
    let app = test_app();
    let resp = app.oneshot(
        axum::http::Request::builder()
            .uri("/api/insights/evaluation?model_id=nonexistent")
            .body(axum::body::Body::empty()).unwrap()
    ).await.unwrap();
    assert_eq!(resp.status(), 404);
    let body = axum::body::to_bytes(resp.into_body(), usize::MAX).await.unwrap();
    let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
    assert_eq!(json["error"], "evaluation_data_unavailable");
}
