//! Integration test: End-to-end model serving flow
//! Tests: load data → train → list models → predict → stats → cache evict

use automl::server::{AppState, ServerConfig, create_router};
use std::sync::Arc;
use axum::body::Body;
use axum::http::{Request, StatusCode};
use tower::ServiceExt;

fn serve_test_app() -> (axum::Router, Arc<AppState>) {
    let config = ServerConfig {
        host: "127.0.0.1".to_string(),
        port: 0,
        static_dir: None,
        data_dir: "/tmp/automl-test-serve-data".to_string(),
        models_dir: "/tmp/automl-test-serve-models".to_string(),
        max_upload_size: 10 * 1024 * 1024,
    };
    std::fs::create_dir_all(&config.data_dir).ok();
    std::fs::create_dir_all(&config.models_dir).ok();
    let state = Arc::new(AppState::new(config.clone()));
    let app = create_router(state.clone(), &config);
    (app, state)
}

// ============================================================================
// Model Listing Tests
// ============================================================================

#[tokio::test]
async fn test_list_models_returns_json_array() {
    let (app, _) = serve_test_app();
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

    let bytes = axum::body::to_bytes(response.into_body(), 1024 * 64).await.unwrap();
    let json: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    assert!(json["models"].is_array(), "models field must be an array");
}

#[tokio::test]
async fn test_list_models_empty_on_fresh_server() {
    let (app, _) = serve_test_app();
    let response = app
        .oneshot(
            Request::builder()
                .uri("/api/models")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    let bytes = axum::body::to_bytes(response.into_body(), 1024 * 64).await.unwrap();
    let json: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(json["models"].as_array().unwrap().len(), 0);
}

// ============================================================================
// Prediction Error Handling Tests
// ============================================================================

#[tokio::test]
async fn test_predict_with_invalid_json() {
    let (app, _) = serve_test_app();
    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/api/predict")
                .header("content-type", "application/json")
                .body(Body::from("not valid json"))
                .unwrap(),
        )
        .await
        .unwrap();
    // Axum returns 422 for deserialization failures
    let status = response.status();
    assert!(
        status == StatusCode::UNPROCESSABLE_ENTITY || status == StatusCode::BAD_REQUEST,
        "Expected 422 or 400 for invalid JSON, got: {}",
        status
    );
}

#[tokio::test]
async fn test_predict_with_nonexistent_model_id() {
    let (app, _) = serve_test_app();
    let body = serde_json::json!({
        "model_id": "does-not-exist",
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
async fn test_predict_batch_with_empty_data() {
    let (app, _) = serve_test_app();
    let body = serde_json::json!({
        "data": []
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
    // Should fail: either 404 (no model) or 400 (empty data)
    let status = response.status();
    assert!(
        status == StatusCode::NOT_FOUND || status == StatusCode::BAD_REQUEST,
        "Expected 404 or 400 for empty batch, got: {}",
        status
    );
}

#[tokio::test]
async fn test_predict_proba_with_nonexistent_model() {
    let (app, _) = serve_test_app();
    let body = serde_json::json!({
        "model_id": "ghost-model",
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

// ============================================================================
// Inference Stats & Cache Tests
// ============================================================================

#[tokio::test]
async fn test_inference_stats_returns_404_for_uncached_model() {
    let (app, _) = serve_test_app();
    let response = app
        .oneshot(
            Request::builder()
                .uri("/api/predict/stats/uncached-model")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::NOT_FOUND);
}

#[tokio::test]
async fn test_evict_cache_succeeds_even_if_not_cached() {
    let (app, _) = serve_test_app();
    let response = app
        .oneshot(
            Request::builder()
                .method("DELETE")
                .uri("/api/predict/cache/nonexistent")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);
}

// ============================================================================
// Model CRUD Tests
// ============================================================================

#[tokio::test]
async fn test_get_model_returns_404_for_unknown() {
    let (app, _) = serve_test_app();
    let response = app
        .oneshot(
            Request::builder()
                .uri("/api/models/unknown-id")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::NOT_FOUND);
}

#[tokio::test]
async fn test_download_model_returns_404_for_unknown() {
    let (app, _) = serve_test_app();
    let response = app
        .oneshot(
            Request::builder()
                .uri("/api/models/unknown-id/download")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::NOT_FOUND);
}

#[tokio::test]
async fn test_delete_model_returns_404_for_unknown() {
    let (app, _) = serve_test_app();
    let response = app
        .oneshot(
            Request::builder()
                .method("DELETE")
                .uri("/api/models/unknown-id")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::NOT_FOUND);
}

// ============================================================================
// End-to-End: Load → Train → List → Predict
// ============================================================================

#[tokio::test]
async fn test_e2e_load_train_list_predict() {
    let (app, _) = serve_test_app();

    // 1. Load iris dataset
    let response = app
        .clone()
        .oneshot(
            Request::builder()
                .uri("/api/data/sample/iris")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK, "Failed to load iris data");

    // 2. Start training
    let train_body = serde_json::json!({
        "target_column": "species",
        "task_type": "classification",
        "model_type": "decision_tree"
    });
    let response = app
        .clone()
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
    let train_status = response.status();
    assert!(
        train_status == StatusCode::OK || train_status == StatusCode::ACCEPTED,
        "Training should start, got: {}",
        train_status
    );

    let bytes = axum::body::to_bytes(response.into_body(), 1024 * 64).await.unwrap();
    let train_json: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    let job_id = train_json["job_id"].as_str().unwrap_or("");

    // 3. Poll training status (up to 30s)
    let mut completed = false;
    for _ in 0..30 {
        tokio::time::sleep(std::time::Duration::from_secs(1)).await;
        let resp = app
            .clone()
            .oneshot(
                Request::builder()
                    .uri(&format!("/api/train/status/{}", job_id))
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();

        if resp.status() == StatusCode::OK {
            let b = axum::body::to_bytes(resp.into_body(), 1024 * 64).await.unwrap();
            let j: serde_json::Value = serde_json::from_slice(&b).unwrap();
            if j["status"]["Completed"].is_object() {
                completed = true;
                break;
            }
            if j["status"]["Failed"].is_object() {
                panic!("Training failed: {:?}", j["status"]["Failed"]);
            }
        }
    }
    assert!(completed, "Training should complete within 30s");

    // 4. List models - should have at least 1
    let response = app
        .clone()
        .oneshot(
            Request::builder()
                .uri("/api/models")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    let bytes = axum::body::to_bytes(response.into_body(), 1024 * 64).await.unwrap();
    let models_json: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    let models = models_json["models"].as_array().unwrap();
    assert!(!models.is_empty(), "Should have at least one model after training");

    let model_id = models[0]["id"].as_str().unwrap();

    // 5. Predict with the trained model
    let predict_body = serde_json::json!({
        "model_id": model_id,
        "data": [[5.1, 3.5, 1.4, 0.2]]
    });
    let response = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/api/predict")
                .header("content-type", "application/json")
                .body(Body::from(serde_json::to_string(&predict_body).unwrap()))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK, "Prediction should succeed");
    let bytes = axum::body::to_bytes(response.into_body(), 1024 * 64).await.unwrap();
    let pred_json: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(pred_json["success"], true);
    assert!(pred_json["predictions"].is_array());
    assert!(!pred_json["predictions"].as_array().unwrap().is_empty());

    // 6. Batch predict
    let batch_body = serde_json::json!({
        "model_id": model_id,
        "data": [[5.1, 3.5, 1.4, 0.2], [6.7, 3.0, 5.2, 2.3], [4.9, 3.1, 1.5, 0.1]]
    });
    let response = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/api/predict/batch")
                .header("content-type", "application/json")
                .body(Body::from(serde_json::to_string(&batch_body).unwrap()))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK, "Batch prediction should succeed");
    let bytes = axum::body::to_bytes(response.into_body(), 1024 * 64).await.unwrap();
    let batch_json: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(batch_json["success"], true);
    assert_eq!(batch_json["count"], 3);

    // 7. Get inference stats
    let response = app
        .clone()
        .oneshot(
            Request::builder()
                .uri(&format!("/api/predict/stats/{}", model_id))
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK, "Stats should be available after predictions");
    let bytes = axum::body::to_bytes(response.into_body(), 1024 * 64).await.unwrap();
    let stats_json: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    assert!(stats_json["stats"]["total_predictions"].as_u64().unwrap() > 0);

    // 8. Get model details
    let response = app
        .clone()
        .oneshot(
            Request::builder()
                .uri(&format!("/api/models/{}", model_id))
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);

    // 9. Evict cache
    let response = app
        .clone()
        .oneshot(
            Request::builder()
                .method("DELETE")
                .uri(&format!("/api/predict/cache/{}", model_id))
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);

    // 10. Stats should be 404 after eviction
    let response = app
        .oneshot(
            Request::builder()
                .uri(&format!("/api/predict/stats/{}", model_id))
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::NOT_FOUND, "Stats should be gone after cache eviction");
}
