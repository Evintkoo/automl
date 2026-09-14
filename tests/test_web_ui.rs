//! Integration test: Web UI structure, tabs, and sub-tabs

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
        data_dir: "/tmp/automl-test-webui-data".to_string(),
        models_dir: "/tmp/automl-test-webui-models".to_string(),
        max_upload_size: 10 * 1024 * 1024,
    };
    std::fs::create_dir_all(&config.data_dir).ok();
    std::fs::create_dir_all(&config.models_dir).ok();
    let state = Arc::new(AppState::new(config.clone()));
    create_router(state, &config)
}

async fn get_index_html() -> String {
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
    let bytes = axum::body::to_bytes(response.into_body(), 1024 * 1024).await.unwrap();
    String::from_utf8(bytes.to_vec()).unwrap()
}

// ============================================================================
// Tab Structure Tests
// ============================================================================

#[tokio::test]
async fn test_index_has_five_navigation_tabs() {
    let html = get_index_html().await;
    let expected_tabs = ["dashboard", "data", "train", "analysis", "reports", "monitor"];
    for tab in &expected_tabs {
        assert!(
            html.contains(&format!("data-tab=\"{}\"", tab)),
            "Missing nav tab: {}",
            tab
        );
    }
}

#[tokio::test]
async fn test_index_has_five_tab_panels() {
    let html = get_index_html().await;
    let expected_panels = [
        "et-dashboard", "et-data", "et-train", "et-analysis", "et-reports", "et-monitor",
    ];
    for panel in &expected_panels {
        assert!(
            html.contains(&format!("id=\"{}\"", panel)),
            "Missing tab panel: {}",
            panel
        );
    }
}

#[tokio::test]
async fn test_dashboard_tab_is_active_by_default() {
    let html = get_index_html().await;
    assert!(html.contains("id=\"et-dashboard\" class=\"tab-panel active\""));
}

// ============================================================================
// Train Sub-Tab Tests
// ============================================================================

#[tokio::test]
async fn test_train_tab_has_subtabs() {
    let html = get_index_html().await;
    let subtabs = ["autotune", "automl", "single", "hyperopt", "ensemble"];
    for sub in &subtabs {
        assert!(
            html.contains(&format!("id=\"esp-train-{}\"", sub)),
            "Missing train sub-panel: {}",
            sub
        );
    }
}

#[tokio::test]
async fn test_autotune_is_default_active_subtab() {
    let html = get_index_html().await;
    assert!(
        html.contains("id=\"esp-train-autotune\" class=\"sub-panel active\""),
        "Auto-Tune should be the default active sub-tab"
    );
}

// ============================================================================
// Analysis Sub-Tab Tests
// ============================================================================

#[tokio::test]
async fn test_analysis_tab_has_subtabs() {
    let html = get_index_html().await;
    let subtabs = ["explain", "anomaly", "dimred"];
    for sub in &subtabs {
        assert!(
            html.contains(&format!("id=\"esp-analysis-{}\"", sub)),
            "Missing analysis sub-panel: {}",
            sub
        );
    }
}

// ============================================================================
// Visualization Canvas Tests
// ============================================================================

#[tokio::test]
async fn test_chart_canvases_present() {
    let html = get_index_html().await;
    let canvases = [
        "e-scatter", "e-corr", "e-at-bar", "e-aml-chart",
        "e-train-radar", "e-train-bars", "e-ho-cvg", "e-ho-best",
        "e-anom-scatter", "e-anom-donut", "e-dr-scatter",
        "e-spark-cpu", "e-spark-mem",
    ];
    for canvas in &canvases {
        assert!(
            html.contains(&format!("id=\"{}\"", canvas)),
            "Missing chart canvas: {}",
            canvas
        );
    }
}

// ============================================================================
// Auto-Tune Integration Tests
// ============================================================================

#[tokio::test]
async fn test_auto_tune_has_target_and_task() {
    let html = get_index_html().await;
    assert!(html.contains("id=\"e-at-target\""), "Auto-Tune missing target column selector");
    assert!(html.contains("id=\"e-at-task\""), "Auto-Tune missing task type selector");
    assert!(html.contains("id=\"e-at-btn\""), "Auto-Tune missing start button");
}

#[tokio::test]
async fn test_auto_tune_calls_api() {
    let html = get_index_html().await;
    assert!(html.contains("/api/autotune"), "Auto-tune must call /api/autotune");
    assert!(html.contains("/api/train/status/"), "Auto-tune must poll /api/train/status/");
}

// ============================================================================
// Content-Security-Policy Integration Test
// ============================================================================

#[tokio::test]
async fn test_index_response_headers() {
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

    let ct = response.headers().get("content-type").unwrap().to_str().unwrap();
    assert!(ct.contains("text/html"), "Index should be served as HTML");

    let cc = response.headers().get("cache-control").unwrap().to_str().unwrap();
    assert!(cc.contains("no-cache"), "Index should have no-cache header");
}

// ============================================================================
// CSS & Styling Tests
// ============================================================================

#[tokio::test]
async fn test_index_has_inline_styles() {
    let html = get_index_html().await;
    assert!(html.contains("<style>"), "Index must have inline styles");
    assert!(html.contains(".sub-tab"), "Must have sub-tab CSS class");
    assert!(html.contains(".sub-panel"), "Must have sub-panel CSS class");
}

#[tokio::test]
async fn test_index_has_progress_styles() {
    let html = get_index_html().await;
    assert!(html.contains(".tp-card"), "Must have training progress card styles");
    assert!(html.contains(".tp-ring"), "Must have progress ring styles");
    assert!(html.contains(".mc-grid"), "Must have model card grid styles");
    assert!(html.contains(".mc-card"), "Must have model card styles");
}

// ============================================================================
// Essential Element IDs
// ============================================================================

#[tokio::test]
async fn test_critical_element_ids_present() {
    let html = get_index_html().await;
    let ids = [
        "e-target", "e-task", "e-model",
        "e-train-btn", "e-ho-btn", "e-ens-btn", "e-at-btn",
        "e-anomaly-btn", "e-explain-btn", "e-dr-btn",
    ];
    for id in &ids {
        assert!(
            html.contains(&format!("id=\"{}\"", id)),
            "Missing critical element: {}",
            id
        );
    }
}

// ============================================================================
// UMAP / PCA (Dimensionality Reduction) Tests
// ============================================================================

#[tokio::test]
async fn test_dimred_controls_present() {
    let html = get_index_html().await;
    assert!(html.contains("id=\"e-dr-method\""), "Missing dim. reduction method selector");
    assert!(html.contains("id=\"e-dr-neighbors\""), "Missing UMAP neighbors input");
    assert!(html.contains("id=\"e-dr-color\""), "Missing color-by selector");
    assert!(html.contains("id=\"e-dr-btn\""), "Missing dim. reduction run button");
}

#[tokio::test]
async fn test_dimred_result_elements_present() {
    let html = get_index_html().await;
    assert!(html.contains("id=\"e-dr-result\""), "Missing dim. reduction result container");
    assert!(html.contains("id=\"e-dr-scatter\""), "Missing dim. reduction scatter canvas");
    assert!(html.contains("id=\"e-dr-empty\""), "Missing dim. reduction empty state");
}

#[tokio::test]
async fn test_dimred_javascript_function_exists() {
    let html = get_index_html().await;
    assert!(html.contains("function eRunDimRed"), "Missing eRunDimRed JS function");
}

#[tokio::test]
async fn test_dimred_calls_api() {
    let html = get_index_html().await;
    assert!(html.contains("/api/visualization/umap"), "Must call UMAP API endpoint");
    assert!(html.contains("/api/visualization/pca"), "Must call PCA API endpoint");
}

// ============================================================================
// Training Progress UI Tests
// ============================================================================

#[tokio::test]
async fn test_training_progress_helpers_exist() {
    let html = get_index_html().await;
    assert!(html.contains("function eProgressRing"), "Missing eProgressRing helper");
    assert!(html.contains("function eStepBar"), "Missing eStepBar helper");
    assert!(html.contains("function eElapsed"), "Missing eElapsed helper");
}

#[tokio::test]
async fn test_core_training_functions_exist() {
    let html = get_index_html().await;
    let fns = [
        "function eTrain", "function ePoll",
        "function eAutoTune", "function eAtPoll",
        "function eHyperOpt", "function eHoPoll",
        "function eEnsemble", "function eEnsPoll",
        "function eRunAutoML", "function eAMLPoll",
        "function eExplain", "function eAnomaly",
    ];
    for f in &fns {
        assert!(html.contains(f), "Missing JS function: {}", f);
    }
}

// ============================================================================
// Chart Drawing Functions
// ============================================================================

#[tokio::test]
async fn test_chart_drawing_functions_exist() {
    let html = get_index_html().await;
    let fns = [
        "function eDrawLine", "function eDrawBars",
        "function eDrawScatter", "function eDrawDonut",
        "function eDrawRadar", "function eDrawCorrelation",
    ];
    for f in &fns {
        assert!(html.contains(f), "Missing chart function: {}", f);
    }
}

// ============================================================================
// API Endpoint References
// ============================================================================

#[tokio::test]
async fn test_api_endpoints_referenced() {
    let html = get_index_html().await;
    let endpoints = [
        "/api/train", "/api/autotune", "/api/hyperopt",
        "/api/ensemble/train", "/api/automl/run",
        "/api/anomaly/detect", "/api/explain/importance",
        "/api/system/status", "/api/models",
    ];
    for ep in &endpoints {
        assert!(html.contains(ep), "Missing API endpoint reference: {}", ep);
    }
}

// ============================================================================
// Reports Tab
// ============================================================================

#[tokio::test]
async fn test_reports_tab_exists() {
    let html = get_index_html().await;
    assert!(html.contains("id=\"et-reports\""), "Missing reports tab panel");
    assert!(html.contains("data-tab=\"reports\""), "Missing reports sidebar button");
    assert!(html.contains("reports:'Reports'"), "Missing reports in eTabNames");
}

#[tokio::test]
async fn test_reports_elements_exist() {
    let html = get_index_html().await;
    let ids = [
        "e-rpt-empty", "e-rpt-nomodels", "e-rpt-select", "e-rpt-models",
        "e-rpt-btn", "e-rpt-content", "e-rpt-date", "e-rpt-summary",
        "e-rpt-best", "e-rpt-table", "e-rpt-rankings",
        "e-rpt-bars", "e-rpt-radar", "e-rpt-time", "e-rpt-scatter",
    ];
    for id in &ids {
        assert!(html.contains(&format!("id=\"{}\"", id)), "Missing report element: {}", id);
    }
}

#[tokio::test]
async fn test_reports_javascript_functions_exist() {
    let html = get_index_html().await;
    let fns = [
        "function eLoadReport", "function eRptSelectAll",
        "function eGenerateReport", "function eRptBack", "function eRptPrint",
    ];
    for f in &fns {
        assert!(html.contains(f), "Missing report JS function: {}", f);
    }
}
