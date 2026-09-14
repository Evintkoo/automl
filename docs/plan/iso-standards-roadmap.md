# AutoML — ISO Standards Compliance Roadmap

> Local-only document. Tracked via `.git/info/exclude`, never committed.

## Overview

AutoML is a pure-Rust AutoML framework (v0.5.0, ~52,885 LOC, 425 tests)
with 23 modules covering ML training (17+ models), preprocessing, hyperparameter
optimization, inference, explainability, drift detection, and a 40+ endpoint REST
API (Axum). This roadmap maps the codebase to applicable ISO standards and defines
concrete implementation tasks to close compliance gaps.

---

## 1. Applicable ISO Standards

### 1.1 AI-Specific Standards

| Standard | Title | Priority | Certifiable |
|----------|-------|----------|-------------|
| ISO/IEC 42001:2023 | AI Management System (AIMS) | HIGH | Yes |
| ISO/IEC 23053:2022 | Framework for AI using ML | HIGH | No (guidance) |
| ISO/IEC 23894:2023 | AI Risk Management | HIGH | No (guidance) |
| ISO/IEC 22989:2022 | AI Concepts and Terminology | MEDIUM | No (vocabulary) |
| ISO/IEC 5338:2023 | AI System Lifecycle Processes | HIGH | No (guidance) |
| ISO/IEC 5259 (series) | Data Quality for AI | HIGH | No (guidance) |
| ISO/IEC 24029 (series) | Robustness of Neural Networks | MEDIUM | No (guidance) |
| ISO/IEC TR 24027:2021 | Bias in AI Systems | HIGH | No (tech report) |
| ISO/IEC TR 24028:2020 | Trustworthiness in AI | HIGH | No (tech report) |
| ISO/IEC 38507:2022 | Governance of AI | LOW | No (guidance) |

### 1.2 Software Quality Standards

| Standard | Title | Priority | Certifiable |
|----------|-------|----------|-------------|
| ISO/IEC 25010:2023 | Software Quality Model (SQuaRE) | HIGH | No (model) |
| ISO/IEC 25012:2008 | Data Quality Model | MEDIUM | No (model) |
| ISO/IEC 25040:2011 | Quality Evaluation Process | MEDIUM | No (process) |
| ISO/IEC 12207:2017 | Software Lifecycle Processes | MEDIUM | No (process) |

### 1.3 Security and Privacy Standards

| Standard | Title | Priority | Certifiable |
|----------|-------|----------|-------------|
| ISO/IEC 27001:2022 | Information Security Management | HIGH | Yes |
| ISO/IEC 27002:2022 | Security Controls | HIGH | No (controls catalog) |
| ISO/IEC 27701:2019 | Privacy Information Management | MEDIUM | Yes (extension to 27001) |
| ISO/IEC 27017:2015 | Cloud Security | LOW | No (guidance) |

### 1.4 Quality Management

| Standard | Title | Priority | Certifiable |
|----------|-------|----------|-------------|
| ISO 9001:2015 | Quality Management Systems | MEDIUM | Yes |
| ISO/IEC 90003:2014 | Software Engineering — QMS Guidelines | MEDIUM | No (guidance) |

---

## 2. Current Compliance Status — Detailed Audit

### 2.1 What Exists (by module, with specifics)

#### Security (`src/security/` — 1,043 LOC) → ISO 27001
| Control | Status | Implementation |
|---------|--------|----------------|
| Authentication | DONE | `ApiKeyVerifier` (constant-time), `JwtVerifier` (configurable expiry) |
| Rate limiting | DONE | `RateLimiter` — sliding window, token bucket, fixed window algorithms |
| Audit logging | PARTIAL | `AuditEntry` — 10,000-entry capped history, in-memory only |
| Input validation | DONE | SQL injection, XSS, directory traversal detection in `SecurityManager` |
| Security headers | DONE | HSTS, X-Content-Type-Options, X-Frame-Options, CSP |
| TLS | PARTIAL | `TlsManager` / `TlsConfig` struct exists, `CertificateInfo` for cert details |
| Secrets | PARTIAL | `SecretsManager` with `SecretType` / `SecretMetadata`, no encryption at rest |
| RBAC | MISSING | No role or permission model |
| Data encryption at rest | MISSING | No AES/ChaCha20 for stored models or data |
| Tamper-evident audit log | MISSING | No hash chain, no immutable storage |

#### Explainability (`src/explainability/` — 1,199 LOC) → ISO TR 24028
| Feature | Status | Implementation |
|---------|--------|----------------|
| Permutation importance | DONE | `PermutationImportance<F>`, `ModelPermutationImportance` |
| Partial Dependence Plots | DONE | `PartialDependence<F>` — 1D, 2D, ICE/c-ICE |
| Local explanations | DONE | `LocalExplainer<F>` — kernel SHAP-like sampling |
| SHAP summary | DONE | `ShapSummary` aggregated from local explanations |
| Feature contributions | DONE | `FeatureContribution` with positive/negative split |
| Decision path extraction | MISSING | No tree traversal path logging |
| Counterfactual explanations | MISSING | No "what-if" analysis |
| Explanation stability/confidence | MISSING | No bootstrap intervals on explanations |

#### Monitoring (`src/monitoring/` — 1,239 LOC) → ISO 25010
| Feature | Status | Implementation |
|---------|--------|----------------|
| Latency tracking | DONE | Rolling window, p50/p95/p99 percentiles |
| Histogram buckets | DONE | Prometheus-style `HistogramBucket` |
| Alert system | DONE | `AlertManager` with conditions, cooldowns, handlers |
| System metrics | DONE | `SystemMonitor` — CPU, memory |
| Metrics export (Prometheus) | MISSING | No `/metrics` endpoint, no OpenTelemetry |
| Distributed tracing | MISSING | `tracing` crate used but no OTLP exporter |
| SLA/SLO definitions | MISSING | No configurable thresholds with enforcement |
| Dashboard data API | PARTIAL | `/api/monitoring/dashboard` exists |

#### Tracking (`src/tracking/` — 1,036 LOC) → ISO 5338
| Feature | Status | Implementation |
|---------|--------|----------------|
| Experiment management | DONE | `ExperimentTracker`, `Experiment`, `Run` |
| Parameter logging | DONE | Key-value param storage per run |
| Metric logging | DONE | Step/epoch tracking with history |
| Artifact tracking | PARTIAL | Path logging only, no actual artifact storage |
| Run comparison | DONE | `best_run()` with maximize/minimize |
| Report generation | DONE | Text report generation |
| Storage backend | PARTIAL | `LocalStorage` only, `StorageBackend` trait exists |
| Model versioning | PARTIAL | `ModelVersion` in export, no rollback workflow |
| Reproducibility | MISSING | No seed management or env fingerprinting |

#### Drift Detection (`src/drift/` — 1,518 LOC) → ISO 42001 / 5338
| Feature | Status | Implementation |
|---------|--------|----------------|
| Statistical tests | DONE | KS test, PSI, Jensen-Shannon divergence |
| Online detection | DONE | DDM, ADWIN, Page-Hinkley, EDDM |
| Multi-feature monitoring | DONE | `FeatureDriftMonitor` with aggregation |
| Severity levels | DONE | 0=none, 1=warning, 2=critical |
| Multivariate drift | MISSING | No cross-feature correlation drift |
| Root cause analysis | MISSING | No attribution of drift to specific features |
| Auto-retrain trigger | MISSING | No integration with training engine |

#### Calibration (`src/calibration/` — 1,184 LOC) → ISO TR 24028
| Feature | Status | Implementation |
|---------|--------|----------------|
| Platt scaling | DONE | Newton's method optimization |
| Isotonic regression | DONE | Pool-adjacent-violators algorithm |
| Temperature scaling | DONE | Gradient descent optimization |
| Beta calibration | DONE | 3-parameter model |
| Calibration metrics | DONE | ECE, MCE, Brier score, reliability diagram |
| Multi-class calibration | MISSING | Single-class focus |

#### Export (`src/export/` — 2,029 LOC) → ISO 5338 / 23053
| Feature | Status | Implementation |
|---------|--------|----------------|
| Binary serialization | DONE | Magic bytes, format versioning |
| JSON serialization | DONE | Portable, human-readable |
| ONNX export | DONE | `ONNXExporter` with operator support |
| PMML export | DONE | `PMMLExporter` |
| Model metadata | DONE | `ModelMetadata` — features, hyperparams, metrics |
| Model registry | DONE | `ModelRegistry` catalog |
| Model versioning | DONE | `ModelVersion`, `VersionedModel` |
| Model signing | MISSING | No cryptographic signature verification |
| Model cards | MISSING | No auto-generated documentation |

#### Preprocessing (`src/preprocessing/` — 3,630 LOC) → ISO 5259
| Feature | Status | Implementation |
|---------|--------|----------------|
| Scalers | DONE | Standard, MinMax, Robust |
| Encoders | DONE | OneHot, Label, Target, Hash |
| Imputation | DONE | Mean, median, mode, forward/backward fill |
| Transformers | DONE | Log, Power, Box-Cox, Yeo-Johnson |
| Outlier detection | DONE | IQR, Z-score, Isolation Forest |
| Feature selection | DONE | Correlation, variance, statistical |
| Data profiling | DONE | `FeatureStats`, `ColumnType` |
| Data provenance | MISSING | No source/lineage tracking |
| Data quality scoring | MISSING | No completeness/validity metrics |

#### API (`src/server/api.rs` — 40+ endpoints) → ISO 25010
| Area | Endpoints | Status |
|------|-----------|--------|
| Data management | 7 endpoints | DONE |
| Preprocessing | 2 endpoints | DONE |
| Training | 3 endpoints | DONE |
| Model management | 4 endpoints | DONE |
| Inference | 5 endpoints | DONE |
| Hyperopt | 5 endpoints | DONE |
| Advanced ML | 8 endpoints | DONE |
| Monitoring | 5 endpoints | DONE |
| Export/Quantize | 4 endpoints | DONE |
| Security/Audit | 3 endpoints | DONE |
| Fairness | 0 endpoints | MISSING |
| Provenance/Lineage | 0 endpoints | MISSING |
| Model cards | 0 endpoints | MISSING |
| Compliance reporting | 0 endpoints | MISSING |

### 2.2 Overall Readiness Score

| Standard | Readiness | Blocking Gaps |
|----------|-----------|---------------|
| ISO/IEC 42001 (AIMS) | 30% | No AIMS policy, risk register, impact assessment |
| ISO/IEC 23053 (ML Framework) | 75% | Missing provenance, model cards |
| ISO/IEC 23894 (AI Risk) | 20% | No risk identification, assessment, or treatment |
| ISO/IEC 5338 (AI Lifecycle) | 60% | Missing versioning workflow, deprecation, reproducibility |
| ISO/IEC 5259 (Data Quality) | 50% | Missing provenance, quality scoring, datasheets |
| ISO/IEC TR 24027 (Bias) | 15% | No fairness metrics, protected attributes, bias detection |
| ISO/IEC TR 24028 (Trustworthiness) | 65% | Missing model cards, decision paths, uncertainty |
| ISO/IEC 27001 (ISMS) | 55% | Missing RBAC, encryption at rest, tamper-evident logs |
| ISO/IEC 25010 (SQuaRE) | 70% | Missing SLOs, metrics export, formal QA process |
| ISO/IEC 27701 (Privacy) | 10% | No PII detection, anonymization, retention policies |

---

## 3. Implementation Plan

### Phase 1: Foundation — Documentation and Governance

**Standards:** ISO 42001, ISO 23894, ISO 27001, ISO 22989
**Type:** Documentation only (no code changes)
**Location:** `docs/plan/governance/`

#### 3.1.1 — AIMS Policy Document
- **File:** `docs/plan/governance/aims-policy.md`
- **Content:**
  - Scope: AutoML as an AI system provider
  - AI principles: fairness, transparency, accountability, safety, privacy
  - Roles: AI system owner, data steward, model reviewer, security officer
  - Risk appetite: acceptable levels for bias, drift, security incidents
  - Review cadence: quarterly management review cycle
- **ISO refs:** 42001 Clauses 4.1-4.4, 5.1-5.3, 6.1

#### 3.1.2 — AI Risk Register
- **File:** `docs/plan/governance/risk-register.md`
- **Content (risk matrix):**

| ID | Risk | Likelihood | Impact | Treatment |
|----|------|-----------|--------|-----------|
| R01 | Training data poisoning | Medium | High | Input validation, anomaly detection on upload |
| R02 | Model theft via API | Low | High | Rate limiting, auth, model encryption |
| R03 | Adversarial inputs at inference | Medium | Medium | Input validation, anomaly scoring |
| R04 | Demographic bias in predictions | High | High | Fairness metrics (Phase 2) |
| R05 | Data drift causing silent degradation | High | High | Drift detection (already implemented) |
| R06 | PII leakage in training data | Medium | Critical | PII detection (Phase 5) |
| R07 | Model inversion attack | Low | High | Output perturbation, rate limiting |
| R08 | Supply chain compromise (deps) | Low | Critical | `cargo audit`, dep pinning |
| R09 | Denial of service on API | Medium | Medium | Rate limiting (already implemented) |
| R10 | Unauthorized model deployment | Medium | High | RBAC (Phase 5) |

- **ISO refs:** 23894 Clauses 6-7, 42001 Clause 6.1

#### 3.1.3 — Threat Model (STRIDE)
- **File:** `docs/plan/governance/threat-model.md`
- **Scope:** REST API, web UI, CLI, training pipeline, model storage
- **Data flow diagrams:** User → API Gateway → Auth → Handler → Engine → Storage
- **Trust boundaries:** External (API clients), Internal (server modules), Storage
- **Existing mitigations to document:**
  - `SecurityManager::validate_input()` — injection prevention
  - `RateLimiter` — DoS mitigation
  - `ApiKeyVerifier` / `JwtVerifier` — authn
  - `SecurityMiddleware` — header injection, CORS
- **ISO refs:** 27001 Clause 6.1.2, 27002 Control 5.1

#### 3.1.4 — Data Classification Scheme
- **File:** `docs/plan/governance/data-classification.md`
- **Categories:**
  - **Public:** Sample datasets (iris, diabetes, wine, boston)
  - **Internal:** Training logs, experiment metadata, system metrics
  - **Confidential:** Uploaded user datasets, trained models, predictions
  - **Restricted:** API keys, JWT secrets, TLS certificates, PII
- **Handling rules per category** (storage, access, retention, disposal)
- **Map to existing code:**
  - `SecretsManager` → Restricted
  - `ExperimentTracker` → Internal
  - Upload handler → Confidential
- **ISO refs:** 27001 Annex A.5.12-5.13

#### 3.1.5 — Terminology Glossary
- **File:** `docs/plan/governance/glossary.md`
- **Align with ISO 22989 terms:**
  - AI system, ML model, training data, test data, feature, label, prediction
  - Bias, fairness, explainability, drift, calibration, robustness
  - Map each term to AutoML struct/module names
- **ISO refs:** 22989 full vocabulary

---

### Phase 2: Data Quality and Fairness

**Standards:** ISO 5259, ISO TR 24027
**Type:** Code changes + documentation

#### 3.2.1 — Data Provenance Tracking
- **New module:** `src/provenance/mod.rs`
- **Structs to create:**

```rust
pub struct DataLineage {
    pub dataset_id: String,
    pub source: DataSource,
    pub ingested_at: chrono::DateTime<chrono::Utc>,
    pub transformations: Vec<TransformationRecord>,
    pub schema_snapshot: SchemaSnapshot,
    pub row_count: usize,
    pub hash: String, // SHA-256 of raw data
}

pub enum DataSource {
    FileUpload { filename: String, mime_type: String },
    Kaggle { dataset_ref: String },
    Url { url: String },
    Sample { name: String },
}

pub struct TransformationRecord {
    pub step: String,           // e.g., "StandardScaler"
    pub parameters: serde_json::Value,
    pub applied_at: chrono::DateTime<chrono::Utc>,
    pub rows_before: usize,
    pub rows_after: usize,
    pub columns_affected: Vec<String>,
}

pub struct SchemaSnapshot {
    pub columns: Vec<ColumnSchema>,
}

pub struct ColumnSchema {
    pub name: String,
    pub dtype: String,
    pub nullable: bool,
    pub stats: Option<ColumnStats>,
}
```

- **Integration points:**
  - Hook into `DataPreprocessor::fit_transform()` to record each step
  - Hook into server upload handlers (`/api/data/upload`, `/api/data/import/*`)
  - Store as JSON alongside dataset in `AppState`
- **New API endpoints:**
  - GET `/api/data/lineage` — get lineage for current dataset
  - GET `/api/data/lineage/{dataset_id}` — get lineage by ID
- **ISO refs:** 5259-1 Clause 6 (data quality management), 5338 Clause 6.3.3

#### 3.2.2 — Data Quality Scoring
- **Extend:** `src/preprocessing/mod.rs`
- **New struct:**

```rust
pub struct DataQualityReport {
    pub overall_score: f64,           // 0.0 - 1.0
    pub completeness: f64,            // 1 - (missing / total)
    pub uniqueness: f64,              // unique rows / total rows
    pub consistency: f64,             // columns matching expected types
    pub validity: f64,                // values within expected ranges
    pub per_column: Vec<ColumnQuality>,
    pub warnings: Vec<QualityWarning>,
}

pub struct ColumnQuality {
    pub column: String,
    pub completeness: f64,
    pub distinct_ratio: f64,
    pub outlier_ratio: f64,
    pub type_consistency: f64,
}

pub enum QualityWarning {
    HighMissingness { column: String, ratio: f64 },
    LowVariance { column: String, variance: f64 },
    HighCardinality { column: String, unique_count: usize },
    SuspectedId { column: String },
    ClassImbalance { column: String, ratio: f64 },
}
```

- **Uses existing:** `FeatureStats` from preprocessing, `OutlierDetector`
- **New API endpoint:** GET `/api/data/quality` — data quality report
- **ISO refs:** 5259-1 Clause 7, 25012

#### 3.2.3 — Fairness Metrics
- **New module:** `src/fairness/mod.rs`
- **Structs:**

```rust
pub struct FairnessConfig {
    pub protected_attributes: Vec<String>,
    pub favorable_label: f64,       // e.g., 1.0 for positive class
    pub thresholds: FairnessThresholds,
}

pub struct FairnessThresholds {
    pub disparate_impact_min: f64,  // typically 0.8 (80% rule)
    pub demographic_parity_max: f64,
    pub equalized_odds_max: f64,
}

pub struct FairnessReport {
    pub metrics: Vec<GroupFairnessMetric>,
    pub overall_fair: bool,
    pub violations: Vec<FairnessViolation>,
}

pub struct GroupFairnessMetric {
    pub protected_attribute: String,
    pub group_value: String,
    pub selection_rate: f64,
    pub true_positive_rate: f64,
    pub false_positive_rate: f64,
    pub positive_predictive_value: f64,
}

pub enum FairnessViolation {
    DisparateImpact { attribute: String, ratio: f64, threshold: f64 },
    DemographicParity { attribute: String, diff: f64, threshold: f64 },
    EqualizedOdds { attribute: String, tpr_diff: f64, fpr_diff: f64 },
}
```

- **Metrics to implement:**
  - Disparate impact ratio (selection_rate_unprivileged / selection_rate_privileged)
  - Demographic parity difference (|P(Y=1|A=0) - P(Y=1|A=1)|)
  - Equalized odds (TPR and FPR parity across groups)
  - Predictive parity (PPV across groups)
  - Individual fairness (similar individuals get similar predictions)
- **New API endpoints:**
  - POST `/api/fairness/evaluate` — compute fairness metrics on predictions
  - GET `/api/fairness/report/{model_id}` — cached fairness report
- **ISO refs:** TR 24027 Clauses 6-8

#### 3.2.4 — Bias Detection Pipeline
- **Extend:** `src/fairness/mod.rs` (or `src/fairness/bias.rs`)
- **Pre-training analysis:**
  - Class imbalance ratio per protected attribute
  - Representation analysis (group sizes)
  - Label distribution skew per group
  - Correlation between protected attributes and target
- **Post-training analysis:**
  - Prediction distribution per group
  - Error rate disparity
  - Confidence score disparity
  - Integrate with existing `LocalExplainer` for attribution by group
- **New API endpoint:** POST `/api/fairness/bias-scan` — run full bias analysis
- **ISO refs:** TR 24027 Clauses 6.2, 7

#### 3.2.5 — Datasheet Generation
- **Extend:** `src/export/mod.rs`
- **Auto-generate after data load:**
  - Dataset name, source, size, feature descriptions
  - Collection methodology (if provided)
  - Known limitations, class distribution
  - Quality score (from 3.2.2)
  - Provenance (from 3.2.1)
- **Output format:** JSON + Markdown
- **New API endpoint:** GET `/api/data/datasheet` — auto-generated datasheet
- **ISO refs:** 5259-3 (data documentation), TR 24028

---

### Phase 3: Transparency and Explainability

**Standards:** ISO TR 24028, ISO 23053
**Type:** Code changes extending existing modules

#### 3.3.1 — Model Cards
- **Extend:** `src/export/mod.rs`
- **New struct:**

```rust
pub struct ModelCard {
    pub model_details: ModelDetails,
    pub intended_use: IntendedUse,
    pub metrics: ModelMetricsCard,
    pub training_data_summary: TrainingDataSummary,
    pub ethical_considerations: Vec<String>,
    pub limitations: Vec<String>,
    pub generated_at: chrono::DateTime<chrono::Utc>,
}

pub struct ModelDetails {
    pub name: String,
    pub version: String,
    pub model_type: String,
    pub task_type: String,  // classification / regression
    pub framework: String,  // "AutoML v0.5.0"
    pub hyperparameters: serde_json::Value,
}

pub struct IntendedUse {
    pub primary_use: String,
    pub out_of_scope: Vec<String>,
    pub users: Vec<String>,
}

pub struct ModelMetricsCard {
    pub primary_metric: String,
    pub primary_value: f64,
    pub all_metrics: HashMap<String, f64>,
    pub evaluation_method: String, // "5-fold cross-validation"
    pub confidence_intervals: Option<HashMap<String, (f64, f64)>>,
}

pub struct TrainingDataSummary {
    pub num_samples: usize,
    pub num_features: usize,
    pub feature_names: Vec<String>,
    pub target_distribution: HashMap<String, usize>,
    pub quality_score: Option<f64>,
}
```

- **Auto-generate:** After training completes in `TrainingEngine`
- **Populate `IntendedUse` and `ethical_considerations`:** Provide sensible defaults
  with option for user override via config
- **Output:** JSON + Markdown
- **New API endpoints:**
  - GET `/api/models/{model_id}/card` — get model card
  - PUT `/api/models/{model_id}/card` — update card metadata (intended use, etc.)
- **ISO refs:** TR 24028 Clause 7, 42001 Annex D

#### 3.3.2 — Decision Path Extraction
- **Extend:** `src/explainability/mod.rs`
- **For tree-based models (Decision Tree, Random Forest, GBM, etc.):**
  - Extract the path from root to leaf for a given input
  - Return: sequence of (feature, threshold, direction) tuples
  - Human-readable summary: "Feature X > 3.5 AND Feature Y <= 2.1 → Class A"
- **For linear models:**
  - Return: coefficient * feature_value contribution per feature
  - Human-readable: "Feature X contributed +0.42, Feature Y contributed -0.18"
- **New API endpoint:** POST `/api/explain/decision-path`
- **ISO refs:** TR 24028 Clause 7.3

#### 3.3.3 — Prediction Uncertainty
- **Extend:** `src/calibration/mod.rs` + `src/inference/mod.rs`
- **For ensemble models (Random Forest, GBM):**
  - Compute prediction variance across trees (epistemic uncertainty)
- **For all classification models:**
  - Use calibrated probabilities (existing `Calibrator` trait)
  - Entropy of prediction distribution as uncertainty measure
- **New fields in prediction response:**

```rust
pub struct PredictionWithUncertainty {
    pub prediction: f64,
    pub confidence: f64,          // calibrated probability
    pub uncertainty: f64,         // entropy or variance
    pub uncertainty_type: String, // "epistemic" | "aleatoric" | "total"
}
```

- **API:** Extend existing `/api/predict` response to include uncertainty fields
- **ISO refs:** TR 24028 Clause 7.2

#### 3.3.4 — Prediction Audit Trail
- **Extend:** `src/security/audit.rs`
- **Add new audit event type:**

```rust
pub enum AuditEventType {
    Authentication { method: String, success: bool },
    DataUpload { filename: String, rows: usize },
    TrainingStarted { model_type: String, config: serde_json::Value },
    TrainingCompleted { model_id: String, metrics: serde_json::Value },
    Prediction { model_id: String, input_hash: String, output: f64, confidence: f64 },
    ModelExported { model_id: String, format: String },
    DriftDetected { severity: u8, details: String },
    FairnessViolation { attribute: String, metric: String, value: f64 },
}
```

- **Storage:** Currently in-memory (10,000 cap). Extend with optional file-based
  append-only log using `BufWriter` to `audit.jsonl`
- **Queryable:** Filter by event type, time range, model ID
- **New API endpoints:**
  - GET `/api/audit/predictions?model_id=X&from=&to=` — prediction audit log
  - GET `/api/audit/events?type=fairness_violation` — filtered events
- **ISO refs:** TR 24028 Clause 8, 42001 Clause 9.2

---

### Phase 4: Lifecycle Management

**Standards:** ISO 5338, ISO 42001
**Type:** Code changes extending tracking and export modules

#### 3.4.1 — Model Versioning Workflow
- **Extend:** `src/export/mod.rs` (existing `ModelRegistry`, `ModelVersion`)
- **Existing:** `ModelVersion` struct and `VersionedModel` exist but lack workflow
- **Add:**
  - `ModelStatus` enum: `Active`, `Staging`, `Deprecated`, `Retired`
  - Rollback: store previous version references, restore by ID
  - Hash verification: SHA-256 of serialized model bytes
  - `ModelRegistry::promote(id, from_status, to_status)`
  - `ModelRegistry::rollback(id)` → revert to previous active version
- **New API endpoints:**
  - POST `/api/models/{model_id}/promote` — staging → active
  - POST `/api/models/{model_id}/deprecate` — active → deprecated
  - POST `/api/models/{model_id}/rollback` — restore previous version
  - GET `/api/models/{model_id}/versions` — version history
- **ISO refs:** 5338 Clause 6.4.7, 42001 Clause 8.4

#### 3.4.2 — Reproducibility
- **Extend:** `src/tracking/mod.rs`
- **Add to `Run`:**

```rust
pub struct EnvironmentFingerprint {
    pub rust_version: String,
    pub automl_version: String,
    pub os: String,
    pub arch: String,
    pub cpu_features: Vec<String>,  // from existing src/device/
    pub random_seed: Option<u64>,
    pub num_threads: usize,         // rayon thread count
}
```

- **Capture automatically** when `ExperimentTracker::start_run()` is called
- **Seed management:** Propagate a single seed through all random operations
  (training, cross-validation, hyperopt) via `StdRng::seed_from_u64()`
- **Deterministic mode:** Set `rayon::ThreadPoolBuilder::num_threads(1)` +
  fixed seed for bit-exact reproducibility
- **ISO refs:** 5338 Clause 6.3.6, 23053 Clause 8

#### 3.4.3 — Deployment Validation
- **Extend:** `src/monitoring/mod.rs`
- **Pre-deployment checks:**
  - Schema match: prediction input columns match training columns
  - Performance threshold: model metric >= configurable minimum
  - Drift check: run drift detection against reference data
  - Fairness check: run fairness evaluation against threshold
- **Post-deployment smoke test:**
  - Send reference inputs, verify outputs within expected range
  - Latency within SLO
- **New API endpoint:** POST `/api/models/{model_id}/validate` — run validation suite
- **ISO refs:** 5338 Clause 6.4.8, 25010

#### 3.4.4 — Model Deprecation and Retirement
- **Extend:** `src/export/mod.rs` (`ModelRegistry`)
- **Deprecation flow:**
  1. Mark model as deprecated → API returns `Deprecation` header on predictions
  2. Set retirement date → after date, predictions return 410 Gone
  3. Archive model artifact → compress and move to archive storage
  4. Delete model → remove from registry after retention period
- **Audit:** All status transitions logged via audit trail (3.3.4)
- **ISO refs:** 5338 Clause 6.4.10

---

### Phase 5: Security Hardening

**Standards:** ISO 27001, ISO 27002, ISO 27701
**Type:** Code changes + policy documentation

#### 3.5.1 — Role-Based Access Control (RBAC)
- **Extend:** `src/security/mod.rs`
- **New structs:**

```rust
pub enum Role {
    Admin,      // full access
    DataOwner,  // upload, preprocess, view data
    Trainer,    // train, evaluate, export models
    Consumer,   // predict only
    Auditor,    // read-only access to audit logs, metrics, reports
}

pub struct Permission {
    pub resource: Resource,
    pub action: Action,
}

pub enum Resource {
    Data, Model, Training, Prediction, Audit, System, Fairness,
}

pub enum Action {
    Create, Read, Update, Delete,
}
```

- **Integration:** Check permissions in `SecurityMiddleware` before handler dispatch
- **Existing JWT:** Embed role claim in JWT payload
- **ISO refs:** 27001 Annex A.5.15-5.18, 27002 Control 8.3

#### 3.5.2 — PII Detection
- **New module:** `src/privacy/mod.rs`
- **Pattern-based detection:**
  - Email: regex `[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}`
  - Phone: regex for international formats
  - SSN/ID: country-specific patterns
  - Credit card: Luhn algorithm validation
  - Names: high-cardinality string columns with name-like distribution
  - IP addresses, dates of birth
- **Scan on upload:** Auto-scan in `/api/data/upload` handler
- **Response:** `PiiScanResult` with column-level PII classifications and recommendations
- **New API endpoint:** POST `/api/privacy/scan` — PII scan
- **ISO refs:** 27701 Clauses 7.2-7.5

#### 3.5.3 — Data Anonymization
- **Extend:** `src/privacy/mod.rs`
- **Methods:**
  - Column suppression (drop PII columns)
  - Generalization (age → age range, ZIP → first 3 digits)
  - Pseudonymization (consistent hash-based replacement)
  - K-anonymity verification (k=5 default)
- **New API endpoint:** POST `/api/privacy/anonymize` — apply anonymization
- **ISO refs:** 27701 Clause 7.4

#### 3.5.4 — Tamper-Evident Audit Logging
- **Extend:** `src/security/audit.rs`
- **Change from in-memory to file-based:**
  - Append-only JSONL file: `audit.jsonl`
  - Each entry includes SHA-256 hash of previous entry (hash chain)
  - Periodic integrity verification
- **Format:**

```json
{"id": 1, "prev_hash": "0000...", "timestamp": "...", "event": {...}, "hash": "a3b4..."}
{"id": 2, "prev_hash": "a3b4...", "timestamp": "...", "event": {...}, "hash": "c5d6..."}
```

- **Verification:** `verify_audit_chain()` function
- **ISO refs:** 27001 Annex A.8.15, 27002 Control 8.15

#### 3.5.5 — Model Encryption at Rest
- **Extend:** `src/export/mod.rs`
- **Use `ring` or `chacha20poly1305` crate:**
  - Encrypt serialized model bytes before writing to disk
  - Key derivation from configurable master key (HKDF)
  - Model files: magic bytes + encrypted payload
- **Integration:** Optional via `ExportConfig { encrypt: bool }`
- **ISO refs:** 27001 Annex A.8.24, 27002 Control 8.24

#### 3.5.6 — Data Retention Policies
- **New module or extend:** `src/privacy/retention.rs`
- **Configurable per data classification:**
  - Public: indefinite
  - Internal: 1 year (logs, metrics)
  - Confidential: 90 days or configurable
  - Restricted: 30 days, immediate on request
- **Automatic purging:** Background task (tokio) that checks retention
- **Right to deletion:** API endpoint to request data deletion
- **New API endpoint:** DELETE `/api/data/{dataset_id}` — delete with audit trail
- **ISO refs:** 27701 Clauses 7.4.5, 7.4.7

---

### Phase 6: Monitoring and Continuous Compliance

**Standards:** ISO 42001, ISO 25010
**Type:** Code changes + process documentation

#### 3.6.1 — SLA/SLO Definitions
- **Extend:** `src/monitoring/mod.rs`
- **Configurable SLOs:**

```rust
pub struct ServiceLevelObjectives {
    pub latency_p50_ms: f64,     // e.g., 10.0
    pub latency_p95_ms: f64,     // e.g., 50.0
    pub latency_p99_ms: f64,     // e.g., 200.0
    pub availability_percent: f64, // e.g., 99.9
    pub error_rate_max: f64,      // e.g., 0.001
}
```

- **Integration:** Compare actual `PerformanceMetrics` against SLOs
- **Alert:** Fire alert via existing `AlertManager` when SLO is breached
- **API:** GET `/api/monitoring/slo` — current SLO status and error budget
- **ISO refs:** 25010 (reliability, performance efficiency)

#### 3.6.2 — Fairness Monitoring in Production
- **Extend:** `src/drift/mod.rs` + `src/fairness/mod.rs`
- **Continuous monitoring:**
  - Collect predictions per protected group in a sliding window
  - Compute fairness metrics periodically (e.g., every N predictions)
  - Alert when fairness threshold is breached
- **Integration with existing `FeatureDriftMonitor`:**
  - Add fairness metrics as a drift signal
- **API:** GET `/api/monitoring/fairness` — real-time fairness dashboard data
- **ISO refs:** TR 24027 Clause 8, 42001 Clause 9.1

#### 3.6.3 — Metrics Export (Prometheus/OTLP)
- **Extend:** `src/monitoring/mod.rs`
- **New endpoint:** GET `/metrics` — Prometheus exposition format
- **Metrics to export:**
  - `automl_prediction_latency_seconds` (histogram)
  - `automl_predictions_total` (counter)
  - `automl_prediction_errors_total` (counter)
  - `automl_model_drift_score` (gauge)
  - `automl_fairness_disparate_impact` (gauge)
  - `automl_data_quality_score` (gauge)
- **Optional:** OpenTelemetry integration via `tracing-opentelemetry`
- **ISO refs:** 25010 (operability)

#### 3.6.4 — Compliance Reporting
- **New module:** `src/compliance/mod.rs`
- **Auto-generate compliance status report:**
  - For each ISO standard: list of controls, status (pass/fail/partial), evidence
  - Evidence: link to audit logs, test results, config values
  - Gap summary with recommended actions
- **Output:** JSON + Markdown
- **New API endpoint:** GET `/api/compliance/report` — full compliance report
- **ISO refs:** 42001 Clause 9.2, 27001 Clause 9.2

#### 3.6.5 — Feedback Loop and Retraining Triggers
- **Extend:** `src/server/handlers.rs` + `src/monitoring/mod.rs`
- **Feedback collection:**
  - POST `/api/predict/feedback` — submit actual outcome for a past prediction
  - Store: prediction ID, actual value, timestamp
- **Retraining triggers:**
  - Drift detected (severity >= critical)
  - Fairness violation sustained
  - Prediction accuracy dropped below threshold (from feedback)
  - Manual trigger
- **Integration:** Connect to existing `ExperimentTracker` for retraining lineage
- **ISO refs:** 42001 Clause 10, 5338 Clause 6.4.9

---

## 4. Execution Order and Dependencies

```
Phase 1 (Governance Docs)
    |
    +---> Phase 2 (Data Quality + Fairness)
    |         |
    |         +---> Phase 3 (Transparency)
    |                   |
    |                   +---> Phase 6 (Monitoring + Compliance)
    |
    +---> Phase 5 (Security Hardening)
    |         |
    |         +---> Phase 6 (Monitoring + Compliance)
    |
    +---> Phase 4 (Lifecycle Management)
```

**Recommended order:** 1 → 2 → 3 → 5 → 4 → 6

- Phase 1 is pure documentation, no code risk
- Phase 2 creates fairness module needed by Phase 3 (audit events) and Phase 6 (monitoring)
- Phase 3 depends on Phase 2 (fairness data for audit trail)
- Phase 5 is independent but benefits from Phase 1 threat model
- Phase 4 is independent but lower priority
- Phase 6 integrates everything

---

## 5. New Modules Summary

| Module | Phase | Purpose | Est. LOC |
|--------|-------|---------|----------|
| `src/provenance/mod.rs` | 2 | Data lineage and source tracking | ~400 |
| `src/fairness/mod.rs` | 2 | Fairness metrics, bias detection | ~800 |
| `src/privacy/mod.rs` | 5 | PII detection, anonymization, retention | ~600 |
| `src/compliance/mod.rs` | 6 | Compliance status reporting | ~300 |

**Existing modules to extend:**

| Module | Phases | Changes |
|--------|--------|---------|
| `src/preprocessing/` | 2 | Data quality scoring |
| `src/explainability/` | 3 | Decision path extraction |
| `src/export/` | 2, 3, 4, 5 | Datasheets, model cards, versioning workflow, encryption |
| `src/calibration/` | 3 | Uncertainty estimation |
| `src/security/` | 3, 5 | Audit events, RBAC, tamper-evident logs |
| `src/monitoring/` | 4, 6 | Validation checks, SLOs, Prometheus export |
| `src/tracking/` | 4 | Reproducibility, env fingerprint |
| `src/drift/` | 6 | Fairness-aware drift monitoring |
| `src/server/` | All | New API endpoints (~15 new) |

---

## 6. Standards Cross-Reference Matrix

| Module | 42001 | 23053 | 5338 | 5259 | 24027 | 24028 | 27001 | 27701 | 25010 |
|--------|-------|-------|------|------|-------|-------|-------|-------|-------|
| `preprocessing/` | | X | X | X | X | | | | X |
| `training/` | X | X | X | | X | | | | X |
| `inference/` | | X | X | | | X | | | X |
| `optimizer/` | | X | X | | | | | | X |
| `explainability/` | X | | | | X | X | | | |
| `drift/` | X | | X | | X | | | | X |
| `anomaly/` | | X | | | | | X | | X |
| `calibration/` | | | | | | X | | | X |
| `security/` | X | | | | | | X | | X |
| `server/` | | | X | | | | X | | X |
| `monitoring/` | X | | X | | | | | | X |
| `tracking/` | X | | X | | | X | | | |
| `export/` | | X | X | | | X | | | |
| `provenance/` (new) | | | X | X | | X | | | |
| `fairness/` (new) | X | | | | X | X | | | |
| `privacy/` (new) | X | | | X | | | X | X | |
| `compliance/` (new) | X | | | | | | X | | |

---

## 7. Key Deliverables Checklist

### Documentation (Phase 1)
- [ ] AIMS policy — `docs/plan/governance/aims-policy.md`
- [ ] Risk register — `docs/plan/governance/risk-register.md`
- [ ] Threat model — `docs/plan/governance/threat-model.md`
- [ ] Data classification — `docs/plan/governance/data-classification.md`
- [ ] Terminology glossary — `docs/plan/governance/glossary.md`

### Code — New Modules
- [ ] `src/provenance/mod.rs` — data lineage
- [ ] `src/fairness/mod.rs` — fairness metrics + bias detection
- [ ] `src/privacy/mod.rs` — PII detection + anonymization + retention
- [ ] `src/compliance/mod.rs` — compliance reporting

### Code — Module Extensions
- [ ] `src/preprocessing/` — data quality scoring
- [ ] `src/explainability/` — decision path extraction
- [ ] `src/export/` — model cards, datasheets, encryption, versioning workflow
- [ ] `src/calibration/` + `src/inference/` — prediction uncertainty
- [ ] `src/security/` — RBAC, tamper-evident audit, expanded event types
- [ ] `src/monitoring/` — SLOs, Prometheus export, deployment validation
- [ ] `src/tracking/` — env fingerprint, seed management
- [ ] `src/drift/` — fairness-aware monitoring
- [ ] `src/server/` — ~15 new API endpoints

### New API Endpoints (by phase)
- [ ] GET `/api/data/lineage` (Phase 2)
- [ ] GET `/api/data/quality` (Phase 2)
- [ ] GET `/api/data/datasheet` (Phase 2)
- [ ] POST `/api/fairness/evaluate` (Phase 2)
- [ ] GET `/api/fairness/report/{model_id}` (Phase 2)
- [ ] POST `/api/fairness/bias-scan` (Phase 2)
- [ ] GET `/api/models/{model_id}/card` (Phase 3)
- [ ] PUT `/api/models/{model_id}/card` (Phase 3)
- [ ] POST `/api/explain/decision-path` (Phase 3)
- [ ] GET `/api/audit/predictions` (Phase 3)
- [ ] GET `/api/audit/events` (Phase 3)
- [ ] POST `/api/models/{model_id}/promote` (Phase 4)
- [ ] POST `/api/models/{model_id}/deprecate` (Phase 4)
- [ ] POST `/api/models/{model_id}/rollback` (Phase 4)
- [ ] GET `/api/models/{model_id}/versions` (Phase 4)
- [ ] POST `/api/models/{model_id}/validate` (Phase 4)
- [ ] POST `/api/privacy/scan` (Phase 5)
- [ ] POST `/api/privacy/anonymize` (Phase 5)
- [ ] DELETE `/api/data/{dataset_id}` (Phase 5)
- [ ] GET `/metrics` (Phase 6)
- [ ] GET `/api/monitoring/slo` (Phase 6)
- [ ] GET `/api/monitoring/fairness` (Phase 6)
- [ ] GET `/api/compliance/report` (Phase 6)
- [ ] POST `/api/predict/feedback` (Phase 6)

---

## 8. Notes

- All new code must follow existing patterns: pure Rust, serde-serializable structs,
  comprehensive tests, `tracing` for logging, `thiserror` for error types.
- New modules must be registered in `src/lib.rs` and re-exported in the prelude
  where appropriate.
- Governance documents in `docs/plan/governance/` remain local (excluded via
  `.git/info/exclude`). Production-ready docs should later move to a proper
  docs directory if they need to be shared.
- ISO 42001 and 27001 are the only certifiable standards listed. Certification
  requires external audit — this plan prepares the technical evidence.
- This is a living document. Update task statuses as work progresses.
