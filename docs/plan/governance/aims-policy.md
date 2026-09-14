# AI Management System (AIMS) Policy

**Document ID:** GOV-AIMS-001
**Version:** 1.0
**Effective Date:** 2026-02-11
**Review Date:** 2026-08-11
**Classification:** Internal
**Standard Reference:** ISO/IEC 42001:2023 (Clauses 4, 5, 6)

---

## 1. Purpose

This document establishes the AI Management System (AIMS) policy for AutoML, a pure-Rust automated machine learning framework (v0.5.0). It defines the principles, governance structure, risk appetite, and review cadence that guide the responsible development, deployment, and operation of AI systems built with or managed by AutoML.

---

## 2. Scope

### 2.1 System Description

AutoML is an open-source, high-performance AutoML framework implemented entirely in Rust. The system encompasses:

- **ML Training Pipeline** -- Automated model selection and training across 17+ algorithms via `TrainEngine` and `TrainingConfig` (`src/training/`), including gradient boosting, random forests, SVMs, naive Bayes, neural architectures (TabNet, FT-Transformer), and ensemble methods (voting, stacking, blending).
- **Hyperparameter Optimization** -- Bayesian optimization, Gaussian processes, and ASHT search via `HyperOptX`, `BayesianOptimizer`, `GaussianProcess` (`src/optimizer/`).
- **Inference Engine** -- Batch, streaming, and real-time prediction via `InferenceEngine` with `predict`, `predict_proba`, and `predict_streaming` methods (`src/inference/`).
- **REST API** -- Axum-based HTTP server exposing training, prediction, model management, and monitoring endpoints (`src/server/`).
- **Web UI** -- Browser-based dashboard using htmx and Alpine.js for experiment management, visualization, and model serving configuration.
- **CLI** -- Command-line interface for training, evaluation, export, and serving (`src/cli/`).
- **Data Processing** -- Preprocessing (`DataPreprocessor`), feature engineering (`PolynomialFeatures`, `FeatureInteractions`, `TfidfVectorizer`), imputation (`MICEImputer`, `KNNImputer`), and synthetic data generation (`SMOTE`, `ADASYN`).
- **Model Lifecycle** -- Export to ONNX/PMML (`ONNXExporter`, `PMMLExporter`), versioning via `ModelRegistry` and `ModelVersion`, experiment tracking via `ExperimentTracker`.
- **Monitoring & Drift** -- `DataDriftDetector` (KS test, PSI, JS divergence), `ConceptDriftDetector` (DDM, ADWIN, EDDM, Page-Hinkley), `FeatureDriftMonitor`, `SystemMonitor`, `AlertManager`.
- **Security** -- `SecurityManager`, `ApiKeyVerifier`, `JwtVerifier`, `RateLimiter`, `SecurityMiddleware`, `SecretsManager`, `TlsManager`.

### 2.2 Applicability

This policy applies to:

- All contributors to the AutoML codebase.
- All operators deploying AutoML in production environments.
- All end users interacting with models trained, served, or managed by the framework.
- Third-party integrations consuming the REST API or importing/exporting models.

### 2.3 Exclusions

- Upstream Rust crate dependencies (ndarray, polars, linfa, smartcore, tokio, axum) are governed by their respective maintainers. AutoML tracks their CVEs via supply-chain auditing (see risk register R08).
- User-supplied datasets and external models imported into the system are the responsibility of the data owner, though AutoML provides tooling to support responsible use.

---

## 3. AI Principles

### 3.1 Fairness

AutoML shall not systematically disadvantage any demographic group. Implementations must:

- Provide demographic parity and equalized odds metrics through the `ModelMetrics` framework (`src/training/models.rs`).
- Support stratified cross-validation (`StratifiedKFold` in `CVStrategy`, `src/training/cross_validation.rs`) to ensure representative evaluation.
- Enable bias detection via `PermutationImportance` and `LocalExplainer` (`src/explainability/`) to identify features with disproportionate impact on protected groups.
- Document known limitations of sample datasets and require bias audits before production deployment.

### 3.2 Transparency

All AI decisions made through AutoML shall be explainable and auditable:

- `ExperimentTracker` (`src/tracking/tracker.rs`) records all training runs, hyperparameters, metrics, and artifacts with immutable run IDs and timestamps.
- `PermutationImportance`, `PartialDependence` (PDP/ICE), and `LocalExplainer` (SHAP-like) in `src/explainability/` provide global and local model interpretability.
- `CalibrationMetrics` (ECE, MCE, Brier score) in `src/calibration/metrics.rs` measure prediction reliability.
- The `SecurityManager` maintains an audit log (capped at 10,000 entries via `MAX_AUDIT_LOG_ENTRIES`) recording all API access events.

### 3.3 Accountability

Clear ownership and decision authority shall be maintained at every stage of the AI lifecycle:

- All model versions are tracked with semantic versioning (`ModelVersion`) in `ModelRegistry` (`src/export/versioning.rs`).
- Run status (`Running`, `Finished`, `Failed`, `Killed`) and parameters are persisted in `ExperimentTracker`.
- Deployment decisions require explicit approval through the model review process (see Section 4).
- Incident response follows the escalation path defined in Section 4.

### 3.4 Safety

AutoML shall include safeguards to prevent harm from AI system failures:

- `DriftDetector` trait and implementations (`DataDriftDetector`, `ConceptDriftDetector`, `FeatureDriftMonitor`) in `src/drift/` continuously monitor for data and concept drift with configurable severity levels (0=none, 1=warning, 2=critical).
- `AlertManager` and `AlertCondition` in `src/monitoring/alerts.rs` trigger alerts on threshold breaches.
- `RateLimiter` (`SlidingWindow`, `TokenBucket`, `FixedWindow` algorithms) in `src/security/rate_limiter.rs` prevents abuse and resource exhaustion.
- `BatchProcessor` with `Priority` queues (`src/batch/`) ensures graceful degradation under load.
- `MemoryPool` (`src/memory/`) prevents out-of-memory conditions during training and inference.

### 3.5 Privacy

Personal and sensitive data shall be protected throughout the AI lifecycle:

- `SecretsManager` (`src/security/secrets.rs`) encrypts credentials with HMAC-SHA256 integrity verification, supports automatic rotation detection (90-day policy), and provides secret strength assessment.
- `TlsManager` (`src/security/tls.rs`) enforces encrypted transport for all API communications.
- Data classification (see `data-classification.md`) governs handling of PII, uploaded datasets, and model artifacts.
- No training data is persisted in model artifacts by default; only learned parameters are serialized via `Model::to_bytes()`.

---

## 4. Roles and Responsibilities

### 4.1 AI System Owner

**Accountability:** Overall governance of AutoML as an AI system.

| Responsibility | AutoML Context |
|---|---|
| Approve AIMS policy and amendments | Signs off on this document and all governance updates |
| Set organizational risk appetite | Defines acceptable risk levels in the risk register |
| Authorize production deployments | Approves model promotion through `ModelRegistry` |
| Ensure compliance with applicable regulations | Maps regulatory requirements to AutoML controls |
| Commission periodic AIMS reviews | Triggers semi-annual governance review cycle |

### 4.2 Data Steward

**Accountability:** Integrity, quality, and compliance of all data flowing through AutoML.

| Responsibility | AutoML Context |
|---|---|
| Classify datasets per data classification scheme | Applies labels per `data-classification.md` |
| Approve data ingestion via upload handlers | Reviews datasets before loading via `DataLoader` (`src/utils/data_loader.rs`) |
| Monitor data drift dashboards | Reviews `DriftReport` from `FeatureDriftMonitor` and PSI/KS scores |
| Enforce data retention and deletion policies | Manages experiment artifacts and `ExperimentTracker` storage lifecycle |
| Validate preprocessing pipelines | Audits `DataPreprocessor`, `FeatureSelector`, and `AutoPipeline` configurations |

### 4.3 Model Reviewer

**Accountability:** Technical validation that models meet quality, fairness, and safety standards before deployment.

| Responsibility | AutoML Context |
|---|---|
| Review model metrics before promotion | Evaluates `ModelMetrics` (accuracy, F1, AUC-ROC, R2, RMSE) from `ExperimentTracker` |
| Validate explainability artifacts | Reviews `PermutationImportance`, PDP, and `LocalExplanation` outputs |
| Assess calibration quality | Checks ECE, MCE, Brier score from `CalibrationMetrics` |
| Verify drift monitoring is configured | Confirms `DataDriftDetector` and `ConceptDriftDetector` thresholds |
| Sign off on model version bumps | Approves `ModelVersion` major/minor/patch increments in `ModelRegistry` |
| Conduct fairness audits | Runs stratified evaluations across demographic slices |

### 4.4 Security Officer

**Accountability:** Security posture of AutoML infrastructure and AI-specific attack surfaces.

| Responsibility | AutoML Context |
|---|---|
| Maintain threat model | Updates `threat-model.md` based on changes to API, UI, and training pipeline |
| Manage API key and JWT lifecycle | Operates `ApiKeyVerifier`, `JwtVerifier`, and `SecretsManager` rotation policies |
| Configure rate limiting | Tunes `RateLimitConfig` (requests/window, burst size, algorithm) |
| Review security audit logs | Monitors `AuditEntry` records from `SecurityManager` |
| Manage TLS certificates | Operates `TlsManager` and `CertificateInfo` renewal |
| Conduct penetration testing | Tests threat model mitigations quarterly |
| Respond to security incidents | Executes IP blocking via `SecurityConfig.blocked_ips` and emergency key rotation |

---

## 5. Risk Appetite

### 5.1 Risk Appetite Statement

AutoML operates with a **moderate** overall risk appetite, acknowledging that machine learning systems inherently involve probabilistic outputs and irreducible uncertainty. The following risk appetite levels apply per domain:

| Domain | Appetite | Rationale |
|---|---|---|
| Model accuracy degradation | Moderate | Statistical models have inherent variance; drift detection mitigates silent failure |
| Data privacy breach | Very Low | PII exposure carries regulatory and reputational consequences |
| Security compromise | Very Low | Model theft and API abuse undermine trust in the framework |
| Fairness violation | Low | Biased predictions cause direct harm to affected groups |
| Availability disruption | Moderate | AutoML workloads can tolerate brief outages with retry logic |
| Supply chain compromise | Very Low | Compromised dependencies can affect all downstream users |

### 5.2 Risk Tolerance Thresholds

- **Critical** risks (residual score >= 20): Must be treated within 7 days or system component taken offline.
- **High** risks (residual score 12-19): Must have active treatment plan within 30 days.
- **Medium** risks (residual score 6-11): Must be reviewed quarterly; treatment within 90 days.
- **Low** risks (residual score 1-5): Accepted with monitoring; reviewed semi-annually.

---

## 6. Review Cadence

| Activity | Frequency | Owner | Artifact |
|---|---|---|---|
| AIMS policy review | Semi-annual | AI System Owner | Updated `aims-policy.md` |
| Risk register review | Quarterly | AI System Owner + Security Officer | Updated `risk-register.md` |
| Threat model update | Quarterly or on architecture change | Security Officer | Updated `threat-model.md` |
| Data classification audit | Semi-annual | Data Steward | Updated `data-classification.md` |
| Model fairness audit | Per deployment + quarterly | Model Reviewer | Fairness report artifact in `ExperimentTracker` |
| Drift monitoring review | Monthly | Data Steward + Model Reviewer | `DriftReport` summaries |
| Security penetration test | Quarterly | Security Officer | Penetration test report |
| Dependency audit | Monthly | Security Officer | `cargo audit` report |
| Incident retrospective | Per incident | All roles | Incident report |

---

## 7. ISO 42001 Clause Mapping

| ISO 42001 Clause | Section in This Document |
|---|---|
| **4.1** Context of the organization | Section 2 (Scope) |
| **4.2** Needs and expectations of interested parties | Section 2.2 (Applicability) |
| **4.3** Scope of the AIMS | Section 2.1 (System Description) |
| **4.4** AI Management System | Sections 3-6 (Principles, Roles, Risk, Review) |
| **5.1** Leadership and commitment | Section 4.1 (AI System Owner) |
| **5.2** AI policy | Section 3 (AI Principles) |
| **5.3** Organizational roles, responsibilities, and authorities | Section 4 (Roles and Responsibilities) |
| **6.1** Actions to address risks and opportunities | Section 5 (Risk Appetite), cross-ref `risk-register.md` |
| **6.2** AI objectives and planning to achieve them | Section 6 (Review Cadence) |
| **6.3** Planning of changes | Section 6 (Review Cadence), change triggers |

---

## 8. Related Documents

| Document | Path |
|---|---|
| AI Risk Register | `docs/plan/governance/risk-register.md` |
| Threat Model | `docs/plan/governance/threat-model.md` |
| Data Classification | `docs/plan/governance/data-classification.md` |
| Glossary | `docs/plan/governance/glossary.md` |
| ISO Standards Roadmap | `docs/plan/iso-standards-roadmap.md` |

---

## 9. Approval

| Role | Name | Date | Signature |
|---|---|---|---|
| AI System Owner | _________________ | ____/____/____ | _________________ |
| Security Officer | _________________ | ____/____/____ | _________________ |
| Data Steward | _________________ | ____/____/____ | _________________ |
| Model Reviewer | _________________ | ____/____/____ | _________________ |
