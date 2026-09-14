# Insights Tab — Educational ML Visualizations

**Date:** 2026-03-15
**Status:** Approved

---

## Overview

Add a top-level **Insights** tab to the AutoML frontend, sitting alongside Train / Predict / Analysis. It contains four sub-tabs, each a Canvas 2D visualization connected to the user's live session (trained model + uploaded dataset). The goal is to let users understand *how their model thinks* — not just what it outputs.

No external charting libraries. All rendering uses the native HTML5 Canvas 2D API, consistent with the rest of the frontend.

---

## Architecture

### Frontend

- **Location:** `src/server/handlers.rs` — the embedded HTML/JS/CSS string (~6500 lines). The Insights tab is added as a new top-level tab alongside the existing Train/Predict/Analysis tabs.
- **Tab structure:** Top-level `Insights` tab → 4 sub-tabs inside: Step-Through, Concepts, Playground, Metrics.
- **Rendering:** Native Canvas 2D API. Reuse existing helper functions (`eDrawLine`, `eDrawBars`, `eDrawScatter`, `eDrawDonut`, `eDrawMiniHist`, `eDrawSparkline`, `eDrawRadar`) where applicable. New canvas drawing logic added inline.
- **Data connection:** All visualizations are driven by the current session state. When no model is loaded, each sub-tab shows a clear "Train a model first" placeholder.
- **JS state:** The frontend uses `eLastModelId` (most recent model ID string), `eData` (dataset info), and `eTraining` (boolean). These are the actual globals to hook into.
- **model_id routing:** Both new GET endpoints accept a `?model_id=` query parameter. The frontend passes `eLastModelId` as this parameter.

### Backend (Rust / Axum)

Two new API endpoints:

| Endpoint | Method | Purpose |
|---|---|---|
| `GET /api/insights/model-structure?model_id=` | GET | Returns serialized tree nodes or per-epoch training history |
| `GET /api/insights/evaluation?model_id=` | GET | Returns confusion matrix + ROC/PR data (classification) or residuals (regression) |

Existing endpoints reused:
- POST `/api/predict` — Playground sub-tab
- POST `/api/explain/importance` (body: `{ "model_id": "..." }`) — Concepts feature importance panel
- POST `/api/explain/local` (body: `{ "model_id": "...", "instance": [f64, ...] }`) — Playground attribution arrows
- GET `/api/data/analyze` — Playground slider min/max/distinct values (returns `column_stats` with `min`, `max`, `top_categories`)

Note: existing `/api/visualization/umap` and `/api/visualization/pca` are POST (not GET), consistent with the router in `src/server/api.rs`.

### Backend state extensions required

Three non-trivial extensions to the Rust layer are needed before the new endpoints can be implemented:

**1. Tree node serialization (`src/training/`)**

`TreeNode` in `src/training/decision_tree.rs` already derives `Serialize`/`Deserialize`. A new method must be added to `TrainEngine`:

```rust
pub fn tree_nodes(&self) -> Option<Vec<SerializedNode>>
```

This performs a breadth-first walk of the trained `TreeNode` structure and returns a flat list of `SerializedNode` structs:

```rust
pub struct SerializedNode {
    pub id: usize,
    pub parent_id: Option<usize>,
    pub feature_name: Option<String>,  // None for leaf nodes
    pub threshold: Option<f64>,        // None for leaf nodes
    pub gini: f64,
    pub samples: usize,
    pub is_leaf: bool,
    pub is_classification: bool,
    pub predicted_class: Option<String>,  // leaf classification nodes
    pub predicted_value: Option<f64>,     // leaf regression nodes
    pub left_child: Option<usize>,
    pub right_child: Option<usize>,
}
```

`feature_name` is resolved from the engine's stored `feature_names` vec. For Random Forest, expose only `estimators[0]`; response includes `total_trees` and `shown_tree_index: 0`.

**2. Per-epoch training history (`epoch_history` field on `TrainEngine`)**

The new field is named `epoch_history` — distinct from the existing `training_history: Vec<ModelComparison>` which stores cross-model comparison data and must not be changed.

```rust
pub epoch_history: Vec<EpochRecord>

pub struct EpochRecord {
    pub epoch: usize,
    pub train_loss: f64,
    pub val_loss: Option<f64>,
    pub train_acc: Option<f64>,
    pub val_acc: Option<f64>,
}
```

**Which model types produce epoch history:**
- `NeuralNetwork`: yes — has an explicit epoch loop
- `SGD` / linear models with iterative solvers: yes — instrument the iteration loop

**Which model types do NOT produce epoch history (no training loop to instrument):**
- `DecisionTree`, `RandomForest`, `KNN`, `NaiveBayes`, `SVM`, `LinearRegression`, `LogisticRegression` (closed-form / single-pass — structurally cannot produce epoch history)
- `GradientBoosting`, `XGBoost`, `LightGBM`, `CatBoost` (iterative round-by-round, but per-round history is deferred to future work — show "This model's iteration history is not yet visualized" message, no retrain prompt)

For these, `epoch_history` remains empty and the Step-Through sub-tab shows the tree animation (for tree types) or the message *"This model type doesn't have a training loop to animate — it learns in a single pass."* (not "retrain", which would be misleading).

If a NeuralNetwork/SGD model was trained before `epoch_history` was added, `epoch_history` is empty and the message reads *"No epoch history recorded — retrain to see the animation."*

**3. `InsightsEvaluation` cache with coordinated eviction**

New field on `AppState`:

```rust
pub insights_cache: DashMap<String, InsightsEvaluation>
```

```rust
pub struct InsightsEvaluation {
    pub task: TaskType,                    // Classification | Regression
    pub classes: Vec<String>,             // classification only
    pub y_true: Vec<f64>,
    pub y_pred: Vec<f64>,                 // class index (classification) or value (regression)
    pub y_prob: Option<Vec<Vec<f64>>>,    // per-class probabilities (classification)
}
```

The `/api/insights/evaluation` handler derives confusion matrix, ROC points (100 evenly-spaced threshold values from 0.0 to 1.0 — sufficient granularity for meaningful click-to-set interaction), PR points (same 100-threshold grid), and residuals from these stored predictions at request time.

**Eviction:** `insights_cache` is evicted in lockstep with `train_engines` inside `evict_stale_entries()`. When a `train_engines` entry is removed, the corresponding `insights_cache` entry is removed in the same call. `MAX_INSIGHTS_CACHE` matches `MAX_TRAIN_ENGINES` (50).

If no entry exists, the endpoint returns `{ "error": "evaluation_data_unavailable" }` and the Metrics sub-tab shows *"Retrain your model to unlock this view."*

---

## Sub-tab 1 — Step-Through (`▶ How It Learned`)

### Purpose
Animate the internals of the trained model — show the user *how* it was built, one step at a time.

### Behaviour
- **Decision Tree / Random Forest:** Renders the tree on Canvas as a node-link diagram (root at top, leaves at bottom). Steps through splits one by one:
  - Highlights the active node in blue
  - Shows the split condition (`petal_length ≤ 2.45`), Gini impurity before/after, and sample counts left/right
  - Sidebar annotation explains the split in plain English: *"The model chose this feature because splitting here reduced uncertainty the most"*
  - For Random Forest: displays tree #1 (index 0) from the ensemble with a label "Showing tree 1 of N"
- **NeuralNetwork / iterative SGD models:** Renders the per-epoch loss/accuracy curve from `epoch_history`, animating one epoch at a time. Annotations at inflection points (*"Validation loss stopped improving — early stop triggered"*). If `epoch_history` is empty on a retrain-eligible model, shows: *"No epoch history recorded — retrain to see the animation."*
- **Single-pass models (KNN, NaiveBayes, SVM, linear, tree-based ensembles without epochs):** Shows: *"This model type learns in a single pass — there is no training loop to animate."* (no retrain prompt — it can never produce epoch history).
- **Controls:** Play / Pause / Step Back / Step Forward / Reset. Speed slider (slow / normal / fast).

### Data source
`GET /api/insights/model-structure?model_id=<eLastModelId>`

### Canvas layout
- Left ~70%: animated diagram
- Right ~30%: annotation panel (plain-text callouts for current step)

---

## Sub-tab 2 — Concept Explainer (`📐 Understand the Model`)

### Purpose
Three annotated diagrams drawn from real session data that explain foundational ML concepts in context of the user's model.

### Panels (stacked vertically, each with a heading and plain-English caption)

**1. Bias-Variance (Training Curve)**
- Line chart: training accuracy vs validation accuracy per epoch (from `model-structure` response `epoch_history`)
- Overfitting zone shaded red if validation diverges from training by >5%
- Underfitting zone shaded yellow if both curves plateau below 70%
- Falls back to a static annotated diagram with generic curves if `epoch_history` is empty or model type doesn't produce it
- Labels: *"Your model fits the training data well but struggles on unseen data"* or *"Good balance"* depending on curve shape

**2. Feature Importance**
- Horizontal bar chart (POST to `/api/explain/importance` with body `{ "model_id": eLastModelId }`)
- Top 10 features, sorted descending
- Bars colour-coded by magnitude (blue gradient: dark = most important)
- Plain-English caption: *"petal_length drove predictions 3× more than the next feature"*

**3. Data Split Breakdown**
- Visual stacked bar showing train / validation / test row counts from `eData` session state
- Colour-coded segments (blue = train, yellow = val, green = test)
- Caption: *"Your model trained on N rows and was evaluated on M rows it had never seen"*

---

## Sub-tab 3 — Live Playground (`🎛 What If?`)

### Purpose
Feature sliders tied to the real model. Every change fires a prediction and updates the display live.

### Behaviour
- **Slider metadata:** On tab load, GET `/api/data/analyze` (no body required — uses current dataset in session). Response `column_stats` provides `min`, `max`, and `top_categories` per column. The target column (identified from `eLastModelId`'s `ModelInfo.task_type` and the dataset's label) is excluded from sliders. Numeric columns → `<input type="range">` with `min`/`max`/`step` from `column_stats`. Categorical/string columns → `<select>` dropdown populated from `top_categories`.
- On every input change: POST `/api/predict` with `{ model_id: eLastModelId, features: { ... } }`, debounced 150ms
- Updates:
  - Large prediction label with confidence badge (*"Iris-virginica — 92%"*)
  - Horizontal confidence bar per class on Canvas
  - *"What pushed the prediction"* panel: POST `/api/explain/local` with `{ "model_id": eLastModelId, "instance": [f64, ...] }` (ordered by feature index). Response returns per-feature attribution values. Top 3 by absolute magnitude rendered as directional arrows on Canvas — direction and magnitude reflect the local explanation, not global importance. Labelled: *"Local explanation — shows how this specific input affected the prediction."*
- Debounced 150ms on predict; local explanation called after predict resolves (sequential, not parallel)

### Canvas layout
- Top section: prediction output (label + confidence bars)
- Bottom section: feature attribution arrows

### Data sources
POST `/api/predict`, POST `/api/explain/local`, GET `/api/data/analyze` (once on tab load), `eLastModelId` for model identification.

---

## Sub-tab 4 — Metrics Deep Dive (`📊 How Well Did It Do?`)

### Purpose
Visual explanation of model evaluation metrics using stored test-split predictions.

### Classification models

**Confusion Matrix**
- Grid on Canvas, colour-coded: TP (green), TN (blue), FP (orange), FN (red)
- Cells fill in one prediction at a time on load animation
- Click a cell: tooltip explaining the cell (*"3 samples were Setosa but predicted as Virginica"*)

**ROC Curve**
- Drawn incrementally on Canvas, AUC area shaded
- Click-to-set threshold (not drag): user clicks a point on the curve to set the threshold; the confusion matrix re-animates to reflect the new threshold. Hit-testing snaps to the nearest pre-computed `roc_points` entry.
- Diagonal random-classifier baseline drawn in grey

**Precision-Recall Curve**
- Drawn alongside ROC in a split layout
- Current threshold highlighted with a labelled dot

### Regression models

**Residual Plot**
- Scatter: actual vs predicted on Canvas, colour-coded by residual magnitude (green = small error, red = large)
- Perfect prediction line (y=x) in grey

**Error Distribution**
- Histogram of residuals on Canvas

**Summary metrics** (R², MAE, RMSE) as stat cards with plain-English size comparison: *"On average, predictions are off by ±0.3 — about 4% of the target range"*.

### Error state
If `insights_cache` has no entry for the model (trained before state extension): *"Retrain your model to unlock this view."*

### Data source
`GET /api/insights/evaluation?model_id=<eLastModelId>` — derives confusion matrix, ROC points, PR points, and residuals at request time from `InsightsEvaluation` stored in `AppState::insights_cache`.

---

## New API Endpoints

### `GET /api/insights/model-structure?model_id=<id>`

**Response (decision tree):**
```json
{
  "type": "tree",
  "algorithm": "DecisionTreeClassifier",
  "total_trees": 1,
  "shown_tree_index": 0,
  "nodes": [
    {
      "id": 0,
      "parent_id": null,
      "feature": "petal_length",
      "threshold": 2.45,
      "gini": 0.667,
      "samples": 120,
      "left_child": 1,
      "right_child": 2,
      "is_leaf": false,
      "predicted_class": null
    }
  ],
  "training_history": {
    "train_accuracy": [0.6, 0.75, 0.88, 0.92],
    "val_accuracy": [0.58, 0.72, 0.85, 0.89]
  }
}
```

**Response (non-tree):**
```json
{
  "type": "convergence",
  "algorithm": "LogisticRegression",
  "nodes": null,
  "training_history": {
    "train_loss": [0.9, 0.6, 0.4, 0.3],
    "val_loss": [0.95, 0.65, 0.45, 0.35],
    "train_accuracy": [0.6, 0.75, 0.88, 0.92],
    "val_accuracy": [0.58, 0.72, 0.85, 0.89]
  }
}
```

`training_history` is `null` for models trained before per-epoch logging was added.

### `GET /api/insights/evaluation?model_id=<id>`

**Response (classification):**
```json
{
  "task": "classification",
  "classes": ["Iris-setosa", "Iris-versicolor", "Iris-virginica"],
  "confusion_matrix": [[13, 0, 0], [0, 10, 1], [0, 2, 14]],
  "roc_points": [{"threshold": 0.9, "fpr": 0.0, "tpr": 0.0}, ...],
  "pr_points": [{"threshold": 0.9, "precision": 1.0, "recall": 0.0}, ...],
  "auc": 0.972,
  "accuracy": 0.925,
  "per_class": [{"class": "Iris-setosa", "precision": 1.0, "recall": 1.0, "f1": 1.0}]
}
```

**Response (regression):**
```json
{
  "task": "regression",
  "residuals": [{"actual": 1.2, "predicted": 1.1}, ...],
  "r2": 0.94,
  "mae": 0.31,
  "rmse": 0.45,
  "target_range": [0.1, 7.9]
}
```

**Error response (no cached data):**
```json
{ "error": "evaluation_data_unavailable" }
```

---

## Error States

- **No model loaded:** Each sub-tab shows a centred placeholder: *"Train a model first to unlock this visualization"*
- **API error:** Toast notification + canvas shows error text inline
- **No training history (old model):** Step-Through and Concepts training curve show *"Retrain your model to see this animation"*
- **No evaluation cache (old model):** Metrics sub-tab shows *"Retrain your model to unlock this view"*

---

## Implementation Constraints

- All Canvas rendering: native Canvas 2D API only, no new JS dependencies
- Animations use `requestAnimationFrame`
- ROC threshold interaction: click-to-set (snap to nearest point), not continuous drag
- The frontend HTML string in `handlers.rs` grows by approximately 800–1200 lines
- Rust backend requires three state extensions (tree export method, per-epoch history accumulation, `InsightsEvaluation` cache) before the new endpoints can be wired up

---

## Out of Scope

- Exporting visualizations as images
- Side-by-side model comparison
- Exact SHAP values (attribution approximated from feature importance × input deviation)
- Multi-tree navigation for Random Forest (always shows tree index 0)
