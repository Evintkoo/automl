# Pipeline Quality Frontend Visibility — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Surface the QualityReport data (gate sub-scores, findings, calibration, OOD) in the Train tab and Insights Playground so users can see exactly what the quality pipeline did and why.

**Architecture:** Add `GateScores` to `QualityReport` so sub-scores are in the JSON; add a quality panel below each training result area (single, auto-tune, automl) that fetches `/api/quality/report/:model_id` on completion; add an OOD warning banner in the Insights Playground prediction output.

**Tech Stack:** Rust (serde), inline HTML/CSS/JS in `src/server/handlers.rs`, Canvas API for Pareto scatter plot.

---

## File Map

| File | Change |
|---|---|
| `src/quality/mod.rs` | Add `GateScores` struct + field to `QualityReport`; populate in `from_context` |
| `src/server/handlers.rs` | Add CSS (~80 lines), HTML placeholders (4 divs), JS functions (~200 lines), hook into 3 poll completions, OOD banner logic |

**Security note:** Several quality report fields (column names, findings messages, model IDs) can originate from user-uploaded CSV data. All such values MUST be passed through an `escHtml` helper before being placed into innerHTML. The plan includes this helper and marks every user-derived string call-site.

---

## Task 1: Add GateScores to QualityReport

**Files:**
- Modify: `src/quality/mod.rs`

The four sub-scores (pre, training, hyperopt, post_training) are computed transiently in `from_context` but not persisted. Without them in the JSON, the frontend cannot display per-gate scores. This task adds them.

- [ ] **Step 1: Write the failing test**

Add this test at the bottom of the `#[cfg(test)]` block in `src/quality/mod.rs` (after the last test):

```rust
#[test]
fn test_report_gate_scores_populated() {
    let ctx = QualityContext {
        hyperopt_converged: true,
        ensemble_beat_single: true,
        ece_before: Some(0.20),
        ece_after: Some(0.05),
        ..Default::default()
    };
    let report = QualityReport::from_context("m1".into(), &ctx);
    // pre_score=1.0 (no critical findings)
    assert!((report.gate_scores.pre - 1.0).abs() < 1e-9);
    // training_score=1.0 (ensemble beat single)
    assert!((report.gate_scores.training - 1.0).abs() < 1e-9);
    // hyperopt_score=1.0 (converged)
    assert!((report.gate_scores.hyperopt - 1.0).abs() < 1e-9);
    // post_score=(0.20-0.05)/0.20=0.75
    assert!((report.gate_scores.post_training - 0.75).abs() < 1e-9);
}
```

- [ ] **Step 2: Run the test to verify it fails**

```bash
cargo test -p automl test_report_gate_scores_populated 2>&1 | tail -20
```

Expected: compile error — `gate_scores` field does not exist on `QualityReport`.

- [ ] **Step 3: Add the GateScores struct**

In `src/quality/mod.rs`, add the struct after the `PostTrainingReport` struct (before `/// Full quality report`):

```rust
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct GateScores {
    pub pre: f64,
    pub training: f64,
    pub hyperopt: f64,
    pub post_training: f64,
}
```

- [ ] **Step 4: Add gate_scores field to QualityReport**

Add `pub gate_scores: GateScores,` to the `QualityReport` struct:

```rust
/// Full quality report returned alongside ModelMetrics.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct QualityReport {
    pub model_id: String,
    pub overall_quality_score: f64,
    pub gate_scores: GateScores,          // ← add this line
    pub pre_training: PreTrainingReport,
    pub training: TrainingReport,
    pub hyperopt: HyperoptReport,
    pub post_training: PostTrainingReport,
}
```

- [ ] **Step 5: Populate gate_scores in from_context**

In `QualityReport::from_context`, the four sub-scores are already computed as `pre_score`, `training_score`, `hyperopt_score`, `post_score`. Add `gate_scores` to the returned struct. Find the `QualityReport {` construction block and add the field after `overall_quality_score: overall,`:

```rust
QualityReport {
    model_id,
    overall_quality_score: overall,
    gate_scores: GateScores {
        pre: pre_score,
        training: training_score,
        hyperopt: hyperopt_score,
        post_training: post_score,
    },
    pre_training: PreTrainingReport { /* keep existing */ },
    // ... rest unchanged
}
```

(Keep all existing field assignments exactly as-is — only add `gate_scores`.)

- [ ] **Step 6: Run all quality tests**

```bash
cargo test -p automl quality 2>&1 | tail -30
```

Expected: all tests pass including `test_report_gate_scores_populated`.

- [ ] **Step 7: Commit**

```bash
git add src/quality/mod.rs
git commit -m "feat(quality): add GateScores to QualityReport for frontend sub-score display"
```

---

## Task 2: Add HTML scaffolding and CSS

**Files:**
- Modify: `src/server/handlers.rs`

Add quality panel placeholder divs (hidden by default) in each training result area, the OOD warning banner in the Playground, and all CSS needed for Task 3.

- [ ] **Step 1: Add CSS for quality panel**

Find the end of the CSS section in `src/server/handlers.rs` (the line `        /* Config summary */`). Add the following CSS **before** `    </style>`:

```css
        /* ── Quality panel ────────────────────────────────────────────────────── */
        .qp-wrap{margin-top:16px;background:#0d0e0f;border:1px solid #1e2a3a;border-radius:12px;padding:20px}
        .qp-header{display:flex;justify-content:space-between;align-items:center;margin-bottom:16px}
        .qp-title{font-size:15px;font-weight:700;color:#e5e7eb}
        .qp-overall{font-size:22px;font-weight:800;padding:4px 14px;border-radius:20px;color:#fff}
        .qp-overall.green{background:#16a34a}.qp-overall.yellow{background:#ca8a04}.qp-overall.red{background:#dc2626}
        .qp-grid{display:grid;grid-template-columns:1fr 1fr;gap:12px}
        .qp-card{background:#111827;border:1px solid #1e2a3a;border-radius:10px;padding:14px}
        .qp-card-header{display:flex;justify-content:space-between;align-items:center;margin-bottom:8px;cursor:pointer;user-select:none}
        .qp-card-title{font-size:13px;font-weight:600;color:#d1d5db}
        .qp-sub{font-size:12px;font-weight:700;padding:2px 9px;border-radius:10px;color:#fff}
        .qp-sub.green{background:#16a34a}.qp-sub.yellow{background:#ca8a04}.qp-sub.red{background:#dc2626}
        .qp-chevron{font-size:12px;color:#6b7280;transition:transform .2s}
        .qp-chevron.open{transform:rotate(180deg)}
        .qp-body{font-size:12px;color:#9ca3af;display:flex;flex-direction:column;gap:6px}
        .qp-row{display:flex;gap:6px;align-items:flex-start}
        .qp-sev-c{color:#ef4444}.qp-sev-w{color:#f59e0b}.qp-sev-i{color:#60a5fa}
        .qp-tag{display:inline-flex;align-items:center;padding:1px 7px;border-radius:8px;font-size:11px;background:#1e2a3a;color:#93c5fd;margin:1px}
        .qp-label{font-size:11px;font-weight:600;color:#6b7280;text-transform:uppercase;letter-spacing:.5px;margin-top:4px}
        .qp-skeleton{background:linear-gradient(90deg,#1a1b1c 25%,#232425 50%,#1a1b1c 75%);background-size:400% 100%;animation:qp-shimmer 1.4s infinite;border-radius:8px;height:120px}
        @keyframes qp-shimmer{0%{background-position:100% 0}100%{background-position:-100% 0}}
        .qp-pending{text-align:center;padding:24px;color:#6b7280;font-size:13px}
        /* OOD banner */
        .ood-banner{background:#78350f20;border:1px solid #d97706;border-radius:8px;padding:10px 14px;display:flex;justify-content:space-between;align-items:flex-start;gap:8px;margin-bottom:10px}
        .ood-banner-text{font-size:13px;color:#fbbf24;line-height:1.5}
        .ood-banner-close{background:none;border:none;color:#6b7280;cursor:pointer;font-size:16px;padding:0;line-height:1}
```

- [ ] **Step 2: Add quality panel div after single-model e-result**

Find in `src/server/handlers.rs`:
```html
                                <div id="e-result"></div>
                                <div id="e-train-viz"
```

Replace with:
```html
                                <div id="e-result"></div>
                                <div id="e-quality-panel" style="display:none"></div>
                                <div id="e-train-viz"
```

- [ ] **Step 3: Add quality panel div after Auto-Tune result**

Find:
```html
                                <div id="e-at-result" class="empty">Load data and click Start Auto-Tune to begin</div>
```

Replace with:
```html
                                <div id="e-at-result" class="empty">Load data and click Start Auto-Tune to begin</div>
                                <div id="e-quality-panel-at" style="display:none"></div>
```

- [ ] **Step 4: Add quality panel div after AutoML leaderboard**

Find:
```html
                                    <div id="e-aml-leaderboard"></div>
```

Replace with:
```html
                                    <div id="e-aml-leaderboard"></div>
                                    <div id="e-quality-panel-aml" style="display:none"></div>
```

- [ ] **Step 5: Add OOD warning banner in Insights Playground**

Find (around line 2164):
```html
                            <div style="display:flex;gap:16px;padding:16px 0">
                                <div id="e-ins-pg-sliders"
```

Replace with:
```html
                            <div id="e-ins-pg-ood-warn" class="ood-banner" style="display:none"><span class="ood-banner-text">&#9888; This input may be out of distribution. The model was trained on different data patterns &mdash; prediction confidence may be lower than expected.</span><button class="ood-banner-close" onclick="this.parentElement.style.display=\'none\'">&#215;</button></div>
                            <div style="display:flex;gap:16px;padding:16px 0">
                                <div id="e-ins-pg-sliders"
```

- [ ] **Step 6: Build to confirm HTML/CSS is valid**

```bash
cargo build 2>&1 | grep -E "^error" | head -20
```

Expected: no errors.

- [ ] **Step 7: Commit**

```bash
git add src/server/handlers.rs
git commit -m "feat(frontend): add quality panel scaffolding and OOD warning banner HTML/CSS"
```

---

## Task 3: Add quality JS and wire into training completion handlers

**Files:**
- Modify: `src/server/handlers.rs`

Add all JavaScript for the quality panel: XSS-safe HTML builder, fetch, render, gate cards, pareto scatter, OOD banner. Hook into the three poll completion handlers.

- [ ] **Step 1: Add quality JS functions before closing </script>**

Find `    </script>` (the closing script tag just before `</body>`, around line 3965). Add the following JS **before** it:

```javascript
    // ── Pipeline Quality Panel ────────────────────────────────────────────────
    function escHtml(s){var d=document.createElement('div');d.textContent=String(s==null?'':s);return d.innerHTML}
    function eQualityColor(score){return score>=80?'green':score>=60?'yellow':'red'}
    function eQualityBadgeHtml(score){var s=Math.round(score);return '<span class="qp-overall '+eQualityColor(s)+'">'+s+'</span>'}
    function eQualitySubHtml(score){var s=Math.round(score*100);return '<span class="qp-sub '+eQualityColor(s)+'">'+s+'</span>'}
    function eQualityFetch(modelId,panelId){
      var panel=document.getElementById(panelId);if(!panel)return;
      panel.style.display='block';
      panel.innerHTML='<div class="qp-wrap"><div class="qp-skeleton"></div></div>';
      fetch('/api/quality/report/'+encodeURIComponent(modelId))
        .then(function(r){return r.json()})
        .then(function(d){
          if(d.error){panel.innerHTML='<div class="qp-wrap"><div class="qp-pending">Quality analysis not available for this model.</div></div>';return}
          eQualityRender(d,panel);
        })
        .catch(function(){panel.innerHTML='<div class="qp-wrap"><div class="qp-pending">Could not load quality report.</div></div>'})
    }
    function eQualityToggle(btn){
      var body=btn.parentElement.parentElement.querySelector('.qp-body');
      var chev=btn.querySelector('.qp-chevron');
      if(!body)return;
      var hidden=body.style.display==='none';
      body.style.display=hidden?'flex':'none';
      if(chev)chev.className='qp-chevron'+(hidden?' open':'');
    }
    function eQualityRender(r,panel){
      var gs=r.gate_scores||{pre:0,training:0,hyperopt:0,post_training:0};
      var safeId=escHtml(r.model_id).replace(/[^a-z0-9]/gi,'_');
      var h='<div class="qp-wrap">';
      h+='<div class="qp-header"><div class="qp-title">&#128202; Pipeline Quality Report <span style="font-size:12px;font-weight:400;color:#6b7280">'+escHtml(r.model_id)+'</span></div>'+eQualityBadgeHtml(r.overall_quality_score*100)+'</div>';
      h+='<div class="qp-grid">';
      h+=eQualityPreCard(r.pre_training,gs.pre);
      h+=eQualityTrainCard(r.training,gs.training);
      h+=eQualityHyperoptCard(r.hyperopt,gs.hyperopt,safeId);
      h+=eQualityPostCard(r.post_training,gs.post_training);
      h+='</div></div>';
      panel.innerHTML=h;
      var pf=r.hyperopt&&r.hyperopt.pareto_front;
      if(pf&&pf.length>=2){eQualityPareto('qp-pareto-'+safeId,pf)}
    }
    function eQualityCardHeader(title,subscore){
      return '<div class="qp-card-header" onclick="eQualityToggle(this)"><div style="display:flex;align-items:center;gap:8px"><span class="qp-card-title">'+escHtml(title)+'</span>'+eQualitySubHtml(subscore)+'</div><span class="qp-chevron open">&#9660;</span></div>';
    }
    function eQualityPreCard(pt,score){
      var h='<div class="qp-card">'+eQualityCardHeader('Pre-training',score);
      h+='<div class="qp-body" style="display:flex">';
      var findings=pt&&pt.findings||[];
      if(findings.length===0){h+='<span style="color:#4b5563;font-style:italic">No findings</span>'}
      else{findings.forEach(function(f){
        var icon=f.severity==='Critical'?'<span class="qp-sev-c">&#9679;</span>':f.severity==='Warning'?'<span class="qp-sev-w">&#9651;</span>':'<span class="qp-sev-i">&#9675;</span>';
        var col=f.column?' <span class="qp-tag">'+escHtml(f.column)+'</span>':'';
        h+='<div class="qp-row">'+icon+'<span>'+escHtml(f.message)+col+'</span></div>';
      })}
      var dropped=pt&&pt.dropped_features||[];
      if(dropped.length>0){h+='<div class="qp-label">Dropped features</div><div>'+dropped.map(function(f){return'<span class="qp-tag">'+escHtml(f)+'</span>'}).join('')+'</div>'}
      var leakage=pt&&pt.leakage_suspects||[];
      if(leakage.length>0){h+='<div class="qp-label">Leakage suspects</div><div>'+leakage.map(function(f){return'<span class="qp-tag" style="background:#3b1515;color:#fca5a5">'+escHtml(f)+'</span>'}).join('')+'</div>'}
      var hicol=pt&&pt.high_cardinality_columns||[];
      if(hicol.length>0){h+='<div class="qp-label">High cardinality</div><div>'+hicol.map(function(f){return'<span class="qp-tag">'+escHtml(f)+'</span>'}).join('')+'</div>'}
      h+='</div></div>';return h;
    }
    function eQualityTrainCard(tr,score){
      var h='<div class="qp-card">'+eQualityCardHeader('Training',score);
      h+='<div class="qp-body" style="display:flex">';
      h+='<div class="qp-row"><span class="qp-label" style="margin:0;margin-right:6px">CV</span><span>'+escHtml(tr&&tr.cv_strategy||'—')+'</span></div>';
      var excl=tr&&tr.algorithms_excluded||[];
      if(excl.length>0){h+='<div class="qp-label">Algorithms excluded</div>';excl.forEach(function(e){h+='<div class="qp-row"><span style="color:#f59e0b">&#9888;</span><span>'+escHtml(String(e).replace(/^EXCLUDED\s+/,''))+'</span></div>'})}
      var ensResult=(tr&&tr.ensemble_beat_single)?'<span style="color:#22c55e">&#10003; Ensemble beat single model</span>':'<span style="color:#6b7280">Ensemble did not improve on best single model</span>';
      h+='<div class="qp-row">'+ensResult+'</div>';
      h+='</div></div>';return h;
    }
    function eQualityHyperoptCard(ho,score,safeId){
      var h='<div class="qp-card">'+eQualityCardHeader('Hyperopt',score);
      h+='<div class="qp-body" style="display:flex">';
      var trials=ho&&ho.trials_run!=null?ho.trials_run:0;
      h+='<div class="qp-row"><span>'+escHtml(trials)+' trials</span></div>';
      h+='<div class="qp-row">'+(ho&&ho.converged?'<span style="color:#22c55e">&#10003; Converged</span>':'<span style="color:#f59e0b">&#9651; Did not converge</span>')+'</div>';
      var pf=ho&&ho.pareto_front||[];
      if(pf.length>=2){h+='<canvas id="qp-pareto-'+safeId+'" width="200" height="120" style="border-radius:6px;background:#0d0e0f;margin-top:4px"></canvas>'}
      h+='</div></div>';return h;
    }
    function eQualityPostCard(po,score){
      var h='<div class="qp-card">'+eQualityCardHeader('Post-training',score);
      h+='<div class="qp-body" style="display:flex">';
      if(po&&po.calibration_method){h+='<div class="qp-row"><span class="qp-label" style="margin:0;margin-right:6px">Cal</span><span>'+escHtml(po.calibration_method)+'</span></div>'}
      if(po&&po.ece_before!=null&&po.ece_after!=null){h+='<div class="qp-row"><span class="qp-label" style="margin:0;margin-right:6px">ECE</span><span>'+po.ece_before.toFixed(3)+' &#8594; <span style="color:#22c55e">'+po.ece_after.toFixed(3)+'</span></span></div>'}
      if(po&&po.ood_threshold!=null){h+='<div class="qp-row" title="Inputs with normalised distance above this threshold are flagged as out-of-distribution"><span class="qp-label" style="margin:0;margin-right:6px">OOD</span><span>threshold '+po.ood_threshold.toFixed(3)+'</span></div>'}
      if(po&&po.calibration_in_set_fraction!=null){h+='<div class="qp-row"><span class="qp-label" style="margin:0;margin-right:6px">Cal set</span><span>'+(po.calibration_in_set_fraction*100).toFixed(1)+'%</span></div>'}
      if(!po||(!po.calibration_method&&po.ece_before==null)){h+='<span style="color:#4b5563;font-style:italic">Calibration not run</span>'}
      h+='</div></div>';return h;
    }
    function eQualityPareto(canvasId,pf){
      var canvas=document.getElementById(canvasId);if(!canvas)return;
      var ctx=canvas.getContext('2d');var W=canvas.width,H=canvas.height;
      ctx.clearRect(0,0,W,H);
      var metrics=pf.map(function(p){return p.metric_score});
      var latencies=pf.map(function(p){return p.latency_ms});
      var minM=Math.min.apply(null,metrics),maxM=Math.max.apply(null,metrics)||1;
      var minL=Math.min.apply(null,latencies),maxL=Math.max.apply(null,latencies)||1;
      var pad=18;
      ctx.fillStyle='#4b5563';ctx.font='9px sans-serif';ctx.textAlign='center';
      ctx.fillText('metric score',W/2,H-2);
      ctx.save();ctx.translate(9,H/2);ctx.rotate(-Math.PI/2);ctx.fillText('latency ms',0,0);ctx.restore();
      pf.forEach(function(p){
        var x=pad+(p.metric_score-minM)/(maxM-minM||1)*(W-2*pad);
        var y=(H-pad)-(p.latency_ms-minL)/(maxL-minL||1)*(H-2*pad);
        ctx.beginPath();ctx.arc(x,y,4,0,2*Math.PI);
        ctx.fillStyle='#3b82f6';ctx.fill();
        ctx.fillStyle='#9ca3af';ctx.font='8px sans-serif';ctx.textAlign='center';
        ctx.fillText('#'+escHtml(p.trial_id),x,y-7);
      });
    }
```

- [ ] **Step 2: Hook eQualityFetch into ePoll completion (single model)**

In `src/server/handlers.rs`, find this exact string inside the minified `ePoll` function (line ~2362):

```
eLastModelId=models[models.length-1].id;$('e-explain-btn').disabled=false;var safeId
```

Replace with:

```
eLastModelId=models[models.length-1].id;$('e-explain-btn').disabled=false;eQualityFetch(eLastModelId,'e-quality-panel');var safeId
```

- [ ] **Step 3: Hook eQualityFetch into eAtPoll completion (auto-tune)**

In `eAtPoll`, find the end of the `s.Completed` branch. Locate:

```
eDashModels()}else if(s.Failed){$('e-at-progress').style.display
```

Replace with:

```
eDashModels();fetch('/api/models').then(function(r){return r.json()}).then(function(md){var mds=md.models||[];if(mds.length>0)eQualityFetch(mds[mds.length-1].id,'e-quality-panel-at')}).catch(function(){})}else if(s.Failed){$('e-at-progress').style.display
```

- [ ] **Step 4: Hook eQualityFetch into eAMLPoll completion (automl)**

In `eAMLPoll`, find the end of the `d.status==='completed'` branch. Locate:

```
eDashModels()}else if(d.status==='failed')
```

Replace with:

```
eDashModels();fetch('/api/models').then(function(r){return r.json()}).then(function(md){var mds=md.models||[];if(mds.length>0)eQualityFetch(mds[mds.length-1].id,'e-quality-panel-aml')}).catch(function(){})}else if(d.status==='failed')
```

- [ ] **Step 5: Add OOD warning logic to eInsPgShowPrediction**

Find in `src/server/handlers.rs`:

```javascript
    function eInsPgShowPrediction(d) {
      var predLabel = document.getElementById('e-ins-pg-pred-label');
      var predConf = document.getElementById('e-ins-pg-pred-conf');
      // /api/predict returns {predictions: [f64, ...]}
      var preds = d.predictions || [];
```

Replace with:

```javascript
    function eInsPgShowPrediction(d) {
      var predLabel = document.getElementById('e-ins-pg-pred-label');
      var predConf = document.getElementById('e-ins-pg-pred-conf');
      var oodBanner = document.getElementById('e-ins-pg-ood-warn');
      if (oodBanner) oodBanner.style.display = d.ood_warning ? 'flex' : 'none';
      // /api/predict returns {predictions: [f64, ...]}
      var preds = d.predictions || [];
```

- [ ] **Step 6: Build to confirm no compile errors**

```bash
cargo build 2>&1 | grep -E "^error" | head -20
```

Expected: no errors.

- [ ] **Step 7: Run quality tests to confirm no regressions**

```bash
cargo test -p automl quality 2>&1 | tail -20
```

Expected: all tests pass.

- [ ] **Step 8: Commit**

```bash
git add src/server/handlers.rs
git commit -m "feat(frontend): add pipeline quality panel JS, OOD banner, and poll hooks"
```

---

## Self-Review Checklist

### Spec Coverage

| Spec requirement | Task |
|---|---|
| Quality score badge on leaderboard | Panel header badge shows overall score (Task 3) — row-level badges deferred (requires model-name-to-ID mapping, fragile) |
| Quality panel below training results | Task 2 (HTML divs) + Task 3 (JS render) — covers single, auto-tune, automl |
| Pre-training card: findings, dropped, leakage, high-cardinality | Task 3 — `eQualityPreCard` |
| Training card: CV strategy, excluded algorithms, ensemble result | Task 3 — `eQualityTrainCard` |
| Hyperopt card: trials, converged, Pareto scatter | Task 3 — `eQualityHyperoptCard` + `eQualityPareto` |
| Post-training card: calibration, ECE, OOD threshold, cal fraction | Task 3 — `eQualityPostCard` |
| Gate cards expanded by default with chevron toggle | Task 3 — cards open by default; `eQualityToggle` collapses/expands |
| GateScores in QualityReport JSON | Task 1 |
| OOD warning banner in Insights Playground | Task 2 (HTML) + Task 3 step 5 (`eInsPgShowPrediction`) |
| Auto-fetch on training completion | Task 3 steps 2–4 (ePoll, eAtPoll, eAMLPoll hooks) |

### Type Consistency

- `GateScores` fields: `pre`, `training`, `hyperopt`, `post_training` — Rust (Task 1) matches JS `gs.pre`, `gs.training`, `gs.hyperopt`, `gs.post_training` (Task 3) ✓
- Panel IDs: `e-quality-panel` / `e-quality-panel-at` / `e-quality-panel-aml` — consistent between Task 2 HTML and Task 3 JS ✓
- Canvas ID: `qp-pareto-<safeId>` — created in `eQualityHyperoptCard`, drawn in `eQualityRender` via `eQualityPareto` ✓
- OOD banner ID: `e-ins-pg-ood-warn` — consistent between Task 2 HTML and Task 3 step 5 ✓
- XSS: all user-derived strings (column names, model IDs, finding messages, algorithm names, calibration method, trial IDs) pass through `escHtml` ✓
