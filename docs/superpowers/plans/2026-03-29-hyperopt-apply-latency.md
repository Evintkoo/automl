# HyperOpt Apply & Real Latency Tracking — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** (A) Add an "Apply & Retrain" button to the HyperOpt result panel that trains a first-class model with the best params; (B) replace hardcoded SLO latency values with P50/P95/P99 computed from a ring buffer of actual prediction latencies.

**Architecture:** Feature B adds a `std::sync::Mutex<VecDeque<u64>>` ring buffer to `AppState`, wraps the `predict` handler's inference call with `Instant`, and replaces three hardcoded lines in the SLO handler. Feature A is entirely frontend JS: extend `eHoPoll`'s completion render with a params table + button, add `eHoApply` that calls the existing `/hyperopt/apply` endpoint and runs the same post-training chain as a normal train.

**Tech Stack:** Rust (`std::sync::Mutex`, `std::collections::VecDeque`, `std::time::Instant`), inline JS/HTML in `src/server/handlers.rs`

---

## File Map

| File | Change |
|---|---|
| `src/server/state.rs` | Add `prediction_latencies` field + `record_prediction_latency` sync method |
| `src/server/handlers.rs` | (B) Wrap inference in predict handler + add `percentile` fn + fix SLO handler; (A) extend `eHoPoll` completion HTML + add `eHoApply` JS |

---

## Task 1: Add latency ring buffer to AppState

**Files:**
- Modify: `src/server/state.rs`

- [ ] **Step 1: Write failing tests**

Add this `#[cfg(test)]` block at the bottom of `src/server/state.rs` (before the final `}`):

```rust
#[cfg(test)]
mod tests {
    use std::collections::VecDeque;

    fn percentile_us(sorted: &[u64], p: f64) -> f64 {
        if sorted.is_empty() { return 0.0; }
        let idx = ((sorted.len() as f64 - 1.0) * p / 100.0).round() as usize;
        sorted[idx] as f64 / 1000.0
    }

    #[test]
    fn test_percentile_empty() {
        assert_eq!(percentile_us(&[], 50.0), 0.0);
    }

    #[test]
    fn test_percentile_single() {
        // 5000 µs = 5.0 ms
        assert!((percentile_us(&[5000], 50.0) - 5.0).abs() < 1e-9);
    }

    #[test]
    fn test_percentile_multiple() {
        let mut vals = vec![1000u64, 5000, 3000, 2000, 4000];
        vals.sort_unstable();
        // sorted: [1000, 2000, 3000, 4000, 5000]
        // P50 → index 2 → 3000 µs = 3.0 ms
        assert!((percentile_us(&vals, 50.0) - 3.0).abs() < 1e-9);
        // P99 → index 4 → 5000 µs = 5.0 ms
        assert!((percentile_us(&vals, 99.0) - 5.0).abs() < 1e-9);
    }

    #[test]
    fn test_ring_buffer_cap() {
        let mut q: VecDeque<u64> = VecDeque::new();
        for i in 0..1100u64 {
            if q.len() >= 1000 { q.pop_front(); }
            q.push_back(i);
        }
        assert_eq!(q.len(), 1000);
        // First 100 were evicted; front should be 100
        assert_eq!(*q.front().unwrap(), 100u64);
    }
}
```

- [ ] **Step 2: Run tests — expect they compile and pass**

```bash
cargo test -p automl test_percentile -- --nocapture 2>&1 | tail -10
cargo test -p automl test_ring_buffer_cap -- --nocapture 2>&1 | tail -5
```

Expected: all four tests pass (pure logic, no AppState dependency).

- [ ] **Step 3: Add the field to the `AppState` struct**

In `src/server/state.rs`, find the line:

```rust
    /// Store completed hyperopt studies for history
    pub completed_studies: RwLock<Vec<serde_json::Value>>,
```

Add immediately after it:

```rust
    /// Ring buffer of recent prediction latencies in microseconds (capped at 1000).
    pub prediction_latencies: std::sync::Mutex<std::collections::VecDeque<u64>>,
```

- [ ] **Step 4: Initialize in `AppState::new()`**

In the `Self { ... }` constructor block, find:

```rust
            completed_studies: RwLock::new(Vec::new()),
```

Add immediately after it:

```rust
            prediction_latencies: std::sync::Mutex::new(std::collections::VecDeque::new()),
```

- [ ] **Step 5: Add `record_prediction_latency` method**

Inside `impl AppState { ... }`, add this method before the final `}`:

```rust
    /// Push a prediction latency (microseconds) into the ring buffer.
    /// Evicts the oldest entry once the buffer reaches 1000 entries.
    pub fn record_prediction_latency(&self, micros: u64) {
        let mut q = self.prediction_latencies.lock().unwrap();
        if q.len() >= 1000 {
            q.pop_front();
        }
        q.push_back(micros);
    }
```

- [ ] **Step 6: Build to confirm no errors**

```bash
cargo build 2>&1 | tail -15
```

Expected: `Finished` with at most the one existing unused-parens warning.

- [ ] **Step 7: Commit**

```bash
git add src/server/state.rs
git commit -m "feat(slo): add prediction_latencies ring buffer to AppState"
```

---

## Task 2: Instrument predict handler + fix SLO percentiles

**Files:**
- Modify: `src/server/handlers.rs`

- [ ] **Step 1: Add the `percentile` free function**

In `src/server/handlers.rs`, find the comment:

```rust
// ============================================================================
// Sample Data Generators
// ============================================================================
```

Insert this function immediately before that comment block:

```rust
/// Compute the p-th percentile (0-100) of a **sorted** slice of microsecond values.
/// Returns the result in milliseconds. Returns 0.0 for empty slices.
fn percentile(sorted: &[u64], p: f64) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }
    let idx = ((sorted.len() as f64 - 1.0) * p / 100.0).round() as usize;
    sorted[idx] as f64 / 1000.0
}
```

- [ ] **Step 2: Wrap the inference call with Instant**

In `pub async fn predict`, find:

```rust
    // Run prediction
    let predictions = engine.predict_array(&x)
        .map_err(|e| ServerError::Internal(format!("Prediction failed: {}", e)))?;

    let cache_hit = engine.last_prediction_was_cached();
```

Replace with:

```rust
    // Run prediction — measure pure inference latency for SLO tracking
    let t0 = std::time::Instant::now();
    let predictions = engine.predict_array(&x)
        .map_err(|e| ServerError::Internal(format!("Prediction failed: {}", e)))?;
    let inference_us = t0.elapsed().as_micros() as u64;

    let cache_hit = engine.last_prediction_was_cached();
```

- [ ] **Step 3: Record latency and drop engine guard**

In the same `predict` function, find:

```rust
    let cache_hit = engine.last_prediction_was_cached();
    let (cache_hits, cache_misses, cache_hit_rate) = engine.cache_stats();
    let avg_latency = engine.stats().avg_latency_ms;

    // OOD check using stored detector (if available)
```

Replace with:

```rust
    let cache_hit = engine.last_prediction_was_cached();
    let (cache_hits, cache_misses, cache_hit_rate) = engine.cache_stats();
    let avg_latency = engine.stats().avg_latency_ms;
    drop(engine); // release registry guard before async work

    // Record inference latency for SLO percentile tracking
    state.record_prediction_latency(inference_us);

    // OOD check using stored detector (if available)
```

- [ ] **Step 4: Replace hardcoded SLO latency values**

In `pub async fn get_slo_status`, find:

```rust
    let metrics = SloMetrics {
        latency_p50_ms: 5.0,   // TODO: instrument actual latency tracking
        latency_p95_ms: 25.0,
        latency_p99_ms: 100.0,
        availability_percent: availability,
        error_rate,
        total_requests,
        total_errors,
    };
```

Replace with:

```rust
    let latency_sorted = {
        let q = state.prediction_latencies.lock().unwrap();
        let mut v: Vec<u64> = q.iter().copied().collect();
        v.sort_unstable();
        v
    };

    let metrics = SloMetrics {
        latency_p50_ms: percentile(&latency_sorted, 50.0),
        latency_p95_ms: percentile(&latency_sorted, 95.0),
        latency_p99_ms: percentile(&latency_sorted, 99.0),
        availability_percent: availability,
        error_rate,
        total_requests,
        total_errors,
    };
```

- [ ] **Step 5: Build to confirm no errors**

```bash
cargo build 2>&1 | tail -15
```

Expected: `Finished` cleanly.

- [ ] **Step 6: Commit**

```bash
git add src/server/handlers.rs
git commit -m "feat(slo): instrument real prediction latency, replace hardcoded P50/P95/P99"
```

---

## Task 3: HyperOpt — params table + Apply & Retrain button

**Files:**
- Modify: `src/server/handlers.rs` (frontend JS string only)

The JS `eHoPoll` function's completion branch renders the best trial result. We extend it with a styled dark card, a params table, and an Apply button. We then add `eHoApply` and `eHoApplyPoll` functions that call `/hyperopt/apply` and run the full post-training chain on completion.

All user-derived content placed into the page goes through the existing `escHtml()` helper — same pattern used throughout the codebase.

- [ ] **Step 1: Replace the result card HTML in `eHoPoll`**

Find this exact substring in the minified JS (it is all on one line inside `eHoPoll`):

```
var h='<div style="background:#f3fbf4;padding:12px;border-radius:8px;margin-bottom:12px">';if(m.best_trial){h+='<strong>Best Score:</strong> '+((m.best_trial.value||0).toFixed(6))+'<br><small>'+JSON.stringify(m.best_trial.params||{})+'</small>'}h+='</div><p style="font-size:12px;color:#6a6f73">Trials: '+(m.total_trials||0)+' | Time: '+(m.total_duration_secs||0).toFixed(2)+'s</p>';
```

Replace with (keep on one line):

```
var h='';if(m.best_trial){var bp=m.best_trial.params||{};var paramRows='';Object.keys(bp).forEach(function(k){var v=bp[k];paramRows+='<tr><td style="padding:3px 8px;color:#9ca3af;font-size:12px">'+escHtml(k)+'</td><td class="mono" style="padding:3px 8px;font-size:12px;color:#e5e7eb">'+(typeof v==='number'?v.toFixed(4):escHtml(String(v)))+'</td></tr>'});h+='<div style="background:#130e24;border:1px solid #3b2d6e;border-radius:10px;padding:14px;margin-bottom:12px"><div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:10px"><span style="font-size:13px;font-weight:600;color:#d1d5db">Best Trial #'+escHtml(String(m.best_trial.trial_id))+'</span><span style="font-size:20px;font-weight:800;color:#a78bfa">'+((m.best_trial.value||0).toFixed(6))+'</span></div>';if(paramRows){h+='<table style="width:100%;border-collapse:collapse;margin-bottom:10px"><tbody>'+paramRows+'</tbody></table>'}h+='<button onclick="eHoApply(this,'+JSON.stringify(JSON.stringify(m.best_trial))+')" style="background:#8b5cf6;color:#fff;border:none;border-radius:8px;padding:7px 18px;font-size:13px;font-weight:600;cursor:pointer;width:100%">&#10003; Apply &amp; Retrain with Best Params</button></div>'}h+='<p style="font-size:12px;color:#6a6f73">Trials: '+(m.total_trials||0)+' | Time: '+(m.total_duration_secs||0).toFixed(2)+'s</p>';
```

- [ ] **Step 2: Add `eHoApply` and `eHoApplyPoll` functions**

Find the line:

```javascript
    function eHo3dChart(trials,bestTrial){
```

Insert these two functions immediately before that line (keep each function on one line in the minified JS block — they are inside the large `<script>` string literal):

```javascript
    function eHoApply(btn,bestTrialJson){var bestTrial=JSON.parse(bestTrialJson);if(btn){btn.disabled=true;btn.textContent='Retraining\u2026'}var targetCol=$('e-target')?$('e-target').value:'';var taskType=$('e-task')?$('e-task').value:'';var modelType=$('e-ho-model')?$('e-ho-model').value:'';if(!targetCol||!taskType){eNotify('Load a dataset first','err');if(btn){btn.disabled=false;btn.textContent='\u2713 Apply \u0026 Retrain with Best Params'}return}fetch('/hyperopt/apply',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({target_column:targetCol,task_type:taskType,model_type:modelType,params:bestTrial.params||{}})}).then(function(r){return r.json()}).then(function(d){if(!d.job_id){throw new Error(d.error||'No job_id')}eHoApplyPoll(d.job_id,modelType,btn)}).catch(function(e){eNotify('Apply failed: '+e.message,'err');if(btn){btn.disabled=false;btn.textContent='\u2713 Apply \u0026 Retrain with Best Params'}})}
    function eHoApplyPoll(jobId,modelType,btn){fetch('/api/train/status/'+jobId).then(function(r){return r.json()}).then(function(d){var s=d.status;if(s.Completed){var m=s.Completed.metrics||{};var newId=m.model_id||('optimized_'+modelType+'_'+jobId);eLastModelId=newId;var explBtn=$('e-explain-btn');if(explBtn)explBtn.disabled=false;eQualityFetch(newId,'e-quality-panel');eDashModels();eNotify('Optimized model trained!','ok');eLogActivity('HyperOpt Apply: '+modelType);eHistPush({type:'HyperOpt-Apply',model:modelType,metrics:m.metrics||{},target:$('e-target')?$('e-target').value:'',task:$('e-task')?$('e-task').value:'',ts:Date.now()});var safeId=newId.replace(/[^a-zA-Z0-9_-]/g,'');var existingResult=$('e-result');if(existingResult){var umapDiv=document.createElement('div');umapDiv.style.cssText='margin-top:16px';var bw=document.createElement('div');bw.id='umap-bwrap-'+safeId;bw.style.cssText='margin-bottom:4px';var bl=document.createElement('div');bl.style.cssText='font-size:11px;color:#6a6f73;margin-bottom:2px';bl.textContent='Computing UMAP\u2026';bw.appendChild(bl);var bt=document.createElement('div');bt.style.cssText='height:3px;background:#1a1b1c;border-radius:2px';var bf=document.createElement('div');bf.id='umap-bar-'+safeId;bf.style.cssText='height:3px;background:#8b5cf6;border-radius:2px;width:0%;transition:width .3s';bt.appendChild(bf);bw.appendChild(bt);var wd=document.createElement('div');wd.id='umap-wrap-'+safeId;var cv=document.createElement('canvas');cv.id='umap-scatter-'+safeId;cv.style.cssText='width:100%;height:280px;display:block;border-radius:8px;background:#0d0e0f';wd.appendChild(cv);umapDiv.appendChild(bw);umapDiv.appendChild(wd);existingResult.appendChild(umapDiv);eStartUmap(newId)}var met=m.metrics||{};var mh='<div style="background:#0d0e0f;border-radius:8px;padding:12px;margin-top:10px;font-size:12px;color:#9ca3af"><div style="font-weight:600;color:#e5e7eb;margin-bottom:6px">Optimized Model Metrics</div>';Object.keys(met).forEach(function(k){if(typeof met[k]==='number')mh+='<div style="display:flex;justify-content:space-between"><span>'+escHtml(k)+'</span><span class="mono" style="color:#a78bfa">'+met[k].toFixed(4)+'</span></div>'});mh+='</div>';var hoRes=$('e-ho-result');if(hoRes){var applyRes=document.createElement('div');applyRes.innerHTML=mh;hoRes.appendChild(applyRes)}if(btn){btn.disabled=false;btn.textContent='\u2713 Apply \u0026 Retrain with Best Params'}}else if(s.Failed){eNotify('Apply failed: '+(s.Failed.error||'Unknown'),'err');if(btn){btn.disabled=false;btn.textContent='\u2713 Apply \u0026 Retrain with Best Params'}}else{setTimeout(function(){eHoApplyPoll(jobId,modelType,btn)},1500)}}).catch(function(){setTimeout(function(){eHoApplyPoll(jobId,modelType,btn)},3000)})}
```

- [ ] **Step 3: Build to confirm no Rust errors**

```bash
cargo build 2>&1 | tail -10
```

Expected: `Finished` cleanly. Rust does not validate inline JS syntax.

- [ ] **Step 4: Commit**

```bash
git add src/server/handlers.rs
git commit -m "feat(hyperopt): add Apply & Retrain button with params table and first-class model flow"
```

---

## Task 4: Fix the pre-existing unused-parens warning

**Files:**
- Modify: `src/server/handlers.rs`

- [ ] **Step 1: Fix the warning**

Find this line (around line 8266, now possibly shifted by new lines above):

```rust
                    for ((&yt, &prob)) in eval.y_true.iter().zip(probs.iter()) {
```

Replace with:

```rust
                    for (&yt, &prob) in eval.y_true.iter().zip(probs.iter()) {
```

- [ ] **Step 2: Verify zero warnings**

```bash
cargo build 2>&1 | grep -E "^warning|^error|Finished"
```

Expected: only the `Finished` line. No warnings.

- [ ] **Step 3: Commit**

```bash
git add src/server/handlers.rs
git commit -m "fix: remove unnecessary parentheses in for-pattern"
```
