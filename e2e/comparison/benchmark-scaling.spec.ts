import { test, expect, Page } from '@playwright/test';
import * as fs from 'fs';
import * as path from 'path';
import {
  RUST_PLATFORM,
  PYTHON_PLATFORM,
  measureTime,
  rateLimitPause,
} from './helpers';

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const LONG_TIMEOUT = 180_000;
const RUST_BASE = RUST_PLATFORM.baseURL;
const PYTHON_BASE = PYTHON_PLATFORM.baseURL;

const RUST_SIZES = [100, 500, 1_000, 5_000, 10_000, 50_000];

// Python/Gradio available sample datasets (from actual dropdown inspection)
const PYTHON_DATASETS: { name: string; rows: number; target: string; label: string }[] = [
  { name: 'iris', rows: 150, target: 'species', label: 'Iris' },
  { name: 'wine', rows: 178, target: 'target', label: 'Wine' },
  { name: 'boston', rows: 506, target: 'target', label: 'Boston' },
  { name: 'diabetes', rows: 768, target: 'target', label: 'Diabetes' },
];

// Python/Gradio selectors (same as benchmark.spec.ts)
const PY = {
  sampleDatasetDropdown: 'input[aria-label="\ud83c\udfb2 Sample Datasets"]',
  loadSampleBtn: '#component-20',
  tabStep1: '#component-10-button',
  tabStep2: '#component-30-button',
  tabStep3: '#component-59-button',
  tabStep4: '#component-85-button',
  createConfigBtn: '#component-55',
  targetColumnInput: 'textarea[placeholder="e.g., price, category, outcome"]',
  trainBtn: '#component-67',
  predictionInput: 'textarea[placeholder*="5.1, 3.5, 1.4, 0.2"]',
  predictBtn: '#component-93',
};

// ---------------------------------------------------------------------------
// Data structures
// ---------------------------------------------------------------------------

interface ScalingDataPoint {
  platform: 'rust' | 'python';
  dataset: string;
  rows: number;
  uploadTime: number;
  preprocessTime: number;
  trainTime: number;
  predictTime: number;
}

// Temp file path for persisting results across test blocks
const RESULTS_TMP = path.join(__dirname, '.scaling-results-tmp.json');

function loadResults(): ScalingDataPoint[] {
  try {
    if (fs.existsSync(RESULTS_TMP)) {
      return JSON.parse(fs.readFileSync(RESULTS_TMP, 'utf-8'));
    }
  } catch { /* ignore */ }
  return [];
}

function saveResult(point: ScalingDataPoint) {
  const results = loadResults();
  results.push(point);
  fs.writeFileSync(RESULTS_TMP, JSON.stringify(results));
}

// ---------------------------------------------------------------------------
// CSV generation helper (Box-Muller) — uses numeric target (0,1,2)
// ---------------------------------------------------------------------------

function generateCSV(numRows: number): string {
  function boxMuller(mu: number, sigma: number, r1: number, r2: number): number {
    return mu + sigma * Math.sqrt(-2.0 * Math.log(r1)) * Math.cos(2.0 * Math.PI * r2);
  }

  const lines: string[] = ['sepal_length,sepal_width,petal_length,petal_width,species'];

  let seed = 42;
  function rand(): number {
    seed = (seed * 1103515245 + 12345) & 0x7fffffff;
    return seed / 0x7fffffff;
  }

  for (let i = 0; i < numRows; i++) {
    const r1 = Math.max(rand(), 1e-10);
    const r2 = rand();
    const r3 = Math.max(rand(), 1e-10);
    const r4 = rand();
    const sl = boxMuller(5.8, 0.8, r1, r2).toFixed(2);
    const sw = boxMuller(3.0, 0.4, r3, r4).toFixed(2);
    const r5 = Math.max(rand(), 1e-10);
    const r6 = rand();
    const r7 = Math.max(rand(), 1e-10);
    const r8 = rand();
    const pl = boxMuller(3.8, 1.7, r5, r6).toFixed(2);
    const pw = boxMuller(1.2, 0.7, r7, r8).toFixed(2);
    // Numeric target: 0, 1, or 2 (avoids one-hot encoding during preprocessing)
    const sp = Math.floor(rand() * 3);
    lines.push(`${sl},${sw},${pl},${pw},${sp}`);
  }

  return lines.join('\n');
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

async function loadPythonPlatform(page: Page) {
  await page.goto(PYTHON_BASE, { waitUntil: 'load', timeout: LONG_TIMEOUT });
  await page.waitForSelector('.gradio-container, .app', { timeout: 60_000 });
  await page.waitForTimeout(2000);
}

async function switchTabPython(page: Page, selector: string) {
  await page.locator(selector).click();
  await page.waitForTimeout(1000);
}

// ---------------------------------------------------------------------------
// Scaling Benchmark Tests
// ---------------------------------------------------------------------------

test.describe('Scaling Benchmark', () => {

  // ========================================================================
  // Warm-up: health check both servers + clear temp results
  // ========================================================================

  test.beforeAll(async ({ browser }) => {
    // Clear previous temp results
    try { fs.unlinkSync(RESULTS_TMP); } catch { /* ignore */ }

    const ctx = await browser.newContext();
    const page = await ctx.newPage();

    // Rust health check
    const rustResp = await page.goto(`${RUST_BASE}/api/health`, { timeout: LONG_TIMEOUT });
    expect(rustResp?.status()).toBe(200);

    // Python health check
    const pyResp = await page.goto(PYTHON_BASE, { timeout: LONG_TIMEOUT });
    expect(pyResp?.status()).toBe(200);

    await ctx.close();
  });

  // ========================================================================
  // Rust Scaling — synthetic CSV at 6 sizes
  // ========================================================================

  test.describe('Scaling: Rust', () => {
    for (const size of RUST_SIZES) {
      test(`Rust pipeline at ${size} rows`, async ({ browser }) => {
        const ctx = await browser.newContext();
        const page = await ctx.newPage();
        await page.goto(RUST_BASE, { waitUntil: 'load', timeout: LONG_TIMEOUT });
        await page.waitForLoadState('networkidle', { timeout: 60_000 });

        const csv = generateCSV(size);

        // Inject retry-aware fetch
        await page.evaluate(() => {
          if (!(window as any).__retryFetch) {
            (window as any).__retryFetch = async (input: RequestInfo | URL, init?: RequestInit, maxRetries = 4, baseDelay = 1000) => {
              for (let attempt = 0; attempt <= maxRetries; attempt++) {
                const res = await fetch(input, init);
                if (res.status !== 429) return res;
                if (attempt === maxRetries) return res;
                const delay = baseDelay * Math.pow(2, attempt) + Math.random() * 500;
                await new Promise(r => setTimeout(r, delay));
              }
              return fetch(input, init);
            };
          }
        });

        // 1. Upload CSV via fetch + FormData
        const { elapsed: uploadTime } = await measureTime(async () => {
          const ok = await page.evaluate(async (args: { url: string; csv: string }) => {
            const rf = (window as any).__retryFetch;
            const blob = new Blob([args.csv], { type: 'text/csv' });
            const form = new FormData();
            form.append('file', blob, 'synthetic.csv');
            const res = await rf(`${args.url}/api/data/upload`, { method: 'POST', body: form });
            return res.ok;
          }, { url: RUST_BASE, csv });
          expect(ok).toBe(true);
        });

        await new Promise(r => setTimeout(r, 300));

        // 2. Preprocess
        const { elapsed: preprocessTime } = await measureTime(async () => {
          const ok = await page.evaluate(async (url: string) => {
            const rf = (window as any).__retryFetch;
            const res = await rf(`${url}/api/preprocess`, {
              method: 'POST',
              headers: { 'Content-Type': 'application/json' },
              body: JSON.stringify({
                task_type: 'classification',
                target_column: 'species',
                scaler: 'standard',
                imputation: 'mean',
                cv_folds: 5,
              }),
            });
            return res.ok;
          }, RUST_BASE);
          expect(ok).toBe(true);
        });

        await new Promise(r => setTimeout(r, 300));

        // 3. Train (logistic regression) + poll for completion
        const { elapsed: trainTime } = await measureTime(async () => {
          const jobId = await page.evaluate(async (url: string) => {
            const rf = (window as any).__retryFetch;
            const res = await rf(`${url}/api/train`, {
              method: 'POST',
              headers: { 'Content-Type': 'application/json' },
              body: JSON.stringify({
                model_type: 'logistic_regression',
                target_column: 'species',
                task_type: 'classification',
              }),
            });
            const data = await res.json();
            return data.job_id;
          }, RUST_BASE);

          // Poll status — check for Completed OR Failed (with rate-limit awareness)
          const result = await page.evaluate(async (args: { url: string; jobId: string }) => {
            const rf = (window as any).__retryFetch;
            for (let i = 0; i < 240; i++) {
              const res = await rf(`${args.url}/api/train/status/${args.jobId}`);
              const data = await res.json();
              if (data.status && data.status.Completed) return { ok: true };
              if (data.status && data.status.Failed) {
                return { ok: false, error: data.status.Failed.error };
              }
              await new Promise(r => setTimeout(r, 500));
            }
            return { ok: false, error: 'Training timed out' };
          }, { url: RUST_BASE, jobId });

          if (!result.ok) throw new Error(`Training failed: ${result.error}`);
        });

        await new Promise(r => setTimeout(r, 300));

        // 4. Batch predict (100 samples)
        const { elapsed: predictTime } = await measureTime(async () => {
          const samples = Array.from({ length: 100 }, () => [5.1, 3.5, 1.4, 0.2]);
          const result = await page.evaluate(async (args: { url: string; samples: number[][] }) => {
            const rf = (window as any).__retryFetch;
            const res = await rf(`${args.url}/api/predict`, {
              method: 'POST',
              headers: { 'Content-Type': 'application/json' },
              body: JSON.stringify({ data: args.samples }),
            });
            return res.json();
          }, { url: RUST_BASE, samples });
          expect(result.success).toBe(true);
        });

        // 5. Clear data for next test
        await page.evaluate(async (url: string) => {
          const rf = (window as any).__retryFetch || fetch;
          await rf(`${url}/api/data/clear`, { method: 'DELETE' });
        }, RUST_BASE);

        // Record to temp file (persists across test blocks)
        const point: ScalingDataPoint = {
          platform: 'rust',
          dataset: `synthetic_${size}`,
          rows: size,
          uploadTime,
          preprocessTime,
          trainTime,
          predictTime,
        };
        saveResult(point);

        console.log(`Rust ${size} rows: upload=${uploadTime}ms preprocess=${preprocessTime}ms train=${trainTime}ms predict=${predictTime}ms`);

        await ctx.close();
      });
    }
  });

  // ========================================================================
  // Python Scaling — 4 built-in sample datasets
  // ========================================================================

  test.describe('Scaling: Python', () => {
    for (const ds of PYTHON_DATASETS) {
      test(`Python pipeline on ${ds.name} (${ds.rows} rows)`, async ({ browser }) => {
        const ctx = await browser.newContext();
        const page = await ctx.newPage();
        await loadPythonPlatform(page);

        // 1. Load sample dataset
        const { elapsed: loadTime } = await measureTime(async () => {
          const dropdown = page.locator(PY.sampleDatasetDropdown);
          await dropdown.waitFor({ state: 'visible', timeout: 30_000 });
          await dropdown.click();

          await page.locator(`[role="option"]:has-text("${ds.label}")`).first().click({ timeout: 10_000 });
          await page.locator(PY.loadSampleBtn).click();
          await page.waitForSelector('text=Sample Dataset Loaded', { timeout: 60_000 });
        });

        // 2. Configure (Step 2)
        const { elapsed: configTime } = await measureTime(async () => {
          await switchTabPython(page, PY.tabStep2);
          await page.locator(PY.createConfigBtn).waitFor({ state: 'visible', timeout: 10_000 });
          await page.locator(PY.createConfigBtn).click();
          await page.waitForFunction(
            () => {
              const els = document.querySelectorAll('.prose, .html-container');
              for (const el of els) {
                const t = el.textContent || '';
                if (t.includes('Configuration') || t.includes('created') || t.includes('Ready')) return true;
              }
              return false;
            },
            { timeout: 30_000 },
          );
        });

        // 3. Train (Step 3)
        const { elapsed: trainTime } = await measureTime(async () => {
          await switchTabPython(page, PY.tabStep3);
          await page.waitForTimeout(1000);

          await page.locator(PY.targetColumnInput).fill(ds.target);
          await page.waitForTimeout(500);

          // Select first algorithm
          const algoInput = page.locator('#component-65 input').first();
          await algoInput.waitFor({ state: 'visible', timeout: 10_000 });
          await algoInput.click();
          await page.waitForTimeout(500);
          try {
            await page.locator('[role="option"]').first().click({ timeout: 10_000 });
          } catch {
            // No algorithms available
          }
          await page.waitForTimeout(500);

          await page.locator(PY.trainBtn).click();
          await page.waitForFunction(
            () => {
              const els = document.querySelectorAll('.prose, .html-container, [data-testid="html"]');
              for (const el of els) {
                const t = el.textContent || '';
                if (t.includes('completed') || t.includes('Accuracy') || t.includes('Training Results') ||
                    t.includes('trained') || t.includes('Model:') || t.includes('Score')) {
                  return true;
                }
              }
              return false;
            },
            { timeout: LONG_TIMEOUT },
          );
        });

        // 4. Single prediction (Step 4)
        let predictTime = 0;
        try {
          await switchTabPython(page, PY.tabStep4);
          const inputLocator = page.locator(PY.predictionInput);
          await inputLocator.waitFor({ state: 'visible', timeout: 10_000 });

          const { elapsed } = await measureTime(async () => {
            await inputLocator.fill('5.1, 3.5, 1.4, 0.2');
            await page.locator(PY.predictBtn).click();
            await page.waitForFunction(
              () => {
                const el = document.querySelector('#component-98');
                return el && el.textContent && el.textContent.length > 10;
              },
              { timeout: 30_000 },
            );
          });
          predictTime = elapsed;
        } catch {
          console.log(`Warning: prediction failed for ${ds.name}, recording 0`);
        }

        const point: ScalingDataPoint = {
          platform: 'python',
          dataset: ds.name,
          rows: ds.rows,
          uploadTime: loadTime,
          preprocessTime: configTime,
          trainTime,
          predictTime,
        };
        saveResult(point);

        console.log(`Python ${ds.name} (${ds.rows}): load=${loadTime}ms config=${configTime}ms train=${trainTime}ms predict=${predictTime}ms`);

        await ctx.close();
      });
    }
  });

  // ========================================================================
  // Report Generation
  // ========================================================================

  test.describe('Scaling: Report', () => {
    test('generate scaling report', async () => {
      const outDir = path.resolve(__dirname);
      const scalingResults = loadResults();

      // Save raw JSON
      const jsonPath = path.join(outDir, 'scaling-results.json');
      fs.writeFileSync(jsonPath, JSON.stringify({
        timestamp: new Date().toISOString(),
        results: scalingResults,
      }, null, 2));
      console.log(`Scaling results saved to ${jsonPath}`);

      // Generate HTML report with Chart.js
      const rustData = scalingResults.filter(r => r.platform === 'rust');
      const pythonData = scalingResults.filter(r => r.platform === 'python');

      const htmlPath = path.join(outDir, 'scaling-report.html');
      const html = generateReportHTML(rustData, pythonData);
      fs.writeFileSync(htmlPath, html);
      console.log(`Scaling report saved to ${htmlPath}`);

      // Verify files exist
      expect(fs.existsSync(jsonPath)).toBe(true);
      expect(fs.existsSync(htmlPath)).toBe(true);

      // Verify we have data
      const json = JSON.parse(fs.readFileSync(jsonPath, 'utf-8'));
      expect(json.results.length).toBeGreaterThan(0);
      console.log(`Total data points collected: ${json.results.length} (rust: ${rustData.length}, python: ${pythonData.length})`);

      // Clean up temp file
      try { fs.unlinkSync(RESULTS_TMP); } catch { /* ignore */ }
    });
  });
});

// ---------------------------------------------------------------------------
// HTML Report Generator
// ---------------------------------------------------------------------------

function generateReportHTML(rustData: ScalingDataPoint[], pythonData: ScalingDataPoint[]): string {
  const rustRows = JSON.stringify(rustData.map(d => d.rows));
  const rustUpload = JSON.stringify(rustData.map(d => d.uploadTime));
  const rustTrain = JSON.stringify(rustData.map(d => d.trainTime));
  const rustPredict = JSON.stringify(rustData.map(d => d.predictTime));

  const pyRows = JSON.stringify(pythonData.map(d => d.rows));
  const pyUpload = JSON.stringify(pythonData.map(d => d.uploadTime));
  const pyTrain = JSON.stringify(pythonData.map(d => d.trainTime));
  const pyPredict = JSON.stringify(pythonData.map(d => d.predictTime));

  // Build summary table rows
  const allData = [...rustData, ...pythonData];
  const tableRows = allData.map(d => `
    <tr>
      <td>${d.platform}</td>
      <td>${d.dataset}</td>
      <td>${d.rows.toLocaleString()}</td>
      <td>${d.uploadTime.toFixed(0)}</td>
      <td>${d.preprocessTime.toFixed(0)}</td>
      <td>${d.trainTime.toFixed(0)}</td>
      <td>${d.predictTime.toFixed(0)}</td>
      <td>${(d.uploadTime + d.preprocessTime + d.trainTime + d.predictTime).toFixed(0)}</td>
    </tr>`).join('');

  // Compute speedup factors at comparable sizes
  const comparablePairs = pythonData.map(py => {
    const sorted = [...rustData].sort((a, b) =>
      Math.abs(a.rows - py.rows) - Math.abs(b.rows - py.rows)
    );
    return { python: py, rust: sorted[0] || null };
  }).filter(p => p.rust !== null);

  const speedupLabels = JSON.stringify(comparablePairs.map(p => `${p.python.dataset}(${p.python.rows}) vs Rust(${p.rust!.rows})`));
  const speedupUpload = JSON.stringify(comparablePairs.map(p => p.rust!.uploadTime > 0 ? +(p.python.uploadTime / p.rust!.uploadTime).toFixed(2) : 0));
  const speedupTrain = JSON.stringify(comparablePairs.map(p => p.rust!.trainTime > 0 ? +(p.python.trainTime / p.rust!.trainTime).toFixed(2) : 0));
  const speedupPredict = JSON.stringify(comparablePairs.map(p => p.rust!.predictTime > 0 ? +(p.python.predictTime / p.rust!.predictTime).toFixed(2) : 0));

  return `<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>AutoML - Dataset Scaling Benchmark</title>
  <script src="https://cdn.jsdelivr.net/npm/chart.js@4"></script>
  <style>
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; background: #0f172a; color: #e2e8f0; padding: 2rem; }
    h1 { text-align: center; font-size: 1.8rem; margin-bottom: 0.5rem; color: #f8fafc; }
    .subtitle { text-align: center; color: #94a3b8; margin-bottom: 2rem; }
    .grid { display: grid; grid-template-columns: 1fr 1fr; gap: 1.5rem; margin-bottom: 2rem; }
    .card { background: #1e293b; border-radius: 12px; padding: 1.5rem; border: 1px solid #334155; }
    .card h2 { font-size: 1rem; color: #94a3b8; margin-bottom: 1rem; }
    canvas { width: 100% !important; }
    table { width: 100%; border-collapse: collapse; font-size: 0.85rem; }
    th, td { padding: 0.5rem 0.75rem; text-align: right; border-bottom: 1px solid #334155; }
    th { color: #94a3b8; font-weight: 600; text-align: right; }
    th:first-child, td:first-child { text-align: left; }
    th:nth-child(2), td:nth-child(2) { text-align: left; }
    tr:hover { background: #334155; }
    .rust { color: #60a5fa; }
    .python { color: #fb923c; }
    .timestamp { text-align: center; color: #64748b; font-size: 0.8rem; margin-top: 2rem; }
    @media (max-width: 900px) { .grid { grid-template-columns: 1fr; } }
  </style>
</head>
<body>
  <h1>AutoML &mdash; Dataset Scaling Benchmark</h1>
  <p class="subtitle"><span class="rust">Rust</span> vs <span class="python">Python/Gradio</span> &mdash; Performance at increasing dataset sizes</p>

  <div class="grid">
    <div class="card">
      <h2>Upload / Load Time vs Dataset Size</h2>
      <canvas id="chartUpload"></canvas>
    </div>
    <div class="card">
      <h2>Training Time vs Dataset Size</h2>
      <canvas id="chartTrain"></canvas>
    </div>
    <div class="card">
      <h2>Prediction Time vs Dataset Size</h2>
      <canvas id="chartPredict"></canvas>
    </div>
    <div class="card">
      <h2>Speedup Factor (Python / Rust) at Comparable Sizes</h2>
      <canvas id="chartSpeedup"></canvas>
    </div>
  </div>

  <div class="card" style="margin-bottom: 2rem;">
    <h2>Raw Measurements</h2>
    <table>
      <thead>
        <tr>
          <th>Platform</th>
          <th>Dataset</th>
          <th>Rows</th>
          <th>Upload (ms)</th>
          <th>Preprocess (ms)</th>
          <th>Train (ms)</th>
          <th>Predict (ms)</th>
          <th>Total (ms)</th>
        </tr>
      </thead>
      <tbody>${tableRows}
      </tbody>
    </table>
  </div>

  <p class="timestamp">Generated: ${new Date().toISOString()}</p>

  <script>
    const rustRows = ${rustRows};
    const rustUpload = ${rustUpload};
    const rustTrain = ${rustTrain};
    const rustPredict = ${rustPredict};

    const pyRows = ${pyRows};
    const pyUpload = ${pyUpload};
    const pyTrain = ${pyTrain};
    const pyPredict = ${pyPredict};

    const speedupLabels = ${speedupLabels};
    const speedupUpload = ${speedupUpload};
    const speedupTrain = ${speedupTrain};
    const speedupPredict = ${speedupPredict};

    const rustColor = 'rgba(96, 165, 250, 1)';
    const rustBg = 'rgba(96, 165, 250, 0.15)';
    const pyColor = 'rgba(251, 146, 60, 1)';
    const pyBg = 'rgba(251, 146, 60, 0.15)';

    const logScaleOpts = {
      type: 'logarithmic',
      title: { display: true, color: '#94a3b8' },
      ticks: { color: '#64748b' },
      grid: { color: '#1e293b' },
    };

    function makeXY(rows, values) {
      return rows.map((r, i) => ({ x: r, y: values[i] }));
    }

    // Chart 1: Upload / Load Time
    new Chart(document.getElementById('chartUpload'), {
      type: 'line',
      data: {
        datasets: [
          { label: 'Rust', data: makeXY(rustRows, rustUpload), borderColor: rustColor, backgroundColor: rustBg, fill: false, tension: 0.3, pointRadius: 5 },
          { label: 'Python', data: makeXY(pyRows, pyUpload), borderColor: pyColor, backgroundColor: pyBg, fill: false, tension: 0.3, pointRadius: 5 },
        ],
      },
      options: {
        responsive: true,
        plugins: { legend: { labels: { color: '#e2e8f0' } } },
        scales: {
          x: { ...logScaleOpts, title: { ...logScaleOpts.title, text: 'Dataset Size (rows)' }, type: 'logarithmic' },
          y: { ...logScaleOpts, title: { ...logScaleOpts.title, text: 'Time (ms)' }, type: 'logarithmic' },
        },
      },
    });

    // Chart 2: Training Time
    new Chart(document.getElementById('chartTrain'), {
      type: 'line',
      data: {
        datasets: [
          { label: 'Rust', data: makeXY(rustRows, rustTrain), borderColor: rustColor, backgroundColor: rustBg, fill: false, tension: 0.3, pointRadius: 5 },
          { label: 'Python', data: makeXY(pyRows, pyTrain), borderColor: pyColor, backgroundColor: pyBg, fill: false, tension: 0.3, pointRadius: 5 },
        ],
      },
      options: {
        responsive: true,
        plugins: { legend: { labels: { color: '#e2e8f0' } } },
        scales: {
          x: { ...logScaleOpts, title: { ...logScaleOpts.title, text: 'Dataset Size (rows)' }, type: 'logarithmic' },
          y: { ...logScaleOpts, title: { ...logScaleOpts.title, text: 'Time (ms)' }, type: 'logarithmic' },
        },
      },
    });

    // Chart 3: Prediction Time (linear)
    new Chart(document.getElementById('chartPredict'), {
      type: 'line',
      data: {
        datasets: [
          { label: 'Rust (batch 100)', data: makeXY(rustRows, rustPredict), borderColor: rustColor, backgroundColor: rustBg, fill: false, tension: 0.3, pointRadius: 5 },
          { label: 'Python (single)', data: makeXY(pyRows, pyPredict), borderColor: pyColor, backgroundColor: pyBg, fill: false, tension: 0.3, pointRadius: 5 },
        ],
      },
      options: {
        responsive: true,
        plugins: { legend: { labels: { color: '#e2e8f0' } } },
        scales: {
          x: { ...logScaleOpts, title: { ...logScaleOpts.title, text: 'Dataset Size (rows)' }, type: 'logarithmic' },
          y: { title: { display: true, text: 'Time (ms)', color: '#94a3b8' }, ticks: { color: '#64748b' }, grid: { color: '#1e293b' } },
        },
      },
    });

    // Chart 4: Speedup Factor (grouped bar)
    new Chart(document.getElementById('chartSpeedup'), {
      type: 'bar',
      data: {
        labels: speedupLabels,
        datasets: [
          { label: 'Upload/Load Speedup', data: speedupUpload, backgroundColor: 'rgba(96, 165, 250, 0.7)', borderColor: rustColor, borderWidth: 1 },
          { label: 'Training Speedup', data: speedupTrain, backgroundColor: 'rgba(52, 211, 153, 0.7)', borderColor: 'rgba(52, 211, 153, 1)', borderWidth: 1 },
          { label: 'Prediction Speedup', data: speedupPredict, backgroundColor: 'rgba(251, 146, 60, 0.7)', borderColor: pyColor, borderWidth: 1 },
        ],
      },
      options: {
        responsive: true,
        plugins: {
          legend: { labels: { color: '#e2e8f0' } },
          tooltip: {
            callbacks: {
              label: function(ctx) { return ctx.dataset.label + ': ' + ctx.parsed.y.toFixed(2) + 'x'; }
            }
          },
        },
        scales: {
          x: { ticks: { color: '#64748b', maxRotation: 45 }, grid: { color: '#1e293b' } },
          y: {
            title: { display: true, text: 'Speedup Factor (Python/Rust)', color: '#94a3b8' },
            ticks: { color: '#64748b' },
            grid: { color: '#1e293b' },
            beginAtZero: true,
          },
        },
      },
    });
  </script>
</body>
</html>`;
}
