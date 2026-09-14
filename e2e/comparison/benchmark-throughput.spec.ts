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

const RUST_MODELS = [
  'logistic_regression',
  'decision_tree',
  'random_forest',
  'gradient_boosting',
  'xgboost',
  'lightgbm',
  'knn',
  'naive_bayes',
  'svm',
  'adaboost',
];

// Python/Gradio selectors (same as scaling tests)
const PY = {
  sampleDatasetDropdown: 'input[aria-label="\ud83c\udfb2 Sample Datasets"]',
  loadSampleBtn: '#component-20',
  tabStep2: '#component-30-button',
  tabStep3: '#component-59-button',
  tabStep4: '#component-85-button',
  createConfigBtn: '#component-55',
  targetColumnInput: 'textarea[placeholder="e.g., price, category, outcome"]',
  algoDropdownInput: '#component-65 input',
  trainBtn: '#component-67',
  predictionInput: 'textarea[placeholder*="5.1, 3.5, 1.4, 0.2"]',
  predictBtn: '#component-93',
};

// ---------------------------------------------------------------------------
// Data structures
// ---------------------------------------------------------------------------

interface ThroughputResult {
  platform: 'rust' | 'python';
  model: string;
  trainTime: number;
  predictTime: number;
  success: boolean;
  accuracy?: number;
  error?: string;
}

// Temp file path for persisting results across test blocks
const RESULTS_TMP = path.join(__dirname, '.throughput-results-tmp.json');

function loadResults(): ThroughputResult[] {
  try {
    if (fs.existsSync(RESULTS_TMP)) {
      return JSON.parse(fs.readFileSync(RESULTS_TMP, 'utf-8'));
    }
  } catch { /* ignore */ }
  return [];
}

function saveResult(point: ThroughputResult) {
  const results = loadResults();
  results.push(point);
  fs.writeFileSync(RESULTS_TMP, JSON.stringify(results));
}

// File to persist discovered Python algorithms between test blocks
const PYTHON_ALGOS_TMP = path.join(__dirname, '.python-algos-tmp.json');

function loadPythonAlgos(): string[] {
  try {
    if (fs.existsSync(PYTHON_ALGOS_TMP)) {
      return JSON.parse(fs.readFileSync(PYTHON_ALGOS_TMP, 'utf-8'));
    }
  } catch { /* ignore */ }
  return [];
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

/** Load iris on Python side and create config (Steps 1-2) */
async function setupPythonIris(page: Page) {
  // Step 1: Load iris sample dataset
  const dropdown = page.locator(PY.sampleDatasetDropdown);
  await dropdown.waitFor({ state: 'visible', timeout: 30_000 });
  await dropdown.click();
  await page.locator('[role="option"]:has-text("Iris")').first().click({ timeout: 10_000 });
  await page.locator(PY.loadSampleBtn).click();
  await page.waitForSelector('text=Sample Dataset Loaded', { timeout: 60_000 });

  // Step 2: Create config
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
}

// ---------------------------------------------------------------------------
// Name mapping: normalize model names for comparison
// ---------------------------------------------------------------------------

const PYTHON_TO_RUST_MAP: Record<string, string> = {
  'logistic regression': 'logistic_regression',
  'decision tree': 'decision_tree',
  'random forest': 'random_forest',
  'gradient boosting': 'gradient_boosting',
  'xgboost': 'xgboost',
  'lightgbm': 'lightgbm',
  'knn': 'knn',
  'k-nearest neighbors': 'knn',
  'k nearest neighbors': 'knn',
  'naive bayes': 'naive_bayes',
  'gaussian nb': 'naive_bayes',
  'svm': 'svm',
  'support vector machine': 'svm',
  'adaboost': 'adaboost',
  'ada boost': 'adaboost',
  'extra trees': 'random_forest',
};

/** Strip leading checkmarks, category prefixes like "✓ Tree-Based - ", and normalize */
function normalizeModelName(name: string): string {
  let n = name.toLowerCase().trim();
  // Strip leading checkmark/tick + whitespace
  n = n.replace(/^[✓✔☑]\s*/, '');
  // Strip "category - " prefix (e.g. "tree-based - ", "linear models - ")
  n = n.replace(/^[a-z\s-]+-\s+/i, '');
  return n.trim();
}

function findRustEquivalent(pythonLabel: string): string | null {
  const normalized = normalizeModelName(pythonLabel);
  return PYTHON_TO_RUST_MAP[normalized] || null;
}

// ---------------------------------------------------------------------------
// Throughput Benchmark Tests
// ---------------------------------------------------------------------------

test.describe('Throughput Benchmark', () => {

  // ========================================================================
  // Warm-up: health check both servers + clear temp results
  // ========================================================================

  test.beforeAll(async ({ browser }) => {
    // Clear previous temp results
    try { fs.unlinkSync(RESULTS_TMP); } catch { /* ignore */ }
    try { fs.unlinkSync(PYTHON_ALGOS_TMP); } catch { /* ignore */ }

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
  // Rust Throughput — 10 classification models on Iris
  // ========================================================================

  test.describe('Throughput: Rust', () => {
    for (const model of RUST_MODELS) {
      test(`Rust ${model}`, async ({ browser }) => {
        const ctx = await browser.newContext();
        const page = await ctx.newPage();
        await page.goto(RUST_BASE, { waitUntil: 'load', timeout: LONG_TIMEOUT });
        await page.waitForLoadState('networkidle', { timeout: 60_000 });

        let trainTime = 0;
        let predictTime = 0;
        let success = false;
        let accuracy: number | undefined;
        let error: string | undefined;

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

        try {
          // 1. Load iris sample dataset
          const loadRes = await page.evaluate(async (url: string) => {
            const rf = (window as any).__retryFetch;
            const r = await rf(`${url}/api/data/sample/iris`);
            return r.json();
          }, RUST_BASE);
          expect(loadRes.success).toBe(true);

          await new Promise(r => setTimeout(r, 300));

          // 2. Preprocess
          const prepRes = await page.evaluate(async (url: string) => {
            const rf = (window as any).__retryFetch;
            const r = await rf(`${url}/api/preprocess`, {
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
            return r.json();
          }, RUST_BASE);
          expect(prepRes.success).toBe(true);

          await new Promise(r => setTimeout(r, 300));

          // 3. Train + poll for completion
          const { elapsed: tTime } = await measureTime(async () => {
            const jobId = await page.evaluate(async (args: { url: string; model: string }) => {
              const rf = (window as any).__retryFetch;
              const res = await rf(`${args.url}/api/train`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                  model_type: args.model,
                  target_column: 'species',
                  task_type: 'classification',
                }),
              });
              const data = await res.json();
              return data.job_id;
            }, { url: RUST_BASE, model });

            // Poll status — check for Completed OR Failed (rate-limit aware)
            const result = await page.evaluate(async (args: { url: string; jobId: string }) => {
              const rf = (window as any).__retryFetch;
              for (let i = 0; i < 240; i++) {
                const res = await rf(`${args.url}/api/train/status/${args.jobId}`);
                const data = await res.json();
                if (data.status && data.status.Completed) {
                  return {
                    ok: true,
                    accuracy: data.status.Completed.metrics?.accuracy,
                  };
                }
                if (data.status && data.status.Failed) {
                  return { ok: false, error: data.status.Failed.error };
                }
                await new Promise(r => setTimeout(r, 500));
              }
              return { ok: false, error: 'Training timed out' };
            }, { url: RUST_BASE, jobId });

            if (!result.ok) throw new Error(`Training failed: ${result.error}`);
            accuracy = result.accuracy;
          });
          trainTime = tTime;

          await new Promise(r => setTimeout(r, 300));

          // 4. Batch predict (50 samples)
          const { elapsed: pTime } = await measureTime(async () => {
            const samples = Array.from({ length: 50 }, () => [5.1, 3.5, 1.4, 0.2]);
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
          predictTime = pTime;
          success = true;

        } catch (e: unknown) {
          error = e instanceof Error ? e.message : String(e);
          console.log(`Rust ${model} failed: ${error}`);
        }

        // 5. Clear data for next test
        await page.evaluate(async (url: string) => {
          const rf = (window as any).__retryFetch || fetch;
          await rf(`${url}/api/data/clear`, { method: 'DELETE' });
        }, RUST_BASE);

        const point: ThroughputResult = {
          platform: 'rust',
          model,
          trainTime,
          predictTime,
          success,
          accuracy,
          error,
        };
        saveResult(point);

        console.log(`Rust ${model}: train=${trainTime}ms predict=${predictTime}ms success=${success} accuracy=${accuracy?.toFixed(4) ?? 'N/A'}`);

        await ctx.close();
      });
    }
  });

  // ========================================================================
  // Python Throughput — discover algorithms then test each
  // ========================================================================

  test.describe('Throughput: Python', () => {

    test('discover available algorithms', async ({ browser }) => {
      const ctx = await browser.newContext();
      const page = await ctx.newPage();
      await loadPythonPlatform(page);
      await setupPythonIris(page);

      // Switch to Step 3 (Training)
      await switchTabPython(page, PY.tabStep3);
      await page.waitForTimeout(1000);

      // Fill target column first
      await page.locator(PY.targetColumnInput).fill('species');
      await page.waitForTimeout(500);

      // Open algorithm dropdown and scrape all options
      const algoInput = page.locator(PY.algoDropdownInput).first();
      await algoInput.waitFor({ state: 'visible', timeout: 10_000 });
      await algoInput.click();
      await page.waitForTimeout(1000);

      const algorithms = await page.evaluate(() => {
        const options = document.querySelectorAll('[role="option"]');
        return Array.from(options).map(opt => (opt.textContent || '').trim()).filter(t => t.length > 0);
      });

      console.log(`Discovered ${algorithms.length} Python algorithms: ${algorithms.join(', ')}`);

      // Save discovered algorithms for subsequent tests
      fs.writeFileSync(PYTHON_ALGOS_TMP, JSON.stringify(algorithms));

      expect(algorithms.length).toBeGreaterThan(0);

      await ctx.close();
    });

    // We use a dynamic test generation approach: the discovery test runs first,
    // then we generate tests for each algorithm found.
    // Since Playwright needs tests defined at parse time, we define a fixed set
    // of indices and skip if no algorithm exists at that index.

    const MAX_PYTHON_ALGOS = 15; // generous upper bound

    for (let idx = 0; idx < MAX_PYTHON_ALGOS; idx++) {
      test(`Python algorithm #${idx}`, async ({ browser }) => {
        const algorithms = loadPythonAlgos();
        if (idx >= algorithms.length) {
          test.skip();
          return;
        }

        const algorithm = algorithms[idx];
        console.log(`Testing Python algorithm: "${algorithm}"`);

        const ctx = await browser.newContext();
        const page = await ctx.newPage();
        await loadPythonPlatform(page);
        await setupPythonIris(page);

        let trainTime = 0;
        let predictTime = 0;
        let success = false;
        let error: string | undefined;

        try {
          // Step 3: Training
          await switchTabPython(page, PY.tabStep3);
          await page.waitForTimeout(1000);

          // Set target column
          await page.locator(PY.targetColumnInput).fill('species');
          await page.waitForTimeout(500);

          // Select algorithm from dropdown
          const algoInput = page.locator(PY.algoDropdownInput).first();
          await algoInput.waitFor({ state: 'visible', timeout: 10_000 });
          await algoInput.click();
          await page.waitForTimeout(500);

          // Click the exact algorithm option
          await page.locator(`[role="option"]:has-text("${algorithm}")`).first().click({ timeout: 10_000 });
          await page.waitForTimeout(500);

          // Train and measure time
          const { elapsed: tTime } = await measureTime(async () => {
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
          trainTime = tTime;

          // Step 4: Single prediction
          try {
            await switchTabPython(page, PY.tabStep4);
            const inputLocator = page.locator(PY.predictionInput);
            await inputLocator.waitFor({ state: 'visible', timeout: 10_000 });

            const { elapsed: pTime } = await measureTime(async () => {
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
            predictTime = pTime;
          } catch {
            console.log(`Warning: prediction failed for Python ${algorithm}, recording 0`);
          }

          success = true;

        } catch (e: unknown) {
          error = e instanceof Error ? e.message : String(e);
          console.log(`Python ${algorithm} failed: ${error}`);
        }

        const point: ThroughputResult = {
          platform: 'python',
          model: algorithm,
          trainTime,
          predictTime,
          success,
          error,
        };
        saveResult(point);

        console.log(`Python "${algorithm}": train=${trainTime}ms predict=${predictTime}ms success=${success}`);

        await ctx.close();
      });
    }
  });

  // ========================================================================
  // Report Generation
  // ========================================================================

  test.describe('Throughput: Report', () => {
    test('generate throughput report', async () => {
      const outDir = path.resolve(__dirname);
      const allResults = loadResults();

      const rustResults = allResults.filter(r => r.platform === 'rust');
      const pythonResults = allResults.filter(r => r.platform === 'python');

      // Save raw JSON
      const jsonPath = path.join(outDir, 'throughput-results.json');
      fs.writeFileSync(jsonPath, JSON.stringify({
        timestamp: new Date().toISOString(),
        results: allResults,
      }, null, 2));
      console.log(`Throughput results saved to ${jsonPath}`);

      // Build model name mapping for shared models
      const sharedModels: { rustModel: string; pythonLabel: string; rustResult: ThroughputResult; pythonResult: ThroughputResult }[] = [];

      for (const pyResult of pythonResults) {
        if (!pyResult.success) continue;
        const rustEquiv = findRustEquivalent(pyResult.model);
        if (rustEquiv) {
          const rustResult = rustResults.find(r => r.model === rustEquiv && r.success);
          if (rustResult) {
            sharedModels.push({
              rustModel: rustEquiv,
              pythonLabel: pyResult.model,
              rustResult,
              pythonResult: pyResult,
            });
          }
        }
      }

      const rustOnlyModels = rustResults.filter(r => r.success && !sharedModels.find(s => s.rustModel === r.model));

      // Generate HTML report
      const htmlPath = path.join(outDir, 'throughput-report.html');
      const html = generateThroughputReportHTML(rustResults, pythonResults, sharedModels, rustOnlyModels);
      fs.writeFileSync(htmlPath, html);
      console.log(`Throughput report saved to ${htmlPath}`);

      // Verify files exist
      expect(fs.existsSync(jsonPath)).toBe(true);
      expect(fs.existsSync(htmlPath)).toBe(true);

      // Verify we have data
      const json = JSON.parse(fs.readFileSync(jsonPath, 'utf-8'));
      expect(json.results.length).toBeGreaterThan(0);
      console.log(`Total results: ${json.results.length} (rust: ${rustResults.length}, python: ${pythonResults.length})`);
      console.log(`Shared models for comparison: ${sharedModels.length}`);
      console.log(`Rust-only models: ${rustOnlyModels.length}`);

      // Clean up temp files
      try { fs.unlinkSync(RESULTS_TMP); } catch { /* ignore */ }
      try { fs.unlinkSync(PYTHON_ALGOS_TMP); } catch { /* ignore */ }
    });
  });
});

// ---------------------------------------------------------------------------
// HTML Report Generator
// ---------------------------------------------------------------------------

function generateThroughputReportHTML(
  rustResults: ThroughputResult[],
  pythonResults: ThroughputResult[],
  sharedModels: { rustModel: string; pythonLabel: string; rustResult: ThroughputResult; pythonResult: ThroughputResult }[],
  rustOnlyModels: ThroughputResult[],
): string {
  // Chart 1 & 2: Shared models — training and prediction time (grouped bar)
  const sharedLabels = JSON.stringify(sharedModels.map(s => s.rustModel));
  const sharedRustTrain = JSON.stringify(sharedModels.map(s => s.rustResult.trainTime));
  const sharedPythonTrain = JSON.stringify(sharedModels.map(s => s.pythonResult.trainTime));
  const sharedRustPredict = JSON.stringify(sharedModels.map(s => s.rustResult.predictTime));
  const sharedPythonPredict = JSON.stringify(sharedModels.map(s => s.pythonResult.predictTime));

  // Chart 3: Rust-only models (horizontal bar)
  const rustOnlyLabels = JSON.stringify(rustOnlyModels.map(r => r.model));
  const rustOnlyTrain = JSON.stringify(rustOnlyModels.map(r => r.trainTime));
  const rustOnlyPredict = JSON.stringify(rustOnlyModels.map(r => r.predictTime));

  // Chart 4: Speedup factors for shared models
  const speedupLabels = JSON.stringify(sharedModels.map(s => s.rustModel));
  const trainSpeedups = JSON.stringify(sharedModels.map(s =>
    s.rustResult.trainTime > 0 ? +(s.pythonResult.trainTime / s.rustResult.trainTime).toFixed(2) : 0
  ));
  const predictSpeedups = JSON.stringify(sharedModels.map(s =>
    s.rustResult.predictTime > 0 ? +(s.pythonResult.predictTime / s.rustResult.predictTime).toFixed(2) : 0
  ));

  // Build summary table rows for ALL results
  const allResults = [...rustResults, ...pythonResults];
  const tableRows = allResults.map(d => `
    <tr class="${d.success ? '' : 'failed'}">
      <td class="${d.platform}">${d.platform}</td>
      <td>${d.model}</td>
      <td>${d.trainTime.toFixed(0)}</td>
      <td>${d.predictTime.toFixed(0)}</td>
      <td>${d.success ? 'Yes' : 'No'}</td>
      <td>${d.accuracy !== undefined ? (d.accuracy * 100).toFixed(2) + '%' : 'N/A'}</td>
      <td>${d.error || ''}</td>
    </tr>`).join('');

  return `<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>AutoML - Model Throughput Benchmark</title>
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
    tr.failed { opacity: 0.5; }
    .rust { color: #60a5fa; }
    .python { color: #fb923c; }
    .timestamp { text-align: center; color: #64748b; font-size: 0.8rem; margin-top: 2rem; }
    @media (max-width: 900px) { .grid { grid-template-columns: 1fr; } }
  </style>
</head>
<body>
  <h1>AutoML &mdash; Model Throughput Benchmark</h1>
  <p class="subtitle"><span class="rust">Rust</span> vs <span class="python">Python/Gradio</span> &mdash; Training throughput across ML model types (Iris dataset, 150 rows)</p>

  <div class="grid">
    <div class="card">
      <h2>Chart 1: Training Time by Model (Shared Models)</h2>
      <canvas id="chartTrainShared"></canvas>
    </div>
    <div class="card">
      <h2>Chart 2: Prediction Time by Model (Shared Models)</h2>
      <canvas id="chartPredictShared"></canvas>
    </div>
    <div class="card">
      <h2>Chart 3: Rust-Only Models (Training + Prediction)</h2>
      <canvas id="chartRustOnly"></canvas>
    </div>
    <div class="card">
      <h2>Chart 4: Speedup Factor for Shared Models (Python / Rust)</h2>
      <canvas id="chartSpeedup"></canvas>
    </div>
  </div>

  <div class="card" style="margin-bottom: 2rem;">
    <h2>Raw Measurements</h2>
    <table>
      <thead>
        <tr>
          <th>Platform</th>
          <th>Model</th>
          <th>Train (ms)</th>
          <th>Predict (ms)</th>
          <th>Success</th>
          <th>Accuracy</th>
          <th>Error</th>
        </tr>
      </thead>
      <tbody>${tableRows}
      </tbody>
    </table>
  </div>

  <p class="timestamp">Generated: ${new Date().toISOString()}</p>

  <script>
    const rustColor = 'rgba(96, 165, 250, 0.8)';
    const rustBorder = 'rgba(96, 165, 250, 1)';
    const pyColor = 'rgba(251, 146, 60, 0.8)';
    const pyBorder = 'rgba(251, 146, 60, 1)';
    const greenColor = 'rgba(52, 211, 153, 0.8)';
    const greenBorder = 'rgba(52, 211, 153, 1)';

    // Chart 1: Training Time — Shared Models (grouped bar)
    new Chart(document.getElementById('chartTrainShared'), {
      type: 'bar',
      data: {
        labels: ${sharedLabels},
        datasets: [
          { label: 'Rust', data: ${sharedRustTrain}, backgroundColor: rustColor, borderColor: rustBorder, borderWidth: 1 },
          { label: 'Python', data: ${sharedPythonTrain}, backgroundColor: pyColor, borderColor: pyBorder, borderWidth: 1 },
        ],
      },
      options: {
        responsive: true,
        plugins: { legend: { labels: { color: '#e2e8f0' } } },
        scales: {
          x: { ticks: { color: '#64748b', maxRotation: 45 }, grid: { color: '#1e293b' } },
          y: { title: { display: true, text: 'Training Time (ms)', color: '#94a3b8' }, ticks: { color: '#64748b' }, grid: { color: '#1e293b' }, beginAtZero: true },
        },
      },
    });

    // Chart 2: Prediction Time — Shared Models (grouped bar)
    new Chart(document.getElementById('chartPredictShared'), {
      type: 'bar',
      data: {
        labels: ${sharedLabels},
        datasets: [
          { label: 'Rust (batch 50)', data: ${sharedRustPredict}, backgroundColor: rustColor, borderColor: rustBorder, borderWidth: 1 },
          { label: 'Python (single)', data: ${sharedPythonPredict}, backgroundColor: pyColor, borderColor: pyBorder, borderWidth: 1 },
        ],
      },
      options: {
        responsive: true,
        plugins: { legend: { labels: { color: '#e2e8f0' } } },
        scales: {
          x: { ticks: { color: '#64748b', maxRotation: 45 }, grid: { color: '#1e293b' } },
          y: { title: { display: true, text: 'Prediction Time (ms)', color: '#94a3b8' }, ticks: { color: '#64748b' }, grid: { color: '#1e293b' }, beginAtZero: true },
        },
      },
    });

    // Chart 3: Rust-Only Models (horizontal bar — train + predict stacked)
    new Chart(document.getElementById('chartRustOnly'), {
      type: 'bar',
      data: {
        labels: ${rustOnlyLabels},
        datasets: [
          { label: 'Training', data: ${rustOnlyTrain}, backgroundColor: rustColor, borderColor: rustBorder, borderWidth: 1 },
          { label: 'Prediction', data: ${rustOnlyPredict}, backgroundColor: greenColor, borderColor: greenBorder, borderWidth: 1 },
        ],
      },
      options: {
        indexAxis: 'y',
        responsive: true,
        plugins: { legend: { labels: { color: '#e2e8f0' } } },
        scales: {
          x: { title: { display: true, text: 'Time (ms)', color: '#94a3b8' }, ticks: { color: '#64748b' }, grid: { color: '#1e293b' }, beginAtZero: true },
          y: { ticks: { color: '#64748b' }, grid: { color: '#1e293b' } },
        },
      },
    });

    // Chart 4: Speedup Factor (bar chart)
    new Chart(document.getElementById('chartSpeedup'), {
      type: 'bar',
      data: {
        labels: ${speedupLabels},
        datasets: [
          { label: 'Training Speedup', data: ${trainSpeedups}, backgroundColor: rustColor, borderColor: rustBorder, borderWidth: 1 },
          { label: 'Prediction Speedup', data: ${predictSpeedups}, backgroundColor: greenColor, borderColor: greenBorder, borderWidth: 1 },
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
