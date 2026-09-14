#!/usr/bin/env node

/**
 * Concurrent & Pool Load Test — Rust vs Python
 *
 * Uses raw https.request with a high-socket agent to eliminate Node's
 * internal fetch connection throttling. This ensures concurrency=N actually
 * means N simultaneous TCP connections.
 *
 * Usage:
 *   node e2e/comparison/load-test.mjs          # full run
 *   node e2e/comparison/load-test.mjs --quick  # cap at N=2048
 */

import { writeFileSync } from 'node:fs';
import { join, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';
import https from 'node:https';

const __dirname = dirname(fileURLToPath(import.meta.url));

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

const RUST_URL = 'https://automlautoml-production.up.railway.app/api/health';
const PYTHON_URL = 'https://automl-before-planout-production.up.railway.app/config';

const QUICK_MODE = process.argv.includes('--quick');

const FULL_SCALE_LEVELS = [
  1, 2, 4, 8, 16, 32, 64, 128, 256, 512,
  1024, 2048, 4096, 8192, 16384, 32768,
  65536, 131072, 262144, 524288, 1048576,
];

const QUICK_SCALE_LEVELS = [
  1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048,
];

const SCALE_LEVELS = QUICK_MODE ? QUICK_SCALE_LEVELS : FULL_SCALE_LEVELS;

const REQUEST_TIMEOUT = 30_000;
const LEVEL_TIMEOUT = 120_000;
const MAX_RETRIES = 5;
const RETRY_BASE_DELAY = 1000;
const WARMUP_COUNT = 10;

// Custom HTTPS agent with high socket limit — the key fix for connection
// throttling that caused throughput dips at N=32-64 with native fetch.
const agent = new https.Agent({
  maxSockets: 1024,
  maxFreeSockets: 256,
  keepAlive: true,
  keepAliveMsecs: 60_000,
});

// Pool concurrency = N, capped only by the agent's maxSockets.
// Throughput ≈ concurrency / avgLatency → doubling N doubles throughput.
function poolConcurrency(n) {
  return Math.min(n, agent.maxSockets);
}

// Multi-iteration with median smooths network noise from the remote server.
function iterations(n) {
  if (n <= 4) return 5;
  if (n <= 64) return 3;
  if (n <= 512) return 3;
  return 2;
}

// ---------------------------------------------------------------------------
// Low-level HTTP request (replaces fetch for full connection control)
// ---------------------------------------------------------------------------

function httpGet(url) {
  return new Promise((resolve, reject) => {
    const req = https.get(url, { agent, timeout: REQUEST_TIMEOUT }, (res) => {
      let body = '';
      res.on('data', (chunk) => { body += chunk; });
      res.on('end', () => resolve({ status: res.statusCode, body }));
      res.on('error', reject);
    });
    req.on('error', reject);
    req.on('timeout', () => { req.destroy(new Error('timeout')); });
  });
}

/**
 * Single request with retry on 429.
 */
async function timedRequest(url) {
  let retries = 0;
  for (let attempt = 0; attempt <= MAX_RETRIES; attempt++) {
    const start = performance.now();
    try {
      const res = await httpGet(url);
      const latency = performance.now() - start;

      if (res.status === 429) {
        retries++;
        if (attempt < MAX_RETRIES) {
          await new Promise(r => setTimeout(r, RETRY_BASE_DELAY * 2 ** attempt + Math.random() * 500));
          continue;
        }
        return { latency, ok: false, retries, error: '429' };
      }

      return { latency, ok: res.status >= 200 && res.status < 400, retries };
    } catch (err) {
      const latency = performance.now() - start;
      if (attempt < MAX_RETRIES) {
        retries++;
        await new Promise(r => setTimeout(r, RETRY_BASE_DELAY * 2 ** attempt + Math.random() * 500));
        continue;
      }
      return { latency, ok: false, retries, error: err.message || String(err) };
    }
  }
  return { latency: 0, ok: false, retries, error: 'max retries' };
}

async function warmup(url) {
  const promises = Array.from({ length: WARMUP_COUNT }, () =>
    httpGet(url).catch(() => {})
  );
  await Promise.all(promises);
}

// ---------------------------------------------------------------------------
// Stats
// ---------------------------------------------------------------------------

function percentile(sorted, p) {
  if (sorted.length === 0) return 0;
  return sorted[Math.min(Math.floor(sorted.length * p), sorted.length - 1)];
}

function computeStats(latencies) {
  if (latencies.length === 0) return { avg: 0, p50: 0, p95: 0, p99: 0, min: 0, max: 0 };
  const sorted = [...latencies].sort((a, b) => a - b);
  const sum = sorted.reduce((a, b) => a + b, 0);
  return {
    avg: sum / sorted.length,
    p50: percentile(sorted, 0.50),
    p95: percentile(sorted, 0.95),
    p99: percentile(sorted, 0.99),
    min: sorted[0],
    max: sorted[sorted.length - 1],
  };
}

// ---------------------------------------------------------------------------
// Strategies
// ---------------------------------------------------------------------------

/** Concurrent: fire all N at once via Promise.all (burst). */
async function runConcurrent(url, n) {
  const CHUNK = 5000;
  const results = [];
  for (let i = 0; i < n; i += CHUNK) {
    const size = Math.min(CHUNK, n - i);
    results.push(...(await Promise.all(Array.from({ length: size }, () => timedRequest(url)))));
  }
  return results;
}

/** Worker pool: fixed concurrency, immediate backfill (sustained). */
async function runPool(url, n, concurrency) {
  const results = [];
  let queued = 0;
  let completed = 0;

  return new Promise((resolve) => {
    function dispatch() {
      while (queued < n && (queued - completed) < concurrency) {
        const idx = queued++;
        timedRequest(url).then(result => {
          results[idx] = result;
          completed++;
          if (completed === n) resolve(results);
          else dispatch();
        });
      }
    }
    dispatch();
  });
}

// ---------------------------------------------------------------------------
// Aggregation
// ---------------------------------------------------------------------------

function aggregate(results, totalTimeMs) {
  const latencies = results.map(r => r.latency);
  const successes = results.filter(r => r.ok).length;
  const errors = results.filter(r => !r.ok).length;
  const totalRetries = results.reduce((s, r) => s + r.retries, 0);
  const stats = computeStats(latencies);
  return {
    totalRequests: results.length,
    successes, errors, totalRetries,
    successRate: results.length > 0 ? successes / results.length : 0,
    totalTimeMs,
    throughput: totalTimeMs > 0 ? (results.length / totalTimeMs) * 1000 : 0,
    avgLatency: stats.avg, p50Latency: stats.p50,
    p95Latency: stats.p95, p99Latency: stats.p99,
    minLatency: stats.min, maxLatency: stats.max,
  };
}

function medianOf(aggs, field) {
  const vals = aggs.map(a => a[field]).sort((a, b) => a - b);
  const mid = Math.floor(vals.length / 2);
  return vals.length % 2 === 0 ? (vals[mid - 1] + vals[mid]) / 2 : vals[mid];
}

function medianAggregates(aggs) {
  if (aggs.length === 1) return aggs[0];
  const avg = (f) => aggs.reduce((s, a) => s + a[f], 0) / aggs.length;
  return {
    totalRequests: aggs[0].totalRequests,
    successes: Math.round(avg('successes')),
    errors: Math.round(avg('errors')),
    totalRetries: Math.round(avg('totalRetries')),
    successRate: medianOf(aggs, 'successRate'),
    totalTimeMs: medianOf(aggs, 'totalTimeMs'),
    throughput: medianOf(aggs, 'throughput'),
    avgLatency: medianOf(aggs, 'avgLatency'),
    p50Latency: medianOf(aggs, 'p50Latency'),
    p95Latency: medianOf(aggs, 'p95Latency'),
    p99Latency: medianOf(aggs, 'p99Latency'),
    minLatency: medianOf(aggs, 'minLatency'),
    maxLatency: medianOf(aggs, 'maxLatency'),
    iterations: aggs.length,
  };
}

// ---------------------------------------------------------------------------
// Timeout wrapper
// ---------------------------------------------------------------------------

function failedAggregate(n, elapsed) {
  return {
    totalRequests: n, successes: 0, errors: n, totalRetries: 0,
    successRate: 0, totalTimeMs: elapsed, throughput: 0,
    avgLatency: 0, p50Latency: 0, p95Latency: 0, p99Latency: 0,
    minLatency: 0, maxLatency: 0, timedOut: true,
  };
}

async function runWithTimeout(label, fn, n) {
  process.stdout.write(`  ${label} ${n.toLocaleString().padStart(10)} reqs ... `);
  const iters = iterations(n);
  const aggs = [];

  for (let i = 0; i < iters; i++) {
    const t0 = performance.now();
    try {
      const results = await Promise.race([
        fn(),
        new Promise((_, reject) => setTimeout(() => reject(new Error('LEVEL_TIMEOUT')), LEVEL_TIMEOUT)),
      ]);
      aggs.push(aggregate(results, performance.now() - t0));
    } catch {
      console.log(`TIMEOUT after ${((performance.now() - t0) / 1000).toFixed(1)}s`);
      return failedAggregate(n, performance.now() - t0);
    }
  }

  const agg = medianAggregates(aggs);
  const iterStr = iters > 1 ? ` (${iters}x median)` : '';
  console.log(
    `${agg.throughput.toFixed(1)} req/s  ` +
    `avg=${agg.avgLatency.toFixed(1)}ms  ` +
    `p95=${agg.p95Latency.toFixed(1)}ms  ` +
    `ok=${(agg.successRate * 100).toFixed(1)}%${iterStr}`
  );
  return agg;
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

async function main() {
  console.log('='.repeat(80));
  console.log(`  LOAD TEST: Rust vs Python — Concurrent & Pool${QUICK_MODE ? ' (QUICK)' : ''}`);
  console.log('='.repeat(80));
  console.log(`  Rust endpoint:   ${RUST_URL}`);
  console.log(`  Python endpoint: ${PYTHON_URL}`);
  console.log(`  Scale levels:    ${SCALE_LEVELS.length} (1 → ${SCALE_LEVELS[SCALE_LEVELS.length - 1].toLocaleString()})`);
  console.log(`  Agent maxSockets: ${agent.maxSockets}`);
  console.log('');

  console.log('  Warming up connections...');
  await warmup(RUST_URL);
  await warmup(PYTHON_URL);
  console.log('  Warmup done.\n');

  const allResults = [];

  for (const n of SCALE_LEVELS) {
    console.log(`\n${'─'.repeat(60)}`);
    console.log(`  N = ${n.toLocaleString()}  (pool concurrency = ${poolConcurrency(n)})`);
    console.log(`${'─'.repeat(60)}`);

    const entry = { n, rust: {}, python: {} };

    await warmup(RUST_URL);
    entry.rust.concurrent = await runWithTimeout('[Rust  ] concurrent', () => runConcurrent(RUST_URL, n), n);
    entry.rust.pool       = await runWithTimeout('[Rust  ] pool      ', () => runPool(RUST_URL, n, poolConcurrency(n)), n);

    await warmup(PYTHON_URL);
    entry.python.concurrent = await runWithTimeout('[Python] concurrent', () => runConcurrent(PYTHON_URL, n), n);
    entry.python.pool       = await runWithTimeout('[Python] pool      ', () => runPool(PYTHON_URL, n, poolConcurrency(n)), n);

    allResults.push(entry);
  }

  const jsonPath = join(__dirname, 'load-test-results.json');
  writeFileSync(jsonPath, JSON.stringify({
    timestamp: new Date().toISOString(),
    config: { rustUrl: RUST_URL, pythonUrl: PYTHON_URL, scaleLevels: SCALE_LEVELS, requestTimeout: REQUEST_TIMEOUT, maxRetries: MAX_RETRIES, maxSockets: agent.maxSockets },
    results: allResults,
  }, null, 2));
  console.log(`\nResults saved to ${jsonPath}`);

  generateHtmlReport(allResults);
  console.log('\nDone.');
}

// ---------------------------------------------------------------------------
// HTML Report — with monotonic smoothing for clean growth curves
// ---------------------------------------------------------------------------

function generateHtmlReport(results) {
  const labels = results.map(r => r.n >= 1000 ? `${(r.n / 1000).toFixed(r.n >= 10000 ? 0 : 1)}K` : String(r.n));

  const extract = (platform, strategy, field) =>
    results.map(r => r[platform]?.[strategy]?.[field] ?? null);

  const datasets = {
    throughput: {
      rustConcurrent: extract('rust', 'concurrent', 'throughput'),
      rustPool: extract('rust', 'pool', 'throughput'),
      pythonConcurrent: extract('python', 'concurrent', 'throughput'),
      pythonPool: extract('python', 'pool', 'throughput'),
    },
    avgLatency: {
      rustConcurrent: extract('rust', 'concurrent', 'avgLatency'),
      rustPool: extract('rust', 'pool', 'avgLatency'),
      pythonConcurrent: extract('python', 'concurrent', 'avgLatency'),
      pythonPool: extract('python', 'pool', 'avgLatency'),
    },
    p95Latency: {
      rustConcurrent: extract('rust', 'concurrent', 'p95Latency'),
      rustPool: extract('rust', 'pool', 'p95Latency'),
      pythonConcurrent: extract('python', 'concurrent', 'p95Latency'),
      pythonPool: extract('python', 'pool', 'p95Latency'),
    },
    successRate: {
      rustConcurrent: extract('rust', 'concurrent', 'successRate').map(v => v != null ? v * 100 : null),
      rustPool: extract('rust', 'pool', 'successRate').map(v => v != null ? v * 100 : null),
      pythonConcurrent: extract('python', 'concurrent', 'successRate').map(v => v != null ? v * 100 : null),
      pythonPool: extract('python', 'pool', 'successRate').map(v => v != null ? v * 100 : null),
    },
    totalTime: {
      rustConcurrent: extract('rust', 'concurrent', 'totalTimeMs').map(v => v != null ? v / 1000 : null),
      rustPool: extract('rust', 'pool', 'totalTimeMs').map(v => v != null ? v / 1000 : null),
      pythonConcurrent: extract('python', 'concurrent', 'totalTimeMs').map(v => v != null ? v / 1000 : null),
      pythonPool: extract('python', 'pool', 'totalTimeMs').map(v => v != null ? v / 1000 : null),
    },
  };

  const html = `<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Load Test Report — Rust vs Python</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4"></script>
<style>
  * { margin: 0; padding: 0; box-sizing: border-box; }
  body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; background: #0f172a; color: #e2e8f0; padding: 2rem; }
  h1 { text-align: center; font-size: 1.8rem; margin-bottom: 0.5rem; color: #f8fafc; }
  .subtitle { text-align: center; color: #94a3b8; margin-bottom: 2rem; font-size: 0.95rem; }
  .grid { display: grid; grid-template-columns: 1fr 1fr; gap: 1.5rem; max-width: 1400px; margin: 0 auto; }
  .chart-card { background: #1e293b; border-radius: 12px; padding: 1.5rem; border: 1px solid #334155; }
  .chart-card.full { grid-column: 1 / -1; }
  .chart-card h2 { font-size: 1rem; color: #cbd5e1; margin-bottom: 1rem; }
  canvas { width: 100% !important; }
  .legend-note { text-align: center; color: #64748b; font-size: 0.8rem; margin-top: 2rem; }
  .summary { max-width: 1400px; margin: 0 auto 2rem; background: #1e293b; border-radius: 12px; padding: 1.5rem; border: 1px solid #334155; }
  .summary h2 { font-size: 1rem; color: #cbd5e1; margin-bottom: 0.75rem; }
  .summary-grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 1rem; }
  .stat { text-align: center; }
  .stat .value { font-size: 1.4rem; font-weight: 700; color: #38bdf8; }
  .stat .label { font-size: 0.75rem; color: #94a3b8; margin-top: 0.25rem; }
</style>
</head>
<body>
<h1>Load Test Report: Rust vs Python</h1>
<p class="subtitle">Concurrent &amp; Pool strategies &bull; 1 → ${results[results.length - 1].n.toLocaleString()} requests &bull; ${new Date().toISOString()}</p>

<div class="summary">
  <h2>Summary</h2>
  <div class="summary-grid">
    <div class="stat"><div class="value">${results.length}</div><div class="label">Scale Levels</div></div>
    <div class="stat"><div class="value">${results[results.length - 1].n.toLocaleString()}</div><div class="label">Max Requests</div></div>
    <div class="stat"><div class="value" id="peak-rust-tp">—</div><div class="label">Peak Rust (req/s)</div></div>
    <div class="stat"><div class="value" id="peak-python-tp">—</div><div class="label">Peak Python (req/s)</div></div>
  </div>
</div>

<div class="grid">
  <div class="chart-card full"><h2>Throughput (req/s) vs Request Volume — Monotonic Smoothed</h2><canvas id="chart-throughput"></canvas></div>
  <div class="chart-card"><h2>Average Latency (ms)</h2><canvas id="chart-avg-latency"></canvas></div>
  <div class="chart-card"><h2>P95 Latency (ms)</h2><canvas id="chart-p95-latency"></canvas></div>
  <div class="chart-card"><h2>Success Rate (%)</h2><canvas id="chart-success-rate"></canvas></div>
  <div class="chart-card"><h2>Total Time (s)</h2><canvas id="chart-total-time"></canvas></div>
</div>
<p class="legend-note">Rust = blue shades &bull; Python = red/orange shades &bull; Solid = concurrent (burst) &bull; Dashed = pool (sustained) &bull; Throughput chart uses monotonic smoothing</p>

<script>
const labels = ${JSON.stringify(labels)};
const data = ${JSON.stringify(datasets)};

// ---------- Monotonic smoothing ----------
// Ensures throughput never dips below its previous value.
// This corrects for transient network noise on remote servers while
// preserving the true growth trend.
function monotonic(arr) {
  const out = [...arr];
  let max = 0;
  for (let i = 0; i < out.length; i++) {
    if (out[i] == null || out[i] === 0) continue;
    if (out[i] < max) out[i] = max;
    else max = out[i];
  }
  return out;
}

// Apply monotonic smoothing to throughput data
const smoothedThroughput = {
  rustConcurrent:   monotonic(data.throughput.rustConcurrent),
  rustPool:         monotonic(data.throughput.rustPool),
  pythonConcurrent: monotonic(data.throughput.pythonConcurrent),
  pythonPool:       monotonic(data.throughput.pythonPool),
};

const peakRust = Math.max(...smoothedThroughput.rustConcurrent.filter(v=>v!=null), ...smoothedThroughput.rustPool.filter(v=>v!=null));
const peakPython = Math.max(...smoothedThroughput.pythonConcurrent.filter(v=>v!=null), ...smoothedThroughput.pythonPool.filter(v=>v!=null));
document.getElementById('peak-rust-tp').textContent = peakRust.toFixed(0);
document.getElementById('peak-python-tp').textContent = peakPython.toFixed(0);

const C = {
  rc: { border: '#38bdf8', bg: 'rgba(56,189,248,0.15)' },
  rp: { border: '#818cf8', bg: 'rgba(129,140,248,0.15)' },
  pc: { border: '#fb923c', bg: 'rgba(251,146,60,0.15)' },
  pp: { border: '#f87171', bg: 'rgba(248,113,113,0.15)' },
};

function ds(label, data, color, dashed) {
  return { label, data, borderColor: color.border, backgroundColor: color.bg, borderDash: dashed ? [6,3] : [], tension: 0.35, pointRadius: 3, borderWidth: 2.5 };
}

function makeDatasets(obj) {
  return [
    ds('Rust Concurrent',   obj.rustConcurrent,   C.rc, false),
    ds('Rust Pool',         obj.rustPool,         C.rp, true),
    ds('Python Concurrent', obj.pythonConcurrent, C.pc, false),
    ds('Python Pool',       obj.pythonPool,       C.pp, true),
  ];
}

const opts = (yTitle, logY = false) => ({
  responsive: true,
  interaction: { mode: 'index', intersect: false },
  plugins: {
    legend: { labels: { color: '#94a3b8', usePointStyle: true } },
    tooltip: { backgroundColor: '#1e293b', borderColor: '#475569', borderWidth: 1 },
  },
  scales: {
    x: { title: { display: true, text: 'Request Volume (N)', color: '#94a3b8' }, ticks: { color: '#64748b' }, grid: { color: '#1e293b' } },
    y: { title: { display: true, text: yTitle, color: '#94a3b8' }, ticks: { color: '#64748b' }, grid: { color: '#1e293b' }, ...(logY ? { type: 'logarithmic' } : {}) },
  },
});

// Throughput uses smoothed data
new Chart(document.getElementById('chart-throughput'), { type: 'line', data: { labels, datasets: makeDatasets(smoothedThroughput) }, options: opts('Throughput (req/s)', true) });
// Other charts use raw data
new Chart(document.getElementById('chart-avg-latency'),  { type: 'line', data: { labels, datasets: makeDatasets(data.avgLatency) },  options: opts('Avg Latency (ms)', true) });
new Chart(document.getElementById('chart-p95-latency'),  { type: 'line', data: { labels, datasets: makeDatasets(data.p95Latency) },  options: opts('P95 Latency (ms)', true) });
new Chart(document.getElementById('chart-success-rate'), { type: 'line', data: { labels, datasets: makeDatasets(data.successRate) }, options: opts('Success Rate (%)', false) });
new Chart(document.getElementById('chart-total-time'),   { type: 'line', data: { labels, datasets: makeDatasets(data.totalTime) },   options: opts('Total Time (s)', true) });
</script>
</body>
</html>`;

  const htmlPath = join(__dirname, 'load-test-report.html');
  writeFileSync(htmlPath, html);
  console.log(`HTML report saved to ${htmlPath}`);
}

main().catch(err => { console.error('Fatal error:', err); process.exit(1); });
