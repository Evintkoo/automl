import { Page } from '@playwright/test';
import * as fs from 'fs';
import * as path from 'path';

// ---------------------------------------------------------------------------
// Platform abstraction — encapsulates selector differences between Rust & Python
// ---------------------------------------------------------------------------

export interface Platform {
  name: 'rust' | 'python';
  baseURL: string;
  healthEndpoint: string;
  selectors: {
    sampleDatasetIris: string;
    dataLoadedIndicator: string;
    tabs: Record<string, string>;
    training: {
      taskTypeSelect: string;
      targetColumnSelect: string;
      modelSelect: string;
      trainButton: string;
      trainingComplete: string;
    };
    prediction: {
      inputField: string;
      submitButton: string;
      resultContainer: string;
    };
  };
}

export const RUST_PLATFORM: Platform = {
  name: 'rust',
  baseURL: 'https://automlautoml-production.up.railway.app',
  healthEndpoint: '/api/health',
  selectors: {
    sampleDatasetIris: 'button.pill:has-text("Iris")',
    dataLoadedIndicator: '#data-rows',
    tabs: {
      data: 'button.nav-tab[data-tab="data"]',
      train: 'button.nav-tab[data-tab="train"]',
      compare: 'button.nav-tab[data-tab="compare"]',
      predict: 'button.nav-tab[data-tab="predict"]',
      analyze: 'button.nav-tab[data-tab="analyze"]',
      serve: 'button.nav-tab[data-tab="serve"]',
      system: 'button.nav-tab[data-tab="system"]',
    },
    training: {
      taskTypeSelect: '#cfg-taskType',
      targetColumnSelect: '#cfg-targetColumn',
      modelSelect: '#model-checkboxes input[type="checkbox"]',
      trainButton: '#btn-train',
      trainingComplete: '#train-results',
    },
    prediction: {
      inputField: '#prediction-input',
      submitButton: '#btn-predict',
      resultContainer: '#prediction-output',
    },
  },
};

export const PYTHON_PLATFORM: Platform = {
  name: 'python',
  baseURL: 'https://automl-before-planout-production.up.railway.app',
  healthEndpoint: '/',
  selectors: {
    sampleDatasetIris: 'label:has-text("Sample Dataset") + div input, [data-testid="dropdown"]',
    dataLoadedIndicator: '.table-wrap, .dataframe, [data-testid="dataframe"]',
    tabs: {
      data: 'button:has-text("Step 1")',
      train: 'button:has-text("Step 3")',
      configure: 'button:has-text("Step 2")',
      predict: 'button:has-text("Step 4")',
      compare: 'button:has-text("Step 5")',
      system: 'button:has-text("System")',
    },
    training: {
      taskTypeSelect: '[data-testid="dropdown"]:near(:text("Task Type"))',
      targetColumnSelect: '#component-64 textarea, [data-testid="textbox"]:near(:text("Target"))',
      modelSelect: '[data-testid="dropdown"]:near(:text("Algorithm"))',
      trainButton: 'button:has-text("Start Multi-Model Training")',
      trainingComplete: '.prose, [data-testid="html"]:has-text("Training")',
    },
    prediction: {
      inputField: '#component-90 textarea, [data-testid="textbox"]:near(:text("Feature"))',
      submitButton: 'button:has-text("Make Prediction")',
      resultContainer: '#component-98, [data-testid="html"]:near(:text("Prediction"))',
    },
  },
};

export const PLATFORMS: Platform[] = [RUST_PLATFORM, PYTHON_PLATFORM];

// ---------------------------------------------------------------------------
// Statistical utilities
// ---------------------------------------------------------------------------

export interface Stats {
  min: number;
  max: number;
  avg: number;
  p50: number;
  p95: number;
  p99: number;
  stddev: number;
  cv: number;         // coefficient of variation (stddev/avg * 100%)
  ci95_lower: number; // 95% confidence interval lower bound
  ci95_upper: number; // 95% confidence interval upper bound
  n: number;          // sample count
  values: number[];
}

export function computeStats(values: number[]): Stats {
  const n = values.length;
  if (n === 0) {
    return { min: 0, max: 0, avg: 0, p50: 0, p95: 0, p99: 0, stddev: 0, cv: 0, ci95_lower: 0, ci95_upper: 0, n: 0, values: [] };
  }

  const sorted = [...values].sort((a, b) => a - b);
  const sum = sorted.reduce((a, b) => a + b, 0);
  const avg = sum / n;

  // Standard deviation (sample)
  const variance = n > 1
    ? sorted.reduce((acc, v) => acc + (v - avg) ** 2, 0) / (n - 1)
    : 0;
  const stddev = Math.sqrt(variance);

  // Coefficient of variation (%)
  const cv = avg > 0 ? (stddev / avg) * 100 : 0;

  // 95% confidence interval (t-distribution approximation for small samples)
  // For small n, use t-value approximation
  const tValues: Record<number, number> = { 1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228 };
  const t = tValues[n] || (n <= 30 ? 2.042 : 1.96);
  const margin = t * (stddev / Math.sqrt(n));

  return {
    min: sorted[0],
    max: sorted[n - 1],
    avg,
    p50: sorted[Math.floor(n * 0.5)],
    p95: sorted[Math.min(Math.floor(n * 0.95), n - 1)],
    p99: sorted[Math.min(Math.floor(n * 0.99), n - 1)],
    stddev,
    cv,
    ci95_lower: avg - margin,
    ci95_upper: avg + margin,
    n,
    values: sorted,
  };
}

/** Cohen's d effect size between two sample arrays */
export function cohensD(a: number[], b: number[]): number {
  if (a.length === 0 || b.length === 0) return 0;
  const avgA = a.reduce((s, v) => s + v, 0) / a.length;
  const avgB = b.reduce((s, v) => s + v, 0) / b.length;
  const varA = a.length > 1 ? a.reduce((s, v) => s + (v - avgA) ** 2, 0) / (a.length - 1) : 0;
  const varB = b.length > 1 ? b.reduce((s, v) => s + (v - avgB) ** 2, 0) / (b.length - 1) : 0;
  const pooledSD = Math.sqrt(((a.length - 1) * varA + (b.length - 1) * varB) / (a.length + b.length - 2));
  return pooledSD > 0 ? (avgA - avgB) / pooledSD : 0;
}

/** Interpret Cohen's d magnitude */
export function effectSizeLabel(d: number): string {
  const abs = Math.abs(d);
  if (abs < 0.2) return 'negligible';
  if (abs < 0.5) return 'small';
  if (abs < 0.8) return 'medium';
  if (abs < 1.2) return 'large';
  return 'very large';
}

// ---------------------------------------------------------------------------
// Rate-limit-aware fetch helper
// ---------------------------------------------------------------------------

/**
 * Fetch with automatic retry on 429 (Too Many Requests).
 * Uses exponential backoff with jitter.
 */
export async function fetchWithRetry(
  input: RequestInfo | URL,
  init?: RequestInit,
  maxRetries: number = 5,
  baseDelay: number = 1000,
): Promise<Response> {
  for (let attempt = 0; attempt <= maxRetries; attempt++) {
    const res = await fetch(input, init);
    if (res.status !== 429) return res;

    if (attempt === maxRetries) return res; // return the 429 on final attempt

    // Exponential backoff with jitter
    const delay = baseDelay * Math.pow(2, attempt) + Math.random() * 500;
    await new Promise(r => setTimeout(r, delay));
  }
  // unreachable, but TypeScript needs it
  return fetch(input, init);
}

/** Small delay to avoid hitting rate limits between rapid requests */
export async function rateLimitPause(ms: number = 300): Promise<void> {
  await new Promise(r => setTimeout(r, ms));
}

// ---------------------------------------------------------------------------
// Timing utilities
// ---------------------------------------------------------------------------

export interface TimingResult {
  platform: string;
  metric: string;
  value: number;
  unit: string;
  iteration?: number;
}

export async function measureTime<T>(fn: () => Promise<T>): Promise<{ result: T; elapsed: number }> {
  const start = Date.now();
  const result = await fn();
  const elapsed = Date.now() - start;
  return { result, elapsed };
}

export async function benchmark(
  fn: () => Promise<void>,
  iterations: number = 3,
): Promise<Stats> {
  const times: number[] = [];
  for (let i = 0; i < iterations; i++) {
    const { elapsed } = await measureTime(fn);
    times.push(elapsed);
  }
  return computeStats(times);
}

// ---------------------------------------------------------------------------
// Page load performance helpers
// ---------------------------------------------------------------------------

export interface PageLoadMetrics {
  ttfb: number;
  domContentLoaded: number;
  fullLoad: number;
  transferSize: number;
  requestCount: number;
}

export async function collectPageLoadMetrics(page: Page, url: string): Promise<PageLoadMetrics> {
  let requestCount = 0;
  let transferSize = 0;

  page.on('response', (response) => {
    requestCount++;
    const headers = response.headers();
    const contentLength = parseInt(headers['content-length'] || '0', 10);
    transferSize += contentLength;
  });

  await page.goto(url, { waitUntil: 'load', timeout: 120_000 });

  const timing = await page.evaluate(() => {
    const nav = performance.getEntriesByType('navigation')[0] as PerformanceNavigationTiming;
    return {
      ttfb: nav.responseStart - nav.requestStart,
      domContentLoaded: nav.domContentLoadedEventEnd - nav.requestStart,
      fullLoad: nav.loadEventEnd - nav.requestStart,
    };
  });

  return {
    ...timing,
    transferSize,
    requestCount,
  };
}

// ---------------------------------------------------------------------------
// Report generation — enhanced with detailed statistics
// ---------------------------------------------------------------------------

export interface ComparisonEntry {
  metric: string;
  category: string;
  rust: number;
  python: number;
  unit: string;
  improvement: string;
  rustOnly: boolean;
  // Raw sample arrays for detailed statistics
  rustSamples?: number[];
  pythonSamples?: number[];
  rustStats?: Stats;
  pythonStats?: Stats;
  effectSize?: number;
  effectLabel?: string;
}

export interface ComparisonReport {
  timestamp: string;
  summary: {
    totalMetrics: number;
    headToHeadMetrics: number;
    rustOnlyMetrics: number;
    rustWins: number;
    pythonWins: number;
    ties: number;
    avgSpeedupFactor: number;
    categories: string[];
  };
  entries: ComparisonEntry[];
}

const reportEntries: ComparisonEntry[] = [];

/** Record a comparison metric — enhanced version */
export function recordComparison(
  category: string,
  metric: string,
  rust: number,
  python: number,
  unit: string,
  lowerIsBetter: boolean = true,
  rustSamples?: number[],
  pythonSamples?: number[],
) {
  const rustOnly = python === 0 && rust > 0;
  let improvement: string;

  if (rustOnly) {
    improvement = `Rust-only (${rust.toFixed(1)}${unit})`;
  } else if (lowerIsBetter) {
    if (rust < python) {
      const factor = python / rust;
      improvement = factor >= 2
        ? `${factor.toFixed(1)}x faster`
        : `${((1 - rust / python) * 100).toFixed(1)}% faster`;
    } else if (rust > python) {
      const factor = rust / python;
      improvement = factor >= 2
        ? `${factor.toFixed(1)}x slower`
        : `${((rust / python - 1) * 100).toFixed(1)}% slower`;
    } else {
      improvement = 'equal';
    }
  } else {
    if (rust > python) {
      improvement = `+${((rust - python) * 100).toFixed(2)}pp better`;
    } else if (rust < python) {
      improvement = `-${((python - rust) * 100).toFixed(2)}pp worse`;
    } else {
      improvement = 'equal';
    }
  }

  const entry: ComparisonEntry = { metric, category, rust, python, unit, improvement, rustOnly };

  if (rustSamples && rustSamples.length > 0) {
    entry.rustSamples = rustSamples;
    entry.rustStats = computeStats(rustSamples);
  }
  if (pythonSamples && pythonSamples.length > 0) {
    entry.pythonSamples = pythonSamples;
    entry.pythonStats = computeStats(pythonSamples);
  }
  if (rustSamples && pythonSamples && rustSamples.length > 0 && pythonSamples.length > 0) {
    entry.effectSize = cohensD(rustSamples, pythonSamples);
    entry.effectLabel = effectSizeLabel(entry.effectSize);
  }

  reportEntries.push(entry);
}

/** Print the detailed statistical comparison report */
export function printComparisonTable() {
  const W = 120;
  const sep = '='.repeat(W);
  const thin = '-'.repeat(W);

  console.log('\n' + sep);
  console.log('  AUTOML PERFORMANCE COMPARISON: RUST vs PYTHON — DETAILED STATISTICAL REPORT');
  console.log(sep);

  // ---- Executive summary ----
  const headToHead = reportEntries.filter(e => !e.rustOnly);
  const rustOnlyEntries = reportEntries.filter(e => e.rustOnly);
  const rustWins = headToHead.filter(e => e.rust < e.python).length;
  const pythonWins = headToHead.filter(e => e.rust > e.python).length;
  const ties = headToHead.filter(e => e.rust === e.python).length;

  const speedups = headToHead
    .filter(e => e.rust > 0 && e.python > 0)
    .map(e => e.python / e.rust);
  const avgSpeedup = speedups.length > 0 ? speedups.reduce((a, b) => a + b, 0) / speedups.length : 0;
  const medianSpeedup = speedups.length > 0 ? computeStats(speedups).p50 : 0;
  const maxSpeedup = speedups.length > 0 ? Math.max(...speedups) : 0;
  const minSpeedup = speedups.length > 0 ? Math.min(...speedups) : 0;

  console.log(`\n  EXECUTIVE SUMMARY`);
  console.log(`  ${thin}`);
  console.log(`  Total metrics collected:     ${reportEntries.length}`);
  console.log(`  Head-to-head comparisons:    ${headToHead.length}`);
  console.log(`  Rust-only features tested:   ${rustOnlyEntries.length}`);
  console.log(`  ${thin}`);
  console.log(`  Rust wins:                   ${rustWins} / ${headToHead.length}  (${((rustWins / Math.max(headToHead.length, 1)) * 100).toFixed(1)}%)`);
  console.log(`  Python wins:                 ${pythonWins} / ${headToHead.length}  (${((pythonWins / Math.max(headToHead.length, 1)) * 100).toFixed(1)}%)`);
  console.log(`  Ties:                        ${ties}`);
  console.log(`  ${thin}`);
  console.log(`  Average speedup factor:      ${avgSpeedup.toFixed(2)}x  (Rust faster)`);
  console.log(`  Median speedup factor:       ${medianSpeedup.toFixed(2)}x`);
  console.log(`  Max speedup:                 ${maxSpeedup.toFixed(1)}x`);
  console.log(`  Min speedup:                 ${minSpeedup.toFixed(2)}x`);

  // ---- Head-to-head detailed table ----
  console.log(`\n\n  HEAD-TO-HEAD COMPARISON (Rust vs Python)`);
  console.log(`  ${sep}`);

  const h2hCategories = [...new Set(headToHead.map(e => e.category))];

  for (const cat of h2hCategories) {
    const entries = headToHead.filter(e => e.category === cat);
    console.log(`\n  [${cat.toUpperCase()}]`);
    console.log(`  ${thin}`);

    // Header
    const hdr =
      '  ' +
      'Metric'.padEnd(30) +
      'Rust'.padStart(12) +
      'Python'.padStart(12) +
      'Unit'.padStart(8) +
      'Speedup'.padStart(10) +
      'Effect'.padStart(14) +
      'Result'.padStart(22);
    console.log(hdr);
    console.log(`  ${thin}`);

    for (const e of entries) {
      const rustStr = e.rust.toFixed(2);
      const pyStr = e.python.toFixed(2);
      const speedup = e.python > 0 && e.rust > 0 ? (e.python / e.rust).toFixed(2) + 'x' : 'N/A';
      const effect = e.effectLabel ? `${e.effectLabel} (${(e.effectSize || 0).toFixed(2)})` : '—';
      const line =
        '  ' +
        e.metric.padEnd(30) +
        rustStr.padStart(12) +
        pyStr.padStart(12) +
        e.unit.padStart(8) +
        speedup.padStart(10) +
        effect.padStart(14) +
        e.improvement.padStart(22);
      console.log(line);

      // Print detailed stats if available
      if (e.rustStats && e.pythonStats) {
        const rs = e.rustStats;
        const ps = e.pythonStats;
        console.log(
          `    Rust  :  n=${rs.n}  avg=${rs.avg.toFixed(1)}  stddev=${rs.stddev.toFixed(1)}  ` +
          `CI95=[${rs.ci95_lower.toFixed(1)}, ${rs.ci95_upper.toFixed(1)}]  ` +
          `CV=${rs.cv.toFixed(1)}%  p50=${rs.p50.toFixed(1)}  p95=${rs.p95.toFixed(1)}  range=[${rs.min.toFixed(1)}, ${rs.max.toFixed(1)}]`
        );
        console.log(
          `    Python:  n=${ps.n}  avg=${ps.avg.toFixed(1)}  stddev=${ps.stddev.toFixed(1)}  ` +
          `CI95=[${ps.ci95_lower.toFixed(1)}, ${ps.ci95_upper.toFixed(1)}]  ` +
          `CV=${ps.cv.toFixed(1)}%  p50=${ps.p50.toFixed(1)}  p95=${ps.p95.toFixed(1)}  range=[${ps.min.toFixed(1)}, ${ps.max.toFixed(1)}]`
        );
      }
    }
  }

  // ---- Rust-only features ----
  if (rustOnlyEntries.length > 0) {
    console.log(`\n\n  RUST-ONLY FEATURES (no Python equivalent)`);
    console.log(`  ${sep}`);

    const rustOnlyCategories = [...new Set(rustOnlyEntries.map(e => e.category))];

    for (const cat of rustOnlyCategories) {
      const entries = rustOnlyEntries.filter(e => e.category === cat);
      console.log(`\n  [${cat.toUpperCase()}]`);
      console.log(`  ${thin}`);

      const hdr =
        '  ' +
        'Metric'.padEnd(35) +
        'Latency'.padStart(12) +
        'Unit'.padStart(8) +
        'Stats'.padStart(50);
      console.log(hdr);
      console.log(`  ${thin}`);

      for (const e of entries) {
        let statsStr = '—';
        if (e.rustStats) {
          const rs = e.rustStats;
          statsStr = `n=${rs.n} avg=${rs.avg.toFixed(1)} stddev=${rs.stddev.toFixed(1)} CV=${rs.cv.toFixed(1)}% p50=${rs.p50.toFixed(1)} p95=${rs.p95.toFixed(1)}`;
        }
        const line =
          '  ' +
          e.metric.padEnd(35) +
          e.rust.toFixed(2).padStart(12) +
          e.unit.padStart(8) +
          statsStr.padStart(50);
        console.log(line);
      }
    }

    // Rust-only aggregate stats
    const allRustTimes = rustOnlyEntries.filter(e => e.unit === 'ms').map(e => e.rust);
    if (allRustTimes.length > 0) {
      const aggStats = computeStats(allRustTimes);
      console.log(`\n  Rust-only API Aggregate (${allRustTimes.length} endpoints):`);
      console.log(`    Avg latency:     ${aggStats.avg.toFixed(1)} ms`);
      console.log(`    Median (p50):    ${aggStats.p50.toFixed(1)} ms`);
      console.log(`    p95:             ${aggStats.p95.toFixed(1)} ms`);
      console.log(`    p99:             ${aggStats.p99.toFixed(1)} ms`);
      console.log(`    Fastest:         ${aggStats.min.toFixed(1)} ms`);
      console.log(`    Slowest:         ${aggStats.max.toFixed(1)} ms`);
      console.log(`    Stddev:          ${aggStats.stddev.toFixed(1)} ms`);

      // Latency distribution buckets
      const sub50 = allRustTimes.filter(t => t < 50).length;
      const sub100 = allRustTimes.filter(t => t >= 50 && t < 100).length;
      const sub200 = allRustTimes.filter(t => t >= 100 && t < 200).length;
      const sub500 = allRustTimes.filter(t => t >= 200 && t < 500).length;
      const sub1000 = allRustTimes.filter(t => t >= 500 && t < 1000).length;
      const over1000 = allRustTimes.filter(t => t >= 1000).length;
      console.log(`\n    Latency Distribution:`);
      console.log(`      <50ms:         ${sub50} endpoints  (${((sub50 / allRustTimes.length) * 100).toFixed(0)}%)`);
      console.log(`      50-100ms:      ${sub100} endpoints  (${((sub100 / allRustTimes.length) * 100).toFixed(0)}%)`);
      console.log(`      100-200ms:     ${sub200} endpoints  (${((sub200 / allRustTimes.length) * 100).toFixed(0)}%)`);
      console.log(`      200-500ms:     ${sub500} endpoints  (${((sub500 / allRustTimes.length) * 100).toFixed(0)}%)`);
      console.log(`      500ms-1s:      ${sub1000} endpoints  (${((sub1000 / allRustTimes.length) * 100).toFixed(0)}%)`);
      console.log(`      >1s:           ${over1000} endpoints  (${((over1000 / allRustTimes.length) * 100).toFixed(0)}%)`);
    }
  }

  console.log('\n' + sep + '\n');
}

/** Save the report to a JSON file */
export function saveReport() {
  const headToHead = reportEntries.filter(e => !e.rustOnly);
  const rustOnlyEntries = reportEntries.filter(e => e.rustOnly);

  const speedups = headToHead
    .filter(e => e.rust > 0 && e.python > 0)
    .map(e => e.python / e.rust);

  const report: ComparisonReport = {
    timestamp: new Date().toISOString(),
    summary: {
      totalMetrics: reportEntries.length,
      headToHeadMetrics: headToHead.length,
      rustOnlyMetrics: rustOnlyEntries.length,
      rustWins: headToHead.filter(e => e.rust < e.python).length,
      pythonWins: headToHead.filter(e => e.rust > e.python).length,
      ties: headToHead.filter(e => e.rust === e.python).length,
      avgSpeedupFactor: speedups.length > 0 ? speedups.reduce((a, b) => a + b, 0) / speedups.length : 0,
      categories: [...new Set(reportEntries.map(e => e.category))],
    },
    entries: reportEntries,
  };

  const outPath = path.join(__dirname, 'results.json');
  fs.writeFileSync(outPath, JSON.stringify(report, null, 2));
  console.log(`Report saved to ${outPath}`);
}

export function getEntries(): ComparisonEntry[] {
  return reportEntries;
}
