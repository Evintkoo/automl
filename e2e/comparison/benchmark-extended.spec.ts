/**
 * Extended benchmark suite: Tests all Rust API endpoints, AutoML features,
 * data analysis, model management, visualizations, ML pipelines, and more.
 *
 * These tests exercise the full breadth of the Rust AutoML platform
 * and compare with the Python/Gradio version where applicable.
 */
import { test, expect, Page } from '@playwright/test';
import {
  RUST_PLATFORM,
  PYTHON_PLATFORM,
  measureTime,
  computeStats,
  recordComparison,
  printComparisonTable,
  saveReport,
  rateLimitPause,
} from './helpers';

const LONG_TIMEOUT = 180_000;
const RUST = RUST_PLATFORM.baseURL;
const PYTHON = PYTHON_PLATFORM.baseURL;

// ---------------------------------------------------------------------------
// Shared setup: ensure Iris data is loaded + preprocessed on Rust server
// ---------------------------------------------------------------------------

/**
 * Inject a rate-limit-aware fetch wrapper into the page context.
 * Call this once per new page before using `page.evaluate` with fetch.
 */
async function injectRetryFetch(page: Page) {
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
}

async function setupRustPipeline(page: Page) {
  await injectRetryFetch(page);
  await page.evaluate(async (url: string) => {
    const rf = (window as any).__retryFetch;
    await rf(`${url}/api/data/sample/iris`);
    await new Promise(r => setTimeout(r, 300));
    await rf(`${url}/api/preprocess`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ task_type: 'classification', target_column: 'species', scaler: 'standard', imputation: 'mean', cv_folds: 5 }),
    });
  }, RUST);
}

async function trainModelRust(page: Page, modelType: string): Promise<string> {
  await injectRetryFetch(page);
  const jobId = await page.evaluate(async (args: { url: string; model: string }) => {
    const rf = (window as any).__retryFetch;
    const res = await rf(`${args.url}/api/train`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ model_type: args.model, target_column: 'species', task_type: 'classification' }),
    });
    const data = await res.json();
    return data.job_id;
  }, { url: RUST, model: modelType });

  // Poll until complete (with rate-limit awareness)
  await page.evaluate(async (args: { url: string; jobId: string }) => {
    const rf = (window as any).__retryFetch;
    for (let i = 0; i < 120; i++) {
      const res = await rf(`${args.url}/api/train/status/${args.jobId}`);
      const data = await res.json();
      if (data.status?.Completed || data.status?.Failed) return;
      await new Promise(r => setTimeout(r, 500));
    }
  }, { url: RUST, jobId });

  return jobId;
}

// ==========================================================================
// 9. DATA ANALYSIS & QUALITY API
// ==========================================================================

test.describe('9. Data Analysis & Quality', () => {
  test('data/analyze — comprehensive analysis speed', async ({ browser }) => {
    const context = await browser.newContext();
    const page = await context.newPage();
    await page.goto(RUST, { waitUntil: 'load', timeout: LONG_TIMEOUT });
    await injectRetryFetch(page);
    await setupRustPipeline(page);

    const times: number[] = [];
    for (let i = 0; i < 5; i++) {
      if (i > 0) await new Promise(r => setTimeout(r, 300));
      const { elapsed } = await measureTime(async () => {
        const res = await page.evaluate(async (url: string) => {
          const rf = (window as any).__retryFetch;
          const r = await rf(`${url}/api/data/analyze`);
          return r.json();
        }, RUST);
        expect(res.success).toBe(true);
        expect(res.column_stats.length).toBeGreaterThan(0);
      });
      times.push(elapsed);
    }
    const stats = computeStats(times);
    recordComparison('Data Analysis', 'Analyze API avg', stats.avg, 0, 'ms', true, times);
    recordComparison('Data Analysis', 'Analyze API p50', stats.p50, 0, 'ms');
    await context.close();
  });

  test('data/quality — quality metrics speed', async ({ browser }) => {
    const context = await browser.newContext();
    const page = await context.newPage();
    await page.goto(RUST, { waitUntil: 'load', timeout: LONG_TIMEOUT });
    await injectRetryFetch(page);
    await setupRustPipeline(page);

    const times: number[] = [];
    for (let i = 0; i < 5; i++) {
      if (i > 0) await new Promise(r => setTimeout(r, 300));
      const { elapsed } = await measureTime(async () => {
        const res = await page.evaluate(async (url: string) => {
          const rf = (window as any).__retryFetch;
          const r = await rf(`${url}/api/data/quality`);
          return r.json();
        }, RUST);
        expect(res.quality_report).toBeDefined();
        expect(res.quality_report.overall_score).toBeGreaterThan(0);
      });
      times.push(elapsed);
    }
    const stats = computeStats(times);
    recordComparison('Data Analysis', 'Quality API avg', stats.avg, 0, 'ms', true, times);
    await context.close();
  });

  test('data/info + data/preview speed', async ({ browser }) => {
    const context = await browser.newContext();
    const page = await context.newPage();
    await page.goto(RUST, { waitUntil: 'load', timeout: LONG_TIMEOUT });
    await injectRetryFetch(page);
    await setupRustPipeline(page);

    // data/info
    const infoTimes: number[] = [];
    for (let i = 0; i < 5; i++) {
      if (i > 0) await new Promise(r => setTimeout(r, 300));
      const { elapsed } = await measureTime(async () => {
        const res = await page.evaluate(async (url: string) => {
          const rf = (window as any).__retryFetch;
          const r = await rf(`${url}/api/data/info`);
          return r.json();
        }, RUST);
        expect(res.rows).toBe(150);
        expect(res.columns).toBe(5);
      });
      infoTimes.push(elapsed);
    }

    // data/preview
    const previewTimes: number[] = [];
    for (let i = 0; i < 5; i++) {
      if (i > 0) await new Promise(r => setTimeout(r, 300));
      const { elapsed } = await measureTime(async () => {
        const res = await page.evaluate(async (url: string) => {
          const rf = (window as any).__retryFetch;
          const r = await rf(`${url}/api/data/preview`);
          return r.json();
        }, RUST);
        expect(res.rows).toBe(10);
      });
      previewTimes.push(elapsed);
    }

    recordComparison('Data Analysis', 'Data Info API avg', computeStats(infoTimes).avg, 0, 'ms', true, infoTimes);
    recordComparison('Data Analysis', 'Data Preview API avg', computeStats(previewTimes).avg, 0, 'ms', true, previewTimes);
    await context.close();
  });

  test('data/datasheet + data/lineage', async ({ browser }) => {
    const context = await browser.newContext();
    const page = await context.newPage();
    await page.goto(RUST, { waitUntil: 'load', timeout: LONG_TIMEOUT });
    await injectRetryFetch(page);
    await setupRustPipeline(page);

    const { elapsed: datasheetTime } = await measureTime(async () => {
      const res = await page.evaluate(async (url: string) => {
        const rf = (window as any).__retryFetch;
        const r = await rf(`${url}/api/data/datasheet`);
        return r.json();
      }, RUST);
      expect(res.datasheet).toBeDefined();
      expect(res.markdown).toBeDefined();
    });

    const { elapsed: lineageTime } = await measureTime(async () => {
      const res = await page.evaluate(async (url: string) => {
        const rf = (window as any).__retryFetch;
        const r = await rf(`${url}/api/data/lineage`);
        return r.json();
      }, RUST);
      expect(res).toBeDefined();
    });

    recordComparison('Data Analysis', 'Datasheet API', datasheetTime, 0, 'ms');
    recordComparison('Data Analysis', 'Lineage API', lineageTime, 0, 'ms');
    await context.close();
  });
});

// ==========================================================================
// 10. MULTI-MODEL TRAINING COMPARISON
// ==========================================================================

test.describe('10. Multi-Model Training', () => {
  test('train 5 different models and compare training times', async ({ browser }) => {
    const context = await browser.newContext();
    const page = await context.newPage();
    await page.goto(RUST, { waitUntil: 'load', timeout: LONG_TIMEOUT });
    await injectRetryFetch(page);
    await setupRustPipeline(page);

    const models = ['logistic_regression', 'decision_tree', 'random_forest', 'naive_bayes', 'knn'];
    const results: Record<string, { time: number; accuracy: number }> = {};

    for (const model of models) {
      const { elapsed } = await measureTime(async () => {
        const jobId = await trainModelRust(page, model);
        const status = await page.evaluate(async (args: { url: string; jobId: string }) => {
          const rf = (window as any).__retryFetch;
          const r = await rf(`${args.url}/api/train/status/${args.jobId}`);
          return r.json();
        }, { url: RUST, jobId });

        const metrics = status.status?.Completed?.metrics;
        results[model] = {
          time: (metrics?.training_time_secs || 0) * 1000,
          accuracy: metrics?.accuracy || 0,
        };
      });

      recordComparison('Multi-Model Training', `${model} total roundtrip`, elapsed, 0, 'ms');
    }

    // Log server-side training times
    for (const [model, res] of Object.entries(results)) {
      console.log(`  ${model}: ${res.time.toFixed(2)}ms server time, accuracy=${res.accuracy.toFixed(4)}`);
    }

    await context.close();
  });
});

// ==========================================================================
// 11. MODEL MANAGEMENT API
// ==========================================================================

test.describe('11. Model Management', () => {
  test('list models API', async ({ browser }) => {
    const context = await browser.newContext();
    const page = await context.newPage();
    await page.goto(RUST, { waitUntil: 'load', timeout: LONG_TIMEOUT });
    await injectRetryFetch(page);

    const times: number[] = [];
    for (let i = 0; i < 5; i++) {
      if (i > 0) await new Promise(r => setTimeout(r, 300));
      const { elapsed } = await measureTime(async () => {
        const res = await page.evaluate(async (url: string) => {
          const rf = (window as any).__retryFetch;
          const r = await rf(`${url}/api/models`);
          return r.json();
        }, RUST);
        expect(res.models).toBeDefined();
        expect(res.models.length).toBeGreaterThan(0);
      });
      times.push(elapsed);
    }

    const stats = computeStats(times);
    recordComparison('Model Management', 'List Models avg', stats.avg, 0, 'ms', true, times);
    await context.close();
  });

  test('get model details + metrics', async ({ browser }) => {
    const context = await browser.newContext();
    const page = await context.newPage();
    await page.goto(RUST, { waitUntil: 'load', timeout: LONG_TIMEOUT });
    await injectRetryFetch(page);

    // Get first model ID
    const modelId = await page.evaluate(async (url: string) => {
      const rf = (window as any).__retryFetch;
      const r = await rf(`${url}/api/models`);
      const data = await r.json();
      return data.models[0]?.id;
    }, RUST);
    expect(modelId).toBeDefined();

    const { elapsed: detailTime } = await measureTime(async () => {
      const res = await page.evaluate(async (args: { url: string; id: string }) => {
        const rf = (window as any).__retryFetch;
        const r = await rf(`${args.url}/api/models/${args.id}`);
        return r.json();
      }, { url: RUST, id: modelId });
      expect(res).toBeDefined();
    });

    const { elapsed: metricsTime } = await measureTime(async () => {
      const res = await page.evaluate(async (args: { url: string; id: string }) => {
        const rf = (window as any).__retryFetch;
        const r = await rf(`${args.url}/api/models/${args.id}/metrics`);
        return r.json();
      }, { url: RUST, id: modelId });
      expect(res).toBeDefined();
    });

    const { elapsed: cardTime } = await measureTime(async () => {
      const res = await page.evaluate(async (args: { url: string; id: string }) => {
        const rf = (window as any).__retryFetch;
        const r = await rf(`${args.url}/api/models/${args.id}/card`);
        return r.json();
      }, { url: RUST, id: modelId });
      expect(res).toBeDefined();
    });

    recordComparison('Model Management', 'Model Detail API', detailTime, 0, 'ms');
    recordComparison('Model Management', 'Model Metrics API', metricsTime, 0, 'ms');
    recordComparison('Model Management', 'Model Card API', cardTime, 0, 'ms');
    await context.close();
  });
});

// ==========================================================================
// 12. AUTOML PIPELINE
// ==========================================================================

test.describe('12. AutoML Pipeline', () => {
  test('run full AutoML pipeline and poll for completion', async ({ browser }) => {
    const context = await browser.newContext();
    const page = await context.newPage();
    await page.goto(RUST, { waitUntil: 'load', timeout: LONG_TIMEOUT });
    await injectRetryFetch(page);
    await setupRustPipeline(page);

    const { elapsed } = await measureTime(async () => {
      // Start AutoML
      const jobId = await page.evaluate(async (url: string) => {
        const rf = (window as any).__retryFetch;
        const res = await rf(`${url}/api/automl/run`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ target_column: 'species', task_type: 'classification' }),
        });
        const data = await res.json();
        return data.job_id;
      }, RUST);
      expect(jobId).toBeDefined();

      // Poll for completion (up to 120s)
      await page.evaluate(async (args: { url: string; jobId: string }) => {
        const rf = (window as any).__retryFetch;
        for (let i = 0; i < 240; i++) {
          const res = await rf(`${args.url}/api/automl/status/${args.jobId}`);
          const data = await res.json();
          if (data.status === 'completed' || data.status === 'Completed' ||
              data.completed === true || data.phase === 'completed') return;
          await new Promise(r => setTimeout(r, 500));
        }
      }, { url: RUST, jobId });
    });

    recordComparison('AutoML Pipeline', 'Full AutoML Run', elapsed, 0, 'ms');
    await context.close();
  });
});

// ==========================================================================
// 13. BATCH PREDICTION
// ==========================================================================

test.describe('13. Batch Prediction', () => {
  test('batch predict — 100 samples', async ({ browser }) => {
    const context = await browser.newContext();
    const page = await context.newPage();
    await page.goto(RUST, { waitUntil: 'load', timeout: LONG_TIMEOUT });
    await injectRetryFetch(page);
    await setupRustPipeline(page);
    await trainModelRust(page, 'logistic_regression');

    // Create 100 sample rows
    const batchData = Array.from({ length: 100 }, (_, i) => [
      5.0 + Math.random() * 2,
      2.5 + Math.random() * 1.5,
      1.0 + Math.random() * 5,
      0.1 + Math.random() * 2,
    ]);

    const times: number[] = [];
    for (let i = 0; i < 3; i++) {
      if (i > 0) await new Promise(r => setTimeout(r, 300));
      const { elapsed } = await measureTime(async () => {
        const res = await page.evaluate(async (args: { url: string; data: number[][] }) => {
          const rf = (window as any).__retryFetch;
          const r = await rf(`${args.url}/api/predict/batch`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ data: args.data }),
          });
          return r.json();
        }, { url: RUST, data: batchData });
        expect(res.success).toBe(true);
        expect(res.count).toBe(100);
      });
      times.push(elapsed);
    }

    const stats = computeStats(times);
    recordComparison('Batch Prediction', 'Batch 100 samples avg', stats.avg, 0, 'ms', true, times);
    recordComparison('Batch Prediction', 'Batch 100 samples p50', stats.p50, 0, 'ms');
    await context.close();
  });

  test('predict probabilities', async ({ browser }) => {
    const context = await browser.newContext();
    const page = await context.newPage();
    await page.goto(RUST, { waitUntil: 'load', timeout: LONG_TIMEOUT });
    await injectRetryFetch(page);

    const times: number[] = [];
    for (let i = 0; i < 5; i++) {
      if (i > 0) await new Promise(r => setTimeout(r, 300));
      const { elapsed } = await measureTime(async () => {
        const res = await page.evaluate(async (url: string) => {
          const rf = (window as any).__retryFetch;
          const r = await rf(`${url}/api/predict/proba`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ data: [[5.1, 3.5, 1.4, 0.2]] }),
          });
          return r.json();
        }, RUST);
        expect(res.success).toBe(true);
        expect(res.probabilities).toBeDefined();
      });
      times.push(elapsed);
    }

    const stats = computeStats(times);
    recordComparison('Batch Prediction', 'Predict Proba avg', stats.avg, 0, 'ms', true, times);
    await context.close();
  });
});

// ==========================================================================
// 14. VISUALIZATION APIs (UMAP, PCA)
// ==========================================================================

test.describe('14. Visualization', () => {
  test('UMAP generation speed', async ({ browser }) => {
    const context = await browser.newContext();
    const page = await context.newPage();
    await page.goto(RUST, { waitUntil: 'load', timeout: LONG_TIMEOUT });
    await injectRetryFetch(page);
    await setupRustPipeline(page);

    const times: number[] = [];
    for (let i = 0; i < 3; i++) {
      if (i > 0) await new Promise(r => setTimeout(r, 300));
      const { elapsed } = await measureTime(async () => {
        const res = await page.evaluate(async (url: string) => {
          const rf = (window as any).__retryFetch;
          const r = await rf(`${url}/api/visualization/umap`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({}),
          });
          return r.json();
        }, RUST);
        expect(res.success).toBe(true);
        expect(res.points.length).toBe(150);
      });
      times.push(elapsed);
    }

    const stats = computeStats(times);
    recordComparison('Visualization', 'UMAP Generation avg', stats.avg, 0, 'ms', true, times);
    await context.close();
  });

  test('PCA generation speed', async ({ browser }) => {
    const context = await browser.newContext();
    const page = await context.newPage();
    await page.goto(RUST, { waitUntil: 'load', timeout: LONG_TIMEOUT });
    await injectRetryFetch(page);
    await setupRustPipeline(page);

    const times: number[] = [];
    for (let i = 0; i < 3; i++) {
      if (i > 0) await new Promise(r => setTimeout(r, 300));
      const { elapsed } = await measureTime(async () => {
        const res = await page.evaluate(async (url: string) => {
          const rf = (window as any).__retryFetch;
          const r = await rf(`${url}/api/visualization/pca`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({}),
          });
          return r.json();
        }, RUST);
        expect(res.success).toBe(true);
        expect(res.points.length).toBe(150);
        expect(res.explained_variance_ratio.length).toBe(2);
      });
      times.push(elapsed);
    }

    const stats = computeStats(times);
    recordComparison('Visualization', 'PCA Generation avg', stats.avg, 0, 'ms', true, times);
    await context.close();
  });
});

// ==========================================================================
// 15. CLUSTERING
// ==========================================================================

test.describe('15. Clustering', () => {
  test('K-Means clustering speed', async ({ browser }) => {
    const context = await browser.newContext();
    const page = await context.newPage();
    await page.goto(RUST, { waitUntil: 'load', timeout: LONG_TIMEOUT });
    await injectRetryFetch(page);
    await setupRustPipeline(page);

    const times: number[] = [];
    for (let i = 0; i < 3; i++) {
      if (i > 0) await new Promise(r => setTimeout(r, 300));
      const { elapsed } = await measureTime(async () => {
        const res = await page.evaluate(async (url: string) => {
          const rf = (window as any).__retryFetch;
          const r = await rf(`${url}/api/clustering/run`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ algorithm: 'kmeans', n_clusters: 3 }),
          });
          return r.json();
        }, RUST);
        expect(res.success).toBe(true);
        expect(res.result.n_clusters).toBe(3);
        expect(res.result.labels.length).toBe(150);
      });
      times.push(elapsed);
    }

    const stats = computeStats(times);
    recordComparison('Clustering', 'K-Means (k=3) avg', stats.avg, 0, 'ms', true, times);
    await context.close();
  });
});

// ==========================================================================
// 16. ANOMALY DETECTION
// ==========================================================================

test.describe('16. Anomaly Detection', () => {
  test('isolation forest anomaly detection speed', async ({ browser }) => {
    const context = await browser.newContext();
    const page = await context.newPage();
    await page.goto(RUST, { waitUntil: 'load', timeout: LONG_TIMEOUT });
    await injectRetryFetch(page);
    await setupRustPipeline(page);

    const times: number[] = [];
    for (let i = 0; i < 3; i++) {
      if (i > 0) await new Promise(r => setTimeout(r, 300));
      const { elapsed } = await measureTime(async () => {
        const res = await page.evaluate(async (url: string) => {
          const rf = (window as any).__retryFetch;
          const r = await rf(`${url}/api/anomaly/detect`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({}),
          });
          return r.json();
        }, RUST);
        expect(res.success).toBe(true);
        expect(res.total_samples).toBe(150);
        expect(res.anomaly_rate).toBeGreaterThan(0);
      });
      times.push(elapsed);
    }

    const stats = computeStats(times);
    recordComparison('Anomaly Detection', 'Isolation Forest avg', stats.avg, 0, 'ms', true, times);
    await context.close();
  });
});

// ==========================================================================
// 17. HYPERPARAMETER OPTIMIZATION
// ==========================================================================

test.describe('17. Hyperparameter Optimization', () => {
  test('HyperOpt on logistic regression (3 trials)', async ({ browser }) => {
    const context = await browser.newContext();
    const page = await context.newPage();
    await page.goto(RUST, { waitUntil: 'load', timeout: LONG_TIMEOUT });
    await injectRetryFetch(page);
    await setupRustPipeline(page);

    const { elapsed } = await measureTime(async () => {
      const jobRes = await page.evaluate(async (url: string) => {
        const rf = (window as any).__retryFetch;
        const r = await rf(`${url}/api/hyperopt`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            model_type: 'logistic_regression',
            target_column: 'species',
            task_type: 'classification',
            n_trials: 3,
          }),
        });
        return r.json();
      }, RUST);
      expect(jobRes.success).toBe(true);
    });

    recordComparison('HyperOpt', 'LogReg 3 trials', elapsed, 0, 'ms');

    // Also check history endpoint
    const { elapsed: historyTime } = await measureTime(async () => {
      const res = await page.evaluate(async (url: string) => {
        const rf = (window as any).__retryFetch;
        const r = await rf(`${url}/api/hyperopt/history`);
        return r.json();
      }, RUST);
      expect(res.studies).toBeDefined();
      expect(res.total).toBeGreaterThan(0);
    });

    recordComparison('HyperOpt', 'History API', historyTime, 0, 'ms');

    // Check search space
    const { elapsed: spaceTime } = await measureTime(async () => {
      const res = await page.evaluate(async (url: string) => {
        const rf = (window as any).__retryFetch;
        const r = await rf(`${url}/api/hyperopt/search-space/random_forest`);
        return r.json();
      }, RUST);
      expect(res).toBeDefined();
    });

    recordComparison('HyperOpt', 'Search Space API', spaceTime, 0, 'ms');
    await context.close();
  });

  test('HyperOpt config endpoint', async ({ browser }) => {
    const context = await browser.newContext();
    const page = await context.newPage();
    await page.goto(RUST, { waitUntil: 'load', timeout: LONG_TIMEOUT });
    await injectRetryFetch(page);

    const { elapsed } = await measureTime(async () => {
      const res = await page.evaluate(async (url: string) => {
        const rf = (window as any).__retryFetch;
        const r = await rf(`${url}/api/hyperopt/config`);
        return r.json();
      }, RUST);
      expect(res.samplers).toBeDefined();
      expect(res.model_search_spaces).toBeDefined();
    });

    recordComparison('HyperOpt', 'Config API', elapsed, 0, 'ms');
    await context.close();
  });
});

// ==========================================================================
// 18. EXPLAINABILITY
// ==========================================================================

test.describe('18. Explainability', () => {
  test('feature importance API', async ({ browser }) => {
    const context = await browser.newContext();
    const page = await context.newPage();
    await page.goto(RUST, { waitUntil: 'load', timeout: LONG_TIMEOUT });
    await injectRetryFetch(page);
    await setupRustPipeline(page);

    // Get a model ID
    const modelId = await page.evaluate(async (url: string) => {
      const rf = (window as any).__retryFetch;
      const r = await rf(`${url}/api/models`);
      const data = await r.json();
      return data.models[0]?.id;
    }, RUST);

    if (modelId) {
      const { elapsed } = await measureTime(async () => {
        const res = await page.evaluate(async (args: { url: string; id: string }) => {
          const rf = (window as any).__retryFetch;
          const r = await rf(`${args.url}/api/explain/importance`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ model_id: args.id }),
          });
          return r.json();
        }, { url: RUST, id: modelId });
        // May return importance scores or error if model doesn't support it
        expect(res).toBeDefined();
      });

      recordComparison('Explainability', 'Feature Importance API', elapsed, 0, 'ms');
    }
    await context.close();
  });
});

// ==========================================================================
// 19. SYSTEM STATUS & MONITORING
// ==========================================================================

test.describe('19. System & Monitoring', () => {
  test('system/status API', async ({ browser }) => {
    const context = await browser.newContext();
    const page = await context.newPage();
    await page.goto(RUST, { waitUntil: 'load', timeout: LONG_TIMEOUT });
    await injectRetryFetch(page);

    const times: number[] = [];
    for (let i = 0; i < 5; i++) {
      if (i > 0) await new Promise(r => setTimeout(r, 300));
      const { elapsed } = await measureTime(async () => {
        const res = await page.evaluate(async (url: string) => {
          const rf = (window as any).__retryFetch;
          const r = await rf(`${url}/api/system/status`);
          return r.json();
        }, RUST);
        expect(res.status).toBe('healthy');
        expect(res.system.cpu_count).toBeGreaterThan(0);
        expect(res.system.total_memory_gb).toBeGreaterThan(0);
      });
      times.push(elapsed);
    }

    const stats = computeStats(times);
    recordComparison('System & Monitoring', 'System Status avg', stats.avg, 0, 'ms', true, times);
    await context.close();
  });

  test('monitoring endpoints', async ({ browser }) => {
    const context = await browser.newContext();
    const page = await context.newPage();
    await page.goto(RUST, { waitUntil: 'load', timeout: LONG_TIMEOUT });
    await injectRetryFetch(page);

    const { elapsed: alertsTime } = await measureTime(async () => {
      const res = await page.evaluate(async (url: string) => {
        const rf = (window as any).__retryFetch;
        const r = await rf(`${url}/api/monitoring/alerts`);
        return r.json();
      }, RUST);
      expect(res).toBeDefined();
    });

    const { elapsed: perfTime } = await measureTime(async () => {
      const res = await page.evaluate(async (url: string) => {
        const rf = (window as any).__retryFetch;
        const r = await rf(`${url}/api/monitoring/performance`);
        return r.json();
      }, RUST);
      expect(res.models_count).toBeDefined();
      expect(res.throughput).toBeDefined();
    });

    const { elapsed: sloTime } = await measureTime(async () => {
      const res = await page.evaluate(async (url: string) => {
        const rf = (window as any).__retryFetch;
        const r = await rf(`${url}/api/monitoring/slo`);
        return r.json();
      }, RUST);
      expect(res.slo_status).toBeDefined();
      expect(res.slo_status.all_met).toBe(true);
    });

    recordComparison('System & Monitoring', 'Alerts API', alertsTime, 0, 'ms');
    recordComparison('System & Monitoring', 'Performance API', perfTime, 0, 'ms');
    recordComparison('System & Monitoring', 'SLO Status API', sloTime, 0, 'ms');
    await context.close();
  });

  test('device info + optimal configs', async ({ browser }) => {
    const context = await browser.newContext();
    const page = await context.newPage();
    await page.goto(RUST, { waitUntil: 'load', timeout: LONG_TIMEOUT });
    await injectRetryFetch(page);

    const { elapsed: deviceTime } = await measureTime(async () => {
      const res = await page.evaluate(async (url: string) => {
        const rf = (window as any).__retryFetch;
        const r = await rf(`${url}/api/device/info`);
        return r.json();
      }, RUST);
      expect(res.cpu).toBeDefined();
      expect(res.memory).toBeDefined();
    });

    const { elapsed: configTime } = await measureTime(async () => {
      const res = await page.evaluate(async (url: string) => {
        const rf = (window as any).__retryFetch;
        const r = await rf(`${url}/api/device/optimal-configs`);
        return r.json();
      }, RUST);
      expect(res.training).toBeDefined();
      expect(res.cache).toBeDefined();
    });

    recordComparison('System & Monitoring', 'Device Info API', deviceTime, 0, 'ms');
    recordComparison('System & Monitoring', 'Optimal Configs API', configTime, 0, 'ms');
    await context.close();
  });
});

// ==========================================================================
// 20. SECURITY & COMPLIANCE
// ==========================================================================

test.describe('20. Security & Compliance', () => {
  test('security status + rate limiting', async ({ browser }) => {
    const context = await browser.newContext();
    const page = await context.newPage();
    await page.goto(RUST, { waitUntil: 'load', timeout: LONG_TIMEOUT });
    await injectRetryFetch(page);

    const { elapsed: secTime } = await measureTime(async () => {
      const res = await page.evaluate(async (url: string) => {
        const rf = (window as any).__retryFetch;
        const r = await rf(`${url}/api/security/status`);
        return r.json();
      }, RUST);
      expect(res.security_enabled).toBe(true);
      expect(res.guards).toBeDefined();
      expect(res.rate_limiting).toBeDefined();
    });

    const { elapsed: auditTime } = await measureTime(async () => {
      const res = await page.evaluate(async (url: string) => {
        const rf = (window as any).__retryFetch;
        const r = await rf(`${url}/api/security/audit-log`);
        return r.json();
      }, RUST);
      expect(res).toBeDefined();
    });

    recordComparison('Security & Compliance', 'Security Status API', secTime, 0, 'ms');
    recordComparison('Security & Compliance', 'Audit Log API', auditTime, 0, 'ms');
    await context.close();
  });

  test('compliance report', async ({ browser }) => {
    const context = await browser.newContext();
    const page = await context.newPage();
    await page.goto(RUST, { waitUntil: 'load', timeout: LONG_TIMEOUT });
    await injectRetryFetch(page);

    const { elapsed } = await measureTime(async () => {
      const res = await page.evaluate(async (url: string) => {
        const rf = (window as any).__retryFetch;
        const r = await rf(`${url}/api/compliance/report`);
        return r.json();
      }, RUST);
      expect(res.compliance_report).toBeDefined();
      expect(res.compliance_report.overall_score).toBeGreaterThan(0);
      expect(res.compliance_report.standard_summaries).toBeDefined();
    });

    recordComparison('Security & Compliance', 'Compliance Report API', elapsed, 0, 'ms');
    await context.close();
  });

  test('audit events', async ({ browser }) => {
    const context = await browser.newContext();
    const page = await context.newPage();
    await page.goto(RUST, { waitUntil: 'load', timeout: LONG_TIMEOUT });
    await injectRetryFetch(page);

    const { elapsed } = await measureTime(async () => {
      const res = await page.evaluate(async (url: string) => {
        const rf = (window as any).__retryFetch;
        const r = await rf(`${url}/api/audit/events`);
        return r.json();
      }, RUST);
      expect(res).toBeDefined();
    });

    recordComparison('Security & Compliance', 'Audit Events API', elapsed, 0, 'ms');
    await context.close();
  });
});

// ==========================================================================
// 21. PREPROCESSING CONFIG
// ==========================================================================

test.describe('21. Preprocessing', () => {
  test('preprocess config + pipeline speed', async ({ browser }) => {
    const context = await browser.newContext();
    const page = await context.newPage();
    await page.goto(RUST, { waitUntil: 'load', timeout: LONG_TIMEOUT });
    await injectRetryFetch(page);

    // Load data first
    await page.evaluate(async (url: string) => {
      const rf = (window as any).__retryFetch;
      await rf(`${url}/api/data/sample/iris`);
    }, RUST);

    const { elapsed: configTime } = await measureTime(async () => {
      const res = await page.evaluate(async (url: string) => {
        const rf = (window as any).__retryFetch;
        const r = await rf(`${url}/api/preprocess/config`);
        return r.json();
      }, RUST);
      expect(res.scalers).toBeDefined();
      expect(res.imputation_strategies).toBeDefined();
    });

    const times: number[] = [];
    for (let i = 0; i < 3; i++) {
      if (i > 0) await new Promise(r => setTimeout(r, 300));
      const { elapsed } = await measureTime(async () => {
        const res = await page.evaluate(async (url: string) => {
          const rf = (window as any).__retryFetch;
          const r = await rf(`${url}/api/preprocess`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ task_type: 'classification', target_column: 'species', scaler: 'standard', imputation: 'mean', cv_folds: 5 }),
          });
          return r.json();
        }, RUST);
        expect(res.success).toBe(true);
      });
      times.push(elapsed);
    }

    const stats = computeStats(times);
    recordComparison('Preprocessing', 'Config API', configTime, 0, 'ms');
    recordComparison('Preprocessing', 'Pipeline Run avg', stats.avg, 0, 'ms', true, times);
    await context.close();
  });
});

// ==========================================================================
// 22. BATCH PROCESSING STATS
// ==========================================================================

test.describe('22. Batch Processing', () => {
  test('batch stats API', async ({ browser }) => {
    const context = await browser.newContext();
    const page = await context.newPage();
    await page.goto(RUST, { waitUntil: 'load', timeout: LONG_TIMEOUT });
    await injectRetryFetch(page);

    const { elapsed } = await measureTime(async () => {
      const res = await page.evaluate(async (url: string) => {
        const rf = (window as any).__retryFetch;
        const r = await rf(`${url}/api/batch/stats`);
        return r.json();
      }, RUST);
      expect(res.total_batches_processed).toBeDefined();
    });

    recordComparison('Batch Processing', 'Batch Stats API', elapsed, 0, 'ms');
    await context.close();
  });
});

// ==========================================================================
// 23. SAMPLE DATASETS (All Types)
// ==========================================================================

test.describe('23. Sample Dataset Loading', () => {
  const datasets = ['iris', 'diabetes', 'boston', 'wine', 'breast_cancer'];

  for (const ds of datasets) {
    test(`load ${ds} sample dataset`, async ({ browser }) => {
      const context = await browser.newContext();
      const page = await context.newPage();
      await page.goto(RUST, { waitUntil: 'load', timeout: LONG_TIMEOUT });
      await injectRetryFetch(page);

      // Clear any existing data first to avoid conflicts
      await page.evaluate(async (url: string) => {
        const rf = (window as any).__retryFetch;
        await rf(`${url}/api/data/clear`, { method: 'DELETE' }).catch(() => {});
      }, RUST);

      const { elapsed } = await measureTime(async () => {
        const res = await page.evaluate(async (args: { url: string; name: string }) => {
          const rf = (window as any).__retryFetch;
          const r = await rf(`${args.url}/api/data/sample/${args.name}`);
          const text = await r.text();
          try { return JSON.parse(text); } catch { return { status: r.status, body: text }; }
        }, { url: RUST, name: ds });
        expect(res.success).toBe(true);
        expect(res.rows).toBeGreaterThan(0);
      });

      recordComparison('Sample Datasets', `Load ${ds}`, elapsed, 0, 'ms');
      await context.close();
    });
  }
});

// ==========================================================================
// 24. CONCURRENT API STRESS — Multiple Endpoints
// ==========================================================================

test.describe('24. Multi-Endpoint Stress', () => {
  test('10 parallel mixed API requests', async ({ browser }) => {
    const context = await browser.newContext();
    const page = await context.newPage();
    await page.goto(RUST, { waitUntil: 'load', timeout: LONG_TIMEOUT });
    await injectRetryFetch(page);

    // Ensure data is loaded for data/* endpoints
    await setupRustPipeline(page);

    const result = await page.evaluate(async (baseURL: string) => {
      const rf = (window as any).__retryFetch;
      const endpoints = [
        '/api/health',
        '/api/system/status',
        '/api/models',
        '/api/data/info',
        '/api/data/quality',
        '/api/monitoring/alerts',
        '/api/monitoring/performance',
        '/api/device/info',
        '/api/security/status',
        '/api/preprocess/config',
      ];

      const start = Date.now();
      const results = await Promise.all(
        endpoints.map(async (ep) => {
          const t0 = Date.now();
          const res = await rf(`${baseURL}${ep}`);
          const t1 = Date.now();
          return { endpoint: ep, status: res.status, time: t1 - t0 };
        }),
      );
      const total = Date.now() - start;

      return {
        total,
        individual: results.map((r) => r.time),
        statuses: results.map((r) => r.status),
        endpoints: results.map((r) => ({ ep: r.endpoint, time: r.time, ok: r.status === 200 })),
      };
    }, RUST);

    // All should succeed
    expect(result.statuses.every((s: number) => s === 200)).toBe(true);

    const stats = computeStats(result.individual);
    recordComparison('Multi-Endpoint Stress', 'Total (10 mixed)', result.total, 0, 'ms');
    recordComparison('Multi-Endpoint Stress', 'Avg per endpoint', stats.avg, 0, 'ms');
    recordComparison('Multi-Endpoint Stress', 'P95 per endpoint', stats.p95, 0, 'ms');

    // Log individual times
    for (const ep of result.endpoints) {
      console.log(`  ${ep.ep}: ${ep.time}ms ${ep.ok ? '✓' : '✗'}`);
    }
    await context.close();
  });
});

// ==========================================================================
// 25. PRIVACY & DATA RETENTION
// ==========================================================================

test.describe('25. Privacy', () => {
  test('privacy retention API', async ({ browser }) => {
    const context = await browser.newContext();
    const page = await context.newPage();
    await page.goto(RUST, { waitUntil: 'load', timeout: LONG_TIMEOUT });
    await injectRetryFetch(page);

    const { elapsed } = await measureTime(async () => {
      const res = await page.evaluate(async (url: string) => {
        const rf = (window as any).__retryFetch;
        const r = await rf(`${url}/api/privacy/retention`);
        return r.json();
      }, RUST);
      expect(res).toBeDefined();
    });

    recordComparison('Privacy', 'Retention Status API', elapsed, 0, 'ms');
    await context.close();
  });
});

// ==========================================================================
// 26. END-TO-END PIPELINE — Full Workflow via API
// ==========================================================================

test.describe('26. E2E Pipeline via API', () => {
  test('full pipeline: load → preprocess → train → predict → explain', async ({ browser }) => {
    const context = await browser.newContext();
    const page = await context.newPage();
    await page.goto(RUST, { waitUntil: 'load', timeout: LONG_TIMEOUT });
    await injectRetryFetch(page);

    const stepTimes: Record<string, number> = {};

    // Step 1: Load data
    const { elapsed: loadTime } = await measureTime(async () => {
      const res = await page.evaluate(async (url: string) => {
        const rf = (window as any).__retryFetch;
        const r = await rf(`${url}/api/data/sample/iris`);
        return r.json();
      }, RUST);
      expect(res.success).toBe(true);
    });
    stepTimes['load'] = loadTime;

    // Step 2: Preprocess
    const { elapsed: preprocessTime } = await measureTime(async () => {
      const res = await page.evaluate(async (url: string) => {
        const rf = (window as any).__retryFetch;
        const r = await rf(`${url}/api/preprocess`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ task_type: 'classification', target_column: 'species', scaler: 'standard', imputation: 'mean', cv_folds: 5 }),
        });
        return r.json();
      }, RUST);
      expect(res.success).toBe(true);
    });
    stepTimes['preprocess'] = preprocessTime;

    // Step 3: Train
    const { elapsed: trainTime } = await measureTime(async () => {
      await trainModelRust(page, 'random_forest');
    });
    stepTimes['train'] = trainTime;

    // Step 4: Predict
    const { elapsed: predictTime } = await measureTime(async () => {
      const res = await page.evaluate(async (url: string) => {
        const rf = (window as any).__retryFetch;
        const r = await rf(`${url}/api/predict`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ data: [[5.1, 3.5, 1.4, 0.2]] }),
        });
        return r.json();
      }, RUST);
      expect(res.success).toBe(true);
    });
    stepTimes['predict'] = predictTime;

    // Step 5: Explain
    const modelId = await page.evaluate(async (url: string) => {
      const rf = (window as any).__retryFetch;
      const r = await rf(`${url}/api/models`);
      const data = await r.json();
      return data.models[0]?.id;
    }, RUST);

    const { elapsed: explainTime } = await measureTime(async () => {
      if (modelId) {
        await page.evaluate(async (args: { url: string; id: string }) => {
          const rf = (window as any).__retryFetch;
          await rf(`${args.url}/api/explain/importance`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ model_id: args.id }),
          });
        }, { url: RUST, id: modelId });
      }
    });
    stepTimes['explain'] = explainTime;

    const totalTime = Object.values(stepTimes).reduce((a, b) => a + b, 0);

    recordComparison('E2E Pipeline', 'Load Data', stepTimes['load'], 0, 'ms');
    recordComparison('E2E Pipeline', 'Preprocess', stepTimes['preprocess'], 0, 'ms');
    recordComparison('E2E Pipeline', 'Train (Random Forest)', stepTimes['train'], 0, 'ms');
    recordComparison('E2E Pipeline', 'Predict', stepTimes['predict'], 0, 'ms');
    recordComparison('E2E Pipeline', 'Explain', stepTimes['explain'], 0, 'ms');
    recordComparison('E2E Pipeline', 'Total Pipeline', totalTime, 0, 'ms');

    console.log('\n  E2E Pipeline Breakdown:');
    for (const [step, time] of Object.entries(stepTimes)) {
      console.log(`    ${step}: ${time.toFixed(0)}ms`);
    }
    console.log(`    TOTAL: ${totalTime.toFixed(0)}ms`);
    await context.close();
  });
});

// ==========================================================================
// 27. EXTENDED REPORT
// ==========================================================================

test.describe('27. Extended Report', () => {
  test('generate extended comparison report', async () => {
    printComparisonTable();
    saveReport();

    const fs = await import('fs');
    const path = await import('path');
    const reportPath = path.join(__dirname, 'results.json');
    expect(fs.existsSync(reportPath)).toBe(true);

    const report = JSON.parse(fs.readFileSync(reportPath, 'utf-8'));
    console.log(`\nExtended metrics collected: ${report.entries.length}`);
  });
});
