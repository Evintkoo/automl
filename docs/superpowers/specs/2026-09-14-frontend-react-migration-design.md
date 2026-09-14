# Frontend Migration: embedded HTML string → React/Vite in `web/`

**Goal:** Replace the ~2,650-line `EMBEDDED_INDEX_HTML` string (and the smaller monitoring-dashboard template) hand-rolled inside `src/server/handlers.rs` with a proper React + TypeScript frontend in a new `web/` folder, built with Vite, styled with plain CSS (no Tailwind/shadcn), while keeping the frontend embedded in and served from the same Rust binary on the same origin.

**Scope:** Full rewrite of all seven current UI panels (Dashboard, Data, Train, Analysis, Insights, Reports, Monitor), cut over once parity is reached. No backend API changes, no change to deployment topology.

**Relationship to prior work:** Directly modeled on the sibling project `~/Documents/projects/evint/torch-inference`'s `docs/superpowers/specs/2026-09-13-frontend-react-migration-design.md`, which solved the identical problem (giant embedded-HTML Rust server → React/Vite `web/`, `rust-embed` for single-binary serving). This design deliberately diverges from that one on two points: no shadcn/Tailwind (plain CSS instead) and no TanStack Query (hand-rolled fetch hooks instead) — both per explicit preference for the smallest reasonable dependency footprint on this project.

---

## Section 1 — Why React + Vite, plain CSS, no query library

| Option | Fit | Trade-off |
|---|---|---|
| **React + TypeScript + Vite (chosen)** | Matches the sibling project's stack and the current UI's SPA/tab-based navigation model. Vite produces a trivially embeddable static `dist/`. | React+ReactDOM runtime (~45KB gzip) — irrelevant for a locally-served ops dashboard. |
| Vanilla JS rewrite (keep current approach, just split into files) | Zero new dependencies. | Doesn't solve the actual problem — no component model, no type safety, the 1,880-line inline script stays a maintenance liability either way. |

**Styling: plain CSS**, not Tailwind/shadcn. The current embedded HTML already has a complete, coherent theme (`:root` CSS variables for `--bg`, `--surface`, `--border`, `--accent`, etc., Geist Mono for UI chrome) — that gets ported verbatim into `styles/globals.css` and consumed via a handful of small hand-built primitives in `components/ui/` (`Button`, `Card`, `Tabs`, `Badge`, `Sparkline`). No component-library dependency, no build-time CSS framework.

**Data fetching: hand-rolled hooks**, not TanStack Query. The API surface is ~20 REST endpoints plus one SSE stream — small enough that a single `useFetch`-style hook in `lib/api-client.ts` (loading/error/data state, no caching layer) covers every panel without pulling in a query library. This is the one place this design trades a bit of convenience (manual retry/cache) for a smaller dependency tree, matching the "simple component" brief.

**Charts:** the current UI does NOT use the vendored `chart.min.js` (0 `Chart.` call sites found) — every 2D chart is hand-rolled `<canvas>` drawing (25 `getContext` call sites: sparklines, stat trends, PCA/UMAP scatter plots). The one exception is the HyperOpt 3D surface, which uses Plotly loaded via CDN `<script defer>`. The new FE keeps this split: small local canvas-drawing components for 2D (no dependency), and `plotly.js-dist-min` as an npm dependency wrapped in one `components/ui/plotly-3d.tsx` for the 3D case.

---

## Section 2 — Folder layout

```
web/                              # new FE root — untouched by `cargo build`
  index.html
  package.json
  package-lock.json
  vite.config.ts
  tsconfig.json
  src/
    main.tsx
    App.tsx                       # flat panel-list -> AppLayout, mirrors torch-inference's App.tsx
    app/
      layout.tsx                  # sidebar shell: 7 nav items, matches current tab bar 1:1
    components/
      ui/                         # hand-built primitives, plain CSS: Button, Card, Badge, Tabs,
                                   #   Sparkline, StatCard, Alert, plotly-3d.tsx
    features/
      dashboard/                  # system status cards, sparklines (was et-dashboard)
      data/                       # upload/import/preview/analyze (was et-data)
      train/                      # subtabs: autotune, automl, single, hyperopt, ensemble (was et-train)
      analysis/                   # explainability, anomaly detection, PCA/UMAP scatter (was et-analysis)
      insights/                   # (was et-insights)
      reports/                    # model comparison report (was et-reports)
      monitor/                    # CPU/mem/queue live stats (was et-monitor)
    lib/
      api-client.ts               # fetch wrapper: base URL, error envelope, the useFetch hook
      sse-client.ts               # EventSource wrapper for /api/visualization/umap/stream
      utils.ts                    # escHtml-equivalent helpers, formatters
    styles/
      globals.css                 # CSS variables ported verbatim from the current embedded <style>
  public/
    favicon.svg
```

Each `features/<name>/` folder is self-contained: its panel component(s), a `types.ts`, and any panel-local helpers — no shared state beyond what `lib/api-client.ts` provides. Adding a feature later means adding one folder + one entry in `App.tsx`'s panel array, matching torch-inference's pattern exactly.

---

## Section 3 — Data flow / API integration

The new FE talks to the same endpoints the current embedded UI already calls — no backend changes in scope. Representative surface (exact params unchanged): `/api/data/{upload,import/url,analyze,sample/:id}`, `/api/train`, `/api/train/status/:id`, `/api/automl/{run,status/:id}`, `/api/autotune`, `/api/hyperopt`, `/hyperopt/apply`, `/api/ensemble/train`, `/api/predict`, `/api/explain/{importance,local}`, `/api/anomaly/detect`, `/api/visualization/{pca,umap}`, `/api/models`, `/api/system/status`, `/api/security/status`, `/api/security/audit-log`, `/api/quality/report/:id`.

- **REST:** `fetch` via `lib/api-client.ts`'s `useFetch` hook (loading/error/data state per call site, manual refetch trigger — no cache layer).
- **SSE (UMAP progressive scatter):** `/api/visualization/umap/stream` wrapped by `lib/sse-client.ts`, mirroring the current `eStartUmap()` reconnect/close semantics.
- No WebSocket usage found in the current UI — none needed in the new one either.

No new backend auth/CORS surface: same-origin embedded serving, existing auth middleware applies unchanged.

---

## Section 4 — Build → embed integration

Goals: keep `cargo build`/CI Node-free by default, keep single-binary single-port deployment.

- **New Makefile targets** (alongside the existing `build`/`server`/`dev`):
  - `web` — `cd web && npm ci && npm run build`, producing `web/dist/`.
  - `web-dev` — `cd web && npm install && npm run dev` (Vite dev server, proxies `/api/*` to `http://localhost:$(PORT)` for hot reload).
- **`web/dist/` and `web/node_modules/` are gitignored** — build output, not committed.
- **Embedding:** add the `rust-embed` crate. Replace `EMBEDDED_INDEX_HTML`/`serve_index()`'s disk-fallback logic and the separate `get_monitoring_dashboard()` template with an embedded-asset struct over `web/dist/`. `cargo build` gets `#[allow_missing = true]` on the embed so it still compiles with zero Node installed / before `web/dist/` exists (matching torch-inference's `web_assets.rs`).
- **Routing:** `/` continues to serve `index.html` from the embedded bundle; a new static-asset route serves the rest of the hashed JS/CSS chunks Vite produces. The standalone monitoring-dashboard route folds into the React app as the `monitor` panel rather than staying a second hand-rolled HTML template.
- **Static assets:** `automl-web/static/` (`chart.min.js` — unused, drop it; `remixicon.css`/`.woff2`) — Remix Icons switches to the `remixicon` npm package bundled by Vite; the vendored files in `automl-web/` are removed once nothing references them.

---

## Section 5 — Migration execution order

1. **Scaffold + shell** — `npm create vite` (react-ts template), port `:root` theme vars into `globals.css`, build `components/ui/` primitives, `lib/api-client.ts` + `useFetch`, app shell with sidebar nav (7 items, empty panels). Wire into `rust-embed` + Makefile so the empty shell serves end-to-end through the real binary before any panel is built.
2. **Dashboard panel** — system/health stat cards, first use of the canvas sparkline components.
3. **Data panel** — upload/import/preview/analyze flows.
4. **Train panel** — autotune/automl/single/hyperopt/ensemble subtabs (`autotune` is the default-active one); largest panel, includes the Plotly 3D wrapper for HyperOpt viz and the `/hyperopt/apply` flow (recently added — must not regress).
5. **Analysis panel** — explainability, anomaly detection, PCA/UMAP scatter; validates `sse-client.ts` against the real UMAP stream.
6. **Insights panel.**
7. **Reports panel** — model comparison, checkbox-driven selection.
8. **Monitor panel** — folds the standalone monitoring-dashboard template into this panel; live CPU/mem/queue stats.
9. **Cutover** — swap `/` to serve the new embedded bundle, delete `EMBEDDED_INDEX_HTML`, `get_monitoring_dashboard()`, and all now-dead string-building code in `handlers.rs`, remove `automl-web/static/chart.min.js`, in the same commit.

---

## Section 6 — Testing strategy

- **`tests/test_web_ui.rs`** currently asserts on the raw embedded-HTML string (tab counts, subtabs, active-tab default) by parsing the served HTML. This becomes meaningless once the served HTML is a Vite-built React shell — deleted at cutover (Step 9), not ported, since it was testing implementation details of the old string-templating approach rather than behavior.
- **`e2e/comparison/pca.spec.ts`** and **`umap.spec.ts`** (Playwright) currently select against the old hand-rolled DOM — selectors are rewritten panel-by-panel as each panel lands (Steps 2–8), not batched at the end, since the DOM structure changes completely with React-rendered markup.
- New FE-level unit tests (Vitest, colocated `*.test.ts(x)` per component) are added starting at scaffold time for `lib/` helpers and any non-trivial component logic (e.g. the UMAP SSE reducer).
- No new backend integration tests — no API contract changes.

---

## Section 7 — Non-goals

- No backend API changes.
- No change to deployment topology (single binary, single origin, same auth model).
- No SSR — client-rendered SPA, matching the current UI's nature as a local ops dashboard.
- No Tailwind/shadcn, no TanStack Query (explicit choice, Section 1).
- No PWA/offline support.

---

## Success criteria

- All seven current panels have a working equivalent in the new FE, verified via updated Playwright specs.
- Server still ships as a single binary; `web/dist/` is not present in the committed repo, only its build output is embedded at compile time.
- `EMBEDDED_INDEX_HTML`, `get_monitoring_dashboard()`, and their wiring are deleted after cutover — no dead code left behind.
- `tests/test_web_ui.rs` removed; Playwright specs pass against the new DOM.
- No regression in the recently-added HyperOpt Apply & Retrain flow or the real prediction-latency SLO tracking (both landed very recently per git log — must carry over working, not just visually).
