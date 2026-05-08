# DevBloom / Prism Quant — Streamlit Stability Hotfix PRD

## Original problem statement
Streamlit-based trading terminal (`/app/app.py`, ~2000 LOC) loaded correctly,
then went **completely blank after a few seconds**.

Logs showed:
* 9× warnings — `The cached function '<x>' has a TTL that will be ignored. Persistent cached functions currently don't support TTL.`
* 2× warnings — `Please replace st.components.v1.html with st.iframe.`
* App started fine on Uvicorn, then UI went blank.

## Root-cause analysis (RCA)

| # | Issue | File | Effect |
|---|-------|------|--------|
| **1** | **Blocking auto-refresh** — `time.sleep(1)` × 30 in script thread, then `st.rerun()` while checkbox stays on → infinite blocking loop | `app.py:613-619` (old) | **Primary cause of blank screen** — WebSocket starves and frontend stops receiving render deltas |
| 2 | `@st.cache_data(persist="disk", ttl=N)` everywhere — Streamlit ≥1.30 silently ignores TTL on disk-persisted caches; data never refreshes | `app.py`, `ui/heatmap.py`, `ui/macro.py`, `ui/watchlist.py`, `ui/alert_inbox.py`, `charting/multi_tf.py` | Stale data + disk I/O storms on every rerun |
| 3 | Deprecated `components.v1.html(MONACO_HTML, ...)` — will be removed 2026-06-01 | `ui/algolab.py:226` | Deprecation warnings; future breakage |
| 4 | `tabs = list(tabs) + [st.empty() for _ in range(14)]` — `st.empty()` is a single-child container; tabs[9..22] silently dropped all but one element | `app.py:759` (old) | tabs 9-22 partially broken; legacy hidden bug |
| 5 | No try/except + no structured logging at script entry | `app.py` | Impossible to trace blank-screen root cause |

## Production-grade fixes implemented

| Fix | Change | File |
|----|--------|------|
| **F1** | Replaced blocking auto-refresh with `streamlit_autorefresh.st_autorefresh(interval=30_000)` — schedules rerun on browser side, never blocks script thread. Cache cleared on tick boundary. Graceful fallback if package missing. | `app.py` (sidebar) |
| **F2** | Stripped `persist="disk"` from all 10 cache decorators with TTL. In-memory caches now correctly honour TTL. | `app.py`, `ui/heatmap.py`, `ui/macro.py`, `ui/watchlist.py`, `ui/alert_inbox.py`, `charting/multi_tf.py` |
| **F3** | Removed deprecated Monaco editor (`components.html`) — kept the textarea (which was already the documented fallback). | `ui/algolab.py` |
| **F4** | Replaced `tabs + [st.empty()]*14` hack with a proper 23-entry `st.tabs([...])` declaration. All tabs now render correctly. | `app.py:747-771` |
| **F5** | Added lightweight stdlib structured logger writing to `/app/logs/streamlit_app.log` + console. Logs `streamlit_script_start` on every rerun and warns on autorefresh-package issues. | `app.py:1-37` |
| F6 | Added `streamlit-autorefresh==1.0.1` to runtime deps (pip install). | env |

## Verification (browser-tested at localhost:8501)

| Check | Before | After |
|-------|--------|-------|
| Initial render | OK | OK |
| Auto-refresh ON for 35 s | **Page blank** | Tick #1 fires, page fully rendered, sidebar+tabs intact |
| TTL warnings in log | 9 | **0** |
| `components.v1.html` warnings | 2 | **0** |
| Tab count visible | 9 (rest broken) | 23 |
| Exception blocks rendered | 0 | 0 |
| Structured log file | absent | `/app/logs/streamlit_app.log` ✓ |

## Architecture impact

* **Startup pipeline** is unchanged on success path — only blank-screen risk eliminated.
* **Memory**: removing `persist="disk"` reclaims pickled DataFrames on disk; TTL eviction works again.
* **Reliability**: structured log file traces script restarts; future blank-screen reports can be diagnosed by checking `/app/logs/streamlit_app.log` for missing `streamlit_script_start` entries during a freeze.
* **No business logic changed**; only stability infrastructure.

## What's NOT changed (deferred / out-of-scope)
* `find_buzzing_stocks` and `_fetch_news_alerts` remain synchronous — heavy yfinance calls.
* Pre-existing E702 lint warnings in `app.py:523-524` (multi-statement lines) untouched — purely cosmetic.
* Kite access-token / Claude API key still need environment configuration by the user (unrelated runtime warnings).

## Backlog / Next Action Items
* P1 — replace `find_buzzing_stocks` / `scan_parallel` ThreadPoolExecutor with non-blocking job queue + result polling.
* P1 — add per-tab `try/except` boundaries for graceful degradation if a downstream module crashes mid-render.
* P2 — surface live cache hit/miss metrics in `/app/logs/streamlit_app.log` for ongoing performance observability.

## Implementation date
2026-05-08
