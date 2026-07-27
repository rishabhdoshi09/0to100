# Phase 0 — Truth & Contradiction Audit

**Scope:** compare documented/intended behaviour against actual implementation.
**Rule:** code is evidence of *current* behaviour; docs are evidence of *intended*
behaviour. Where they differ, the code wins and the gap is logged below.

**Classification legend:**
`MONEY_CRITICAL` (can lose real money) · `EVIDENCE_CRITICAL` (can corrupt a research
conclusion) · `RELIABILITY` (can silently stop working) · `ARCHITECTURE` (structural
debt) · `DOCUMENTATION` (docs overstate reality) · `COSMETIC`.

Contradictions are ordered by severity. Money- and evidence-critical items are the
mandatory first fixes.

---

## C-01 · "No fake data, ever" is false in research-reachable paths
**Class:** EVIDENCE_CRITICAL
**Claim:** `CLAUDE.md` invariant #1 — "a symbol with no real data is skipped, never
simulated. Demo-data fallbacks were removed deliberately."
**Reality:** yfinance and/or Google Finance are reachable from research- and
scan-grade code paths — `scan/bulk_fetcher.py`, `scan/relative_strength.py`,
`scan/breakout_health.py`, `scan/quality_engine.py`, `scan/conviction.py`,
`data/market_data.py`, and `gauntlet/momentum.py` (`--source yf`). EXP-005 was run
directly on yfinance data and produced a survivorship-inflated "PASS."
**Impact:** research grade cannot currently guarantee its inputs are trustworthy. A
strategy can be validated on a source that silently substitutes biased/adjusted-
differently data.
**Fix direction:** every dataset declares a trust class (see `DATA_CLASSIFICATION.md`);
`RESEARCH_GRADE` execution refuses any `OPERATIONAL_ONLY`/`DISPLAY_ONLY`/`PROHIBITED`
source. yfinance/Google become `DISPLAY_ONLY` and are unreachable from research.

## C-02 · Portfolio metrics are synthesised from independent per-trade R
**Class:** EVIDENCE_CRITICAL
**Claim:** the gauntlet reports CAGR / Sharpe / drawdown per strategy.
**Reality:** `gauntlet/runner.py::_equity_and_drawdown` builds a "modelled account"
by compounding each trade's R at a *fixed 1%-risk-per-trade, independently* —
`np.cumprod(1 + 0.01*r)`. There is **no portfolio NAV, no cash constraint, no
overlapping-position accounting, no turnover, no concurrency limit.** Trades that
could never have been taken together (insufficient capital, same day, same cluster)
are all "taken." The label is `modeled CAGR`, but it is consumed as if comparable to
Nifty's real CAGR. (The momentum gauntlet, `gauntlet/momentum.py`, is better — it
uses monthly *portfolio* returns — but the primary signal path does not.)
**Impact:** any CAGR/Sharpe/alpha from the per-trade path is not a portfolio result
and must not back a deployment claim. This is the core reason for Decision C.
**Fix direction:** the chronological portfolio simulator (§9 of the directive) with
an immutable daily NAV ledger becomes the *only* source of portfolio metrics.

## C-03 · Universe is survivorship-biased; point-in-time is a documented stub
**Class:** EVIDENCE_CRITICAL
**Claim:** `data/nse_universe.py::point_in_time_universe(as_of)` provides
survivorship-aware membership.
**Reality:** with no `logs/universe_history.json` on disk (there is none), it returns
**today's** constituents with `survivorship_complete=False`. Every experiment to date
(EXP-002…005) ran on the current universe → survivorship-biased. There is no security
master, no listing/delisting dates, no bitemporal knowledge tracking.
**Impact:** all historical results are optimistically biased; EXP-005 demonstrated the
magnitude (a 37% CAGR mirage).
**Fix direction:** bitemporal security master + point-in-time reconstruction (§8).

## C-04 · Corporate actions detected but not event-backed or applied
**Class:** EVIDENCE_CRITICAL
**Claim:** `data/corporate_actions.py` back-adjusts prices; `core/data_integrity.py`
verifies.
**Reality:** adjustment requires `logs/ca_events.json`, which does not exist →
`load_events()` returns `{}` → **no adjustment is applied.** Research ran on raw,
unadjusted prices (phantom split/bonus gaps present). `data_integrity` only *detects*
gaps; in research-grade mode a CA must be event-backed, never inferred from a gap.
**Impact:** un-adjusted prices fabricate stop-hits, breakdowns and returns.
**Fix direction:** corporate-action ledger (§8) sourced from real CA events; research
grade fails closed if `verify_ca_adjustment()` is not PASS.

## C-04b · Live autopilot is enabled and Telegram taps can reach paper orders
**Class:** MONEY_CRITICAL
**Claim:** directive §15 requires live autopilot disabled during the overhaul.
**Reality:** `execution/autopilot.py` supports live arming; `alerts/telegram_actions.py`
provides button-tap actions. Invariant #4 restricts Telegram to paper-only, but the
path exists and must be hard-isolated during the overhaul.
**Impact:** an un-graduated strategy could place live orders.
**Fix direction:** disable live arming behind an explicit feature flag; document
graduation criteria (`EXECUTION_SAFETY.md`); no live path until forward-paper evidence.

## C-05 · Streamlit owns the lifecycle of every background service
**Class:** RELIABILITY
**Claim (docs):** "Background daemons started once in app.py."
**Reality:** `app.py` (≈line 142) starts `market_monitor`, `auto_scan`,
`telegram_actions` listener and `us_scanner` from within the Streamlit script, guarded
only by `st.session_state["monitor_started"]` — a **per-session** guard. Multiple
browser sessions / Streamlit reruns / multipage navigation can spawn duplicate
workers. There are no PIDs, health, heartbeats, graceful shutdown or idempotency.
**Impact:** duplicate scans, duplicate Telegram pushes, duplicate paper fills; workers
die silently with the tab; no observability.
**Fix direction:** independently runnable services with explicit entry points (§7).

## C-06 · Critical startup and evidence paths fail OPEN (silent `except: pass`)
**Class:** RELIABILITY + EVIDENCE_CRITICAL
**Reality:** every daemon start in `app.py` is wrapped in `except Exception: pass`;
`scan/auto_scan.py::_emit_trade_record`'s ledger emission is wrapped in
`except Exception: pass`; many `_probe_*`/data reads fail-open to `{}`/`None`.
**Impact:** a service that fails to start, or an evidence write that fails, is
indistinguishable from success. Directive Decision D requires evidence capture and
execution to fail *closed*.
**Fix direction:** split fail-open (news/LLM/UI) from fail-closed (evidence/execution);
transactional evidence writes that raise on failure.

## C-07 · Experiment "reproducibility" hashes a fingerprint, not the data
**Class:** EVIDENCE_CRITICAL
**Claim:** `gauntlet/registry.py` stamps experiments with a dataset hash for
reproducibility.
**Reality:** `dataset_hash()` hashes a small *fingerprint dict* (row counts, symbol
list), not the actual bytes of the input data. There is no immutable raw archive, no
per-file SHA-256, no dataset snapshot ID. Two different datasets with the same counts
collide; the same experiment on re-downloaded data cannot be proven identical.
**Impact:** results are not bit-for-bit reproducible from a snapshot ID.
**Fix direction:** immutable raw archive with SHA-256 per source file; dataset
manifests with `snapshot_id` and `source_file_hashes` (§8).

## C-08 · Multiple-testing accounting does not persist across experiments
**Class:** EVIDENCE_CRITICAL
**Reality:** the harness applies DSR/FDR *within a single run*, but there is no
persistent **trial ledger** across EXP-002…005 and their many inspected variants
(raw vs trend, 7y vs 15y, bhav vs yf). Each renamed/re-parameterised strategy escapes
cross-experiment multiplicity accounting.
**Impact:** the true family-wise error rate across everything we've tried is
unaccounted; a "survivor" could be the luckiest of a large hidden search.
**Fix direction:** a durable trial ledger; every inspected variant counts (§10).

## C-09 · Feature-schema versions can silently mix
**Class:** EVIDENCE_CRITICAL
**Reality:** `research/feature_store.py` carries a `SCHEMA_VERSION`, but there is no
enforced cohort selection — reads can mix rows written under different schema
versions without an explicit migration.
**Fix direction:** reads must pin a schema version or an explicit migration/cohort.

## C-10 · Documentation overstates maturity in several places
**Class:** DOCUMENTATION
**Reality:** `CLAUDE.md` presents CA adjustment, survivorship-awareness and the
research stack as operational; in fact they are interfaces awaiting real data
(`ca_events.json`, `universe_history.json` absent). `RESEARCH_LOG.md` is accurate and
should be the template for honesty.
**Fix direction:** docs must state implemented vs tested vs unproven vs blocked.

## C-11 · Regime labels leak / break over research windows
**Class:** RELIABILITY
**Reality:** EXP-005 bucketed 51/162 months into a `nan` regime because
`_nifty_regime_series` didn't cover the older yfinance dates. Regime evidence is
silently partial.
**Fix direction:** regime series must be point-in-time and coverage-checked; missing
coverage fails the regime test rather than emitting `nan` buckets.

## C-12 · Costs/slippage are modelled constants, never reconciled
**Class:** EVIDENCE_CRITICAL (for deployment claims)
**Reality:** `core/costs.py` uses assumed 0.22% + 0.10% slippage; there is no
reconciliation against realised fills. Fills are simulated *at the pivot* (optimistic).
**Fix direction:** cost model stays modelled and *labelled*; slippage graduates to
`OBSERVED` only after forward-paper reconciliation (§15/§16).

---

## Money- and evidence-critical fix order (mandatory first)
1. **C-04b** disable live autopilot + hard-isolate the Telegram order path.
2. **C-06** make evidence capture + execution fail closed.
3. **C-01 / C-05 (DATA_CLASSIFICATION)** trust classes; research grade refuses
   non-`RESEARCH_GRADE` sources.
4. **C-02** portfolio NAV ledger replaces per-trade compounding as the metric source.
5. **C-03 / C-04 / C-07** security master + CA ledger + immutable raw archive +
   snapshot IDs (the point-in-time data platform).
6. **C-08 / C-09 / C-11** trial ledger, schema cohorts, point-in-time regimes.

`RELIABILITY`/`ARCHITECTURE`/`DOCUMENTATION` items are addressed as their owning phase
lands (see `IMPLEMENTATION_PLAN.md`).
