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

## C-04b · Live autopilot must be migration-locked during the overhaul
**Class:** MONEY_CRITICAL · **Status:** live-lock FIXED (2026-07-27); Telegram
paper-only VERIFIED (2026-07-28)
**Claim:** directive §15 requires live autopilot disabled during the overhaul.
**Reality (original):** `execution/autopilot.py` supported live arming.
**Clarification (important):** Telegram *paper* ordering is **intended behaviour**
(invariant #4), not itself a contradiction. The contradiction was solely that LIVE
autopilot arming was reachable during the overhaul.
**Fix applied:** `_live_enabled()` gates LIVE arming behind `QT_LIVE_ENABLED`, fail
closed / default off (Milestone 1a). **`QT_LIVE_ENABLED` is a TEMPORARY migration
interlock, not strategy graduation** — setting it does not assert any strategy earned
live capital; it removes exactly one of many blocks. Full graduation (and a future
deployment-manifest gate) is documented in `EXECUTION_SAFETY.md`. PAPER autopilot and
Telegram paper actions remain fully operational; LIVE stays locked by default.
**Verification (Telegram paper-only) — see C-04c below.**

## C-04c · VERIFICATION: every Telegram order path is paper-only
**Class:** MONEY_CRITICAL (safety proof) · **Status:** VERIFIED · 2026-07-28
**Assertion:** no Telegram tap or command can ever place a LIVE order (invariant #4).
**Evidence:** `alerts/telegram_actions.py` has exactly **one** order path,
`_do_paper_trade()`, which calls `execution.trade_executor.place_trade(..., paper=True)`
— and `place_trade` forces paper whenever `paper=True` (`if paper or not kite_ready()`).
No other `place_trade(` call exists in the module. Locked by test
`TestTelegramCommands::test_telegram_order_path_is_always_paper`, which (a) asserts the
tap passes `paper=True` even with the app armed LIVE, and (b) fails if a second,
un-audited `place_trade(` call is ever added or the `paper=True` is dropped.

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

## C-13 · Day-P&L / circuit-breaker day boundary — RESOLVED (money-safety milestone)
**Class:** RELIABILITY (money-adjacent) · **Status:** FIXED · 2026-07-28
**Original finding:** surfaced at a UTC/IST date boundary (UTC 2026-07-27 23:58 =
IST 2026-07-28). `test_circuit_breaker_disarms` and `test_pnl_snapshot_live_and_day`
inserted trades with naive `datetime.now()` (the machine clock = **UTC** on the CI
box / a VPS) while the autopilot filters the trading day by IST `today_ist()`. Across
the boundary a genuine "today" trade was excluded, day-realised P&L read 0, and the
**daily-loss circuit breaker could fail to fire**. MONEY-adjacent.

**Root cause (confirmed):** the production India money-path was in fact IST-*correct*
already — `execution/trade_executor.py` persists `placed_at` as naive **IST**
wall-clock, and every autopilot day-query resolves "today" via IST `today_ist()`. The
defect was an **implicit, undocumented convention**: nothing single-sourced "storage =
naive IST" or "compare only via the IST trading day," so (a) a naive machine
`datetime.now()` could be compared against an IST date and silently mis-bucket, and
(b) the tests wrote machine-local timestamps and depended on the wall-clock instant
pytest ran.

**Fix applied (this milestone):**
- Canonical contract single-sourced in `core/market_clock.py`:
  `now_ist_naive()` (the STORAGE clock), `ist_day_of(ts)` and `is_ist_today(ts, today)`
  (the only sanctioned day-bucketing — accepts naive-IST *or* tz-aware and converts).
  The storage convention is documented inline (naive-IST legacy; tz-aware-UTC migration
  deferred but the query layer already tolerates it).
- `execution/trade_executor.py` now stamps `placed_at` via `now_ist_naive()`.
- `execution/autopilot.py` routes all three money-critical day-queries — day-P&L
  snapshot, EOD digest, and the **circuit breaker** `day_realized` — through
  `is_ist_today(placed_at, today)` (was a raw `str(placed_at)[:10] == today`). The
  per-day trade limit / traded-symbol dedup / daily-disarm keys already keyed off IST
  `today_ist()`; they are now covered by boundary tests.
- Tests corrected to write via the IST storage clock and made **deterministic** — a
  new `TestAutopilotDayBoundary` pins the IST "today" (monkeypatches `_ist_today`) so
  results are independent of both the machine timezone and the instant pytest runs.

**Tests added (all deterministic, network-free):** `market_clock` contract
(23:59 IST / 00:01 IST / UTC-date≠IST-date / tz-aware round-trip); circuit breaker
counts a 00:01-IST loss even when the UTC instant is the prior day; breaker ignores a
23:59-IST prior-day loss; day-realised counts only the IST day with no double-count;
PAPER **and** LIVE closes both IST-filtered; per-day trade limit resets on the IST day.

**Residual (deferred, tracked as C-14):** other `datetime.now()`/`date.today()` sites
outside the NSE money-path (US-paper autopilot, F&O expiry, Telegram display strings)
are not yet IST/ET-explicit. They cannot affect the NSE circuit breaker.

## C-14 · Non-NSE day/time sites are not timezone-explicit (scoped follow-up)
**Class:** RELIABILITY · **Status:** OPEN (scoped, out of the C-13 money-milestone)
**Reality:** the C-13 fix hardened the NSE India money-path (circuit breaker, day-P&L,
per-day limits). Remaining naive `datetime.now()` / `date.today()` calls persist in:
`execution/us_autopilot.py` (US-paper day keys + `datetime.now()` age math — the US
path is otherwise ET-explicit via pytz), `execution/fo_executor.py` (F&O expiry via
`date.today()` — F&O is paper-first/opt-in), and display strings in
`alerts/telegram_alerts.py` / `alerts/telegram_actions.py` (labelled "IST" but computed
from machine-local time → wrong text on a non-IST server).
**Impact:** on a non-IST/non-ET server the US-paper day boundary and F&O expiry can
shift; Telegram timestamps can misread. **None of these touch the NSE circuit breaker
or NSE live-order path.** Bounded to US-paper / F&O-paper / display.
**Fix direction:** extend the `market_clock` discipline (an ET-explicit analogue for
the US path; explicit tz for F&O expiry; format Telegram strings from `now_ist()`).
Addressed in the owning phase, not in this focused money-safety milestone.

## C-15 · Valuation & sector data are not point-in-time (surfaced, fail-closed)
**Class:** EVIDENCE_CRITICAL · **Status:** SURFACED + FAIL-CLOSED for the momentum
framework (2026-07-28); full fix deferred to the data-platform phase.
**Reality:** `data/fundamentals_cache.db` is `(symbol, data_json, fetched_at)` — one
CURRENT row per symbol with NO publication-date history, so historical valuation
cannot be reconstructed. Sector membership (`scan/sector_heat.py`) is likewise not
historically dated. Using either as-of a past bar would leak the future.
**Handling (EXP-006, ADR-002):** the Institutional Momentum Breakout framework treats
valuation as `VALUATION_DATA_UNAVAILABLE` unless a record carries a real `available_ts`
proving it pre-dates the observation (never forward-filled); valuation is CONTEXT only
(flags `EXTREME_PE`/`HIGH_EXPECTATION_RISK`, never rejects the primary momentum
candidate). Sector membership carries `SECTOR_MEMBERSHIP_NOT_PIT`. Both are recorded on
every observation as explicit limitations rather than silently assumed.
**Fix direction:** a bitemporal fundamentals ledger with publication dates and a dated
sector-membership history (the point-in-time data platform, ADR-001 §8). Until then the
framework fails closed on these inputs.
**Update (2026-07-28, EXP-006 evidence run):** the EXP-006 runner enforces this — its
data-quality gate fails closed on absent/corrupt data, and its **research-grade verdict
gate** downgrades any would-be PASS on survivorship-incomplete / CA-unadjusted data to
INCONCLUSIVE. In this environment the run returned INCONCLUSIVE(DATA_UNAVAILABLE): no
point-in-time NSE dataset exists here (empty bhav/index stores, no network, no
universe/CA/fundamental history). A defensible PASS is not attainable until at least
survivorship + CA reach research grade; a FAIL is attainable now.
**Update (2026-07-28, run executed + committed):** the frozen runner was executed and
its auditable artifact set committed (originally `docs/overhaul/exp006_run/`, commit
`6a865c8`). **Corrected status language:** that run was a **run attempt completed but
BLOCKED before candidate generation** → **INCONCLUSIVE — DATA_UNAVAILABLE**; the
economic hypothesis is **UNEVALUATED**; it is **NOT a historical evidence verdict**.
`TestCommittedRunRecord` guards the persisted record against a false verdict.

**Update (2026-07-28, data-acquisition milestone):** the blocked record was relocated
into an append-only run tree `docs/overhaul/experiments/EXP-006/runs/0001-blocked/`
(content unchanged) with a `run_manifest.json` (artifact SHA-256s). The
data-acquisition decision + contracts live in `docs/overhaul/data_acquisition/`
(source = NSE official archives; yfinance stays DISPLAY_ONLY). A **FAIL-direction
verdict gate** was added to `runner._decide`: an economic FAIL is retained only when
data limitations are one-directional favourable (survivorship); a CA-raw (either-way)
limitation downgrades a FAIL to INCONCLUSIVE. No EXP-006 threshold/feature/config-hash
changed.

## C-21 · Autonomous Research Brain: full self-driving research loop, one human gate
**Class:** ARCHITECTURE / safety · **Status:** ADDED · 2026-07-30
**Reality (user ask):** the system had no self-running automation — a human had to press
every button. Request: it should study the market on its own, reason in the open, build a
"data thread", reject weak ideas, and improve trades WITHOUT human intervention.
**Mitigation (built + tested, `research/auto_research/`):** a headless brain runs the whole
loop by itself — `loop.run_cycle()` observes data readiness → generates grammar candidates
→ reasons through each (structural leakage/complexity/PIT + evidence sample/concentration/
cost/drawdown) → REJECTS the weak ones → shortlists survivors → auto-advances the lifecycle
using **SYSTEM-only** transitions up to exactly **one** gate, `AWAITING_USER_APPROVAL`, and
**stops**. Every step is written to an **append-only** `ResearchThread` (JSONL, deterministic
content, wall-clock only as provenance) so a human can watch it think. `LearningLedger`
tracks per-family **decay vs improvement** across cycles and PROPOSES re-tested child
versions (`bump_version` → new config hash, old evidence never transfers) — proposals only,
never mutations of an active strategy. `scheduler.AutoResearchBrain` runs cycles on an
interval, surviving errors, sharing memory across cycles. **Safety proven by tests:** the
cycle report carries `acted_on_market=False` / `approved_anything=False` always; `_advance_
to_gate` uses only actor="system" hops and the step beyond the gate raises `LifecycleError`
for system (user-only); **synthetic** evidence is never presented as market evidence and
never becomes a proposal; **no research-grade data ⇒ honest `Discovery unavailable …`** and
zero proposals (fails closed to red). No module imports an order path; `ui/auto_research_
page.py` is read-only (approval still happens by a person in Strategy Studio); PAPER
autopilot, Telegram paper-only and the LIVE migration lock are unchanged.

## C-20 · Strategy Studio: autonomous discovery is research-only, user-gated
**Class:** ARCHITECTURE / safety · **Status:** ADDED · 2026-07-28
**Risk:** an autonomous strategy generator could silently implement / paper-deploy /
promote an idea, or sell it to the user with a single score.
**Mitigation (built + tested, `research/strategy_studio/`):** the system may generate and
REJECT strategies but a **USER-only lifecycle transition** is required to approve one for
PAPER (research code raises `LifecycleError`); approval is **immutable + PAPER-only**,
bound to one config hash (a tweak invalidates it); paper activation is a **separate
confirmation**; every material tweak makes a **new version + config hash** and must be
re-tested; discovery records **every attempt** (incl. rejects), applies multiple-testing
burden + untouched-test isolation + simpler-baseline + complexity penalty; confidence is
shown as **five separate** measures; and **synthetic fixtures are labelled non-evidence**.
No module imports an order path; PAPER autopilot, Telegram paper-only and the LIVE
migration lock are unchanged. Real market evidence still requires a research-grade dataset.

## C-19 · Historical Data Setup frontend (data management; presentation + ingestion only)
**Class:** DOCUMENTATION / usability · **Status:** ADDED · 2026-07-28
**Reality:** materialising real data required operator shell commands; a local layman
had no guided way to provide data, validate it, and run EXP-006.
**Fix (data-management only):** `research/momentum_breakout/data_setup.py` (pure engine:
safe ZIP extraction, content validation, readiness green/amber/red, deterministic
snapshot, overwrite-protected save into the CANONICAL stores via new
`bhavcopy_store.build_from_local` / `index_store.build_from_local` — no parallel DB) +
`ui/data_setup_page.py` (thin Streamlit page under More Tools) + guide
`docs/user-guide/HISTORICAL_DATA_SETUP.md`. The page runs the UNCHANGED frozen EXP-006
runner into a NEW immutable run dir (never overwrites `0001-blocked`), refuses a red
readiness gate, and contains NO order actions. EXP-006 thresholds/config-hash/detector/
entry/stop/exits/ablations are unchanged; research stays isolated from execution.

## C-18 · Test-suite completeness: test_scan_core reclassified as integration
**Class:** RELIABILITY / DOCUMENTATION · **Status:** RESOLVED · 2026-07-28
**Reality:** `tests/test_scan_core.py` uses synthetic data + mocks but its import chain
reaches heavy operational `scan/*` modules that make lazy data/network calls and stall
without network. Earlier runs called the suite "green" only by ad-hoc `--ignore`.
**Fix:** moved to `tests/integration/test_scan_core.py` and marked
`pytest.mark.integration`; `tests/conftest.py` excludes `tests/integration` from the
default run by classification (env-gated `QT_INTEGRATION`), so the **canonical
network-free suite is simply `python -m pytest`** (no ad-hoc `--ignore`). Integration
runs separately: `QT_INTEGRATION=1 python -m pytest tests/integration`. CI runs the
canonical suite (blocking) + integration (non-blocking).

## C-16 · Detector NaN-safety + O(n²) base scan (fixed during EXP-006 run)
**Class:** RELIABILITY · **Status:** FIXED · 2026-07-28
**Reality:** the frozen detector (`_detect_base`) and the simulator did not guard
against NaN/missing bars, so on real gappy data a missing session could fabricate a
spurious pivot/candidate or an unrealistic fill; and `_detect_base` rescanned every
base length at every bar (O(base_max²)), making a whole-market historical run
intractable.
**Fix:** both now fail closed on NaN/missing bars; `_detect_base` rewritten to an
O(base_max) incremental scan with **identical output** (all 39 pre-existing detector
tests unchanged). Pure robustness + performance — the tested hypothesis, thresholds,
pivot/base definition and config-hash semantics are unchanged (no new experiment id).

---

## Money- and evidence-critical fix order (mandatory first)
0. **C-13** ✅ DONE — IST-consistent day boundaries across the NSE money-path
   (circuit breaker / day-P&L / per-day limits) + deterministic boundary tests.
1. **C-04b** ✅ DONE (arming) — live autopilot migration-locked behind
   `QT_LIVE_ENABLED`; **C-04c** ✅ Telegram order path verified paper-only.
2. **C-06** make evidence capture + execution fail closed.
3. **C-01 / C-05 (DATA_CLASSIFICATION)** trust classes; research grade refuses
   non-`RESEARCH_GRADE` sources.
4. **C-02** portfolio NAV ledger replaces per-trade compounding as the metric source.
5. **C-03 / C-04 / C-07** security master + CA ledger + immutable raw archive +
   snapshot IDs (the point-in-time data platform).
6. **C-08 / C-09 / C-11** trial ledger, schema cohorts, point-in-time regimes.

`RELIABILITY`/`ARCHITECTURE`/`DOCUMENTATION` items are addressed as their owning phase
lands (see `IMPLEMENTATION_PLAN.md`).
