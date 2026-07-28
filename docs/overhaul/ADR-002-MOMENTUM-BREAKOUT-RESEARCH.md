# ADR-002 — Institutional Momentum Breakout Research Framework (EXP-006)

**Status:** accepted · **Date:** 2026-07-28 · **Branch:** `overhaul/evidence-lab`
**Supersedes:** none · **Related:** ADR-001 (Evidence Lab), C-01/C-03/C-04 (TRUTH_AUDIT)

## Context

The milestone asks for a point-in-time, reproducible **research** framework that
studies whether stocks with prior leadership, a long contracting base, a confirmed
breakout, small structural risk and strong sector support have positive forward
expectancy after realistic Indian cash-equity costs. The hypothesis is **not
assumed valid** — the result may be PASS, FAIL or INCONCLUSIVE. Candidate detection
is research-only and must never reach any execution path.

The repository-first requirement: inspect what exists, reuse correct canonical
pieces, do not duplicate ATR/MA/RS/volume/breakout under new names, and flag any
existing feature that is **not point-in-time safe** before reusing it.

## Repository inspection — what exists and how it is used

**Reused as-is (already correct Evidence-Lab contracts):**

| Component | Reused for |
|---|---|
| `research/harness.py` | The evidence gate: `evaluate()` (DSR/PSR, alpha vs benchmark, block-bootstrap CI, BH-FDR). The framework invents **no** new statistics. |
| `gauntlet/ledger.py::TradeRecord` | Immutable per-trade ledger shape + benchmark window returns. |
| `gauntlet/registry.py`, `gauntlet/freeze.py` | Provenance: experiment id, git commit, dataset hash, config hash, seed. |
| `research/feature_store.py` + `feature_schema.py` | Immutable, schema-versioned observation snapshots (the store the framework's observations can be frozen into). |
| `scan/momentum.py` | Pure, array-based momentum/trend math patterns (already PIT-safe). |
| `data/nse_universe.point_in_time_universe(as_of)` | Survivorship-aware membership (flags `survivorship_complete=False` honestly). |
| `data/bhavcopy_store.get_ohlcv` | CA-adjusted-on-read EOD history. |

**Inspected and deliberately NOT reused — not point-in-time safe:**

| Component | Why it cannot be used in research |
|---|---|
| `scan/relative_strength.py` | Fetches Nifty live (Kite/yfinance) and anchors on `date.today()`. Cannot be evaluated "as of" a historical bar → would leak the future. |
| `scan/unified_scanner.py`, `scan/breakout_sniper.py` (ATR/pattern) | Coupled to live scan context and score calibration; not a pure as-of computation. |

This is the crux of requirement #4. The existing ATR/RS/breakout code is
**operational**, not research-grade. Reusing it would silently import look-ahead.

## Decision

1. **Add one canonical, pure, point-in-time primitive module** —
   `research/momentum_breakout/pit.py` — providing ATR, SMA/EMA, returns/momentum,
   drawdown, CLV, volume-z / rvol / dry-up, and cross-sectional relative strength.
   Every function takes an explicit observation index `i` and may read only
   `arr[:i+1]`; `window()` and `assert_no_future_read()` make that testable and
   fail closed (`FutureLeak`). This is **not** duplication-for-its-own-sake: it is
   the single research-grade primitive, and the live operational implementations
   remain untouched for the scanner. The tension in the milestone
   ("don't duplicate" vs "flag non-PIT-safe") is resolved by making PIT-safety the
   discriminator.

2. **Canonical research object** — `MomentumBreakoutObservation`
   (`observation.py`): raw component features (never an opaque score), transparent
   component scores, structural stop + initial risk, six distinct timestamps,
   data-quality flags, eligibility + every rejection reason, and the full
   reproducibility stamp. Its `event_id()` is the deduplication identity
   (symbol + base + pivot + breakout-day + detector/config).

3. **Valuation fails closed.** `data/fundamentals_cache.db` is
   `(symbol, data_json, fetched_at)` — one *current* row per symbol with **no
   publication-date history**. Therefore historical valuation is treated as
   `VALUATION_DATA_UNAVAILABLE` unless a record carries a real `available_ts`
   proving it pre-dates the observation. Current fundamentals are **never**
   forward-filled into the past. Valuation is CONTEXT: it flags
   `EXTREME_PE` / `HIGH_EXPECTATION_RISK` but **never** rejects the primary
   momentum candidate; the experiment separately measures whether expensive
   candidates behave differently.

4. **Sector membership is not historically dated.** Breadth/sector-RS are usable
   from current members' PIT price series, but every observation carries
   `SECTOR_MEMBERSHIP_NOT_PIT` so the limitation is never hidden.

5. **Event identity is a research construct** — derived only from price structure,
   **never** from paper/live order journals. Deduplication here has nothing to do
   with execution de-duplication.

6. **Research-only isolation.** The package imports nothing from `execution/`,
   `alerts/`, the broker, or GTT. A regression test enforces this and re-asserts
   the prior milestone's money invariants (PAPER autopilot operational, Telegram
   paper-only, LIVE migration-locked).

## Consequences

- The framework can be run on `RESEARCH_GRADE` data as it becomes available; until
  then it fails closed on the parts that are not point-in-time (valuation; full
  survivorship; dated sector membership) rather than fabricating them.
- Reproducibility is bound to `config_hash` (thresholds + primitive/detector/
  feature/scoring versions). Changing a primary threshold after seeing results is a
  new experiment id, not a silent edit.
- No new statistic is introduced — the existing gauntlet/harness decides the verdict.

## Not done in this milestone (explicitly out of scope)

Service extraction, portfolio-simulator, live-strategy deployment, UI redesign,
broad Phase-1 refactoring. No execution/paper/Telegram/GTT wiring. A Research Center
view is deferred (would be display-only, no execution buttons) to avoid broad UI work.
