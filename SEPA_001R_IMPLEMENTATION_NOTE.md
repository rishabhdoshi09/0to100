# SEPA-001R — Research Hardening Note

**Experiment:** Canonical SEPA eligibility retest after SEPA-001  
**Status:** Research only — no paper, autopilot, broker, GTT, or live wiring  
**Eligibility version:** `sepa-001r.v1`  
**Parent evidence:** `docs/overhaul/experiments/SEPA-001/` (immutable)  
**This namespace:** `docs/overhaul/experiments/SEPA-001R/`

SEPA-001 delivered a canonical object and tests. Its research verdict was
**MODIFY AND RETEST**. Core SEPA (variants E/F) was **INCONCLUSIVE** because it
barely traded. This note records the diagnosis and the hardening plan **before**
code changes. Rules will not be widened merely to manufacture fills.

---

## 1. Causes of SEPA-001 failure / inconclusiveness

| Cause | What SEPA-001 actually showed | Why it blocks a claim |
|-------|-------------------------------|------------------------|
| VCP evaluated after the move | Median distance to pivot ≈ **+9.9%**; 72% of detections already **>1.5%** extended | Specific-entry SEPA never got a fill; E/F ≈ 0 trades |
| Pivot too early in the base | Pivot = **highest** contraction swing high (usually the *first* high) | Price coils under later, lower resistance; buy-zone around the early high is already behind the market |
| Coarse sampling | `sample_step=10` (and a 20-step diagnostic) | A 1–3 session buy-zone window is routinely skipped |
| No setup lifecycle | Each sampled bar is an independent row | Even with daily bars, one VCP would become many pseudo-trades |
| Corporate actions absent | `logs/ca_events.json` missing; `pit_safe=false` | Unadjusted splits/bonuses fabricate Stage-2 failures and fake bases |
| Tiny, biased universe | Top **80** names by *recent* rupee turnover | Survivorship + liquidity bias; n too small for D/E/F |
| PIT overstated | `survivorship_complete=true` from local first/last sessions, not an official listing archive | Cannot claim PIT_STRONG |
| Wrong fill model on D | Variant D still used scanner `entry=last price` and 2×ATR / 4×ATR | Point estimate (+0.36R, n=34) is not a SEPA result |
| RS/Stage-2 on scanner path only | B and C improved A, still UNDERPOWERED | Useful filters, not evidence that *core SEPA* has edge |
| No walk-forward / multiple-testing | One in-sample grid; RS≥80/90 on n=22–45 | Cannot adopt thresholds from postcards |

**What was already correct (do not rewrite):** strict 8/8 AND template,
cross-sectional `rs_cs_v1` (not Nifty excess), fail-closed missing data, buy-zone
hard gate (`entry = last price` forbidden), structural stop as truth / ATR
diagnostic only, no fabricated pivots, research-only wiring.

---

## 2. Corporate-action plan

**Policy (unchanged):** never invent factors from price gaps.

1. Audit `data.corporate_actions` + `data.nse_ca_ingest` + `core.data_integrity.verify_ca_adjustment`.
2. If an official NSE corporates-API ledger can be fetched, ingest **only**
   unambiguous share-count events (split / bonus / consolidation). Cash
   dividends stay provenance-only (`ca_sharecount_v1`).
3. If the ledger is absent or verify does not PASS:
   - `ca_complete = false` (events on disk are **not** enough)
   - scan raw frames for consecutive-session phantom gaps
   - **exclude** unresolved-gap symbols from the research book
   - mark the run **PIT_DEGRADED** (or PIT_UNVERIFIED if we cannot even scan)
4. Every run records: CA source, coverage period, symbols covered, unresolved
   symbols/events, adjustment version/hash, verification result.

`ca_complete = true` only when `verify_ca_adjustment` actually passes.

---

## 3. Universe plan

1. Start from official bhavcopy symbols with complete OHLCV — not today's
   Kite/NSE list applied backward.
2. Point-in-time membership: `logs/universe_history.json` if present.
   - Official/operator archive → membership may be PIT-usable
   - `bhav_inferred` first/last sessions → **PIT_DEGRADED**, never
     `survivorship_complete=true` as a research claim
   - missing file → today's survivors, **PIT_UNVERIFIED**
3. Investability screen (as-of, not today):
   - min close ≥ ₹20
   - min 20-session turnover (close×volume)
   - min 260 sessions of history at as-of
   - required OHLCV columns present
   - not on the unresolved-CA gap list
4. Report starting size, eligible size by year, exclusions, turnover
   distribution, survivorship class.

Target: broadest **historically defensible** NSE book in the local store
(hundreds, not 80), still liquid enough to be tradable.

---

## 4. VCP timing diagnosis (hypothesis)

The SEPA-001 zigzag is **causal on completed swings** (a swing exists only after
a `min_reversal_pct` reversal). It is **not** the fractal `find_swings(left,right)`
path, which *would* need future bars.

Lag enters in three other ways:

1. **Pivot definition.** Highest swing high in the contraction sequence is
   typically the **first** (widest) high. After two tightenings, price is near
   the *last* resistance, often ~5–15% below the first high. A 1.5% band around
   the first high is already in the rear-view mirror. This matches median +10%.
2. **Pattern recognition vs entry.** `TOO_FAR_BELOW_PIVOT` (price < 92% of that
   early pivot) **fails the VCP itself**, so the engine never tracks the coil
   while it is still forming. Entry validity is the wrong layer for that test.
3. **Independent 10-bar snapshots.** No memory of “this base is live.” The
   1–3 session window inside the buy-zone is easy to miss.

**Not the fix:** widening the buy-zone to 5–10% so late detections become fills.

**The fix:** last-contraction (actionable) pivot; causal confirmation timestamps;
incremental state; daily evaluation with **one observation per setup**.

---

## 5. Pivot hypothesis

| ID | Definition | Structural rationale | How we will *not* choose it |
|----|------------|----------------------|-----------------------------|
| `pivot_pattern_high_v1` | Max contraction swing high (SEPA-001) | Pattern high / measured-move origin | Keep as diagnostic only |
| `pivot_last_contraction_v1` | Last confirmed contraction swing high | The resistance the stock is actually coiling under; knowable when that high is confirmed by the last pullback | **Default candidate** — measured, not fitted |
| `pivot_tight_area_v1` | Last contraction high (same object; tightness already required on final depth) | Minervini “tight area” is the last contraction, not a future high | No look-ahead shelf using future highs |

Primary research pivot: **`pivot_last_contraction_v1`**.

For each definition we will record detection latency, distance to breakout,
% of setups inside 0–1.5% when first actionable, false-break rate, expectancy,
and n. We will **not** pick whichever maxes historical R.

---

## 6. Daily evaluation architecture

- Primary resolution: **every session** (`sample_step=1`).
- Comparison grid (same book, same horizon): step 10 / 5 / 1 — to measure how
  much of the near-zero E/F problem was sampling, not structure.
- Signal known at **session close**; fill attempt is **next session open**
  (EXP-006 convention).
- Fill only if next open is **inside** the versioned buy-zone.
- Classifications, not silent market chases: `VALID_FILL` | `MISSED` |
  `GAP_THROUGH` | `EXTENDED` | `INVALIDATED`.
- Horizon 20 sessions; mark-to-market if neither stop nor +2R. CNC costs via
  `core.costs.net_r`. Structural stop only — no 2×ATR/4×ATR geometry on E/F.

---

## 7. Setup deduplication design

Unit of observation = **unique setup**, not a daily stock row.

- Identity frozen at first structural detection:
  `setup_id = sha256(symbol | base_start_date | eligibility_version | vcp_version | pivot_version)[:16]`
- Pivot used for *entry* may update only as new **confirmed** contractions appear;
  a new `base_start_date` is a new setup.
- Lifecycle: `FORMING → PIVOT_DEFINED → ENTRY_READY → {FILLED | GAP_THROUGH | EXTENDED | FAILED | EXPIRED}`.
- One fill attempt per setup, on the first `ENTRY_READY` close (next open).
- Symbol-level embargo while a setup is open/pending so the same coil cannot
  emit 15 trades in 15 days.

---

## 8. Statistical-validation design

Reuse `research.harness` (no new statistics):

- Expectancy net of CNC costs
- Block-bootstrap CI for mean R (`require_block_ci` / `block_bootstrap_mean_ci`)
- PSR; DSR with `n_trials` = number of **predeclared** variants (A–F plus the
  documented sensitivity tables, not an unbounded search)
- Year-by-year, regime, sector breakdowns
- Walk-forward: earlier years = research/training (stability of the *region*,
  not a single peak); later unseen block = confirmation. Canonical parameters
  stay the SEPA-001 defaults (RS≥70, buy-zone +1.5%, 8/8) unless a **stable
  region** is obvious on train *and* holds on test
- Do not treat `n≥30` as sufficient by itself
- Promotion gate (all must hold) is listed in the SEPA-001R brief; default
  stance is **do not promote**

Variants (same as SEPA-001, plus timing-corrected VCP):

| ID | Definition |
|----|------------|
| A | Production scanner baseline |
| B | A + Stage-2 structure (7 MA/52w rules) |
| C | B + RS percentile ≥ 70 |
| D | C + **corrected causal** structural VCP (scanner fill — not SEPA fills) |
| E | D gates + valid pivot/buy-zone + structural stop + next-open SEPA fill |
| F | Core SEPA eligibility without requiring scanner BUY |

Additional studies (accounted in `n_trials`, not silent mining): buy-zone
0.25–5%; RS 70/80/90 + percentile buckets; contraction count 2 vs 3; volume
dry-up required vs ranking vs off; Stage-2 7-rule vs 8/8 vs near-SEPA 7/8.

---

## 9. Files expected to change

**Immutable (do not overwrite):**
`docs/overhaul/experiments/SEPA-001/**`, root `SEPA_001_*.md`.

**New research outputs:**
`docs/overhaul/experiments/SEPA-001R/**`

**Engine (research only):**

| File | Change |
|------|--------|
| `research/sepa/config.py` | `sepa-001r.v1`, `vcp_causal_v1`, `pivot_last_contraction_v1` |
| `research/sepa/vcp.py` | Freeze `detect_vcp_legacy`; causal zigzag + last-contraction pivot |
| `research/sepa/vcp_state.py` | Incremental state machine |
| `research/sepa/types.py` | setup_id, vcp_state, knowable dates, pit_class (additive) |
| `research/sepa/engine.py` | Wire new VCP fields; still research-only |
| `research/sepa/entry.py` | Next-open fill classifier (no `entry=price` fallback) |
| `research/sepa/frames.py` | Honest `ca_complete` (verify must pass) |
| `research/sepa/integrity.py` | CA / universe / timestamp report |
| `research/sepa/setups.py` | Setup IDs + lifecycle |
| `research/sepa/universe_screen.py` | Investability screen |
| `research/sepa/ablation_r.py` | Daily / deduped / walk-forward runner |
| `research/sepa/timing.py` | Old vs new detection diagnostics |
| `research/sepa/__main__.py` | `001r` command |
| `tests/test_sepa_001r_*.py` | Causality, CA, dedup, entry, PIT |

**Must not change:** `execution/*`, autopilot, GTT, broker, production
`UnifiedScanner` BUY gates, Ideas/Ready scoring, Telegram live path.

---

## 10. Promotion stance entering the retest

Until the gate in the brief is fully met, the only admissible outcomes are
**KEEP RESEARCH-ONLY**, **MODIFY AND RETEST**, or **REJECT**. Paper promotion
is not on the table at the start of this work.
