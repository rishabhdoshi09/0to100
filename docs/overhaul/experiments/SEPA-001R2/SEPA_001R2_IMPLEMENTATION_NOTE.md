# SEPA-001R2 — Research Validity Corrections

**Experiment:** SEPA-001R2  
**Status:** Research only — no paper, autopilot, broker, GTT, or live BUY wiring  
**Eligibility version (planned):** `sepa-001r2.v1`  
**Parent evidence (immutable):**  
- `docs/overhaul/experiments/SEPA-001/`  
- `docs/overhaul/experiments/SEPA-001R/`  
**This namespace:** `docs/overhaul/experiments/SEPA-001R2/`

This note is written **before** eligibility/research-runner changes. The goal is
not to manufacture fills. The goal is to remove remaining research-validity
defects, then obtain historically defensible evidence for whether the
causally executable SEPA technical core has an edge on NSE equities.

SEPA-001R's KEEP RESEARCH-ONLY verdict is retained as history. Its numbers
are **not** final: several validity bugs can move Stage-2, RS, unique-setup
counts, and E/F fills.

---

## 1. Code-audit findings (SEPA-001R as landed)

### P0-1 — End-of-sample universe (LOOK-AHEAD)

**YES — the 001R historical universe is look-ahead-biased.**

`research/sepa/universe_screen.py` `load_research_frames()`:

- calls `screen_investable(raw)` with `as_of=None`, which uses **each frame's
  last bar** for price, session count, and 20-session turnover
- then caps `max_symbols` by ranking `_turnover(df)` on the **full** frame
- `run_ablation_r` reuses that frozen `frames` dict for every earlier `as_of`
- `build_rs_table(..., universe=list(frames))` therefore ranks against a
  **2026-selected top-100** on every historical date

A name that was illiquid or unlisted in 2025 but liquid in 2026 can sit in the
2025 RS denominator. A name that was liquid in 2025 but failed the 2026 cap
never appears.

### P0-2 — Top-100 called “NSE cross-sectional RS”

The canonical 001R book is 100 names after an end-of-sample screen of 2660.
That is a diagnostic cohort, not an NSE cross-section. RS≥70 on that book is
not the same gate as RS≥70 vs the as-of investable universe.

### P0-3 — Contraction truncation (stale pivot)

`research/sepa/vcp.py` `_contractions_from_swings()`:

```
if len(contractions) >= max_contractions:
    break
```

Contractions are collected in **time order**. The first `max_contractions` (6)
are kept. `_evaluate_structure` then does `seq = contractions[-max_contractions:]`,
which is a no-op after the early break. If a lookback contains 8 coils, the
actionable last-contraction pivot can be an **old** high, not the live coil.

### P0-4 — Config parameters vs code

| Parameter | Used? | Where | Behaviour |
|-----------|-------|-------|-----------|
| `min_contractions` | yes | `_evaluate_structure` | too-few fail |
| `max_contractions` | yes | `_contractions_from_swings` **and** seq slice | currently truncates from the **start** |
| `min_reversal_pct` | yes | `causal_zigzag` | swing confirmation |
| `vcp_lookback` | yes | `detect_vcp` | rolling window (drives setup-id drift) |
| `depth_expand_tol` | yes | expanding-pullback count |
| `final_vs_first` | yes | NOT_TIGHTENING |
| `max_final_depth_pct` | yes | FINAL_CONTRACTION_LOOSE |
| `max_base_depth_pct` | yes | BASE_TOO_DEEP |
| `volume_dry_up_max` / `volume_dry_up_required` | yes | VOLUME_EXPANDING |
| `near_pivot_frac` | diagnostic | `far_below_pivot`; fail only if `fail_vcp_if_far_below_pivot` (001R default False) |
| `min_recovery_bounce` | **NO** | config only | dead — will be removed from eligibility config |
| `swing_left` / `swing_right` | **NO** on money path | only `find_swings` (look-ahead diagnostic) | not causal zigzag |

### P0-5 — Setup identity drift

`setup_id = sha256(symbol \| base_start_date \| versions)`.
`base_start_date` is `seq[0]["high_date"]` inside the **current 120-bar
window**. When the first contraction ages out, a continuing base mints a new
ID. 001R's 944 unique setups can be inflated.

### P0-6 — Left-censor / EXTENDED

`SetupRegistry` treats `EXTENDED` as terminal. First snapshot of a base that
is already through the pivot is counted as an extended refusal, not
left-censored. Returning to the pivot after EXTENDED can re-open a fill if a
new `base_start` appears — a silent pivot-retest.

### P0-7 — CA / ABFRL

Ledger has 290 share-count events (2024-01-02 → 2026-08-21), hash
`e260673881d9e5c3`. `verify_ca_adjustment` FAIL, `gap_rate=0.0125`, flagged
**ABFRL**. ABFRL has **zero** ledger events. 001R excluded 38 gap names from
the *end-of-sample* book; it did not classify ABFRL (likely restructuring /
demerger) vs bad print vs genuine gap. Threshold will not be lowered.

### P0-8 — History depth

Local official bhavcopy: **424 sessions**, 2024-12-24 → 2026-08-21.
~252 sessions are warm-up for 200-DMA / 252d RS. Post-warm-up sample is
roughly **one year**. Inadequate for walk-forward or rare E/F.

Membership: `logs/universe_history.json` source=`bhav_inferred`, 2751 rows.
No official listing/delisting archive in-repo. Class remains at best
`PIT_DEGRADED` unless an archive is found.

### P1 harness language

`research.harness.evaluate()` can return **PROMOTE** on statistical PSR/DSR
alone. 001R labelled B/C `harness PROMOTE*` while PIT_DEGRADED and CI crossed
zero. R2 will separate `STATISTICAL_SIGNAL` from `DEPLOYMENT_ELIGIBLE`.

---

## 2. Corporate-action plan

- Keep `ca_sharecount_v1`: split / bonus / consolidation only. Never infer
  factors from price gaps.
- Re-ingest NSE corporates API over the **full downloaded history**, not just
  2024–2026.
- Classify unresolved discontinuities: split, bonus, consolidation, rights,
  demerger/spin-off, merger, special distribution, symbol/restructuring,
  genuine market gap, bad print, unknown.
- **ABFRL and similar:** Option B — quarantine the symbol/segment from trend,
  VCP, and return research. Do not fabricate a demerger factor.
- `ca_complete=true` only if configured verify PASSES on the research book
  after quarantine. Do not lower `gap_rate` 0.002.
- Future events still apply only for `index < ex_date` (`adjust_frame`).

## 3. Universe plan

At **every** `as_of_date`:

1. Membership candidates known as of that date (`bhav_inferred` listed≤as_of
   and not yet delisted — labelled degraded, not official).
2. Slice OHLCV through as_of.
3. Min history, min price, trailing turnover, CA quarantine — all as-of.
4. Rank liquidity on trailing data known then.
5. Canonical RS denominator = **full as-of investable set**.
6. Top-100 / 250 / 500 are **sensitivity studies only**.

Persist per decision: universe date, candidate count, investable count, RS
denominator, source, membership hash, selection reason.

## 4. VCP / pivot plan

- Collect **all** confirmed contraction legs in the lookback; do not break at
  `max_contractions`.
- Active sequence = most recent consecutive contractions ending at the latest
  low, length in `[min_contractions, max_contractions]`.
- If that live sequence is invalid, the **current** setup fails — do not
  resurrect an older valid coil as today's pivot.
- Pivot remains `pivot_last_contraction_v1` of the **active** sequence.
- Adding older bars before the active base must not move the current pivot
  backward.

## 5. Setup lifecycle / left-censor

Persistent ledger per symbol:

- Immutable setup ID frozen at first knowable active base (original base
  start, not rolling-window first high).
- Same ID while contraction dates overlap / last-contraction continues.
- New ID only after terminal reset or a documented new advance + new base.
- First observation already through the buy-zone → `LEFT_CENSORED` (not a
  miss, not a fill, excluded from entry-frequency).
- Forming below pivot → keep tracking.
- Observed ENTRY_READY then escape → `EXTENDED` / `MISSED` (canonical).
- Later return to pivot → `PIVOT_RETEST` research variant, **not** core F.

## 6. Daily evaluation / layers

- Evaluation dates: session calendar after warm-up.
- Layer 1 (primary): signal quality on unique setups.
- Layer 2: portfolio simulation only if a variant qualifies at Layer 1.
- A–D remain scanner-gated **signal** studies; do not call 1463 daily hits
  a portfolio.
- Add G (research-only): Stage-2 + RS without scanner BUY.

## 7. Statistical / walk-forward design (predeclared)

Calendar blocks (filled after data coverage is known):

- Warm-up: first 252 sessions (200-DMA + RS 252).
- Development / validation / unseen test: remaining complete calendar years,
  last year held out. Parameters not chosen on the last block.
- Canonical RS threshold stays **70** unless the *development* block alone
  justifies a change (it will not be optimized on the full sample).
- Buy-zone stays `[-0.25%, +1.50%]` unless development shows a **stable
  region** — never widen to mint trades.

Gate:

- Statistical: `STATISTICAL_SIGNAL` / `UNDERPOWERED` / `REJECT`
- `DEPLOYMENT_ELIGIBLE` requires PIT acceptable, CA verify, OOS, CI rule,
  effective sample. Paper shadow is observation-only.

## 8. Files expected to change

| File | Role |
|------|------|
| `research/sepa/vcp.py` | collect all contractions; active sequence |
| `research/sepa/config.py` | `sepa-001r2.v1`; drop dead eligibility knobs |
| `research/sepa/universe_pit.py` | **new** as-of investable universe |
| `research/sepa/lifecycle.py` | **new** persistent setup + left-censor |
| `research/sepa/ca_audit.py` | **new** classify / quarantine |
| `research/sepa/gates.py` | **new** statistical vs deployment |
| `research/sepa/ablation_r2.py` | **new** date-major runner + funnel |
| `research/sepa/study_r2.py` | **new** long-history study |
| `research/sepa/types.py` / `engine.py` | universe-as-of + lifecycle fields |
| `tests/test_sepa_001r2.py` | causality / PIT / CA / lifecycle tests |
| `docs/overhaul/experiments/SEPA-001R2/*` | reports + machine JSON |

Not changed: live execution, paper, autopilot, broker, GTT, UI, SEPA-001 /
SEPA-001R experiment files.

## 9. What would still not be claimed

- Official listing membership (`PIT_STRONG`) without an archive.
- `ca_complete` if verify still fails after quarantine.
- Promotion from a handful of E/F fills.
- That Minervini is “proven” on NSE. Evidence can say no.
