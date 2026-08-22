# SEPA-003 research protocol

**Status:** Predeclared. Written and committed **before** SEPA-003
feature tables, decay numbers, component survival results, or the
decision file exist.

**Experiment:** SEPA-003 — Edge Decay Attribution & Component Survival  
**Prior evidence (immutable):** SEPA-001, SEPA-001R, SEPA-001R2,
SEPA-001R2.1  
**Frozen core F identity:** `sepa-001r2.v1` / `vcp_causal_v2` /
`pivot_last_contraction_v1` / config hash `76acdb2bb188a5f4`

This protocol does **not** change any core-SEPA threshold. It does **not**
re-open 2025–2026 as untouched out-of-sample. It does **not** authorise
paper, live, broker, GTT, or autopilot work.

SEPA-001 through SEPA-001R2.1 configs, JSON, ledgers, walk-forwards, and
decision files are **not modified** by this experiment.

---

## 1. Why this experiment exists

SEPA-001R2.1 tested core F honestly enough for the predeclared
confirmation block to **REJECT** (n=975, E[R]=−0.122).

That failure is consumed evidence. The next job is **attribution**, not
rescue:

1. Why the apparent 2020–2023 edge decayed.
2. Which individual SEPA *concepts* still carry information.
3. Whether those concepts belong as **features / filters** later.
4. Whether core F should be retired as a standalone strategy candidate.

A new rule found here is a **NEW_HYPOTHESIS**. It is never
`VALIDATED_EDGE`, never `untouched confirmation`, never
`production validated`.

---

## 2. Consumed blocks (do not relabel)

Calendar assignment is by signal `as_of`, same as R2.1.

| Era label in SEPA-003 | Dates | Role now |
|---|---|---|
| Winning historical era | first eligible date → **2023-12-31** | Diagnosis / descriptive |
| Weak era | **2024-01-01** → **2026-08-21** | Diagnosis (validation + consumed confirmation) |
| Year slices | 2020, 2021, 2022, 2023, 2024, 2025, 2026 YTD | Stability, not OOS |

R2.1 names (development / validation / confirmation) remain the **historical
record** of how F was tested. SEPA-003 may **cite** those numbers. It may
not treat 2025–2026 as a fresh holdout for a newly fitted rule.

If a later experiment wants a new untouched window, it must wait for
**dates after this milestone’s last bar** and write a new protocol first.

---

## 3. Primary predeclared hypotheses

These eight are locked now. All other cuts are **exploratory**.

| ID | Claim | Primary contrast | Success language |
|---|---|---|---|
| **H1** | Stage-2 *strength* reduces adverse outcomes versus otherwise comparable baseline candidates. | R2.1 A vs B (already computed) plus reconstructed MAE / fail-rate / left-tail among Stage-2 vs non-Stage-2 names in the same year. | Adverse-selection reduction. **Not** “an edge”. |
| **H2** | Cross-sectional RS has a stable positive relationship with outcome. | Prespecified RS buckets 50–69 / 70–79 / 80–89 / 90–94 / 95–99; continuous rank. Must not be a one-year spike. | Relationship, not a new cutoff. |
| **H3** | VCP **binary** status adds incremental predictive value beyond Stage-2 + RS. | VCP vs matched non-VCP among Stage-2 + RS, same year/regime/RS bucket. Frozen detector only. | Incremental information. |
| **H4** | Tighter **final** contractions improve breakout outcomes. | Final contraction depth (continuous + quartile). Frozen VCP geometry. | Direction + stability. |
| **H5** | Volume dry-up improves breakout outcomes. | `dry_up_ratio` (continuous + quartile). | Direction + stability. |
| **H6** | Lower pivot-entry extension improves MAE and failure rate. | Distance from pivot / buy-zone position / gap %. **No new buy-zone.** | MAE / fail-rate, not optimized expectancy. |
| **H7** | SEPA performance is materially conditional on broader market regime. | PIT regime states below. Requires a working classifier (R2.1 was `unknown`). | Material difference in E[R] or fail-rate; not a gate. |
| **H8** | Sector / group leadership adds information beyond stock RS. | Leading vs neutral/weak groups after controlling for stock RS. If mapped coverage is too thin: **INSUFFICIENT_PIT_SECTOR_DATA**. | Incremental information. No guessed labels. |

Exploratory hypotheses (not primary): contraction count, base depth,
base duration, tightness composite, SMA distances, 52-week location,
stop width, breakout CLV, liquidity, individual Trend Template rules,
shallow-tree interactions.

---

## 4. Labels that are forbidden / required

Forbidden on any SEPA-003 output:

- untouched confirmation
- final OOS
- production validated
- VALIDATED_EDGE
- PROMOTE (as a deployment word)
- paper-eligible / DEPLOYMENT_ELIGIBLE

Required when a new conditional cut looks interesting:

- `NEW_HYPOTHESIS`
- `requires_future_validation`
- `2025_2026_already_observed`

Harness `PROMOTE` from PSR/DSR, if it appears, is remapped to
`STATISTICAL_SIGNAL` and is **not** a trading licence.

---

## 5. Data and reconstruction

**Prices:** official NSE bhav already on disk (2019-08-23 → 2026-08-21).
No fabricated bars. CA policy remains `ca_sharecount_v1` (no inferred
factors). Causal CA segments from R2.1 may be reused for lookback
isolation; they are not rewritten.

**R2.1 trade rows were not persisted.** SEPA-003 reconstructs unique
core-F setups from `docs/overhaul/experiments/SEPA-001R2/setups.jsonl`
(variant `F`) using the **frozen** eligibility/VCP/entry functions.
Reconstruction must:

- use `date_step = 1` semantics for the fill search after detection
- use next-open fill + actual costs as in R2.1
- store `entry_date`, `exit_date`, `hold_sessions`
- report reconstruction n and E[R] beside the frozen R2.1 F table
- **not** retune if the reconstructed n differs

Ladder-level A/B/C/D/E/G numbers are **cited from R2.1**, not recomputed
as a second official A–G run.

**G rows** (`g_signal_rows.jsonl`) are a Stage-2+RS forward-% panel for
H2 / matching. They are not SEPA R.

---

## 6. Regime classifier (PIT)

R2.1 called `_nifty_regimes()` → official index store. That store was
empty / unaligned, so every F row was `unknown`. That is a data-plumbing
failure, not a market fact.

SEPA-003 builds `regime_pit_v1`:

**Index series (priority):**

1. Official NSE index OHLC if a historically deep local store exists.
2. Else a **documented** equal-weight NIFTY 50 proxy from official bhav
   closes of `data.nse_universe.NIFTY50` names that have a bar on that
   session (`NIFTY50_EQUALWEIGHT_PROXY_BHAV`). Survivorship of the
   *member list* is today’s list — class `PIT_DEGRADED`. The *prices*
   are as-of.

**States** (deterministic, no future returns):

| State | Rule (all use information ≤ as_of) |
|---|---|
| STRONG_BULL | Close > SMA50 > SMA200, SMA200 rising (21-session), 20d return > +6% |
| BULL | Close > SMA50 and 20d return > +2%, not STRONG_BULL |
| BEAR | Close < SMA50 and close < SMA200 and (20d return < −8% or SMA200 falling) |
| CORRECTION | Close < SMA50 and 20d return < −2%, not BEAR |
| SIDEWAYS | Residual classified session |
| UNKNOWN | <200 proxy sessions or undefined SMAs |

Also persist, PIT-safe: Nifty trend state, distance vs 50DMA / 200DMA,
50DMA slope, 200DMA slope, 20d / 63d index return, breadth
(% investable above own SMA50 / SMA200) when the as-of screen exists.

**Invariance test:** appending future bars must not change any historical
as-of label. No future return may enter the current-date rule.

No regime **gate** is added in this milestone.

---

## 7. Sector / industry map

R2.1 used `scan.sector_heat.sector_of`, which only parses NIFTY500
comment groups. NIFTY50 names (RELIANCE, INFY, …) were therefore
UNKNOWN — a coverage hole, not “no industry”.

SEPA-003 map `sector_map_v1`:

- Parse NIFTY500 comment groups (same source as production heat).
- Overlay documented large-cap industries for NIFTY50 / residual
  NIFTY100 names **not** in those comments.
- Unmapped names stay `UNKNOWN`. Never invent a sector from price
  behaviour.

The map is a **current** classification applied historically.
`sector_identity_pit = false`. Sector *returns* and ranks at as_of use
only members with bars ≤ as_of.

Persist coverage by year. H8 may conclude
`INSUFFICIENT_PIT_SECTOR_DATA` if mapped F coverage is too thin or the
identity is too degraded to support a claim.

No sector **gate**.

---

## 8. Feature table (setup level)

One row per reconstructed unique F fill (and, separately, matched
controls). Minimum fields:

**Trend:** eight Trend Template booleans + distances above SMA50 / 150 /
200, SMA200 slope, distance from 52-week high / low.

**RS:** percentile, raw weighted score, 3m/6m/9m/12m components,
benchmark excess if a PIT index proxy exists.

**VCP:** contraction count, each depth, first / final depth, final/first
ratio, base depth, base duration, tightness, quality score, dry-up
ratio, recent/base volume, pivot revisions, setup maturity, left-censor
flag.

**Entry:** pivot distance, buy-zone position, gap %, next-open distance,
breakout volume / CLV when bars exist, ATR, stop % , stop type, stop/ATR.

**Context:** regime at detection, regime at entry, sector, sector RS /
rank, stock vs sector, n strong names in group, liquidity, turnover,
price, year. Market-cap proxy only if PIT-safe (turnover×price is a
liquidity proxy, not official mcap).

**Outcome:** net R, MAE R, MFE R, +1R, +2R, stop-before-1R, hold
sessions, failed breakout, CA-censored flag.

Manifest versions: `sepa-003.v1` feature set + frozen R2.1 detector
versions.

---

## 9. Decay analysis

Compare winning era (→2023-12-31) vs weak era (2024-01-01→).

For each important feature: n, median, Q1, Q3, Cliff’s δ or rank-biserial
effect size, and outcome conditional on the feature.

Specifically test whether later setups show: weaker regimes, weaker
group leadership, poorer dry-up, deeper VCPs, wider stops, larger gaps,
worse pivot geometry, lower liquidity, different contraction counts,
more extended leadership, worse follow-through.

Allowed decay verdicts:

- `MARKET_CHANGED` — feature distributions similar; outcomes worse in the
  same bins (especially regime mix).
- `POPULATION_CHANGED` — setup/feature mix shifted.
- `UNSTABLE_EDGE` — neither mix nor regime explains the collapse;
  year-level sign flips remain.
- `INCONCLUSIVE` — cannot distinguish.

---

## 10. Component survival and matching

Prespecified RS buckets only. Do **not** pick a threshold from the best
bin.

A feature is **not** durable if: one year drives the plus, one sector
dominates, the sign flips repeatedly, or n is postcard-sized
(n < 30 in the slice used for the claim).

Classifications: `ROBUST_POSITIVE` | `CONTEXT_DEPENDENT` | `UNSTABLE` |
`NO_SIGNAL` | `INSUFFICIENT_DATA`.

Matching for H3/H4/H5/H6: stratification on year + RS bucket + regime
(+ sector when mapped). Compare VCP / dry-up / tightness / pivot
proximity against Stage-2+RS controls **without** that characteristic.
Simple stratified mean difference + bootstrap CI. Regression / Huber /
logistic are explanatory, not production models. Nested/rolling
coefficient signs are reported; the fitted model is **not deployed**.

“Loses less money than A” is **not** an edge. It may still be
`adverse_selection_reducer` for a later ensemble.

---

## 11. Multiple testing

- Count primary (8) and exploratory tests separately.
- Inferential claims on primary H1–H8 use Benjamini–Hochberg FDR at q=0.10
  across those eight (two-sided where a direction was not strictly
  one-sided).
- Exploratory p-values are descriptive.
- Bootstrap CIs (block where serial dependence is obvious).
- n < 30 → `INSUFFICIENT_DATA`, no significance language.

---

## 12. Forward observation (research ledger only)

Specify a passive recorder for dates **after** this milestone’s last
bar:

- candidate, frozen features, hypothetical entry/stop, later outcome,
  regime, sector, data-quality flags

It must not place orders, open paper tickets, or hook autopilot.
If wiring it into `app.py` would change production runtime, **only
document** the design. Do not activate.

---

## 13. Decision rule (exactly one)

After the reports, answer the fourteen component questions, then choose
exactly one:

- **A** — RETIRE CORE SEPA; RETAIN SELECT FEATURES
- **B** — KEEP CORE SEPA AS RESEARCH BENCHMARK ONLY
- **C** — CONDITIONAL SEPA HYPOTHESIS WARRANTS FUTURE VALIDATION
- **D** — EVIDENCE DOES NOT SUPPORT FURTHER SEPA WORK

C requires a clearly labelled `NEW_HYPOTHESIS` and an explicit future
window. It still does **not** promote paper or live.

Stop after the decision. No paper integration.

---

## 14. Required artefacts (after this protocol is committed)

- `SEPA_003_REGIME_AUDIT.md`
- `SEPA_003_SECTOR_AUDIT.md`
- `SEPA_003_FEATURE_DATASET.md`
- `SEPA_003_DECAY_ANALYSIS.md`
- `SEPA_003_COMPONENT_SURVIVAL.md`
- `SEPA_003_MATCHED_CONTROLS.md`
- `SEPA_003_ENTRY_ANALYSIS.md`
- `SEPA_003_RESULTS.md`
- `SEPA_003_DECISION.md`
- `sepa_003_features.parquet` (or jsonl if parquet is impractical)
- `sepa_003_stats.json`
- `sepa_003_hypotheses.json`
- `sepa_003_feature_manifest.json`
- `SEPA_003_FORWARD_LEDGER.md` (design)

---

## 15. Commit barrier

No results file, no decision, and no expectancy table may be committed
until **this protocol is on the branch**.
