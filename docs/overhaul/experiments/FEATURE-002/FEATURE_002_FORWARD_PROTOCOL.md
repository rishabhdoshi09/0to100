# FEATURE-002 — Forward Protocol (immutable)

**Experiment:** FEATURE-002 — Future Shadow Ranking Validation  
**Claim class until mature:** `FORWARD VALIDATION ACTIVE`  
**Forbidden until FEATURE-003:** production ranking, BUY, Ready, paper, autopilot, GTT, broker.

This file is frozen **before** any FEATURE-002 outcome is reviewed.

---

## 0. FEATURE-001 freeze (consumed history)

| Field | Value |
|---|---|
| FEATURE-001 final commit | `aa2dc3b3ef5ff611b2cdd25faeabff93f80dae58` |
| Message | Record FEATURE-001 explanatory Trend and RS attribution |
| Trend version | `trend_features_v1` |
| RS version | `rs_features_v1` wrapping `rs_cs_v1` |
| Last FEATURE-001 **candidate** session | `2026-07-23` |
| FEATURE-001 sample grid | every 5th official session, 2020-09-28 → 2026-07-23 |
| FEATURE-001 verdicts | Trend and RS = `FORWARD-VALIDATE AS RANK FEATURE` (explanatory, not VALIDATED_EDGE) |

FEATURE-001 also used up to 20 **forward outcome bars** after each candidate date. Those bars are outcomes, not new candidate dates. They do **not** reopen 2026-07-23 as a FEATURE-002 observation.

---

## 1. Forward start (strict)

| Field | Value |
|---|---|
| `FORWARD_START_DATE` | `2026-07-24` (first calendar date strictly after last FEATURE-001 sample) |
| `FORWARD_START_TS_IST` | `2026-08-22T00:00:00+05:30` (protocol activation; live primary rows also require `recorded_at` ≥ this) |
| Exchange | NSE |
| Eligible primary row | `session_date >= 2026-07-24` **and** `source == live_scan` **and** `recorded_at >= FORWARD_START_TS_IST` **and** `feature_set_version == feature-002.v1` |

Any row with `session_date < 2026-07-24` is implementation-test only.  
Any replay / synthetic / FEATURE-001-panel row is `source != live_scan` and **cannot** enter primary statistics.

---

## 2. What this experiment is

For every future production scan cycle, **after** production has already ranked and decided:

1. Freeze Trend + RS feature vectors as-of that session (official bhav, no future bars).
2. Compute **shadow ranks** inside that cycle’s candidate set.
3. Later, resolve forward returns / MAE / MFE from later official bars.
4. Ask whether RS and/or Trend ordering beat production ordering.

Production continues exactly as before. Shadow ranks are never read by BUY, Ready, autopilot, sizer, GTT, Telegram, or broker code.

---

## 3. Frozen primary rankings

| ID | Name | Formula | Role |
|---|---|---|---|
| **R0** | Production rank | Final `auto_scan` order after sector heat, conviction, live overlay, edge veto, EV tag, prime tag. Rank 1 = first in that list. | Primary baseline |
| **R1** | RS-only | Higher `rs_percentile` (`rs_cs_v1`) is better. Ties: higher `rs_score`, then symbol. | Primary |
| **R2** | Trend-only | Higher `n_structure_passed` is better. Ties: higher `pct_above_sma200`, then higher `ma_spread_50_200_pct`, then symbol. | Primary |
| **R3** | Rank-aggregation composite | Within-set percentile ranks (average ranks for ties): `0.67 * PctlRank(rs_percentile) + 0.33 * PctlRank(n_structure_passed)`. Higher is better. Ties: R1 then R2 then symbol. | **Exploratory only** |

R3 is **not** a primary graduation candidate. It is frozen here so it cannot be tuned on FEATURE-002 outcomes.

**Why 0.67 / 0.33 (FEATURE-001, not FEATURE-002):** FEATURE-001 within-day top−bottom E[R] was RS 0.395 vs Trend 0.141 (~2.8:1). That is the sole justification for overweighting RS. The FEATURE-001 `score + RS/10` blend is **forbidden** (it diluted RS-alone).

`production_rank_version` = `auto_scan.final_order.v1`  
`shadow_rank_version` = `feature-002.ranks.v1`  
`feature_set_version` = `feature-002.v1`

A version bump is a **new experiment**. Old rows stay labelled with their version and are excluded from primary stats of the new version.

---

## 4. Candidate sets

Ranking is defined only among signals that existed at the **same scan cycle**.

- `candidate_set_id` = hash(`feature-002.v1`, `scan_cycle_id`)
- Members = every `StockSignal` the production scan serialized (one row per symbol)
- Family identity is retained on the row (`families[]`, `primary_family`)
- Do not rank across timestamps
- Do not pool families before reporting family-level metrics

---

## 5. Families (repository truth)

`scan/unified_scanner.py` `SIGNAL_META` keys. Evaluate independently. New keys are logged and held out until n is adequate.

`BREAKOUT_52W`, `BREAKOUT_RES`, `GOLDEN_CROSS`, `VOL_SQUEEZE`, `VCP`, `FLAT_BASE`, `CUP_HANDLE`, `HIGH_TIGHT_FLAG`, `ASC_TRIANGLE`, `DOUBLE_BOTTOM`, `PRE_BREAKOUT`, `ACCUMULATION`, `DELIVERY_SPIKE`, `NR7_COIL`, `POCKET_PIVOT`, `MOMENTUM`, `PULLBACK_SUPPORT`

---

## 6. Outcomes (resolved later)

From official history after `session_date`, never rewriting frozen features:

- next-session open
- 1 / 5 / 10 / 20 session close-to-close returns
- MAE / MFE vs session close (or production entry if present)
- +1R / +2R **only** when a valid production stop exists (`entry > stop`)
- actual production-trade outcome if the live journal later marks it traded

Unresolved stays `NULL`, never `0`.

---

## 7. Maturity (predeclared — do not peek-and-edit)

| Stage | Rule | Verdict allowed |
|---|---|---|
| QUIET | < 100 resolved primary observations | none |
| EARLY | 100–499 | descriptive only |
| INTERIM | 500–1,999 | spreads + CIs, no graduation |
| DECISION-CAPABLE | ≥ 2,000 resolved **and** ≥ 250 multi-candidate sets **and** ≥ 100 resolved in each family considered for policy **and** ≥ 6 calendar months of primary live-scan span | one of the four labels below |

Prefer 9–12 months and more than one regime before graduation.

---

## 8. Allowed final labels (only when DECISION-CAPABLE)

Exactly one per feature:

`GRADUATE_RANK_FEATURE` | `EXTEND_FORWARD_VALIDATION` | `KEEP_RESEARCH_ONLY` | `RETIRE`

`GRADUATE_RANK_FEATURE` does **not** change production. It only permits designing FEATURE-003 (controlled paper-ranking). FEATURE-003 is not started here.

Until DECISION-CAPABLE:

**`FORWARD VALIDATION ACTIVE — INSUFFICIENT NEW DATA`**

---

## 9. Failure conditions (when mature)

Fail (cannot graduate) if after adequate sample/time:

- rank correlation ≈ 0
- top−bottom spread unstable
- production rank equal or better
- benefit only in one family or one month/regime
- feature mainly duplicates production rank
- improvement smaller than likely execution noise

---

## 10. Regime and sector

Record as **context** (`regime_label`, `sector`, `sector_map_version`).  
Do **not** multiply ranks or gate on them. Any regime-conditioned ranker is a later hypothesis.

---

## 11. Safety

If shadow logging throws or is slow:

- production scan, rank, Ready, autopilot, Telegram, GTT continue
- error is logged at debug/warning
- research never blocks trading

Threshold / Stage-2∧RS trade-or-not studies are **prohibited** as primary FEATURE-002 tests.
