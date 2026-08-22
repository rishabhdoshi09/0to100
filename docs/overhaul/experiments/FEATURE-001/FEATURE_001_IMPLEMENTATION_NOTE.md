# FEATURE-001 — Implementation Note (Step 0 audit)

**Experiment:** FEATURE-001 — Trend & Relative-Strength Feature Graduation  
**Status of this note:** written from a repository audit **before** study results.  
**Core SEPA:** `CORE_SEPA_STATUS = RETIRED_RESEARCH_BENCHMARK` (`research/sepa/status.py`).  
**Claim class of any later result:** `EXPLANATORY` on already-consumed 2019–2026 history. Not `VALIDATED_EDGE`.

This note records **where Trend and RS already participate**, what FEATURE-001 will measure, and what it will not touch. It is not a results paper.

---

## 0. Boundary (locked)

Do **not**:

- resurrect Core SEPA F
- retune VCP, buy-zone, or RS ≥ 70 against SEPA history
- add a new SEPA rule or market-regime gate
- change broker / GTT / live autopilot behaviour
- promote anything to paper or live
- label any finding from this history `VALIDATED_EDGE`

SEPA-001 → 001R → 001R2/R2.1 → 003 remain an **immutable research benchmark**. FEATURE-001 asks a different question: *conditional on an existing QuantTerm family firing, do Stage-2/trend quality and `rs_cs_v1` improve outcomes as ranking / quality / risk-context / loss-avoidance features?*

The answer is not assumed to be yes.

---

## 1. Production decision path (repository truth)

### 1.1 Conductor

`app.py` starts daemons and the Streamlit shell. Live scan orchestration is `scan/auto_scan.py` (background brain). Cards and tickets flow through `ui/scanner.py`, `ui/command_center.py`, `product/trade_desk.py` (Ready), and `product/recommendations_workspace.py` (Ideas / Top Stocks).

Execution remains `execution/trade_executor.py` → Kite + GTT OCO. FEATURE-001 does not call this path and does not change it.

### 1.2 Signal families (`scan/unified_scanner.py` `SIGNAL_META`)

These are the production families. FEATURE-001 uses this table, not an illustrative list.

| Key | User label | Category | Base weight |
|---|---|---|---|
| `BREAKOUT_52W` | 52-week high breakout | Breakout | 30 |
| `BREAKOUT_RES` | Resistance break on volume | Breakout | 26 |
| `GOLDEN_CROSS` | Golden cross (50/200 SMA) | Breakout | 22 |
| `VOL_SQUEEZE` | Squeeze breakout | Breakout | 22 |
| `VCP` | VCP — tightening base | Pattern | 28 |
| `FLAT_BASE` | Flat base near breakout | Pattern | 24 |
| `CUP_HANDLE` | Cup & handle | Pattern | 24 |
| `HIGH_TIGHT_FLAG` | High tight flag | Pattern | 30 |
| `ASC_TRIANGLE` | Ascending triangle | Pattern | 24 |
| `DOUBLE_BOTTOM` | Double bottom | Pattern | 22 |
| `PRE_BREAKOUT` | Breakout ke kareeb | PreBreakout | 26 |
| `ACCUMULATION` | Smart-money accumulation | PreBreakout | 24 |
| `DELIVERY_SPIKE` | Delivery buying rising | PreBreakout | 18 |
| `NR7_COIL` | Coiled — tightest day in 7 | PreBreakout | 14 |
| `POCKET_PIVOT` | Pocket pivot volume | PreBreakout | 20 |
| `MOMENTUM` | Strong momentum | Momentum | 20 |
| `PULLBACK_SUPPORT` | Uptrend pullback to support | Pullback | 26 |

`scan/short_scanner.py` is a **separate** bearish detector (paper-first). It is out of FEATURE-001 scope (long-side families only).

`scan/setup_engine.py` archetypes (`VCP_BREAKOUT`, `ACCUMULATION_BREAKOUT`, …) feed `scan/ranking_engine.py` playbooks. They are **not** the same object as `SIGNAL_META` and **not** `research.sepa` Core F. FEATURE-001 attributes **`SIGNAL_META` fires** from `UnifiedScanner._analyze`, because that is the production detector and the walk-forward backtest (`scan/signal_backtest.py`) already uses it.

### 1.3 Scoring, ranking, BUY / WATCH

`UnifiedScanner._analyze` builds:

```
base = Σ SIGNAL_META[s][2] × calib[s] × regime_calib[s]
score = min(100, base + trend_bonus + mom_score × 0.2)
trend_bonus = 10 if close > SMA200 else 0
```

BUY if `score ≥ 55` and ≥ 2 signals, **or** `BREAKOUT_52W` / `HIGH_TIGHT_FLAG`. Extension / falling-knife / RSI / CLV guards **demote** to WATCH (`chase_risk`). They are quality gates, not Core SEPA.

Calibration: `scan/signal_backtest.py` walk-forward → `logs/signal_backtest.json`; `scan/live_edge.py` blends live `signal_log` conservatively (≥30 outcomes). `scan/ev_engine.py` is the north-star EV ranker on live outcomes.

**Conviction:** `scan/conviction.py` enriches a **top-N** slice with news + earnings. That path is **not PIT-safe** for 2019–2026. FEATURE-001’s “existing ranking” is the **technical `score`** (and `mom_score` / `momentum_5d`). News/Yahoo/restated fundamentals are excluded from the historical panel.

**Ready / Ideas:**

- Ready (`product/trade_desk.py`) ranks last-scan names with ATQ + cached Ideas overlay. Copy currently says “Stage-2 is Minervini SEPA…”.
- Ideas / Best Setups (`product/sepa_setup.py` `rank_best_setups`) scores a **7-rule point template** (100 pts). This is **not** `research.sepa` 8/8 AND-gate + VCP + buy-zone.

**Autopilot eligibility:** live autopilot reads scanner BUY + portfolio/risk rails. It does **not** require Core F. FEATURE-001 must not add a SEPA or Trend/RS hard gate here.

---

## 2. Where Trend currently participates

| Surface | What it is | Class |
|---|---|---|
| `UnifiedScanner` `above_sma200` → `+10` score | Soft rank bonus | **Ranking (production)** |
| `StockSignal.above_sma50` / `above_sma200` | Flags for breadth / conviction | **Ranking / context** |
| `breakout_conviction` trend-stage +7.5/+7.5 | Soft points if above SMA50/200 | **Ranking (production)** |
| `product/sepa_setup.py` 7-rule point score | Ideas / Best Setups / monitor | **UI + Ideas ranking** |
| `product/monitor_context.py` Weinstein/Minervini stage 1–4 | Context on the same SMA stack | **UI context** |
| `research/sepa/trend.py` `evaluate_trend` | Strict 8-rule AND-gate (7 structure + RS) | **Research-only** |
| Core F (`research/sepa/engine.py`) | Structure ∧ RS≥70 ∧ VCP ∧ buy-zone | **Research-only; RETIRED as strategy** |

Production does **not** implement the strict 7-rule structure AND-gate as a BUY licence. It uses a **single SMA200 boolean** plus a **different** 7-rule point scorer on Ideas.

FEATURE-001 will expose the **strict template as a feature vector**, not as a new gate.

---

## 3. Where RS currently participates

| Surface | Methodology | Class |
|---|---|---|
| `research/sepa/rs.py` `rs_cs_v1` | `0.40·r63 + 0.20·r126 + 0.20·r189 + 0.20·r252`; cross-sectional percentile on the PIT investable set | **Research-only canonical** — FEATURE-001 **reuses this unchanged** |
| `research/sepa003/fastrs.py` `FastRS` | Same formula, vectorized as-of | **Research-only** |
| `scan/relative_strength.py` | Nifty / sector excess; Kite or **yfinance** | **Live overlay — not PIT-safe**; forbidden on the FEATURE-001 panel |
| Ideas `monitor_context` RS vs Nifty | 63-session benchmark excess on official history | **UI context**, not `rs_cs_v1` |
| `screener/vcp_scanner.py` / `ui/vcp_page.py` RS | Separate 1M/3M/6M vs Nifty50 mix | **UI / screener**, not `rs_cs_v1` |
| `breakout_conviction` `rs_outperf` | Soft points for Nifty excess | **Ranking (production)**, different series |

FEATURE-001 primary RS object is **`rs_cs_v1`**. The live yfinance RS path must not enter the historical panel. Canonical RS ≥ 70 remains a **descriptive flag**, not a retuned cutoff.

---

## 4. Hard gating vs execution vs research

| Mechanism | Hard gate? | FEATURE-001 action |
|---|---|---|
| Core F eligibility | Research-only AND-gate | Freeze. Mark retired benchmark. Do not call from production. |
| VCP / buy-zone in `research.sepa` | Research-only | Freeze. No rescue. |
| Scanner BUY score + guards | Production hard/soft gates | Unchanged. |
| Telegram buttons | Paper-only | Unchanged. |
| GTT OCO / broker | Execution | Unchanged. |
| Autopilot | Production | Unchanged. No Core F plan. |
| Lab keep/skip (`signal_backtest.trading_playbook`) | Demote-only on scan | Unchanged. |

---

## 5. Research / evidence infrastructure

| Piece | Role in FEATURE-001 |
|---|---|
| `scan/signal_backtest.py` | Production walk-forward: `sample_step=5`, **last ~250 sessions only**, horizon 10, `_simulate_timed` + costs. Default lookback is too short for 2020–2026 stability. FEATURE-001 uses the **same detector and simulator**, on a **full-history** session grid (documented sampling — not a production change). |
| `core/signal_outcome_tracker.py` / `signal_log` | Live tracked outcomes. Too short / too recent for this study. |
| `research/harness.py` | DSR / PSR / Reality Check / BH-FDR / block-bootstrap CI. FEATURE-001 uses FDR + block-bootstrap CIs for inferential claims. |
| `research/sepa/universe_pit.py` | `load_store_frames`, `FastInvestable`, PIT membership. Reused. |
| `gauntlet/` | Strategy battery. Not a FEATURE-001 promotion path. |
| `core/evidence_levels.py` | E0–E6. This study stays **explanatory** (consumed history). |
| `core/decision_journal.py` | Live decision log. Not the FEATURE-001 panel. |
| Feature registry | **None exists.** FEATURE-001 introduces `trend_features_v1` / `rs_features_v1` as the first versioned feature manifests. |

Production vs research parity for **detection**: `research/sepa/scanner_research.py` already states the research scanner **is** `UnifiedScanner._analyze`. FEATURE-001 keeps that invariant.

---

## 6. Misleading production / UI semantics (Step 2 plan)

These strings can be read as “approved SEPA trade eligibility.” Field **keys** stay (`sepa_score`, `sepa_verdict`) so persistence does not break. **Headlines and desk copy** become Trend Quality / research-context language.

| Location | Current implication | Planned wording |
|---|---|---|
| `product/sepa_setup.py` `_verdict` | `STRONG — MEETS SEPA` | `STRONG — TREND QUALITY INTACT` (research qualify, not Core F) |
| same, WEAK advice | “does not meet Minervini's SEPA criteria” | Trend Quality not intact; not a Core SEPA licence |
| same, disclaimer | “SEPA here is Mark Minervini's published Stage-2…” | Trend Quality 7-rule template; **not Core SEPA**, not a buy |
| `product/sepa_setup.py` Best Setups note | “ranked on Minervini's 7-rule Stage-2 template” | 7-rule Stage-2 / Trend Quality template; research, not a buy |
| `product/trade_desk.py` Ready disclaimer | “Stage-2 is Minervini SEPA” | Trend Quality (7-rule template, research context, not Core SEPA) |
| Ready empty-next | “Ideas SEPA” | Ideas Trend Quality |
| `product/top_stocks.py` `tape_policy` | “Minervini SEPA / Trend Template” | Trend Quality template + stage/RS context |
| `product/recommendations_workspace.py` load_note | “Minervini's 7-rule template” | 7-rule Trend Quality template |
| `ui/vcp_page.py` | VCP page titled as Minervini licence | Add research-context line; do not redesign the page |

Not in scope for this milestone: renaming `sepa_*` JSON keys, deleting `research/sepa`, rewriting playbooks, or a UI redesign.

---

## 7. Canonical feature versions (Steps 3–4)

### 7.1 `trend_features_v1`

Retain the **strict** Trend Template arithmetic (`research/sepa/trend.py` / `SepaConfig` SMA lengths, 21-session SMA200 slope, 252-session 52-week window, 30% off-low / 25% near-high structure rules). Expose **atomic** flags and continuous distances. **Do not collapse to one Stage-2 bit.**

Persisted at minimum:

- booleans: `price_gt_sma50`, `price_gt_sma150`, `price_gt_sma200`, `sma50_gt_sma150`, `sma50_gt_sma200`, `sma150_gt_sma200`, `sma200_rising`
- `dist_above_52w_low_pct`, `dist_from_52w_high_pct`
- `structure_pass` (original 7-rule AND-gate, RS **excluded**)
- `n_structure_passed` (0–7)
- continuous: `pct_above_sma50/150/200`, `sma50/150/200_slope_pct`, `ma_spread_50_200_pct`

Descriptive buckets (prespecified, not optimized):

- `strict` = `structure_pass`
- `near` = not strict and `n_structure_passed ≥ 5`
- `non` = `n_structure_passed < 5`

### 7.2 `rs_features_v1`

Wrap `rs_cs_v1` only. No formula change.

- `rs_percentile`, `rs_score`
- `r63`, `r126`, `r189`, `r252` (3m / 6m / 9m / 12m session proxies)
- `rs_ge_70` descriptive flag
- `rs_pct_chg_21` = percentile(t) − percentile(t−21) when both tables exist
- Nifty-relative 63d return **only** if official index history is present (`data/index_store.py`); never yfinance

Prespecified RS buckets: `<50`, `50–69`, `70–79`, `80–89`, `90–94`, `95–99`.

---

## 8. Study design (Steps 5–16) — locked before results

### 8.1 Panel

- Official NSE bhavcopy via `load_store_frames` (CA applied on read, store stays raw).
- PIT investable snapshot per sample date (`FastInvestable`: min price 20, min turnover ₹50L, min sessions 260).
- Detector: `UnifiedScanner._analyze` on history `≤ as_of` (last 280 bars — covers SMA200 slope + 52w).
- Sample grid: every **5th session** on a shared calendar (same cadence as production backtest `sample_step=5`). Full-span dates, not the production “last 250 sessions” window. This is FEATURE-001 sampling, not a live change.
- Outcome: `_simulate_timed` + CNC costs, **horizon 20** (so +1R / +2R / MAE / MFE can realize). Research horizon; production nightly default remains 10.
- One **event** row per (symbol, date, fire). Explode to **family rows** by each `SIGNAL_META` key that fired. Family baselines are never blended before being reported.

### 8.2 Forbidden inputs

No news, no Yahoo, no restated fundamentals, no `scan/relative_strength.py` live RS, no future bars in features.

### 8.3 Primary hypotheses (also `feature_001_hypotheses.json`)

- **H1** Trend strength improves **breakout-family** outcomes.
- **H2** RS improves **breakout / momentum** family outcomes.
- **H3** Trend/RS reduce adverse-selection tails (`RISK_FILTER_VALUE`, not automatically ALPHA).
- **H4** Trend and RS add information beyond existing momentum scoring (`mom_score`, `momentum_5d`).
- **H5** The value of Trend/RS differs materially by strategy family.

Inferential claims use BH-FDR (`q = 0.10`) and block-bootstrap CIs. Everything else is exploratory. Test-count is recorded in the hypothesis registry after the run.

### 8.4 Classification vocabulary (per family, Trend and RS separately)

Exactly one of: `POSITIVE_RANK_FEATURE` | `RISK_FILTER_VALUE` | `REDUNDANT` | `NEGATIVE` | `UNSTABLE` | `INSUFFICIENT_DATA`.

Rules are prespecified in `research/feature001/analyze.py` (`classify_family_feature`). Evidence decides the cell — the example table in the program brief is **not** a target.

### 8.5 Final feature status (exactly one each for Trend and RS)

`FORWARD-VALIDATE AS RANK FEATURE` | `FORWARD-VALIDATE AS RISK FILTER` | `KEEP RESEARCH-ONLY` | `RETIRE`

No paper, no live, no `VALIDATED_EDGE`.

---

## 9. Forward observation (Step 17)

Passive **shadow feature logging** for dates **strictly after** the experiment freeze. If wiring into `app.py` / `auto_scan` could affect runtime reliability, **do not activate** in this milestone — specify the ledger only (`FEATURE_001_FORWARD_LEDGER.md` / `research/feature001/forward_spec.py`).

---

## 10. Deprecation map (preview; full text in `FEATURE_001_SEPA_DEPRECATION.md`)

**Keep:** `trend_features_v1` arithmetic, `rs_cs_v1`, SEPA-001…003 docs/results, generic structural helpers.

**Research-only:** Core F eligibility, VCP hard-gate, pivot/buy-zone experiments.

**Deprecate from production semantics:** “SEPA Ready / MEETS SEPA” as a money licence; any future autopilot plan that requires Core F.

**Do not delete** SEPA code in this milestone.

---

## 11. Deliverables

Under `docs/overhaul/experiments/FEATURE-001/`:

- this note
- `FEATURE_001_BASELINES.md`
- `FEATURE_001_TREND_STUDY.md`
- `FEATURE_001_RS_STUDY.md`
- `FEATURE_001_STRATEGY_INTERACTIONS.md`
- `FEATURE_001_RANKING_STUDY.md`
- `FEATURE_001_RISK_FILTER_STUDY.md`
- `FEATURE_001_TEMPORAL_STABILITY.md`
- `FEATURE_001_SEPA_DEPRECATION.md`
- `FEATURE_001_RESULTS.md`
- `FEATURE_001_DECISION.md`
- machine-readable: feature dataset, per-strategy results, hypothesis registry, feature manifest, ranking output

Production trading behaviour must remain unchanged after this milestone.
