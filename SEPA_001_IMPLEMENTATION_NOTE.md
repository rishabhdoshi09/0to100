# SEPA-001 — Implementation Note

**Experiment:** Canonical SEPA eligibility & research validation  
**Status:** Research architecture only — no live execution, broker, GTT, or autopilot wiring  
**Authority:** SEPA audit (PARTIAL / 47) — system is SEPA-*inspired*, not end-to-end SEPA  
**Eligibility version:** `sepa-001.v1`

This note exists so implementation does not invent a second source of truth
and does not promote a filename (`vcp`, `sepa`, `setup`) as if it were the
methodology.

---

## 1. Existing reusable components

| File | Function / class | Current purpose | PIT status | Production / research | Reuse decision | Reason |
|------|------------------|-----------------|------------|------------------------|----------------|--------|
| `product/sepa_setup.py` | `score_sepa`, `minervini_trend_template_7` | 7-rule **weighted** Stage-2 quiz (100 pts). MIXED at 40/100. Missing history = 0, not a pass. | Frame is as-of if caller slices; live Ideas uses last bar of bhavcopy | Research overlay (Ideas / Ready). **Not** a money-path gate | **Reuse SMA/52w arithmetic; do not reuse scoring or floors** | Correct MA math on official OHLCV. Wrong as SEPA: scored not AND-gated; 25% off 52w low (strict wants 30%); no RS-70 rule |
| `product/sepa_setup.py` | `rank_best_setups` | Ranks scan shortlist by SEPA score, 45s budget, min_score 40 | As-of of loaded frames | Ideas Best Setups only | **Do not reuse for eligibility** | Ranking + MIXED floor is UI, not qualification |
| `product/trade_desk.py` | `SEPA_READY_FLOOR = 40` | MIXED names can land on Ready | n/a | Ready overlay | **Do not touch in SEPA-001** | Live/UI path. Strict engine is separate |
| `product/monitor_context.py` | `rs_vs_benchmark` | 63-session return minus Nifty 50; LEADER if +5pp | PIT-OK on sliced frames | Overlay context | **Retain as diagnostic, never as RS rank** | Benchmark-relative excess ≠ cross-sectional percentile |
| `product/monitor_context.py` | `classify_stage` | Weinstein 1–4 from price vs 50/200 | PIT-OK on sliced frames | Overlay | Optional diagnostic on eligibility object | Not a substitute for 8/8 template |
| `scan/unified_scanner.py` | `_analyze`, `grade_breakout` | Production scanner: 16 signals, chase_risk, CLV, ATR stop/target | Walk-forward in `signal_backtest` (`hist = df.iloc[:t]`) | **Money path** (scan → Ready/autopilot) | **Baseline A only.** Do not change production BUY | SEPA-001 must not alter live BUY behaviour |
| `scan/unified_scanner.py` | `_detect_patterns`, `_pullback_depths` | “VCP” = 120 bars split into ~30-bar **calendar windows**, monotone shrinking, last ≤12%, price ≥95% of 40d high. **No volume dry-up. No swings.** | As-of of passed frame | Production VCP *label* | **Do not reuse as VCP** | Range-coil, not nested contractions. False positives documented in audit |
| `scan/unified_scanner.py` | pivot / entry | Pivot = first pattern pivot else 41-day high. **`entry = pivot if pivot > price else last price`** | As-of | Production | **Forbidden in SEPA engine** | Through-pivot becomes a market chase. Audit P0 |
| `scan/unified_scanner.py` | `chase_risk` | +10%/5d and >10% above EMA20, or >20% above SMA50 → BUY→WATCH | As-of | Production demote | **Do not copy thresholds** | Extension guard for scanner, not a pivot buy-zone |
| `scan/unified_scanner.py` | `_atr` | 14-period mean TR | As-of | Production | **Reuse as sanity metric only** | Must not overwrite structural stop |
| `screener/vcp_scanner.py` | `_check_vcp`, swing high/low | Swing *values* (not indexed legs). Pullback < prior×0.75, need 2 shrinks. yfinance primary | Not PIT; Yahoo | Streamlit VCP page — **off Trade-desk path** | **Do not promote.** Borrow *idea* of swings, rewrite with dated indices | Swing-low matching uses `l < h_prev` on prices, not time; yfinance not allowed as research source |
| `scan/setup_engine.py` | `_check_vcp` | 15-bar segments, 0.9 shrink, last >15% reject, vol 0.85, pivot 20d high, dist >8% reject; stop = 20d low | Fetch via `_fetch` (not bhav as-of) | ScanPipeline / Streamlit JARVIS — **not** `market_scan_service` | **Do not promote.** Stop-from-20d-low is a hint, not canonical | Calendar chunks again; unreachable from desk |
| `scan/pipeline.py` | `ScanPipeline` | 8-stage institutional scan | Mixed | Streamlit, not React desk | **Out of scope** | Off money path |
| `scan/signal_backtest.py` | `run_backtest`, `_simulate_timed` | Walk-forward scanner accuracy; fill when price reaches entry; costs via `core.costs` | `iloc[:t]` — good | Research + Lab playbook | **Reuse simulation + cost convention** | Do not reuse ATR 2:1 geometry as SEPA R/R |
| `risk/position_sizer.py` | `size_equity_trade` | 1% capital / (entry−stop), 10% name cap | n/a | Production | **Not in SEPA-001** | Sizing is execution; engine only reports stop distance |
| `data/bhavcopy_store.py` / `data/bhavcopy_runtime.py` | `get_ohlcv` | Official NSE EOD; CA applied **on read** if events exist | Caller must slice | Primary history | **Canonical price source** | Slice `index <= as_of`. Never invent bars |
| `data/corporate_actions.py` | `load_events`, `adjust_frame` | Back-adjust OHLC ÷ factor, vol ×; missing file → `{}` (raw, unadjusted) | Events dated by `ex_date` | On-read in `get_ohlcv` | **Reuse; label degraded when file absent** | No fake CA. `logs/ca_events.json` currently **absent** in this environment |
| `data/nse_universe.py` | `point_in_time_universe` | Membership from `logs/universe_history.json`; no file → today’s survivors + `survivorship_complete=False` | Honest fallback | Live scan uses **today’s** list | **Required for RS and ablation** | Never label survivor fallback as PIT-safe |
| `data/index_store.py` | `get_index_ohlcv` | Nifty / VIX / sectors | As-of if sliced | Regime / gauntlet | Diagnostic RS vs Nifty only | Not the RS percentile universe |
| `research/harness.py` | `evaluate`, PSR/DSR/FDR/Reality Check | Anti-overfitting gate on R-streams | n/a | Research | **Reuse on ablation R-streams** | No new statistics |
| `gauntlet/registry.py` | `register` | Experiment id + git/dataset/config hash | n/a | Gauntlet | Stamp SEPA-001 runs | Optional; ablation writes its own fingerprint |
| `gauntlet/ledger.py` | `TradeRecord` | Immutable trade row | n/a | Gauntlet | Optional emission | Ablation can stay JSONL without forcing gauntlet abort gates |
| `research/intelligence/data/pit_contract.py` | `PitContract` | Facade over snapshot / universe / CA | Explicit NOT_PIT_SAFE | Research A1 | **Optional.** SEPA-001 reads bhavcopy + PIT universe directly so ablation does not require a Snapshot bind | Contract forbids silent survivor success — same policy |
| `research/momentum_breakout/` | EXP-006 sim | Next-bar open, gap-through-stop at open | PIT | Research | **Reuse entry convention for ablation fills** | Signal known at close; trade next open |
| `scan/conviction.py` | Yahoo `earningsQuarterlyGrowth` | ±10 conviction | **Not PIT** (restated `.info`) | Top-40 spice | **Do not use in SEPA-001** | Fundamentals deferred |
| `core/costs.py` | `cost_in_r` | CNC round-trip + slippage in R | n/a | Backtest + live R | **Subtract from ablation R** | Honest net expectancy |

---

## 2. Existing problems (audit-backed)

1. **Fragmented SEPA logic** — 7-rule scorer (Ideas), calendar VCP (production scanner), swing VCP (Streamlit screener), segment VCP (SetupEngine), Ready floor 40. No single `eligible` answer.
2. **UI-only logic** — `SEPA_READY_FLOOR`, Stage-2 overlay, Ideas ranking. Autopilot never calls `score_sepa`.
3. **Unreachable setup logic** — `screener/vcp_scanner.py`, `scan/setup_engine.py` / `ScanPipeline` are not on `market_scan_service` → Trade desk.
4. **Duplicate VCP** — three detectors; production one is shrinking **windows**, not swings.
5. **Non-canonical pivots** — 40-day high / 41-day high / 20-day high / last price.
6. **Scoring where gating is required** — 40/100 MIXED can be “Ready”; template is not 8/8 AND.
7. **Benchmark RS vs cross-sectional RS** — +5pp vs Nifty ≠ RS percentile ≥ 70 vs as-of universe.
8. **Last price after pivot** — production `entry = last price` once through the pivot.
9. **Look-ahead / survivorship** — daily scan uses today’s EQ list; CA table optional (raw gaps); Yahoo fundamentals restated; SEPA overlay is not what `signal_backtest` measures.

---

## 3. Proposed architecture

```
Universe (PIT membership as-of, or explicit degraded survivors)
  → OHLCV through as_of_date only (bhavcopy, CA on-read)
  → Strict Trend Template (8 independent rules, fail-closed on missing history)
  → Cross-sectional RS percentile (as-of universe, versioned formula)
  → Structural VCP (swing legs + volume dry-up measurements)
  → Canonical pivot (from VCP resistance) or None
  → Buy-zone validity (configurable extension; never snap to last price)
  → Structural stop (final contraction low); ATR is sanity only
  → Reward context (measured move or UNKNOWN — never fake 2:1)
  → SepaEligibility (eligible / rejection codes / evidence)
```

**Layers the object must keep distinct**

| Layer | Pass means | Fail example |
|-------|------------|--------------|
| Good stock | 8/8 template + RS ≥ 70 | Stage 1, or RS 55 |
| Good setup | Structural VCP + pivot exists | Calendar coil, or no swings |
| Good entry | Price inside versioned buy-zone | ₹535 vs pivot ₹500 |
| Trade eligibility | All of the above + stop defined and not excessively wide | Excellent stock, invalid entry → `eligible=False` |

A stock may be exceptional and still return **NO TRADE — INVALID ENTRY**.

---

## 4. Strict Trend Template (`trend_template_v1`)

AND of eight rules. Missing input → that rule is **not a pass** (`passed=None` / fail-closed for `trend_template_pass`).

| # | Rule | Threshold | Notes vs legacy scorer |
|---|------|-----------|------------------------|
| 1 | Close > SMA150 and SMA200 | strict `>` | Legacy combined 20 pts |
| 2 | SMA150 > SMA200 | strict `>` | Legacy 10 pts |
| 3 | SMA200 rising ~1 month | SMA200(t) > SMA200(t−21) | Same 21-session proxy; Minervini is discretionary “about a month” |
| 4 | SMA50 > SMA150 and SMA200 | strict `>` | Legacy 20 pts |
| 5 | Close > SMA50 | strict `>` | Legacy 15 pts |
| 6 | Close ≥ 30% above 52-week low | **30%** (not 25%) | Legacy scorer stays 25% for UI compatibility |
| 7 | Close within 25% of 52-week high | 25% | Same |
| 8 | Cross-sectional RS percentile ≥ 70 | 70 | **New.** Not in the 7 |

`near_sepa` / watch: 7/8 with RS still computed — **never equal to strict eligibility**.

Legacy `score_sepa` is **not deleted**.

---

## 5. Cross-sectional RS (`rs_cs_v1`)

**Not** IBD proprietary. Documented research formula:

Let \(r_n\) = total return over the last \(n\) sessions (close[t]/close[t−n] − 1), using only bars ≤ as_of.

\[
s = 0.40\,r_{63} + 0.20\,r_{126} + 0.20\,r_{189} + 0.20\,r_{252}
\]

(heavier on ~3 months; 63/126/189/252 ≈ 3/6/9/12 months of sessions.)

- Universe = PIT membership as-of that date when `survivorship_complete` and `research_grade`.
- Else: rank among symbols with a bhav bar on as_of, set `universe_complete=False`, `pit_safe=False`.
- A name needs all four horizons or RS is **unavailable** (fail-closed for rule 8).
- Percentile = percent of *valid* universe scores strictly below this name (0–100).
- Threshold: `>= 70`.
- Retain `rs_vs_benchmark` on the object as `benchmark_rs` (diagnostic).

---

## 6. Structural VCP (`vcp_swing_v1`)

- Swings from **high/low** with a 4-bar fractal (left=right=4). Dated indices.
- A contraction = swing high → later swing low (pullback) → recovery swing high.
- Minimum **2** contractions; cap recorded at 6.
- Depths generally decrease: each later depth ≤ prior × **1.15** (noise), and **final < first × 0.75**.
- Final depth ≤ **12%**.
- Base depth (pattern high−low)/high ≤ **35%** (reject loose bases).
- Price in the last 5 bars ≥ **92%** of pivot (tightening toward resistance).
- Last close above the last contraction low (no breakdown).
- Volume: mean volume of final contraction vs first; dry-up ratio = final/first. **Required** `dry_up_ratio ≤ 0.90` for `vcp_pass`. Expanding volume → fail with `VOLUME_EXPANDING` (measurements still reported).
- Calendar-window coils **without** ≥2 swing contractions → reject.

This is **not** `_pullback_depths`.

---

## 7. Pivot, buy-zone, stop, R/R

**Pivot:** highest swing high in the accepted contraction sequence (resistance the pattern is coiling under). If VCP fails, `pivot=None` — do not fall back to a 41-day high.

**Buy-zone (`buy_zone_v1` defaults, research parameters, not claimed optimal):**

- Lower: `pivot × (1 − 0.25%)` (tiny coil under the level still “at the pivot”)
- Upper: `pivot × (1 + 1.5%)`
- `price > upper` → `ENTRY_EXTENDED` / **NO TRADE — INVALID ENTRY**
- `price < lower` → `ENTRY_BELOW_PIVOT` (setup may be good; trade not yet)
- Never set `entry = last price` because it is through the pivot

Ablation will also test upper bounds: 0.25, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0%.

**Stop:** last contraction swing low. If missing → no trade.  
If `(entry − stop)/entry > 8%` **or** stop distance > **3×ATR** → `WIDE_STRUCTURAL_STOP` (reject; do not tighten).  
ATR is reported, never the stop.

**Reward:** measured move = first contraction depth × pivot, projected up from pivot. If first depth missing → `reward_r = null` / `UNKNOWN`. No 4×ATR target.

---

## 8. PIT policy

| Input | Policy |
|-------|--------|
| Price | `frame[frame.index <= as_of]` only |
| SMAs | Computed on that slice (no future bars in the window) |
| 52-week | Last 252 **available** sessions in the slice |
| Universe | `point_in_time_universe(as_of)`; degraded if incomplete |
| CA | On-read adjust; `ca_complete=False` if `load_events()` empty |
| RS | Rank only names in the as-of universe with valid scores |
| Fundamentals / news | **Not used** (Yahoo `.info` is not PIT) |

`eligible` is a technical verdict. `pit_safe` is true only when universe membership is complete **and** a CA table is loaded. Ablation must not advertise survivor-biased runs as PIT-safe.

---

## 9. Integration boundary

```text
research.sepa.evaluate_sepa_eligibility(symbol, as_of_date, ...) -> SepaEligibility
```

Callable from research, backtests, later scanner/UI/autopilot.

**SEPA-001 wires research/backtest only.** No changes to `execution/autopilot.py`, `trade_executor`, GTT, or production BUY gates.

---

## 10. Ablation plan

Walk-forward on official bhavcopy, `hist = df.iloc[:t]`, next-bar open fill (EXP-006 convention), CNC costs, structural stop as invalidation, horizon 20 sessions, mark-to-market if neither stop nor +2R.

| ID | Definition |
|----|------------|
| A | Production scanner (`UnifiedScanner._analyze`) — any signal with valid entry/stop (current research baseline) |
| B | A **and** strict 8/8 template |
| C | B **and** RS ≥ 70 |
| D | C **and** structural VCP |
| E | D **and** buy-zone valid |
| F | Core SEPA only: 8/8 + RS + VCP + pivot + buy-zone + structural stop (**not** requiring scanner BUY) |

Plus: buy-zone width sweep; RS 70/80/90; template 7/8 vs 8/8 (research only).

Metrics: trade count, trades/year, expectancy R (net), total/median R, avg win/loss, win rate, PF, max DD (R and %), Sharpe/Sortino when n allows, payoff, hold days, MAE/MFE, failed-break rate, % +1R / +2R, % stop before +1R, regime/sector/year if available. Harness `evaluate()` on each R-stream. Small n → INCONCLUSIVE, never a fake promote.

---

## 11. What this milestone will not do

Broker, live autopilot, order routing, GTT, pyramiding, sell-into-strength, climax exits, AI/NLP catalysts, full fundamental SEPA, portfolio concentration redesign, production BUY changes, UI redesign.

---

## 12. Implementation sequence

1. This note (no engine yet).
2. `research/sepa/` types + config + trend + RS + VCP + entry + `evaluate_sepa_eligibility`.
3. Tests (synthetic + PIT invariance).
4. Ablation runner + replay of NSE examples.
5. `SEPA_001_RESULTS.md` + architecture recommendation A/B/C/D from evidence.
