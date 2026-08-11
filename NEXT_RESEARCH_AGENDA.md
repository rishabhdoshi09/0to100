# QuantTerm — Next Research Agenda

> **Agenda only.** No experiments executed. No Phase B. Production unchanged.  
> Global dataset trust remains **`OPERATIONAL_ONLY`**.  
> Scientific work may use the **scoped certified 29-name panel**
> (snapshot `a7a9828ec37e09e4`) where noted.

---

## Plain-English summary

QuantTerm finished a hard round of scientific tests. Four big ideas failed.
A fifth idea looked promising once, then **failed when checked on different
dates** — and even flipped direction. So that idea is closed too.

We are **not** going to keep twisting the same failed ideas (more networks,
more AI models, more momentum tweaks).

Instead, this agenda asks a simpler question:

> Given only the historical data we can actually trust, what *different*
> economic stories are worth testing next?

**Recommended next work (at most three, not started yet):**

1. **Short-term bounce-back** — stocks that fell hard recently sometimes recover.
2. **Quieter stocks** — lower-volatility names may deliver better risk-adjusted results.
3. **Calm-before-move as risk context** — when volatility shrinks, does the *shape*
   of future outcomes change (risk), not necessarily the average direction (alpha)?

---

## 1. Settled evidence / closed branches

| Branch | IDs | Final | Status |
|--------|-----|-------|--------|
| Dynamic market structure | EXP-A5-01 / `81b8889792f53113` | FAIL | **REJECT — closed** |
| Standalone portfolio-network incrementality | EXP-A6-01 / `590571a11ee06fc2` | FAIL | **REJECT — closed** |
| Multi-horizon CS momentum family (5/10/22/66d) | EXP-A2-01 / `775b4a0fce7d5b83` | FAIL | **REJECT — closed** |
| Simple logistic challenger | EXP-A3-01 / `7842a46ee335685a` | FAIL | **REJECT — closed** |
| `signal_x_network_concentration` discovery | EXP-A5A6-01 / `3734b8a0a9124a60` | PASS_RISK (discovery only) | superseded by confirmation |
| Independent confirmation of that interaction | EXP-A6-CONF-01 / `e6092c83f98fba20` | **FAILED_CONFIRMATION** | **REJECT — closed** |

### Critical negative lesson (preserved in scientific memory)

> An apparent FDR-surviving interaction was discovered on one sample but
> reversed sign on independent confirmation and had no incremental economic
> value.
>
> Discovery δcorr = **+0.36** → confirmation δcorr = **−0.23**.  
> No incrementality vs pairwise/sector controls. No economic risk value.  
> **Branch closed.** Do not reopen via threshold fishing, centrality swaps,
> or “similar” network interactions.

### Permanently closed *for now* (do not propose)

- more network interactions / centrality / graph variants  
- different clustering methods  
- more momentum horizons or momentum rescue  
- RF / GBM / XGBoost / deep learning / ensembles / RL  
- “combine everything” / unrestricted indicator mining  

Reopen only with **materially new external evidence** or a **fundamentally
different** preregistered hypothesis — not a cosmetic rename.

---

## 2. Trustworthy data inventory

**Global:** `OPERATIONAL_ONLY` (unresolved CA outside the panel; gauntlet not earned).  
**Scoped scientific surface:** 29-name panel, snapshot `a7a9828ec37e09e4`,
approx. **2023-08-23 → 2026-08-11**, CA-verified, identity-verified.

| Input | Class | Notes |
|-------|--------|--------|
| OHLC (adjusted bhav) | **RESEARCH_READY** | Scoped 29-name panel only |
| Volume | **RESEARCH_READY** | Same scoped bars |
| Delivery % | **PARTIAL** | In bhav files; **not** scoped-certified |
| Index (Nifty etc.) | **PARTIAL** | Present (~360 sessions); thinner than equity; not A.5-certified as primary benchmark |
| India VIX | **PARTIAL** | Present; not scoped-certified for A.5 |
| Corporate actions | **PARTIAL** | **Panel verified**; global still unresolved (ABFRL, ETFs, …) |
| Fixed-universe identity (29) | **RESEARCH_READY** | ISIN security_ids verified |
| Full PIT universe membership | **PARTIAL** | Ledger exists; not required for fixed panel; dual-ISIN incomplete |
| PIT sector history | **MISSING** | Static map only |
| Fundamentals / valuations | **MISSING** | No PIT valuation ledger |
| Earnings / results history | **OPERATIONAL_ONLY** | Live/heuristic — not PIT research archive |
| Announcements | **OPERATIONAL_ONLY** | Live news path |
| FII/DII | **OPERATIONAL_ONLY** | Short cache; not multi-year certified |
| Futures history | **OPERATIONAL_ONLY** | Live/membership — no research panel |
| Options EOD history | **OPERATIONAL_ONLY** | Live chains; durable EOD store missing here |
| Ownership / shareholding | **OPERATIONAL_ONLY** | As-of-now scrapes — not PIT |

### Research gaps (block entire families)

| Gap | Blocks |
|-----|--------|
| No PIT fundamentals | value / quality / profitability / investment factors |
| No PIT earnings archive | PEAD, surprise, results-gap studies |
| No PIT sector history | sector-neutral designs that need historical sector membership |
| Global CA unresolved | any claim on the full 2,800+ universe as RESEARCH_GRADE |
| Thin / uncertified index+VIX for A.5 scope | market-beta / VIX-state tests need a **fresh scoped cert** before promotion-grade claims |
| Mega-cap-only 29 panel | weak power for liquidity premia; survivorship/large-cap bias in all panel tests |

---

## 3. Candidate economic hypotheses (data-filtered)

Only ideas the **current RESEARCH_READY panel** can support honestly.
Each row answers the quality filter (mechanism, persistence, data, baseline,
falsifier, distinctness, multiplicity, sample, costs).

### H1 — Short-horizon cross-sectional reversal (ALPHA)

**User-facing:**  
“Stocks that fall much more than their peers over a few days sometimes bounce
back. We want to test whether that bounce is real after trading costs.”

| Filter | Answer |
|--------|--------|
| Mechanism | Overreaction / temporary liquidity demand; mean reversion after short shocks |
| Why persist? | Continual flow-driven overshoot in liquid names; costly to arb overnight risk |
| Data | Adjusted OHLC on certified 29 panel — **READY** |
| Baseline | Flat / equal-weight panel; **and** explicit contrast to *rejected* 60d momentum |
| Falsifier | Cost-aware OOS mean R ≤ 0 after FDR across preregistered short horizons; or indistinguishable from noise |
| Distinct from failed work? | **Yes** — economically opposite of EXP-A2-01 momentum family; different horizons (e.g. 1–5d formation → 5–10d hold), not “another 60d rank” |
| Multiplicity | Small fixed horizon set (≤4); BH-FDR; no post-hoc horizon crowning |
| Sample | ~700 sessions × 29 ≈ adequate for CS ranks; OOS still thin — gate on n_eff |
| Costs | **High sensitivity** (short horizon, high turnover) — CNC round-trip mandatory |

**Class:** ALPHA  

---

### H2 — Low-volatility (low-risk) effect within panel (ALPHA / risk-adjusted)

**User-facing:**  
“Quieter stocks sometimes give a smoother ride. We want to test whether
lower-volatility names in this group actually produce better results after
adjusting for risk and costs.”

| Filter | Answer |
|--------|--------|
| Mechanism | Leverage aversion / lottery preference → underpricing of low-vol names |
| Why persist? | Institutional constraints; behavioural preference for volatile “story” stocks |
| Data | OHLC → realized vol — **READY** |
| Baseline | Equal-weight panel; high-vol quintile; raw return *and* Sharpe/vol-scaled |
| Falsifier | Low-vol minus high-vol cost-aware OOS edge ≤ 0; or only mechanical vol scaling with no economic content |
| Distinct? | **Yes** — not momentum, not network, not structure clusters |
| Multiplicity | One primary sort (trailing realized vol); optional 1 secondary horizon preregistered |
| Sample | Adequate on 29 names; mega-caps may compress the effect |
| Costs | Medium (lower turnover than 1d reversal if monthly rebalance) |

**Class:** ALPHA (primary metric = risk-adjusted / cost-aware long-short)  

---

### H3 — Realized-volatility compression → forward *distribution* (RISK)

**User-facing:**  
“When a stock’s price swings settle down, the *next* period’s outcomes may
change shape — bigger gaps, worse tails, or calmer continuation. We want to
test that as a **risk** clue, not as a buy signal.”

| Filter | Answer |
|--------|--------|
| Mechanism | Compression precedes expansion of risk (regime shift in variance / tails) |
| Why persist? | Structural underreaction to quiet regimes; options/hedging frictions (even if we lack options history) |
| Data | OHLC realized vol — **READY** |
| Baseline | Unconditional return distribution; high-vol state as control |
| Falsifier | No material change in downside probability / left-tail / MAE after compression vs controls |
| Distinct? | **Yes** — RISK/MARKET STATE, not stock-picking alpha; not network concentration |
| Multiplicity | One primary definition of compression (e.g. 10d vol / 60d vol < frozen threshold from **train split only**) |
| Sample | Adequate event counts on 29×~700 |
| Costs | N/A for pure risk diagnostic; if later used as filter, measure opportunity cost separately |

**Class:** RISK / MARKET STATE — **not** a BUY/SELL signal  

---

### H4 — Abnormal volume / turnover state (ALPHA candidate — lower priority)

**User-facing:**  
“Unusual trading activity might mean something is changing. We want to know
if that helps predict the next move after accounting for the recent price trend.”

| Filter | Answer |
|--------|--------|
| Mechanism | Information arrival / attention |
| Data | Volume **READY** on panel |
| Distinct? | Partially — must **control for** short-term return so it is not stealth momentum/reversal |
| Concern | All 29 names are mega-liquid → effect may be weak / null; high false-hope risk |
| Priority | **Back-burner** unless H1/H2 need a volume control |

**Class:** ALPHA (conditional)  

---

### H5 — Panel-internal breadth / dispersion (RISK / MARKET STATE — lower priority)

**User-facing:**  
“When stocks in the group move together or scatter apart, future portfolio
risk might change.”

| Filter | Answer |
|--------|--------|
| Mechanism | Correlation regime / opportunity set |
| Data | OHLC cross-section **READY**; full-market breadth **not** PIT-certified |
| Distinct? | Yes from failed *network concentration interaction*; must not rebuild graph/centrality |
| Concern | Easy to reinvent failed A.5/A.6 structure/network stories — keep **simple** (e.g. fraction up, cross-sectional std) |
| Priority | Optional RISK diagnostic after H1–H3 |

**Class:** MARKET STATE / RISK  

---

### Explicitly deferred (data not ready)

| Idea | Why deferred |
|------|----------------|
| PEAD / earnings surprise | Earnings history **OPERATIONAL_ONLY** / not PIT |
| Value / quality / profitability | Fundamentals **MISSING** |
| Sector-neutral factors needing PIT sectors | Sector history **MISSING** |
| FII/DII timing | **OPERATIONAL_ONLY** short cache |
| Options-implied / futures basis | History **OPERATIONAL_ONLY** / missing EOD store |
| Full-NSE dynamic universe alpha | Global trust **OPERATIONAL_ONLY** |

---

## 4. Reasons to reject weak candidates *now*

| Weak idea | Why rejected |
|-----------|--------------|
| “Try more indicators / AI patterns” | Not an economic hypothesis; multiplicity explosion |
| Network / centrality / cluster variants | **Closed** by A.5+A.6; sign flip on confirmation |
| More momentum horizons / nonlinear momentum | **Closed** by EXP-A2-01 + EXP-A3-01 |
| RF/GBM/XGBoost/DL rescue | Escalation without mechanism; forbidden |
| Residual 60d momentum vs market | Too close to rejected CS momentum — exclude unless formation ≤5d *and* preregistered as reversal |
| Delivery-spike alpha | Delivery only **PARTIAL** — certify first |
| Anything needing PIT fundamentals/earnings | Data gap |

---

## 5. Ranked shortlist

| HYPOTHESIS | ECONOMIC RATIONALE | DATA READINESS | DISTINCT FROM FAILED WORK? | EXPECTED SAMPLE SIZE | COST SENSITIVITY | IMPLEMENTATION COMPLEXITY | SCIENTIFIC VALUE | PRIORITY |
|------------|-------------------|----------------|----------------------------|----------------------|------------------|---------------------------|------------------|----------|
| H1 Short-horizon reversal | Overreaction / liquidity | RESEARCH_READY (panel OHLC) | Yes (≠ momentum) | Medium–High CS observations | **High** | Low | High | **1** |
| H2 Low-volatility effect | Leverage/lottery preference | RESEARCH_READY | Yes | Medium | Medium | Low | High | **2** |
| H3 Vol-compression risk | Variance regime shift | RESEARCH_READY | Yes (RISK not alpha) | Medium events | Low (diagnostic) | Low–Med | High (product risk) | **3** |
| H4 Abnormal volume | Attention/info | RESEARCH_READY volume | Partial | Medium | High | Low | Medium | 4 (defer) |
| H5 Panel breadth/dispersion | Correlation regime | RESEARCH_READY CS | Yes if kept simple | Medium | Low | Low | Medium | 5 (defer) |

---

## 6. Recommended next experiments (≤3) — **not executed**

### Recommendation set

| # | Working ID | Hypothesis | Class | Priority |
|---|------------|------------|-------|----------|
| 1 | **EXP-NEXT-01** | Short-horizon CS reversal | ALPHA | Run first |
| 2 | **EXP-NEXT-02** | Low-volatility premium within panel | ALPHA | Second |
| 3 | **EXP-NEXT-03** | Vol-compression → downside distribution | RISK | Third (parallel-ok after 01) |

If capacity allows only **one**: choose **EXP-NEXT-01**.

---

### EXP-NEXT-01 — Short-horizon cross-sectional reversal

**User-facing:**  
“Stocks that fall much more than the group over a few days sometimes bounce
back. We will test whether that effect survives trading costs on verified history.”

| Item | Spec (preregistration outline) |
|------|--------------------------------|
| **Hypothesis** | Cross-sectional short-horizon losers outperform short-horizon winners over a preregistered short holding window after costs. |
| **Null** | No positive cost-aware OOS edge in the preregistered reversal family after multiple-testing control. |
| **Universe** | Fixed preregistered 29-name panel (same as Phase A.5 freeze) |
| **Snapshot** | `a7a9828ec37e09e4` (or successor scoped cert with identical panel rules) |
| **Formation** | Preregistered set only, e.g. `{1d, 3d, 5d}` return rank (losers = bottom quintile) |
| **Hold / horizon** | Preregistered set only, e.g. `{5d, 10d}` — **no** 22d/66d momentum reuse |
| **Inputs** | Adjusted close only |
| **Portfolio** | Long bottom 20% / short top 20% by formation return; equal weight; close→close |
| **Benchmark** | (i) zero; (ii) equal-weight panel; (iii) **explicit non-equivalence check** vs EXP-A2-01 60d momentum |
| **Costs** | CNC `round_trip_cost_pct`; conservative turnover assumption |
| **Primary metric** | Deflated Sharpe / harness verdict on OOS net R; mean net R |
| **Secondary** | n, n_eff, CI, profit factor, max DD, cost drag |
| **Falsification** | All preregistered pairs have mean_net ≤ 0 **or** no FDR survivor with DSR gate |
| **Multiple testing** | BH-FDR across \|formation\|×\|hold\| cells; n_trials = that count in DSR |
| **Train/OOS** | Freeze `oos_start` date *before* run (do not reuse A.5 discovery OOS blindly without stating independence) |
| **Data certification required** | Scoped panel READY (already); no delivery/index required |
| **Not allowed** | Expanding horizons after seeing results; ML challengers |

---

### EXP-NEXT-02 — Low-volatility effect within panel

**User-facing:**  
“Quieter stocks in this group may produce a better risk-adjusted result. We
will test that with verified prices and realistic costs.”

| Item | Spec |
|------|------|
| **Hypothesis** | Names in the lowest trailing realized-volatility quintile deliver higher cost-aware risk-adjusted OOS performance than the highest-vol quintile. |
| **Null** | No positive low-minus-high vol edge after costs; any raw return gap is explained by mechanical volatility. |
| **Universe / snapshot** | Same certified 29 panel / `a7a9828ec37e09e4` |
| **Vol definition** | Trailing 20d (primary) realized stdev of daily returns — frozen |
| **Rebalance** | Every 21 sessions (preregistered) |
| **Hold** | 21d forward |
| **Portfolio** | Long low-vol quintile / short high-vol quintile, equal weight |
| **Primary metric** | OOS mean net R **and** Sharpe of long-short; harness DSR |
| **Benchmark** | Equal-weight panel; vol-scaled EW control |
| **Costs** | CNC round-trip |
| **Falsification** | mean_net ≤ 0 and Sharpe not FDR/DSR-clear; or edge vanishes vs vol-scaled control |
| **Multiple testing** | Single primary; optional secondary 60d vol definition only if preregistered (n_trials=2) |
| **Data certification** | Scoped OHLC READY |
| **Class** | ALPHA (risk-adjusted) — do **not** auto-convert into a Brain veto |

---

### EXP-NEXT-03 — Volatility compression as risk context

**User-facing:**  
“When price swings go quiet, the next stretch of outcomes may get riskier in
the tails. We will test that as a **warning**, not as a buy tip.”

| Item | Spec |
|------|------|
| **Hypothesis** | After realized-vol compression, forward downside risk (loss rate / left tail / MAE proxy) is materially worse than in non-compressed states, for otherwise similar names. |
| **Null** | No material worsening of preregistered downside metrics after compression. |
| **Universe / snapshot** | Same certified 29 panel |
| **Compression** | e.g. `vol_10 / vol_60 < τ` with **τ frozen from training partition only** (never fit on final OOS) |
| **Outcome** | 10d forward return distribution metrics (not mean return as success) |
| **Primary metric** | Loss-rate gap and/or 5th-percentile gap (compression vs not), preregistered thresholds for “material” |
| **Controls** | Trailing return level; trailing vol level (so compression ≠ “just high vol”) |
| **Costs** | Reporting-only unless a later *separate* policy experiment tests filtering |
| **Falsification** | Downside gaps fail materiality **or** disappear after controls |
| **Multiple testing** | Single primary definition; no family of τ |
| **Class** | **RISK** — CONFIRMED ≠ production veto; needs separate policy experiment later |
| **Data certification** | Scoped OHLC READY |

---

## 7. Data required for each (checklist)

| Experiment | OHLC panel | Volume | Index/VIX | CA/identity | New cert needed? |
|------------|------------|--------|-----------|-------------|------------------|
| EXP-NEXT-01 | Yes | No | No | Yes (existing scoped) | No (reuse `a7a9828ec37e09e4`) |
| EXP-NEXT-02 | Yes | No | No | Yes | No |
| EXP-NEXT-03 | Yes | No | No | Yes | No |

Optional later upgrade: protocol-scoped **index/VIX** certification if market-beta
neutralization is added — **not** required for the three outlines above.

---

## 8. Preregistration outline (process)

Before any code run for EXP-NEXT-0k:

1. Write `PHASE_A7_FROZEN_PROTOCOLS.md` (or per-experiment freeze JSON) with the
   tables above locked.  
2. Register hypothesis IDs in the isolated research DB **before** evaluation.  
3. Bind snapshot_id + panel + costs + FDR rules.  
4. Run once on certified snapshot via PIT contracts.  
5. Record PASS / FAIL / INCONCLUSIVE — no retuning.  
6. Independent confirmation plan ready *before* celebrating any PASS.

**Do not implement in this milestone.**

---

## 9. Areas explicitly NOT worth researching now

1. Network / graph / centrality / community interactions  
2. Dynamic clustering vs sectors as alpha  
3. Momentum horizon fishing or ML challengers on momentum features  
4. Full-universe RESEARCH_GRADE claims without CA remediation  
5. Fundamental / earnings / FII / options strategies without PIT ledgers  
6. Delivery-based alpha before delivery series certification  
7. “Residual momentum” that is 60d momentum in disguise  
8. Policy filters built on **unconfirmed** risk stories  

---

## 10. Scientific-memory updates

Recorded negative belief (Phase A.5/A.6 memory):

- FDR-surviving `signal_x_network_concentration` **reversed** on independent
  confirmation and lacked incremental economic value — **do not rediscover
  trivial variants**.

Closed REJECT freezes remain for EXP-A5/A6/A2/A3.

---

## 11. Production / Phase B confirmation

| Flag | Value |
|------|-------|
| Production behaviour changed | **No** |
| Phase B started | **No** |
| Experiments implemented | **No** |
| Global trust | **OPERATIONAL_ONLY** |

---

## Final ask for review

Approve **EXP-NEXT-01** (and optionally 02/03) for formal freeze + preregistration
in a later milestone. Until then: **STOP**.

_Agenda generated after Phase A.5 scientific rerun + Phase A.6
FAILED_CONFIRMATION. Snapshot reference: `a7a9828ec37e09e4`._
