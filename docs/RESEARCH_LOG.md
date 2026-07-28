# QuantTerm — Research Log

The project's **lab notebook**. Every experiment — including blocked and negative
ones — gets a one-page, append-only record here. This is deliberately *separate
from code*: it is the immutable scientific history that prevents hindsight bias
("I think we already tried that…"), enforces pre-registration, and makes the
evidence trail auditable by an outside committee.

**Append-only.** Never edit or delete a past entry. A superseded conclusion is
corrected by a *new* entry that references the old one. Git is the immutability
guarantee.

---

## Standing research principles (the operating contract)

1. **Evidence is allowed to kill the project.** A gauntlet or paper failure
   against the pre-registered criteria defaults to *rejecting the hypothesis* —
   not tweaking a parameter, indicator, stop, or universe. A new hypothesis may
   begin only after a written post-mortem stating why the last one failed and
   what genuinely new information (not a new knob) justifies the next experiment.
2. **Negative evidence is as valuable as positive evidence.** A clean, documented
   rejection is progress — it rules out a hypothesis and saves time and capital.
   Success is measured by *reducing uncertainty*, not by producing a profitable
   backtest.
3. **Pre-register before running.** Hypothesis, success criteria, datasets, and
   stopping rules are fixed *before* execution and frozen (`gauntlet/freeze.py`).
   A knob that moves mid-experiment voids the run.
4. **No claim without evidence.** Evidence Levels rise only through the objective
   gated workflow (`core/evidence_levels.py`); belief does not promote.
5. **Default action is to run experiments, not write code.** Infrastructure is
   built only when a completed experiment proves it scientifically necessary.

---

## Entry template

```
## EXP-NNN — <title>
- Hypothesis:
- Null hypothesis:
- Pre-registered success criteria:
- Pre-registered failure criteria:
- Required datasets:
- Execution date:
- Status: PASS / FAIL / INCONCLUSIVE / BLOCKED
- Reason for outcome:
- Evidence generated:
- Evidence Level change:
- Decision made:
- Next action:
- Supersedes / references:
```

---

## EXP-001 — Historical Gauntlet (first run attempt)

- **Hypothesis:** The QuantTerm signal set possesses a statistically significant,
  durable, benchmark-beating edge after realistic costs.
- **Null hypothesis:** Net-of-cost expectancy ≤ 0, or all return is explained by
  market beta (α ≤ 0).
- **Pre-registered success criteria (→ E3 PASS):** harness `PROMOTE` **and**
  FDR-significant — Deflated Sharpe > 0.95 (deflated across #strategies),
  block-bootstrap CI lower > 0, `beats_benchmark` (α>0, p<0.05), profit factor
  ≥ 1.3, net expectancy ≥ +0.15R, positive in ≥2 regimes, White's Reality-Check
  p < 0.05 for the best strategy.
- **Pre-registered failure criteria:** non-positive net expectancy, OR α not
  significant (long-beta), OR fails the Reality Check → **FAIL**. Positive but
  underpowered / correlation-fragile → **INCONCLUSIVE** (no claim).
- **Required datasets:** CA-adjusted bhav price history; point-in-time
  survivorship-complete universe (+ delisted EOD); Nifty 50 + India VIX history;
  `logs/ca_events.json`; `logs/universe_history.json`. Window must span a real
  drawdown.
- **Execution date:** 2026-07-26 (attempted).
- **Status:** **BLOCKED — External Data Dependency.**
- **Reason for outcome:** The required NSE datasets are inaccessible in the
  execution environment — `nsearchives.nseindia.com` and `www.nseindia.com` are
  policy-denied at the network gateway (403 CONNECT rejected); no local CA or
  universe-history files exist. Fabricating data is forbidden (invariant #1).
  `python -m gauntlet` correctly **ABORTED** (exit 2) at the dataset-validation
  gate — every required check failed except symbol-mismatch.
- **Evidence generated:** None about the market. About the system: the
  dataset-validation abort gate is confirmed working end-to-end against a real
  empty environment — the harness cannot be induced to produce a verdict from
  absent data.
- **Evidence Level change:** **None.** Strategy alpha remains **E0** (unproven).
- **Decision made:** No engineering work authorized. No workaround, synthetic
  data, or alternate source. Experiment paused, not failed.
- **Next action:** Acquire the four datasets in a network-enabled environment (or
  supply the files into `logs/`), confirm `core.data_integrity.verify_ca_adjustment().passed`,
  then rerun the pre-registered experiment exactly as specified.
- **Supersedes / references:** —

## EXP-002 — Historical Gauntlet (EXPLORATORY, biased data)

- **Hypothesis:** Same as EXP-001 — the QuantTerm signal set has a significant,
  durable, benchmark-beating edge after costs.
- **Pre-registered success criteria:** as EXP-001 (harness PROMOTE + FDR).
- **Execution date:** 2026-07-27. Experiment id `614670961b1e`.
- **Data:** ~765 sessions / 2886 symbols of NSE bhav (real) + index/VIX. Run with
  `--skip-validation`: **CURRENT-universe (survivorship-BIASED) and NOT
  CA-adjusted**. This is an OPTIMISTIC data set — the biases inflate results.
- **Status:** **FAIL.** 5054 trades, 17 signals → **PASS 0 · FAIL 14 ·
  INCONCLUSIVE 3.**
- **Reason for outcome:** Nearly every signal is NEGATIVE expectancy after costs
  (VCP −0.66R, FLAT_BASE −0.55R, NR7_COIL −0.45R, ASC_TRIANGLE −0.32R, PRE_BREAKOUT
  −0.16R, DOUBLE_BOTTOM −0.13R, MOMENTUM/POCKET_PIVOT/CUP_HANDLE all slightly neg).
  Every signal has NEGATIVE alpha vs Nifty. White's Reality-Check p = 0.62 → the
  BEST of 17 is indistinguishable from luck. The 3 INCONCLUSIVE are two tiny-sample
  (n=13, n=15) and one breakeven (DELIVERY_SPIKE +0.007R, n=295). No FDR survivors.
- **Evidence generated:** The current signal set, as implemented, has **no
  tradeable edge after costs** — and this held on FAVOURABLE (biased) data, so
  clean CA-adjusted / survivorship-complete data would be WORSE, not better. A FAIL
  here is a strong, cheap kill.
- **Evidence Level change:** **None — stays E0.** (A biased-data FAIL cannot lower
  a level that is already at the floor, and cannot raise anything.)
- **Decision made:** Do NOT trade real capital on the current signal set. Do NOT
  loosen gates to manufacture a pass (that is the overfitting trap). Autopilot stays
  conservative/paper.
- **Next action:** Post-mortem (below), then EITHER pre-register a genuinely NEW
  hypothesis (different exits / holding period / a single most-promising signal) OR
  accept that a pure technical-breakout system on NSE after costs lacks edge and
  pivot. The clean-data run is now LOW priority — an optimistic FAIL rarely flips.
- **Post-mortem notes:** (a) Costs are decisive for the near-breakeven signals —
  several are ~0 gross but −0.03 to −0.10R net. (b) US-origin chart patterns (VCP,
  CUP_HANDLE, FLAT_BASE, HIGH_TIGHT_FLAG) transfer poorly / are mis-parameterised on
  NSE. (c) Fills are modeled AT the pivot (optimistic) — real fills make this worse.
  (d) Exit logic (2×ATR stop + breakeven trail + fixed target) is a candidate to
  re-examine as a NEW hypothesis, not a tweak to force EXP-002 to pass.
- **Supersedes / references:** extends EXP-001 (which was BLOCKED); this is the
  first run that actually produced trades and a verdict.

## EXP-003 — Cross-Sectional Momentum factor (PRE-REGISTERED)

- **Hypothesis:** A low-turnover cross-sectional momentum strategy has POSITIVE
  net-of-cost expectancy AND positive alpha vs Nifty on NSE — because (a) monthly
  rebalancing amortises the trading costs that sank EXP-002's short-term signals,
  and (b) momentum is a globally-replicated anomaly (strong prior → less likely to
  be data-mining than a bespoke pattern).
- **Null hypothesis:** net monthly return ≤ 0, OR alpha vs Nifty ≤ 0 (i.e. it's
  just beta / no skill over the index).
- **Pre-registered rule (ONE config — NO sweep):** universe = liquid names (median
  daily turnover ≥ ₹5 cr); rank by 12-1 momentum (return from ~12 months ago to ~1
  month ago, i.e. lookback 252d, skip 21d); buy the top 20 equal-weight; rebalance
  monthly (21 trading days); 0.32% round-trip cost charged on the fraction of the
  book that turns over. Implemented in `scan/momentum.py`, judged by the SAME
  harness battery (alpha, block-bootstrap CI, Deflated Sharpe, regimes).
- **Success criteria (→ investigate further):** harness PROMOTE — mean monthly > 0
  with block-CI lower > 0, `beats_benchmark` True (alpha>0, p<0.05), and the
  strategy's Sharpe/CAGR beats Nifty's.
- **Failure criteria:** non-positive mean or no alpha → FAIL (momentum adds nothing
  over the index after costs). Positive-but-underpowered (few months of data) →
  INCONCLUSIVE, meaning "need a longer history", NOT a pass.
- **Data note:** first pass runs on the same ~3-year (~36-month) window as EXP-002,
  still survivorship-BIASED (current universe). So a PASS is not proof (bias
  inflates); a FAIL is a strong kill. With only ~36 monthly points the honest
  likely outcome is INCONCLUSIVE.
- **Status:** **RUN 2026-07-27 → INCONCLUSIVE (underpowered), point estimate
  NEGATIVE.** Experiment id `379b22314843`.
- **Result:** Only **14 monthly rebalances** (below the 30 floor → no statistical
  claim). But the evidence we have is clearly bad, not neutral: mean monthly
  **−1.61%**, **alpha vs Nifty −1.77%/mo**, beta 1.46, block-CI [−5.18, +3.01]%
  (includes 0). Strategy CAGR **−21.4%** / Sharpe −0.61 / maxDD 41% vs **Nifty
  CAGR +0.09% / Sharpe +0.08 / maxDD 15%**. Regimes: BULL +1.2% (n=4), CHOP −6.4%
  (n=7), BEAR +5.8% (n=3) — the choppy stretch wrecked it. `beats_benchmark`:
  False.
- **Reading:** The verdict is INCONCLUSIVE only because 14 months is too few for a
  claim — momentum is a multi-YEAR anomaly with occasional crash periods, and this
  ~3-year window (survivorship-biased) is both short and momentum-hostile (high-beta
  midcap winners crashed in the chop). So this is NOT evidence momentum works; it is
  an UNDERPOWERED test whose point estimate is decidedly negative.
- **Evidence Level:** stays **E0**.
- **Decision:** Do NOT trade this. Do NOT sweep parameters to find a passing variant
  (anti-fishing commitment holds). The real limitation surfaced is **history length**
  — 14–36 monthly points cannot test a factor. The disciplined fix is MORE DATA
  (5–10+ years), NOT a different config. Combined signal from EXP-002 + EXP-003:
  nothing tested so far beats buy-and-hold Nifty after costs on this window.
- **Next action (two honest paths):** (A) extend the bhav/index history to 5–10
  years and RE-RUN the SAME EXP-003 config — a fair, powered test of momentum; or
  (B) accept the converging evidence and treat low-cost index / factor-ETF exposure
  as the benchmark that active strategies have not yet beaten.
- **Supersedes / references:** motivated by EXP-002's post-mortem (cost drag +
  negative alpha on high-turnover short-term signals).

### EXP-003 RE-RUN (powered) — 2026-07-27, after fixing the index-history depth bug

- **What changed:** the first run was capped at 14 months because
  `build_index_store` ignored a deeper `days` request (fixed, commit e300a1e).
  Re-ran the IDENTICAL config on ~9 years of index history + ~7 years of bhav →
  **66 monthly rebalances.** Experiment id `467beaf4835e`.
- **Result — the sign FLIPPED positive** (the 14-month window had been a momentum-
  crash stretch): mean monthly **+1.66%**, **alpha vs Nifty +0.31%/mo**, positive in
  ALL regimes (BULL +1.93, BEAR +2.17, CHOP +1.14). Strategy **CAGR 17.2% vs Nifty
  11.8%** — momentum beat the index on absolute return. Deflated Sharpe 0.95,
  p=0.0525.
- **Status:** **INCONCLUSIVE — promising, NOT proven.** Three honest brakes: (1) the
  alpha is NOT significant — correlation-aware CI [−0.6, +4.0]%/mo includes zero,
  `beats_benchmark` False, p just misses 0.05; (2) **risk-adjusted it LOSES to
  Nifty** — Sharpe 0.70 vs 0.97, because vol is 28% vs 12% and max drawdown **44.6%
  vs 12.3%**; you took ~2.3× the risk for ~1.45× the return; (3) still
  survivorship-biased (inflates).
- **Reading:** Unlike EXP-002 (dead) and the 14-month crash window (−21%), 5.5 years
  of momentum shows a REAL, regime-consistent positive edge that beats the index on
  return. But the RAW strategy is not clearly better than just holding Nifty once
  risk is accounted for — the weakness is excess volatility / a 44% drawdown, not the
  signal itself.
- **Evidence Level:** still **E0** (promising ≠ proven; alpha not significant, data
  biased).
- **Decision / next (two legitimate, non-fishing paths):** (A) **EXP-004 — risk-
  managed momentum**: a genuinely NEW hypothesis that targets the specific weakness
  the data exposed (the 44% drawdown / 28% vol) — e.g. a trend filter (hold only
  while Nifty > 200-DMA, else cash) or volatility scaling. NOT a parameter tweak of
  EXP-003; a new pre-registration. (B) acquire survivorship-free + CA-adjusted data
  to PROVE the current momentum, now that it shows promise and the data work is
  finally worth it.

## EXP-004 — Risk-Managed Momentum (200-DMA trend filter) (PRE-REGISTERED)

- **Hypothesis:** Adding a market trend filter to EXP-003's momentum — hold the
  top-20 book only while Nifty is AT/ABOVE its 200-day SMA, else sit in CASH — cuts
  the drawdown and RAISES the risk-adjusted return (Sharpe) above Nifty's, WITHOUT
  needing the raw alpha to be larger. Directly targets the specific weakness EXP-003
  exposed (44.6% drawdown, 28% vol, Sharpe 0.70 < Nifty 0.97), not its parameters.
- **Why this is a NEW hypothesis, not a tweak:** it adds a distinct mechanism (a
  regime/trend overlay) motivated by a measured weakness, and it is judged on a
  DIFFERENT success metric (risk-adjusted return / drawdown), not on nudging
  EXP-003's alpha across a significance line.
- **Null hypothesis:** the filter does NOT improve risk-adjusted return — Sharpe ≤
  EXP-003's and ≤ Nifty's, or drawdown not materially reduced.
- **Pre-registered rule (ONE config — NO sweep):** everything in EXP-003 (12-1,
  monthly, top-20, liquid, 0.32% cost) PLUS: at each rebalance, if Nifty close <
  its 200-day SMA → hold cash (0% that month, conservative — real cash would earn
  ~0.5%/mo), else hold the momentum book. Implemented via
  `scan.momentum.trend_gate_from` + `build_momentum_series(trend_gate=…)`;
  `python -m gauntlet.momentum --trend`.
- **Success criteria:** Sharpe > Nifty's (0.97) AND max drawdown materially below
  EXP-003's 44.6% (target ≤ ~25%), with mean return still clearly positive — i.e.
  the risk-managed version is a better risk/reward than simply holding the index.
  A harness PROMOTE (significant positive alpha) is a bonus, not required for this
  hypothesis (which is about risk-adjusted improvement).
- **Failure criteria:** Sharpe not above Nifty's, or drawdown not materially cut →
  the filter adds nothing; momentum's risk problem stands.
- **Data note:** same ~9-year index / ~7-year bhav history, still survivorship-
  biased. A good result is promising, not proof.
- **Status:** **RUN 2026-07-27 → FAIL (per pre-registered criteria).** Experiment id
  `f1640867338e`. 76 monthly rebalances, 71.1% of months invested / 28.9% in cash.
- **Result — the filter helped on RISK, but not enough to beat the index:**
  - Directional wins vs EXP-003: max DD **44.6% → 32.7%**, vol **28% → 22%**, beta
    **1.35 → 0.58**, alpha **+0.31% → +0.71%/mo**, Sharpe **0.70 → 0.83**. The trend
    overlay did exactly what it should.
  - But the **pre-registered success bar was NOT met:** strategy Sharpe 0.83 is
    still BELOW Nifty's (Nifty Sharpe over this window = **1.33**), and drawdown
    32.7% is well above the ~25% target and 2.6× Nifty's 12.3%. Same CAGR (17.2 vs
    17.3) at ~1.7× the vol.
  - Harness verdict FAIL: alpha +0.71%/mo is economically meaningful but NOT
    statistically significant (t-test p=0.17, block-CI includes 0), so it does not
    clear the benchmark gate.
- **Reading:** The hypothesis "a trend filter cuts the risk" was CONFIRMED
  directionally — but even risk-managed, momentum does not beat a Nifty that has run
  at a Sharpe of ~1.3 over this post-2020 window. Note the window is strongly
  benchmark-FAVOURABLE: Nifty's long-run Sharpe is nearer 0.7-0.9, so a 1.33 here is
  an exceptionally high bar.
- **Evidence Level:** still **E0**.
- **Converging conclusion across EXP-002/003/004:** on this ~6-9 year (survivorship-
  biased) NSE data, NOTHING tested — breakout signals, raw momentum, or risk-managed
  momentum — beats low-cost Nifty buy-and-hold on a RISK-ADJUSTED basis after costs.
  The strategies keep matching Nifty's return but with more risk. That is strong
  (though window-limited) evidence for the passive conclusion.
- **Decision:** Do NOT trade these. **Do NOT sweep the 200-DMA window** or try
  top-30 / dual-momentum / etc. to manufacture a pass — four experiments in, that
  would be textbook fishing, and the anti-fishing commitment holds. The honest,
  money-relevant read is: for this period, a low-cost Nifty index fund/ETF is the
  best risk/reward found, and no active variant has beaten it.
- **Legitimate remaining moves (NOT more tweaks):** (i) test on a LONGER / different
  window (pre-2020, or 15+ years) with clean survivorship-free data — the current
  window is index-favourable and biased, so the conclusion is provisional; (ii)
  otherwise, accept the passive conclusion.
- **Supersedes / references:** builds on EXP-003's powered re-run.

## EXP-005 — Momentum on ~15y yfinance data → "PASS" but REJECTED AS INVALID

- **Setup:** same 12-1 momentum config, `--source yf --years 15` (162 monthly
  rebalances, ~14.6y). Experiment id `14fecc41e59f`.
- **Headline (looks spectacular):** Verdict **PASS** — CAGR **37.4%** vs Nifty
  11.6%, Sharpe **1.48** vs 0.80, alpha **+1.70%/mo**, Deflated Sharpe 1.0,
  p≈1e-7, block-CI excludes zero, beats_benchmark True.
- **VERDICT: INVALID — this PASS is a survivorship-bias artifact, NOT an edge.**
  Rejected on multiple independent grounds:
  1. **Severe survivorship bias.** The universe is the CURRENT Nifty-500, and yf
     only returns the names that survived ~15 years — a pre-selected winners club.
     The 44% drawdowns and blow-ups (delisted names) that a real 2010-2025 momentum
     book would have taken are simply ABSENT from the data. The bias inflates the
     result — exactly as pre-registered ("a longer-window PASS is still not proof").
  2. **Broken/partial download.** 88 tickers failed — including large, clearly-NOT-
     delisted names (AXISBANK, SIEMENS, MRF, NTPC, TITAN, IOC, GRASIM, BAJAJ-AUTO…)
     via `OperationalError('unable to open database file')` (yfinance SQLite cache
     under threads). So only 236 of 500 names loaded — a noisy, non-representative
     subset of the survivors. (threads=False fix applied; but even a complete
     survivor set is still biased.)
  3. **Broken regime split.** 51 of 162 months bucket to a `nan` regime — the
     index regime series doesn't cover the older yf dates, so the regime evidence
     is unusable.
- **What it actually PROVES (the real lesson):** put EXP-003 (clean 7y bhav:
  momentum ≈ Nifty, no clear edge) next to EXP-005 (biased 15y yf: momentum
  "returns 37%/yr"). The enormous gap between them IS the survivorship bias, shown
  live. The stronger the "edge" on biased data, the more the bias — not the alpha.
- **Evidence Level:** stays **E0.** A contaminated PASS raises nothing.
- **Decision:** REJECT this result; do NOT trade on it; do NOT let a 37%-CAGR
  number tempt a real allocation. Free data cannot answer this question:
  bhav is clean but too short (7y); yfinance is long but survivorship-biased. A
  genuine 15-year verdict needs **survivorship-free (delisted-inclusive), CA-
  adjusted data — a paid vendor.** Absent that, the honest standing conclusion is
  EXP-004's: on clean data nothing has beaten low-cost Nifty risk-adjusted.
- **Supersedes / references:** long-window robustness attempt on EXP-003's config;
  demonstrates why survivorship-free data is non-negotiable for a real claim.

## META-001 — Architectural discovery: per-trade R cannot back a portfolio-alpha claim

- **Type:** methodology correction (append-only; erases nothing above).
- **Date:** 2026-07-27. Context: start of the Evidence Lab overhaul (`overhaul/evidence-lab`).
- **Discovery:** EXP-002…005 derived CAGR/Sharpe/drawdown from a stream of
  INDEPENDENTLY-compounded per-trade R-multiples (`gauntlet/runner.py` modelled a fixed
  1%-risk-per-trade account). That is not a portfolio: it ignores cash constraints,
  overlapping positions, concurrency limits, turnover and idle capital. No CAGR, Sharpe,
  alpha or benchmark-outperformance claim may rest on independently compounded trades.
- **Which prior results remain informative (NOT erased):**
  - EXP-002 (breakout signals net-negative after costs) — the sign and the cost lesson
    stand; per-signal expectancy is valid *attribution*.
  - EXP-003/004 (momentum real but risk-heavy; trend filter cuts risk, doesn't beat
    the index risk-adjusted) — these ALREADY used monthly *portfolio* returns
    (`gauntlet/momentum.py`), so they are closer to correct and remain the standing
    evidence.
  - EXP-005 (survivorship-biased yfinance "PASS") — remains a permanent example of an
    INVALID result the system correctly rejected.
- **What must be recomputed under the new standard:** any CAGR/Sharpe/drawdown/alpha
  that came from the per-trade path (the non-momentum gauntlet) — once the chronological
  portfolio simulator + daily NAV ledger (Phase 4) exists.
- **Why we do NOT erase the earlier experiments:** negative and cautionary evidence is
  permanent; the log records how our understanding improved, not a cleaned-up story.
- **New standard:** portfolio metrics come only from an immutable daily NAV ledger
  produced by a chronological, point-in-time simulator (see ADR-001 / IMPLEMENTATION_PLAN).
- **Supersedes / references:** governs the interpretation of EXP-002…005; does not
  invalidate their qualitative conclusions.

---

## EXP-006 — Institutional Momentum Breakout v1 (PRE-REGISTERED — not yet run)
- **Status:** PRE-REGISTERED. Framework implemented + unit-tested (synthetic,
  network-free); NOT yet run on `RESEARCH_GRADE` NSE data. No verdict claimed.
- **Date:** 2026-07-28. Branch `overhaul/evidence-lab`. Framework:
  `research/momentum_breakout/` (see ADR-002). Next unused id after EXP-005/META-001.
- **Primary hypothesis:** Stocks with (1) meaningful prior leadership, (2) a long
  contracting base, (3) a confirmed breakout above a pre-existing pivot, (4) small,
  structurally-justified initial risk and (5) strong sector participation have
  **positive forward expectancy after realistic Indian cash-equity costs**. The
  hypothesis is NOT assumed valid — the correct result may be PASS, FAIL or
  INCONCLUSIVE, decided by the existing `research/harness.py` evidence gate.
- **Momentum principle (pre-registered tension):** valuation is NOT a mandatory
  reject in the primary hypothesis — price action governs whether a position stays
  valid, and a name can keep rising despite extreme valuation. Valuation is captured
  as a point-in-time explanatory/risk feature; the experiment SEPARATELY tests
  whether extreme valuation changes forward returns, drawdowns, failure rates or gap
  risk. Fundamentals are never used before their real availability timestamp.
- **Entry convention (frozen):** signal known only AFTER the breakout bar closes;
  entry no earlier than the NEXT tradable bar's open; explicit slippage; gap-through
  the stop fills at the (worse) open, never at the stop price; same-bar ordering is
  pessimistic (stop before target); no fill when it could not realistically execute.
- **Structural stop (frozen):** the primary stop is the highest point-in-time stop
  candidate below entry (swing low / tight-range low / breakout-bar low / pivot−ATR /
  base support); initial risk in % and ATR units; setups over the configurable
  maximum structural risk (~2–8% research range; primary value versioned in
  `config.py`) are rejected. Thresholds are NOT to be optimised against the result.
- **Primary comparisons:** vs (a) Nifty over equal holding windows, (b) all eligible
  liquid stocks, (c) cross-sectional momentum WITHOUT the base+sector conditions,
  (d) breakout candidates WITHOUT the strong-sector requirement.
- **Primary exit (frozen):** initial structural stop + trend-following exit
  (`structural_trend`). Secondary (labelled) variants: `structural_ema_trail`,
  `structural_maxhold`. The best variant will NOT be swapped in post-hoc.
- **Decision metrics:** n_candidates, n_trades, expectancy R + CI, profit factor,
  win rate, avg win/loss, MAE, MFE, max drawdown, turnover, cost drag, Sharpe,
  benchmark-relative, regime breakdown, sector concentration, and the harness's
  DSR / block-bootstrap / BH-FDR multiple-testing controls.
- **Pre-registered ablations (FDR-controlled):** prior-only → +breakout → +long base
  → +small risk → +strong sector → +participation; plus diagnostic splits
  (valuation-extreme vs not, ATH vs non-ATH, strong vs weak sector, low vs high
  structural risk, shorter vs longer bases). Diagnostic only; no component is
  promoted on one favourable slice.
- **Point-in-time safety (fail closed):** six clocks separated (market / signal /
  data-availability / ingestion / entry / fundamental). Guards refuse future bars in
  base/pivot construction, same-day-close entry, future sector membership, future or
  forward-filled fundamentals, and target/exit leakage into entry features. Known
  limitations recorded, not papered over: valuation has no PIT publication dates in
  the repo (fails closed to UNAVAILABLE); sector membership is not historically
  dated (`SECTOR_MEMBERSHIP_NOT_PIT`); universe survivorship is incomplete until
  `logs/universe_history.json` is supplied.
- **Reproducibility:** every observation stamps experiment id, config hash
  (thresholds + primitive/detector/feature/scoring versions), dataset snapshot id,
  and code commit; identical data+config+code reproduce identical observations and
  event ids (unit-tested).
- **No post-result optimisation:** a material threshold change after seeing the
  primary result requires a NEW experiment id (new config hash), per the standing
  research principles above.
- **Supersedes / references:** builds on the momentum evidence (EXP-003/004) and the
  portfolio-metric standard (META-001); introduces the breakout-structure hypothesis
  those did not test. Awaits a run on point-in-time data before any verdict.

---

## EXP-006 — Historical Evidence Run · RESULT: INCONCLUSIVE (data unavailable in run environment)
- **Type:** evidence run of the pre-registered EXP-006 framework (commit `6e7968e`).
  Append-only; does not edit the EXP-006 pre-registration above.
- **Date:** 2026-07-28. Branch `overhaul/evidence-lab`. Runner:
  `research/momentum_breakout/runner.py` + `dataset.py`.
- **Primary verdict:** **INCONCLUSIVE**. Reason: `DATA_UNAVAILABLE`. The point-in-time
  NSE dataset does not exist in this execution environment — the bhavcopy store is
  empty (`logs/bhav/` = 0 files), the NSE index store is empty, there is no network
  to NSE (HTTP 000), and no `universe_history.json` / `ca_events.json` / PIT
  fundamentals. The data-quality gate therefore **failed closed** and the runner
  emitted INCONCLUSIVE(DATA_UNAVAILABLE) rather than fabricating a PASS/FAIL. This is
  the correct, honest outcome — **not** strategy evidence.
- **What WAS delivered (implementation, tested — NOT evidence):** a complete,
  deterministic, network-free historical runner that, given a real point-in-time
  dataset, executes the frozen EXP-006 spec end to end: data-quality gate (fails
  closed on non-positive prices / HLOC inconsistency / duplicate dates / absent
  data), dataset snapshot manifest (reproducible `snapshot_id`), chronological
  candidate generation (one event per breakout, structural dedup), the pre-registered
  gap-aware next-bar-entry simulator (primary + two secondary exits), the six frozen
  ablations, benchmark comparisons, regime / sector / valuation breakdowns, the
  existing harness (DSR / alpha / block-CI) and BH-FDR multiple-testing control, and
  machine-readable artifacts. Verified on a synthetic research-grade dataset: the
  runner produces trades and a coherent verdict, and — critically — **refuses to
  claim a PASS on a small sample** (8 synthetic trades → UNDERPOWERED → INCONCLUSIVE),
  exactly as the evidence standard requires.
- **Verdict-mapping discipline (frozen in the runner):** harness PROMOTE → PASS,
  REJECT → FAIL, UNDERPOWERED/INCONCLUSIVE → INCONCLUSIVE. **Research-grade gate:** a
  would-be PASS on a dataset that is survivorship-incomplete or CA-unadjusted is
  DOWNGRADED to INCONCLUSIVE (a biased PASS is not defensible — the EXP-005 lesson);
  a FAIL is retained (meaningful even on optimistically-biased data). Secondary exits,
  ablations and slices NEVER override the primary verdict.
- **Bug fixes during the run (implementation contradicted robustness, demonstrated by
  tests; hypothesis unchanged, no new experiment id needed):** (1) `_detect_base` and
  the simulator now fail closed on NaN/missing bars so a data gap can never fabricate a
  candidate or an unrealistic fill (previously NaN could slip through on real gappy
  data); (2) `_detect_base` rewritten from an O(base_max²) rescan to an O(base_max)
  incremental scan with **identical output** (verified: all 39 pre-existing detector
  tests still pass) — a pure performance fix that makes a whole-market historical run
  tractable. Both documented; neither changes the tested hypothesis, thresholds,
  pivot/base definition, or config hash semantics.
- **Operator reproduction (where the data lives — e.g. the Mac with a built bhav
  store):** `python -m research.momentum_breakout.runner --out logs/experiments/EXP-006`.
  It builds the store if needed, freezes the snapshot, runs the frozen spec, writes
  the full artifact set, and prints the verdict JSON. The result is reproducible from
  the snapshot manifest (source identities, date range, symbol/row counts, adjustment
  + universe policy, benchmark identity, cost model, code commit, config hash).
- **Material limitations that will bound any real-data verdict (recorded, not hidden):**
  universe survivorship incomplete until `universe_history.json` is supplied; corporate
  actions applied only if `ca_events.json` present (else raw → phantom-gap risk); sector
  membership not historically dated (`SECTOR_MEMBERSHIP_NOT_PIT`); no point-in-time
  fundamentals (`VALUATION_DATA_UNAVAILABLE`, never substituted with current values);
  no delivery-volume PIT data. Under the research-grade gate, a PASS is not attainable
  until at least survivorship + CA are research-grade; a FAIL remains attainable now.
- **Supersedes / references:** executes EXP-006 (pre-registered above); inherits the
  EXP-005 anti-survivorship-mirage discipline (META/EXP-005) as the research-grade gate.
  No verdict on the hypothesis's economic validity is claimed — that awaits a run on
  research-grade point-in-time NSE data.

---

## EXP-006 — Evidence run EXECUTED + artifacts COMMITTED · RESULT: INCONCLUSIVE (DATA_UNAVAILABLE)
- **Type:** execution of the frozen EXP-006 runner (commit `a634be3`) with the
  resulting auditable artifact set persisted into version control. Append-only; does
  not alter the EXP-006 pre-registration or the earlier EXP-006 RESULT entry above.
- **Date:** 2026-07-28. Branch `overhaul/evidence-lab`.
- **What is new vs the prior EXP-006 RESULT entry:** the earlier entry recorded that
  the runner existed and returned INCONCLUSIVE, but its artifacts were transient
  (scratchpad/tmp) and `logs/` is git-ignored. This entry records that the runner was
  **run** and its machine-readable artifacts were **committed** to
  `docs/overhaul/exp006_run/` (force-added past the repo's global `*.json` ignore, on
  purpose, so the evidence record is auditable from the repo).
- **Verdict:** **INCONCLUSIVE — DATA_UNAVAILABLE** (unchanged; the data reality is
  unchanged). Confirmed freshly: `logs/bhav/` = 0 files, `is_ready()` = False, NSE
  `HTTP 000`; a bounded 45s `BhavDataProvider` build attempt **timed out** (no network);
  no universe/CA/fundamental history. The data-quality gate failed closed. **Not
  strategy evidence.**
- **Reproducibility identities (in the committed manifest):** snapshot_id
  `ad652107580ddae1`; EXP-006 config hash `4f638f99e13bf939` (identical to the frozen
  framework — no config drift); code commit `a634be3`; cost model 0.22% round-trip +
  0.10% slippage (modelled aggregate, NOT a broker contract-note replication);
  universe survivorship_complete = false; corporate actions RAW; benchmark/sector/
  fundamental identities unavailable.
- **Artifacts committed:** `data_quality.json`, `snapshot_manifest.json`,
  `experiment_spec.json`, `config_snapshot.json`, `limitations.json`, `verdict.json`,
  `artifact_index.json`, `README.md` (index + reproduce guide). The full observation/
  ledger/ablation/benchmark set is produced only when real data is present.
- **Guarded by tests:** `TestCommittedRunRecord` asserts the persisted verdict stays
  INCONCLUSIVE/DATA_UNAVAILABLE, the manifest matches the frozen config hash, and the
  data-quality gate is failed-closed — so the committed record cannot be edited into a
  false verdict without failing CI.
- **No PASS is attainable on the current data policy:** the runner's research-grade
  gate downgrades a would-be PASS on survivorship-incomplete / CA-unadjusted data to
  INCONCLUSIVE; a FAIL would remain meaningful. A verdict on the hypothesis's economic
  validity still awaits a run on research-grade point-in-time NSE data (operator step;
  `python -m research.momentum_breakout.runner --out logs/experiments/EXP-006`).
- **Supersedes / references:** executes EXP-006 (pre-registered); records the committed
  artifact set for the prior EXP-006 RESULT; inherits the EXP-005 anti-mirage discipline.
