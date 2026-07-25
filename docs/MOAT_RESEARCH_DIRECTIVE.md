# QuantTerm — Moat Research Directive

> A strategic R&D roadmap for durable competitive advantage. Written to be
> scientifically honest first and ambitious second. If a line here sounds like
> hype, it has failed. Companion to `SYSTEM_OVERVIEW.md` and `CLAUDE.md`.

---

## 0. The uncomfortable thesis (read this before anything else)

**You cannot out-alpha Citadel, and you should stop trying.** The directive asks
for "market microstructure, order-flow proxies, auction dynamics, hidden factor
models." With the data QuantTerm actually has — **NSE end-of-day bhavcopy plus a
delayed index-API snapshot** — most of that is not accessible. There is no
Level-2 order book, no true tick tape, no colocation, no paid alt-data. Building
"order-flow alpha" on EOD bars does not produce a moat; it produces **overfit
noise that will lose real money** and violate the system's own first invariant
(*no fake data, evidence over vibes*). A great quant's first move is to ask
"what does my data actually support?" — not "what sounds sophisticated?"

**So where is the real, defensible moat?** Not in the signals. Every technical
signal QuantTerm has can be replicated by Chartink or TradingView in a weekend.
The moat is the thing competitors structurally *cannot* copy:

> **QuantTerm's durable advantage is the closed-loop learning system built on one
> trader's longitudinal ledger of decisions, rejections, and outcomes — a
> data asset that compounds with every trade and is unique to that trader.**

TradingView has 50M users and zero memory of what any of them decided and how it
turned out. QuantTerm already logs **every decision (taken *and* rejected) with
its prediction and its 5-day outcome** (`decision_journal.py`). That ledger is
the crown jewel, and it is almost entirely unexploited. The moats worth building
are the ones that turn that ledger into compounding intelligence:

1. **Rigor as a product** — an anti-overfitting research harness. This is the
   RenTech/Jane Street edge: not exotic signals, but ferocious discipline
   against fooling yourself. **Nothing else on this list is safe to build until
   this exists.**
2. **Counterfactual gate attribution** — what did each filter *cost* vs *earn*?
   The rejected-trade ledger answers this. Nobody else has the data.
3. **Concept-drift detection** — catch edge decay statistically, before the
   equity curve rolls over.
4. **Market memory (analog retrieval)** — "this environment resembles these past
   periods; here is the distribution of what followed." Honest, explainable,
   needs a historical feature store you build once.
5. **Personalized Quant DNA** — learn the *trader*, not just the market. The
   ultimate moat, because it is literally their data. Impossible to replicate.
6. **A self-driving research scientist** — the meta-moat that generates and
   vets hypotheses on top of all the above, gated by the harness in #1.

Everything else in the directive either collapses into these, gets scoped down
because the data is thin, or gets honestly shelved. The rest of this document
argues each case with the 11-point framework you asked for.

**The prioritization principle** used throughout:

```
priority  ≈   defensibility  ×  feasibility_with_current_data  ×  (1 / overfit_risk)  ×  expected_impact
```

A brilliant idea that overfits on 40 samples has negative expected value. A
modest idea that compounds a unique data asset with low overfit risk is a moat.

---

## 1. The moat map (prioritized)

| # | Capability | Defensible? | Feasible now? | Overfit risk | Verdict |
|---|---|---|---|---|---|
| **F** | **Research Governance Harness** | Very high (process) | Yes | N/A (it *reduces* it) | **BUILD FIRST — prerequisite** |
| 7 | Counterfactual gate attribution | Very high (needs the ledger) | Yes | Low | **Tier 1** |
| 3 | Concept-drift detection | High | Yes | Low | **Tier 1** |
| 5 | Market memory / analog engine | High | Yes (build feature store) | Low–Med (as context) | **Tier 1** |
| 10 | Personalized Quant DNA | Highest (their data) | Incrementally | Low | **Tier 1 (compounding)** |
| 4 | AI research scientist | High (meta-moat) | Yes, *if* gated by F | **High if naive** | **Tier 1 — only after F** |
| 2 | Automatic regime discovery | Medium | Partly | High | **Tier 2 — scoped, context-only** |
| 6 | Portfolio optimization | Medium | Yes (robust version) | Medium | **Tier 2 — robust, not Markowitz** |
| 8 | Execution intelligence | Medium | Log now, learn later | Low | **Tier 2 — instrument now** |
| 1 | Microstructure/order-flow alpha | — | **No (no L2 data)** | Extreme | **Tier 3 — reframe, don't chase** |
| 9 | Unsupervised signal discovery | Low as signals | Exploratory only | **Extreme** | **Fold into #4 as hypothesis source** |

---

## FOUNDATION — Research Governance Harness (build this first)

**This is not optional and it is not glamorous. It is the thing that makes
everything else scientifically valid instead of a slow-motion blow-up.**

**1. Problem.** The directive proposes clustering, hidden factor models,
automated hypothesis testing, unsupervised discovery. Run dozens of models
against a few hundred outcomes and you *will* find spurious "edges" — this is
data-mining bias / the multiple-comparisons problem, the single most common way
quants destroy themselves. QuantTerm's real outcome sample is small and grows
slowly (a few trades/day). Naive backtesting on it is a lie generator.

**2. Why competitors rarely do it.** Retail "AI trading" products are *built on*
the overfit — a curve-fit backtest with a 3.0 Sharpe is the marketing. Honest
out-of-sample discipline produces less impressive numbers, so nobody ships it.
That is exactly why it is defensible: it is culturally hard, not technically
hard.

**3. Expected impact.** Better calibration; better robustness; earlier and
*trustworthy* edge detection. It doesn't add return directly — it *prevents
negative return* from every other subsystem. This is the highest-ROI item on the
list precisely because it de-risks all the others.

**4. Statistical justification.** Standard, well-established methods:
- **Walk-forward / purged & embargoed cross-validation** (López de Prado) so
  training never peeks across the outcome horizon (5-day overlap → leakage).
- **Multiple-testing correction:** Benjamini–Hochberg FDR across a batch of
  tested hypotheses; **Deflated Sharpe Ratio** (adjusts an observed Sharpe for
  the number of trials and non-normality).
- **Minimum-sample gates:** reuse the system's existing ≥30 rule, but make it a
  *first-class, enforced* gate with power analysis (how many trades to detect a
  0.1R edge at 80% power?).
- **Out-of-sample holdout that is never touched** until a strategy is promoted.

**5. Data requirements.** Only what already exists: `signal_log`,
`decisions.db`, `signal_backtest.json`, the bhavcopy store. No new data.

**6. Computational cost.** Low. Purged CV and FDR are cheap; DSR is a formula.

**7. Implementation complexity.** Medium — it's a discipline layer more than an
algorithm. A `research/harness.py` with: `purged_cv_split()`, `deflated_sharpe()`,
`bh_fdr()`, `min_sample_power()`, and a `HoldoutVault` that refuses to reveal
holdout performance until a strategy is registered.

**8. Overfitting risk.** This *is* the anti-overfitting subsystem.

**9. Testing methodology.** Unit-test the statistics against known cases (a
50/50 coin must not pass; a genuine +0.3R signal on 200 samples must). Inject
synthetic pure-noise "strategies" and confirm the harness rejects ~all of them
(false-discovery-rate control demonstrably works).

**10. Integration.** Every learning subsystem (`live_edge`, `signal_backtest`,
`ev_engine`, and the research scientist below) routes its claims *through* the
harness before they can change a weight or raise an alert. It becomes the
gatekeeper between "observed pattern" and "acted-upon belief."

**11. Belongs in:** a **new `research/` subsystem** — the scientific spine the
learning loop currently lacks.

---

## TIER 1 — the defensible core

### 7 · Counterfactual Gate Attribution — *the most underexploited asset you own*

**1. Problem.** QuantTerm has ~14 autopilot gates and a stack of scanner demotes.
Nobody knows which ones *make* money and which ones *cost* it. A gate that
rejects trades which would have won is a silent tax. Right now the gates are
believed on faith.

**2. Why competitors don't.** They have no rejected-trade ledger. You can only
ask "what would the rejected trade have done?" if you *logged the rejection with
a reference price* — which `decision_journal.py` already does (TAKEN **and**
REJECTED, each with entry_ref, EV, p_win, and a 5-day outcome). This is a genuine
structural advantage.

**3. Expected impact.** Better decision quality + higher expected return: prune
or loosen gates that are net-negative, tighten those that are net-positive. Even
finding one bad gate is real money. Directly answers "which gates EARN vs COST."

**4. Statistical justification.** This is causal-inference-lite: per gate,
compare the outcome distribution of trades it *rejected* vs a matched set it let
through (matched on regime, score band, sector — to reduce confounding). Report
the average treatment effect of the gate with a confidence interval, run through
the harness (min-sample + FDR across gates). Where a gate's rejects
systematically outperform its passes, the gate is inverting alpha.

**5. Data requirements.** `decisions.db` (already logging). Needs enough
*rejected* outcomes per gate — the main constraint; some gates will stay "THIN"
for months. That's honest and fine.

**6. Computational cost.** Trivial.

**7. Complexity.** Low–Medium (the matching is the only subtlety).

**8. Overfitting risk.** Low — it's measurement of already-collected outcomes,
not prediction. The risk is *confounding*, handled by matching + honest CIs.

**9. Testing methodology.** Back-test the attribution itself on synthetic gates
with known effects; confirm it recovers them. Require ≥N rejects per gate before
any verdict.

**10. Integration.** Feeds the Brain ("gate X has cost +0.4R over 60 rejects —
consider loosening") and, once a gate is proven net-negative through the harness,
proposes a config change for human approval. Never auto-removes a safety gate.

**11. Belongs in:** the **learning loop** (`core/decision_journal.py` extended +
`research/gate_attribution.py`).

---

### 3 · Concept-Drift Detection — *catch decay before the drawdown*

**1. Problem.** Every edge decays. The system already blends "recent vs lifetime"
expectancy, but that's a lagging eyeball check. By the time the lifetime average
turns, you've bled. You need to detect the *change point* early and per-signal:
"breakout edge is deteriorating," "momentum is strengthening," "delivery signal
is no longer predictive."

**2. Why competitors don't.** They ship static screeners. A screener has no
concept of its own decay because it never tracked outcomes in the first place.

**3. Expected impact.** Earlier edge detection → lower drawdown. Demote a
decaying signal *before* it costs a month of losses; lean into a strengthening
one sooner.

**4. Statistical justification.** Online change-point detection on the per-signal
outcome R-stream: **Page–Hinkley test** and **CUSUM** for mean shifts;
**ADWIN** (adaptive windowing) for distribution change; a rolling **SPRT**
(sequential probability ratio test) for "is this signal still ≥ 0R?". These are
textbook streaming-statistics methods designed exactly for "has the process
changed?" with controlled false-alarm rates.

**5. Data requirements.** `signal_log` outcome stream (already there), ordered by
time. More history = earlier detection.

**6. Computational cost.** Very low (online, O(1) per new outcome).

**7. Complexity.** Medium.

**8. Overfitting risk.** Low — change-point tests have explicit false-alarm-rate
control; tune to a conservative ARL (average run length) so it doesn't cry wolf.

**9. Testing methodology.** Synthetic streams with a known change point at t*:
measure detection delay and false-alarm rate; require both within target before
shipping. Replay historical signal_log to see what it would have flagged.

**10. Integration.** A first-class **Brain directive** and a scanner
calibration input: a signal in active drift gets a demote-only caution
("📉 breakout edge weakening — 3 weeks of decay, size down"). This is the
directive's explicit ask ("this should become part of the Brain").

**11. Belongs in:** **learning loop → Brain** (`research/drift.py` feeding
`core/brain.py`).

---

### 5 · Market Memory Engine — *"this resembles these historical periods"*

**1. Problem.** The Brain says "tape is TRENDING_BULL." A trader wants "the last
five times breadth, VIX, sector dispersion, and macro looked like *today*, here's
the distribution of what the next 5–20 days did." Retrieval of statistically
similar regimes, with honest outcome distributions — not a point prediction.

**2. Why competitors don't.** It requires a *historical feature store of daily
market state* and a distance metric — infra nobody builds for a screener. And
done right it's explainable ("nearest neighbours: 12 Mar '22, 18 Sep '23 …"),
which is a Bloomberg-terminal-grade feature retail never gets.

**3. Expected impact.** Better explainability + better calibration of
expectations + context for the posture. Not a direct return signal — a *prior*
that sharpens sizing and stops ("in analogous tapes, breakouts failed 60% —
tighten").

**4. Statistical justification.** k-NN / analog retrieval over a standardized
market-state vector: breadth, % above DMAs, realized vol, VIX, sector-return
dispersion, cross-sectional correlation, macro mood (from `macro_pulse`), options
skew, FII flow. Distance = Mahalanobis (accounts for feature covariance) or a
learned metric. Crucially: **purge overlapping/adjacent days** so "nearest
neighbours" aren't just yesterday. Report the forward-return *distribution* of
neighbours, never a single number.

**5. Data requirements.** A daily **feature store** built once from the bhavcopy
store + index store + options + flows history. This is the real work; the
retrieval is easy after.

**6. Computational cost.** Low (a few thousand daily vectors; k-NN is instant).

**7. Complexity.** Medium (feature engineering + honest purging).

**8. Overfitting risk.** Low–Medium. Low if used as *context/prior*; Medium if
turned into a hard signal — so it stays context by design.

**9. Testing methodology.** Leave-one-period-out: does the neighbour-distribution
forward return beat the unconditional base rate out-of-sample? If not, it's just
a nice UI, and we say so.

**10. Integration.** A **Brain** context block + a Daily Pulse panel ("today's
tape ≈ these 5 dates; breakouts then returned +X% median, 55% hit"). Feeds
sizing priors, not gates.

**11. Belongs in:** **new subsystem** `research/market_memory.py` + a shared
`research/feature_store.py` (also used by regime discovery and the research
scientist — build it once).

---

### 10 · Personalized Quant DNA — *the moat that is literally their data*

**1. Problem.** The system learns the *market* but not the *trader*. The same
setup should be handled differently for a user who exits winners too early, or
who overtrades after a loss, or who only makes money in trending tapes. This is
where "impossible to replicate" becomes literal.

**2. Why competitors don't.** It needs a long, honest record of *this person's*
actual decisions and outcomes — which only exists because QuantTerm logs it.
A new platform starts this user from zero. The DNA compounds; it is a
network-effect-of-one.

**3. Expected impact.** Better decision quality + better capital allocation, personalized:
surface the setups *this* trader converts, warn on the mistakes *this* trader
repeats, size up in *their* strong regimes. `trade_coach.py` already gestures at
this (overtrading/revenge detection) — DNA makes it quantitative and predictive.

**4. Statistical justification.** A behavioral fingerprint from their trade log:
realized vs available R (exit-timing bias), holding-time asymmetry between
winners and losers (disposition effect — a well-documented, measurable bias),
post-loss trade frequency (tilt), per-sector and per-regime hit rate and
expectancy, earnings-week performance. Each is a simple, robust statistic with a
CI; run through the harness so we don't over-personalize on 20 trades.

**5. Data requirements.** `trades.db` + `decisions.db` (own trades). Grows with
use — the whole point.

**6. Computational cost.** Trivial.

**7. Complexity.** Low–Medium (mostly honest statistics + a good UI/report).

**8. Overfitting risk.** Low if gated on sample size (don't tell someone they're
"bad at earnings weeks" on 4 data points).

**9. Testing methodology.** Split the user's history in half; do behavioral
tendencies measured on the first half persist and predict on the second? Report
only tendencies that are stable out-of-sample.

**10. Integration.** A **new `core/trader_dna.py`** feeding: the Brain (personalized
directives), the ticket (a nudge when a setup matches a known weakness), and the
coach. Advice-only, always — it never overrides the human.

**11. Belongs in:** **new subsystem** `core/trader_dna.py`, plus a Pulse report.

---

### 4 · Autonomous Research Scientist — *the meta-moat (only after the Foundation)*

**1. Problem.** Hypotheses currently come from developers. The directive's vision:
the system continuously asks "what should I test next?" — *does delivery matter
more in strong breadth? does ATR behave differently at high VIX? which signal
combos consistently underperform?* — then backtests, rejects the weak, promotes
the strong, writes a report, and proposes weight updates.

**2. Why competitors don't.** It's genuinely hard, and without the Foundation
harness it is an overfitting machine that will confidently recommend garbage.
Most teams can't build the *discipline*, not the automation.

**3. Expected impact.** Compounding edge discovery — the system improves itself
faster than any dev team could. This is the closest thing to a self-improving
quant on the list. Potentially the largest long-run impact, *conditional on* the
harness being real.

**4. Statistical justification.** A structured hypothesis grammar over the
feature store ("conditional edge of signal S given feature F in bucket B"),
generated by an LLM (DeepSeek/JARVIS) *and* by unsupervised mining (this is where
#9 belongs — as a hypothesis *source*, never a live signal). Every hypothesis is:
pre-registered → tested with purged CV → corrected for multiple testing (FDR /
deflated Sharpe across the batch) → validated on untouched holdout → promoted
only if it survives all four. **Combinatorial Purged Cross-Validation** to
estimate the probability of backtest overfitting (PBO).

**5. Data requirements.** The feature store (#5) + outcome ledgers + the harness
(F). No external data.

**6. Computational cost.** Medium — batched backtests, run off-hours (the system
already has a nightly backtest slot).

**7. Complexity.** High — the flagship. Build last of Tier 1.

**8. Overfitting risk.** **High if naive — this is the whole danger.** Fully
mitigated only by routing 100% of claims through the Foundation harness and a
sacrosanct holdout. Under no circumstances does a hypothesis change a live weight
without surviving the full gauntlet + human sign-off initially.

**9. Testing methodology.** Feed it pure-noise features; it must promote ~none
(PBO near 1, DSR insignificant). Feed it a planted true edge; it must find it.
Track its *own* live hit rate — the research scientist is itself outcome-tracked.

**10. Integration.** Runs in the nightly slot; outputs a **research report**
(new Pulse tab / Telegram digest) and *proposed* calibration deltas that a human
approves before they touch the scorer. Later, high-confidence promotions can
auto-apply within tight bounds.

**11. Belongs in:** **new `research/scientist.py`**, orchestrating the harness,
feature store, and backtester. The capstone.

---

## TIER 2 — real, but scope carefully

### 2 · Automatic Regime Discovery — *do it, but as context, not gates*

Replace hand-labeled regimes (BULL/CHOP/BEAR) with unsupervised ones (HMM,
Gaussian mixture, or HDBSCAN over the market-state feature vector), selected by
BIC and **required to be economically interpretable**. *The catch:* unsupervised
regimes are unstable and overfit easily on limited history; label-switching and
spurious micro-regimes are common. **Verdict:** build it on top of the feature
store, validate regime persistence out-of-sample, and use discovered regimes as
*additional context* for `live_edge`/`ev_engine` conditioning and market-memory
retrieval — **never** as a hard hard-coded gate replacement until they prove
stable over ≥1 year. Overfit risk: **High** → harness-gated. Belongs in:
`research/regimes.py` feeding the Brain.

### 6 · Portfolio Optimization — *robust, not Markowitz*

The system has `portfolio_intel.py` (rotation advice) and `correlation.py`
(clusters). Extend to: marginal-EV contribution per candidate, diversification
benefit (does it join an existing correlation cluster or open a new bet?),
opportunity cost of each allocation. **The catch:** full mean-variance
optimization on noisy EV + covariance estimates is a classic overfit — it
maximizes estimation error. **Verdict:** use *robust* methods — rank by marginal
EV net of correlation-cluster penalty, cap by the existing risk rails, maybe
hierarchical-risk-parity for weights (López de Prado — no matrix inversion, far
more stable than Markowitz). Overfit risk: Medium. Belongs in: `risk/` +
`core/portfolio_intel.py`.

### 8 · Execution Intelligence — *instrument now, learn later*

Learn from slippage, fill quality, spread, time-of-day, partial fills. **The
honest catch:** paper fills are *simulated* (`cost_model.py`); real fills exist
only for live trades, of which there are currently few. Learning a slippage model
on simulated fills teaches you your own assumptions. **Verdict:** build the
**logging and attribution now** (capture requested vs filled price, spread,
time-of-day, latency on every live order), accumulate, and only fit the model
once there's a real sample. Low overfit risk, low immediate value, high future
value. Belongs in: `execution/` + a new `execution/exec_analytics.py`.

---

## TIER 3 — challenge and reframe

### 1 · "Alpha from microstructure / order flow" — *not with this data*

Be blunt: **without Level-2 depth, tick order flow, or paid alt-data, genuine
microstructure alpha is off the table**, and pretending otherwise manufactures
overfit noise. What *is* feasible and genuinely underexploited by retail, using
data you already have:

- **Cross-asset lead–lag:** crude, USD/INR, US indices (overnight), India VIX,
  bond yields → conditioning variables and short-horizon lead-lag on NSE
  sectors. `index_store` + `macro_pulse` make this buildable. *This* is the
  honest "new source of alpha," and it's real.
- **Relative leadership / rotation:** cross-sectional RS ranking and sector
  leadership *transitions* (which the directive lists) — feasible from bhavcopy,
  partly present, worth deepening into a proper leadership-rotation model.
- **Institutional footprint proxies you already have:** delivery % (real
  ownership), FII/DII flows, bulk deals, options positioning. Deepen these
  (e.g., delivery-% *conditioned on breadth* — a research-scientist hypothesis)
  rather than inventing order-flow you can't see.

**Verdict:** reframe #1 as "cross-asset + leadership conditioning," build it
through the research scientist as *conditional* edges, and explicitly **shelve
true microstructure** until/unless the data budget changes. Honesty here is the
edge — it's why the system won't blow up chasing a mirage.

### 9 · Unsupervised Signal Discovery — *fold into #4, never a live signal*

Clustering trades into "hidden winner/loser archetypes" is a superb *hypothesis
generator* and a **catastrophic direct signal source** (pure data-mining, extreme
overfit). **Verdict:** it lives *inside* the research scientist (#4) as one
hypothesis source, subject to the identical harness gauntlet. It never emits a
live signal on its own. Belongs in: `research/scientist.py`.

---

## 2. Why this is a moat and not just features

Three compounding advantages, none copyable by a screener:

1. **Data network-effect-of-one.** The decision+outcome ledger and the trader
   DNA grow with every day of use and are unique to the user. A competitor
   starts every user at zero memory. Switching cost rises monotonically.
2. **Process rigor as a brand.** A platform that *demonstrably* controls its own
   false-discovery rate and shows its holdout results is trustworthy in a market
   full of curve-fit hype. That trust is defensible because it's culturally hard
   to fake.
3. **Self-improvement compounding.** Once the research scientist + harness loop
   runs, the edge-discovery rate compounds. Competitors ship static screeners;
   QuantTerm gets smarter while it sleeps — safely, because of the harness.

---

## 3. Recommended build sequence (dependencies matter)

```
  ┌─ F. Research Governance Harness  ──────────────┐  (prerequisite for all learning claims)
  │                                                │
  ├─ 7. Counterfactual gate attribution  ← ledger already exists; fastest real win
  ├─ 3. Concept-drift detection          ← streaming stats on signal_log
  │
  ├─ FEATURE STORE (shared) ──┬─ 5. Market memory / analog engine
  │                           ├─ 2. Regime discovery (context-only)
  │                           └─ 4. Research scientist  ← capstone, needs F + store
  │
  ├─ 10. Trader DNA           ← parallel; compounds from day one
  ├─ 6. Robust portfolio opt  ← extends existing modules
  └─ 8. Execution logging     ← instrument now, model later
```

**If you build only three things:** the **Harness (F)**, **Counterfactual gate
attribution (7)**, and **Concept-drift detection (3)**. They are low-overfit,
feasible today on existing data, exploit the unique ledger, and de-risk
everything after. That is the minimum viable moat.

---

## 4. Governing principles (the kill criteria)

Every proposed subsystem must clear these before it ships, or it dies:

- **It survives the harness.** Out-of-sample, multiple-testing-corrected, above
  the minimum-sample power threshold. No exceptions, including for ideas I
  proposed here.
- **It maps to a principle metric:** higher expected return · lower drawdown ·
  better calibration · earlier edge detection · better allocation · better
  execution · better decision quality · better robustness · better
  explainability. If it maps to none, it is complexity for its own sake — cut it.
- **It respects the seven invariants.** Especially: no fake data (a thin sample
  stays "unknown," never a confident number), and no live action a human didn't
  authorize while the subsystem is unproven.
- **It is outcome-tracked itself.** Every new intelligence layer logs its own
  predictions and gets graded, exactly like a BUY. The research scientist is
  measured by whether its promoted hypotheses actually paid.
- **Explainability is a feature, not an afterthought.** "This resembles these
  dates," "this gate cost +0.4R over 60 rejects," "your DNA converts pullbacks
  but chases breakouts" — legible reasons are half the moat.

---

## 5. What I am *not* recommending (and why)

- **Deep-learning price prediction / LSTMs on OHLCV.** Overfits ferociously on
  this sample size; unexplainable; competitors' graveyard. No.
- **Sentiment-from-social-media alpha.** Noisy, gameable, no proven edge at this
  latency; `macro_pulse` already captures the honest, corroboration-gated
  version as *context*, not signal.
- **High-frequency / intraday microstructure.** Data and latency you don't have.
- **A bigger signal zoo.** More indicators ≠ more edge; it multiplies the
  overfitting surface. The moat is the loop around the signals, not the signals.

---

*The objective is not a feature-rich product. It is a product that gets
measurably, defensibly smarter every day — while being ruthlessly honest about
what its data can and cannot prove.*
