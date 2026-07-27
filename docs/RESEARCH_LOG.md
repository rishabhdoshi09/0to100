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
