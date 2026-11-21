# Why the desk's setups lose: attribution on discovery data (21 Nov 2026)

**Research only.** Descriptive, not confirmatory. It spends no alpha and changes no gate.

`research/loss_attribution.py` runs the **production** walk-forward on the offline SSD copy:

- `edge_diagnosis.collect` → `scan.signal_backtest`;
- the one rulebook and production costs;
- the session-anchored grid and hash sample from this cycle;
- 400 symbols, decisions every 5th session.

Only **STRICT-tier** decisions (whole 52-week lookback corporate-action adjusted) with verified point-in-time membership are kept. Only **discovery** decisions (before 2025-11-01) are measured. The 2,196 trades on or after that date are counted, not measured.

- **Sample:** 7,394 filled trades and 12,199 detections, decisions 2022-09-07 to 2025-10-29, 78 independent 10-session blocks.
- **Intervals:** overlap-robust (`research.overlap_inference`), with no multiplicity correction.

Reproduce: `nice -n 19 venv/bin/python -m research.loss_attribution --root <SSD copy root> --symbols 400 --accept-unbound-coverage --out <json>`

## 1. The headline: the signals do not select

| Measure (STRICT, discovery) | Mean [95% CI] |
|---|---|
| Every detection, entered at the decision close, 10-session return | +0.75% [−0.13, +1.63]% |
| **Every detection, 10-session excess over the equal-weight market** | **−0.17% [−0.49, +0.15]%** |
| Filled detections only, excess (look-ahead: a buy-stop fills only after a rise) | +1.79% [+1.32, +2.26]% |
| Unfilled detections, excess (same look-ahead, other side) | −3.20% [−3.79, −2.60]% |
| Production trades, gross R | −0.025R [−0.082, +0.032] |
| **Production trades, net R** | **−0.137R [−0.208, −0.066]** |
| Mean round-trip cost | **0.112R per trade** |
| Trade net return vs the market's over the same window | −1.42% per trade [−1.80, −1.03]% |
| Beta of trade returns on the equal-weight market (OLS, descriptive) | 0.55; alpha per trade −0.99% |

**Reading.**

- Taken unconditionally, the scanner's setups pick stocks that do **what the equal-weight market does** over the next 10 sessions. There is no stock-selection alpha to protect.
- The apparent selection edge of filled trades (+1.8%) is look-ahead conditioning. The buy-stop fills on stocks that then rose, and the unfilled ones are its mirror image (−3.2%).
- The rulebook's trade construction then turns a zero-alpha selection into a **gross ≈ 0R** trade (−0.025R).
- Costs of about **0.11R per trade** make it **−0.14R**. They are large in R because stops sit about 3% from entry, so the 0.32% round trip is about a ninth of the risk unit.

## 2. Where the R goes

| Exit | Share of trades | Mean net R | Sum net R |
|---|---|---|---|
| LOSS (stop) | 31.2% | −0.80 | −1,847 |
| SCRATCH (breakeven trail) | 33.6% | −0.05 | −114 |
| FLAT (10-session time exit) | 27.5% | −0.03 | −64 |
| WIN (target) | 7.7% | +1.78 | +1,009 |

| Holding period | Trades | Net R [95% CI] |
|---|---|---|
| 1–3 sessions | 3,335 | −0.20 [−0.28, −0.13] |
| 4–6 | 1,975 | −0.12 [−0.23, −0.00] |
| 7–10 | 2,084 | −0.05 [−0.23, +0.12] |

- Fast failures carry the loss: 45% of trades end within 3 sessions.
- The target is hit by only 7.7% of trades inside 10 sessions.

## 3. Geometry counterfactuals on the same fills (pre-declared, not searched)

| Exit rule | Net R [95% CI] |
|---|---|
| Production (stop, target, breakeven trail) | −0.137 [−0.208, −0.066] |
| Time exit at 10 sessions, no stop | −0.074 [−0.190, +0.042] |
| Stop only, time exit at 10 | −0.315 [−0.408, −0.223] |
| 2× wider stop, time exit at 10 | −0.135 [−0.248, −0.023] |

No exit construction turns these fills positive. The stop limits losses, and removing the trail and target does not reveal a hidden edge. **The stop/target geometry is not destroying a positive selection; there is none to destroy.**

## 4. Slices (each a hypothesis at most; no multiplicity correction)

- **Regime:** BULL −0.14R, CHOP −0.16R, BEAR −0.11R (insufficient blocks). No regime is positive.
- **Breadth** (share of names above their 50-session mean): BROAD −0.13R, MIXED −0.16R, NARROW −0.12R. No breadth state is positive.
- **Signals:** every signal with at least 30 trades has a negative point estimate. The least bad are BREAKOUT_52W (n = 41, −0.01R), CUP_HANDLE (−0.04R), PULLBACK_SUPPORT (−0.06R) and MOMENTUM (−0.06R). The worst are FLAT_BASE (−0.70R) and VCP (−0.47R).
- **More confirmation is worse:** 1 signal −0.11R, 2 signals −0.15R, 3 or more −0.16R.
- **Sectors:** 76% of trades fall in the UNMAPPED sector bucket (the sector map covers 337 symbols). One mapped sector (Railways, n = 75, +0.45R) is a lone positive among about ten tested. It is a multiple-comparison artefact until shown otherwise and is not pursued.
- **Quality concentration could not be measured.** The trade ledger's `confidence` is 0 for every trade, because the scanner does not populate it. This matches the learning corpus, where 18 of 30 decision features are constant (`docs/audits/2026-11-21-cycle2-checkpoint.md` §4).

## 5. What follows

1. **Tuning stops, targets, entries or confirmations on this family is not justified.** Selection alpha is about zero, so geometry can only redistribute a zero-mean outcome and pay costs. Searching geometries on these outcomes would be curve fitting.
2. **The economically different hypotheses tested this week:**
   - 12-1 momentum: NOT_SIGNIFICANT, beta 1.18.
   - Low volatility: PASS in discovery (p = 0.049), NOT_SIGNIFICANT in the single held-out look (alpha −0.30%/month, p = 0.70).
   - Neither is supported out of sample (`docs/research/lowvol_discovery_2026-11-21.md`).
3. **The scientifically honest conclusion is that further evidence is necessary.**
   - The already-frozen prospective 52-week-high test (`research/preregistered/pr_2026_11_20_52wh.json`) is the next clean evidence source. Wiring it into the shadow ledger is an owner decision.
   - Every retrospective look at this history now carries the disclosure of what has been seen.
