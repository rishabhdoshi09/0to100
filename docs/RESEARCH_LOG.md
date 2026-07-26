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
