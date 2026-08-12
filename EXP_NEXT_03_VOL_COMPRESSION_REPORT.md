# EXP-NEXT-03 — Volatility Compression (Risk Context)

> Scientific report. Production unchanged. Not a live trading authorization.

## WHAT WE TESTED / WHAT HAPPENED / WHAT IT MEANS / WHAT QUANTTERM WILL DO

**What we tested:** Whether unusually calm (compressed) price movement changes future downside risk — as a warning, not a buy tip.

**What happened:** After trading costs and the frozen checks, the effect was not reliable (or did not survive independent confirmation).

**What it means:** The idea does not currently show a proven advantage QuantTerm should use.

**What QuantTerm will do:** Nothing. The strategy / risk rule will not be used. Branch closed.


---

## Technical layer

- Experiment ID: `EXP-NEXT-03`
- Hypothesis ID: `ada9d05390b78be1`
- Type: `RISK`
- Snapshot: `a7a9828ec37e09e4`
- Discovery verdict: `FAIL`
- Confirmation: `None`
- Final verdict: **FAIL**
- Next action: `REJECT_CLOSE_BRANCH`
- Production authority: `False`
- Registry: `REJECTED`
- Result hash: `c11659c0ecfafb42`

### Discovery detail

```json
{
  "verdict": "FAIL",
  "reason": "no material downside gap under frozen criteria",
  "tau": 0.6934158713155164,
  "n": 7337,
  "n_compressed": 1922,
  "n_not": 5415,
  "loss_rate_compressed": 0.513,
  "loss_rate_not": 0.4933,
  "loss_rate_gap": 0.0197,
  "left_tail_p05_compressed": -0.0814,
  "left_tail_p05_not": -0.0756,
  "left_tail_gap": -0.0058,
  "mean_fwd_gap_comp_minus_not": -0.0036,
  "incremental_to_abs_vol": true,
  "incr_loss_gap": 0.0783,
  "incr_tail_gap": -0.0098,
  "note": "mean_fwd_gap is descriptive only \u2014 not a BUY criterion"
}
```

### Confirmation detail

```json
null
```

_Generated 2026-08-11T17:14:50.646341+00:00_
