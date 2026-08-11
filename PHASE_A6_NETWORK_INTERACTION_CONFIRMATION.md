# Phase A.6 — Network Interaction Confirmation

> Independent confirmation of the sole surviving Phase A.5 finding (`signal_x_network_concentration`). **Not Phase B. Production unchanged. Not a BUY/SELL signal.**

## Plain English

QuantTerm noticed that some otherwise-good trade signals performed differently when the portfolio was already crowded into stocks behaving similarly.

The earlier result did not repeat on new data, so QuantTerm will not use it.

Close this research branch. Do not mine similar interactions.

---

## 1. Discovery result reference

- Parent experiment: `EXP-A5A6-01` / `3734b8a0a9124a60`
- Snapshot: `a7a9828ec37e09e4`
- Surviving interaction: `signal_x_network_concentration`
- Discovery delta_corr=0.3637 p=0.0091 (FDR-cleared)
- Interpretation: PASS_RISK / context effect only — NOT standalone return prediction. Confirmation asks whether elevated portfolio-network concentration worsens the risk profile of otherwise comparable signals.

### Rejected branches (frozen)

- `EXP-A5-01` `dynamic_market_structure_incrementality` → **REJECT** (do not retune / escalate)
- `EXP-A6-01` `standalone_portfolio_network_predictability` → **REJECT** (do not retune / escalate)
- `EXP-A2-01` `multi_horizon_momentum_family` → **REJECT** (do not retune / escalate)
- `EXP-A3-01` `simple_logistic_challenger` → **REJECT** (do not retune / escalate)

## 2. Frozen confirmation hypothesis

- Protocol file: `/workspace/docs/overhaul/PHASE_A6_FROZEN_CONFIRMATION.json`
- Protocol sha256: `b53eba391069d0367da8b47815beaf7bdc9ba42d0473167d5f8d37d670f47dc5`
- Primary H1: higher `portfolio_network_concentration` context changes signal→10d-forward association in the **discovery direction** (`delta_corr > 0`), as risk/context — not standalone alpha.
- No post-hoc `network_concentration > X` threshold invented from confirmation data (within-sample median split only).

## 3. New experiment ID

- Experiment: `EXP-A6-CONF-01`
- Hypothesis ID: `e6092c83f98fba20`
- Registry status: `REJECTED`
- Does **not** overwrite `EXP-A5A6-01` / `3734b8a0a9124a60`.

## 4. Confirmation dataset certification

- Status: `READY`
- Global trust: `OPERATIONAL_ONLY`
- Parent scoped cert: `READY_FOR_SCIENTIFIC_RERUN`
- Snapshot: `a7a9828ec37e09e4`
- Sample mode: `untouched_historical_holdout_excluded_from_discovery_eval`
- Forward post-discovery available: `False`
- Note: No certified sessions after discovery end (2026-08-11). Preferred later-period confirmation is NOT available.

## 5. Proof of sample independence

Confirmation evaluation dates are disjoint from discovery evaluation dates; each confirmation forward window ends before discovery oos_start 2025-09-19.

- Discovery eval dates: 2025-09-19 → 2026-07-13 (n=11)
- Confirmation eval dates: 2023-11-22 → 2025-08-12 (n=22)
- Overlap: `[]`
## 6. Exact metric definitions

- Signal: 60d cross-sectional momentum rank on frozen 29-name panel.
- Network concentration: Herfindahl of correlation-community weights (ρ≥0.70) for top-5 momentum names, 60d lookback.
- Primary: median-split Fisher-z δcorr(signal, 10d fwd) high−low.
- Controls: top-5 pairwise cluster concentration + sector HHI.
- Economic risk: among signal≥0.5, require loss-rate gap ≥5pp OR left-tail (p05) gap ≤ −1pp for high vs low concentration.

## 7. Baseline / control comparison

```json
{
  "ok": true,
  "controls": [
    "pairwise_conc",
    "sector_hhi"
  ],
  "residual_interaction": {
    "corr_low": 0.0443,
    "corr_high": -0.2033,
    "delta_corr": -0.2475,
    "p": 0.0052,
    "n_low": 464,
    "n_high": 174,
    "median_split": 0.2
  },
  "incremental": false
}
```

## 8. Statistical result

```json
{
  "corr_low": 0.0435,
  "corr_high": -0.1894,
  "delta_corr": -0.2329,
  "p": 0.0086,
  "n_low": 464,
  "n_high": 174,
  "median_split": 0.2
}
```

- Discovery reference delta_corr=0.3637 vs confirmation delta_corr=-0.2329

## 9. Economic / risk result

```json
{
  "ok": true,
  "median_net_conc": 0.2,
  "n_high": 90,
  "n_low": 240,
  "loss_rate_high": 0.3556,
  "loss_rate_low": 0.4375,
  "loss_rate_gap": -0.0819,
  "left_tail_p05_high": -0.0539,
  "left_tail_p05_low": -0.0664,
  "left_tail_gap": 0.0125,
  "mean_negative_fwd_high": -0.0326,
  "mean_negative_fwd_low": -0.033,
  "mean_fwd_high": 0.0143,
  "mean_fwd_low": 0.0103,
  "max_adverse_proxy_p05_high": -0.0539,
  "economic_risk_meaning": false,
  "opportunity_cost": {
    "n_signal_cohort": 330,
    "n_would_demote": 90,
    "demote_rate": 0.2727,
    "mean_fwd_demoted": 0.0143,
    "mean_fwd_kept": 0.0103,
    "mean_fwd_gap_kept_minus_demoted": -0.004,
    "cost_pct_reporting_only": 0.32
  }
}
```

## 10. Opportunity-cost analysis

```json
{
  "n_signal_cohort": 330,
  "n_would_demote": 90,
  "demote_rate": 0.2727,
  "mean_fwd_demoted": 0.0143,
  "mean_fwd_kept": 0.0103,
  "mean_fwd_gap_kept_minus_demoted": -0.004,
  "cost_pct_reporting_only": 0.32
}
```

## 11. Regime / subperiod diagnostics

Not preregistered beyond the single holdout window; no post-hoc regime mining performed.

## 12. Confirmation verdict

- **FAILED_CONFIRMATION**
- Reason: primary interaction not replicated (delta_corr=-0.2329, p=0.0086)
- Next action: `REJECT`

## 13. Scientific-memory update

- Four rejected A.5 branches frozen as negative evidence.
- Confirmation outcome recorded under `e6092c83f98fba20` (discovery id untouched).

## 14. Production behaviour confirmation

- production_behaviour_changed: `False`
- production_authority: `False`
- phase_b_started: `False`
- Brain / ranking / sizing / risk vetoes / execution / broker: **unchanged**.

## 15. Plain-English explanation

The earlier result did not repeat on new data, so QuantTerm will not use it.

---

## Final matrix

| Field | Value |
|---|---|
| DISCOVERY RESULT | PASS_RISK / `signal_x_network_concentration` δcorr=0.3637 |
| CONFIRMATION RESULT | `FAILED_CONFIRMATION` |
| INCREMENTAL TO EXISTING CONTROLS? | `False` |
| ECONOMICALLY MEANINGFUL? | `False` |
| FINAL VERDICT | **FAILED_CONFIRMATION** |
| NEXT ACTION | `REJECT` |

STOP. Do not begin the policy experiment. Do not begin Phase B.

_Evaluated at: 2026-08-11T17:04:18.354885+00:00_
_git_sha: `79fc9dede097c2666adb9cae353618785665df47`_
