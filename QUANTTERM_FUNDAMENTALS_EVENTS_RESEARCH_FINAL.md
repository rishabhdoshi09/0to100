# QuantTerm Fundamentals + Events Research — Final

> End-to-end cycle for EXP-FUND-01..04. No Phase B. No AI/ML. Production unchanged.
> Global trust `OPERATIONAL_ONLY`. Closed OHLCV branches not reopened.

## Plain English

QuantTerm tested four company-information ideas using only facts that were public at each historical date: post-earnings drift, quality, earnings growth, and cheap-vs-expensive (trailing PE). One or more hypotheses confirmed on this certified scope. Follow-up research only — no production authorization.

## Data scope

- Foundation package: `46ff79f58ee21c9e`
- Parent OHLCV: `2f683be0c73eaa33`
- Events / fundamentals / valuations: package ledgers (AVAILABLE_AT enforced via PitContract)
- Partitions: discovery `2023-01-01→2024-06-30`; confirm `2024-07-01→2025-03-18`
- Costs: CNC 0.32 pct points; turnover one-way=1.0
- Multiple-testing: per-test DSR n_trials=4; cycle BH-FDR α=0.05
- FDR rejected (discovery): `['EXP-FUND-03', 'EXP-FUND-04']`

## Results table

| EXPERIMENT | TYPE | DISCOVERY | CONFIRMATION | NET ECONOMIC VALUE | FINAL VERDICT | NEXT ACTION |
|---|---|---|---|---|---|---|
| EXP-FUND-01 | ALPHA_EVENT | INCONCLUSIVE | — | 0.013544 | **INCONCLUSIVE** | HOLD_NO_TUNING |
| EXP-FUND-02 | ALPHA_FUNDAMENTAL | INCONCLUSIVE | — | 0.001788 | **INCONCLUSIVE** | HOLD_NO_TUNING |
| EXP-FUND-03 | ALPHA_FUNDAMENTAL | PASS | PASS | 0.007073 | **CONFIRMED** | ELIGIBLE_FOR_FOLLOWUP_RESEARCH |
| EXP-FUND-04 | ALPHA_FUNDAMENTAL | PASS | INCONCLUSIVE | 0.011434 | **INCONCLUSIVE** | HOLD_NO_TUNING |

## Positive evidence

- `EXP-FUND-03` → `CONFIRMED` (disc net=0.007073)

## Negative evidence

- None.

## Inconclusive evidence

- `EXP-FUND-01` → INCONCLUSIVE (n=12, net=0.013544)
- `EXP-FUND-02` → INCONCLUSIVE (n=85, net=0.001788)
- `EXP-FUND-04` → INCONCLUSIVE (n=85, net=0.011434)

## Scientific-memory updates

- Each experiment recorded REJECT/WATCH beliefs with hypothesis ids.
- Closed OHLCV branches remain closed.

## Production unchanged

| Surface | Status |
|---------|--------|
| Brain / ranking / risk / sizing / execution | Unchanged |
| Autopilot / broker / alerts | Unchanged |
| Any CONFIRMED result | `ELIGIBLE_FOR_FOLLOWUP_RESEARCH` only |

## What NOT to build next

- ML/AI to rescue failed fundamentals hypotheses
- Generic mining across all announcement types
- Sector-neutral redesigns without PIT sector history
- Shareholding factors without AVAILABLE_AT ownership ledger
- Reopening momentum / reversal / low-vol / network branches

## Overall decision

**FOLLOWUP_RESEARCH_FOR_CONFIRMED_ONLY**

One or more hypotheses confirmed on this certified scope. Follow-up research only — no production authorization.

## Status card

| Field | Value |
|-------|--------|
| FOUNDATION | `46ff79f58ee21c9e` |
| OHLCV | `2f683be0c73eaa33` |
| OVERALL | **FOLLOWUP_RESEARCH_FOR_CONFIRMED_ONLY** |
| CONFIRMED | 1 |
| FAIL | 0 |
| INCONCLUSIVE | 3 |

_Generated 2026-08-11T18:15:45.911117+00:00_
_git_sha `378cd4895c6b00fa116a3076e1f0eedac5d84324`_

---

## Follow-up (2026-08-11) — EXP-FUND-03 only

Parent EXP-FUND-03 remains **CONFIRMED** / `ELIGIBLE_FOR_FOLLOWUP_RESEARCH` (not overwritten).

Follow-up experiment **EXP-FUND-03-FOLLOWUP** completed under frozen protocol
`docs/overhaul/EXP_FUND_03_FOLLOWUP_FROZEN_PROTOCOL.json`.

| Item | Result |
|---|---|
| Reproduction | PASS |
| Follow-up verdict | **INCONCLUSIVE_FOLLOWUP** |
| Next action | `RECORD_EVIDENCE_NO_TUNING` |
| Report | `EXP_FUND_03_EARNINGS_GROWTH_FOLLOWUP.md` |

Blocking reasons for `ROBUST_CONFIRMED`: placebo controls not clean; little
incremental information vs 60d momentum after residualization. No production
authority. No Phase B. FUND-01/02/04 remain HOLD_NO_TUNING.
