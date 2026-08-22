# EDGE-004 — Implementation Note

Written before results. FEATURE-002 frozen. EDGE-001/002/003 not retuned.

## Why this hypothesis

EDGE-001 tested **medium-term continuation** (12-1 winners). EDGE-002 tested low-vol. EDGE-003 tested trend *inclusion*. Short-horizon **reversal** (Jegadeesh: last-month losers outperform) is the remaining simple CS idea that (a) uses the same PIT path and (b) is not a silent mutation of those specs.

EXP-NEXT-01 FAIL on 1/3/5-day reversal used a 29-name panel. That does not answer a full-universe monthly loser book. It does consume the short-horizon reversal *family* on that panel — this protocol cannot claim a pristine lifetime OOS for “reversal in general.”

## Reuse

Same universe, same-session print, next-open, CNC costs, month-end calendar, EW-universe bench, `incl_momentum` / `skip_momentum` from `research.edge001.momentum`.

## Signal (locked)

At T: R1 = close[T] / close[T-21] − 1, bars ≤ T. Score = −R1. Top 20.

## PIT

Append-future at the same j must not change R1. Fail-closed on short history.
