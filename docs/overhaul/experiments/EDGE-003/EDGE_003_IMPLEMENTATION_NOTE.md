# EDGE-003 — Implementation Note

Written before results. FEATURE-002 frozen. EDGE-001/002 not retuned.

## Why this hypothesis

SEPA-003 retained Trend / Stage-2 as a *concept*. FEATURE-001 tested Trend as a rank feature **on scanner fires**. Nobody has tested Trend as a standalone **inclusion filter** on the full PIT investable universe.

This is not CS 12-1 (EDGE-001) and not low-vol (EDGE-002).

## Reuse

Same universe, same-session print, next-open, CNC costs, month-end calendar, EW-universe bench, regime descriptive tables.

## Signal (locked)

At T: SMA200 = mean of last 200 closes ≤ T. SMA200 rising = SMA200 > SMA200 from 21 sessions earlier. Include iff close[T] > SMA200 and SMA200 rising.

## Portfolio (locked)

Equal-weight **all** qualifiers (not Top 20). Monthly. Next open. No stop.

If the qualifier set is huge, that is a result (the filter is then close to the market), not a reason to switch to Top 20 after seeing returns.

## PIT

SMA windows use bars ≤ T only. Append-future at the same j must not change the flag.
