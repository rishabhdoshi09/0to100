# Missing-data policy

Never silently:

- fill fundamentals with zero
- forward-fill across unknown filing dates
- copy today's sector backward and call it PIT
- use stale prices indefinitely
- treat a missing benchmark return as 0

Machine-readable rules: `research.data_foundation.policy.POLICY`.

| Field | Missing | Carry-forward | Max staleness | Notes |
|---|---|---|---|---|
| OHLCV close | FAIL | last official bar only inside the session window | 5 sessions | Then FAIL (not investable). |
| Fundamentals metric | UNKNOWN | none across unknown filings | usable only after `available_at` | Restatement is a new row. |
| Derived ratio | UNKNOWN | none | n/a | Requires PIT inputs + `calc_version`. |
| Earnings surprise | FAIL | n/a | n/a | Forbidden without consensus history. |
| Sector | UNKNOWN | static map only if labelled STATIC_BACKFILL | n/a | Never upgrade to PIT_SECTOR_STRONG. |
| Benchmark return | UNKNOWN | none | n/a | Not 0. |
| Universe membership | FAIL | none | delisted is out | Do not invent listings. |
| CA factor | FAIL | none | n/a | No gap-inferred factors. |

`decide(field, value)` returns FAIL / UNKNOWN / OK.
