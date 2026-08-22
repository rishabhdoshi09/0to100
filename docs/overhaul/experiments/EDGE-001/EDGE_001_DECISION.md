# EDGE-001 — Decision

## `RESEARCH-ONLY`

Exactly one label is allowed. None of them authorise live trading, paper autopilot, FEATURE-002 changes, or production BUY edits.

- Failures: ['confirmation_reverses_development']
- Notes: ['H3 falsified: including the last month beat 12-1. Do not switch inside EDGE-001.']
- Later-block mean excess vs EW: 3.30%
- Later-block mean excess vs Nifty: 6.16%
- Live authorised: `False`
- Paper authorised: `False`
- FEATURE-002 change authorised: `False`

H1 is not rejected on full-sample CAGR, but monthly net excess vs the equal-weight investable universe has a block-bootstrap CI that includes zero and a hit rate below 50%. Confirmation 2025–2026 reverses development. H2 holds on the pooled decile means (ordered slope) and fails as a year-by-year law. H3 (skip last month) does not improve on 12-0. H4 is descriptive: bull months are stronger; correction months are negative. No regime gate is added.

Do **not** rescue this milestone with stops, sector caps, news, or AI. A 9-1 or crash-aware variant would be a **new** protocol, not a silent retune of M1/Top20/monthly. A stop overlay would be EDGE-002 only after a surviving primary effect.
