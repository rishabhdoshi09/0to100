# SEPA-003 regime audit

R2.1 labelled every core-F row `unknown` because
`data.index_store.get_index_ohlcv("^NSEI")` had no historically deep
local series (the default store is a short recent window). That was
plumbing, not a market fact.

## Classifier

`regime_pit_v1` on `NIFTY50_EQUALWEIGHT_PROXY_BHAV`:

- Equal-weight daily return of `data.nse_universe.NIFTY50` names that
  have an official bhav bar that session, compounded to a level.
- Member *list* is today’s NIFTY50 (`PIT_DEGRADED`).
- Prices and rolling SMAs use bars ≤ as_of only.

States: STRONG_BULL / BULL / SIDEWAYS / CORRECTION / BEAR / UNKNOWN.

Append-future invariance is tested in `tests/test_sepa_003.py`.

No regime **gate** is added.

## Reconstructed F by entry regime

| Regime | n | E[R] | CI |
|---|---|---|---|
| STRONG_BULL | 359 | +0.281 | [−0.034, +0.638] |
| BULL | 1,341 | +0.319 | [+0.131, +0.537] |
| SIDEWAYS | 1,346 | +0.035 | [−0.120, +0.203] |
| CORRECTION | 356 | −0.139 | [−0.334, +0.069] |
| BEAR | 30 | +2.149 | [+1.138, +3.226] |
| UNKNOWN | 0 | — | — |

BEAR n=30 is below the postcard floor for a claim. H7 p=0.26.

## Mix shift (diagnosis, not a new holdout)

| Era | BULL | STRONG_BULL | SIDEWAYS | CORRECTION | BEAR |
|---|---|---|---|---|---|
| Winning →2023-12-31 | 789 | 318 | 688 | 207 | 18 |
| Weak 2024–2026 | 552 | 41 | 658 | 149 | 12 |

STRONG_BULL almost disappears in the weak era. That is consistent with
`MARKET_CHANGED` as a *contributor*, but year flips (2022 already
negative) keep the official decay label at **UNSTABLE_EDGE**.
