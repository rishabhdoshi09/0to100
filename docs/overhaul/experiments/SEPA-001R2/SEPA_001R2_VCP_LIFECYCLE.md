# SEPA-001R2 VCP lifecycle

## Contraction sequence

SEPA-001R `_contractions_from_swings` stopped at `max_contractions`, keeping the **earliest** coils. With 8+ contractions in lookback, `pivot_last_contraction_v1` could be a stale high.

R2:

1. Collect every causally confirmed high→low leg
2. `select_active_sequence` takes windows that **end on the latest contraction**
3. Longest valid window ≤ `max_contractions` wins
4. An older valid coil is never preferred over a newer invalid live structure

Tests: 2 / 3 / 6 / 8 contractions; extra old coils cannot move the current last-contraction pivot backward.

## Setup identity

001R `setup_id(symbol, rolling base_start)` minted a new ID when the first contraction aged out of 120 bars.

R2 `PersistentSetupLedger` freezes `original_base_start` at first observation of the live coil and keeps one ID while the symbol’s open setup is not core-terminal. A new ID requires FAILED / FILLED / GAP_THROUGH / MISSED / EXTENDED / EXPIRED, not a rolling array.

## Left-censor and retest

| Case | Label | Core F |
|---|---|---|
| First eval observation already through the zone | `LEFT_CENSORED` | no, excluded from opportunity stats |
| Observed ENTRY_READY then escape | `EXTENDED` / `MISSED` / `GAP_THROUGH` | no fill |
| Forming below pivot | track | not a miss |
| Return to zone after EXTENDED | `PIVOT_RETEST` | **not** a core SEPA entry |

Warm-up (252 sessions) initializes pattern state; counting starts at the first post-warm-up as-of (2020-09-16 on this book).
