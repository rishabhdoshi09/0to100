# A1 — Unified Point-in-Time Access Facade

**Milestone:** Phase A / A1  
**Status:** Research access contract only — no production behaviour change  
**Authority:** `QUANTTERM_INSTITUTIONAL_AI_AUDIT.md`

## Goal

Give research code one explicit way to ask:

> What information was legitimately available as of timestamp T?

## Reuse (no second store)

| Need | Canonical source |
|------|------------------|
| Immutable OHLCV / benchmark | `research.intelligence.data.Snapshot` / `SnapshotStore` / `SnapshotBarProvider` |
| Automation + evidence tier | `research.intelligence.data_state` (`READY`/`DEGRADED`/`STALE` + tiers) |
| Survivorship membership | `data.nse_universe.point_in_time_universe` + `data.universe_history` |
| Corporate actions | `data.corporate_actions.load_events` / `ledger_status` |
| PIT valuations | `data.pit_valuations.get_valuation` |
| Feature observation freeze | `research.feature_store` (unchanged; not reimplemented) |

## What was added

- `research/intelligence/data/pit_contract.py` — thin facade: `history` / `latest` / `as_of` / `coverage`
- PIT-read status constants on `data_state`: `INCOMPLETE`, `NOT_PIT_SAFE`, `BLOCKED` (plus reused `READY`/`DEGRADED`/`STALE`)
- This note + contract section in tests

## What was deliberately not built

- No new SnapshotStore / FeatureStore / data warehouse
- No network / live quote path inside frozen historical reads
- No silent “today’s universe” fallback presented as PIT-safe
- No fundamentals/news as READY (remain `NOT_PIT_SAFE` until dated ledgers exist)

## Semantics

- `as_of` / `history` for bars never return observations dated after the request timestamp.
- Requesting `as_of` beyond the pinned snapshot’s last date → `BLOCKED`.
- Missing universe / CA / valuation ledgers → `INCOMPLETE` or `NOT_PIT_SAFE`, never fabricated rows.
- Current screener fundamentals and static sectors → `NOT_PIT_SAFE`.
