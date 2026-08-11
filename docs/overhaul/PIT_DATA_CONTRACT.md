# QuantTerm Point-in-Time Data Contract (Phase A / A1)

Research code must ask for information **as of** a timestamp. Future-known or
revised current information must never silently contaminate historical tests.

## Facade

```python
from research.intelligence.data import PitContract, SnapshotStore

store = SnapshotStore(root)
contract = PitContract.from_store(store, snapshot_id)

contract.history("bars", symbol="RELIANCE", through="2024-01-15")
contract.latest("bars", symbol="RELIANCE", as_of="2024-01-15")
contract.as_of("universe", when="2024-01-15")                 # snapshot membership
contract.as_of("universe", when="2024-01-15", universe_source="ledger")
contract.as_of("valuations", when="2024-01-15", symbol="RELIANCE")
contract.coverage(as_of="2024-01-15")
```

This is a **thin facade**. It does not copy data into a new store. Canonical
sources remain `Snapshot` / `SnapshotStore`, `data_state`, universe / CA /
valuation ledgers.

## Domains

| Domain | Behaviour |
|--------|-----------|
| `bars` | `Snapshot.bars(symbol, through=as_of)` — never past `as_of` |
| `benchmark` | `Snapshot.benchmark(through=as_of)` |
| `universe` | Snapshot bar-contemporaneous membership, or membership ledger |
| `corporate_actions` | Operator CA ledger events with `ex_date <= as_of` |
| `valuations` | `pit_valuations.get_valuation` (`available_ts <= as_of`) |
| `fundamentals` | Always `NOT_PIT_SAFE` (as-of-now caches) |
| `sectors` | Always `NOT_PIT_SAFE` (static maps, not historically dated) |

## Status vocabulary

Reuses `research.intelligence.data_state` strings where they already exist:

| Status | Meaning for a PIT read |
|--------|------------------------|
| `READY` | Usable, PIT-honest payload |
| `DEGRADED` | Usable with explicit limitations |
| `STALE` | Reserved / freshness signalling (automation) |
| `INCOMPLETE` | Required slice or ledger missing — no fabrication |
| `NOT_PIT_SAFE` | Source exists but is not point-in-time safe; refuse biased fallback |
| `BLOCKED` | Request refused (future `as_of`, no snapshot, unknown domain) |

Automation `DATA_STATES` is unchanged — `INCOMPLETE` / `NOT_PIT_SAFE` / `BLOCKED`
are research-read statuses and do not grant `allows_new_entries`.

## Hard rules

1. No fake fallback data (especially: no “today’s survivors” as READY universe).
2. No network / live fetch inside `PitContract` (`allow_network=True` raises).
3. `as_of` beyond the pinned snapshot’s last trading date → `BLOCKED`.
4. Production runtime may keep using `Snapshot` / `SnapshotBarProvider` directly.
5. Evidence tier claims still go through `data_state.classify_tier`.

See also: `docs/overhaul/A1_PIT_CONTRACT_NOTE.md`.
