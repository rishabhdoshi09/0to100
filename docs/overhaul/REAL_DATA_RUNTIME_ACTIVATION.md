# Real-Data Runtime Activation — Implementation Note

Completes the one blocked seam: imported NSE files → **immutable snapshot** → **active pointer**
→ **snapshot-reading provider** → **pinned cycle context** → `run_intelligence_cycle`. No broker
work. Reuses existing normalization/manifest; does not redesign the two brains.

## Existing storage to reuse
- `research/momentum_breakout/data_setup.py`: archive extraction + bhav/index normalization
  (`safe_extract_zip`, `ingest_files`, `_normalize_to_bhav`) + `dataset_snapshot` (per-file sha256).
- `research/momentum_breakout/dataset.py`: `snapshot_manifest` (deterministic id) — reused as the
  manifest idea; the new store makes it immutable + content-addressed + activatable.
- `data/bhavcopy_store.py` (CA-adjusted OHLCV), `data/nse_universe.point_in_time_universe`,
  `data/index_store.py`, `core/market_clock.py` (IST calendar) — referenced by the provider.

## Current data flow (before)
`run_intelligence_cycle_day` → `_build_intel_ctx` → `intel_registry_fn` (unset in prod) →
`CycleContext(data_ok=False, strategies=[])` → honest no-op.

## Exact missing interfaces (added here)
- `research/intelligence/data/snapshot_store.py` — `SnapshotStore`: commit (immutable, content-
  addressed, atomic), `activate` (atomic pointer swap + audit), `get_active`, `open`, `list`,
  `verify`.
- `research/intelligence/data/snapshot.py` — `Snapshot` read-only accessor: `bars(symbol, through,
  adjustment)`, `universe(on_date)`, `benchmark(through)`, `latest_available_date`, `health`,
  `coverage_for(spec)` — all point-in-time, never past `through`.
- `research/intelligence/data/provider.py` — `SnapshotBarProvider`: pinned-snapshot-only, no
  internet/synthetic fallback.
- `research/intelligence/runtime/context_builder.py` — `build_context_from_snapshot(...)`: builds a
  real `CycleContext` (deployable strategies + PIT universe data + benchmark + tier + forward
  eligibility) pinned to ONE snapshot id.

## Snapshot directory / schema
```
logs/snapshots/
  ACTIVE                      # atomic pointer file → snapshot_id (+ audit line)
  <snapshot_id>/
    manifest.json             # full manifest + checksum
    bars_equity.csv           # normalized, sorted (symbol,date,ohlcv,series)
    index_daily.csv           # optional benchmark series
```
`snapshot_id = sha256(canonical_equity_csv + canonical_index_csv + schema_ver + parser_ver)[:16]`
— identical normalized content + versions ⇒ same id (idempotent commit); any change ⇒ new id
(successor).

## Activation & cycle pinning
`activate` verifies then `os.replace`s a temp pointer (atomic; crash leaves old OR new, never
partial). Each cycle pins `ctx.data_snapshot_id` at creation; `cycle_id` already includes the
snapshot, so a re-run is deterministic and a mid-run activation can't switch the pinned data.

## Provider integration seam
`run_intelligence_cycle_day`: `SnapshotStore.get_active()` → `verify` → `open` → `SnapshotBarProvider`
→ `build_context_from_snapshot` → `run_intelligence_cycle`. No active/verified snapshot ⇒ honest
no-op (unchanged behaviour).

## Recovery behaviour
On use: verify manifest checksum + data-file hashes + active pointer resolves; on failure, block
new risk and keep the last verified state. A committed snapshot is never mutated; corrections make
a successor.

## Compatibility risks
- Large multi-year CSV read cost — correctness first; columnar/partitioned storage is deferred.
- Freshness must be trading-calendar aware (weekends/holidays), not wall-clock — minimal calendar
  logic here, full holiday table deferred.

## Implementation order (this milestone ★)
1. ★ SnapshotStore (commit/verify/activate/get_active/open/list) + manifest + content id.
2. ★ Snapshot accessor (PIT bars/universe/benchmark/health/coverage).
3. ★ SnapshotBarProvider (pinned, no fallback).
4. ★ context_builder + data-aware registry readiness + forward-eligibility gate.
5. ★ Wire scheduler; end-to-end snapshot→cycle→paper position; no-snapshot no-op.
6. UI: active-snapshot + readiness surfacing.
7. Deferred (documented): full validation/quarantine severity engine, incremental successor
   reconcile depth, full recovery matrix, scalability benchmarks, full Data-Setup UI overhaul.
