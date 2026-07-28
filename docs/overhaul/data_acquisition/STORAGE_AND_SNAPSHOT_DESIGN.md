# Local research store & snapshot design

Materialise source data into a reproducible local research store, keeping **raw source
files distinguishable from transformed research data**.

## Layout

```
logs/bhav/                         # RAW: per-day NSE archive CSVs + pickled cache (git-ignored)
logs/index/                        # RAW: NSE index CSVs (git-ignored)
logs/ca_events.json                # RAW: corporate-action ledger (operator-supplied)
logs/universe_history.json         # RAW: listing/delisting master (operator-supplied)
docs/overhaul/experiments/EXP-006/ # DERIVED (committed): immutable run records + manifests
```

Raw market data is **never committed** (git-ignored `logs/`, and licensing forbids
redistribution). Only derived provenance (hashes, counts, date ranges, verdicts) is
committed under `docs/`.

## Snapshot record (what every dataset snapshot must carry)

| Field | Source |
|---|---|
| raw source identity | provider + archive URL family |
| retrieval time | fetch timestamp (provenance; not in the reproducible artifacts) |
| **raw content hash** | SHA-256 per raw source file *(planned — see gap below)* |
| **transformed content hash** | SHA-256 of the materialised store *(planned)* |
| transformation version | `pit_indicators`/detector/features/scoring versions (in config hash) |
| schema version | store schema |
| date range, symbols, sessions, rows | data-quality report |
| adjustment policy | RAW vs CA-adjusted |
| universe policy | survivorship_complete |
| benchmark identity | `^NSEI` via index store |
| corporate-action identity | ca_events.json hash (when present) |
| known limitations | data-quality report |

**Equivalent content ⇒ same `snapshot_id`; any material change ⇒ new `snapshot_id`.**

## Current implementation vs gap

- **Implemented:** `dataset.snapshot_manifest()` computes a deterministic `snapshot_id`
  from source identities + date range + symbol/row counts + config hash, and records
  adjustment/universe/benchmark/cost identities + code commit. `data_quality_report()`
  validates integrity and fails closed.
- **Gap (planned, not blocking the decision):** per-file **raw** and **transformed**
  SHA-256 content hashes (C-07 in `TRUTH_AUDIT.md`). Today the snapshot hashes a
  *fingerprint* (counts + identities), not raw bytes. This is documented and reserved
  for the data-platform phase; it does not affect the current honest verdict, which is
  data-unavailable.

## Immutable run records

Runs live under `docs/overhaul/experiments/EXP-006/runs/<run-id>/` (append-only). Each
run keeps: `run_manifest.json` (run id, code commit, config hash, snapshot id, start/
completion, data-quality status, verdict, artifact SHA-256s) + the artifact set. The
existing no-data record is preserved as the immutable **`0001-blocked`** run. Future
runs get new ids; **nothing is overwritten**.
