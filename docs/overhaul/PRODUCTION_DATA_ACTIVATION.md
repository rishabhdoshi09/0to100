# Production Data Activation — Audit & Implementation Map

Repository-first audit (Phase 0). Reuse-focused: much of the data foundation already exists;
this milestone connects it to the running loop and expands genuine strategy support.

## Current import pipeline
`ui/data_setup_page.py` → `research/momentum_breakout/data_setup.py`:
`safe_extract_zip` (traversal/bomb/size guards) + `ingest_files` (csv/json/md/pdf, flexible
column aliasing `_ALIAS`, multi-day split) → `validate_dataset` (OHLC sanity, dupes, dates) →
`readiness` (green/amber/red) → `dataset_snapshot` (sha256 per file + coverage) →
`save_into_canonical` → `materialize` (builds `bhavcopy_store` / `index_store`).

## Currently accepted formats
`.csv`, `.md`, `.json` (ca_events/universe_history), `.pdf` (best-effort), and `.zip`
containing those. Flexible headers via `_ALIAS`/`_colmap`.

## Exact reasons realistic NSE packages fail today
- **Nested compression**: `BhavCopy_..._F_0000.csv.zip` (a `.csv.zip` member, or a zip-of-zips)
  is not recursively unpacked → the inner CSV is never seen. `.csv.gz` unsupported.
- **No file classification report**: result is "usable/could not read", not per-file
  Accepted/Quarantined/Duplicate/Unsupported with counts and reasons.
- **Row-level quarantine**: a few bad rows can fail a file rather than quarantining the rows.

## Existing reusable components (do NOT duplicate)
- `research/momentum_breakout/dataset.py`: `BhavDataProvider` (symbols/get_ohlcv),
  `snapshot_manifest` (deterministic `snapshot_id`), `data_quality_report`.
- `data/bhavcopy_store.py`: CA-adjusted `get_ohlcv` (`corporate_actions.adjust_frame`).
- `data/nse_universe.py`: `point_in_time_universe(as_of)` (survivorship-aware).
- `data/index_store.py`: index/VIX series for regime + benchmark.
- `data_setup.py`: ingestion + validation + snapshot + canonical save.

## Duplicate implementations to AVOID
Do not build a second snapshot id, second OHLCV provider, second universe filter, or second CA
adjuster. Extend `dataset.py` / `data_setup.py` / the stores.

## Missing point-in-time / protection fields
- Explicit `knowledge_available_timestamp` per bar/CA (only ex-date today).
- Evidence-eligibility TIER on a dataset (all-or-nothing readiness today).
- System-wide DATA STATE with partial-degradation isolation (Boolean readiness today).
- Production registry of frozen strategy specs (none — the loop has no registry).
- Bar-by-bar adapters for ~9 registered families (only breakout + vol-contraction).

## Current production provider gaps
`providers.daily_bars` reads today's bhav; `signals_for` reuses the SCANNER (not per-strategy
rules). The runtime path needs: snapshot-pinned bars through as_of, PIT universe, raw/adjusted
state, and per-strategy signals — never internet/synthetic fallback.

## Recommended implementation order (this milestone delivers ★)
1. ★ **Genuine strategy adapters** (Phase 12): trend-following + pullback (single-symbol),
   cross-sectional momentum + relative-strength (cross-sectional). Unified `signals()` interface.
2. ★ **Production strategy registry** (Phase 11): build from frozen specs + startup validation
   (missing adapter / dup id / unknown family / bad universe) → disable-with-reason, never crash.
3. ★ **Data operating states + evidence tiers** (Phases 16, 19): NO_DATA/READY/…/FAILED with
   partial isolation; OPERATIONAL_ONLY→FORWARD_ELIGIBLE tiers surfaced in Evidence Cards.
4. ★ **Wire registry + snapshot provider into the cycle** so PAPER_AUTO runs on a validated
   snapshot (fixture-injected in tests; honest no-op with no data).
5. ★ **Strategy Coverage UI** (Phase 18): registered / runtime-supported / data-sufficient /
   evidence-state / paper-eligible, with the exact missing component per strategy.
6. Nested-archive ingestion (`.csv.zip`, `.csv.gz`, zip-of-zips) + per-file classification
   report + row-level quarantine — extends `data_setup.py`. **(next increment)**
7. Immutable snapshot store + incremental successors + snapshot pinning per cycle (reuse
   `snapshot_manifest`). **(next increment)**

## Honesty
The sandbox has no NSE data and no network, so nothing real can be ingested here. Every new
component degrades to an explicit no-data/no-action state and is exercised by deterministic
fixtures — never by synthetic data presented as evidence.
