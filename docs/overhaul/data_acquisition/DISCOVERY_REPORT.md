# Existing-data discovery report

**Question:** does a usable point-in-time NSE research dataset already exist anywhere
reachable from this project? **Answer: NO.** Machine-readable: `discovery_report.json`.

**Do not conclude "no data" from an empty local cache alone** — so every plausible
location was inspected, not just `logs/bhav`.

## Locations inspected

| Location | Result |
|---|---|
| `logs/bhav/` (bhavcopy cache) | **0 files** (absent/empty) |
| `logs/index*` | absent |
| `logs/universe_history.json` (survivorship) | **absent** |
| `logs/ca_events.json` (corporate actions) | **absent** |
| `data/fundamentals_cache.db` | present but **current snapshot only, no publication dates** |
| `/mnt/user-data` (mounted user dir) | **empty** |
| `/opt/rclone` (cloud tool mount) | read-only tool; **no remotes configured** |
| `models/` | ML model only; no market data |
| Env vars (`QT_*`/`BHAV`/`NSE`/`DATASET`/`*DIR`) | none point to external market data |
| CI artifacts (`.github/workflows`) | CI only runs pytest; stores no data |
| Broker-export / cloud object-storage config | none configured |
| Filesystem-wide search (`bhav`/`ohlcv`/`nifty`/`universe_history`/`ca_events`) | no matches (only pytest temp fixtures — synthetic) |

## Searched-for artifacts (all absent)

NSE daily OHLCV/bhavcopy · index history · historical symbol masters · IPO/delisting ·
corporate-action records · symbol-change mappings · dated sector classifications ·
delivery data · historical fundamentals with publication dates.

Note: **delivery %** *is* a field in the NSE bhavcopy file (`DELIV_PER`), so it comes
for free once bhav files exist — but no bhav files are present here.

## Network

- NSE archives (`nsearchives.nseindia.com`) → **HTTP 000** (unreachable; not in the
  environment's proxy allow-list).
- A bounded 45s `BhavDataProvider` build attempt **timed out** (blocks on the NSE fetch).

## Documented rebuild path (exists, needs network)

`data/bhavcopy_store.build_store()` (equities EOD) + `data/index_store.build_index_store()`
(Nifty/indices), both pulling official NSE archive CSVs.

## Conclusion

No existing research dataset, and none can be built in this environment (no NSE
network). Acquisition requires either network access to NSE archives **or** an
operator-supplied bhav store plus `universe_history.json` and `ca_events.json`. The
source decision and contracts in this directory define exactly what to acquire and how
it becomes research-grade.
