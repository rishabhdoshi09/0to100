# Data-source decision, licensing & reproducibility

No existing dataset was found (see `DISCOVERY_REPORT.md`). This selects **one** primary
source, reusing the repository's existing ingestion — **no competing ingestion system
is introduced** (per the milestone).

## Candidate sources

| Field | **NSE official archives (SELECTED)** | yfinance (`.NS`) | Licensed vendor (e.g. data provider) |
|---|---|---|---|
| Provider | NSE (`nsearchives.nseindia.com`) | Yahoo (unofficial) | Commercial |
| Status | Official, free EOD archive | Unofficial scrape | Licensed |
| History | ~ several years of daily EOD | ~15y+ | Vendor-defined |
| Symbols | Full NSE EQ series | Current `.NS` names | Vendor-defined |
| OHLCV fields | O/H/L/C/Volume **+ DELIV_PER (delivery)** | O/H/L/C/Volume | Full |
| Adjustment | **RAW / unadjusted** | Split-adjusted (differently) | Usually adjusted |
| Corporate actions | Separate NSE CA filings (must be added) | Baked-in, opaque | Provided |
| Delisted coverage | Via archive dates (needs mapping) | **Missing (survivorship)** | Usually included |
| Historical universe | Needs `universe_history.json` | Not supported | Often included |
| Index data | `ind_close_all` (Nifty etc.) | `^NSEI` | Provided |
| Sector data | Not dated in-repo | No | Sometimes dated |
| Delivery data | **Yes (in bhavcopy)** | No | Sometimes |
| Fundamental pub timestamps | No (separate) | No | Sometimes |
| Rate limits | Archive fetch, modest | Flaky/throttled | Contractual |
| Licensing / redistribution | EOD archive for own use; **do not redistribute raw** | Yahoo ToS — not for redistribution | Per contract |
| Reliability | High (official) | Low (flaky, changes) | High |
| Storage | Small (CSV/day, pickled) | Small | Varies |
| Reproducibility | High (stable official files, hashable) | **Low (values change under you)** | High |

## Decision

**Primary source = NSE official archives**, via the existing
`data/bhavcopy_store.py` (equities EOD, incl. delivery) + `data/index_store.py` (Nifty
benchmark). Rationale:

1. **Official & reproducible** — stable archive files that can be content-hashed; the
   opposite of yfinance, whose adjusted values silently change (the EXP-005 survivorship
   mirage came from yfinance).
2. **Already the repo's canonical path** — reusing it honours "do not implement multiple
   competing ingestion systems".
3. **Delivery data included** — `DELIV_PER` needs no extra source.

**yfinance is rejected for research** (classified `DISPLAY_ONLY` in
`DATA_CLASSIFICATION.md`): survivorship-biased, adjusted-differently, non-reproducible.
It may remain an *exploratory* source but can never yield a research PASS.

## Licensing / redistribution assessment

- NSE EOD archive data may be fetched for the project's own analysis. **Raw NSE files
  are NOT redistributed in this repository** (they are git-ignored under `logs/`). Only
  *derived, non-redistributable* provenance (hashes, counts, date ranges, verdicts) is
  committed under `docs/`. If wider redistribution is ever needed, obtain an appropriate
  data licence.
- yfinance/Yahoo data is **not** committed and **not** used for research verdicts.

## Reproducibility implications

Because the primary source is official archive files, a snapshot is reproducible from
`{source identity, date range, per-file content hashes, transformation version}`. The
same inputs must yield the same `snapshot_id` (see `STORAGE_AND_SNAPSHOT_DESIGN.md`).
