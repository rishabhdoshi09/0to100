# Minimum acceptable dataset contract (first real EXP-006 run)

The first real EXP-006 run requires **at least** the following. Each item lists what
provides it and its current status.

| # | Requirement | Provided by | Status |
|---|---|---|---|
| 1 | Daily NSE equity OHLCV | `data/bhavcopy_store` (`sec_bhavdata_full`) | ready-when-fetched |
| 2 | Nifty benchmark history | `data/index_store` (`ind_close_all` → `^NSEI`) | ready-when-fetched |
| 3 | Stable symbol & session identities | bhav store keyed by symbol + trading date | present (current-symbol keyed) |
| 4 | IPO / first-observed-session handling | first finite bar per symbol (runner treats leading gaps as pre-listing) | implemented in runner/detector |
| 5 | Delisting / terminal-history handling | trailing gaps → no fill; data-quality counts terminal history | implemented in runner |
| 6 | Documented corporate-action policy | `CORPORATE_ACTION_POLICY.md` + `data/corporate_actions` | policy defined; needs `ca_events.json` for RESEARCH_GRADE |
| 7 | Deterministic local materialisation | bhav pickle cache + index store | implemented |
| 8 | Data-quality validation | `dataset.data_quality_report()` (fails closed) | implemented + tested |
| 9 | Stable content hashes & snapshot identity | `dataset.snapshot_manifest()` + `STORAGE_AND_SNAPSHOT_DESIGN.md` | snapshot id implemented; per-file raw hashes = planned |
| 10 | Chronological point-in-time retrieval | runner scans bar-by-bar; PIT primitives read only `≤ i` | implemented + tested |

## Optional (mark unavailable; never fabricate)

- **Historical valuation / fundamentals** — no publication-dated source in-repo →
  `VALUATION_DATA_UNAVAILABLE`; current fundamentals are NEVER substituted.
- **Delivery data** — available from bhavcopy (`DELIV_PER`); if absent it stays MISSING,
  never 0.
- **Fully dated sector membership** — not available → `SECTOR_MEMBERSHIP_NOT_PIT`
  (see `UNIVERSE_HISTORY_POLICY.md`).

## Strong-sector membership decision (frozen before running)

The primary EXP-006 hypothesis requires **positive sector strength**. If dated sector
membership cannot be reconstructed reliably, the choice is registered **before** any
result — one of:

1. **Primary hypothesis stays BLOCKED** (default; no verdict) — chosen when sector
   strength materially drives eligibility and cannot be supported.
2. **Limitation-compatible run** — allowed ONLY if the sector requirement can be met
   from PIT price series of *current* members with the limitation explicitly flagged
   (`SECTOR_MEMBERSHIP_NOT_PIT`), AND the verdict gate treats the resulting bias
   honestly (a would-be PASS is downgraded; a FAIL is retained only if limitations are
   one-directional favourable — see the runner's verdict gate).
3. **New experiment ID** — required if the sector definition is *changed* (that is a
   material change to the frozen experiment, not a limitation).

**The frozen experiment is never silently weakened.** Until this decision is recorded
for a real dataset, the run stays blocked.
