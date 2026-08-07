# Universe-history (survivorship & membership) policy

**Do not silently use current membership as historical truth.**

## Point-in-time universe

- `data/nse_universe.point_in_time_universe(as_of)` returns survivorship-aware
  membership **only** when `logs/universe_history.json` (listing/delisting dates) is
  supplied. Without it, it returns **today's survivors** with
  `survivorship_complete=False` and a note — and every consumer must treat that as
  survivorship-biased.
- The EXP-006 runner stamps `survivorship_complete` on every observation and in the
  snapshot manifest, and its verdict gate downgrades a would-be PASS accordingly.

## Sector membership

- Sector classification (`scan/sector_heat`) is **current**, not historically dated.
  Any sector-strength/breadth computed from it carries `SECTOR_MEMBERSHIP_NOT_PIT`.
- The framework never claims survivorship-safe historical sector breadth. Results that
  *depend* on the sector requirement are identifiable via that flag, and the strong-
  sector decision in `MINIMUM_DATASET_CONTRACT.md` governs whether a primary verdict is
  even attempted.

## Required artifacts to reach RESEARCH_GRADE universe

1. `universe_history.json` — per symbol `{symbol, listed, delisted?}` from NSE archives.
2. A symbol-change master (renames/mergers) — to keep identities stable across time.
3. (Ideally) a dated sector-membership history — otherwise sector stays a flagged limitation.

Until (1) exists, the universe is `survivorship_complete=False` and a PASS is not
attainable; a FAIL remains attainable (survivorship bias is one-directional favourable).
