# Sample manual / auditive filing chronology

**Question:** would a researcher querying each historical date see only what was publicly available then?

**Method:** `known_as_of(symbol, as_of)` on the ingested PIT ledger. Mid-sample cut `2022-12-31` vs full `2026-08-21`. Future leak = any row with `available_at` after the cut appearing in the earlier query.

All twelve names: **future_leak_at_2022-12-31 = 0**.

| Symbol | Bucket | Rows now | Events now | Known at 2022-12-31 | Filings after that date | First / last available | PIT-safe at cut? |
|---|---|---|---|---|---|---|---|
| RELIANCE | large / energy | 14 | 98 | 3 | 11 | 2019-05-06 / 2025-01-16 | yes |
| TCS | large / IT | 14 | 104 | 3 | 11 | 2019-04-23 / 2025-01-09 | yes |
| HDFCBANK | large / financial | 13 | 76 | 2 | 11 | 2019-04-23 / 2025-01-23 | yes |
| INFY | large / IT | 12 | 97 | 1 | 11 | 2022-10-13 / 2025-01-16 | yes |
| MARUTI | mid-large / auto | 13 | 84 | 2 | 11 | 2019-05-24 / 2025-01-29 | yes |
| PIDILITIND | mid / industrial | 14 | 102 | 3 | 11 | 2019-06-25 / 2025-01-23 | yes |
| TRENT | mid / consumer | 11 | 84 | 3 | 8 | 2019-05-06 / 2025-02-06 | yes |
| PERSISTENT | mid / IT | 14 | 99 | 3 | 11 | 2019-05-06 / 2025-01-22 | yes |
| ASTRAL | mid / industrial | 14 | 102 | 3 | 11 | 2019-06-25 / 2025-01-30 | yes |
| KAYNES | newly listed / industrial | 11 | 22 | 0 | 11 | 2023-01-31 / 2025-01-28 | yes (empty before listing-era filings) |
| TATAMOTORS | large / auto + CA | 0 | 0 | 0 | 0 | — | n/a — **missing from official results dump** |
| WIPRO | large / IT | 14 | 99 | 3 | 11 | 2019-04-23 / 2025-01-17 | yes |

## Notes

- INFY’s first ingested XBRL in the **bounded** set is 2022-10-13 even though result **events** exist earlier. Event timeline ≠ statement ledger.
- KAYNES has no fundamentals as of 2022-12-31 (listed Nov 2022; first ingested filing 2023-01-31). Correctly empty.
- TATAMOTORS does not appear in this host’s NSE `corporates-financial-results` raw JSON. Treat as an identity / filing-symbol hole (possible post-demerger vehicle). Do not invent rows.
- Consolidated vs standalone are labelled on each row. Do not compare them across quarters without checking `consol_basis`.

**Answer:** for every name that has rows, an as-of query does **not** leak later filings. Absence is returned as missing, not as a restated live website value.
