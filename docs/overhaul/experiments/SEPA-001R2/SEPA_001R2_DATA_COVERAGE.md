# SEPA-001R2 data coverage

Official NSE `sec_bhavdata_full` CSVs via `data.bhavcopy_store`. No fabricated bars.

| Field | Value |
|---|---|
| CSV files on disk | 1788 |
| Store sessions | 1787 |
| Store symbols | 3156 (3054 with ≥80 bars) |
| First session | **2019-08-23** |
| Last session | **2026-08-21** |
| 2018 / early-2019 | NSE `sec_bhavdata_full` 404 — not invented |
| Warm-up | 252 sessions (~2020-09-16 first *indicator* date; canonical eval starts at **260** sessions so min_sessions and Stage-2 252 can both be known) |
| Post-warm-up calendar years | **2021, 2022, 2023, 2024, 2025** complete; 2020 remainder; 2026 through 21 Aug |

## Sessions by year (CSV)

| Year | Files |
|---|---|
| 2019 | 68 |
| 2020 | 261 |
| 2021 | 261 |
| 2022 | 259 |
| 2023 | 260 |
| 2024 | 261 |
| 2025 | 251 |
| 2026 | 167 |

Post-warm-up sample is **≥5 complete calendar years**. That meets the R2 minimum. It is not 8–10 years: 2018 files are not served on this endpoint.

One parse failure: `2022-08-08` CSV tokenizer error (skipped, not fabricated).

Corporate-action coverage is separate (`SEPA_001R2_CA_AUDIT.md`).
