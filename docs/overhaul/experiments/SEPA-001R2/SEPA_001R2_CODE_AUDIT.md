# SEPA-001R2 code audit

Audit of SEPA-001R as landed on `cursor/sepa-001r-research-942f`, plus the
R2 corrections. SEPA-001 / SEPA-001R result files were not modified.

## P0 findings

| ID | Finding | Evidence | R2 treatment |
|----|---------|----------|--------------|
| P0-1 | End-of-sample universe | `universe_screen.load_research_frames` calls `screen_investable(..., as_of=None)` then ranks `_turnover(df)` on the **full** frame; `ablation_r` reuses that dict and `build_rs_table(..., universe=list(frames))` | `universe_pit.screen_investable_as_of` / `FastInvestable.snapshot` at every as-of |
| P0-2 | Top-100 called NSE RS | 001R `max_symbols=100` | Canonical `top_n=None`; 100/250/500 are sensitivity only |
| P0-3 | Earliest contractions kept | `vcp._contractions_from_swings` `break` at `max_contractions` | Collect all; `select_active_sequence` ends at the latest coil |
| P0-4 | Dead knobs | `min_recovery_bounce` never read; `swing_left/right` only on look-ahead `find_swings` | Documented; left in `SepaConfig` so `sepa-001r.v1` hashes stay stable; unused on R2 money path |
| P0-5 | Setup-id drift | `setup_id(symbol, window base_start)` | `PersistentSetupLedger` freezes `original_base_start` |
| P0-6 | Left-censor / retest | `EXTENDED` terminal; first snapshot already through pivot counted as miss | `LEFT_CENSORED` excluded; `PIVOT_RETEST` is not core F |
| P0-7 | ABFRL verify FAIL | Ledger 290 events; **ABFRL has 0 events** | Classify + quarantine; never infer a factor |
| P0-8 | 424 sessions | Local bhav 2024-12-24 → 2026-08-21 | Expand official `sec_bhavdata_full` as far as NSE serves |

## Config parameter table

| Parameter | Used in code? | File/function | Behaviour | Tests |
|-----------|---------------|---------------|-----------|-------|
| `min_contractions` | yes | `vcp._evaluate_structure` | TOO_FEW | 001 / 001r2 |
| `max_contractions` | yes | `select_active_sequence` (not early break) | window cap from the **end** | 001r2 6/8 coil tests |
| `min_reversal_pct` | yes | `causal_zigzag` | confirmation | 001r causality |
| `vcp_lookback` | yes | `detect_vcp` | rolling window | lifecycle rolling-id test |
| `depth_expand_tol` | yes | `_tightening_fail_reasons` | EXPANDING_PULLBACKS | planted widening |
| `final_vs_first` | yes | `_tightening_fail_reasons` | NOT_TIGHTENING | planted widening |
| `max_final_depth_pct` | yes | `_tightening_fail_reasons` | FINAL_CONTRACTION_LOOSE | planted |
| `max_base_depth_pct` | yes | `_evaluate_structure` | BASE_TOO_DEEP | planted deep |
| `volume_dry_up_max` / `required` | yes | `_evaluate_structure` | VOLUME_EXPANDING | planted expand vol |
| `near_pivot_frac` | diagnostic | `far_below_pivot` | not a VCP fail unless `fail_vcp_if_far_below_pivot` | 001R default False |
| `fail_vcp_if_far_below_pivot` | yes | `_evaluate_structure` | 001 True / 001R-R2 False | legacy vs new |
| `min_recovery_bounce` | **no** | config only | zigzag already requires a reversal | documented unused |
| `swing_left` / `swing_right` | **no** on money path | `find_swings` only | look-ahead fractal | 001r documents leak |
| `buy_zone_*` | yes | `entry.evaluate_entry` | hard zone | gap-through tests |
| `max_stop_pct` | yes | `evaluate_entry` | WIDE_STRUCTURAL_STOP | planted wide |
| `max_stop_atr` | diagnostic | `evidence_atr_wide` | not the stop | 001 |
| `rs_threshold` | yes | engine | canonical 70 | RS bucket study |
| `rs_horizons` / `weights` | yes | `rs.py` | `rs_cs_v1` | PIT RS tests |

## VCP lifecycle semantics (R2)

| Case | Label | Core F fill? | Opportunity stats? |
|------|-------|--------------|--------------------|
| A Research starts after the move | `LEFT_CENSORED` | no | excluded |
| B Observed ENTRY_READY then escapes | `EXTENDED` / `MISSED` / `GAP_THROUGH` | no | yes (refusal) |
| C Forming below pivot | track | no yet | not a miss |
| D Return to pivot after EXTENDED | `PIVOT_RETEST` | **no** | research variant only |

## Files not rewritten

Canonical 8/8 Stage-2 (`trend.py`), `rs_cs_v1`, hard buy-zone, structural stop,
fail-closed missing data, next-open fill classifier (zone before stop).
Live execution, paper, autopilot, broker, GTT, UI: untouched.
