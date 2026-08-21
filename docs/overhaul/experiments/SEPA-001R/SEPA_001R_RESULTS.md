# SEPA-001R Results

**Eligibility version:** `sepa-001r.v1`  
**Pivot:** `pivot_last_contraction_v1`  
**VCP:** `vcp_causal_v1`  
**Generated:** 2026-08-21  
**Data:** official NSE bhavcopy, 424 sessions through 2026-08-21, CA on-read  
**PIT class:** `PIT_DEGRADED` (`ca_complete=false`)  
**Live execution:** not wired

SEPA-001 remains immutable in `docs/overhaul/experiments/SEPA-001/`.

Primary run: 100 liquid names, **daily** (`sample_step=1`), lookback 220, horizon 20, CNC costs, unique-setup embargo on E/F.  
944 unique VCP setups. Payload: `ablation_001r.json`.

## Main comparison

| Variant | Unique Setups | Trades | Expectancy R | PF | Win % | Avg Win | Avg Loss | Max DD R | +1R % | Fail-Break % | CI | Verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| A Baseline scanner | 1207 | 1463 | −0.248 | 0.66 | 38.9 | +1.23 | −1.19 | −430 | 39.8 | 47.7 | [−0.52, −0.06] | REJECT |
| B + Stage-2 structure | 432 | 483 | +0.107 | 1.22 | 44.5 | +1.33 | −0.87 | −24.6 | 42.4 | 40.8 | [−0.05, +0.24] | harness PROMOTE* |
| C + RS ≥ 70 | 267 | 291 | +0.128 | 1.26 | 45.7 | +1.35 | −0.90 | −10.9 | 48.5 | 42.3 | [−0.03, +0.27] | harness PROMOTE* |
| D + causal VCP (scanner fills) | 58 | 62 | +0.132 | 1.26 | 46.8 | +1.35 | −0.94 | −5.3 | 53.2 | 43.6 | [−0.11, +0.37] | UNDERPOWERED |
| E D + buy-zone + structural stop | 2 | 2 | −1.086 | — | 0.0 | — | −1.09 | −1.09 | 50.0 | 100 | [−1.09, −1.08] | UNDERPOWERED |
| F Core SEPA, no scanner required | 4 | 4 | −0.617 | 0.24 | 25.0 | +0.78 | −1.08 | −1.39 | 50.0 | 75.0 | [−1.09, −0.15] | UNDERPOWERED |

\*Harness PROMOTE on B/C is **not** a paper licence: PIT_DEGRADED, block-bootstrap CI **includes zero / is negative**, 2025–2026 only, daily scanner over-sampling vs SEPA-001's step-10.

### Trades / year (252-session convention on 262 as-of dates)

| Variant | Trades/year | MAE R | MFE R | Avg hold |
|---|---|---|---|---|
| A | 1407 | 1.16 | 1.82 | 7.6 |
| B | 465 | 0.83 | 1.05 | 9.1 |
| C | 280 | 0.83 | 1.15 | 9.5 |
| D | 60 | 0.88 | 1.19 | 11.4 |
| E | 1.9 | 1.34 | 1.09 | 7.5 |
| F | 3.9 | 1.29 | 1.33 | 9.5 |

E fill attempts: 126 `EXTENDED`, 2 `VALID_FILL`, 1 `GAP_THROUGH`.  
F fill attempts: 4 `VALID_FILL`, 3 `EXTENDED`, 1 `MISSED`, 1 `GAP_THROUGH`.

## Year breakdown

| Variant | 2025 n | 2025 E[R] | 2026 n | 2026 E[R] |
|---|---|---|---|---|
| A | 588 | −0.36 | 875 | −0.17 |
| B | 146 | +0.18 | 337 | +0.08 |
| C | 28 | −0.04 | 263 | +0.15 |
| D | 5 | −0.90 | 57 | +0.22 |
| E | 0 | — | 2 | −1.09 |
| F | 0 | — | 4 | −0.62 |

Walk-forward (train 2025 → test 2026): core E/F have **no 2025 fills**. C flips from negative (thin) to positive. Not a stable unseen-block confirmation for core SEPA.

## Regime

All fills bucketed `unknown` — the Nifty regime series did not align onto as-of dates in this run. **No new regime gate.**

## Sector (C, names that mapped)

Positive pockets (n≥8): Capital Markets +0.48, Pharma +0.71, Telecom +0.70.  
Negative pockets: NBFC −0.73, Banking −0.52.  
**163 / 291 C trades are `UNKNOWN` sector** (map is not PIT). Do not add a sector leadership gate from this.

## Parameter sensitivity

### Buy-zone (F, 40-name subset, step=2)

See `SEPA_001R_BUYZONE.md`. Widths 0.25–1.0%: **0 fills**. 1.5–3%: 1 loser. 5%: 2 losers. **Do not widen the zone to manufacture trades.**

### RS percentile buckets (20-session forward %, **no VCP/entry gates**)

| Bucket | n | Mean 20d % | Median 20d % |
|---|---|---|---|
| 50–69 | 298 | +1.72 | +1.04 |
| 70–79 | 143 | +2.96 | +2.00 |
| 80–89 | 148 | +3.62 | +2.53 |
| 90–94 | 77 | +4.75 | +3.54 |
| 95–99 | 55 | +9.80 | +7.19 |

Monotone in this book. **Do not adopt RS≥90** from the C-threshold postcard (n=41 on the subset). Canonical threshold stays **70**.

### RS threshold on C / F (40-name subset)

| RS ≥ | C n | C E[R] | F n | F E[R] |
|---|---|---|---|---|
| 70 | 97 | +0.42 | 1 | −1.08 |
| 80 | 81 | +0.33 | 1 | −1.08 |
| 90 | 41 | +0.63 | 1 | −1.08 |

### VCP components (F, same subset)

Removing volume dry-up: 2 fills, still −1.09R. Tightening max base depth to 25%: 0 fills. Contraction count 2 vs 3: same single loser. **Sample too small to attribute signal to a VCP knob.**

### Pivot definition (F, subset)

`last_contraction` vs `pattern_high`: both n=1, −1.08R on this subset. Last-contraction is retained for **structural** reasons (coil resistance), not max R. Full-book unique setups still use last-contraction.

## Sample-quality warnings

- `PIT_DEGRADED`: CA verify FAIL; membership is `bhav_inferred`
- Core E/F n={2,4} — a handful of trades, CI entirely negative
- Daily A over-samples the scanner (1463 trades / ~1 year) vs SEPA-001 step-10; A's sign flipped to negative
- 2025–2026 only; F has no 2025 trade
- Sector and regime incomplete
- 100-name liquidity cap (2660 → 100 after screen)

**Core SEPA (E/F) is not evidence-supported on this retest.**
