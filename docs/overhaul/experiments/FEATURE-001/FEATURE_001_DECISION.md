# FEATURE-001 — Decision

**Claim class:** EXPLANATORY. Already-consumed 2020-09-28 → 2026-07-23 history.  
**Core SEPA:** `RETIRED_RESEARCH_BENCHMARK`.  
**Production:** unchanged. No paper, no live, no `VALIDATED_EDGE`.

Panel: 120,270 filled `UnifiedScanner` fires (identity calibration), 231,145 family rows, 302 sample dates, horizon 20, official bhavcopy. Full event jsonl is regenerable (`python -m research.feature001`) and is not stored in git.

## Final classification (exactly one each)

| Feature | Status |
|---|---|
| Trend (`trend_features_v1`) | FORWARD-VALIDATE AS RANK FEATURE |
| RS (`rs_features_v1` / `rs_cs_v1`) | FORWARD-VALIDATE AS RANK FEATURE |

Forward-validate means: **future dates only**, shadow rank comparison, no hard gate, no ticket change. This history cannot confirm the features.

## Per-family policy

| Family | n | Trend | RS |
|---|---:|---|---|
| BREAKOUT_52W | 323 | INSUFFICIENT_DATA | UNSTABLE |
| BREAKOUT_RES | 932 | RISK_FILTER_VALUE | RISK_FILTER_VALUE |
| GOLDEN_CROSS | 3753 | REDUNDANT | REDUNDANT |
| VOL_SQUEEZE | 13273 | POSITIVE_RANK_FEATURE | POSITIVE_RANK_FEATURE |
| VCP | 1760 | UNSTABLE | UNSTABLE |
| FLAT_BASE | 11833 | UNSTABLE | REDUNDANT |
| CUP_HANDLE | 34112 | POSITIVE_RANK_FEATURE | POSITIVE_RANK_FEATURE |
| HIGH_TIGHT_FLAG | 469 | INSUFFICIENT_DATA | INSUFFICIENT_DATA |
| ASC_TRIANGLE | 25941 | POSITIVE_RANK_FEATURE | POSITIVE_RANK_FEATURE |
| DOUBLE_BOTTOM | 8522 | UNSTABLE | REDUNDANT |
| PRE_BREAKOUT | 30232 | POSITIVE_RANK_FEATURE | POSITIVE_RANK_FEATURE |
| ACCUMULATION | 12093 | POSITIVE_RANK_FEATURE | POSITIVE_RANK_FEATURE |
| DELIVERY_SPIKE | 8369 | POSITIVE_RANK_FEATURE | POSITIVE_RANK_FEATURE |
| NR7_COIL | 9665 | UNSTABLE | POSITIVE_RANK_FEATURE |
| POCKET_PIVOT | 29964 | POSITIVE_RANK_FEATURE | POSITIVE_RANK_FEATURE |
| MOMENTUM | 14535 | POSITIVE_RANK_FEATURE | POSITIVE_RANK_FEATURE |
| PULLBACK_SUPPORT | 25369 | POSITIVE_RANK_FEATURE | POSITIVE_RANK_FEATURE |

`REDUNDANT` on GOLDEN_CROSS means the residual after `mom_score` is tiny. The raw A→B lift is still large (E[R] 0.030 → 0.287). Do not read REDUNDANT as “no association.”

## Answers

1. **Which strategies benefit from Trend?**  
   POSITIVE_RANK_FEATURE: `VOL_SQUEEZE`, `CUP_HANDLE`, `ASC_TRIANGLE`, `PRE_BREAKOUT`, `ACCUMULATION`, `DELIVERY_SPIKE`, `POCKET_PIVOT`, `MOMENTUM`, `PULLBACK_SUPPORT`.  
   RISK_FILTER_VALUE only: `BREAKOUT_RES`.  
   Flagship `BREAKOUT_52W` does **not** show a usable Trend lift (almost all fires are already strict Stage-2).

2. **Which benefit from RS?**  
   The same POSITIVE_RANK_FEATURE set, plus `NR7_COIL`. `BREAKOUT_RES` is RISK_FILTER_VALUE. `BREAKOUT_52W` is UNSTABLE.

3. **Which are harmed?**  
   No family classified NEGATIVE. Joint AND-gate **hurts** `BREAKOUT_52W` and `BREAKOUT_RES` (A 0.109 → D 0.081; A −0.003 → D −0.113). That is a frequency + selection cost, not a labelled NEGATIVE cell.

4. **Is Trend mainly alpha, ranking, or tail-risk?**  
   Mainly **ranking**, with a secondary tail effect. Within-day top−bottom E[R] for Trend = **0.141 [0.100, 0.187]**. Several families also cut bottom-decile rates. Mean-R lifts are modest except where the family is already a trend object (`VOL_SQUEEZE`, `GOLDEN_CROSS` raw).

5. **Is RS mainly alpha, ranking, or tail-risk?**  
   Mainly **ranking**. Within-day top−bottom E[R] for RS = **0.395 [0.319, 0.477]** — larger than production `score` (0.064, CI crosses ~0). Residual after production score remains 0.215. Do **not** move the RS 70 cutoff from these buckets.

6. **Are either redundant with existing momentum?**  
   Not as a duplicate of `mom_score`: corr(RS, mom_score)=0.16; corr(Trend, mom_score)=0.03.  
   They **overlap each other** (corr Trend↔RS = 0.57).  
   After mom residual Spearman is small (0.05 / 0.07) but after production `score` RS still adds (0.215). GOLDEN_CROSS is the one family where Trend/RS are labelled REDUNDANT vs mom.

7. **Does Trend+RS add incremental information?**  
   Sometimes, not as a universal AND-gate. `VOL_SQUEEZE` A 0.180 → D 0.332; `GOLDEN_CROSS` A 0.030 → D 0.393; `CUP_HANDLE` A 0.048 → D 0.097. Naive `score + RS/10` **dilutes** RS-alone ranking (0.095 vs 0.395). Do not ship an additive combo from this study.

8. **Which relationships are stable across years?**  
   RS high−low deltas stay positive in 2021–2026 for `VOL_SQUEEZE`, `PRE_BREAKOUT`, `ACCUMULATION`, `DELIVERY_SPIKE`, `POCKET_PIVOT`, `MOMENTUM`, `PULLBACK_SUPPORT`, `GOLDEN_CROSS`.  
   Trend is stable for `CUP_HANDLE`, `DELIVERY_SPIKE`, `PULLBACK_SUPPORT` (all seven years non-negative where measured).

9. **Which are development-era artifacts?**  
   Whole-tape E[R] flips negative in 2025–2026 (event baseline −0.11).  
   Trend deltas flip negative in 2025–2026 for `ASC_TRIANGLE`, `PRE_BREAKOUT`, `NR7_COIL`, `GOLDEN_CROSS` (2026).  
   `VCP` / `FLAT_BASE` year signs whip around — UNSTABLE.  
   2020 is thin (first sample date 2020-09-28) and several 2020 Trend deltas are the opposite sign of later years.

10. **Hard gate now?**  
    **No.** Neither feature becomes a production hard gate. An AND-gate would destroy useful frequency on breakouts that are already in Stage-2.

11. **Ranking feature now?**  
    **Not in production.** Status is forward-validate as a rank feature on **future** fires. Preferred research rank key is `rs_percentile`, then `n_structure_passed`. Do not replace `score` with an untested linear blend.

12. **Remain research-only?**  
    Core SEPA, VCP gates, buy-zone, and any “SEPA Ready” licence: research-only / deprecated semantics.  
    Trend and RS themselves: **forward-validate**, still not paper/live.

13. **Which SEPA production/UI semantics should be deprecated?**  
    `MEETS SEPA`, Ready copy that Stage-2 **is** Minervini SEPA for money, Ideas SEPA-as-eligibility, `sepa_score >= 40` as Core F, any autopilot plan requiring Core F. Desk copy now says Trend Quality / research setup context. See `FEATURE_001_SEPA_DEPRECATION.md`.

## Hypotheses (FDR q = 0.10)

| ID | Result | Note |
|---|---|---|
| H1 | FDR reject (pooled) | Heterogeneous; not a licence to gate all breakouts |
| H2 | FDR reject (pooled) | Stronger than H1; still family-specific |
| H3 | FDR reject | Tail rate down ~6.8pp for strict structure; RISK_FILTER_VALUE ≠ ALPHA |
| H4 | FDR reject, small after mom | Larger residual after production score than after mom_score |
| H5 | not rejected (no p) | Policy table shows material family differences |

## What this is not

- Not a Core F resurrection
- Not a VCP retune
- Not a new RS cutoff
- Not paper, not live, not VALIDATED_EDGE

## Forward observation

Shadow feature logging only, dates **strictly after** this freeze. **Not wired** into `app.py` / `auto_scan` in this milestone.

See `FEATURE_001_FORWARD_LEDGER.md`.
