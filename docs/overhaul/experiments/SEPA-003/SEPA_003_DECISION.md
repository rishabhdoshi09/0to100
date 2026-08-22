# SEPA-003 decision

**Revision:** SEPA-003  
**Prior sequence (immutable):** SEPA-001 → SEPA-001R → SEPA-001R2 → SEPA-001R2.1  
**Protocol:** `SEPA_003_RESEARCH_PROTOCOL.md` (committed before results)  
**Official core-F confirmation (R2.1, consumed):** n=975, E[R]=−0.122, REJECT  

This file does **not** re-open 2025–2026 as untouched OOS.  
Any new cut is a **NEW_HYPOTHESIS**, never `VALIDATED_EDGE`.  
Paper, live, broker, GTT, and autopilot are **not** authorised.

---

# A — RETIRE CORE SEPA; RETAIN SELECT FEATURES

Core F failed its predeclared confirmation. Reconstruction and matching
do not revive a standalone SEPA trade rule. Stage-2 structure and
cross-sectional RS remain the only components worth keeping as
**research features** (adverse-selection / ranking), with future
validation required. Binary VCP as a hard gate should be retired.

C is not selected: a bull-regime or tightness cut would still be a
NEW_HYPOTHESIS fitted on consumed years.  
D is not selected: Stage-2 and RS still consistently reduce how badly
the scanner baseline loses; that is not an edge, but it is not noise
either.

---

## Fourteen answers

1. **Strict Stage-2** — useful as a quality / loss-avoidance feature, not
   a standalone edge. Frozen R2.1: A −0.130R → B −0.019R. Confirmation
   both REJECT (A −0.303, B −0.069). “Loses less” is not an edge.

2. **RS** — weak ranking feature, not a hard ≥70 trade rule. On the G
   Stage-2+RS 20d panel (prespecified buckets): 70–79 +1.88% (n=8,657),
   80–89 +2.18% (n=6,506), 90–94 +2.19% (n=3,093), 95–99 +2.89%
   (n=2,484). Monotone but small. F R-buckets rise toward 95–99
   (+0.544R, n=382) but confirmation C is still REJECT. Do not retune
   the cutoff from this file.

3. **VCP as a binary gate** — **no incremental value**. R2.1 C +0.026R →
   D −0.017R. Matched VCP vs Stage-2+RS no-VCP (year × RS bucket ×
   regime, 20d %): 94 strata, mean diff **−0.34pp** (n_vcp=3,408,
   n_ctrl=7,932).

4. **VCP continuous features** — **UNSTABLE**. Final-contraction
   quartiles are not monotone; year high−low flips sign (2020 −0.69R,
   2026 +0.34R). FDR flagged a pooled H4 contrast; year stability
   overrides that. Tightness class: UNSTABLE.

5. **Pivot-entry geometry** — **UNSTABLE**. Explanatory only. No new
   buy-zone. H6 FDR reject on MAE does not survive year stability.

6. **Volume dry-up** — **UNSTABLE** / FDR not rejected (p=0.28).

7. **Contraction tightness** — does not carry a durable breakout signal
   as a gate or as a stable continuous score. See (3)–(4).

8. **Regime-conditional** — classifier now works (0 UNKNOWN on 3,432
   fills). BULL n=1,341 E[R]=+0.319; STRONG_BULL n=359 +0.281;
   SIDEWAYS n=1,346 +0.035; CORRECTION n=356 −0.139. BEAR n=30 +2.15
   is postcard-sized — do not claim a bear edge. H7 p=0.26 (not FDR
   rejected). Decay verdict **UNSTABLE_EDGE**. No regime gate added.
   A bull-only SEPA rule would be a NEW_HYPOTHESIS, not this decision.

9. **Sector leadership** — **INSUFFICIENT_PIT_SECTOR_DATA**. Mapped
   885 / 3,432 (25.8%). Identity is a current map, not a PIT archive.
   Leading vs weak groups are not interpretable (and the thin mapped
   slice even points the wrong way).

10. **Features that survive across time** — Trend Template as a quality
    screen; RS percentile as a ranking input. Nothing in the VCP / entry
    stack is ROBUST_POSITIVE.

11. **Development-era artifacts** — pooled F plus (R2.1 +0.123R;
    reconstructed +0.172R on 3,432 fills) is 2021/2023. 2022 is already
    negative inside the winning calendar era. Binary VCP as a profit
    engine. Treating daily scanner n as F n. BEAR +2R on 30 trades.

12. **QuantTerm should retain** — `trend_template` (7 structural rules)
    and `rs_cs_v1` percentile as research features / ranking inputs
    (`NEW_HYPOTHESIS`, future dates only).

13. **QuantTerm should retire** — core F as a standalone candidate;
    VCP binary as a hard BUY/eligibility gate; any plan to retune
    buy-zone / RS / stop on 2025–2026 and call it OOS.

14. **Further historical optimization of core F?** — **No.** The
    confirmation block is consumed. Fitting a new F against 2024–2026
    is in-sample theatre.

---

## Strategic conclusion

**A — RETIRE CORE SEPA; RETAIN SELECT FEATURES**

Stop. No paper integration.
