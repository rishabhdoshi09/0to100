# QuantTerm Research Program — Synthesis

**As of:** 2026-08-22  
**Stop condition:** **STOP A** (five new primary EDGE hypotheses exhausted) and **STOP F** (further CS strategy discovery on this bhav store is no longer the main bottleneck).

Production firewall unchanged. FEATURE-002 remains frozen and immature.

---

## Evidence table

| Hypothesis | Economic idea | PIT quality | Net result | Robustness | Verdict | Future status |
|---|---|---|---|---|---|---|
| Core SEPA | Stage-2 + VCP eligibility | PIT_DEGRADED | Confirmation Core F E[R] < 0 | Failed confirmation | RETIRE CORE; RETAIN FEATURES | Retired benchmark |
| FEATURE-001 Trend/RS | Rank features on scanner fires | Consumed history | Explanatory association | Not a standalone edge | FORWARD-VALIDATE AS RANK FEATURE | FEATURE-002 only |
| FEATURE-002 R1–R3 | Future shadow ranks | Future-only | 0 primary live_scan rows | Not mature | INSUFFICIENT NEW DATA | Wait; do not retune |
| EDGE-001 12-1 Top20 | CS momentum continuation | PIT_DEGRADED listing | +4.4% vs EW; 30% vs 25.6% | Conf −5.4%; CI includes 0 | RESEARCH-ONLY | Archived; no 9-1 rescue |
| EDGE-002 126d low-vol | Low-vol anomaly | PIT_DEGRADED listing | 6.4% vs EW 25.6% | Spearman −0.54 | REJECT | Archived |
| EDGE-003 T1 inclusion | Price>SMA200 & rising | PIT_DEGRADED listing | +1.27% vs EW | CI includes 0; harness INCONCLUSIVE; conf +0.17% | RESEARCH-ONLY | No inclusion shadow |
| EDGE-004 21d losers | 1-month reversal | PIT_DEGRADED listing | 12.6% vs EW 25.6% | Flat deciles; both tails lose | REJECT | Archived |
| EDGE-005 52w-high Top20 | George–Hwang proximity | PIT_DEGRADED listing | 19.5% vs EW 25.6% | Spearman 0.73; D10 < D9; harness REJECT | RESEARCH-ONLY | Slope ≠ book |
| EDGE-006 high ADV Top20 | Liquidity as quality | PIT_DEGRADED listing | 18.1% vs EW 25.6% | Spearman −0.87 | REJECT | Archived |
| EXP-NEXT-01/03 | Short-horizon / vol-compress | 29-name panel | FAIL | Tiny panel | FAIL | Closed |
| EXP-NEXT-02 | 20d L/S low-vol | 29-name panel | INCONCLUSIVE | Tiny panel | INCONCLUSIVE | Superseded by EDGE-002 |

New-EDGE budget used: **5/5** (EDGE-002 … EDGE-006). EDGE-001 and FEATURE-002 do not count.

---

## Answers

### 1. Which hypotheses genuinely survived?

None as a **tradable historical edge**.

What survived as *research objects*:

- FEATURE-001 / FEATURE-002 rank-feature program (unresolved; wait for future data).
- EDGE-001: a pooled 12-1 *decile slope*, not the Top20 book after confirmation.
- EDGE-003: inclusion *content* (mean share 54%; included beat excluded ~68 bps), not +1.3% vs EW.
- EDGE-005: a mid-high proximity slope (D9 > D1), not the nearest-high Top20 book.
- Scanner laggard demote: farthest-from-high and 21d losers both lose badly.

### 2. Which only looked good in development?

- EDGE-001: development and validation positive; **confirmation reversed**.
- EDGE-003: small positive in all blocks; confirmation economically flat.
- EDGE-004 / EDGE-005: confirmation *looked* better vs EW; development/validation did not. Do not rewrite those books as “2025-only.”

### 3. Which features repeatedly reduce adverse selection?

- **Avoiding laggards / last-month knives** (EDGE-004 REJECT, EDGE-005 LAG −20%, scanner demote).
- **Not buying the most extended near-high names as a concentrated Top20** (EDGE-005 D10 dip).
- Trend/RS as *within-set ranks* (FEATURE-001) — unvalidated forward.
- Low-vol and high-ADV as *positive* ranks did the opposite of helping.

### 4. Which edges are redundant?

12-1, SMA200 inclusion, and 52w-high proximity are one **continuation / strength** family. Testing more lookbacks of the same family is not independent research.

1-month reversal, low-vol, and high-ADV were independent and all failed as long-only Top20 books.

### 5. Which strategy families should be retired?

- Core SEPA / binary VCP / “SEPA Ready” (already).
- Long-only CS **Top20 monthly** on this store for: low-vol, 21d losers, high-ADV.
- Further silent mutations of 12-1 (9-1, stops, sector caps, residual-on-same-history).

### 6. Where does QuantTerm have the strongest evidence?

1. **Process and PIT hygiene** (same-session print, next-open, centralized CNC costs, walk-forward, red-team).
2. **EW investable universe** as the right benchmark (CAGR ~25.6% over 2020–2026). Beating Nifty is not enough.
3. **Rank-feature program** (FEATURE-001 → FEATURE-002), not a new book.
4. **Laggard avoidance** as a *demote*, not as a standalone long.

### 7. Is there evidence for an ensemble?

**No.** Mandate: at least two independently credible components. We have zero PROMISING books and one immature shadow. Combining RESEARCH-ONLY + REJECT does not create alpha.

### 8. Is more strategy research justified?

Not another CS monthly Top20 on official bhav. The five-experiment budget is exhausted. Remaining untested families need **data we do not have** (PIT fundamentals, earnings, official sector history) or are a priori non-viable (daily overnight CNC).

### 9. Is the bigger bottleneck data rather than strategy?

**Yes.** Usable PIT series: bhav OHLCV, inferred listing, degraded sector map, short official Nifty. Missing: PIT fundamentals, earnings tape, Nifty 500, long official index, survivorship-complete membership file. Strategy search on what remains has been run.

### 10. Should QuantTerm focus next on execution / portfolio construction instead?

**After FEATURE-002 matures — and not before a signal is validated.** Mandate §25: do not optimize allocation of unvalidated edges.

Recommended next *research domain* (not production):

1. Keep FEATURE-002 passive; write its decision report when gates hit.
2. Improve live_scan logging quality so the shadow can actually accumulate rows.
3. Do not change BUY, ranking, autopilot, GTT, Telegram, or sizing.
4. If a later mandate opens, prefer **data acquisition** (PIT fundamentals / earnings / official sectors) over another CS book.
5. Illiquidity premium (EDGE-006 L0 +1.35%) is a **new ID** if ever tested; it is consumed-history and not prioritized.

---

## Production restrictions (repeat)

This synthesis authorises **no** live, paper, FEATURE-002, or BUY changes.
