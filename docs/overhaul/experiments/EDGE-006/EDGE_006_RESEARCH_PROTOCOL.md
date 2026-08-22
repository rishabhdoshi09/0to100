# EDGE-006 — Research Protocol (frozen before backtests)

**Last primary EDGE in the autonomous five-experiment budget. Do not change L1 after later blocks.**

---

## Hypotheses

**H1 (primary).** Equal-weight of the 20 **highest** 20-session rupee-ADV names in the PIT investable universe earns **positive net excess** vs unconstrained EW investable after CNC costs.

**H2.** ADV deciles are positively related to next-month open-to-open return (D10 = most liquid).

Economic rationale: among names that already clear ₹50L ADV, the most liquid names may have less junk / better sponsorship. The flat 0.32% CNC model does **not** credit tighter spreads, so this is a *return* test, not a cost-advantage test.

This is **not** the min-turnover floor (already applied) and **not** an illiquidity-premium book.

---

## Frozen primary

| Knob | Value |
|---|---|
| Signal | **L1**: 20-session mean `close×volume` ending at T |
| Sort | Descending ADV |
| Book | Top 20, EW, long only |
| Rebalance | Monthly last session |
| Fill | Next open; no stop |
| Costs | CNC RT on one-way TO |
| Universe | Same PIT screen + bar on T |

Fail-closed if ADV is non-finite or ≤ 0.

---

## Comparators

- L0: Top 20 **lowest** ADV (illiquidity diagnostic)
- L1 Top 50
- Cadence 4-week / 2-month / quarterly on L1-Top20

Benchmarks: EW investable; Nifty-50 EW proxy. Same walk-forward blocks as EDGE-001–005.

---

## Failure → REJECT

- Net does not beat EW and does not beat Nifty in val+conf
- Deciles not ordered (Spearman < 0.20 or D10 does not beat D1)
- Confirmation reverses development
- Costs destroy a thin gross edge
- Only L0 (illiquids) works — that is a different hypothesis

## Labels

`PROMISING — FORWARD VALIDATION WARRANTED` / `RESEARCH-ONLY` / `MODIFY HYPOTHESIS` / `REJECT`

PROMISING requires the §17 robustness bar (CI excludes 0 or harness decisive; confirmation not economically flat).

None authorise paper, live, or FEATURE-002 changes.

After this experiment: write `docs/research_program/RESEARCH_PROGRAM_SYNTHESIS.md` and stop new strategy discovery (STOP A).
