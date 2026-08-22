# Hypothesis backlog

Scored for the autonomous mandate. Scores are 1–5 (5 = favourable).  
**Do not execute the whole list.** One primary EDGE at a time.

Independence notes matter: EDGE-001 consumed CS 12-1; FEATURE-001 consumed Trend/RS *on scanner fires*; EXP-NEXT consumed reversal / 20d L/S low-vol / vol-compression on a **29-name** panel.

| ID | Hypothesis | Econ | Indep | Data | PIT | TO | Retail | Redund | DoF | Falsify | Priority |
|---|---|---|---|---|---|---|---|---|---|---|---|
| H-LV | Low trailing realized vol names outperform high-vol names (long-only, full PIT universe, monthly) | 5 | 4 | 5 | 4 | 4 | 5 | 3 | 4 | 5 | **DONE EDGE-002 REJECT** |
| H-TS | Own-history trend inclusion (price>SMA200 and SMA200 rising) beats EW investable | 4 | 3 | 5 | 4 | 4 | 4 | 3 | 4 | 5 | **1 — EDGE-003** |
| H-OVN | Close-to-open vs open-to-close contribution after costs | 3 | 5 | 5 | 4 | 2 | 2 | 5 | 5 | 5 | 3 (costs may kill daily) |
| H-RES | Residual CS momentum after market (and sector if map allows) | 4 | 2 | 5 | 3 | 3 | 4 | 2 | 3 | 4 | 4 — born from EDGE-001; consumed |
| H-91 | 9-1 CS momentum Top20 | 3 | 1 | 5 | 4 | 3 | 4 | 1 | 2 | 4 | 5 — EDGE-001 sensitivity; consumed |
| H-REV1M | 1-month CS reversal | 3 | 3 | 5 | 4 | 3 | 3 | 3 | 3 | 4 | 6 — EXP-NEXT-01 FAIL on short horizon; H3 said skip-month did not help |
| H-BREADTH | Breadth / %above-200 as a cash vs equity allocator | 3 | 4 | 4 | 4 | 5 | 4 | 3 | 3 | 4 | 7 |
| H-QUAL | Quality / profitability long-only | 4 | 5 | 1 | 1 | 4 | 4 | 4 | 3 | 3 | postponed — no PIT fundamentals |
| H-ERN | Post-earnings drift | 4 | 5 | 1 | 1 | 3 | 3 | 4 | 3 | 3 | postponed — no PIT earnings tape |
| H-SEC | Sector-momentum overlay / caps | 3 | 2 | 2 | 2 | 3 | 3 | 2 | 3 | 3 | postponed — sector map PIT_DEGRADED |

## Selection for EDGE-002

**H-LV** wins the rubric.

- Simple: one number (realized vol), one rank, one book.
- Orthogonal to EDGE-001 (risk anomaly vs relative-strength winners) and to FEATURE-001 (not a scanner rank).
- EXP-NEXT-02 was INCONCLUSIVE on 29 names, 20d vol, long-short, 21d hold. That does **not** answer whether a retail long-only low-vol book works on the full PIT universe. It **does** consume the economic idea on 2024–2026 for that tiny panel — EDGE-002 must say so and cannot claim pristine OOS.
- Data and PIT path already exist (`FastInvestable`, same-session print, next-open, `core.costs`).
- Expected turnover lower than 12-1 momentum.
- Clean reject: no monotonic vol deciles and/or no net excess vs EW after costs.

Not selected now:

- H-91 / H-RES: mutating EDGE-001 on consumed history.
- H-TS: reserved; Trend was already studied as a *feature on fires*.
- H-QUAL / H-ERN: fabricate-nothing rule.

## Scoring key

Econ = published / economic rationale. Indep = not a silent retune. Data = we have the fields. PIT = causality. TO = likely survivable turnover. Retail = capacity. Redund = 5 means not redundant. DoF = few knobs. Falsify = clear kill rule.
