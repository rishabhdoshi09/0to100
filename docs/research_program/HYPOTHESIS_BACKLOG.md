# Hypothesis backlog

Scored for the autonomous mandate. Scores are 1–5 (5 = favourable).  
**Do not execute the whole list.** One primary EDGE at a time.

Independence notes matter: EDGE-001 consumed CS 12-1; FEATURE-001 consumed Trend/RS *on scanner fires*; EXP-NEXT consumed reversal / 20d L/S low-vol / vol-compression on a **29-name** panel.

| ID | Hypothesis | Econ | Indep | Data | PIT | TO | Retail | Redund | DoF | Falsify | Priority |
|---|---|---|---|---|---|---|---|---|---|---|---|
| H-LV | Low trailing realized vol names outperform high-vol names (long-only, full PIT universe, monthly) | 5 | 4 | 5 | 4 | 4 | 5 | 3 | 4 | 5 | **DONE EDGE-002 REJECT** |
| H-TS | Own-history trend inclusion (price>SMA200 and SMA200 rising) beats EW investable | 4 | 3 | 5 | 4 | 4 | 4 | 3 | 4 | 5 | **DONE EDGE-003 RESEARCH-ONLY** |
| H-OVN | Close-to-open vs open-to-close contribution after costs | 3 | 5 | 5 | 4 | 2 | 2 | 5 | 5 | 5 | postponed — daily CNC 0.32% RT is a priori fatal |
| H-RES | Residual CS momentum after market (and sector if map allows) | 4 | 2 | 5 | 3 | 3 | 4 | 2 | 3 | 4 | 4 — born from EDGE-001; consumed |
| H-91 | 9-1 CS momentum Top20 | 3 | 1 | 5 | 4 | 3 | 4 | 1 | 2 | 4 | 5 — EDGE-001 sensitivity; consumed |
| H-REV1M | 1-month CS reversal | 3 | 3 | 5 | 4 | 3 | 3 | 3 | 3 | 4 | **DONE EDGE-004 REJECT** |
| H-52W | Names nearest 52-week high outperform (George–Hwang) | 4 | 3 | 5 | 4 | 3 | 4 | 3 | 4 | 5 | **DONE EDGE-005 RESEARCH-ONLY** |
| H-LIQ | Highest 20d ADV names outperform (liquidity as CS quality rank) | 3 | 4 | 5 | 4 | 4 | 5 | 4 | 5 | 5 | **DONE EDGE-006 REJECT** |
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

## Selection for EDGE-006 (final budget slot)

**H-LIQ** — highest 20d rupee ADV among already-investable names.

- ADV is already computed for the liquidity floor and capacity flags; it has **never** been the ranker.
- Orthogonal to return, vol, SMA, losers, and 52w-high.
- Flat CNC cost model will not credit tighter spreads, so this tests whether liquid names have higher *returns*, not cheaper execution.
- H-BREADTH postponed: allocator (§25) and `%above-SMA200` was described in EDGE-003 (consumed threshold risk).

## Selection for EDGE-005

**H-52W** (George–Hwang proximity to 52-week high).

- Simple: one number (close / 252-session max), one rank, one book.
- Orthogonal to 12-1 *return* (EDGE-001), realized vol (EDGE-002), SMA200 inclusion (EDGE-003), and 21d losers (EDGE-004).
- Scanner already *demotes* laggards (>30% below 52w high) on fires. That is not a test of a standalone near-high Top20 book. FEATURE-001 did not isolate this as a portfolio.
- Independence is 3, not 5: the scanner quality gate consumed the *direction* of the idea on fires. Confirmation cannot be sold as a first look at “near highs are good.”
- H-BREADTH is an allocator (§25) and uses `%above-SMA200` already described in EDGE-003.

## Selection for EDGE-004

**H-REV1M** wins the remaining-budget rubric.

- Simple: one number (prior-month return), one rank, one long-only book.
- Orthogonal to EDGE-001 (short-horizon *reversal* vs 12-1 *continuation*), EDGE-002 (vol), and EDGE-003 (own-history trend inclusion).
- EXP-NEXT-01 FAIL was 1/3/5-day reversal on a **29-name** panel. That does not answer full-universe monthly Jegadeesh. It **does** consume the short-horizon reversal idea on that panel — EDGE-004 must say so.
- H-OVN is more independent but a tradable daily overnight CNC book is a priori dead (0.32% RT × ~250). Not a useful primary EDGE.
- H-BREADTH is an allocator. Mandate: edge discovery precedes allocation; EDGE-003 also consumed `%above-SMA200` variation.
- H-91 / H-RES remain consumed-history mutations of EDGE-001.

Not selected now: H-OVN (a priori cost), H-BREADTH (allocation / consumed), H-QUAL / H-ERN / H-SEC (no PIT data).

## Scoring key

Econ = published / economic rationale. Indep = not a silent retune. Data = we have the fields. PIT = causality. TO = likely survivable turnover. Retail = capacity. Redund = 5 means not redundant. DoF = few knobs. Falsify = clear kill rule.
