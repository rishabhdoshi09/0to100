# RESEARCH_GRADE Remediation Report

**Date:** 2026-08-11  
**Branch:** `cursor/institutional-ai-audit-80a2`  
**Phase B:** not started  
**Phase A.5 scientific rerun:** **not run** (gate still not earned)  
**Production trading behaviour:** **unchanged**

---

## In plain English

**Conclusion:** Research quality is still **Not ready** — but we now understand
*why*, and several real defects were fixed without inventing data.

**What improved:**
- Many “failed price adjustments” were not broken corporate actions at all —
  they were stocks that skipped trading for weeks/months.
- Official NSE split text that QuantTerm could not read before (for example
  `From Rs10/- … To Re 1/-`) is now parsed from the **official subject string**.
- Same-day bonus + split pairs (example: Bajaj Finance) now both apply.
- On the frozen Phase A.5 29-name panel, every consecutive-session corporate
  discontinuity we found is now **verified**.

**What still blocks certification:**
- A small set of consecutive-session jumps still lack an official parseable
  factor (notably a demerger without a published share ratio, some ETF unit
  splits, and a few names with no matching NSE CA subject).
- Dual-ISIN confirmation of historical renames remains low (old tickers are
  rarely still on today’s ISIN master) — renames are evidenced as PARTIAL from
  the official symbol-change file, not guessed.

**What this means for a normal user:**  
QuantTerm will not treat strategy tests on this full dataset as scientific
proof yet. It also will not invent missing corporate-action ratios just to
“make the score look green.”

---

## 1. Root cause of gap_rate ≈ 0.16

### How it was calculated (before remediation)

| Piece | Behaviour |
|-------|-----------|
| Detector | `core.data_integrity.phantom_gaps` |
| Definition of “gap” | Adjacent bars in a symbol’s series with \|close-to-close %\| ≥ **35** (`QT_INTEGRITY_GAP_PCT`, default 35) |
| Series tested | **Adjusted** prices via `get_ohlcv` → `adjust_frame` (raw store stays immutable) |
| Aggregate `gap_rate` | Share of **sampled symbols** with ≥1 such adjacent-bar move |
| Pass bar | `gap_rate ≤ 0.002` **and** CA events loaded |

### What that conflated

On a 150-symbol sample before remediation:

| Kind | Approx. share of “failures” |
|------|-----------------------------|
| Sparse / suspension spans (calendar gap ≫ 3 days between bars) | **majority** (~15/25 flagged adj gaps) |
| True consecutive-session CA still missing/mis-parsed | minority |
| Genuine one-day market moves ≥35% | rare under NSE circuits; not the driver |

So **0.16 was largely “large move frequency including suspensions”**, not
“unresolved unexplained structural discontinuities.”

### Correct quality measure (after R1/R6)

`verify_ca_adjustment` now reports:

**`unresolved_consecutive_session_symbol_rate`**

— symbols with ≥1 **UNRESOLVED** discontinuity on near-consecutive sessions
(calendar span ≤ 3 days). Suspension/sparse spans are classified
`SUSPENSION_OR_RELISTING` and do **not** alone fail the research metric.

Threshold **unchanged** (≤ 0.002). This is a **correctness fix**, not a
relaxation of PIT/CA/identity requirements.

---

## 2. Discontinuity classification

Classes used:

| Class | Meaning |
|-------|---------|
| `SUPPORTED_CA` | Official CA near ex-date; adjusted continuity restored |
| `GENUINE_MARKET_MOVE` | Large raw move but adjusted continuity OK |
| `SUSPENSION_OR_RELISTING` | Adjacent bars span >3 calendar days |
| `IDENTITY_TRANSITION` | Symbol-change evidence near the date |
| `DATA_ERROR` | Reserved for proven bad prints |
| `UNRESOLVED` | Consecutive-session structural jump without safe official factor |

### Sample after remediation (500 symbols)

| Class | Count |
|-------|------:|
| SUPPORTED_CA | 44 |
| SUSPENSION_OR_RELISTING | 53 |
| UNRESOLVED | 7 |
| others | 0 |

**Unresolved symbol rate:** ≈ **0.014** (was legacy ~0.16).

---

## 3. Verified CA events (high impact)

Parser fix: official NSE subjects like  
`Face Value Split (Sub-Division) - From Rs10/- Per Share To Re 1/- Per Share`  
now yield a factor from the **subject text** (not from prices).

Share-count events in ledger after re-ingest: **378** (was 183).

### Frozen Phase A.5 panel (29 names) — consecutive CA jumps

| Symbol | Classification | Status | Notes |
|--------|----------------|--------|-------|
| RELIANCE | SUPPORTED_CA | VERIFIED | Bonus 1:1 |
| BPCL | SUPPORTED_CA | VERIFIED | Bonus 1:1 |
| WIPRO | SUPPORTED_CA | VERIFIED | Bonus 1:1 |
| HDFCBANK | SUPPORTED_CA | VERIFIED | Bonus 1:1 |
| NESTLEIND | SUPPORTED_CA | VERIFIED | Split 10:1 + later Bonus 1:1 |
| DRREDDY | SUPPORTED_CA | VERIFIED | Split 5→1 |
| KOTAKBANK | SUPPORTED_CA | VERIFIED | Split 5→1 (was missing — parser) |
| BAJFINANCE | SUPPORTED_CA | VERIFIED | **Bonus 4:1 + Split 2→1 same day** (multi-event) |

**Phase A.5 unresolved consecutive CA list:** **empty**.

Raw prices remain immutable; adjustment is still on-read via
`adjust_frame` + policy `ca_sharecount_v1` (dividends not auto-applied).

---

## 4. Unresolved CA events (still blocking full-universe gate)

Ranked by research impact:

| Rank | Symbol | Why unresolved | Impact |
|------|--------|----------------|--------|
| 1 | *(none on Phase A.5 panel)* | — | Panel CA OK |
| 2 | ABFRL | Official subject **Demerger** with **no parseable ratio** | High liquidity; MISSING_SOURCE — must not invent factor from ~3× price hint |
| 3 | ALPL30IETF / AUTOIETF / BANKNIFTY1 / similar | ETF unit splits; no parseable EQ CA subject in API pull | Peripheral to equity panel; still pollute full-universe rate |
| 4 | AMIORG | No matching bonus/split subject in NSE CA pull for the jump date | Mid liquidity |
| 5 | CALSOFT, BIOFILCHEM | No official parseable CA; could be data/noise | Peripheral |

**Authoritative factors were never taken from `price_before/price_after`.**  
`ratio_hint` is investigative only.

---

## 5–6. Lineage coverage

| Item | Result |
|------|--------|
| Sources | EQUITY_L (ISIN+listing), symbolchange.csv, delisted.csv |
| Bug fixed | EQUITY_L headers had leading spaces → ISINs were dropped; now stripped |
| Transitions classified | 1045 |
| CONFIRMED (dual ISIN) | 0 (old symbols usually absent from today’s master) |
| PARTIAL (official rename evidence) | 1045 |
| CONFLICT / UNRESOLVED | 0 |
| Phase A.5 focus blocking | **none** |
| `symbol_lineage_complete` (no conflict/unresolved) | **True** |
| `isin_confirmed_rate` | **0.0** (honest — reported separately) |

User-facing:

> Stock history link — official rename notices are on file. Matching ISINs
> confirm some links; others remain evidence-backed renames without dual-ISIN
> closure.

---

## 7. Validator corrections

| Change | Rationale |
|--------|-----------|
| `verify_ca_adjustment` uses unresolved consecutive-session rate | Correctness: stop treating suspensions as CA failures |
| `discontinuity_audit` classifier + tests | Explicit SUPPORTED_CA / SUSPENSION / UNRESOLVED |
| NSE split subject parser hardened | Official text variants; not price inference |
| Multi-event same ex-date already multiplied in `adjust_frame` | BAJFINANCE bonus×split now both ingested |
| EQUITY_L key strip for ISIN | Identity evidence bugfix |
| Gate lineage check uses PARTIAL-ok / CONFLICT-fail | Matches R4 statuses; no guessing |

Regression tests: `tests/test_discontinuity_audit.py`, extended
`tests/test_nse_ca_ingest.py`.

**Not changed:** 35% magnitude threshold; 0.002 pass bar; no fabricated CA;
no yfinance; raw store immutable.

---

## 8. Frozen-protocol impact

| Question | Answer |
|----------|--------|
| Were Phase A.5 hypotheses/horizons/costs/criteria changed? | **No** |
| Did remediation silently shrink the frozen universe? | **No** |
| Would excluding ETFs/demergers to earn RESEARCH_GRADE be allowed by the frozen protocol? | **Not stated** → treating that as a **new exclusion** would be `ORIGINAL_PROTOCOL_BLOCKED` if used to force a PASS |
| Proposed path if needed later | Separately **versioned** protocol with explicit ETF/demerger exclusion — **not** silent rewrite of IDs `81b8889792f53113` … |

---

## 9. Final data trust class

| | Previous | After remediation |
|--|----------|-------------------|
| Trust class | `OPERATIONAL_ONLY` | **`OPERATIONAL_ONLY`** (unchanged class) |
| Legacy any-large-move rate | ≈ 0.16 | still high if measured the old way (suspensions) |
| Unresolved consecutive rate | *(not measured)* | ≈ **0.013–0.014** |
| Phase A.5 panel CA | several missing/partial | **all verified** |
| Lineage gate | fail (False flag / no ISIN) | **pass** (PARTIAL official, no conflict) |
| `evaluate_research_grade().earned` | False | **False** |

**RESEARCH_GRADE was not earned. No manual override.**

---

## 10. Remaining blockers (exact)

1. **Unresolved consecutive discontinuities** above the 0.002 bar on the
   full sampled equity store — especially:
   - ABFRL demerger without official ratio
   - ETF unit splits without parseable CA subjects
   - AMIORG / CALSOFT / BIOFILCHEM without matching official factors  
2. **ISIN-confirmed rename rate = 0** for historical old tickers (PARTIAL only).
   Not currently gating alone after evidence classification, but dual-ISIN
   closure is still incomplete as a scientific master.
3. Gauntlet `no_phantom_gaps` still fails because adjustment verify has not
   PASSed under the (correct) unresolved metric.

**STOP condition met for certification:** do not fabricate demerger/ETF
factors; do not use yfinance; do not weaken the standard.

---

## 11. Production behaviour confirmation

**No changes** to Brain, live ranking, risk limits, portfolio authority,
execution, broker interaction, trade sizing, or live signals.

Remediation touched research data integrity, CA parsing, identity ISIN
ingestion, and presentation helpers only.

---

## Artifacts

| Path | Role |
|------|------|
| `research/intelligence/data/discontinuity_audit.py` | Classifier + audit |
| `core/data_integrity.py` | Corrected verify metric |
| `data/nse_ca_ingest.py` | Official split-text parser fix |
| `data/security_identity.py` | ISIN header fix + lineage report |
| `logs/research_grade/discontinuity_audit.json` | Latest audit dump (gitignored) |
| `logs/research_grade/verify_ca.json` | Verify result |
| `logs/research_grade/lineage_coverage.json` | Lineage dump |
| `logs/research_grade/gate_after_remediation.json` | Gate dump |

---

## Request for Phase A.5 rerun?

**No — not requested.**  
Gate has not earned `RESEARCH_GRADE`. Frozen protocols remain frozen.
Await approval only after a genuine PASS.

**Do not begin Phase B.**
