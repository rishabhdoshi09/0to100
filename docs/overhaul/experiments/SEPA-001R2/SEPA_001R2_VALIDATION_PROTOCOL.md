# SEPA-001R2 validation protocol

**Status:** Predeclared. Written **before** the canonical R2.1 long-history
run and **before** looking at final A–G performance.

**Experiment:** SEPA-001R2.1 (runner-integrity correction of SEPA-001R2)  
**Eligibility / VCP / pivot (frozen):** `sepa-001r2.v1` / `vcp_causal_v2` /
`pivot_last_contraction_v1`  
**This document does not change any strategy threshold.**

SEPA-001, SEPA-001R, and existing SEPA-001R2 audit files remain immutable
evidence. This protocol only freezes how R2.1 **will be scored**.

---

## 1. Why a protocol exists

Canonical SEPA parameters are already frozen (buy-zone, RS≥70, Stage-2
template, VCP contractions, structural stop, next-open fill). The remaining
risk is **evaluation leakage**: choosing a walk-forward split, a sampling
cadence, or a CA treatment after seeing expectancy.

This file locks the split, the statistical units, and the deployment rule
**now**. After the run, `SEPA_001R2_WALK_FORWARD.md` and
`SEPA_001R2_DECISION.md` may only **report** these blocks. They may not
retune RS, VCP, Stage-2, stop, or the buy-zone because a block looked weak.

---

## 2. Calendar blocks (fixed)

Official NSE history on disk at freeze: **2019-08-23 → 2026-08-21**.

| Block | Inclusive dates | Role |
|---|---|---|
| Warm-up | 2019-08-23 through the first date with 252 completed sessions | Indicators only. **Not** a performance sample. |
| Development / descriptive | first eligible evaluation date through **2023-12-31** | Describe behaviour. May inform **narrative**, never thresholds. |
| Validation | **2024-01-01** through **2024-12-31** | Confirmatory for the already-frozen spec. Not a licence to retune. |
| Untouched confirmation | **2025-01-01** through **2026-08-21** | The only block that may satisfy `has_unseen_block` for deployment. |

Assignment is by **signal `as_of` date** (the decision session), not by exit
date. A trade decided on 2023-12-29 whose exit falls in 2024 stays in
development.

**Why this split (declared before results):**

- It is a **calendar** cut, not a performance-adaptive cut.
- Development covers multiple post-warm-up years (2020–2023) so Stage-2/RS/VCP
  have a descriptive sample after the 252-session lookback exists.
- Validation is a full calendar year that the authors have not used to change
  SEPA thresholds in this milestone.
- Confirmation is the later, still-unseen (for threshold decisions) window
  through the last official bar. Deployment **must** use this block’s
  evidence. `has_unseen_block=False` hardcoded forever is forbidden.

If the first eligible date is after 2023-12-31 (should not happen with 2019
history), development would be empty and the run must abort rather than
silently fold validation into development.

No other split is used. If a future experiment needs a different cut, it
must write a new protocol **before** seeing that experiment’s results.

---

## 3. Observation semantics (not optional)

Canonical A–G comparison uses **every exchange session** after warm-up
until the last date that still has a full outcome horizon:

- `date_step = 1`
- scanner evaluation **daily** (`scanner_step = 1`)
- Variant E **observes** setup lifecycle daily; the scanner gate only
  decides whether that day is an **E opportunity / fill attempt**

Five-session sampling is **not** a canonical observation. A step-5
counterfactual may be counted as a diagnostic (how many E entry-ready
days would have been missed) but must not replace the primary ladder.

---

## 4. Statistical units

| Variants | Raw unit | Deduplicated unit | Must not confuse |
|---|---|---|---|
| A, B, C, D | signal-day (scanner row that passes that variant’s gates) | same symbol embargoed until the **actual exit session** | 1,500 daily A rows ≠ 50 unique F setups |
| G | Stage-2+RS signal-day with a next-open print | same symbol embargoed until the last **uncensored** forward session used in that row | G is **not** SEPA R |
| E, F | persistent setup identity (`PersistentSetupLedger`) | one core opportunity per continuing base | left-censored / pivot-retest are not core fills |

Expectancy and harness statistics for A–D/E/F use the **deduplicated**
unit. Raw signal-day tables are reported alongside for conditional
forward-return analysis.

G is a **pure signal study** (forward %, MAE %, MFE %, hit +5% / +10%).
It is not fed to an R-trade harness and is not called SEPA expectancy.

---

## 5. Corporate-action research rule (causal segments)

Unresolved discontinuities are enumerated exhaustively. Treatment is
**date/segment-aware**, never a static symbol blacklist applied to every
historical as-of.

- Before event date `D`: prior observations remain valid if lookback and
  forward outcome do not cross `D`.
- Forward path that crosses `D`: `CA_CENSORED_OUTCOME` — excluded from
  expectancy, counted in the funnel. No fabricated through-gap return.
- At/after `D`: indicators whose lookback crosses `D` are invalid. The
  name re-enters only after every required feature (including 252-session
  Stage-2/RS and VCP lookback) can be computed from **clean post-event**
  bars.

No adjustment factor is inferred. Global `ca_complete` remains whatever
`verify_ca_adjustment` says. A separate `ca_research_acceptable` flag may
be true only when the causal audit’s six conditions hold (see R2.1
spec). That flag does **not** rewrite the global verifier.

---

## 6. Deployment gate (paper shadow is still not live)

`has_unseen_block` is true only when the **confirmation block** contains
the required effective sample of **core F** (and only F is the deployment
candidate).

Paper shadow, if ever granted later, remains observation / fake-money.
This protocol does **not** authorise broker execution, GTT, live
autopilot, or paper-integration code.

Decision after the run must be exactly one of:

- `PROMOTE TO PAPER SHADOW`
- `KEEP RESEARCH-ONLY`
- `MODIFY AND RETEST`
- `REJECT CORE SEPA`

Thresholds will not be modified in response to validation or
confirmation numbers in this milestone.
