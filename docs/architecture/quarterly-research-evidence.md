# Quarterly evidence briefs — read-only contract (first increment)

Owner: Decision / company due diligence. Canonical entry point is
StockResearchEngine / build_due_diligence. Do not start a second scanner,
a second filing warehouse, or a second decision authority.

## Why

The November 2026 user-supplied long-term tracker shows a useful research
presentation: price context, thesis, anti-thesis, quarter-specific figures,
material risk, and next falsifiable checkpoints. It is an unverified secondary
research artifact, not a certified NSE/BSE feed. Statements, calculated
percentages and exact source dates must be independently checked.

In the supplied example, CCL's displayed target ₹1,251 and CMP ₹970 imply
28.97% upside rather than the stated 25.9%; EMIL's ₹223/₹144 implies
54.86% rather than the stated 39.5%. Neither discrepancy proves the cited
buy/target prices are correct. It establishes arithmetic QA and source checks.

## Current implementation

- Existing stock-investigator GET returns a quarterly_brief projection.
- The projection reads only existing measured KPI records and risk flags.
- Each displayed finding carries KPI ID, source, publication label and period.
- Missing inputs stay missing. Future ISO-dated claims are withheld.
- Non-ISO source dates carry an unverified status, not a verified badge.
- Price QA recalculates both buy-to-CMP and CMP-to-target returns;
  reported numbers are advisory and discrepancies are retained.
- Source is never asserted to be independently verified by this projection.
- No effect on scanners, confidence, portfolio sizing, execution,
  Telegram permissions or live-money interlocks.
- The frontend renders these findings inside the existing Investigator.

## Following increments — gated

1. Source lineage: from the existing PIT warehouse, attach evidence IDs,
   document SHA, fiscal period, publication and available_from, revision.
2. Automated acquisition: use only the existing Research Data acquisition
   owner for changed filings; never start a new polling singleton. Bound
   requests, retries, staleness and last-known-good behavior.
3. Financial reconciler: consolidated/standalone isolation, quarter-versus-
   annual units, guidance versus actual, one-off PAT adjustments, comparability
   after corporate actions, ratio validation.
4. Hypothesis ledger: each positive/negative claim has measurable next event,
   threshold, source and expiration. UNKNOWN must never be a pass.
5. Long-term tracker: durable, versioned multi-symbol review with as-of
   snapshots and publication-triggered updates; keep restated history.
6. Evaluation: PIT-safe replay, counterfactuals and failure-rate testing;
   never mistake target-price gap for realized edge.
7. Release: official-source fixtures, parser QA, threat model, exact-head CI,
   actual Mac latency/recovery/storage and explicit release approval.

## Fail-closed rules

- PDF research notes are quoted secondary claims, not official exchange facts.
- A cached claim is not fresh merely because someone opened the app.
- Missing filing publication times cannot be used for historical replay.
- Valuation is unavailable without defensible shares, adjustments, source
  timestamps and units. A price target is not an intrinsic value.
- YoY quarter comparisons must use matching periods and consolidation scope.
- This projection cannot produce a BUY, SELL, probability, or broker order.
