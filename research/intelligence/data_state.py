"""
🧭 Data operating states (Phase 16) + evidence-eligibility tiers (Phase 19).

Data health is NOT one global Boolean. A system state governs automation, and an evidence tier
governs what claims a dataset may support. Both are surfaced in the UI and stamped on Evidence
Cards so nothing over-claims what the data can honestly support.
"""
from __future__ import annotations

# ── system-wide data state ───────────────────────────────────────────────────────
NO_DATA = "NO_DATA"
IMPORTING = "IMPORTING"
VALIDATING = "VALIDATING"
READY = "READY"
READY_WITH_WARNINGS = "READY_WITH_WARNINGS"
DEGRADED = "DEGRADED"            # some strategies/symbols unusable; isolate, no new risk there
STALE = "STALE"                 # data too old; block new entries, keep managing exits
CONFLICTED = "CONFLICTED"       # reconciliation conflict; block new risk
FAILED = "FAILED"

DATA_STATES = (NO_DATA, IMPORTING, VALIDATING, READY, READY_WITH_WARNINGS,
               DEGRADED, STALE, CONFLICTED, FAILED)

# ── research PIT-read vocabulary (facade; does NOT alter automation DATA_STATES) ─
# Reuses READY / DEGRADED / STALE string values. Extra statuses describe honest
# research-read outcomes that automation DATA_STATES never needed to name.
INCOMPLETE = "INCOMPLETE"         # required ledger/slice missing or partial
NOT_PIT_SAFE = "NOT_PIT_SAFE"     # source exists but is not point-in-time safe
BLOCKED = "BLOCKED"               # request refused (future as_of, no snapshot, …)

PIT_READ_STATES = (READY, DEGRADED, STALE, INCOMPLETE, NOT_PIT_SAFE, BLOCKED)


def allows_new_entries(state: str) -> bool:
    """Only genuinely healthy states may OPEN new positions."""
    return state in (READY, READY_WITH_WARNINGS)


def manages_positions(state: str) -> bool:
    """Whether existing positions should still be managed/exited (almost always yes)."""
    return state not in (FAILED,)


def data_ok(state: str) -> bool:
    return state in (READY, READY_WITH_WARNINGS, DEGRADED, STALE)


# ── evidence-eligibility tier ─────────────────────────────────────────────────────
OPERATIONAL_ONLY = "OPERATIONAL_ONLY"       # UI inspection only; no research/forward claims
LIMITED_RESEARCH = "LIMITED_RESEARCH"       # exploratory only, with loud warnings
RESEARCH_ELIGIBLE = "RESEARCH_ELIGIBLE"     # meets historical-validation standards
FORWARD_ELIGIBLE = "FORWARD_ELIGIBLE"       # may drive the automated paper loop for current dates

TIERS = (OPERATIONAL_ONLY, LIMITED_RESEARCH, RESEARCH_ELIGIBLE, FORWARD_ELIGIBLE)


def classify_tier(coverage: dict) -> tuple:
    """Return (tier, reasons). `coverage` fields (all optional, default worst-case):
      has_prices, has_benchmark, has_universe_history, adjustment_consistent,
      missing_session_rate (0..1), corporate_action_coverage (0..1), freshness_days,
      validation_errors (int).
    Conservative: anything missing downgrades — never upgrades on assumption."""
    c = coverage or {}
    reasons = []
    if not c.get("has_prices"):
        return OPERATIONAL_ONLY, ["no validated price history"]
    if int(c.get("validation_errors", 0)) > 0:
        reasons.append("unresolved validation errors")
        return OPERATIONAL_ONLY, reasons
    if not c.get("adjustment_consistent", False):
        reasons.append("inconsistent split/bonus adjustment")
    if c.get("missing_session_rate", 1.0) > 0.05:
        reasons.append("missing-session rate above 5%")
    if reasons:
        return LIMITED_RESEARCH, reasons

    research_ok = (c.get("has_benchmark") and c.get("has_universe_history")
                   and c.get("corporate_action_coverage", 0.0) >= 0.9)
    if not research_ok:
        if not c.get("has_benchmark"):
            reasons.append("no benchmark history")
        if not c.get("has_universe_history"):
            reasons.append("no point-in-time universe (survivorship risk)")
        if c.get("corporate_action_coverage", 0.0) < 0.9:
            reasons.append("incomplete corporate-action coverage")
        return LIMITED_RESEARCH, reasons

    # forward-eligible additionally needs FRESH data
    if c.get("freshness_days", 999) > 5:
        return RESEARCH_ELIGIBLE, ["data not fresh enough for live forward-testing"]
    return FORWARD_ELIGIBLE, []


def forward_eligible(tier: str) -> bool:
    return tier == FORWARD_ELIGIBLE
