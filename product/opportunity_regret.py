"""Opportunity / regret engine.

After a market day: given what was knowable at decision time, how good was
capital allocation? Future prices are used only for evaluation. They never
rewrite the historical decision. Output feeds research hypotheses — it does
not automatically alter hard controls.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

SCHEMA_VERSION = 1

DISCOVERY_FAILURE = "DISCOVERY_FAILURE"
RANKING_FAILURE = "RANKING_FAILURE"
DD_FAILURE = "DD_FAILURE"
POLICY_FAILURE = "POLICY_FAILURE"
ENTRY_TIMING_FAILURE = "ENTRY_TIMING_FAILURE"
PORTFOLIO_ALLOCATION_FAILURE = "PORTFOLIO_ALLOCATION_FAILURE"
CORRECT_ABSTENTION = "CORRECT_ABSTENTION"
INCONCLUSIVE = "INCONCLUSIVE"

AFFECTS_HARD_CONTROLS = False


def _f(value: Any) -> float | None:
    try:
        if value is None or value == "":
            return None
        out = float(value)
        return out if out == out else None
    except (TypeError, ValueError):
        return None


def _ret(row: Mapping[str, Any]) -> float | None:
    for key in ("forward_return_pct", "realized_return_pct", "fwd_ret"):
        val = _f(row.get(key))
        if val is not None:
            return val
    return None


def classify_one(
    row: Mapping[str, Any],
    *,
    taken_symbols: set[str],
    taken_returns: list[float],
) -> dict[str, Any]:
    symbol = str(row.get("symbol") or "").upper()
    decision = str(row.get("decision") or row.get("group") or "").upper()
    reason = str(row.get("reason_code") or "")
    fwd = _ret(row)
    status = str(row.get("status") or decision)

    if fwd is None:
        failure = INCONCLUSIVE
        detail = "no forward return for evaluation"
    elif status in {"TAKEN", "ENTER_NOW"} or symbol in taken_symbols:
        failure = INCONCLUSIVE if abs(fwd) < 1.0 else (
            INCONCLUSIVE  # taken trades are outcomes, not allocation failures by themselves
        )
        detail = "taken trade evaluated separately"
        if fwd is not None:
            failure = INCONCLUSIVE
    elif reason in {"DD_GATE_FAILED", "DD_BLOCK"}:
        failure = CORRECT_ABSTENTION if fwd <= 0 else DD_FAILURE
        detail = "DD blocked; forward return used only to grade the abstention"
    elif reason in {"EVIDENCE_POLICY_BLOCK", "POLICY_FAILURE"}:
        failure = CORRECT_ABSTENTION if fwd <= 0 else POLICY_FAILURE
        detail = "policy blocked; forward return grades the policy"
    elif reason in {"ENTRY_TOO_EXTENDED", "WAIT_FOR_ENTRY", "WAIT"}:
        failure = ENTRY_TIMING_FAILURE if fwd >= 3.0 else (
            CORRECT_ABSTENTION if fwd <= 0 else INCONCLUSIVE
        )
        detail = "wait/chase gate; future used only to score timing"
    elif reason in {"NOT_SURFACED"}:
        failure = DISCOVERY_FAILURE if fwd >= 5.0 else (
            CORRECT_ABSTENTION if fwd <= 0 else INCONCLUSIVE
        )
        detail = "scan saw it but recommendation did not surface it"
    elif reason in {"NOT_TOP_OF_PORTFOLIO", "SECTOR_CAP", "CORRELATION_CAP", "MAX_PORTFOLIO_RISK"}:
        best_taken = max(taken_returns) if taken_returns else None
        if fwd is not None and best_taken is not None and fwd > best_taken + 2.0:
            failure = PORTFOLIO_ALLOCATION_FAILURE
            detail = "skipped name later beat the funded names"
        elif fwd is not None and fwd <= 0:
            failure = CORRECT_ABSTENTION
            detail = "portfolio skip avoided a loss"
        else:
            failure = INCONCLUSIVE
            detail = "portfolio skip; sample too close to call"
    elif reason in {"LOW_QUALITY_SETUP", "WATCH_TIER"}:
        failure = CORRECT_ABSTENTION if (fwd or 0) <= 0 else RANKING_FAILURE
        detail = "tier below auto-enter"
    else:
        failure = INCONCLUSIVE
        detail = "unclassified skip"

    return {
        "symbol": symbol,
        "decision_reason": reason,
        "decision_as_of": row.get("decision_as_of") or row.get("as_of") or "",
        "forward_return_pct": fwd,
        "failure_source": failure,
        "detail": detail,
        "rewrote_historical_decision": False,
        "affects_hard_controls": AFFECTS_HARD_CONTROLS,
    }


def evaluate_day(
    *,
    taken: Sequence[Mapping[str, Any]] | None = None,
    rejected: Sequence[Mapping[str, Any]] | None = None,
    waits: Sequence[Mapping[str, Any]] | None = None,
    not_surfaced: Sequence[Mapping[str, Any]] | None = None,
    competing: Sequence[Mapping[str, Any]] | None = None,
    as_of: str = "",
) -> dict[str, Any]:
    taken = list(taken or [])
    groups = list(rejected or []) + list(waits or []) + list(not_surfaced or []) + list(competing or [])
    taken_symbols = {str(r.get("symbol") or "").upper() for r in taken}
    taken_returns = [v for v in (_ret(r) for r in taken) if v is not None]
    rows = [classify_one(r, taken_symbols=taken_symbols, taken_returns=taken_returns) for r in groups]
    counts: dict[str, int] = {}
    for row in rows:
        counts[row["failure_source"]] = counts.get(row["failure_source"], 0) + 1
    hypotheses = []
    if counts.get(DISCOVERY_FAILURE):
        hypotheses.append("scan-to-recommendation coverage may be dropping winners")
    if counts.get(RANKING_FAILURE):
        hypotheses.append("tier/rank cut may be too harsh on later winners")
    if counts.get(DD_FAILURE):
        hypotheses.append("DD false-negative rate deserves a challenger, not an automatic gate change")
    if counts.get(POLICY_FAILURE):
        hypotheses.append("learned policy may be blocking good setups — keep as challenger evidence")
    if counts.get(ENTRY_TIMING_FAILURE):
        hypotheses.append("chase/wait timing may be leaving edge on the table")
    if counts.get(PORTFOLIO_ALLOCATION_FAILURE):
        hypotheses.append("portfolio concentration may be starving independent names")
    return {
        "schema_version": SCHEMA_VERSION,
        "as_of": as_of,
        "affects_hard_controls": AFFECTS_HARD_CONTROLS,
        "taken_n": len(taken),
        "evaluated_n": len(rows),
        "counts": counts,
        "rows": rows,
        "research_hypotheses": hypotheses,
        "live_locked": True,
    }
