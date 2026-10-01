"""Objective outcome grading for shadow decisions.

The Common Outcome Resolver principle: Champion and every Challenger are
graded against the SAME realized price path, resolved once from official
data -- no policy ever gets to define its own convenient outcome after the
fact. Reuses core.outcome_resolver.first_touch_path (official bhavcopy bars
only; returns None until the horizon has actually elapsed -- it never
fabricates a "future" outcome) and product.counterfactual_learning's
freeze-then-settle grading (now covering both taken and not-taken
decisions, see its WINNER_TAKEN/LOSER_TAKEN extension).

Grading never rewrites an already-settled row: once a shadow decision has
an `outcome`, this module treats it as immutable, mirroring the identity
freeze itself (a decision's evidence can't change after the fact; its
realized outcome can't either, once resolved).
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from core.outcome_resolver import PATH_HORIZON_SESSIONS, first_touch_path


def resolve_forward_outcome(
    symbol: str, as_of: str, entry: float, stop: float, target: float,
    *, horizon: int | None = None,
) -> dict[str, Any] | None:
    """None means NOT YET DUE (the horizon hasn't elapsed in official data) --
    never grade early, never fabricate a future bar."""
    result = first_touch_path(
        symbol, as_of, entry, stop, target,
        horizon=horizon or PATH_HORIZON_SESSIONS,
    )
    if result is None:
        return None
    exit_price, forward_return_pct, worked = result
    return {
        "exit_price": exit_price,
        "forward_return_pct": forward_return_pct,
        "worked": worked,
    }


def grade_shadow_decision(
    row: Mapping[str, Any], *, horizon: int | None = None,
    path: str | Path | None = None,
) -> dict[str, Any] | None:
    """Grade ONE shadow decision row in place (persisted). Returns None if
    the row is not yet due for grading (outcome horizon hasn't elapsed, or
    the row has no entry/stop/target to resolve a path against). Returns the
    row unchanged if it was already graded -- grading is idempotent and
    never re-settles a resolved outcome."""
    if row.get("outcome") is not None:
        return dict(row)

    symbol = str(row.get("symbol") or "")
    as_of = str(row.get("as_of") or "")
    entry, stop, target = row.get("entry"), row.get("stop"), row.get("target")
    if not symbol or not as_of or entry is None or stop is None or target is None:
        return None

    forward = resolve_forward_outcome(symbol, as_of, float(entry), float(stop), float(target), horizon=horizon)
    if forward is None:
        return None

    selected = str(row.get("decision") or "") == "ENTER_NOW"
    from product.counterfactual_learning import settle

    settled = settle(
        dict(row, hypothetical_entry=entry, hypothetical_stop=stop, hypothetical_target=target),
        forward_return_pct=forward["forward_return_pct"],
        selected=selected,
    )
    settled["graded_at"] = datetime.now(timezone.utc).isoformat()
    settled["resolved_exit_price"] = forward["exit_price"]
    settled["resolved_worked"] = forward["worked"]

    from product.evolution.shadow_decisions import save_graded_decision

    return save_graded_decision(settled, path=path)


def grade_pending_decisions(
    *, limit: int | None = None, horizon: int | None = None,
    path: str | Path | None = None,
) -> list[dict[str, Any]]:
    """Batch-grade every ungraded shadow decision whose horizon has elapsed.
    A broken/failed grading attempt for one row must never block the rest --
    this is off-market batch work, not the live Champion execution path."""
    from logger import get_logger
    from product.evolution.shadow_decisions import list_shadow_decisions

    log = get_logger(__name__)
    pending = list_shadow_decisions(ungraded_only=True, path=path)
    if limit is not None:
        pending = pending[:limit]

    graded: list[dict[str, Any]] = []
    for row in pending:
        try:
            result = grade_shadow_decision(row, horizon=horizon, path=path)
        except Exception as exc:
            log.warning("evolution_grading_failed", shadow_id=row.get("shadow_id"), error=str(exc))
            continue
        if result is not None and result.get("outcome") is not None:
            graded.append(result)
    return graded
