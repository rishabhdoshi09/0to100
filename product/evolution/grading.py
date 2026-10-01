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


def resolve_directional_underlying_forward_outcome(
    symbol: str,
    as_of: str,
    direction: str,
    entry: float,
    stop: float,
    target: float,
    *,
    horizon: int | None = None,
) -> dict[str, Any] | None:
    """Direction-aware first-touch resolver for F&O UNDERLYING shadows.

    Uses the same official bhavcopy source as core.outcome_resolver but
    supports SHORT geometry (stop above entry / target below entry). Return
    percentage is normalized so positive always means the policy's desired
    direction won, which lets the shared counterfactual taxonomy compare
    LONG and SHORT policies on one R sign convention.
    """
    from data.bhavcopy_store import get_ohlcv
    import pandas as pd

    direction = str(direction or "").upper()
    if direction not in {"LONG", "SHORT"}:
        return None
    try:
        entry_f, stop_f, target_f = float(entry), float(stop), float(target)
    except (TypeError, ValueError):
        return None
    if entry_f <= 0 or stop_f <= 0 or target_f <= 0:
        return None
    if direction == "LONG" and not (stop_f < entry_f < target_f):
        return None
    if direction == "SHORT" and not (target_f < entry_f < stop_f):
        return None

    try:
        df = get_ohlcv(str(symbol or "").upper())
    except Exception:
        return None
    if df is None or getattr(df, "empty", True):
        return None
    if not {"high", "low", "close"} <= set(df.columns):
        return None
    try:
        since = df[df.index >= pd.Timestamp(str(as_of or "")[:10])]
    except Exception:
        return None
    if since is None or getattr(since, "empty", True):
        return None
    try:
        highs = since["high"].to_numpy(dtype=float)
        lows = since["low"].to_numpy(dtype=float)
        closes = since["close"].to_numpy(dtype=float)
    except Exception:
        return None

    limit = int(horizon or PATH_HORIZON_SESSIONS)
    cap = min(len(highs), limit)
    filled = False
    for i in range(cap):
        if not filled:
            filled = highs[i] >= entry_f if direction == "LONG" else lows[i] <= entry_f
            if not filled:
                continue

        # Conservative ambiguity rule: adverse stop is checked first.
        if direction == "LONG":
            if lows[i] <= stop_f:
                exit_px, worked = stop_f, 0
                break
            if highs[i] >= target_f:
                exit_px, worked = target_f, 1
                break
        else:
            if highs[i] >= stop_f:
                exit_px, worked = stop_f, 0
                break
            if lows[i] <= target_f:
                exit_px, worked = target_f, 1
                break
    else:
        if not filled:
            return {"exit_price": 0.0, "forward_return_pct": 0.0, "worked": -1} if len(highs) >= limit else None
        if len(highs) < limit:
            return None
        exit_px = float(closes[cap - 1])
        worked = 1 if (
            exit_px >= entry_f if direction == "LONG" else exit_px <= entry_f
        ) else 0

    normalized_pct = (
        (float(exit_px) - entry_f) / entry_f * 100.0
        if direction == "LONG"
        else (entry_f - float(exit_px)) / entry_f * 100.0
    )
    return {
        "exit_price": float(exit_px),
        "forward_return_pct": normalized_pct,
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
    if str(row.get("grading_mode") or "") == "PAPER_FORWARD_CONTRACT_ONLY":
        # No trustworthy historical NSE option-chain path exists. Contract
        # shadows are graded only by record_observed_contract_shadow_outcome()
        # when a genuine PAPER-forward option observation is available.
        return None

    symbol = str(row.get("symbol") or "")
    as_of = str(row.get("as_of") or "")
    entry, stop, target = row.get("entry"), row.get("stop"), row.get("target")
    if not symbol or not as_of or entry is None or stop is None or target is None:
        return None

    if str(row.get("domain") or "") == "FNO_UNDERLYING":
        forward = resolve_directional_underlying_forward_outcome(
            symbol,
            as_of,
            str(row.get("direction") or ""),
            float(entry),
            float(stop),
            float(target),
            horizon=horizon,
        )
    else:
        forward = resolve_forward_outcome(
            symbol, as_of, float(entry), float(stop), float(target), horizon=horizon,
        )
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


def record_observed_contract_shadow_outcome(
    shadow_id: str,
    *,
    realized_R: float,
    evidence_class: str,
    observed_source: str,
    path: str | Path | None = None,
) -> dict[str, Any]:
    """Grade one F&O contract shadow from a genuine observed PAPER-forward
    option outcome. Historical/counterfactual option-chain evidence is
    deliberately rejected because QuantTerm has no trustworthy source for it.
    """
    from product.evidence_class import PAPER_FORWARD
    from product.evolution.shadow_decisions import get_shadow_decision, save_graded_decision
    from product import counterfactual_learning as CFL

    if str(evidence_class or "") != PAPER_FORWARD:
        raise ValueError("F&O contract Evolution outcomes must be PAPER_FORWARD; historical option evidence is forbidden")
    if not str(observed_source or "").strip():
        raise ValueError("observed_source is required for F&O contract shadow grading")
    row = get_shadow_decision(shadow_id, path=path)
    if row is None:
        raise KeyError(f"unknown contract shadow {shadow_id}")
    if str(row.get("grading_mode") or "") != "PAPER_FORWARD_CONTRACT_ONLY":
        raise ValueError("shadow is not an F&O contract-selection observation")
    if row.get("outcome") is not None:
        return row

    value = float(realized_R)
    updated = dict(row)
    updated.update({
        "outcome": "OBSERVED_PAPER_FORWARD_OPTION",
        "counterfactual_R": value,
        "classification": CFL.WINNER_TAKEN if value > 0 else CFL.LOSER_TAKEN,
        "graded_at": datetime.now(timezone.utc).isoformat(),
        "evidence_class": PAPER_FORWARD,
        "observed_source": str(observed_source),
        "not_pnl": True,
    })
    return save_graded_decision(updated, path=path)
