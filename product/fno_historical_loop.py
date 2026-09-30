"""Checkpointed, incremental driver for the F&O historical walk-forward
counterfactual simulation (product.fno_historical_walkforward).

Mirrors the existing HISTORICAL_PAPER_CYCLE pattern (product
.historical_paper_loop / research.autonomy.jobs.run_historical_paper_cycle):
runs only while the cash market is closed, never touches execution or paper
positions, and carries a durable cursor so it can never silently repeat a
session it has already graded. Deliberately much simpler than that module's
batch/phase state machine -- there is no "wait for the future bar to
arrive" step here, because this walks PAST sessions where the settlement
bar already exists in the official bhavcopy store the moment the candidate
session does. One call processes a small, bounded number of new sessions
and returns; it is designed to be invoked repeatedly from a scheduled job,
never to run the whole history in one call.

Evidence weight discipline (see product.fno_historical_walkforward and
product.fno_evidence): every outcome this loop settles is
evidence_class=COUNTERFACTUAL. That evidence is inspectable and reported,
but product.conditional_evidence.ranking_evidence() -- all
product.decision_ranking.rank() ever reads -- refuses to let it move a
ranking score. Only a real settled PAPER_FORWARD trade (product
.fno_evidence.record_fno_settlement, fed by the live F&O paper cycle) can
do that. This loop cannot promote a setup into live-money authority: it
never touches product.live_safety / the broker-boundary interlock, and the
evidence class it writes is, by construction, one ranking_evidence() has
never treated as promotable.
"""
from __future__ import annotations

import json
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from core.runtime_paths import logs_dir
from product.fno_historical_walkforward import (
    classify_walk_forward_outcome,
    evaluate_point_in_time_candidate,
    record_walk_forward_outcome,
)

SCHEMA_VERSION = 1
DEFAULT_MAX_SESSIONS_PER_RUN = 5
DEFAULT_HORIZON_SESSIONS = 1
DEFAULT_LOOKBACK_DAYS = 20


def checkpoint_path(path: str | Path | None = None) -> Path:
    if path is not None:
        return Path(path)
    import os

    override = os.environ.get("QT_FNO_WALKFORWARD_CHECKPOINT")
    if override:
        return Path(override)
    return logs_dir() / "product" / "fno_historical_walkforward_checkpoint.json"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_checkpoint(path: str | Path | None = None) -> dict[str, Any]:
    target = checkpoint_path(path)
    if not target.exists():
        return {
            "schema_version": SCHEMA_VERSION,
            "cursor_date": "",
            "total_sessions_processed": 0,
            "total_candidates_evaluated": 0,
            "total_settled": 0,
            "last_run_at": "",
            "last_error": "",
            "last_error_at": "",
        }
    try:
        payload = json.loads(target.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {
            "schema_version": SCHEMA_VERSION,
            "cursor_date": "",
            "total_sessions_processed": 0,
            "total_candidates_evaluated": 0,
            "total_settled": 0,
            "last_run_at": "",
            "last_error": "corrupt checkpoint file replaced",
            "last_error_at": _now(),
        }
    if not isinstance(payload, dict):
        return {
            "schema_version": SCHEMA_VERSION,
            "cursor_date": "",
            "total_sessions_processed": 0,
            "total_candidates_evaluated": 0,
            "total_settled": 0,
            "last_run_at": "",
            "last_error": "",
            "last_error_at": "",
        }
    return payload


def _save_checkpoint(payload: Mapping[str, Any], path: str | Path | None = None) -> None:
    target = checkpoint_path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(target.suffix + ".tmp")
    tmp.write_text(json.dumps(dict(payload), indent=2, default=str), encoding="utf-8")
    tmp.replace(target)


def _default_universe() -> list[str]:
    """The current F&O underlying list. Never fabricated: any failure to
    load the real instrument master yields an empty universe, which
    run_next_batch already reports as NO_UNIVERSE rather than silently
    doing nothing."""
    try:
        from data.fno_universe import current_fno_universe

        return list(current_fno_universe().symbols())
    except Exception:
        return []


def _default_history_provider(symbol: str):
    """Real historical equity OHLC, column names normalised to what
    evaluate_point_in_time_candidate expects. Never fabricates a bar: a
    symbol with no store history returns None, exactly like
    data.bhavcopy_store.get_ohlcv itself."""
    from data.bhavcopy_store import get_ohlcv

    frame = get_ohlcv(symbol)
    if frame is None or frame.empty:
        return None
    rename = {"open": "Open", "high": "High", "low": "Low", "close": "Close", "volume": "Volume"}
    return frame.rename(columns={k: v for k, v in rename.items() if k in frame.columns})


def run_next_batch(
    *,
    universe: Sequence[str] | None = None,
    history_provider: Callable[[str], Any] | None = None,
    now_ist: datetime | None = None,
    max_sessions_per_run: int = DEFAULT_MAX_SESSIONS_PER_RUN,
    horizon_sessions: int = DEFAULT_HORIZON_SESSIONS,
    lookback_days: int = DEFAULT_LOOKBACK_DAYS,
    default_start_sessions_back: int = 260,
    path: str | Path | None = None,
) -> dict[str, Any]:
    """Process up to ``max_sessions_per_run`` new historical sessions for
    every symbol in ``universe``, settle what can honestly be settled, and
    advance the durable cursor. Safe to call repeatedly and often: it does
    real work only when there is a genuinely new session past the cursor
    with enough real forward history to settle against, and returns a
    cheap, honest "nothing new" result otherwise.
    """
    import pandas as pd

    checkpoint = load_checkpoint(path)
    resolved_universe = list(universe) if universe is not None else _default_universe()
    provider = history_provider or _default_history_provider

    if not resolved_universe:
        result = {
            "status": "NO_UNIVERSE",
            "sessions_processed": [],
            "candidates_evaluated": 0,
            "settled": 0,
            "checked_at": _now(),
        }
        return result

    # Build the union of available session dates across the universe from
    # real store history only -- never invent a calendar. One symbol's
    # provider raising (a corrupt cache file, a bad frame) must never cost
    # the whole batch its other, healthy symbols.
    frames: dict[str, Any] = {}
    all_dates: set = set()
    intake_errors: list[str] = []
    for symbol in resolved_universe:
        try:
            frame = provider(symbol)
            if frame is None or getattr(frame, "empty", True):
                continue
            frame_dates = {pd.Timestamp(ts).date() for ts in frame.index}
        except Exception as exc:
            intake_errors.append(f"{symbol}: history load failed: {type(exc).__name__}: {exc}")
            continue
        frames[symbol] = frame
        all_dates.update(frame_dates)

    if not all_dates:
        result = {
            "status": "NO_HISTORY",
            "sessions_processed": [],
            "candidates_evaluated": 0,
            "settled": 0,
            "checked_at": _now(),
        }
        return result

    sorted_dates = sorted(all_dates)
    last_available = sorted_dates[-1]
    # The last session we could possibly SETTLE is horizon_sessions before
    # the newest session on disk (settlement needs a real forward bar).
    if horizon_sessions > 0:
        # A plain negative-index slice (sorted_dates[:-horizon_sessions]) is
        # wrong once horizon_sessions >= len(sorted_dates): Python would
        # still return a large prefix instead of recognising that nothing
        # has enough forward history to settle against yet.
        settleable_count = max(0, len(sorted_dates) - horizon_sessions)
        settleable_dates = sorted_dates[:settleable_count]
    else:
        settleable_dates = sorted_dates
    if not settleable_dates:
        result = {
            "status": "AWAITING_FORWARD_BAR",
            "sessions_processed": [],
            "candidates_evaluated": 0,
            "settled": 0,
            "checked_at": _now(),
        }
        return result

    cursor_str = str(checkpoint.get("cursor_date") or "")
    if cursor_str:
        cursor_date = date.fromisoformat(cursor_str)
        candidate_dates = [d for d in settleable_dates if d > cursor_date]
    else:
        # First run ever: do not walk the entire store history in one call.
        # Start default_start_sessions_back sessions before the newest
        # settleable date so the very first batch is bounded too.
        start_index = max(0, len(settleable_dates) - default_start_sessions_back)
        candidate_dates = settleable_dates[start_index:]

    batch_dates = candidate_dates[:max_sessions_per_run]
    if not batch_dates:
        checkpoint["last_run_at"] = _now()
        checkpoint["last_error"] = ""
        _save_checkpoint(checkpoint, path)
        return {
            "status": "UP_TO_DATE",
            "sessions_processed": [],
            "candidates_evaluated": 0,
            "settled": 0,
            "cursor_date": cursor_str,
            "checked_at": _now(),
        }

    evaluated = 0
    settled = 0
    classification_counts: dict[str, int] = dict(checkpoint.get("classification_counts") or {})
    errors: list[str] = list(intake_errors)
    try:
        for as_of in batch_dates:
            for symbol, frame in frames.items():
                try:
                    candidate = evaluate_point_in_time_candidate(
                        symbol, as_of, frame, lookback_days=lookback_days,
                    )
                except Exception as exc:
                    errors.append(f"{symbol}@{as_of}: evaluate failed: {type(exc).__name__}: {exc}")
                    continue
                if candidate is None:
                    continue
                evaluated += 1
                forward_rows = frame[frame.index > pd.Timestamp(as_of)]
                if len(forward_rows) < horizon_sessions:
                    continue
                forward_row = forward_rows.iloc[horizon_sessions - 1]
                forward_close = float(forward_row["Close"])
                try:
                    update = record_walk_forward_outcome(
                        candidate,
                        forward_close=forward_close,
                        resolved_at=pd.Timestamp(forward_row.name).isoformat(),
                        path=None,
                    )
                except Exception as exc:
                    errors.append(f"{symbol}@{as_of}: settle failed: {type(exc).__name__}: {exc}")
                    continue
                if update is not None:
                    settled += 1
                # Grading every settled candidate (taken and not-taken alike)
                # against the same CORRECT_REJECTION/MISSED_WINNER/... taxonomy
                # the rest of the desk uses for rejected-candidate review is
                # separate from the R-multiple evidence cell above: it never
                # feeds ranking, it only makes "was this decision graded, and
                # was it graded correctly" honestly reportable in the UI.
                try:
                    classification = classify_walk_forward_outcome(
                        candidate, forward_close=forward_close,
                    )
                except Exception:
                    classification = ""
                if classification:
                    classification_counts[classification] = (
                        int(classification_counts.get(classification) or 0) + 1
                    )
    finally:
        checkpoint["cursor_date"] = batch_dates[-1].isoformat()
        checkpoint["total_sessions_processed"] = int(checkpoint.get("total_sessions_processed") or 0) + len(batch_dates)
        checkpoint["total_candidates_evaluated"] = int(checkpoint.get("total_candidates_evaluated") or 0) + evaluated
        checkpoint["total_settled"] = int(checkpoint.get("total_settled") or 0) + settled
        checkpoint["classification_counts"] = classification_counts
        checkpoint["last_run_at"] = _now()
        checkpoint["universe_size"] = len(resolved_universe)
        checkpoint["last_available_session"] = last_available.isoformat()
        if errors:
            checkpoint["last_error"] = "; ".join(errors[:5])
            checkpoint["last_error_at"] = _now()
        else:
            checkpoint["last_error"] = ""
        _save_checkpoint(checkpoint, path)

    return {
        "status": "OK" if not errors else "PARTIAL_ERROR",
        "sessions_processed": [d.isoformat() for d in batch_dates],
        "candidates_evaluated": evaluated,
        "settled": settled,
        "classification_counts": classification_counts,
        "cursor_date": checkpoint["cursor_date"],
        "remaining_sessions": max(0, len(candidate_dates) - len(batch_dates)),
        "errors": errors,
        "checked_at": _now(),
    }


def status(path: str | Path | None = None) -> dict[str, Any]:
    """Read-only projection for Home/System-health/F&O UI surfaces. Never
    computed from anything other than the durable checkpoint -- if nothing
    has ever run, this says so honestly rather than defaulting to zeros
    that look like a clean, evidence-checked history."""
    checkpoint = load_checkpoint(path)
    has_run = bool(checkpoint.get("last_run_at"))
    cursor = str(checkpoint.get("cursor_date") or "")
    last_available = str(checkpoint.get("last_available_session") or "")
    coverage_complete = bool(cursor) and bool(last_available) and cursor >= last_available
    return {
        "schema_version": SCHEMA_VERSION,
        "available": has_run,
        "last_run_at": checkpoint.get("last_run_at") or "",
        "cursor_date": cursor,
        "last_available_session": last_available,
        "coverage_complete": coverage_complete,
        "total_sessions_processed": int(checkpoint.get("total_sessions_processed") or 0),
        "total_candidates_evaluated": int(checkpoint.get("total_candidates_evaluated") or 0),
        "total_settled": int(checkpoint.get("total_settled") or 0),
        "classification_counts": dict(checkpoint.get("classification_counts") or {}),
        "universe_size": int(checkpoint.get("universe_size") or 0),
        "last_error": checkpoint.get("last_error") or "",
        "last_error_at": checkpoint.get("last_error_at") or "",
        "evidence_class": "COUNTERFACTUAL",
        "affects_ranking": False,
        "affects_live_money": False,
    }
