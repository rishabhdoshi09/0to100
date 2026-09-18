"""Closed-market historical virtual-paper learning loop.

This lane keeps QuantTerm productive when NSE cash is closed without pretending
historical simulation is real forward evidence.

One durable batch:
    point-in-time replay -> virtual long paper trades -> settled historical
    evidence -> learning -> research -> advance cursor.

The batch cursor advances only after research completes, so a restart cannot
silently skip the learning/research stages. Historical trades live in their own
ledger and never mutate the real forward paper book.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import threading
from concurrent.futures import Future, ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from core.runtime_paths import logs_path

SCHEMA_VERSION = 1
PHASE_IDLE = "IDLE"
PHASE_RUNNING = "RUNNING"
PHASE_AWAITING_LEARNING = "AWAITING_LEARNING"
PHASE_AWAITING_RESEARCH = "AWAITING_RESEARCH"
PHASE_FAILED = "FAILED"

DEFAULT_STATE = logs_path("product/historical_paper_loop.json")
DEFAULT_LEDGER = logs_path("product/historical_paper_trades.jsonl")
DEFAULT_MEMORY = logs_path("product/historical_paper_memory.json")

DEFAULT_BATCH_SIZE = 8
DEFAULT_WARMUP_SESSIONS = 60
DEFAULT_HORIZON_SESSIONS = 10
DEFAULT_UNIVERSE_LIMIT = 40

_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="qt-historical-paper")
_lock = threading.Lock()
_future: Future | None = None
_future_batch_id = ""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(dict(payload), indent=2, default=str), encoding="utf-8")
    os.replace(tmp, path)


def load_state(path: str | Path | None = None) -> dict[str, Any]:
    target = Path(path) if path is not None else DEFAULT_STATE
    payload = _read_json(target)
    return {
        "schema_version": SCHEMA_VERSION,
        "phase": str(payload.get("phase") or PHASE_IDLE),
        "current_batch_id": str(payload.get("current_batch_id") or ""),
        "current_sessions": list(payload.get("current_sessions") or []),
        "last_completed_session": str(payload.get("last_completed_session") or ""),
        "last_result": dict(payload.get("last_result") or {}),
        "last_error": str(payload.get("last_error") or ""),
        "updated_at": str(payload.get("updated_at") or ""),
    }


def _save_state(payload: Mapping[str, Any], path: str | Path | None = None) -> dict[str, Any]:
    target = Path(path) if path is not None else DEFAULT_STATE
    state = {**load_state(target), **dict(payload), "schema_version": SCHEMA_VERSION, "updated_at": _now()}
    _write_json(target, state)
    return state


def _official_sessions(sessions_fn: Callable[[], Sequence[Any]] | None = None) -> list[str]:
    if sessions_fn is not None:
        raw = list(sessions_fn() or [])
    else:
        from product.historical_replay import official_sessions
        raw = official_sessions()
    out: list[str] = []
    for value in raw:
        text = str(getattr(value, "isoformat", lambda: value)())[:10]
        if len(text) == 10 and text not in out:
            out.append(text)
    return sorted(out)


def _batch_id(sessions: Sequence[str], universe_limit: int) -> str:
    raw = json.dumps({"sessions": list(sessions), "universe_limit": int(universe_limit)}, sort_keys=True)
    return "hist_" + hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def peek_next_batch(
    *,
    sessions_fn: Callable[[], Sequence[Any]] | None = None,
    state_path: str | Path | None = None,
    batch_size: int = DEFAULT_BATCH_SIZE,
    warmup_sessions: int = DEFAULT_WARMUP_SESSIONS,
    horizon_sessions: int = DEFAULT_HORIZON_SESSIONS,
    universe_limit: int = DEFAULT_UNIVERSE_LIMIT,
) -> dict[str, Any]:
    """Return the next unprocessed fully-settleable historical batch."""
    state = load_state(state_path)
    if state["phase"] in {PHASE_RUNNING, PHASE_AWAITING_LEARNING, PHASE_AWAITING_RESEARCH}:
        return {
            "available": False,
            "reason": "batch_in_progress",
            "phase": state["phase"],
            "batch_id": state["current_batch_id"],
            "sessions": state["current_sessions"],
        }

    sessions = _official_sessions(sessions_fn)
    warmup = max(0, int(warmup_sessions))
    horizon = max(1, int(horizon_sessions))
    if len(sessions) <= warmup + horizon:
        return {"available": False, "reason": "insufficient_settleable_history", "sessions_total": len(sessions)}

    eligible = sessions[warmup: len(sessions) - horizon]
    last = state["last_completed_session"]
    start = 0
    if last:
        try:
            start = eligible.index(last) + 1
        except ValueError:
            # Data retention may have dropped old dates. Continue from the first
            # eligible session strictly newer than the durable cursor.
            start = next((i for i, day in enumerate(eligible) if day > last), len(eligible))
    batch = eligible[start: start + max(1, int(batch_size))]
    if not batch:
        return {
            "available": False,
            "reason": "historical_backlog_caught_up",
            "last_completed_session": last,
            "eligible_sessions": len(eligible),
        }
    return {
        "available": True,
        "batch_id": _batch_id(batch, universe_limit),
        "sessions": batch,
        "period_start": batch[0],
        "period_end": batch[-1],
        "universe_limit": int(universe_limit),
        "horizon_sessions": horizon,
    }


def pending_stage(*, state_path: str | Path | None = None) -> dict[str, Any]:
    state = load_state(state_path)
    return {
        "phase": state["phase"],
        "batch_id": state["current_batch_id"],
        "sessions": state["current_sessions"],
        "last_result": state["last_result"],
        "last_error": state["last_error"],
    }


def _f(value: Any) -> float | None:
    try:
        if value is None or value == "":
            return None
        out = float(value)
        return None if out != out else out
    except (TypeError, ValueError):
        return None


def _close(bar: Mapping[str, Any]) -> float | None:
    for key in ("close", "Close", "CLOSE"):
        if key in bar:
            return _f(bar.get(key))
    return None


def _high(bar: Mapping[str, Any]) -> float | None:
    for key in ("high", "High", "HIGH"):
        if key in bar:
            return _f(bar.get(key))
    return None


def _low(bar: Mapping[str, Any]) -> float | None:
    for key in ("low", "Low", "LOW"):
        if key in bar:
            return _f(bar.get(key))
    return None


def simulate_virtual_trade(
    row: Mapping[str, Any],
    *,
    later_bars_fn: Callable[..., list[dict[str, Any]]] | None = None,
    horizon: int = DEFAULT_HORIZON_SESSIONS,
) -> dict[str, Any] | None:
    """Conservative long-only paper execution on later official daily bars.

    If stop and target are both touched in the same daily bar, stop wins because
    intraday ordering is unknowable from OHLC. This avoids optimistic fill bias.
    """
    if str(row.get("decision") or "").upper() != "BUY":
        return None
    entry = _f(row.get("entry"))
    stop = _f(row.get("stop"))
    target = _f(row.get("target"))
    if entry is None or stop is None or stop >= entry:
        return None

    if later_bars_fn is None:
        from product.decision_outcomes import later_bars as later_bars_fn
    bars = list(later_bars_fn(str(row.get("symbol") or ""), str(row.get("as_of") or ""), horizon=int(horizon)) or [])
    if not bars:
        return None

    exit_price: float | None = None
    exit_date = ""
    exit_reason = "HORIZON_CLOSE"
    same_bar_ambiguous = False
    for bar in bars:
        hi = _high(bar)
        lo = _low(bar)
        stop_hit = lo is not None and lo <= stop
        target_hit = target is not None and hi is not None and hi >= target
        if stop_hit and target_hit:
            exit_price = stop
            exit_date = str(bar.get("date") or "")[:10]
            exit_reason = "STOP_AND_TARGET_SAME_BAR_STOP_FIRST_CONSERVATIVE"
            same_bar_ambiguous = True
            break
        if stop_hit:
            exit_price = stop
            exit_date = str(bar.get("date") or "")[:10]
            exit_reason = "STOP_HIT"
            break
        if target_hit:
            exit_price = target
            exit_date = str(bar.get("date") or "")[:10]
            exit_reason = "TARGET_HIT"
            break

    if exit_price is None:
        last = bars[-1]
        exit_price = _close(last)
        exit_date = str(last.get("date") or "")[:10]
    if exit_price is None:
        return None

    risk = entry - stop
    realized_r = (exit_price - entry) / risk
    return_pct = (exit_price - entry) / entry * 100.0
    trade_id = "hist-paper:" + hashlib.sha256(
        f"{row.get('run_id')}|{row.get('as_of')}|{row.get('symbol')}|{entry}|{stop}|{target}".encode("utf-8")
    ).hexdigest()[:20]
    return {
        "trade_id": trade_id,
        "symbol": str(row.get("symbol") or "").upper(),
        "entry_date": str(row.get("as_of") or "")[:10],
        "exit_date": exit_date,
        "entry_price": round(entry, 6),
        "stop_price": round(stop, 6),
        "target_price": None if target is None else round(target, 6),
        "exit_price": round(exit_price, 6),
        "exit_reason": exit_reason,
        "realized_R": round(realized_r, 6),
        "return_pct": round(return_pct, 6),
        "same_bar_path_ambiguous": same_bar_ambiguous,
        "setup": str(row.get("setup") or ""),
        "sector": str(row.get("sector") or ""),
        "regime": str(row.get("regime") or "UNKNOWN"),
        "decision_reason_code": str(row.get("reason_code") or ""),
        "source_run_id": str(row.get("run_id") or ""),
        "source_freeze_id": str(row.get("freeze_id") or ""),
        "source_decision_id": str(row.get("canonical_decision_id") or row.get("decision_id") or ""),
        "evidence_class": "HISTORICAL_REPLAY",
        "paper_lane": "HISTORICAL_VIRTUAL_PAPER",
        "not_real_pnl": True,
        "not_promotion_evidence": True,
        "live_locked": True,
    }


def _normal_cdf(z: float) -> float:
    return 0.5 * (1.0 + math.erf(float(z) / math.sqrt(2.0)))


def update_historical_setup_policies(
    trades: Sequence[Mapping[str, Any]],
    *,
    path: str | Path | None = None,
) -> list[dict[str, Any]]:
    """Persist setup-level historical priors with explicit uncertainty.

    Historical-only evidence may inform shadow confidence, but it is written as
    backtest_historical_replay and affects_selection=False. Therefore the policy
    store can never make it ACTIVE from this lane alone.
    """
    grouped: dict[str, list[dict[str, Any]]] = {}
    for raw in trades:
        row = dict(raw)
        setup = str(row.get("setup") or "").strip()
        r = _f(row.get("realized_R"))
        if not setup or r is None:
            continue
        grouped.setdefault(setup, []).append(row)

    out: list[dict[str, Any]] = []
    for setup, rows in sorted(grouped.items()):
        rs = [float(_f(r.get("realized_R")) or 0.0) for r in rows]
        n = len(rs)
        mean = sum(rs) / n
        if n > 1:
            variance = sum((x - mean) ** 2 for x in rs) / (n - 1)
            std = math.sqrt(max(variance, 0.0))
            se = std / math.sqrt(n)
        else:
            std = 0.0
            se = float("inf")

        shrinkage_k = 8.0
        shrunk = (n / (n + shrinkage_k)) * mean
        if math.isfinite(se) and se > 0:
            lower = shrunk - 1.96 * se
            upper = shrunk + 1.96 * se
            p_edge = _normal_cdf(shrunk / se)
        elif n > 0:
            lower = upper = shrunk
            p_edge = 1.0 if shrunk > 0 else (0.0 if shrunk < 0 else 0.5)
        else:
            lower = upper = 0.0
            p_edge = 0.5

        sample_cap = 49.0 if n < 8 else (69.0 if n < 20 else (79.0 if n < 30 else 95.0))
        confidence_score = round(max(0.0, min(sample_cap, p_edge * 100.0)), 1)
        reproduced_positive = bool(n >= 20 and lower > 0.0)
        generation_fingerprint = hashlib.sha256(
            "|".join(sorted(str(r.get("trade_id") or "") for r in rows)).encode("utf-8")
        ).hexdigest()[:16]

        from product.learning_policy_store import upsert_policy

        policy = upsert_policy(
            policy_id=f"HIST_SETUP::{setup}",
            dimension="historical_setup",
            bucket=setup,
            sample_size=n,
            expectancy_R=mean,
            source="backtest_historical_replay",
            path=path,
            extra={
                "affects_selection": False,
                "historical_confidence_score": confidence_score,
                "historical_reproduced_positive": reproduced_positive,
                "historical_shrunk_mean_R": round(shrunk, 4),
                "historical_lower_95_R": round(lower, 4),
                "historical_upper_95_R": round(upper, 4),
                "historical_p_edge_positive": round(p_edge, 4),
                "generation_fingerprint": generation_fingerprint,
                "not_promotion_evidence": True,
                "not_real_pnl": True,
            },
        )
        out.append(policy)
    return out


def _load_ledger(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    out: list[dict[str, Any]] = []
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if isinstance(row, dict):
                out.append(row)
    except Exception:
        return []
    return out


def _append_unique(path: Path, rows: Sequence[Mapping[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    existing = {str(r.get("trade_id") or "") for r in _load_ledger(path)}
    fresh = [dict(r) for r in rows if str(r.get("trade_id") or "") and str(r.get("trade_id")) not in existing]
    if not fresh:
        return 0
    with path.open("a", encoding="utf-8") as handle:
        for row in fresh:
            handle.write(json.dumps(row, default=str) + "\n")
    return len(fresh)


def _run_batch(
    batch: Mapping[str, Any],
    *,
    state_path: str | Path | None,
    ledger_path: str | Path | None,
    memory_path: str | Path | None,
    replay_fn: Callable[..., Mapping[str, Any]] | None,
    later_bars_fn: Callable[..., list[dict[str, Any]]] | None,
    sessions_fn: Callable[[], Sequence[Any]] | None,
) -> dict[str, Any]:
    bid = str(batch["batch_id"])
    sessions = list(batch["sessions"])
    horizon = int(batch.get("horizon_sessions") or DEFAULT_HORIZON_SESSIONS)
    universe_limit = int(batch.get("universe_limit") or DEFAULT_UNIVERSE_LIMIT)
    all_sessions = _official_sessions(sessions_fn)
    try:
        end_index = all_sessions.index(sessions[-1])
    except ValueError:
        raise RuntimeError("historical batch sessions disappeared from official history")
    if end_index + 1 >= len(all_sessions):
        raise RuntimeError("historical batch has no sentinel session")
    replay_dates = sessions + [all_sessions[end_index + 1]]

    if replay_fn is None:
        from product.historical_replay import run_historical_replay as replay_fn
    report = dict(replay_fn(
        sessions=len(sessions),
        universe_limit=universe_limit,
        force=True,
        dates_fn=lambda: replay_dates,
        persist_live_reco=False,
    ) or {})
    status = str(report.get("status") or "").upper()
    if status not in {"SUCCEEDED", "DEGRADED"}:
        raise RuntimeError(f"historical replay did not complete: {status or 'UNKNOWN'}")

    trades: list[dict[str, Any]] = []
    for raw in report.get("decisions") or report.get("rows") or []:
        if not isinstance(raw, Mapping):
            continue
        row = dict(raw)
        row.setdefault("run_id", report.get("run_id"))
        trade = simulate_virtual_trade(row, later_bars_fn=later_bars_fn, horizon=horizon)
        if trade is not None:
            trades.append(trade)

    ledger = Path(ledger_path) if ledger_path is not None else DEFAULT_LEDGER
    appended = _append_unique(ledger, trades)
    all_trades = _load_ledger(ledger)
    memory_target = Path(memory_path) if memory_path is not None else DEFAULT_MEMORY
    try:
        from product.paper_learning import build_paper_memory
        memory = build_paper_memory(all_trades, as_of=sessions[-1])
        memory["evidence_class"] = "HISTORICAL_REPLAY"
        memory["paper_lane"] = "HISTORICAL_VIRTUAL_PAPER"
        memory["not_promotion_evidence"] = True
        memory["not_real_pnl"] = True
        _write_json(memory_target, memory)
    except Exception as exc:
        memory = {"error": str(exc)[:240]}

    try:
        setup_policies = update_historical_setup_policies(all_trades)
    except Exception as exc:
        setup_policies = [{"error": str(exc)[:240]}]

    result = {
        "status": "SUCCEEDED",
        "batch_id": bid,
        "period_start": sessions[0],
        "period_end": sessions[-1],
        "sessions": sessions,
        "replay_status": status,
        "decisions": int(report.get("decisions_tested") or len(report.get("decisions") or [])),
        "historical_paper_trades": len(trades),
        "trades_appended": appended,
        "historical_paper_total": len(all_trades),
        "historical_setup_policies": len([p for p in setup_policies if not p.get("error")]),
        "setup_policy_errors": [p.get("error") for p in setup_policies if p.get("error")],
        "memory": {
            "closed_trades": int(memory.get("closed_trades") or 0) if isinstance(memory, dict) else 0,
            "cooldown": len(memory.get("cooldown") or []) if isinstance(memory, dict) else 0,
            "prefer": len(memory.get("prefer") or []) if isinstance(memory, dict) else 0,
        },
        "evidence_class": "HISTORICAL_REPLAY",
        "paper_lane": "HISTORICAL_VIRTUAL_PAPER",
        "not_promotion_evidence": True,
        "not_real_pnl": True,
        "live_locked": True,
    }
    _save_state({
        "phase": PHASE_AWAITING_LEARNING,
        "current_batch_id": bid,
        "current_sessions": sessions,
        "last_result": result,
        "last_error": "",
    }, state_path)
    return result


def ensure_next_batch_started(
    *,
    expected_batch_id: str = "",
    sessions_fn: Callable[[], Sequence[Any]] | None = None,
    state_path: str | Path | None = None,
    ledger_path: str | Path | None = None,
    memory_path: str | Path | None = None,
    replay_fn: Callable[..., Mapping[str, Any]] | None = None,
    later_bars_fn: Callable[..., list[dict[str, Any]]] | None = None,
    batch_size: int = DEFAULT_BATCH_SIZE,
    warmup_sessions: int = DEFAULT_WARMUP_SESSIONS,
    horizon_sessions: int = DEFAULT_HORIZON_SESSIONS,
    universe_limit: int = DEFAULT_UNIVERSE_LIMIT,
) -> dict[str, Any]:
    """Start/poll one historical batch without blocking the supervisor."""
    global _future, _future_batch_id
    state = load_state(state_path)
    if state["phase"] in {PHASE_AWAITING_LEARNING, PHASE_AWAITING_RESEARCH}:
        if not expected_batch_id or state["current_batch_id"] == expected_batch_id:
            return {"status": "SUCCEEDED", **dict(state.get("last_result") or {})}

    batch = peek_next_batch(
        sessions_fn=sessions_fn,
        state_path=state_path,
        batch_size=batch_size,
        warmup_sessions=warmup_sessions,
        horizon_sessions=horizon_sessions,
        universe_limit=universe_limit,
    )
    if not batch.get("available"):
        if state["phase"] == PHASE_RUNNING and state["current_batch_id"]:
            batch = {
                "available": True,
                "batch_id": state["current_batch_id"],
                "sessions": state["current_sessions"],
                "period_start": (state["current_sessions"] or [""])[0],
                "period_end": (state["current_sessions"] or [""])[-1],
                "universe_limit": universe_limit,
                "horizon_sessions": horizon_sessions,
            }
        else:
            return {"status": "IDLE", **batch}

    bid = str(batch["batch_id"])
    if expected_batch_id and bid != str(expected_batch_id):
        return {
            "status": "FAILED",
            "error": f"historical batch identity changed: expected {expected_batch_id}, got {bid}",
            "batch_id": bid,
        }

    with _lock:
        if _future is not None and _future_batch_id == bid:
            if not _future.done():
                return {"status": "RUNNING", "batch_id": bid, "sessions": list(batch["sessions"])}
            try:
                result = dict(_future.result() or {})
            except Exception as exc:
                _save_state({
                    "phase": PHASE_FAILED,
                    "current_batch_id": bid,
                    "current_sessions": list(batch["sessions"]),
                    "last_error": str(exc)[:300],
                }, state_path)
                _future = None
                _future_batch_id = ""
                return {"status": "FAILED", "batch_id": bid, "error": str(exc)[:300]}
            _future = None
            _future_batch_id = ""
            return result

        # A RUNNING persisted state with no live Future means the process was
        # restarted. Re-submit exactly the same batch; ledger writes are idempotent.
        _save_state({
            "phase": PHASE_RUNNING,
            "current_batch_id": bid,
            "current_sessions": list(batch["sessions"]),
            "last_error": "",
        }, state_path)
        _future_batch_id = bid
        _future = _executor.submit(
            _run_batch,
            batch,
            state_path=state_path,
            ledger_path=ledger_path,
            memory_path=memory_path,
            replay_fn=replay_fn,
            later_bars_fn=later_bars_fn,
            sessions_fn=sessions_fn,
        )
        return {"status": "RUNNING", "batch_id": bid, "sessions": list(batch["sessions"])}


def mark_learning_complete(batch_id: str, *, state_path: str | Path | None = None) -> dict[str, Any]:
    state = load_state(state_path)
    if (
        state["current_batch_id"] != str(batch_id)
        or state["phase"] != PHASE_AWAITING_LEARNING
    ):
        return state
    return _save_state({"phase": PHASE_AWAITING_RESEARCH}, state_path)


def mark_research_complete(batch_id: str, *, state_path: str | Path | None = None) -> dict[str, Any]:
    state = load_state(state_path)
    if (
        state["current_batch_id"] != str(batch_id)
        or state["phase"] != PHASE_AWAITING_RESEARCH
    ):
        return state
    sessions = list(state.get("current_sessions") or [])
    return _save_state({
        "phase": PHASE_IDLE,
        "last_completed_session": sessions[-1] if sessions else state.get("last_completed_session", ""),
        "current_batch_id": "",
        "current_sessions": [],
        "last_error": "",
    }, state_path)
