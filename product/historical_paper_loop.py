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
DEFAULT_REPLAY_ROOT = logs_path("product/historical_paper_replays")

DEFAULT_BATCH_SIZE = 8
DEFAULT_WARMUP_SESSIONS = 60
DEFAULT_HORIZON_SESSIONS = 20
DEFAULT_UNIVERSE_LIMIT = 40

_lock = threading.Lock()
_thread: threading.Thread | None = None
_thread_batch_id = ""
_thread_result: dict[str, Any] | None = None
_thread_error = ""


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
        "thesis_hash": str(payload.get("thesis_hash") or ""),
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


def _batch_id(sessions: Sequence[str], universe_limit: int, thesis_hash: str = "") -> str:
    raw = json.dumps(
        {
            "sessions": list(sessions),
            "universe_limit": int(universe_limit),
            "thesis_hash": str(thesis_hash or ""),
        },
        sort_keys=True,
    )
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
    try:
        from product.trading_thesis import manifest as thesis_manifest
        thesis_hash = str(thesis_manifest().get("thesis_hash") or "")
    except Exception:
        thesis_hash = ""
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
    # A materially new thesis must earn its own historical evidence. Once the
    # previous batch is terminal/idle, restart the historical cursor for the new
    # thesis instead of inheriting old-policy confidence.
    last = (
        state["last_completed_session"]
        if str(state.get("thesis_hash") or "") == thesis_hash
        else ""
    )
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
        "batch_id": _batch_id(batch, universe_limit, thesis_hash),
        "sessions": batch,
        "thesis_hash": thesis_hash,
        "period_start": batch[0],
        "period_end": batch[-1],
        "universe_limit": int(universe_limit),
        "horizon_sessions": horizon,
    }


def reset_for_thesis(
    thesis_hash: str,
    *,
    state_path: str | Path | None = None,
) -> dict[str, Any]:
    """Start a clean historical cursor for a newly approved production thesis.

    Any old daemon may finish its computation, but _run_batch/_background guard
    against writing terminal state or confidence for a no-longer-current thesis.
    """
    return _save_state({
        "phase": PHASE_IDLE,
        "current_batch_id": "",
        "current_sessions": [],
        "last_completed_session": "",
        "thesis_hash": str(thesis_hash or ""),
        "last_result": {},
        "last_error": "",
    }, state_path)


def pending_stage(*, state_path: str | Path | None = None) -> dict[str, Any]:
    state = load_state(state_path)
    return {
        "phase": state["phase"],
        "batch_id": state["current_batch_id"],
        "sessions": state["current_sessions"],
        "thesis_hash": state["thesis_hash"],
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
        f"{row.get('thesis_hash')}|{row.get('run_id')}|{row.get('as_of')}|{row.get('symbol')}|{entry}|{stop}|{target}".encode("utf-8")
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
        "thesis_hash": str(row.get("thesis_hash") or ""),
        "selection_score": _f(row.get("selection_score")),
        "evidence_class": "HISTORICAL_REPLAY",
        "paper_lane": "HISTORICAL_VIRTUAL_PAPER",
        "not_real_pnl": True,
        "not_promotion_evidence": True,
        "live_locked": True,
    }



def _paper_bar(row: Mapping[str, Any]) -> tuple[float, ...] | None:
    """Convert one official OHLC row to PaperBook's bar contract."""
    def val(*keys: str) -> float | None:
        for key in keys:
            out = _f(row.get(key))
            if out is not None:
                return out
        return None

    op = val("open", "Open", "OPEN")
    hi = val("high", "High", "HIGH")
    lo = val("low", "Low", "LOW")
    close = val("close", "Close", "CLOSE")
    if hi is None or lo is None or close is None:
        return None
    return (op, hi, lo, close) if op is not None else (hi, lo, close)


def simulate_paper_book_sequence(
    decisions: Sequence[Mapping[str, Any]],
    *,
    official_sessions: Sequence[str],
    later_bars_fn: Callable[..., list[dict[str, Any]]] | None = None,
    horizon: int = DEFAULT_HORIZON_SESSIONS,
    max_new_per_session: int = 3,
) -> dict[str, Any]:
    """Replay historical BUY decisions through the real PaperBook mechanics.

    Unlike independent trade scoring, this preserves overlapping positions,
    duplicate-name blocking, five-position / total-risk / per-name sizing caps,
    realistic paper slippage + India cash costs, and the same portfolio
    selection authority used by PAPER_FORWARD. New positions are opened only
    after prior positions have been marked on that session, so a historical
    entry never sees its own completed daily OHLC.

    Historical evidence remains a separate non-P&L evidence class.
    """
    from product.paper_autopilot import (
        DEFAULT_RISK_PCT,
        ENTER_NOW,
        evaluate_selection_candidate,
    )
    from product.portfolio_selection_authority import apply_portfolio_authority
    from product.strategy_catalog import ENSEMBLE_ID
    from research.auto_research.costs import india_cash_costs
    from research.auto_research.paper_book import PaperBook

    if later_bars_fn is None:
        from product.decision_outcomes import later_bars as later_bars_fn

    sessions = [str(day)[:10] for day in official_sessions if str(day)[:10]]
    if not sessions:
        return {
            "trades": [],
            "rejections": [],
            "book": {},
            "execution_model": "PaperBook",
        }

    candidates: dict[str, list[dict[str, Any]]] = {}
    for raw in decisions:
        row = dict(raw)
        if str(row.get("decision") or "").upper() != "BUY":
            continue
        day = str(row.get("as_of") or "")[:10]
        if day:
            candidates.setdefault(day, []).append(row)

    book = PaperBook(slippage_bps=3.0, cost_model=india_cash_costs)
    bar_cache: dict[str, dict[str, tuple[float, ...]]] = {}
    source_by_intent: dict[str, dict[str, Any]] = {}
    rejections: list[dict[str, Any]] = []

    for day in sessions:
        # First settle/manage positions opened on earlier sessions using only
        # this session's official bar.
        bars: dict[str, tuple[float, ...]] = {}
        for pos in list(book.open.values()):
            bar = (bar_cache.get(str(pos.symbol).upper()) or {}).get(day)
            if bar is not None:
                bars[str(pos.symbol).upper()] = bar
        if bars:
            book.mark(bars, day, allow_entry_session=False)

        raw_day = candidates.get(day) or []
        if not raw_day:
            continue

        ranked = []
        family_risk: dict[str, float] = {}
        cluster_risk: dict[str, float] = {}
        clock = datetime.fromisoformat(f"{day}T15:30:00+05:30")
        for row in raw_day:
            card = row.get("selection_card")
            card = dict(card) if isinstance(card, Mapping) else {}
            if not card:
                # Compatibility for old persisted replays: do not fabricate
                # missing PIT inputs. Such rows can be inspected but are not
                # portfolio-comparable historical paper evidence.
                rejections.append({
                    "symbol": str(row.get("symbol") or "").upper(),
                    "as_of": day,
                    "reason_code": "MISSING_PIT_SELECTION_CARD",
                })
                continue
            try:
                decision = evaluate_selection_candidate(
                    card,
                    book=book,
                    workspace={
                        "point_in_time": True,
                        "scan_scanned_at": f"{day}T15:30:00+05:30",
                    },
                    now=clock,
                    entries_allowed=True,
                    paper_enabled=True,
                    regime=str(row.get("regime") or "UNKNOWN"),
                    enforce_history=False,
                    family_risk=family_risk,
                    cluster_risk=cluster_risk,
                )
            except Exception as exc:
                rejections.append({
                    "symbol": str(row.get("symbol") or "").upper(),
                    "as_of": day,
                    "reason_code": "SELECTION_REPLAY_ERROR",
                    "detail": str(exc)[:200],
                })
                continue
            if decision.decision != ENTER_NOW:
                rejected = decision.as_dict()
                rejected["as_of"] = day
                rejections.append(rejected)
                continue
            ranked.append((float(decision.selection_score or 0.0), decision))

        ranked.sort(key=lambda item: (-item[0], item[1].symbol))
        if not ranked:
            continue

        kept, diverted = apply_portfolio_authority(
            ranked,
            book=book,
            max_new=max_new_per_session,
            regime=str(raw_day[0].get("regime") or "UNKNOWN"),
        )
        for decision in diverted:
            rejected = decision.as_dict()
            rejected["as_of"] = day
            rejections.append(rejected)

        for _score, decision in kept:
            row = next(
                (
                    candidate
                    for candidate in raw_day
                    if str(candidate.get("symbol") or "").upper() == decision.symbol
                ),
                {},
            )
            intent_id = "hist-paper-intent:" + hashlib.sha256(
                f"{row.get('thesis_hash')}|{day}|{decision.symbol}|{row.get('canonical_decision_id')}".encode("utf-8")
            ).hexdigest()[:20]
            qty = int(_f(decision.card.get("approved_quantity")) or 0)
            pos = book.open_position(
                ENSEMBLE_ID,
                decision.symbol,
                float(_f(decision.card.get("entry")) or 0.0),
                float(_f(decision.card.get("stop")) or 0.0),
                float(_f(decision.card.get("target")) or 0.0),
                day,
                int(horizon),
                risk_pct_of_capital=DEFAULT_RISK_PCT,
                quantity=(qty if qty > 0 else None),
                decision_id=str(row.get("canonical_decision_id") or ""),
                paper_intent_id=intent_id,
                context_key=str(row.get("context_key") or ""),
            )
            if pos is None:
                reason = str((book.refusals[-1] if book.refusals else ("", "BOOK_REFUSED"))[1])
                rejections.append({
                    "symbol": decision.symbol,
                    "as_of": day,
                    "reason_code": "BOOK_REFUSED",
                    "detail": reason,
                })
                continue

            # PaperPosition is intentionally a normal dataclass (not slots), so
            # sector can travel with the virtual book for portfolio concentration
            # checks exactly as present selection authority expects.
            pos.sector = str(decision.card.get("sector") or row.get("sector") or "")
            family = str(pos.sector or "")
            family_risk[family] = family_risk.get(family, 0.0) + DEFAULT_RISK_PCT
            cluster_risk[family] = cluster_risk.get(family, 0.0) + DEFAULT_RISK_PCT
            source_by_intent[intent_id] = dict(row)

            future = list(
                later_bars_fn(decision.symbol, day, horizon=int(horizon)) or []
            )
            by_date: dict[str, tuple[float, ...]] = {}
            for raw_bar in future:
                if not isinstance(raw_bar, Mapping):
                    continue
                bar_day = str(raw_bar.get("date") or "")[:10]
                bar = _paper_bar(raw_bar)
                if bar_day and bar is not None:
                    by_date[bar_day] = bar
            bar_cache[decision.symbol] = by_date

    trades: list[dict[str, Any]] = []
    for closed in book.closed:
        source = source_by_intent.get(str(closed.paper_intent_id or "")) or {}
        notional = float(closed.entry_price) * int(closed.qty or 0)
        return_pct = (
            float(closed.pnl) / notional * 100.0
            if notional > 0
            else 0.0
        )
        trade_id = "hist-paper-book:" + hashlib.sha256(
            f"{closed.paper_intent_id}|{closed.exit_date}|{closed.exit_price}".encode("utf-8")
        ).hexdigest()[:20]
        trades.append({
            "trade_id": trade_id,
            "symbol": str(closed.symbol or "").upper(),
            "entry_date": str(closed.entry_date or "")[:10],
            "exit_date": str(closed.exit_date or "")[:10],
            "entry_price": round(float(closed.entry_price), 6),
            "stop_price": round(float(closed.stop_price), 6),
            "target_price": _f(source.get("target")),
            "exit_price": round(float(closed.exit_price), 6),
            "exit_reason": str(closed.exit_reason or ""),
            "realized_R": round(float(closed.realized_R), 6),
            "return_pct": round(return_pct, 6),
            "qty": int(closed.qty or 0),
            "net_pnl": round(float(closed.pnl), 6),
            "setup": str(source.get("setup") or ""),
            "sector": str(source.get("sector") or ""),
            "regime": str(source.get("regime") or "UNKNOWN"),
            "decision_reason_code": str(source.get("reason_code") or ""),
            "source_run_id": str(source.get("run_id") or ""),
            "source_freeze_id": str(source.get("freeze_id") or ""),
            "source_decision_id": str(source.get("canonical_decision_id") or source.get("decision_id") or ""),
            "thesis_hash": str(source.get("thesis_hash") or ""),
            "selection_score": _f(source.get("selection_score")),
            "evidence_class": "HISTORICAL_REPLAY",
            "paper_lane": "HISTORICAL_VIRTUAL_PAPER",
            "execution_model": "PaperBook",
            "slippage_bps": 3.0,
            "cost_model": "india_cash_costs",
            "not_real_pnl": True,
            "not_promotion_evidence": True,
            "live_locked": True,
        })

    return {
        "trades": trades,
        "rejections": rejections,
        "open_unresolved": len(book.open),
        "book": book.snapshot(),
        "execution_model": "PaperBook",
        "slippage_bps": 3.0,
        "cost_model": "india_cash_costs",
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
        thesis_hash = str((rows[0] if rows else {}).get("thesis_hash") or "")
        if not thesis_hash:
            continue
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

        from product.learning_policy_store import load_policies, upsert_policy

        existing = next(
            (
                dict(p)
                for p in (load_policies(path).get("policies") or [])
                if str(p.get("policy_id") or "") == f"HIST_SETUP::{thesis_hash}::{setup}"
            ),
            None,
        )
        if (
            existing
            and str(existing.get("generation_fingerprint") or "") == generation_fingerprint
            and int(existing.get("sample_size") or 0) == n
        ):
            out.append(existing)
            continue

        policy = upsert_policy(
            policy_id=f"HIST_SETUP::{thesis_hash}::{setup}",
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
                "thesis_hash": thesis_hash,
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
    expected_thesis_hash = str(batch.get("thesis_hash") or "")
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
    # Autonomous historical-paper replay has its own batch workspace. The
    # operator Decision Simulator uses the default historical_replay directory;
    # keeping these artifacts separate prevents cross-process progress/report/
    # ledger races while both may legitimately run at the same time.
    replay_directory = Path(DEFAULT_REPLAY_ROOT) / bid
    report = dict(replay_fn(
        sessions=len(sessions),
        universe_limit=universe_limit,
        force=True,
        directory=replay_directory,
        dates_fn=lambda: replay_dates,
        persist_live_reco=False,
    ) or {})
    status = str(report.get("status") or "").upper()
    if status not in {"SUCCEEDED", "DEGRADED"}:
        raise RuntimeError(f"historical replay did not complete: {status or 'UNKNOWN'}")

    decisions = [
        dict(raw)
        for raw in (report.get("decisions") or report.get("rows") or [])
        if isinstance(raw, Mapping)
    ]
    mismatched = [
        row for row in decisions
        if expected_thesis_hash
        and str(row.get("thesis_hash") or "") != expected_thesis_hash
    ]
    if mismatched:
        raise RuntimeError(
            "historical replay thesis mismatch; batch evidence refused"
        )

    try:
        from product.trading_thesis import manifest as thesis_manifest
        current_thesis_hash = str(thesis_manifest().get("thesis_hash") or "")
    except Exception:
        current_thesis_hash = ""
    if expected_thesis_hash and current_thesis_hash != expected_thesis_hash:
        return {
            "status": "OBSOLETE_THESIS",
            "batch_id": bid,
            "thesis_hash": expected_thesis_hash,
            "current_thesis_hash": current_thesis_hash,
            "message": "production thesis changed while historical batch was running; evidence not applied",
            "not_real_pnl": True,
            "not_promotion_evidence": True,
        }

    # Replay the BUY lane through the same persistent-risk mechanics as
    # present paper trading instead of treating every candidate as an independent
    # backtest. This prevents impossible overlapping positions from inflating
    # historical confidence.
    first_index = all_sessions.index(sessions[0])
    last_index = all_sessions.index(sessions[-1])
    timeline_end = min(len(all_sessions), last_index + horizon + 1)
    paper_timeline = all_sessions[first_index:timeline_end]
    for row in decisions:
        row.setdefault("run_id", report.get("run_id"))
    paper_sim = simulate_paper_book_sequence(
        decisions,
        official_sessions=paper_timeline,
        later_bars_fn=later_bars_fn,
        horizon=horizon,
        max_new_per_session=3,
    )
    trades = list(paper_sim.get("trades") or [])

    # The paper-book pass can be materially heavier than decision replay. Refuse
    # to persist its evidence if effective selection behavior changed while it
    # was running; the next supervisor tick will create a new thesis-versioned
    # batch instead of mixing versions.
    try:
        from product.trading_thesis import manifest as thesis_manifest
        post_sim_thesis_hash = str(thesis_manifest().get("thesis_hash") or "")
    except Exception:
        post_sim_thesis_hash = ""
    if expected_thesis_hash and post_sim_thesis_hash != expected_thesis_hash:
        return {
            "status": "OBSOLETE_THESIS",
            "batch_id": bid,
            "thesis_hash": expected_thesis_hash,
            "current_thesis_hash": post_sim_thesis_hash,
            "message": "production thesis changed during paper-book replay; evidence not persisted",
            "not_real_pnl": True,
            "not_promotion_evidence": True,
        }

    ledger = Path(ledger_path) if ledger_path is not None else DEFAULT_LEDGER
    appended = _append_unique(ledger, trades)
    all_trades = _load_ledger(ledger)
    thesis_trades = [
        row for row in all_trades
        if not expected_thesis_hash
        or str(row.get("thesis_hash") or "") == expected_thesis_hash
    ]
    memory_target = Path(memory_path) if memory_path is not None else DEFAULT_MEMORY
    try:
        from product.paper_learning import build_paper_memory
        memory = build_paper_memory(thesis_trades, as_of=sessions[-1])
        memory["evidence_class"] = "HISTORICAL_REPLAY"
        memory["paper_lane"] = "HISTORICAL_VIRTUAL_PAPER"
        memory["not_promotion_evidence"] = True
        memory["not_real_pnl"] = True
        _write_json(memory_target, memory)
    except Exception as exc:
        memory = {"error": str(exc)[:240]}

    try:
        setup_policies = update_historical_setup_policies(thesis_trades)
    except Exception as exc:
        setup_policies = [{"error": str(exc)[:240]}]

    result = {
        "status": "SUCCEEDED",
        "batch_id": bid,
        "thesis_hash": expected_thesis_hash,
        "period_start": sessions[0],
        "period_end": sessions[-1],
        "sessions": sessions,
        "replay_status": status,
        "replay_workspace": str(replay_directory),
        "decisions": int(report.get("decisions_tested") or len(report.get("decisions") or [])),
        "historical_paper_trades": len(trades),
        "trades_appended": appended,
        "historical_paper_total": len(thesis_trades),
        "historical_paper_open_unresolved": int(paper_sim.get("open_unresolved") or 0),
        "historical_paper_rejections": len(list(paper_sim.get("rejections") or [])),
        "historical_execution_model": str(paper_sim.get("execution_model") or "PaperBook"),
        "historical_slippage_bps": paper_sim.get("slippage_bps"),
        "historical_cost_model": paper_sim.get("cost_model"),
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
        "thesis_hash": expected_thesis_hash,
        "last_result": result,
        "last_error": "",
    }, state_path)
    return result


def _background_batch_runner(
    batch: Mapping[str, Any],
    *,
    state_path: str | Path | None,
    ledger_path: str | Path | None,
    memory_path: str | Path | None,
    replay_fn: Callable[..., Mapping[str, Any]] | None,
    later_bars_fn: Callable[..., list[dict[str, Any]]] | None,
    sessions_fn: Callable[[], Sequence[Any]] | None,
) -> None:
    """Run one heavy historical batch in a daemon thread.

    Daemon ownership is deliberate: launcher Ctrl+C must be able to terminate
    the autonomy process immediately instead of waiting for an offline replay.
    Durable phase state makes the same batch restart-safe on the next launch.
    """
    global _thread_result, _thread_error
    try:
        result = _run_batch(
            batch,
            state_path=state_path,
            ledger_path=ledger_path,
            memory_path=memory_path,
            replay_fn=replay_fn,
            later_bars_fn=later_bars_fn,
            sessions_fn=sessions_fn,
        )
        with _lock:
            _thread_result = dict(result or {})
            _thread_error = ""
    except Exception as exc:
        current_state = load_state(state_path)
        # Do not let a stale old-thesis daemon overwrite a reset/new-thesis state.
        if str(current_state.get("thesis_hash") or "") == str(batch.get("thesis_hash") or ""):
            _save_state({
                "phase": PHASE_FAILED,
                "current_batch_id": str(batch.get("batch_id") or ""),
                "current_sessions": list(batch.get("sessions") or []),
                "thesis_hash": str(batch.get("thesis_hash") or ""),
                "last_error": str(exc)[:300],
            }, state_path)
        with _lock:
            _thread_result = None
            _thread_error = str(exc)[:300]


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
    global _thread, _thread_batch_id, _thread_result, _thread_error
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
                "thesis_hash": str(state.get("thesis_hash") or ""),
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
        if _thread is not None and _thread_batch_id == bid:
            if _thread.is_alive():
                return {"status": "RUNNING", "batch_id": bid, "sessions": list(batch["sessions"])}
            if _thread_error:
                error = _thread_error
                _thread = None
                _thread_batch_id = ""
                _thread_result = None
                _thread_error = ""
                return {"status": "FAILED", "batch_id": bid, "error": error}
            if _thread_result is not None:
                result = dict(_thread_result)
                _thread = None
                _thread_batch_id = ""
                _thread_result = None
                return result

        # A RUNNING persisted state with no live daemon means the process was
        # restarted. Re-submit exactly the same batch; all ledgers are idempotent.
        _save_state({
            "phase": PHASE_RUNNING,
            "current_batch_id": bid,
            "current_sessions": list(batch["sessions"]),
            "thesis_hash": str(batch.get("thesis_hash") or ""),
            "last_error": "",
        }, state_path)
        _thread_batch_id = bid
        _thread_result = None
        _thread_error = ""
        _thread = threading.Thread(
            target=_background_batch_runner,
            kwargs={
                "batch": batch,
                "state_path": state_path,
                "ledger_path": ledger_path,
                "memory_path": memory_path,
                "replay_fn": replay_fn,
                "later_bars_fn": later_bars_fn,
                "sessions_fn": sessions_fn,
            },
            name="qt-historical-paper",
            daemon=True,
        )
        _thread.start()
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
