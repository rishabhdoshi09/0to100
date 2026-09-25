"""Forward-only learning for the US PAPER lane.

US scanning and execution are intentionally separate from NSE stores. This module
keeps an append/rewrite decision ledger, settles taken trades plus rejected-name
counterfactuals, and exposes bounded rank adjustments only after enough forward
observations. It never places an order and never affects live money.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from core.runtime_paths import logs_path

SCHEMA_VERSION = 1
LEDGER = logs_path("product", "us_decision_ledger.jsonl")
MODEL = logs_path("product", "us_learning.json")
MIN_SAMPLE = 20
SUPPORT_SAMPLE = 30
PRIOR_STRENGTH = 10.0
_HORIZON = 5
_lock = threading.RLock()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read_jsonl(path: Path = LEDGER) -> list[dict[str, Any]]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except Exception:
        return []
    out: list[dict[str, Any]] = []
    for line in lines:
        try:
            row = json.loads(line)
        except Exception:
            continue
        if isinstance(row, dict):
            out.append(row)
    return out


def _write_jsonl(rows: list[Mapping[str, Any]], path: Path = LEDGER) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(
        "".join(json.dumps(dict(row), default=str) + "\n" for row in rows),
        encoding="utf-8",
    )
    os.replace(tmp, path)


def _session_date() -> str:
    try:
        from zoneinfo import ZoneInfo
        return datetime.now(ZoneInfo("America/New_York")).date().isoformat()
    except Exception:
        return datetime.now(timezone.utc).date().isoformat()


def _bucket_score(value: Any) -> str:
    try:
        v = float(value or 0.0)
    except Exception:
        v = 0.0
    if v >= 80:
        return "80+"
    if v >= 70:
        return "70-79"
    if v >= 60:
        return "60-69"
    return "<60"


def _bucket_conviction(value: Any) -> str:
    try:
        v = float(value or 0.0)
    except Exception:
        v = 0.0
    if v >= 70:
        return "70+"
    if v >= 55:
        return "55-69"
    return "<55"


def _decision_id(session_date: str, symbol: str, action: str) -> str:
    raw = f"{session_date}|{symbol.upper()}|{action.upper()}"
    return "usdec_" + hashlib.sha256(raw.encode("utf-8")).hexdigest()[:18]


def record_decision(
    *,
    symbol: str,
    action: str,
    reason: str,
    entry: float,
    stop: float,
    target: float | None,
    score: float,
    conviction: float,
    meta: Mapping[str, Any] | None = None,
    session_date: str | None = None,
) -> dict[str, Any]:
    """Record at most one TAKE and one REJECT per symbol/session."""
    session = str(session_date or _session_date())[:10]
    symbol = str(symbol or "").upper().strip()
    action = str(action or "").upper().strip()
    if not symbol or action not in {"TAKE", "REJECT"}:
        return {}
    did = _decision_id(session, symbol, action)
    with _lock:
        rows = _read_jsonl()
        prior = next((r for r in rows if str(r.get("decision_id")) == did), None)
        if prior is not None:
            return dict(prior)
        meta = dict(meta or {})
        row = {
            "schema_version": SCHEMA_VERSION,
            "decision_id": did,
            "observed_at": _now(),
            "session_date": session,
            "market": "US",
            "symbol": symbol,
            "action": action,
            "reason": str(reason or ""),
            "entry": float(entry or 0.0),
            "stop": float(stop or 0.0),
            "target": None if target in (None, "") else float(target),
            "score": float(score or 0.0),
            "conviction": float(conviction or 0.0),
            "score_bucket": _bucket_score(score),
            "conviction_bucket": _bucket_conviction(conviction),
            "categories": [str(x) for x in (meta.get("categories") or [])],
            "signals": [str(x) for x in (meta.get("signals") or [])],
            "verdict": str(meta.get("verdict") or ""),
            "learned_rank_score": meta.get("learned_rank_score"),
            "learning_adjustment_at_decision": meta.get("learning_adjustment"),
            "settled": False,
            "outcome_R": None,
            "outcome_kind": "",
            "outcome_at": "",
            "not_pnl": action != "TAKE",
            "evidence_class": "US_PAPER_FORWARD" if action == "TAKE" else "US_FORWARD_COUNTERFACTUAL",
            "live_locked": True,
        }
        rows.append(row)
        _write_jsonl(rows)
        return row


def _closed_trade_outcome(row: Mapping[str, Any]) -> tuple[float, str] | None:
    if str(row.get("action")) != "TAKE":
        return None
    try:
        from execution.us_autopilot import _trades, _WIN, _LOSS
        trades = _trades(_WIN + _LOSS)
    except Exception:
        return None
    symbol = str(row.get("symbol") or "").upper()
    session = str(row.get("session_date") or "")[:10]
    entry = float(row.get("entry") or 0.0)
    stop = float(row.get("stop") or 0.0)
    risk = entry - stop
    if risk <= 0:
        return None
    for trade in trades:
        if str(trade.get("symbol") or "").upper() != symbol:
            continue
        note = str(trade.get("note") or "")
        marker = "session="
        trade_session = ""
        if marker in note:
            trade_session = note.split(marker, 1)[1].split("|", 1)[0].strip()[:10]
        if not trade_session:
            trade_session = str(trade.get("placed_at") or "")[:10]
        if trade_session != session:
            continue
        exit_px = float(trade.get("exit_price") or 0.0)
        if exit_px <= 0:
            continue
        return (exit_px - entry) / risk, str(trade.get("status") or "CLOSED")
    return None


def _counterfactual_outcome(row: Mapping[str, Any]) -> tuple[float, str] | None:
    if str(row.get("action")) != "REJECT":
        return None
    symbol = str(row.get("symbol") or "").upper()
    session = str(row.get("session_date") or "")[:10]
    entry = float(row.get("entry") or 0.0)
    stop = float(row.get("stop") or 0.0)
    target = float(row.get("target") or 0.0)
    risk = entry - stop
    if not symbol or entry <= 0 or risk <= 0:
        return None
    if target <= entry:
        target = entry + 2.0 * risk
    try:
        from data.us_data import get_us_daily
        frame = get_us_daily(symbol, lookback_days=90)
    except Exception:
        frame = None
    if frame is None or len(frame) == 0:
        return None

    future: list[tuple[float, float, float]] = []
    try:
        for idx, bar in frame.iterrows():
            day = getattr(idx, "date", lambda: idx)()
            day_s = day.isoformat() if hasattr(day, "isoformat") else str(day)[:10]
            if day_s <= session:
                continue
            future.append((float(bar["high"]), float(bar["low"]), float(bar["close"])))
    except Exception:
        return None
    if len(future) < _HORIZON:
        return None

    for high, low, _close in future[:_HORIZON]:
        # Same-bar target/stop ordering is unknowable from daily OHLC. Be
        # conservative and count the stop first.
        if low <= stop:
            return -1.0, "WOULD_STOP"
        if high >= target:
            return (target - entry) / risk, "MISSED_TARGET"
    close = future[_HORIZON - 1][2]
    return (close - entry) / risk, "HORIZON_CLOSE"


def settle_pending() -> dict[str, Any]:
    with _lock:
        rows = _read_jsonl()
        updated = 0
        for row in rows:
            if row.get("settled"):
                continue
            result = _closed_trade_outcome(row) or _counterfactual_outcome(row)
            if result is None:
                continue
            outcome_r, kind = result
            row["settled"] = True
            row["outcome_R"] = round(float(outcome_r), 6)
            row["outcome_kind"] = kind
            row["outcome_at"] = _now()
            updated += 1
        if updated:
            _write_jsonl(rows)
    model = rebuild_model(rows=rows)
    return {
        "updated": updated,
        "settled": sum(1 for r in rows if r.get("settled")),
        "pending": sum(1 for r in rows if not r.get("settled")),
        "model": model,
        "live_locked": True,
    }


def _wilson_lower(wins: float, n: float) -> float:
    if n <= 0:
        return 0.0
    z = 1.96
    p = wins / n
    denom = 1.0 + z * z / n
    centre = p + z * z / (2.0 * n)
    margin = z * math.sqrt((p * (1.0 - p) + z * z / (4.0 * n)) / n)
    return max(0.0, (centre - margin) / denom)


def _cells(rows: list[Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    groups: dict[str, list[tuple[float, float]]] = {}
    for row in rows:
        if not row.get("settled") or row.get("outcome_R") is None:
            continue
        weight = 1.0 if str(row.get("action")) == "TAKE" else 0.5
        r = float(row.get("outcome_R") or 0.0)
        keys = [
            f"score:{row.get('score_bucket') or ''}",
            f"conviction:{row.get('conviction_bucket') or ''}",
        ]
        categories = [str(x) for x in (row.get("categories") or []) if str(x)]
        if categories:
            keys.append(f"category:{categories[0]}")
        for key in keys:
            groups.setdefault(key, []).append((r, weight))

    out: dict[str, dict[str, Any]] = {}
    for key, vals in groups.items():
        n = len(vals)
        weighted_n = sum(w for _r, w in vals)
        total = sum(r * w for r, w in vals)
        shrunk = total / (weighted_n + PRIOR_STRENGTH)
        wins = sum(w for r, w in vals if r > 0)
        lower = _wilson_lower(wins, weighted_n)
        adjustment = 0.0
        reason = "INSUFFICIENT_FORWARD_EVIDENCE"
        if n >= MIN_SAMPLE and shrunk <= -0.15:
            adjustment = -2.0
            reason = "FORWARD_NEGATIVE_EXPECTANCY"
        elif n >= SUPPORT_SAMPLE and shrunk >= 0.15 and lower >= 0.45:
            adjustment = 1.0
            reason = "FORWARD_POSITIVE_EXPECTANCY"
        out[key] = {
            "n": n,
            "weighted_n": round(weighted_n, 2),
            "shrunk_mean_R": round(shrunk, 4),
            "win_rate_lower_95": round(lower, 4),
            "adjustment": adjustment,
            "reason": reason,
        }
    return out


def rebuild_model(*, rows: list[Mapping[str, Any]] | None = None) -> dict[str, Any]:
    rows = list(rows if rows is not None else _read_jsonl())
    cells = _cells(rows)
    mature = {k: v for k, v in cells.items() if float(v.get("adjustment") or 0.0) != 0.0}
    payload = {
        "schema_version": SCHEMA_VERSION,
        "updated_at": _now(),
        "status": "ACTIVE_PAPER_RANKING" if mature else "COLLECTING_FORWARD_EVIDENCE",
        "settled_decisions": sum(1 for r in rows if r.get("settled")),
        "taken_settled": sum(1 for r in rows if r.get("settled") and r.get("action") == "TAKE"),
        "counterfactual_settled": sum(1 for r in rows if r.get("settled") and r.get("action") == "REJECT"),
        "mature_cells": mature,
        "cells": cells,
        "paper_only": True,
        "live_locked": True,
        "historical_only_can_promote": False,
    }
    MODEL.parent.mkdir(parents=True, exist_ok=True)
    tmp = MODEL.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    os.replace(tmp, MODEL)
    return payload


def load_model() -> dict[str, Any]:
    try:
        payload = json.loads(MODEL.read_text(encoding="utf-8"))
    except Exception:
        return rebuild_model()
    return payload if isinstance(payload, dict) else rebuild_model()


def adjustment(card: Mapping[str, Any]) -> dict[str, Any]:
    model = load_model()
    cells = dict(model.get("cells") or {})
    keys = [
        f"score:{_bucket_score(card.get('score'))}",
        f"conviction:{_bucket_conviction(card.get('breakout_conviction'))}",
    ]
    categories = [str(x) for x in (card.get("categories") or []) if str(x)]
    if categories:
        keys.append(f"category:{categories[0]}")
    used: list[dict[str, Any]] = []
    raw = 0.0
    for key in keys:
        cell = dict(cells.get(key) or {})
        adj = float(cell.get("adjustment") or 0.0)
        if adj:
            raw += adj
            used.append({"key": key, **cell})
    bounded = max(-5.0, min(3.0, raw))
    return {
        "adjustment": round(bounded, 4),
        "affects_selection": bool(used),
        "evidence": used,
        "model_status": str(model.get("status") or "COLLECTING_FORWARD_EVIDENCE"),
        "paper_only": True,
        "live_locked": True,
    }


def dashboard() -> dict[str, Any]:
    rows = _read_jsonl()
    model = load_model()
    return {
        "available": bool(rows),
        "decisions": len(rows),
        "settled": sum(1 for r in rows if r.get("settled")),
        "pending": sum(1 for r in rows if not r.get("settled")),
        "model": model,
        "selection_learning_active": bool(model.get("mature_cells")),
        "paper_only": True,
        "live_locked": True,
    }
