"""Persistent whole-market scan results with explicit provenance.

The scanner remains the source of truth. This module serializes its output so
UI/API consumers can distinguish *when a scan executed* from *which market
session its data represents*. Unknown facts stay unknown; timestamp corruption
must never be normalized into plausible-looking zeroes.
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

from core.runtime_paths import logs_path


def default_scan_path() -> Path:
    return logs_path("product", "latest_momentum_scan.json")


DEFAULT_SCAN_PATH = default_scan_path()
_SUPPORTED_SCHEMA_VERSIONS = {1, 2}


def _value(obj: Any, name: str, default: Any = None) -> Any:
    if isinstance(obj, Mapping):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _opt_float(obj: Any, name: str) -> float | None:
    """Persist a number only when the scanner actually set it. Missing stays missing."""
    if isinstance(obj, Mapping) and name not in obj:
        return None
    if not isinstance(obj, Mapping) and not hasattr(obj, name):
        return None
    raw = _value(obj, name, None)
    if raw is None or raw == "":
        return None
    try:
        return float(raw)
    except (TypeError, ValueError):
        return None


def _opt_bool(obj: Any, name: str) -> bool | None:
    if isinstance(obj, Mapping):
        if name not in obj:
            return None
        return bool(obj.get(name))
    if not hasattr(obj, name):
        return None
    return bool(getattr(obj, name))


def _as_utc(value: datetime | str | None) -> datetime | None:
    if value is None or value == "":
        return None
    try:
        if isinstance(value, datetime):
            moment = value
        else:
            moment = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        if moment.tzinfo is None:
            moment = moment.replace(tzinfo=timezone.utc)
        return moment.astimezone(timezone.utc)
    except (TypeError, ValueError):
        return None


def _duration_truth(
    started_at: datetime | str | None,
    completed_at: datetime | str | None,
) -> tuple[float | None, str, str]:
    """Return duration only for a provable, correctly ordered interval."""
    start = _as_utc(started_at)
    completed = _as_utc(completed_at)
    if start is None:
        return None, "UNAVAILABLE", "SCAN_START_TIME_UNAVAILABLE"
    if completed is None:
        return None, "UNAVAILABLE", "SCAN_COMPLETION_TIME_UNAVAILABLE"
    delta = (completed - start).total_seconds()
    if delta < 0:
        return None, "UNAVAILABLE", "SCAN_TIMESTAMP_ORDER_INVALID"
    return round(delta, 3), "AVAILABLE", ""


def _history_provenance() -> dict[str, Any]:
    """Read canonical official-history session truth without acquiring data."""
    try:
        from data.bhavcopy_runtime import official_history_freshness

        raw = dict(official_history_freshness(load_cache=True) or {})
    except Exception as exc:
        return {
            "market_session_date": None,
            "price_data_as_of": None,
            "expected_session_date": None,
            "freshness_state": "UNAVAILABLE",
            "provenance_reason": f"HISTORY_PROVENANCE_UNAVAILABLE:{type(exc).__name__}",
        }

    available = str(raw.get("available_session") or "")[:10] or None
    expected = str(raw.get("expected_latest_completed_session") or "")[:10] or None
    reason = str(raw.get("reason_code") or "").strip()
    current = bool(raw.get("current"))
    if current:
        state = "CURRENT"
    elif available:
        state = "STALE"
    else:
        state = "UNAVAILABLE"
    if not reason:
        reason = "CURRENT" if current else "HISTORY_NOT_READY"
    return {
        "market_session_date": available,
        "price_data_as_of": available,
        "expected_session_date": expected,
        "freshness_state": state,
        "provenance_reason": reason,
    }


def _record(signal: Any, names: Mapping[str, str], fno_symbols: set[str]) -> dict[str, Any]:
    symbol = str(_value(signal, "symbol", "") or "").upper()
    signals = [str(x) for x in (_value(signal, "signals", []) or [])]
    reasons = [str(x) for x in (_value(signal, "reasons", []) or [])]
    chase = bool(_value(signal, "chase_risk", False))
    verdict = str(_value(signal, "verdict", "WATCH") or "WATCH")
    if chase:
        status = "Wait for pullback"
    elif verdict == "BUY":
        status = "Ready to trade"
    elif "PRE_BREAKOUT" in signals:
        status = "Watch for breakout"
    else:
        status = "Watch"
    cats_raw = _value(signal, "categories", None)
    if isinstance(cats_raw, (set, tuple, list)):
        categories = sorted({str(c) for c in cats_raw if c})
    else:
        categories = []
    row = {
        "symbol": symbol,
        "company": str(names.get(symbol, symbol)),
        "status": status,
        "verdict": verdict,
        "price": float(_value(signal, "price", 0.0) or 0.0),
        "momentum_5d": float(_value(signal, "momentum_5d", 0.0) or 0.0),
        "score": float(_value(signal, "score", 0.0) or 0.0),
        "rsi": float(_value(signal, "rsi", 0.0) or 0.0),
        "volume_ratio": float(_value(signal, "volume_ratio", 0.0) or 0.0),
        "entry": float(_value(signal, "entry", 0.0) or 0.0),
        "stop": float(_value(signal, "stop", 0.0) or 0.0),
        "target": float(_value(signal, "target", 0.0) or 0.0),
        "chase_risk": chase,
        "fno_available": symbol in fno_symbols,
        "signals": signals,
        "reasons": reasons,
        "why": reasons[0] if reasons else "No explanation recorded",
        "breakout_grade": str(_value(signal, "breakout_grade", "") or ""),
        "categories": categories,
    }
    for key in (
        "change_pct", "pivot_distance_pct", "breakout_conviction", "avg_vol20",
    ):
        value = _opt_float(signal, key)
        if value is not None:
            row[key] = value
    for key in ("above_sma50", "above_sma200"):
        flag = _opt_bool(signal, key)
        if flag is not None:
            row[key] = flag
    return row


def build_scan_payload(
    names: Mapping[str, str],
    results: Iterable[Any],
    fno_symbols: Iterable[str] = (),
    *,
    scanned_at: datetime | None = None,
    scan_started_at: datetime | str | None = None,
    scan_completed_at: datetime | str | None = None,
    scanned: int | None = None,
    approved_universe: int | None = None,
    requested_universe: int | None = None,
    universe_failed: int | None = None,
    history_provenance: Mapping[str, Any] | None = None,
    fundamental_data_as_of: str | None = None,
    news_data_as_of: str | None = None,
    source_set: Iterable[str] = (),
) -> dict[str, Any]:
    """Build schema-v2 scan state from real scan output and explicit provenance.

    ``scanned_at`` is kept as a backward-compatible alias for completion time.
    Duration is emitted only when both endpoints are known and ordered.
    """
    fno = {str(s).upper() for s in fno_symbols}
    records = [_record(row, names, fno) for row in results]
    records.sort(key=lambda row: (-float(row["score"] or 0.0), row["symbol"]))
    momentum = [r for r in records if "MOMENTUM" in r["signals"]]
    near = [r for r in records if "PRE_BREAKOUT" in r["signals"] and "MOMENTUM" not in r["signals"]]
    ready = [r for r in records if r["status"] == "Ready to trade"]

    completed = _as_utc(scan_completed_at) or _as_utc(scanned_at) or datetime.now(timezone.utc)
    start = _as_utc(scan_started_at)
    duration_s, duration_status, duration_reason = _duration_truth(start, completed)
    completed_iso = completed.isoformat()
    start_iso = start.isoformat() if start is not None else None

    approved_n = int(approved_universe) if approved_universe is not None else len(names)
    normalized_n = len({str(s).strip().upper() for s in names if str(s).strip()})
    requested_n = int(requested_universe) if requested_universe is not None else normalized_n
    scanned_n = int(scanned) if scanned is not None else normalized_n
    failed_n = int(universe_failed) if universe_failed is not None else None
    qualified_n = len(records)

    provenance = dict(history_provenance) if history_provenance is not None else _history_provenance()
    market_session_date = str(provenance.get("market_session_date") or "")[:10] or None
    price_data_as_of = str(provenance.get("price_data_as_of") or market_session_date or "")[:10] or None
    expected_session_date = str(provenance.get("expected_session_date") or "")[:10] or None
    freshness_state = str(provenance.get("freshness_state") or "UNAVAILABLE")
    provenance_reason = str(provenance.get("provenance_reason") or "HISTORY_PROVENANCE_UNAVAILABLE")
    sources = sorted({str(item).strip() for item in source_set if str(item).strip()})

    return {
        "schema_version": 2,
        "scan_id": f"scan:{completed_iso}",
        "scanned_at": completed_iso,
        "scan_started_at": start_iso,
        "scan_completed_at": completed_iso,
        "scan_duration_s": duration_s,
        "scan_duration_status": duration_status,
        "scan_duration_reason": duration_reason or None,
        "market_session_date": market_session_date,
        "price_data_as_of": price_data_as_of,
        "expected_session_date": expected_session_date,
        "fundamental_data_as_of": fundamental_data_as_of or None,
        "news_data_as_of": news_data_as_of or None,
        "freshness_state": freshness_state,
        "provenance_reason": provenance_reason,
        "source_set": sources,
        "approved_universe": approved_n,
        "requested_universe": requested_n,
        "universe_requested": requested_n,
        "universe_loaded": normalized_n,
        "scanned": scanned_n,
        "universe_scanned": scanned_n,
        "universe_failed": failed_n,
        "candidate_count": qualified_n,
        "qualified_rows": qualified_n,
        "universe_size": scanned_n,
        "records": records,
        "summary": {
            "with_any_setup": qualified_n,
            "qualified": qualified_n,
            "momentum": len(momentum),
            "fno_momentum": sum(1 for r in momentum if r["fno_available"]),
            "near_breakout": len(near),
            "ready_to_trade": len(ready),
            "extended": sum(1 for r in records if r["chase_risk"]),
        },
    }


def save_scan(payload: Mapping[str, Any], path: str | Path = DEFAULT_SCAN_PATH) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(target.suffix + ".tmp")
    tmp.write_text(json.dumps(dict(payload), indent=2, default=str), encoding="utf-8")
    os.replace(tmp, target)
    return target


def load_scan(path: str | Path = DEFAULT_SCAN_PATH) -> dict[str, Any] | None:
    target = Path(path)
    if not target.exists():
        return None
    try:
        payload = json.loads(target.read_text(encoding="utf-8"))
        if int(payload.get("schema_version", 0)) not in _SUPPORTED_SCHEMA_VERSIONS:
            return None
        if not isinstance(payload.get("records"), list):
            return None
        return payload
    except Exception:
        return None


def watchlist_rows(payload: Mapping[str, Any] | None, limit: int = 25) -> list[dict[str, Any]]:
    if not payload:
        return []
    priority = {"Ready to trade": 0, "Watch for breakout": 1, "Wait for pullback": 2, "Watch": 3}
    rows = list(payload.get("records", []))
    rows.sort(key=lambda r: (priority.get(str(r.get("status")), 9), -float(r.get("score", 0) or 0),
                             str(r.get("symbol", ""))))
    return rows[: max(0, int(limit))]


def scan_age_hours(payload: Mapping[str, Any] | None, *, now: datetime | None = None) -> float | None:
    if not payload or not payload.get("scanned_at"):
        return None
    stamp = _as_utc(str(payload["scanned_at"]))
    current = _as_utc(now or datetime.now(timezone.utc))
    if stamp is None or current is None:
        return None
    seconds = (current - stamp).total_seconds()
    if seconds < 0:
        return None
    return seconds / 3600.0


def scan_artifact_is_fresh(
    path: str | Path = DEFAULT_SCAN_PATH,
    *,
    max_age_s: float,
    now: datetime | None = None,
) -> bool:
    """True only when scan age is provable and within ``max_age_s``."""
    age_h = scan_age_hours(load_scan(path), now=now)
    if age_h is None:
        return False
    try:
        return age_h * 3600.0 <= float(max_age_s)
    except (TypeError, ValueError):
        return False
