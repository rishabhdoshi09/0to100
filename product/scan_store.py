"""Persistent retail scan results and tomorrow-watchlist projection.

The scanner remains the source of truth. This module only serializes its output
so the UI opens instantly instead of rescanning the full market on every rerun.
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

def default_scan_path() -> Path:
    return Path(__file__).resolve().parents[1] / "logs" / "product" / "latest_momentum_scan.json"


DEFAULT_SCAN_PATH = default_scan_path()


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
) -> dict[str, Any]:
    fno = {str(s).upper() for s in fno_symbols}
    records = [_record(row, names, fno) for row in results]
    # deterministic ranking: score descending, symbol as the stable secondary key for ties
    records.sort(key=lambda row: (-float(row["score"] or 0.0), row["symbol"]))
    momentum = [r for r in records if "MOMENTUM" in r["signals"]]
    near = [r for r in records if "PRE_BREAKOUT" in r["signals"] and "MOMENTUM" not in r["signals"]]
    ready = [r for r in records if r["status"] == "Ready to trade"]
    now = scanned_at or datetime.now(timezone.utc)
    return {
        "schema_version": 1,
        "scanned_at": now.isoformat(),
        "universe_size": len(names),
        "records": records,
        "summary": {
            "with_any_setup": len(records),
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
        if int(payload.get("schema_version", 0)) != 1 or not isinstance(payload.get("records"), list):
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
                             str(r.get("symbol", ""))))       # symbol tiebreak → deterministic
    return rows[: max(0, int(limit))]


def scan_age_hours(payload: Mapping[str, Any] | None, *, now: datetime | None = None) -> float | None:
    if not payload or not payload.get("scanned_at"):
        return None
    try:
        stamp = datetime.fromisoformat(str(payload["scanned_at"]).replace("Z", "+00:00"))
        if stamp.tzinfo is None:
            stamp = stamp.replace(tzinfo=timezone.utc)
        current = now or datetime.now(timezone.utc)
        if current.tzinfo is None:
            current = current.replace(tzinfo=timezone.utc)
        return max(0.0, (current - stamp).total_seconds() / 3600.0)
    except Exception:
        return None


def scan_artifact_is_fresh(
    path: str | Path = DEFAULT_SCAN_PATH,
    *,
    max_age_s: float,
    now: datetime | None = None,
) -> bool:
    """True when the canonical scan JSON exists and scanned_at is within max_age_s."""
    age_h = scan_age_hours(load_scan(path), now=now)
    if age_h is None:
        return False
    try:
        return age_h * 3600.0 <= float(max_age_s)
    except (TypeError, ValueError):
        return False
