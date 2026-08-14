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

DEFAULT_SCAN_PATH = Path("logs/product/latest_momentum_scan.json")


def _value(obj: Any, name: str, default: Any = None) -> Any:
    if isinstance(obj, Mapping):
        return obj.get(name, default)
    return getattr(obj, name, default)


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
    return {
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
        "sector": str(_value(signal, "sector", "") or ""),
        # Sniper / live-breakout fields (preserved so autonomy can arm Kite WS)
        "categories": sorted(
            set(_value(signal, "categories", set()) or set())
            or ({"PreBreakout"} if "PRE_BREAKOUT" in signals else set())
        ),
        "pivot_distance_pct": float(_value(signal, "pivot_distance_pct", 0.0) or 0.0),
        "avg_vol20": float(_value(signal, "avg_vol20", 0.0) or 0.0),
        "edge_r": (
            float(_value(signal, "edge_r"))
            if _value(signal, "edge_r", None) is not None
            else None
        ),
        # Breakout quality — used to surface the BEST breakout among peers
        "breakout_grade": str(_value(signal, "breakout_grade", "") or ""),
        "breakout_conviction": float(
            _value(signal, "breakout_conviction", 0.0) or 0.0
        ),
    }


def build_scan_payload(
    names: Mapping[str, str],
    results: Iterable[Any],
    fno_symbols: Iterable[str] = (),
    *,
    scanned_at: datetime | None = None,
) -> dict[str, Any]:
    fno = {str(s).upper() for s in fno_symbols}
    # Apply full-universe backtest evidence before serialization so React /
    # pre-trade / conviction all see the same measured edge.
    result_list = list(results)
    try:
        from scan.measured_edge import apply_measured_edge

        apply_measured_edge(result_list)
    except Exception:
        pass
    records = [_record(row, names, fno) for row in result_list]
    # Rank by verdict tier, then score + measured edge — demoted losers stay down.
    vrank = {"STRONG BUY": 2, "BUY": 1}
    records.sort(
        key=lambda row: (
            -vrank.get(str(row.get("verdict") or ""), 0),
            -float(row["score"] or 0.0) - 40.0 * float(row["edge_r"] or 0.0),
            row["symbol"],
        )
    )
    momentum = [r for r in records if "MOMENTUM" in r["signals"]]
    near = [r for r in records if "PRE_BREAKOUT" in r["signals"] and "MOMENTUM" not in r["signals"]]
    ready = [r for r in records if r["status"] == "Ready to trade"]
    edged = sum(1 for r in records if r.get("edge_r") is not None)
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
            "with_measured_edge": edged,
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
        out = dict(payload)
        same_day = _scanned_on_ist_today(out)
        out["same_ist_day"] = same_day
        # Prior-day file is a SNAPSHOT, never "current live memory".
        out["records_status"] = "CURRENT_DAY" if same_day else "PRIOR_DAY_SNAPSHOT"
        return out
    except Exception:
        return None


def _scanned_on_ist_today(payload: Mapping[str, Any]) -> bool:
    stamp = str(payload.get("scanned_at") or "").strip()
    if not stamp:
        return False
    try:
        from core.market_clock import IST, today_ist
        dt = datetime.fromisoformat(stamp.replace("Z", "+00:00"))
        if dt.tzinfo is None:
            # Product scans historically stored UTC-ish ISO; treat naive as UTC.
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(IST).date() == today_ist()
    except Exception:
        return False


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
