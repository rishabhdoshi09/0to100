"""Research buy / stop / target from real tape fields.

Same geometry as the unified scanner: stop = entry − 2×ATR, target = entry + 4×ATR.
ATR comes from an explicit atr/atr14, then atr_pct, then the long-term row's
vol_pct (daily return std already on file). Last resort is a 5% / 10% band.

Never invents a price. A scanned entry/stop/target that already forms a valid
plan wins. Missing price → no levels (dashes stay dashes).
"""
from __future__ import annotations

from typing import Any, Mapping

STOP_ATR = 2.0
TARGET_ATR = 4.0
STOP_PCT_FALLBACK = 0.05
TARGET_PCT_FALLBACK = 0.10


def _f(value: Any, default: float = 0.0) -> float:
    try:
        if value is None or value == "":
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _atr_rupees(row: Mapping[str, Any], price: float) -> tuple[float | None, str]:
    atr = _f(row.get("atr") if row.get("atr") is not None else row.get("atr14"))
    if atr > 0:
        return atr, "atr"
    atr_pct = _f(row.get("atr_pct"))
    if atr_pct > 0 and price > 0:
        return price * atr_pct / 100.0, "atr_pct"
    vol_pct = _f(row.get("vol_pct"))
    if vol_pct > 0 and price > 0:
        return price * vol_pct / 100.0, "vol_pct"
    return None, ""


def research_levels(row: Mapping[str, Any]) -> dict[str, Any]:
    """Return entry/stop/target plus upside from buy. None when price is missing."""
    price = _f(row.get("price") or row.get("close") or row.get("cmp"))
    existing_entry = _f(row.get("entry") or row.get("entry_price"))
    existing_stop = _f(row.get("stop") or row.get("stop_price"))
    existing_target = _f(row.get("target") or row.get("target_price"))
    entry = existing_entry if existing_entry > 0 else price

    if entry <= 0:
        return {
            "entry": None,
            "stop": None,
            "target": None,
            "levels_source": "",
            "upside_from_buy_pct": None,
        }

    complete_scan = (
        0 < existing_stop < entry
        and existing_target > entry
    )
    if complete_scan:
        return {
            "entry": round(entry, 2),
            "stop": round(existing_stop, 2),
            "target": round(existing_target, 2),
            "levels_source": "scan",
            "upside_from_buy_pct": round((existing_target / entry - 1.0) * 100.0, 1),
        }
    atr, atr_src = _atr_rupees(row, entry)
    if atr and atr > 0:
        stop = existing_stop if 0 < existing_stop < entry else round(entry - STOP_ATR * atr, 2)
        target = existing_target if existing_target > entry else round(entry + TARGET_ATR * atr, 2)
        source = atr_src
    else:
        stop = existing_stop if 0 < existing_stop < entry else round(entry * (1.0 - STOP_PCT_FALLBACK), 2)
        target = existing_target if existing_target > entry else round(entry * (1.0 + TARGET_PCT_FALLBACK), 2)
        source = "pct_fallback"

    if stop >= entry:
        stop = round(entry * (1.0 - STOP_PCT_FALLBACK), 2)
        source = "pct_fallback"
    if target <= entry:
        target = round(entry * (1.0 + TARGET_PCT_FALLBACK), 2)
        if source != "pct_fallback" and not atr:
            source = "pct_fallback"

    return {
        "entry": round(entry, 2),
        "stop": round(stop, 2),
        "target": round(target, 2),
        "levels_source": source,
        "upside_from_buy_pct": round((target / entry - 1.0) * 100.0, 1),
    }


def attach_research_levels(row: Mapping[str, Any]) -> dict[str, Any]:
    """Copy row and fill entry/stop/target when a real price exists."""
    out = dict(row)
    for key, value in research_levels(out).items():
        if value is not None and value != "":
            out[key] = value
    return out


def levels_tag(source: str) -> str:
    if source in {"atr", "atr_pct", "vol_pct"}:
        return "2×ATR stop"
    if source == "pct_fallback":
        return "5% stop"
    return ""
