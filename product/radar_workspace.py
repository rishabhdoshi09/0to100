"""Market radar projections: Breakouts, Momentum and Long-Term Picks lanes.

Pure functions over persisted scan/long-term payloads — no duplicate scanning.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Callable, Mapping, Sequence

BREAKOUT_TAGS = frozenset({"BREAKOUT_52W", "BREAKOUT_RES", "GOLDEN_CROSS", "VOL_SQUEEZE"})
QUALITY_CLASSES = frozenset({"QUALITY_COMPOUNDER", "GARP_CANDIDATE", "QUALITY_BUT_EXPENSIVE"})


def _signals(row: Mapping[str, Any]) -> set[str]:
    return {str(item).upper() for item in (row.get("signals") or [])}


def _f(value: Any, default: float = 0.0) -> float:
    try:
        return float(value if value is not None else default)
    except (TypeError, ValueError):
        return float(default)


def classify_breakout_state(row: Mapping[str, Any]) -> str:
    """Deterministic breakout sub-state for retail scanner tables."""
    status = str(row.get("status", "") or "")
    if bool(row.get("chase_risk")):
        return "extended_after_breakout"
    signals = _signals(row)
    vol_ok = _f(row.get("volume_ratio") or row.get("rvol") or 0) >= 1.0 or "VOL_SQUEEZE" in signals
    if "PRE_BREAKOUT" in signals or status == "Watch for breakout":
        return "near_breakout"
    if signals & BREAKOUT_TAGS:
        if str(row.get("verdict", "")).upper() == "BUY" and status == "Ready to trade":
            if not vol_ok:
                return "breakout_without_volume"
            return "confirmed_breakout"
        if status == "Wait for pullback":
            return "failed_breakout"
        return "breakout_under_observation"
    if status == "Wait for pullback":
        return "failed_or_extended"
    if not signals and status in {"Watch", "Watch for breakout"}:
        return "insufficient_data"
    return "not_in_breakout_lane"


def classify_momentum_state(row: Mapping[str, Any]) -> str:
    """Momentum strength vs extension without inventing new scores."""
    if "MOMENTUM" not in _signals(row):
        return "not_momentum"
    if bool(row.get("chase_risk")):
        return "strong_but_extended"
    status = str(row.get("status", "") or "")
    score = _f(row.get("score"))
    mom5 = _f(row.get("momentum_5d"))
    mom20 = _f(row.get("momentum_20d") or row.get("momentum_21d"))
    vol = _f(row.get("atr_pct") or row.get("volatility"))
    if status == "Ready to trade" and score >= 65:
        if vol >= 4.0:
            return "high_volatility_momentum"
        return "strong_actionable"
    if mom5 > 0 and mom20 > 0 and score >= 60:
        return "steady_leadership"
    if mom5 > 0 and score >= 55:
        return "improving"
    if mom5 < 0:
        return "weakening"
    hist = _f(row.get("history_days") or row.get("sessions") or 0)
    if hist and hist < 120:
        return "insufficient_history"
    return "watch_momentum"


def default_sector_lookup(symbol: str) -> str:
    try:
        from scan.sector_heat import sector_of
        return str(sector_of(symbol) or "")
    except Exception:
        return ""


def enrich_scan_row(
    row: Mapping[str, Any],
    *,
    sector_lookup: Callable[[str], str] | None = None,
    scanned_at: str = "",
) -> dict[str, Any]:
    lookup = sector_lookup or default_sector_lookup
    symbol = str(row.get("symbol", "") or "").upper()
    enriched = dict(row)
    enriched["sector"] = str(row.get("sector") or lookup(symbol) or "—")
    enriched["breakout_state"] = classify_breakout_state(row)
    enriched["momentum_state"] = classify_momentum_state(row)
    enriched["setup_label"] = str(row.get("status") or row.get("verdict") or "Watch")
    enriched["freshness"] = scanned_at or "unknown"
    enriched["change_5d_pct"] = _f(row.get("momentum_5d"))
    enriched["relative_strength"] = _f(row.get("score"))
    enriched["risk_label"] = (
        "Chase risk"
        if bool(row.get("chase_risk"))
        else "Pullback wait"
        if str(row.get("status", "")) == "Wait for pullback"
        else "Normal"
    )
    reason = row.get("reasons") or row.get("why") or []
    enriched["reason"] = str(reason[0] if isinstance(reason, list) and reason else row.get("why") or "")
    return enriched


def enrich_long_term_row(row: Mapping[str, Any], *, scanned_at: str = "") -> dict[str, Any]:
    lookup = default_sector_lookup
    symbol = str(row.get("symbol", "") or "").upper()
    enriched = dict(row)
    enriched["sector"] = str(row.get("sector") or lookup(symbol) or "—")
    enriched["setup_label"] = str(row.get("classification") or "Unclassified")
    enriched["freshness"] = scanned_at or "unknown"
    enriched["reason"] = ", ".join((row.get("quality_factors") or [])[:2]) or "Long-term screen"
    enriched["risk_label"] = ", ".join((row.get("risk_flags") or [])[:1]) or "Review risks"
    cov = row.get("fundamental_coverage")
    enriched["coverage_pct"] = round(_f(cov) * 100, 1) if cov is not None else None
    return enriched


def build_radar_home(
    *,
    scan_payload: Mapping[str, Any] | None,
    long_term_payload: Mapping[str, Any] | None,
    market: Any,
    sector_lookup: Callable[[str], str] | None = None,
) -> dict[str, Any]:
    scan_rows = [dict(r) for r in (scan_payload or {}).get("records", []) or [] if isinstance(r, Mapping)]
    long_rows = [dict(r) for r in (long_term_payload or {}).get("records", []) or [] if isinstance(r, Mapping)]
    scan_at = str((scan_payload or {}).get("scanned_at", "") or "")
    lt_at = str((long_term_payload or {}).get("scanned_at", "") or "")

    enriched = [enrich_scan_row(r, sector_lookup=sector_lookup, scanned_at=scan_at) for r in scan_rows]

    breakouts = [
        r for r in enriched
        if r["breakout_state"] in {
            "confirmed_breakout", "near_breakout", "insufficient_confirmation", "extended_after_breakout",
        }
    ]
    breakouts.sort(key=lambda r: (
        r["breakout_state"] != "confirmed_breakout",
        r["breakout_state"] != "near_breakout",
        bool(r.get("chase_risk")),
        -_f(r.get("score")),
        r.get("symbol", ""),
    ))

    momentum = [r for r in enriched if "MOMENTUM" in _signals(r)]
    momentum.sort(key=lambda r: (bool(r.get("chase_risk")), -_f(r.get("score")), r.get("symbol", "")))

    long_picks = [
        enrich_long_term_row(r, scanned_at=lt_at)
        for r in long_rows
        if str(r.get("classification", "")) in QUALITY_CLASSES
    ]
    long_picks.sort(key=lambda r: (-_f(r.get("combined_score")), r.get("symbol", "")))

    health = str(getattr(market, "health", "") or (market or {}).get("health", "Unavailable") if isinstance(market, Mapping) else "Unavailable")
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "market_session": str(getattr(market, "trade_stance", "") or (market or {}).get("trade_stance", "") if isinstance(market, Mapping) else ""),
        "market_health": health,
        "breadth": str(getattr(market, "breadth", "") or (market or {}).get("breadth", "") if isinstance(market, Mapping) else ""),
        "nifty_change_1d": _f(getattr(market, "nifty_change_1d", 0) if not isinstance(market, Mapping) else market.get("nifty_change_1d")),
        "vix": _f(getattr(market, "vix", 0) if not isinstance(market, Mapping) else market.get("vix")),
        "leaders": list(getattr(market, "leaders", ()) or (market or {}).get("leaders", []) if isinstance(market, Mapping) else []),
        "laggards": list(getattr(market, "laggards", ()) or (market or {}).get("laggards", []) if isinstance(market, Mapping) else []),
        "scan_scanned_at": scan_at,
        "long_term_scanned_at": lt_at,
        "universe_size": int((scan_payload or {}).get("universe_size", 0) or 0),
        "lanes": {
            "breakouts": breakouts[:12],
            "momentum": momentum[:12],
            "long_term_picks": long_picks[:12],
        },
        "counts": {
            "breakouts": len(breakouts),
            "momentum": len(momentum),
            "long_term_picks": len(long_picks),
        },
        "best_setups": [],
        "best_setups_note": "",
    }


def enrich_scanner_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    scanned_at: str = "",
    sector_lookup: Callable[[str], str] | None = None,
) -> list[dict[str, Any]]:
    return [enrich_scan_row(dict(row), sector_lookup=sector_lookup, scanned_at=scanned_at) for row in rows]
