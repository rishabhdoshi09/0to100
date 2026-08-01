"""Pure product projections for the QuantTerm professional workspace.

The module deliberately contains no frontend, network, scanner, broker or mutation
imports. It only projects already-persisted product state into the command center
and unified scanner. Backend engines remain the source of truth.
"""
from __future__ import annotations

from typing import Any, Mapping, Sequence


SCANNER_MODES = (
    "Momentum",
    "Conviction",
    "Breakouts",
    "Pre-Breakout",
    "Long-Term",
    "F&O",
    "Avoid",
)


def _get(source: Any, key: str, default: Any = None) -> Any:
    if isinstance(source, Mapping):
        return source.get(key, default)
    return getattr(source, key, default)


def _f(value: Any, default: float = 0.0) -> float:
    try:
        return float(value if value is not None else default)
    except (TypeError, ValueError):
        return float(default)


def _records(payload: Mapping[str, Any] | None) -> list[dict[str, Any]]:
    if not payload:
        return []
    return [dict(row) for row in payload.get("records", []) if isinstance(row, Mapping)]


def _signals(row: Mapping[str, Any]) -> set[str]:
    return {str(item).upper() for item in (row.get("signals") or [])}


def build_command_center_state(
    *,
    scan_payload: Mapping[str, Any] | None,
    long_term_payload: Mapping[str, Any] | None,
    paper: Any,
    autonomy: Mapping[str, Any] | None,
    market: Any,
) -> dict[str, Any]:
    """Build one truthful, deterministic snapshot for the command center."""
    scan_rows = _records(scan_payload)
    long_rows = _records(long_term_payload)
    autonomy = dict(autonomy or {})

    ready = [
        row for row in scan_rows
        if str(row.get("status", "")) == "Ready to trade"
        and not bool(row.get("chase_risk"))
    ]
    momentum = [row for row in scan_rows if "MOMENTUM" in _signals(row)]
    near = [row for row in scan_rows if "PRE_BREAKOUT" in _signals(row)]
    ready.sort(key=lambda row: (-_f(row.get("score")), str(row.get("symbol", ""))))

    quality_classes = {"QUALITY_COMPOUNDER", "GARP_CANDIDATE", "QUALITY_BUT_EXPENSIVE"}
    quality = [row for row in long_rows if str(row.get("classification", "")) in quality_classes]
    quality.sort(key=lambda row: (-_f(row.get("combined_score")), str(row.get("symbol", ""))))

    capital = _f(_get(paper, "capital", 0.0))
    equity = _f(_get(paper, "equity", capital), capital)
    paper_return = round(((equity / capital - 1.0) * 100.0), 6) if capital > 0 else 0.0
    open_positions = list(_get(paper, "open_positions", ()) or ())

    health = str(_get(market, "health", "Unavailable") or "Unavailable")
    leaders = tuple(str(item) for item in (_get(market, "leaders", ()) or ()))
    laggards = tuple(str(item) for item in (_get(market, "laggards", ()) or ()))

    insights: list[str] = []
    if health.lower() == "healthy":
        insights.append("Market regime supports selective risk-taking.")
    elif health.lower() == "weak":
        insights.append("Market regime is weak; capital protection takes priority.")
    else:
        insights.append("Market regime is mixed; demand stronger confirmation.")
    if leaders:
        insights.append("Leading sectors: " + ", ".join(leaders[:3]) + ".")
    if ready:
        insights.append(f"{len(ready)} entry-ready setup(s) passed the saved scan.")
    elif near:
        insights.append(f"No entry-ready setup; {len(near)} name(s) are near breakout.")
    else:
        insights.append("No qualifying setup is being forced into the shortlist.")
    if quality:
        insights.append(f"{len(quality)} long-horizon candidate(s) have usable quality coverage.")

    return {
        "market_health": health,
        "market_summary": str(_get(market, "summary", "Market context unavailable.")),
        "trade_stance": str(_get(market, "trade_stance", "Use normal caution.")),
        "breadth": str(_get(market, "breadth", "Unavailable")),
        "nifty_change_1d": _f(_get(market, "nifty_change_1d", 0.0)),
        "vix": _f(_get(market, "vix", 0.0)),
        "leaders": leaders,
        "laggards": laggards,
        "scan_universe": int(_f((scan_payload or {}).get("universe_size", 0))),
        "momentum_count": len(momentum),
        "ready_count": len(ready),
        "near_breakout_count": len(near),
        "long_term_count": len(quality),
        "fundamental_coverage_pct": _f((long_term_payload or {}).get("summary", {}).get("coverage_pct", 0.0)),
        "paper_capital": capital,
        "paper_equity": equity,
        "paper_return_pct": paper_return,
        "open_positions": open_positions,
        "open_position_count": len(open_positions),
        "open_risk": _f(_get(paper, "open_risk", 0.0)),
        "paper_enabled": bool(_get(paper, "enabled", False)),
        "autonomy_running": bool(autonomy.get("running", False)),
        "autonomy_state": str(autonomy.get("state", "UNKNOWN")),
        "autonomy_plain_state": str(autonomy.get("plain_state", "Supervisor status unavailable.")),
        "heartbeat_ist": str(autonomy.get("heartbeat_ist", "")),
        "top_setups": ready[:6] if ready else momentum[:6],
        "top_long_term": quality[:6],
        "insights": insights,
    }


def scanner_rows(
    mode: str,
    *,
    scan_payload: Mapping[str, Any] | None,
    long_term_payload: Mapping[str, Any] | None,
    conviction_rows: Sequence[Mapping[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Return ranked records for one unified scanner mode."""
    normalized = str(mode or "Momentum").strip().lower()
    scan_rows = _records(scan_payload)
    long_rows = _records(long_term_payload)

    if normalized == "long-term":
        rows = [dict(row, _source="long_term") for row in long_rows]
        rows.sort(key=lambda row: (-_f(row.get("combined_score")), str(row.get("symbol", ""))))
        return rows

    if normalized == "conviction":
        rows = [dict(row, _source="conviction") for row in (conviction_rows or [])]
        rows.sort(key=lambda row: (-_f(row.get("conviction_score")), str(row.get("symbol", ""))))
        return rows

    if normalized == "momentum":
        rows = [row for row in scan_rows if "MOMENTUM" in _signals(row)]
    elif normalized == "breakouts":
        breakout = {"BREAKOUT_52W", "BREAKOUT_RES", "GOLDEN_CROSS", "VOL_SQUEEZE"}
        rows = [row for row in scan_rows if _signals(row) & breakout]
    elif normalized == "pre-breakout":
        rows = [
            row for row in scan_rows
            if "PRE_BREAKOUT" in _signals(row)
            or str(row.get("status", "")) == "Watch for breakout"
        ]
    elif normalized == "f&o":
        rows = [row for row in scan_rows if bool(row.get("fno_available"))]
    elif normalized == "avoid":
        rows = [
            row for row in scan_rows
            if bool(row.get("chase_risk"))
            or str(row.get("status", "")) == "Wait for pullback"
        ]
        rows.extend(
            dict(row, _source="long_term")
            for row in long_rows
            if str(row.get("classification", "")) == "AVOID_REVIEW"
        )
    else:
        rows = list(scan_rows)

    projected = [dict(row, _source=row.get("_source", "market_scan")) for row in rows]
    projected.sort(
        key=lambda row: (
            bool(row.get("chase_risk")),
            -_f(row.get("score", row.get("combined_score", 0.0))),
            str(row.get("symbol", "")),
        )
    )
    return projected
