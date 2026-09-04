"""Simple defensible portfolio stress. Inspection, not prediction."""
from __future__ import annotations

from typing import Any, Mapping, Sequence

from product.portfolio_heat import measure


def _f(value: Any) -> float | None:
    try:
        if value in (None, ""):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _shock_positions(
    positions: Sequence[Mapping[str, Any]],
    *,
    gap_pct: float,
    sector: str = "",
) -> list[dict[str, Any]]:
    out = []
    for pos in positions:
        row = dict(pos)
        entry = _f(pos.get("entry") or pos.get("entry_price"))
        stop = _f(pos.get("stop") or pos.get("stop_price"))
        last = _f(pos.get("last") or pos.get("ltp") or entry)
        if last is None:
            continue
        shocked = last * (1.0 + gap_pct / 100.0)
        if sector and str(pos.get("sector") or "").lower() != sector.lower():
            shocked = last
        row["last"] = shocked
        row["gap_hit_stop"] = bool(stop is not None and shocked <= stop)
        row["gap_pct_applied"] = gap_pct if shocked != last else 0.0
        out.append(row)
    return out


def run_scenarios(
    positions: Sequence[Mapping[str, Any]],
    *,
    capital: float,
) -> dict[str, Any]:
    """Market gap, sector gap, correlated stops, vol expansion."""
    base = measure(positions, capital=capital)
    market = _shock_positions(positions, gap_pct=-5.0)
    sector = ""
    if positions:
        sector = str(positions[0].get("sector") or "")
    sector_gap = _shock_positions(positions, gap_pct=-8.0, sector=sector) if sector else []
    corr_stops = sum(1 for p in market if p.get("gap_hit_stop"))
    vol = _shock_positions(positions, gap_pct=-3.0)
    return {
        "base_heat": base,
        "market_gap_down_5pct": {
            "stops_hit": sum(1 for p in market if p.get("gap_hit_stop")),
            "n": len(market),
        },
        "sector_gap_8pct": {
            "sector": sector,
            "stops_hit": sum(1 for p in sector_gap if p.get("gap_hit_stop")),
            "n": len(sector_gap),
        },
        "correlated_stops": {
            "names_that_would_stop": corr_stops,
            "note": "If names are correlated, a market gap can take several stops together.",
        },
        "volatility_expansion": {
            "stops_hit": sum(1 for p in vol if p.get("gap_hit_stop")),
            "note": "Wider ranges make ATR-fallback stops more likely to fill.",
        },
        "not_a_forecast": True,
        "live_locked": True,
    }
