"""Plain-language projection of the existing market-regime engine."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping


@dataclass(frozen=True)
class RetailMarketView:
    health: str
    summary: str
    trade_stance: str
    breadth: str
    leaders: tuple[str, ...]
    laggards: tuple[str, ...]
    nifty_change_1d: float
    nifty_change_5d: float
    vix: float
    technical_details: dict


def _get(source: Any, name: str, default: Any = None) -> Any:
    if isinstance(source, Mapping):
        return source.get(name, default)
    return getattr(source, name, default)


def build_market_view(regime: Any) -> RetailMarketView:
    score = float(_get(regime, "regime_score", 50.0) or 50.0)
    risk = str(_get(regime, "risk_mode", "NEUTRAL") or "NEUTRAL")
    breakout = str(_get(regime, "breakout_environment", "NEUTRAL") or "NEUTRAL")
    breadth_label = str(_get(regime, "breadth_label", "NEUTRAL") or "NEUTRAL")
    breadth_strength = int(_get(regime, "breadth_strength", 50) or 0)
    leaders = tuple(str(x) for x in (_get(regime, "leading_sectors", []) or []))
    laggards = tuple(str(x) for x in (_get(regime, "lagging_sectors", []) or []))

    if score >= 68 and risk != "RISK_OFF":
        health = "Healthy"
    elif score <= 35 or risk == "RISK_OFF":
        health = "Weak"
    else:
        health = "Mixed"

    if risk == "RISK_OFF":
        stance = "New paper trades should be selective or paused; protecting capital comes first."
    elif breakout == "FAVORABLE":
        stance = "New paper trades are allowed when stock-level evidence and safety checks also pass."
    elif breakout == "UNFAVORABLE":
        stance = "Avoid chasing breakouts; wait for cleaner entries or pullbacks."
    else:
        stance = "Use normal caution and take only the clearest setups."

    lead_text = ", ".join(leaders[:3]) if leaders else "no clear sector leader"
    lag_text = ", ".join(laggards[:3]) if laggards else "no clear laggard"
    summary = (
        f"Market condition is {health.lower()}. Breadth is {breadth_label.lower()} "
        f"({breadth_strength}/100). Leading: {lead_text}. Lagging: {lag_text}."
    )
    return RetailMarketView(
        health=health,
        summary=summary,
        trade_stance=stance,
        breadth=f"{breadth_label.title()} · {breadth_strength}/100",
        leaders=leaders,
        laggards=laggards,
        nifty_change_1d=float(_get(regime, "nifty_change_1d", 0.0) or 0.0),
        nifty_change_5d=float(_get(regime, "nifty_change_5d", 0.0) or 0.0),
        vix=float(_get(regime, "vix", 0.0) or 0.0),
        technical_details={
            "market_regime": _get(regime, "market_regime", ""),
            "volatility_regime": _get(regime, "volatility_regime", ""),
            "risk_mode": risk,
            "breakout_environment": breakout,
            "institutional_activity": _get(regime, "institutional_activity", ""),
            "regime_score": score,
            "regime_confidence": _get(regime, "regime_confidence", 0.0),
            "quality_multiplier": _get(regime, "quality_multiplier", 1.0),
            "timestamp": _get(regime, "timestamp", ""),
        },
    )


def current_market_view(*, allow_network: bool = True) -> RetailMarketView:
    from core import regime_engine as RE
    view = build_market_view(RE.compute_regime(allow_network=allow_network))
    sources = {
        str(k): str(v)
        for k, v in dict(getattr(RE, "_LAST_SOURCES", {}) or {}).items()
    }
    primary = "kite" if any(v == "kite" for v in sources.values()) else (
        "nse_index_store" if any(v == "nse_index_store" for v in sources.values()) else "unavailable"
    )
    details = dict(view.technical_details or {})
    details["data_sources"] = sources
    details["primary_source"] = primary
    details["yahoo_fallback"] = any(v == "yfinance" for v in sources.values())
    return RetailMarketView(
        health=view.health,
        summary=view.summary,
        trade_stance=view.trade_stance,
        breadth=view.breadth,
        leaders=view.leaders,
        laggards=view.laggards,
        nifty_change_1d=view.nifty_change_1d,
        nifty_change_5d=view.nifty_change_5d,
        vix=view.vix,
        technical_details=details,
    )
