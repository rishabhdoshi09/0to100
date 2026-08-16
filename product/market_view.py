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


def _view_from_official_tape() -> RetailMarketView | None:
    """Desk tape from official NSE files. Never imports Yahoo."""
    from product.local_tape import read_official_tape

    tape = read_official_tape()
    if not tape.usable:
        return None
    breadth = tape.breadth or {}
    verdict = str(breadth.get("verdict") or "MIXED")
    strength = int(round(float(breadth.get("pct_above_50") or 50)))
    nifty_1d = float(tape.nifty_change_1d or 0.0)
    vix = float(tape.vix or 0.0)

    if verdict == "NARROW" or nifty_1d <= -1.0 or (vix and vix >= 18):
        health = "Weak"
    elif verdict == "HEALTHY" and nifty_1d >= 0 and (not vix or vix < 14):
        health = "Healthy"
    else:
        health = "Mixed"

    if health == "Weak" or verdict == "NARROW":
        stance = "Be selective. Breadth or the index is not supporting chase entries."
    elif health == "Healthy":
        stance = "Tape is constructive. Take only setups that are still intact on the live bar."
    else:
        stance = "Use normal caution. Prefer names still near their 20-day high."

    lead_text = ", ".join(tape.leaders[:3]) if tape.leaders else "no clear sector leader"
    lag_text = ", ".join(tape.laggards[:3]) if tape.laggards else "no clear laggard"
    breadth_label = verdict.title() if verdict else "Unknown"
    summary = (
        f"Market condition is {health.lower()}. Breadth is {breadth_label.lower()} "
        f"({strength}/100). Leading: {lead_text}. Lagging: {lag_text}."
    )
    if tape.as_of:
        summary = f"{summary} Prices as of {tape.as_of} EOD."
    return RetailMarketView(
        health=health,
        summary=summary,
        trade_stance=stance,
        breadth=f"{breadth_label} · {strength}/100",
        leaders=tape.leaders,
        laggards=tape.laggards,
        nifty_change_1d=nifty_1d,
        nifty_change_5d=float(tape.nifty_change_5d or 0.0),
        vix=vix,
        technical_details={
            "as_of": tape.as_of,
            "source": tape.source,
            "nifty_close": tape.nifty_close,
            "breadth": breadth,
            "sector_changes": tape.sector_changes,
        },
    )


def current_market_view() -> RetailMarketView:
    """Prefer official local tape. Regime/Yahoo is optional enrichment only."""
    local = _view_from_official_tape()
    if local is not None:
        return local
    from core.regime_engine import compute_regime
    return build_market_view(compute_regime())
