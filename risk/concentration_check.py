"""
Concentration Reality Check — before any trade, computes TRUE sector exposure.
"You already hold 14% in banking. This adds 8% more = 22% total. Limit: 25%."
Single line. Non-blocking. Just honest math.
"""
from __future__ import annotations

from typing import Optional

_SECTOR_MAP: dict[str, str] = {
    # Banking
    "HDFCBANK": "Banking", "ICICIBANK": "Banking", "KOTAKBANK": "Banking",
    "AXISBANK": "Banking", "SBIN": "Banking", "INDUSINDBK": "Banking",
    "BANDHANBNK": "Banking", "FEDERALBNK": "Banking", "IDFCFIRSTB": "Banking",
    # IT
    "INFY": "IT", "TCS": "IT", "WIPRO": "IT", "HCLTECH": "IT", "TECHM": "IT",
    "MPHASIS": "IT", "LTIM": "IT", "PERSISTENT": "IT", "COFORGE": "IT",
    # Energy
    "RELIANCE": "Energy", "ONGC": "Energy", "BPCL": "Energy",
    "IOC": "Energy", "GAIL": "Energy", "POWERGRID": "Energy", "NTPC": "Energy",
    # Auto
    "TATAMOTORS": "Auto", "MARUTI": "Auto", "BAJAJ-AUTO": "Auto", "M&M": "Auto",
    "EICHERMOT": "Auto", "HEROMOTOCO": "Auto", "TVSMOTOR": "Auto",
    # Pharma
    "SUNPHARMA": "Pharma", "DRREDDY": "Pharma", "CIPLA": "Pharma",
    "DIVISLAB": "Pharma", "BIOCON": "Pharma", "LUPIN": "Pharma", "AUROPHARMA": "Pharma",
    # Consumer / FMCG
    "TITAN": "Consumer", "NESTLEIND": "Consumer", "HINDUNILVR": "Consumer",
    "ITC": "Consumer", "BRITANNIA": "Consumer", "DABUR": "Consumer",
    "MARICO": "Consumer", "COLPAL": "Consumer",
    # Metals
    "TATASTEEL": "Metal", "HINDALCO": "Metal", "JSWSTEEL": "Metal",
    "SAIL": "Metal", "VEDL": "Metal", "COALINDIA": "Metal",
    # Financials (non-bank)
    "BAJFINANCE": "Financials", "BAJAJFINSV": "Financials", "HDFC": "Financials",
    "HDFCLIFE": "Financials", "ICICIPRU": "Financials", "SBILIFE": "Financials",
    # Realty
    "DLF": "Realty", "GODREJPROP": "Realty", "OBEROIRLTY": "Realty",
    # Infra / Capital Goods
    "LT": "Infra", "ULTRACEMCO": "Infra", "ADANIPORTS": "Infra",
    "SIEMENS": "Infra", "ABB": "Infra",
}

_DEFAULT_SECTOR_LIMIT = 25.0  # % of portfolio


def _get_sector(symbol: str) -> str:
    """Return sector for a symbol, defaulting to 'Other'."""
    return _SECTOR_MAP.get(symbol.upper(), "Other")


def compute_concentration(
    new_symbol: str,
    new_size_pct: float,
    open_positions: dict,
    sector_limit: float = _DEFAULT_SECTOR_LIMIT,
) -> dict:
    """
    Compute sector concentration after adding a new position.

    Parameters
    ----------
    new_symbol    : NSE symbol to be traded (e.g. "HDFCBANK")
    new_size_pct  : % of portfolio this new trade represents (0-100 scale)
    open_positions: {symbol: size_pct} of current holdings
    sector_limit  : hard cap on single-sector exposure (default 25%)

    Returns
    -------
    dict with keys: sector, existing_sector_pct, new_total_sector_pct,
                    sector_limit, exceeds_limit, warning,
                    total_capital_used, capital_remaining
    """
    sector = _get_sector(new_symbol)

    # Current sector exposure (sum of all open positions in same sector)
    existing_sector_pct = sum(
        size for sym, size in open_positions.items()
        if _get_sector(sym) == sector
    )

    new_total_sector_pct = existing_sector_pct + new_size_pct

    # Total capital already deployed
    total_capital_used = sum(open_positions.values()) + new_size_pct
    capital_remaining = max(0.0, 100.0 - total_capital_used)

    exceeds_limit = new_total_sector_pct > sector_limit

    warning: Optional[str] = None
    if exceeds_limit:
        warning = (
            f"{sector} exposure: {new_total_sector_pct:.0f}% after this trade. "
            f"Limit: {sector_limit:.0f}%."
        )
    elif new_total_sector_pct >= sector_limit * 0.8:
        warning = (
            f"{sector} exposure will reach {new_total_sector_pct:.0f}% "
            f"(approaching {sector_limit:.0f}% limit)."
        )

    return {
        "sector": sector,
        "existing_sector_pct": round(existing_sector_pct, 2),
        "new_total_sector_pct": round(new_total_sector_pct, 2),
        "sector_limit": sector_limit,
        "exceeds_limit": exceeds_limit,
        "warning": warning,
        "total_capital_used": round(total_capital_used, 2),
        "capital_remaining": round(capital_remaining, 2),
    }


def render_concentration_warning(
    symbol: str,
    size_pct: float,
    open_positions: dict,
    sector_limit: float = _DEFAULT_SECTOR_LIMIT,
) -> None:
    """
    Renders concentration warnings in Streamlit — silent when all clear.

    Shows:
    - Sector exposure warning if total would exceed 20% (approaching limit)
    - Portfolio deployment warning if total capital > 80%
    """
    import streamlit as st

    try:
        result = compute_concentration(symbol, size_pct, open_positions, sector_limit)
    except Exception:
        return  # Never break the order form

    # Sector concentration warning
    threshold_warn = sector_limit * 0.8  # warn at 80% of limit
    if result["new_total_sector_pct"] >= threshold_warn:
        sector = result["sector"]
        new_pct = result["new_total_sector_pct"]
        limit = result["sector_limit"]
        if result["exceeds_limit"]:
            st.warning(
                f"⚠️ {sector} exposure: {new_pct:.0f}% after this trade. "
                f"Limit: {limit:.0f}%. Consider reducing size."
            )
        else:
            st.warning(
                f"⚠️ {sector} exposure: {new_pct:.0f}% after this trade "
                f"(approaching {limit:.0f}% limit)."
            )

    # Overall portfolio deployment warning
    if result["total_capital_used"] > 80.0:
        st.warning(
            f"⚠️ Portfolio will be {result['total_capital_used']:.0f}% deployed "
            f"after this trade. Consider smaller size."
        )
