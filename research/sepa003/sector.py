"""Versioned symbol → sector map. Unmapped stays UNKNOWN. No price-inferred labels."""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Mapping

import pandas as pd

from research.sepa003.constants import SECTOR_VERSION

# Contemporaneous large-cap overlay for names the NIFTY500 comment parser misses.
# Not a historical industry archive. sector_identity_pit = false.
_NIFTY50_OVERLAY: dict[str, str] = {
    "RELIANCE": "Energy & Power",
    "TCS": "IT / Software",
    "HDFCBANK": "Banking & Finance",
    "INFY": "IT / Software",
    "ICICIBANK": "Banking & Finance",
    "HINDUNILVR": "FMCG / Consumer Staples",
    "KOTAKBANK": "Banking & Finance",
    "LT": "Infrastructure & Construction",
    "SBIN": "Banking & Finance",
    "BHARTIARTL": "Telecom & Media",
    "AXISBANK": "Banking & Finance",
    "ASIANPAINT": "Paints",
    "MARUTI": "Auto & Auto Ancillary",
    "NESTLEIND": "FMCG / Consumer Staples",
    "ULTRACEMCO": "Cement",
    "BAJFINANCE": "NBFC",
    "WIPRO": "IT / Software",
    "HCLTECH": "IT / Software",
    "TECHM": "IT / Software",
    "SUNPHARMA": "Pharma & Healthcare",
    "TITAN": "Gems & Jewellery",
    "ADANIENT": "Metals & Mining",
    "ADANIPORTS": "Infrastructure & Construction",
    "BAJAJFINSV": "NBFC",
    "BPCL": "Energy & Power",
    "BRITANNIA": "FMCG / Consumer Staples",
    "CIPLA": "Pharma & Healthcare",
    "COALINDIA": "Metals & Mining",
    "DIVISLAB": "Pharma & Healthcare",
    "DRREDDY": "Pharma & Healthcare",
    "EICHERMOT": "Auto & Auto Ancillary",
    "GRASIM": "Cement",
    "HDFCLIFE": "Insurance",
    "HEROMOTOCO": "Auto & Auto Ancillary",
    "HINDALCO": "Metals & Mining",
    "INDUSINDBK": "Banking & Finance",
    "ITC": "FMCG / Consumer Staples",
    "JSWSTEEL": "Metals & Mining",
    "M&M": "Auto & Auto Ancillary",
    "NTPC": "Energy & Power",
    "ONGC": "Energy & Power",
    "POWERGRID": "Energy & Power",
    "SBILIFE": "Insurance",
    "SHRIRAMFIN": "NBFC",
    "TATACONSUM": "FMCG / Consumer Staples",
    "TATAMOTORS": "Auto & Auto Ancillary",
    "TATASTEEL": "Metals & Mining",
    "TRENT": "Consumer & Retail / Apparel",
    "VEDL": "Metals & Mining",
}

_NIFTY100_OVERLAY: dict[str, str] = {
    "ABB": "Engineering",
    "ADANIGREEN": "Energy & Power",
    "ADANIENSOL": "Energy & Power",
    "AMBUJACEM": "Cement",
    "AUROPHARMA": "Pharma & Healthcare",
    "BAJAJ-AUTO": "Auto & Auto Ancillary",
    "BALKRISIND": "Tyres",
    "BANDHANBNK": "Banking & Finance",
    "BANKBARODA": "Banking & Finance",
    "BEL": "Defence",
    "BERGEPAINT": "Paints",
    "BOSCHLTD": "Auto & Auto Ancillary",
    "CANBK": "Banking & Finance",
    "CHOLAFIN": "NBFC",
    "COLPAL": "FMCG / Consumer Staples",
    "DMART": "Consumer & Retail / Apparel",
    "GAIL": "Gas & Energy Distribution",
    "GODREJCP": "FMCG / Consumer Staples",
    "HAVELLS": "White Goods / Consumer Durables",
    "ICICIPRULI": "Insurance",
    "INDIGO": "Logistics & Transport",
    "IOC": "Energy & Power",
    "JUBLFOOD": "Hospitality",
    "LICI": "Insurance",
    "LUPIN": "Pharma & Healthcare",
    "MARICO": "FMCG / Consumer Staples",
    "MCDOWELL-N": "FMCG / Consumer Staples",
    "MUTHOOTFIN": "NBFC",
    "NAUKRI": "Digital / New Economy",
    "OBEROIRLTY": "Real Estate",
    "PAGEIND": "Textiles",
    "PIDILITIND": "Specialty Chemicals",
    "PNB": "Banking & Finance",
    "RECLTD": "Energy & Power",
    "SAIL": "Metals & Mining",
}

_SECTION_RE = re.compile(r"^\s*#\s*([A-Za-z][A-Za-z &/'\-]+)$")
_SYM_RE = re.compile(r'"([A-Z0-9&\-]+)"')


def parse_nifty500_comments() -> dict[str, str]:
    src = (Path(__file__).resolve().parents[2] / "data" / "nse_universe.py").read_text()
    start = src.find("NIFTY500")
    end = src.find("]))", start)
    block = src[start:end] if start >= 0 and end > start else ""
    sector = ""
    mapping: dict[str, str] = {}
    for line in block.splitlines():
        m = _SECTION_RE.match(line)
        if m:
            sector = m.group(1).strip()
            continue
        if sector:
            for sym in _SYM_RE.findall(line):
                mapping.setdefault(sym, sector)
    return mapping


def load_sector_map_v1() -> dict[str, Any]:
    comments = parse_nifty500_comments()
    overlay = {**_NIFTY100_OVERLAY, **_NIFTY50_OVERLAY}
    merged = dict(comments)
    overlay_only = {}
    for sym, sec in overlay.items():
        if sym not in merged:
            merged[sym] = sec
            overlay_only[sym] = sec
    return {
        "version": SECTOR_VERSION,
        "sector_identity_pit": False,
        "source": "nse_universe NIFTY500 comments + documented large-cap overlay",
        "map": merged,
        "n_mapped": len(merged),
        "n_from_comments": len(comments),
        "n_from_overlay": len(overlay_only),
        "never_infers_from_price": True,
    }


def sector_of(symbol: str, mapping: Mapping[str, str] | None = None) -> str:
    m = mapping if mapping is not None else load_sector_map_v1()["map"]
    return str(m.get(str(symbol).upper()) or "UNKNOWN")


def sector_context(
    symbol: str,
    as_of: str,
    frames: Mapping[str, pd.DataFrame],
    mapping: Mapping[str, str],
    *,
    lookback: int = 63,
    rs_threshold: float = 70.0,
    stock_rs: float | None = None,
) -> dict[str, Any]:
    """PIT sector returns among mapped members with bars ≤ as_of."""
    from research.sepa.frames import slice_as_of
    sec = sector_of(symbol, mapping)
    out = {
        "sector": sec,
        "sector_identity_pit": False,
        "sector_ret": None,
        "sector_rank": None,
        "n_sector_members": 0,
        "stock_vs_sector": None,
        "n_strong_in_group": None,
        "sector_rs": None,
    }
    if sec == "UNKNOWN":
        return out
    cutoff = pd.Timestamp(as_of)
    rets: dict[str, float] = {}
    strong = 0
    stock_ret = None
    for sym, lab in mapping.items():
        if lab != sec:
            continue
        hist = slice_as_of(frames.get(sym), cutoff)
        if hist is None or len(hist) < lookback + 1:
            continue
        c = pd.to_numeric(hist["close"], errors="coerce").dropna()
        if len(c) < lookback + 1:
            continue
        r = float(c.iloc[-1] / c.iloc[-lookback - 1] - 1.0)
        rets[sym] = r
        if stock_rs is not None and sym == str(symbol).upper() and float(stock_rs) >= rs_threshold:
            strong += 1
        elif stock_rs is None and r > 0.10:
            strong += 1
    out["n_sector_members"] = len(rets)
    if not rets:
        return out
    out["sector_ret"] = round(float(pd.Series(rets).median()), 6)
    ranked = sorted(rets.values(), reverse=True)
    if str(symbol).upper() in rets:
        stock_ret = rets[str(symbol).upper()]
        out["stock_vs_sector"] = round(stock_ret - out["sector_ret"], 6)
        out["sector_rank"] = int(sum(1 for x in ranked if x > stock_ret) + 1)
    # Cross-section of sector medians for a crude sector RS percentile
    return out


def sector_ranks_as_of(
    as_of: str,
    frames: Mapping[str, pd.DataFrame],
    mapping: Mapping[str, str],
    *,
    lookback: int = 63,
    min_members: int = 3,
) -> dict[str, float]:
    from research.sepa.frames import slice_as_of
    cutoff = pd.Timestamp(as_of)
    bucket: dict[str, list[float]] = {}
    for sym, sec in mapping.items():
        hist = slice_as_of(frames.get(sym), cutoff)
        if hist is None or len(hist) < lookback + 1:
            continue
        c = pd.to_numeric(hist["close"], errors="coerce").dropna()
        if len(c) < lookback + 1:
            continue
        bucket.setdefault(sec, []).append(float(c.iloc[-1] / c.iloc[-lookback - 1] - 1.0))
    med = {s: float(pd.Series(v).median()) for s, v in bucket.items() if len(v) >= min_members}
    if not med:
        return {}
    order = sorted(med, key=lambda s: med[s])
    n = len(order)
    return {s: 100.0 * i / max(n - 1, 1) for i, s in enumerate(order)}
