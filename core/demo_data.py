"""
Demo data provider — returns realistic NSE market data when live sources fail.
Used as final fallback in regime engine and scan pipeline.
"""
from __future__ import annotations

import random
from datetime import datetime, date

_SEED_DATE = date.today().toordinal()  # stable seed per day so data looks consistent


def _rng(offset: int = 0) -> random.Random:
    return random.Random(_SEED_DATE + offset)


# ---------------------------------------------------------------------------
# Regime / Index Data
# ---------------------------------------------------------------------------

DEMO_REGIME = {
    "nifty_price":      24_387.5,
    "nifty_change_1d":  0.68,
    "nifty_change_5d":  2.14,
    "vix":              13.4,
    "sma50":            23_850.0,
    "sma200":           22_310.0,
    "market_regime":    "TRENDING_BULL",
    "volatility_regime": "NORMAL",
    "breadth_strength": 72.0,
    "breadth_label":    "STRONG",
    "breakout_environment": "FAVORABLE",
    "risk_mode":        "NORMAL",
    "institutional_activity": "ACCUMULATION",
    "leading_sectors":  ["BANK", "IT", "AUTO"],
    "lagging_sectors":  ["REALTY", "METAL"],
    "rotation_mode":    "RISK_ON",
    "sector_returns":   {
        "BANK":   1.82,
        "IT":     1.44,
        "AUTO":   0.97,
        "PHARMA": 0.31,
        "FMCG":   0.12,
        "ENERGY": -0.44,
        "METAL":  -0.91,
        "REALTY": -1.23,
    },
    "regime_score":     76.0,
    "quality_multiplier": 1.15,
    "recommended_playbooks": ["VCP_BREAKOUT", "MOMENTUM_EXPANSION", "EARLY_LEADER"],
    "avoid_patterns":   ["MEAN_REVERSION", "FAILED_BREAKOUT_REVERSAL"],
    "timestamp":        datetime.now().strftime("%H:%M"),
}


# ---------------------------------------------------------------------------
# Stock universe demo candidates
# ---------------------------------------------------------------------------

_DEMO_STOCKS: list[dict] = [
    {"symbol": "HDFCBANK",  "price": 1_678.4,  "mcap_cr": 12_80_000, "avg_vol": 8_500_000,  "sector": "BANK"},
    {"symbol": "ICICIBANK", "price": 1_312.7,  "mcap_cr": 9_20_000,  "avg_vol": 11_200_000, "sector": "BANK"},
    {"symbol": "RELIANCE",  "price": 2_948.0,  "mcap_cr": 19_90_000, "avg_vol": 7_300_000,  "sector": "ENERGY"},
    {"symbol": "INFY",      "price": 1_511.2,  "mcap_cr": 6_30_000,  "avg_vol": 9_800_000,  "sector": "IT"},
    {"symbol": "TCS",       "price": 3_722.5,  "mcap_cr": 13_60_000, "avg_vol": 4_200_000,  "sector": "IT"},
    {"symbol": "TATAMOTORS","price": 938.6,    "mcap_cr": 3_45_000,  "avg_vol": 13_400_000, "sector": "AUTO"},
    {"symbol": "BAJFINANCE","price": 7_284.0,  "mcap_cr": 4_40_000,  "avg_vol": 3_100_000,  "sector": "BANK"},
    {"symbol": "AXISBANK",  "price": 1_147.3,  "mcap_cr": 3_55_000,  "avg_vol": 10_200_000, "sector": "BANK"},
    {"symbol": "LT",        "price": 3_612.8,  "mcap_cr": 5_10_000,  "avg_vol": 2_800_000,  "sector": "INFRA"},
    {"symbol": "WIPRO",     "price": 478.4,    "mcap_cr": 2_50_000,  "avg_vol": 7_600_000,  "sector": "IT"},
    {"symbol": "SUNPHARMA", "price": 1_684.9,  "mcap_cr": 4_03_000,  "avg_vol": 3_900_000,  "sector": "PHARMA"},
    {"symbol": "ULTRACEMCO","price": 10_842.0, "mcap_cr": 3_14_000,  "avg_vol": 430_000,    "sector": "CEMENT"},
    {"symbol": "TITAN",     "price": 3_418.5,  "mcap_cr": 3_05_000,  "avg_vol": 1_800_000,  "sector": "CONS"},
    {"symbol": "NESTLEIND", "price": 2_328.7,  "mcap_cr": 2_25_000,  "avg_vol": 620_000,    "sector": "FMCG"},
    {"symbol": "HCLTECH",   "price": 1_612.4,  "mcap_cr": 4_37_000,  "avg_vol": 4_900_000,  "sector": "IT"},
    {"symbol": "MARUTI",    "price": 12_384.0, "mcap_cr": 3_77_000,  "avg_vol": 560_000,    "sector": "AUTO"},
    {"symbol": "SBIN",      "price": 812.6,    "mcap_cr": 7_26_000,  "avg_vol": 22_000_000, "sector": "BANK"},
    {"symbol": "KOTAKBANK", "price": 1_984.3,  "mcap_cr": 3_96_000,  "avg_vol": 4_100_000,  "sector": "BANK"},
    {"symbol": "ASIANPAINT","price": 2_632.0,  "mcap_cr": 2_52_000,  "avg_vol": 1_400_000,  "sector": "CONS"},
    {"symbol": "POWERGRID", "price": 312.7,    "mcap_cr": 2_92_000,  "avg_vol": 8_700_000,  "sector": "ENERGY"},
]


def get_demo_stock(symbol: str) -> dict | None:
    for s in _DEMO_STOCKS:
        if s["symbol"] == symbol:
            return s
    return None


def get_all_demo_symbols() -> list[str]:
    return [s["symbol"] for s in _DEMO_STOCKS]


def make_demo_ohlcv(symbol: str, bars: int = 60) -> "list[dict]":
    """Returns list of OHLCV dicts (newest last) for the symbol."""
    stock = get_demo_stock(symbol)
    base_price = stock["price"] if stock else 1000.0
    avg_vol = stock["avg_vol"] if stock else 1_000_000

    rng = _rng(hash(symbol) % 1000)
    prices = []
    p = base_price * 0.85  # start 15% below current price
    for _ in range(bars):
        drift = rng.gauss(0.0008, 0.014)
        p = p * (1 + drift)
        rng2 = _rng(hash(symbol) % 500)
        spread = p * rng2.uniform(0.004, 0.018)
        o = p + rng.gauss(0, spread * 0.3)
        h = max(o, p) + rng.uniform(0, spread)
        l = min(o, p) - rng.uniform(0, spread)
        c = p
        vol = int(avg_vol * rng.uniform(0.5, 1.8))
        prices.append({"open": o, "high": h, "low": l, "close": c, "volume": vol})

    # force close to land near base_price
    if prices:
        scale = base_price / prices[-1]["close"]
        for bar in prices:
            for k in ("open", "high", "low", "close"):
                bar[k] *= scale

    return prices
