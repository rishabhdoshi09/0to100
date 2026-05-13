"""
regime_engine.py — 5-dimension market regime classifier for NSE India.

Public API:
    compute_regime() -> RegimeState

All data sourced via yfinance. Sector fetches are parallelised.
Module-level cache with 15-minute TTL (no Streamlit dependency).
All external calls degrade gracefully — never raise.
"""

from __future__ import annotations

import time
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Optional

import numpy as np
import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

NIFTY_TICKER = "^NSEI"
VIX_TICKER = "^INDIAVIX"

SECTOR_TICKERS: dict[str, str] = {
    "IT":      "^CNXIT",
    "BANK":    "^NSEBANK",
    "AUTO":    "^CNXAUTO",
    "PHARMA":  "^CNXPHARMA",
    "FMCG":    "^CNXFMCG",
    "METAL":   "^CNXMETAL",
    "ENERGY":  "^CNXENERGY",
    "REALTY":  "^CNXREALTY",
}

OFFENSIVE_SECTORS = {"IT", "AUTO", "METAL"}
DEFENSIVE_SECTORS = {"PHARMA", "FMCG", "ENERGY"}

CACHE_TTL_SECS = 15 * 60  # 15 minutes

_CACHE: dict = {}          # keys: "regime_state", "timestamp"

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class RegimeState:
    # Primary dimensions
    market_regime: str              # TRENDING_BULL | TRENDING_BEAR | CHOPPY | COMPRESSION | EXPANSION | DISTRIBUTION
    volatility_regime: str          # LOW_VOL_COMPRESSION | NORMAL | TREND_VOLATILITY | ELEVATED | PANIC
    breadth_strength: int           # 0-100
    breadth_label: str              # STRONG | NEUTRAL | WEAK
    breakout_environment: str       # FAVORABLE | NEUTRAL | UNFAVORABLE
    risk_mode: str                  # RISK_ON | RISK_OFF | NEUTRAL
    institutional_activity: str     # ACCUMULATION | DISTRIBUTION | NEUTRAL | RISK_ON | RISK_OFF

    # Sector rotation
    leading_sectors: list[str]
    lagging_sectors: list[str]
    rotation_mode: str              # OFFENSIVE | DEFENSIVE | MIXED
    sector_returns: dict[str, float]

    # Nifty raw metrics
    nifty_price: float
    nifty_change_1d: float
    nifty_change_5d: float
    sma50: float
    sma200: float
    vix: float

    # Derived / actionable
    regime_score: float             # 0-100, overall bullishness
    quality_multiplier: float       # applied to downstream setup scores
    recommended_playbooks: list[str]
    avoid_patterns: list[str]

    # Meta
    timestamp: str
    data_age_mins: int

    # ------------------------------------------------------------------
    def to_dict(self) -> dict:
        return asdict(self)

    def summary_line(self) -> str:
        return (
            f"[{self.timestamp[:16]}] "
            f"Regime={self.market_regime} | "
            f"Vol={self.volatility_regime} | "
            f"Breadth={self.breadth_label}({self.breadth_strength}) | "
            f"Risk={self.risk_mode} | "
            f"Score={self.regime_score:.1f} | "
            f"QM={self.quality_multiplier:.2f}"
        )


# ---------------------------------------------------------------------------
# yfinance helpers
# ---------------------------------------------------------------------------

def _fetch_ohlcv(ticker: str, period: str = "1y") -> Optional[pd.DataFrame]:
    """Download OHLCV; return None on failure."""
    try:
        df = yf.download(ticker, period=period, progress=False, auto_adjust=True)
        if df is None or df.empty:
            logger.warning("No data for %s", ticker)
            return None
        df.index = pd.to_datetime(df.index)
        df.sort_index(inplace=True)
        return df
    except Exception as exc:
        logger.warning("Failed to fetch %s: %s", ticker, exc)
        return None


def _latest_close(df: pd.DataFrame) -> float:
    return float(df["Close"].iloc[-1])


def _sma(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window, min_periods=window).mean()


def _ema(series: pd.Series, window: int) -> pd.Series:
    return series.ewm(span=window, adjust=False).mean()


def _atr(df: pd.DataFrame, window: int = 14) -> pd.Series:
    high = df["High"]
    low = df["Low"]
    close = df["Close"]
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(span=window, adjust=False).mean()


def _adx(df: pd.DataFrame, window: int = 14) -> float:
    """Return scalar current ADX value; NaN on failure."""
    try:
        high = df["High"]
        low = df["Low"]
        close = df["Close"]

        plus_dm = high.diff()
        minus_dm = -low.diff()
        plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0.0)
        minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0.0)

        atr_series = _atr(df, window)
        smoothed_plus = plus_dm.ewm(span=window, adjust=False).mean()
        smoothed_minus = minus_dm.ewm(span=window, adjust=False).mean()

        pdi = 100 * smoothed_plus / atr_series.replace(0, np.nan)
        mdi = 100 * smoothed_minus / atr_series.replace(0, np.nan)
        dx = 100 * (pdi - mdi).abs() / (pdi + mdi).replace(0, np.nan)
        adx_series = dx.ewm(span=window, adjust=False).mean()
        return float(adx_series.iloc[-1])
    except Exception:
        return float("nan")


def _bollinger_width(close: pd.Series, window: int = 20) -> pd.Series:
    ma = _sma(close, window)
    std = close.rolling(window, min_periods=window).std()
    upper = ma + 2 * std
    lower = ma - 2 * std
    return (upper - lower) / ma.replace(0, np.nan)


# ---------------------------------------------------------------------------
# A. Market Regime
# ---------------------------------------------------------------------------

def _classify_market_regime(df: pd.DataFrame) -> tuple[str, float, float, float, float, float]:
    """Return (regime_label, price, chg1d, chg5d, sma50, sma200)."""
    fallback = ("CHOPPY", float("nan"), float("nan"), float("nan"), float("nan"), float("nan"))
    if df is None or len(df) < 210:
        return fallback

    close = df["Close"].squeeze()
    volume = df["Volume"].squeeze()

    price = float(close.iloc[-1])
    chg1d = float((close.iloc[-1] / close.iloc[-2] - 1) * 100) if len(close) > 1 else 0.0
    chg5d = float((close.iloc[-1] / close.iloc[-6] - 1) * 100) if len(close) > 5 else 0.0

    sma50_s = _sma(close, 50)
    sma200_s = _sma(close, 200)
    sma50 = float(sma50_s.iloc[-1])
    sma200 = float(sma200_s.iloc[-1])

    # SMA50 slope: compare today vs 5 sessions ago
    sma50_slope = float(sma50_s.iloc[-1] - sma50_s.iloc[-6]) if not sma50_s.iloc[-6:].isna().any() else 0.0

    adx_val = _adx(df)
    high_52w = float(close.rolling(252, min_periods=200).max().iloc[-1])
    atr_series = _atr(df)
    bb_width = _bollinger_width(close)

    # ATR contracting: current ATR < 5-session avg ATR * 0.9
    atr_contracting = (
        float(atr_series.iloc[-1]) < float(atr_series.iloc[-6:-1].mean()) * 0.90
        if len(atr_series) > 6 else False
    )

    # BB width percentile over last 252 sessions
    bb_recent = bb_width.dropna().iloc[-252:]
    bb_current = float(bb_width.iloc[-1]) if not np.isnan(float(bb_width.iloc[-1])) else 0.0
    bb_pct20 = float(bb_recent.quantile(0.20)) if len(bb_recent) >= 50 else float("nan")

    # BB width expanding vs 10-session average
    bb_10avg = float(bb_width.iloc[-11:-1].mean()) if len(bb_width) > 11 else float("nan")
    bb_expanding = (bb_current > bb_10avg * 1.30) if not np.isnan(bb_10avg) else False

    # Volume on up vs down days (last 20 sessions)
    recent = df.iloc[-20:]
    up_days = recent[recent["Close"].squeeze() >= recent["Open"].squeeze()]
    dn_days = recent[recent["Close"].squeeze() < recent["Open"].squeeze()]
    avg_vol_up = float(up_days["Volume"].squeeze().mean()) if len(up_days) > 0 else 0.0
    avg_vol_dn = float(dn_days["Volume"].squeeze().mean()) if len(dn_days) > 0 else 0.0
    volume_on_down = avg_vol_dn > avg_vol_up * 1.10

    # Determine regime
    if (price > sma50 > sma200
            and sma50_slope > 0
            and adx_val > 25
            and price > high_52w * 0.90):
        regime = "TRENDING_BULL"

    elif price < sma50 < sma200 and sma50_slope < 0:
        regime = "TRENDING_BEAR"

    elif (not np.isnan(bb_pct20)
          and bb_current <= bb_pct20
          and atr_contracting):
        regime = "COMPRESSION"

    elif bb_expanding and abs(chg5d) > 1.5:
        regime = "EXPANSION"

    elif (price < sma50
          and volume_on_down
          and sma50_slope <= 0):
        regime = "DISTRIBUTION"

    else:
        # CHOPPY: price between SMA50/SMA200 OR ADX < 20 for 10+ sessions
        adx_recent_low = adx_val < 20  # simplified; ADX already smoothed
        price_between = min(sma50, sma200) <= price <= max(sma50, sma200)
        regime = "CHOPPY" if (price_between or adx_recent_low) else "CHOPPY"

    return regime, price, chg1d, chg5d, sma50, sma200


# ---------------------------------------------------------------------------
# B. Volatility Regime
# ---------------------------------------------------------------------------

def _classify_volatility(vix_df: Optional[pd.DataFrame]) -> tuple[str, float]:
    if vix_df is None or vix_df.empty:
        return "NORMAL", float("nan")

    close = vix_df["Close"].squeeze()
    vix = float(close.iloc[-1])
    ma5 = float(_sma(close, 5).iloc[-1]) if len(close) >= 5 else vix

    if vix > 30:
        label = "PANIC"
    elif vix > 24:
        label = "ELEVATED"
    elif vix > 18:
        label = "TREND_VOLATILITY"
    elif vix > 14:
        label = "NORMAL"
    else:
        label = "LOW_VOL_COMPRESSION" if vix < ma5 else "NORMAL"

    return label, vix


# ---------------------------------------------------------------------------
# C. Breadth
# ---------------------------------------------------------------------------

def _compute_breadth(sector_data: dict[str, Optional[pd.DataFrame]]) -> tuple[int, str, float]:
    """Return (breadth_score 0-100, label, advance_decline_proxy)."""
    scores: list[float] = []
    advances = 0
    declines = 0

    for name, df in sector_data.items():
        if df is None or len(df) < 55:
            continue
        close = df["Close"].squeeze()
        price = float(close.iloc[-1])
        ma50 = float(_sma(close, 50).iloc[-1])
        ret5d = float((close.iloc[-1] / close.iloc[-6] - 1)) if len(close) > 5 else 0.0

        above_50d = 1.0 if price > ma50 else 0.0
        positive_5d = 1.0 if ret5d > 0 else 0.0
        # sector score 0-1
        scores.append((above_50d + positive_5d) / 2.0)

        if ret5d > 0:
            advances += 1
        else:
            declines += 1

    if not scores:
        return 50, "NEUTRAL", 1.0

    raw = float(np.mean(scores)) * 100  # 0-100
    breadth_score = int(round(raw))

    if breadth_score > 65:
        label = "STRONG"
    elif breadth_score >= 40:
        label = "NEUTRAL"
    else:
        label = "WEAK"

    ad_proxy = advances / max(declines, 1)
    return breadth_score, label, ad_proxy


# ---------------------------------------------------------------------------
# D. Sector Rotation
# ---------------------------------------------------------------------------

def _classify_sector_rotation(
    sector_data: dict[str, Optional[pd.DataFrame]]
) -> tuple[list[str], list[str], str, dict[str, float]]:
    """Return (leaders, laggards, rotation_mode, sector_returns)."""
    returns: dict[str, float] = {}
    for name, df in sector_data.items():
        if df is None or len(df) < 7:
            returns[name] = 0.0
            continue
        close = df["Close"].squeeze()
        ret5d = float((close.iloc[-1] / close.iloc[-6] - 1) * 100) if len(close) > 5 else 0.0
        returns[name] = round(ret5d, 3)

    ranked = sorted(returns, key=lambda s: returns[s], reverse=True)
    leaders = ranked[:3]
    laggards = ranked[-3:]

    leaders_set = set(leaders)
    offensive_leading = leaders_set & OFFENSIVE_SECTORS
    defensive_leading = leaders_set & DEFENSIVE_SECTORS

    if len(offensive_leading) >= 2:
        rotation_mode = "OFFENSIVE"
    elif len(defensive_leading) >= 2:
        rotation_mode = "DEFENSIVE"
    else:
        rotation_mode = "MIXED"

    return leaders, laggards, rotation_mode, returns


# ---------------------------------------------------------------------------
# E. Institutional Activity
# ---------------------------------------------------------------------------

def _classify_institutional(
    df: pd.DataFrame,
    market_regime: str,
    vix: float,
    sma50: float,
    sma200: float,
) -> str:
    if df is None or len(df) < 20:
        return "NEUTRAL"

    close = df["Close"].squeeze()
    high = df["High"].squeeze()
    volume = df["Volume"].squeeze()

    # RISK_ON / RISK_OFF conditions (use broad signals)
    price = float(close.iloc[-1])
    high_52w = float(close.rolling(252, min_periods=200).max().iloc[-1])
    low_52w = float(close.rolling(252, min_periods=200).min().iloc[-1])
    range_52w = high_52w - low_52w
    in_upper_half = price > (low_52w + range_52w * 0.5) if range_52w > 0 else False

    vix_declining = False
    if not np.isnan(vix):
        vix_declining = True  # placeholder; actual VIX trend computed in vol regime

    if sma50 > sma200 and in_upper_half and (np.isnan(vix) or vix < 22):
        return "RISK_ON"

    if sma50 < sma200 or (not np.isnan(vix) and vix > 22):
        return "RISK_OFF"

    # ACCUMULATION / DISTRIBUTION from price + volume micro-structure (last 20 bars)
    recent = df.iloc[-20:]
    recent_close = recent["Close"].squeeze()
    recent_open = recent["Open"].squeeze()
    recent_volume = recent["Volume"].squeeze()
    recent_high = recent["High"].squeeze()

    up_mask = recent_close >= recent_open
    dn_mask = ~up_mask
    avg_vol_up = float(recent_volume[up_mask].mean()) if up_mask.sum() > 0 else 0.0
    avg_vol_dn = float(recent_volume[dn_mask].mean()) if dn_mask.sum() > 0 else 0.0

    # Higher lows proxy: last 10 lows trending up
    lows_10 = recent["Low"].squeeze().iloc[-10:]
    higher_lows = (lows_10.diff().dropna() > 0).sum() >= 6

    # Lower highs proxy
    highs_10 = recent_high.iloc[-10:]
    lower_highs = (highs_10.diff().dropna() < 0).sum() >= 6

    vol_up_dominant = avg_vol_up > avg_vol_dn * 1.05
    vol_dn_dominant = avg_vol_dn > avg_vol_up * 1.10

    # Daily range compression (accumulation signature)
    daily_range = (recent_high - recent["Low"].squeeze())
    range_contracting = float(daily_range.iloc[-5:].mean()) < float(daily_range.iloc[-15:-5].mean()) * 0.90

    if higher_lows and vol_up_dominant and range_contracting:
        return "ACCUMULATION"
    if lower_highs and vol_dn_dominant:
        return "DISTRIBUTION"

    return "NEUTRAL"


# ---------------------------------------------------------------------------
# Scoring and playbooks
# ---------------------------------------------------------------------------

def _compute_regime_score(
    market_regime: str,
    volatility_regime: str,
    breadth_score: int,
    institutional: str,
) -> float:
    """Return 0-100 overall bullishness score."""
    market_map = {
        "TRENDING_BULL": 85,
        "EXPANSION":     70,
        "CHOPPY":        50,
        "COMPRESSION":   45,
        "DISTRIBUTION":  25,
        "TRENDING_BEAR": 15,
    }
    vol_map = {
        "LOW_VOL_COMPRESSION": 75,
        "NORMAL":              65,
        "TREND_VOLATILITY":    55,
        "ELEVATED":            35,
        "PANIC":               10,
    }
    inst_map = {
        "RISK_ON":        80,
        "ACCUMULATION":   70,
        "NEUTRAL":        50,
        "DISTRIBUTION":   30,
        "RISK_OFF":       20,
    }
    m = market_map.get(market_regime, 50)
    v = vol_map.get(volatility_regime, 50)
    b = breadth_score
    i = inst_map.get(institutional, 50)

    score = m * 0.35 + v * 0.20 + b * 0.25 + i * 0.20
    return round(min(max(score, 0), 100), 2)


def _quality_multiplier(market_regime: str, volatility_regime: str) -> float:
    base = {
        "TRENDING_BULL": 1.25,
        "EXPANSION":     1.10,
        "CHOPPY":        0.90,
        "COMPRESSION":   0.85,
        "DISTRIBUTION":  0.75,
        "TRENDING_BEAR": 0.65,
    }.get(market_regime, 0.90)

    vol_adj = {
        "LOW_VOL_COMPRESSION": 0.05,
        "NORMAL":              0.00,
        "TREND_VOLATILITY":   -0.05,
        "ELEVATED":           -0.10,
        "PANIC":              -0.20,
    }.get(volatility_regime, 0.0)

    return round(min(max(base + vol_adj, 0.50), 1.40), 2)


def _derive_playbooks(
    market_regime: str,
    volatility_regime: str,
    breadth_label: str,
    rotation_mode: str,
    institutional: str,
) -> tuple[list[str], list[str]]:
    recommended: list[str] = []
    avoid: list[str] = []

    bull_vol = market_regime == "TRENDING_BULL"
    low_vol = volatility_regime in ("LOW_VOL_COMPRESSION", "NORMAL")
    strong_breadth = breadth_label == "STRONG"
    risk_on = institutional in ("RISK_ON", "ACCUMULATION")

    if bull_vol and low_vol:
        recommended += ["VCP_BREAKOUT", "MOMENTUM_EXPANSION", "EARLY_LEADER"]
        avoid += ["MEAN_REVERSION", "COUNTER_TREND_SHORT"]

    if market_regime == "COMPRESSION" and risk_on:
        recommended += ["VCP_BREAKOUT", "ACCUMULATION_BREAKOUT"]
        avoid += ["MOMENTUM_CHASING", "LATE_MOMENTUM"]

    if market_regime == "TRENDING_BEAR":
        recommended += ["FAILED_BREAKOUT_REVERSAL", "MEAN_REVERSION"]
        avoid += ["BREAKOUT_BUY", "MOMENTUM_EXPANSION", "VCP_BREAKOUT"]

    if market_regime == "EXPANSION":
        recommended += ["MOMENTUM_EXPANSION", "SECTOR_BREAKOUT"]
        avoid += ["MEAN_REVERSION"]

    if market_regime == "CHOPPY":
        recommended += ["RANGE_FADE", "MEAN_REVERSION"]
        avoid += ["MOMENTUM_EXPANSION", "BREAKOUT_BUY"]

    if market_regime == "DISTRIBUTION":
        recommended += ["DEFENSIVE_POSITIONING", "CASH_CONSERVATION"]
        avoid += ["BREAKOUT_BUY", "VCP_BREAKOUT", "EARLY_LEADER"]

    if rotation_mode == "OFFENSIVE" and bull_vol:
        recommended += ["SECTOR_MOMENTUM"]
    elif rotation_mode == "DEFENSIVE":
        recommended += ["DEFENSIVE_ROTATION"]
        avoid += ["HIGH_BETA_MOMENTUM"]

    if strong_breadth and bull_vol:
        if "BROAD_MARKET_LONG" not in recommended:
            recommended.append("BROAD_MARKET_LONG")

    if volatility_regime == "PANIC":
        avoid += ["BREAKOUT_BUY", "MOMENTUM_EXPANSION"]
        recommended = [p for p in recommended if p not in avoid]
        recommended.insert(0, "VOLATILITY_MEAN_REVERSION")

    # deduplicate preserving order
    seen: set[str] = set()
    rec_dedup: list[str] = []
    for p in recommended:
        if p not in seen:
            seen.add(p)
            rec_dedup.append(p)

    seen2: set[str] = set()
    avd_dedup: list[str] = []
    for p in avoid:
        if p not in seen2 and p not in set(rec_dedup):
            seen2.add(p)
            avd_dedup.append(p)

    return rec_dedup, avd_dedup


def _breakout_environment(
    market_regime: str,
    volatility_regime: str,
    breadth_label: str,
) -> str:
    favorable = (
        breadth_label == "STRONG"
        and volatility_regime in ("NORMAL", "LOW_VOL_COMPRESSION")
        and market_regime in ("TRENDING_BULL", "EXPANSION")
    )
    unfavorable = (
        market_regime in ("TRENDING_BEAR", "DISTRIBUTION")
        or volatility_regime in ("ELEVATED", "PANIC")
        or breadth_label == "WEAK"
    )
    if favorable:
        return "FAVORABLE"
    if unfavorable:
        return "UNFAVORABLE"
    return "NEUTRAL"


def _risk_mode(
    sma50: float,
    sma200: float,
    vix: float,
    institutional: str,
) -> str:
    if institutional in ("RISK_ON",):
        return "RISK_ON"
    if institutional in ("RISK_OFF",):
        return "RISK_OFF"
    if not np.isnan(sma50) and not np.isnan(sma200):
        if sma50 > sma200 and (np.isnan(vix) or vix < 22):
            return "RISK_ON"
        if sma50 < sma200 or (not np.isnan(vix) and vix > 22):
            return "RISK_OFF"
    return "NEUTRAL"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compute_regime() -> RegimeState:
    """
    Compute and return the current 5-dimension market regime.
    Results are cached for 15 minutes.
    """
    now = time.time()
    cached = _CACHE.get("regime_state")
    cached_ts = _CACHE.get("timestamp", 0.0)

    if cached is not None and (now - cached_ts) < CACHE_TTL_SECS:
        return cached

    # ---- parallel data fetch ------------------------------------------------
    tickers_to_fetch: dict[str, str] = {"NIFTY": NIFTY_TICKER, "VIX": VIX_TICKER}
    tickers_to_fetch.update({name: tick for name, tick in SECTOR_TICKERS.items()})

    raw_data: dict[str, Optional[pd.DataFrame]] = {}

    def _fetch(name: str, ticker: str) -> tuple[str, Optional[pd.DataFrame]]:
        return name, _fetch_ohlcv(ticker, period="1y")

    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = {pool.submit(_fetch, n, t): n for n, t in tickers_to_fetch.items()}
        for fut in as_completed(futures):
            try:
                name, df = fut.result()
                raw_data[name] = df
            except Exception as exc:
                logger.warning("Fetch future failed: %s", exc)
                raw_data[futures[fut]] = None

    nifty_df = raw_data.get("NIFTY")
    vix_df = raw_data.get("VIX")
    sector_data = {name: raw_data.get(name) for name in SECTOR_TICKERS}

    # ---- classify each dimension --------------------------------------------
    market_regime, nifty_price, chg1d, chg5d, sma50, sma200 = _classify_market_regime(nifty_df)
    volatility_regime, vix = _classify_volatility(vix_df)
    breadth_score, breadth_label, ad_proxy = _compute_breadth(sector_data)
    leaders, laggards, rotation_mode, sector_returns = _classify_sector_rotation(sector_data)
    institutional = _classify_institutional(nifty_df, market_regime, vix, sma50, sma200)

    # ---- derived fields -----------------------------------------------------
    regime_score = _compute_regime_score(market_regime, volatility_regime, breadth_score, institutional)
    qm = _quality_multiplier(market_regime, volatility_regime)
    recommended, avoid = _derive_playbooks(
        market_regime, volatility_regime, breadth_label, rotation_mode, institutional
    )
    breakout_env = _breakout_environment(market_regime, volatility_regime, breadth_label)
    risk_mode = _risk_mode(sma50, sma200, vix, institutional)

    fetch_time_utc = datetime.now(timezone.utc)
    data_age_mins = 0  # just fetched

    state = RegimeState(
        market_regime=market_regime,
        volatility_regime=volatility_regime,
        breadth_strength=breadth_score,
        breadth_label=breadth_label,
        breakout_environment=breakout_env,
        risk_mode=risk_mode,
        institutional_activity=institutional,
        leading_sectors=leaders,
        lagging_sectors=laggards,
        rotation_mode=rotation_mode,
        sector_returns=sector_returns,
        nifty_price=round(nifty_price, 2) if not np.isnan(nifty_price) else 0.0,
        nifty_change_1d=round(chg1d, 3) if not np.isnan(chg1d) else 0.0,
        nifty_change_5d=round(chg5d, 3) if not np.isnan(chg5d) else 0.0,
        sma50=round(sma50, 2) if not np.isnan(sma50) else 0.0,
        sma200=round(sma200, 2) if not np.isnan(sma200) else 0.0,
        vix=round(vix, 2) if not np.isnan(vix) else 0.0,
        regime_score=regime_score,
        quality_multiplier=qm,
        recommended_playbooks=recommended,
        avoid_patterns=avoid,
        timestamp=fetch_time_utc.isoformat(),
        data_age_mins=data_age_mins,
    )

    _CACHE["regime_state"] = state
    _CACHE["timestamp"] = now

    logger.info("RegimeState computed: %s", state.summary_line())
    return state


if __name__ == "__main__":
    import json
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    rs = compute_regime()
    print(rs.summary_line())
    print(json.dumps(rs.to_dict(), indent=2, default=str))
