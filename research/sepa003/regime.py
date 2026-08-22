"""PIT market-regime classifier. No future returns enter the as-of label."""
from __future__ import annotations

from typing import Any, Mapping

import numpy as np
import pandas as pd

from research.sepa003.constants import REGIME_VERSION


def official_nifty_close() -> pd.Series | None:
    try:
        from data.index_store import get_index_ohlcv
        df = get_index_ohlcv("^NSEI") or get_index_ohlcv("Nifty 50")
    except Exception:
        df = None
    if df is None or len(df) < 200:
        return None
    col = next((c for c in df.columns if str(c).lower() == "close"), None)
    if col is None:
        return None
    s = pd.to_numeric(df[col], errors="coerce").dropna()
    s.index = pd.DatetimeIndex(s.index).tz_localize(None).normalize()
    return s.sort_index() if len(s) >= 200 else None


def nifty50_equal_weight_proxy(frames: Mapping[str, pd.DataFrame]) -> pd.Series:
    """Equal-weight of today's NIFTY50 list using official bhav closes.

    Member *identity* is contemporaneous (PIT_DEGRADED). Prices are as-of.
    """
    from data.nse_universe import NIFTY50
    members = [s for s in NIFTY50 if s in frames and frames[s] is not None and len(frames[s])]
    if not members:
        return pd.Series(dtype=float)
    rets = []
    for sym in members:
        c = pd.to_numeric(frames[sym]["close"], errors="coerce")
        idx = pd.DatetimeIndex(frames[sym].index).tz_localize(None).normalize()
        c = pd.Series(c.values, index=idx).sort_index()
        r = c.pct_change()
        rets.append(r)
    panel = pd.concat(rets, axis=1)
    daily = panel.mean(axis=1, skipna=True).dropna()
    if daily.empty:
        return pd.Series(dtype=float)
    level = (1.0 + daily).cumprod() * 100.0
    level.index = pd.DatetimeIndex(level.index).normalize()
    return level


def build_index_level(frames: Mapping[str, pd.DataFrame]) -> tuple[pd.Series, str]:
    official = official_nifty_close()
    if official is not None and len(official) >= 200:
        return official, "official_nse_index_store"
    proxy = nifty50_equal_weight_proxy(frames)
    return proxy, "NIFTY50_EQUALWEIGHT_PROXY_BHAV"


def classify_regime_level(close: pd.Series) -> pd.DataFrame:
    """Deterministic states from a price level. Rolling windows use ≤ t only."""
    close = pd.Series(close).astype(float).sort_index()
    close.index = pd.DatetimeIndex(close.index).tz_localize(None).normalize()
    sma50 = close.rolling(50, min_periods=50).mean()
    sma200 = close.rolling(200, min_periods=200).mean()
    sma200_prev = sma200.shift(21)
    ret20 = close.pct_change(20)
    ret63 = close.pct_change(63)
    sma50_prev = sma50.shift(21)
    dist50 = (close / sma50 - 1.0) * 100.0
    dist200 = (close / sma200 - 1.0) * 100.0
    slope50 = (sma50 / sma50_prev - 1.0) * 100.0
    slope200 = (sma200 / sma200_prev - 1.0) * 100.0
    rising200 = sma200 > sma200_prev
    falling200 = sma200 < sma200_prev

    state = pd.Series("UNKNOWN", index=close.index)
    known = sma200.notna() & ret20.notna()
    strong = known & (close > sma50) & (sma50 > sma200) & rising200 & (ret20 > 0.06)
    bull = known & (close > sma50) & (ret20 > 0.02) & ~strong
    bear = known & (close < sma50) & (close < sma200) & ((ret20 < -0.08) | falling200)
    corr = known & (close < sma50) & (ret20 < -0.02) & ~bear
    side = known & ~strong & ~bull & ~bear & ~corr
    state[strong] = "STRONG_BULL"
    state[bull] = "BULL"
    state[bear] = "BEAR"
    state[corr] = "CORRECTION"
    state[side] = "SIDEWAYS"

    out = pd.DataFrame({
        "regime": state,
        "index_close": close,
        "sma50": sma50,
        "sma200": sma200,
        "dist_sma50_pct": dist50,
        "dist_sma200_pct": dist200,
        "slope_sma50_pct": slope50,
        "slope_sma200_pct": slope200,
        "ret20": ret20,
        "ret63": ret63,
        "trend_state": np.where(
            close > sma50,
            np.where(close > sma200, "ABOVE_50_200", "ABOVE_50"),
            np.where(close < sma200, "BELOW_50_200", "BELOW_50"),
        ),
    })
    out.attrs["regime_version"] = REGIME_VERSION
    return out


def regime_at(table: pd.DataFrame | None, as_of: str) -> dict[str, Any]:
    empty = {
        "regime": "UNKNOWN", "index_close": None, "sma50": None, "sma200": None,
        "dist_sma50_pct": None, "dist_sma200_pct": None,
        "slope_sma50_pct": None, "slope_sma200_pct": None,
        "ret20": None, "ret63": None, "trend_state": "UNKNOWN",
    }
    if table is None or len(table) == 0:
        return empty
    try:
        cutoff = pd.Timestamp(as_of)
        sl = table[table.index <= cutoff]
        if sl.empty:
            return empty
        row = sl.iloc[-1]
        out = {k: (None if pd.isna(row[k]) else (str(row[k]) if k in ("regime", "trend_state") else float(row[k])))
               for k in empty}
        return out
    except Exception:
        return empty


def append_future_invariant(close: pd.Series, cut: int) -> bool:
    """Appending bars after `cut` must not change the label at cut-1."""
    if close is None or len(close) < cut + 10:
        return True
    prefix = classify_regime_level(close.iloc[:cut])
    full = classify_regime_level(close)
    ts = prefix.index[-1]
    return str(prefix.loc[ts, "regime"]) == str(full.loc[ts, "regime"])


def breadth_as_of(
    frames: Mapping[str, pd.DataFrame],
    as_of: str,
    symbols: list[str] | None = None,
    *,
    min_n: int = 300,
) -> dict[str, Any]:
    """% of names above own SMA50/200 using bars ≤ as_of only."""
    from research.sepa.frames import slice_as_of
    names = symbols or list(frames)
    above50 = above200 = n = adv = dec = 0
    cutoff = pd.Timestamp(as_of)
    for sym in names:
        df = frames.get(sym)
        hist = slice_as_of(df, cutoff)
        if hist is None or len(hist) < 51:
            continue
        c = pd.to_numeric(hist["close"], errors="coerce").dropna()
        if len(c) < 51:
            continue
        n += 1
        last, prev = float(c.iloc[-1]), float(c.iloc[-2])
        if last > prev:
            adv += 1
        elif last < prev:
            dec += 1
        if last > float(c.iloc[-50:].mean()):
            above50 += 1
        if len(c) >= 200 and last > float(c.iloc[-200:].mean()):
            above200 += 1
    if n < min_n:
        return {"n": n, "pct_above_50": None, "pct_above_200": None,
                "adv_ratio": None, "verdict": "", "usable": False}
    return {
        "n": n,
        "pct_above_50": round(100.0 * above50 / n, 2),
        "pct_above_200": round(100.0 * above200 / n, 2) if n else None,
        "adv_ratio": round(adv / dec, 3) if dec else None,
        "verdict": (
            "HEALTHY" if (adv / max(dec, 1) >= 1.2 and above50 / n >= 0.55) else
            "NARROW" if (adv / max(dec, 1) < 0.8 or above50 / n < 0.40) else
            "MIXED"
        ),
        "usable": True,
    }
