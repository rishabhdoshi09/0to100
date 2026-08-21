"""Synthetic Stage-2 / VCP frames for SEPA research replays (not market data)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from research.sepa.config import SepaConfig


def idx(n: int, start="2020-01-02") -> pd.DatetimeIndex:
    return pd.bdate_range(start, periods=n)


def ohlcv(close, volume=None) -> pd.DataFrame:
    close = np.asarray(close, dtype=float)
    n = len(close)
    vol = np.asarray(volume if volume is not None else np.full(n, 100_000.0), dtype=float)
    high = close + 0.4
    low = close - 0.4
    return pd.DataFrame(
        {"open": close, "high": high, "low": low, "close": close, "volume": vol},
        index=idx(n),
    )


def stage2(n: int = 280, start=80.0, step=0.55, volume=120_000.0) -> pd.DataFrame:
    close = start + np.arange(n) * step
    return ohlcv(close, volume=np.full(n, volume))


def plant_vcp(*, contractions="tight", volume="dry", extend=0.0, wide_stop=False) -> pd.DataFrame:
    base = stage2(260, start=80.0, step=0.5)
    last = float(base["close"].iloc[-1])
    pivot = last
    if contractions == "tight":
        legs = [
            (pivot, pivot * 0.86),
            (pivot * 0.99, pivot * 0.92),
            (pivot, pivot * 0.96),
        ]
    elif contractions == "two":
        legs = [(pivot, pivot * 0.88), (pivot * 0.995, pivot * 0.94)]
    elif contractions == "widening":
        legs = [(pivot, pivot * 0.95), (pivot * 0.99, pivot * 0.88), (pivot, pivot * 0.80)]
    elif contractions == "deep":
        legs = [(pivot, pivot * 0.55), (pivot * 0.90, pivot * 0.70), (pivot, pivot * 0.85)]
    else:
        raise ValueError(contractions)

    highs, lows, closes, vols = [], [], [], []

    def _swing_high(px: float, vol: float):
        for v in (px * 0.985, px * 0.995, px, px * 0.995, px * 0.985):
            highs.append(v + 0.05)
            lows.append(v - 0.8)
            closes.append(v)
            vols.append(vol)

    def _swing_low(px: float, vol: float):
        for v in (px * 1.015, px * 1.005, px, px * 1.005, px * 1.015):
            highs.append(v + 0.8)
            lows.append(v - 0.05)
            closes.append(v)
            vols.append(vol)

    def _recover(from_px: float, to_px: float, vol: float, bars=4):
        seq = np.linspace(from_px, to_px, bars)
        for v in seq:
            highs.append(v + 0.3)
            lows.append(v - 0.3)
            closes.append(v)
            vols.append(vol)

    vol_first = 400_000.0 if volume == "dry" else 80_000.0
    vol_last = 80_000.0 if volume == "dry" else 500_000.0
    for i, (h, lo) in enumerate(legs):
        vol = vol_first if i == 0 else (vol_last if i == len(legs) - 1 else (vol_first + vol_last) / 2)
        _swing_high(h, vol)
        _recover(h * 0.99, lo * 1.01, vol, bars=3)
        _swing_low(lo if not wide_stop else lo * 0.5, vol)
        nxt = legs[i + 1][0] if i + 1 < len(legs) else h * (1.0 + extend)
        _recover(lo * 1.01, nxt, vol_last if i == len(legs) - 1 else vol, bars=4)
    finish = pivot * (1.0 + extend)
    _recover(closes[-1], finish, vol_last, bars=3)
    extra = pd.DataFrame(
        {"open": closes, "high": highs, "low": lows, "close": closes, "volume": vols},
        index=idx(len(closes), start=str(base.index[-1].date() + pd.Timedelta(days=1))),
    )
    return pd.concat([base, extra])


RESEARCH_CFG = SepaConfig(swing_left=2, swing_right=2)
