"""
numta indicator wrapper — 130+ TA-Lib-compatible indicators via numta.

Returns a flat dict of scalar values, consistent with IndicatorEngine output.
Designed to be called from IndicatorEngine.compute() as an optional extension.
"""
from __future__ import annotations

from typing import Any, Dict

import numpy as np
import pandas as pd

try:
    import numta
    _NUMTA_AVAILABLE = True
except ImportError:
    _NUMTA_AVAILABLE = False


def numta_available() -> bool:
    return _NUMTA_AVAILABLE


def add_numta_indicators(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Compute numta indicators from an OHLCV DataFrame.
    Returns a flat dict of {indicator_name: scalar} ready to merge into IndicatorEngine output.
    Missing columns or insufficient data → empty dict (never raises).
    """
    if not _NUMTA_AVAILABLE:
        return {}

    required = {"open", "high", "low", "close", "volume"}
    if not required.issubset(df.columns):
        return {}
    if len(df) < 30:
        return {}

    op = df["open"].to_numpy(dtype=float)
    hi = df["high"].to_numpy(dtype=float)
    lo = df["low"].to_numpy(dtype=float)
    cl = df["close"].to_numpy(dtype=float)
    vo = df["volume"].to_numpy(dtype=float)

    out: Dict[str, Any] = {}

    def _last(arr) -> float | None:
        try:
            v = float(arr[-1])
            return None if np.isnan(v) or np.isinf(v) else round(v, 5)
        except Exception:
            return None

    try:
        # ── Trend ─────────────────────────────────────────────────────────
        out["numta_ema_12"]   = _last(numta.EMA(cl, 12))
        out["numta_ema_26"]   = _last(numta.EMA(cl, 26))
        out["numta_dema_20"]  = _last(numta.DEMA(cl, 20))
        out["numta_tema_20"]  = _last(numta.TEMA(cl, 20))
        out["numta_wma_20"]   = _last(numta.WMA(cl, 20))
        out["numta_kama_10"]  = _last(numta.KAMA(cl, 10))
        out["numta_ht_trendline"] = _last(numta.HT_TRENDLINE(cl))
        out["numta_ht_trendmode"] = _last(numta.HT_TRENDMODE(cl))

        # ── MACD ──────────────────────────────────────────────────────────
        macd, macd_sig, macd_hist = numta.MACD(cl, 12, 26, 9)
        out["numta_macd"]      = _last(macd)
        out["numta_macd_sig"]  = _last(macd_sig)
        out["numta_macd_hist"] = _last(macd_hist)

        # ── Bollinger Bands ────────────────────────────────────────────────
        bb_up, bb_mid, bb_low = numta.BBANDS(cl, 20)
        out["numta_bb_upper"]  = _last(bb_up)
        out["numta_bb_mid"]    = _last(bb_mid)
        out["numta_bb_lower"]  = _last(bb_low)
        if bb_up[-1] and bb_low[-1] and bb_up[-1] != bb_low[-1]:
            out["numta_bb_pct"] = round(
                float((cl[-1] - bb_low[-1]) / (bb_up[-1] - bb_low[-1])), 5
            )

        # ── Momentum ──────────────────────────────────────────────────────
        out["numta_rsi_14"]  = _last(numta.RSI(cl, 14))
        out["numta_mom_10"]  = _last(numta.MOM(cl, 10))
        out["numta_roc_10"]  = _last(numta.ROC(cl, 10))
        out["numta_cmo_14"]  = _last(numta.CMO(cl, 14))
        out["numta_trix_15"] = _last(numta.TRIX(cl, 15))
        out["numta_willr_14"] = _last(numta.WILLR(hi, lo, cl, 14))

        stoch_k, stoch_d = numta.STOCH(hi, lo, cl)
        out["numta_stoch_k"] = _last(stoch_k)
        out["numta_stoch_d"] = _last(stoch_d)

        stochrsi_k, stochrsi_d = numta.STOCHRSI(cl)
        out["numta_stochrsi_k"] = _last(stochrsi_k)
        out["numta_stochrsi_d"] = _last(stochrsi_d)

        # ── Volatility ────────────────────────────────────────────────────
        out["numta_atr_14"]  = _last(numta.ATR(hi, lo, cl, 14))
        out["numta_natr_14"] = _last(numta.NATR(hi, lo, cl, 14))

        # ── Volume ────────────────────────────────────────────────────────
        out["numta_obv"]  = _last(numta.OBV(cl, vo))
        out["numta_mfi_14"] = _last(numta.MFI(hi, lo, cl, vo, 14))
        out["numta_ad"]   = _last(numta.AD(hi, lo, cl, vo))

        # ── Directional ───────────────────────────────────────────────────
        out["numta_adx_14"]      = _last(numta.ADX(hi, lo, cl, 14))
        out["numta_plus_di_14"]  = _last(numta.PLUS_DI(hi, lo, cl, 14))
        out["numta_minus_di_14"] = _last(numta.MINUS_DI(hi, lo, cl, 14))
        out["numta_cci_14"]      = _last(numta.CCI(hi, lo, cl, 14))
        out["numta_aroon_up"], out["numta_aroon_dn"] = (
            _last(numta.AROON(hi, lo, 14)[0]),
            _last(numta.AROON(hi, lo, 14)[1]),
        )
        out["numta_sar"] = _last(numta.SAR(hi, lo))

        # ── Linear Regression ─────────────────────────────────────────────
        out["numta_linreg_slope_14"] = _last(numta.LINEARREG_SLOPE(cl, 14))
        out["numta_linreg_angle_14"] = _last(numta.LINEARREG_ANGLE(cl, 14))

        # ── Pattern Recognition (key reversal patterns) ───────────────────
        out["numta_cdl_engulfing"]   = _last(numta.CDLENGULFING(op, hi, lo, cl))
        out["numta_cdl_hammer"]      = _last(numta.CDLHAMMER(op, hi, lo, cl))
        out["numta_cdl_shooting"]    = _last(numta.CDLSHOOTINGSTAR(op, hi, lo, cl))
        out["numta_cdl_doji"]        = _last(numta.CDLDOJI(op, hi, lo, cl))
        out["numta_cdl_morning"]     = _last(numta.CDLMORNINGSTAR(op, hi, lo, cl))
        out["numta_cdl_evening"]     = _last(numta.CDLEVENINGSTAR(op, hi, lo, cl))

    except Exception as exc:
        # Partial results are fine — don't discard what succeeded
        out["numta_error"] = str(exc)

    return out
