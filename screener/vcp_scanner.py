"""
VCP + Base Formation Breakout Scanner.

Detects institutional-grade setups using:
- Mark Minervini VCP (Volatility Contraction Pattern)
- Stan Weinstein Stage 2 analysis
- Wyckoff accumulation signatures
- CANSLIM-style relative strength

Scoring: Trend(20%) + Base(20%) + Volume Contraction(20%) +
         Volatility Contraction(15%) + RS(15%) + Breakout(10%)
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

from logger import get_logger

log = get_logger(__name__)


# ── Dataclasses ───────────────────────────────────────────────────────────────

@dataclass
class VCPSetup:
    symbol: str
    price: float
    setup_score: float           # 0-100 master score
    category: str                # "Elite Setup" | "Strong Setup" | "Watchlist" | "Avoid"
    base_type: str               # "VCP" | "FLAT" | "CUP" | "ASCENDING" | "DEEP" | "NONE"
    base_depth_pct: float        # max drawdown within base
    base_duration_weeks: int
    breakout_level: float        # pivot/resistance to break
    stop_loss: float             # base low or ATR-adjusted
    atr_risk: float              # current ATR
    risk_reward: float
    volume_contraction_score: float    # 0-100
    volatility_contraction_score: float  # 0-100
    accumulation_score: float    # 0-100
    rs_score: float              # relative strength vs Nifty 0-100
    trend_stage: str             # "Stage 1" | "Stage 2" | "Stage 3" | "Stage 4"
    contraction_sequence: list   # e.g. [15.0, 10.0, 6.0] pullback depths
    breakout_probability: float  # 0-1
    vcp_contractions: int        # number of contractions detected
    pivot_distance_pct: float    # % away from breakout pivot
    weekly_tight: bool           # weekly closes within 1-2%
    pocket_pivot: bool           # pocket pivot detected


# ── Scanner ───────────────────────────────────────────────────────────────────

class VCPScanner:
    """
    Scans a universe of NSE stocks for VCP and base formation breakout setups.
    Uses yfinance as primary data source.  Pure pandas/numpy — no TA-Lib.
    """

    _nifty_returns_cache: Optional[pd.Series] = None   # class-level cache

    def __init__(self, nifty_returns: Optional[pd.Series] = None) -> None:
        if nifty_returns is not None:
            VCPScanner._nifty_returns_cache = nifty_returns
        elif VCPScanner._nifty_returns_cache is None:
            VCPScanner._nifty_returns_cache = self._fetch_nifty_returns()

        self._nifty_returns = VCPScanner._nifty_returns_cache

    # ── Public API ─────────────────────────────────────────────────────────────

    def scan(self, symbols: list[str], top_n: int = 25) -> list[VCPSetup]:
        """Parallel scan across symbols.  Returns top_n results sorted by score."""
        results: list[VCPSetup] = []

        with ThreadPoolExecutor(max_workers=16) as pool:
            futures = {pool.submit(self.analyze, sym): sym for sym in symbols}
            for fut in as_completed(futures):
                sym = futures[fut]
                try:
                    setup = fut.result()
                    if setup is not None and setup.setup_score >= 40:
                        results.append(setup)
                except Exception as exc:
                    log.debug("vcp_scan_failed", symbol=sym, error=str(exc))

        results.sort(key=lambda x: x.setup_score, reverse=True)
        return results[:top_n]

    def analyze(self, symbol: str) -> Optional[VCPSetup]:
        """Full VCP analysis pipeline for one symbol."""
        try:
            df = self._fetch_data(symbol, days=520)
            if df is None or len(df) < 100:
                return None

            close = df["close"].values
            price = float(close[-1])

            # ── Component scores ──────────────────────────────────────────────
            trend_score, stage = self._trend_filter(df)
            base_score, base_type, base_depth, duration_weeks, breakout_level = self._detect_base(df)
            vol_contraction, pocket_pivot = self._volume_contraction_score_full(df)
            volatility_score, contraction_seq = self._volatility_contraction_score(df)
            rs = self._rs_score(
                pd.Series(close).pct_change().dropna(),
                self._nifty_returns,
            )
            breakout_score = self._breakout_score(df, breakout_level)
            accum_score, weekly_tight = self._accumulation_score(df)

            # ── Master score ──────────────────────────────────────────────────
            master = self._master_score(
                trend_score, base_score, vol_contraction,
                volatility_score, rs, breakout_score,
            )

            # ── Stop / target ─────────────────────────────────────────────────
            atr = self._calc_atr(df)
            stop_loss, atr_risk, rr = self._compute_stop_and_target(
                df, base_depth, breakout_level, atr
            )

            # ── Category ──────────────────────────────────────────────────────
            category = self._category(master)

            # ── VCP count ─────────────────────────────────────────────────────
            vcp_count = len(contraction_seq)

            # ── Pivot distance ────────────────────────────────────────────────
            pivot_dist_pct = (breakout_level - price) / price * 100 if price > 0 else 0.0

            # ── Breakout probability (heuristic) ─────────────────────────────
            bp = self._breakout_probability(master, vol_contraction, volatility_score, pivot_dist_pct)

            return VCPSetup(
                symbol=symbol,
                price=round(price, 2),
                setup_score=round(master, 1),
                category=category,
                base_type=base_type,
                base_depth_pct=round(base_depth, 2),
                base_duration_weeks=duration_weeks,
                breakout_level=round(breakout_level, 2),
                stop_loss=round(stop_loss, 2),
                atr_risk=round(atr_risk, 2),
                risk_reward=round(rr, 2),
                volume_contraction_score=round(vol_contraction, 1),
                volatility_contraction_score=round(volatility_score, 1),
                accumulation_score=round(accum_score, 1),
                rs_score=round(rs, 1),
                trend_stage=stage,
                contraction_sequence=contraction_seq,
                breakout_probability=round(bp, 3),
                vcp_contractions=vcp_count,
                pivot_distance_pct=round(pivot_dist_pct, 2),
                weekly_tight=weekly_tight,
                pocket_pivot=pocket_pivot,
            )

        except Exception as exc:
            log.debug("vcp_analyze_error", symbol=symbol, error=str(exc))
            return None

    # ── Private: data ──────────────────────────────────────────────────────────

    def _fetch_data(self, symbol: str, days: int = 520) -> Optional[pd.DataFrame]:
        try:
            import yfinance as yf
            ticker = yf.Ticker(f"{symbol}.NS")
            df = ticker.history(period="2y", interval="1d")
            if df is None or df.empty or len(df) < 60:
                return None
            df = df.copy()
            df.columns = [c.lower() for c in df.columns]
            df = df.reset_index()
            # Ensure numeric columns
            for col in ["open", "high", "low", "close", "volume"]:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors="coerce")
            df = df.dropna(subset=["close"])
            return df
        except Exception as exc:
            log.debug("vcp_fetch_failed", symbol=symbol, error=str(exc))
            return None

    @staticmethod
    def _fetch_nifty_returns() -> Optional[pd.Series]:
        try:
            import yfinance as yf
            df = yf.Ticker("^NSEI").history(period="2y", interval="1d")
            if df is None or df.empty:
                return None
            return df["Close"].pct_change().dropna()
        except Exception as exc:
            log.debug("vcp_nifty_fetch_failed", error=str(exc))
            return None

    # ── Private: trend ─────────────────────────────────────────────────────────

    def _trend_filter(self, df: pd.DataFrame) -> tuple[float, str]:
        """Returns (trend_score 0-100, stage_label)."""
        close = df["close"].values
        if len(close) < 200:
            return 30.0, "Stage 1"

        sma50_series = pd.Series(close).rolling(50).mean().values
        sma200_series = pd.Series(close).rolling(200).mean().values

        sma50 = float(sma50_series[-1])
        sma200 = float(sma200_series[-1])
        price = float(close[-1])

        high_52w = float(np.max(close[-252:]) if len(close) >= 252 else np.max(close))

        # Stage classification
        if price > sma50 > sma200:
            stage = "Stage 2"
            score = 100.0
        elif price > sma200 and sma50 < sma200:
            stage = "Stage 1"
            score = 60.0
        elif price < sma50 and price > sma200:
            stage = "Stage 3"
            score = 30.0
        else:
            stage = "Stage 4"
            score = 0.0

        # Bonus: within 25% of 52-week high
        if high_52w > 0 and (high_52w - price) / high_52w < 0.25:
            score = min(100.0, score + 10.0)

        # Penalty: SMA50 declining last 10 days
        if len(sma50_series) >= 11:
            sma50_10d_ago = float(sma50_series[-11])
            if sma50 < sma50_10d_ago:
                score = max(0.0, score - 20.0)

        return score, stage

    # ── Private: base detection ────────────────────────────────────────────────

    def _detect_base(
        self, df: pd.DataFrame
    ) -> tuple[float, str, float, int, float]:
        """Returns (base_score, base_type, base_depth_pct, duration_weeks, breakout_level)."""
        close = df["close"].values
        high_arr = df["high"].values if "high" in df.columns else close
        low_arr = df["low"].values if "low" in df.columns else close

        window = min(80, len(close))
        base_close = close[-window:]
        base_high = high_arr[-window:]
        base_low = low_arr[-window:]

        if len(base_close) < 20:
            return 0.0, "NONE", 0.0, 0, float(close[-1])

        highest_high = float(np.max(base_high))
        lowest_low = float(np.min(base_low))

        if highest_high == 0:
            return 0.0, "NONE", 0.0, 0, float(close[-1])

        base_depth = (highest_high - lowest_low) / highest_high * 100
        duration_weeks = window // 5

        # Hard reject
        if base_depth > 40:
            return 20.0, "DEEP", base_depth, duration_weeks, highest_high

        # Breakout level = highest high in last 30 days
        breakout_level = float(np.max(high_arr[-30:])) if len(high_arr) >= 30 else highest_high

        # ── VCP detection (priority) ──────────────────────────────────────────
        vcp_score, contraction_seq, vcp_count = self._check_vcp(base_close)
        if vcp_count >= 2 and base_depth < 20:
            return 95.0, "VCP", base_depth, duration_weeks, breakout_level

        # ── FLAT base ─────────────────────────────────────────────────────────
        if base_depth < 12 and duration_weeks >= 4:
            return 90.0, "FLAT", base_depth, duration_weeks, breakout_level

        # ── CUP base (U-shape) ────────────────────────────────────────────────
        if 12 <= base_depth <= 35:
            third = len(base_close) // 3
            if third > 0:
                first_trend = base_close[third] - base_close[0]
                mid_mean = float(np.mean(base_close[third: 2 * third]))
                last_trend = base_close[-1] - base_close[2 * third]
                if first_trend < 0 and last_trend > 0 and mid_mean < base_close[0]:
                    return 85.0, "CUP", base_depth, duration_weeks, breakout_level

        # ── ASCENDING base ────────────────────────────────────────────────────
        if base_depth < 15:
            # Each pullback higher than last
            lows = self._find_swing_lows(base_close)
            if len(lows) >= 2:
                if all(lows[i] > lows[i - 1] for i in range(1, len(lows))):
                    return 80.0, "ASCENDING", base_depth, duration_weeks, breakout_level

        # ── Generic base ──────────────────────────────────────────────────────
        score = max(0.0, 70.0 - base_depth * 1.5)
        return score, "NONE", base_depth, duration_weeks, breakout_level

    def _check_vcp(self, close: np.ndarray) -> tuple[float, list[float], int]:
        """Returns (score, contraction_sequence, count)."""
        swing_highs = self._find_swing_highs(close)
        swing_lows = self._find_swing_lows(close)

        pullbacks: list[float] = []
        for i in range(1, len(swing_highs)):
            h_prev = swing_highs[i - 1]
            # find a low between swing_high[i-1] and swing_high[i]
            lows_between = [l for l in swing_lows if l < h_prev]
            if lows_between:
                l_val = min(lows_between)
                if h_prev > 0:
                    depth = (h_prev - l_val) / h_prev * 100
                    pullbacks.append(round(depth, 1))

        if len(pullbacks) < 2:
            return 0.0, pullbacks, 0

        # Each pullback < previous * 0.75
        contractions = 0
        for i in range(1, len(pullbacks)):
            if pullbacks[i] < pullbacks[i - 1] * 0.75:
                contractions += 1

        if contractions >= 2:
            return 95.0, pullbacks, contractions

        return 0.0, pullbacks, contractions

    def _find_swing_highs(self, close: np.ndarray, window: int = 3) -> list[float]:
        highs = []
        for i in range(window, len(close) - window):
            if close[i] == max(close[i - window: i + window + 1]):
                highs.append(float(close[i]))
        return highs

    def _find_swing_lows(self, close: np.ndarray, window: int = 3) -> list[float]:
        lows = []
        for i in range(window, len(close) - window):
            if close[i] == min(close[i - window: i + window + 1]):
                lows.append(float(close[i]))
        return lows

    # ── Private: volume contraction ────────────────────────────────────────────

    def _volume_contraction_score_full(self, df: pd.DataFrame) -> tuple[float, bool]:
        """Returns (score 0-100, pocket_pivot_bool)."""
        if "volume" not in df.columns or len(df) < 60:
            return 50.0, False

        volume = df["volume"].values.astype(float)
        close = df["close"].values

        vol_60 = volume[-60:]
        if len(vol_60) < 60:
            return 50.0, False

        early = float(np.mean(vol_60[0:20]))
        mid = float(np.mean(vol_60[20:40]))
        late = float(np.mean(vol_60[40:60]))

        score = 0.0
        if early > 0:
            ratio = late / early
            score = max(0.0, min(100.0, (1 - ratio) * 100))

        # Pocket pivot: volume > max(vol[-11:-1]) on an UP day
        pocket_pivot = False
        avg_20 = float(np.mean(volume[-21:-1])) if len(volume) > 21 else 0.0
        if len(volume) >= 12 and len(close) >= 2:
            for i in range(-10, 0):
                idx = len(volume) + i
                if idx >= 1 and close[idx] > close[idx - 1]:
                    prior_max = float(np.max(volume[idx - 11: idx]))
                    if volume[idx] > prior_max:
                        pocket_pivot = True
                        break

        if pocket_pivot:
            score = min(100.0, score + 20.0)

        # Volume dry-up: any day in last 10 where vol < 0.5 * 20d avg
        if avg_20 > 0 and len(volume) >= 10:
            if any(volume[i] < 0.5 * avg_20 for i in range(-10, 0)):
                score = min(100.0, score + 15.0)

        return round(score, 1), pocket_pivot

    def _volume_contraction_score(self, df: pd.DataFrame) -> float:
        score, _ = self._volume_contraction_score_full(df)
        return score

    # ── Private: volatility contraction ───────────────────────────────────────

    def _volatility_contraction_score(
        self, df: pd.DataFrame
    ) -> tuple[float, list[float]]:
        """Returns (score 0-100, contraction_sequence)."""
        close = df["close"].values
        high_arr = df["high"].values if "high" in df.columns else close
        low_arr = df["low"].values if "low" in df.columns else close

        if len(close) < 60:
            return 30.0, []

        # 1. ATR contraction
        atr_series = self._calc_atr_series(df)
        score_atr = 0.0
        if atr_series is not None and len(atr_series) >= 60:
            atr_early = float(atr_series.iloc[-60])
            atr_now = float(atr_series.iloc[-1])
            if atr_early > 0:
                atr_compression = (atr_early - atr_now) / atr_early * 100
                score_atr = min(50.0, atr_compression * 2)

        # 2. Range contraction
        ranges_30 = (high_arr[-30:] - low_arr[-30:]).astype(float)
        score_range = 0.0
        if len(ranges_30) >= 30:
            mean_first = float(np.mean(ranges_30[:15]))
            mean_last = float(np.mean(ranges_30[15:]))
            if mean_first > 0:
                rc = (mean_first - mean_last) / mean_first * 100
                score_range = min(30.0, rc * 1.5)

        # 3. VCP swing contraction
        score_vcp = 0.0
        contraction_seq: list[float] = []

        close_60 = close[-60:]
        high_60 = high_arr[-60:]
        low_60 = low_arr[-60:]

        swing_h_idx = []
        w = 3
        for i in range(w, len(close_60) - w):
            if close_60[i] == max(close_60[i - w: i + w + 1]):
                swing_h_idx.append(i)

        swing_l_idx = []
        for i in range(w, len(close_60) - w):
            if close_60[i] == min(close_60[i - w: i + w + 1]):
                swing_l_idx.append(i)

        pullbacks: list[float] = []
        for i in range(1, len(swing_h_idx)):
            h_idx = swing_h_idx[i - 1]
            h_val = float(high_60[h_idx])
            # find lows between consecutive highs
            lows_between = [low_60[j] for j in swing_l_idx if swing_h_idx[i - 1] < j < swing_h_idx[i]]
            if lows_between and h_val > 0:
                l_val = float(min(lows_between))
                depth = (h_val - l_val) / h_val * 100
                pullbacks.append(round(depth, 1))

        contraction_seq = pullbacks
        if len(pullbacks) >= 3:
            if pullbacks[0] > pullbacks[1] > pullbacks[2]:
                score_vcp = 20.0

        total = max(0.0, min(100.0, score_atr + score_range + score_vcp))
        return round(total, 1), contraction_seq

    def _calc_atr_series(self, df: pd.DataFrame, period: int = 14) -> Optional[pd.Series]:
        if "high" not in df.columns or "low" not in df.columns:
            return None
        h = df["high"].values.astype(float)
        l = df["low"].values.astype(float)
        c = df["close"].values.astype(float)
        trs = []
        for i in range(1, len(c)):
            tr = max(h[i] - l[i], abs(h[i] - c[i - 1]), abs(l[i] - c[i - 1]))
            trs.append(tr)
        tr_series = pd.Series(trs)
        return tr_series.rolling(period).mean()

    # ── Private: relative strength ─────────────────────────────────────────────

    def _rs_score(
        self,
        symbol_returns: pd.Series,
        nifty_returns: Optional[pd.Series],
    ) -> float:
        """Returns 0-100 RS score vs Nifty."""
        if nifty_returns is None or len(symbol_returns) < 21:
            return 50.0

        def _period_rs(sym_ret: pd.Series, nif_ret: pd.Series, n: int) -> float:
            s = sym_ret.iloc[-n:].sum() if len(sym_ret) >= n else sym_ret.sum()
            b = nif_ret.iloc[-n:].sum() if len(nif_ret) >= n else nif_ret.sum()
            return float(s - b)

        rs_1m = _period_rs(symbol_returns, nifty_returns, 21)
        rs_3m = _period_rs(symbol_returns, nifty_returns, 63)
        rs_6m = _period_rs(symbol_returns, nifty_returns, 126)

        raw = rs_1m * 0.40 + rs_3m * 0.35 + rs_6m * 0.25
        # Normalize: raw ~= outperformance in decimal. Clamp ±0.5 → 0-100
        score = (raw + 0.5) / 1.0 * 100
        return max(0.0, min(100.0, score))

    # ── Private: accumulation ──────────────────────────────────────────────────

    def _accumulation_score(self, df: pd.DataFrame) -> tuple[float, bool]:
        """Returns (score 0-100, weekly_tight)."""
        close = df["close"].values
        high_arr = df["high"].values if "high" in df.columns else close
        low_arr = df["low"].values if "low" in df.columns else close
        volume = df["volume"].values.astype(float) if "volume" in df.columns else None

        score = 0.0
        weekly_tight = False

        # 1. Tight weekly closes (group by week)
        try:
            date_col = "date" if "date" in df.columns else df.columns[0]
            df_tmp = df.copy()
            df_tmp["_close"] = close
            df_tmp["_week"] = pd.to_datetime(df_tmp[date_col]).dt.isocalendar().week
            weekly = df_tmp.groupby("_week")["_close"].last()
            if len(weekly) >= 3:
                weekly_vals = weekly.values[-6:]  # last 6 weeks
                weekly_range_pcts = []
                for i in range(len(weekly_vals) - 2):
                    chunk = weekly_vals[i: i + 3]
                    rng = (float(np.max(chunk)) - float(np.min(chunk))) / float(np.mean(chunk)) * 100
                    weekly_range_pcts.append(rng)
                if weekly_range_pcts and min(weekly_range_pcts) < 2.0:
                    score += 25.0
                    weekly_tight = True
        except Exception:
            pass

        # 2. Low-volume pullbacks: red candles have below-average volume
        if volume is not None and len(volume) > 20:
            avg_vol = float(np.mean(volume[-21:-1]))
            if avg_vol > 0:
                red_vols = [volume[i] for i in range(-20, 0) if close[i] < close[i - 1]]
                if red_vols and float(np.mean(red_vols)) < avg_vol:
                    score += 20.0

        # 3. Support defense: price tested a level 2+ times without breaking
        if len(low_arr) >= 20:
            base_low = float(np.min(low_arr[-20:]))
            touches = sum(
                1 for i in range(-20, 0)
                if abs(low_arr[i] - base_low) / (base_low + 1e-9) < 0.01
            )
            if touches >= 2:
                score += 20.0

        # 4. Up-volume dominance
        if volume is not None and len(volume) > 20:
            up_vol = sum(volume[i] for i in range(-20, 0) if close[i] >= close[i - 1])
            dn_vol = sum(volume[i] for i in range(-20, 0) if close[i] < close[i - 1])
            if dn_vol > 0 and up_vol > dn_vol * 1.3:
                score += 20.0

        # 5. Closing range: avg (close-low)/(high-low) > 0.6 for last 20 days
        if len(close) >= 20:
            ranges = high_arr[-20:] - low_arr[-20:]
            close_ranges = close[-20:] - low_arr[-20:]
            valid = [(cr / r) for cr, r in zip(close_ranges, ranges) if r > 0]
            if valid and float(np.mean(valid)) > 0.6:
                score += 15.0

        return min(100.0, score), weekly_tight

    # ── Private: breakout score ────────────────────────────────────────────────

    def _breakout_score(self, df: pd.DataFrame, breakout_level: float) -> float:
        """Returns 0-100 breakout proximity/confirmation score."""
        close = df["close"].values
        volume = df["volume"].values.astype(float) if "volume" in df.columns else None
        high_arr = df["high"].values if "high" in df.columns else close
        low_arr = df["low"].values if "low" in df.columns else close

        price = float(close[-1])
        if price == 0:
            return 0.0

        score = 0.0
        distance_pct = (breakout_level - price) / price * 100

        if distance_pct < 0:
            # Already broke out
            if volume is not None and len(volume) > 20:
                avg_vol = float(np.mean(volume[-21:-1]))
                if avg_vol > 0 and volume[-1] > 1.5 * avg_vol:
                    score += 60.0
                else:
                    score += 20.0
            else:
                score += 20.0
        elif distance_pct < 2:
            score += 40.0
        elif distance_pct < 5:
            score += 20.0

        # Volume on last day vs 20-day avg
        if volume is not None and len(volume) > 20:
            avg_vol = float(np.mean(volume[-21:-1]))
            if avg_vol > 0:
                vol_ratio = volume[-1] / avg_vol
                vol_bonus = min(30.0, (vol_ratio - 0.5) * 15)
                score += max(0.0, vol_bonus)

        # Candle body strength
        o = float(df["open"].values[-1]) if "open" in df.columns else price
        h = float(high_arr[-1])
        l = float(low_arr[-1])
        c = float(close[-1])
        candle_range = h - l
        if candle_range > 0:
            body_strength = abs(c - o) / candle_range
            if body_strength > 0.6:
                score += 10.0

        return min(100.0, score)

    # ── Private: master score ──────────────────────────────────────────────────

    def _master_score(
        self,
        trend: float,
        base: float,
        vol_contraction: float,
        volatility: float,
        rs: float,
        breakout: float,
    ) -> float:
        score = (
            trend * 0.20
            + base * 0.20
            + vol_contraction * 0.20
            + volatility * 0.15
            + rs * 0.15
            + breakout * 0.10
        )
        return max(0.0, min(100.0, score))

    @staticmethod
    def _category(score: float) -> str:
        if score >= 75:
            return "Elite Setup"
        if score >= 60:
            return "Strong Setup"
        if score >= 45:
            return "Watchlist"
        return "Avoid"

    # ── Private: stop/target ───────────────────────────────────────────────────

    def _compute_stop_and_target(
        self,
        df: pd.DataFrame,
        base_depth: float,
        breakout_level: float,
        atr: float,
    ) -> tuple[float, float, float]:
        close = df["close"].values
        low_arr = df["low"].values if "low" in df.columns else close
        price = float(close[-1])

        base_low = float(np.min(low_arr[-80:])) if len(low_arr) >= 80 else float(np.min(low_arr))
        stop_loss = max(base_low, price - 2 * atr)

        measured_move = breakout_level + (breakout_level - stop_loss) * 2
        denom = price - stop_loss
        if denom > 0:
            rr = (measured_move - price) / denom
        else:
            rr = 0.0

        return stop_loss, atr, max(0.0, rr)

    @staticmethod
    def _calc_atr(df: pd.DataFrame, period: int = 14) -> float:
        if "high" not in df.columns or "low" not in df.columns:
            return 0.0
        h = df["high"].values.astype(float)
        l = df["low"].values.astype(float)
        c = df["close"].values.astype(float)
        n = min(period + 1, len(c))
        h, l, c = h[-n:], l[-n:], c[-n:]
        trs = [
            max(h[i] - l[i], abs(h[i] - c[i - 1]), abs(l[i] - c[i - 1]))
            for i in range(1, len(c))
        ]
        return float(np.mean(trs)) if trs else 0.0

    @staticmethod
    def _breakout_probability(
        master: float,
        vol_score: float,
        vola_score: float,
        pivot_dist_pct: float,
    ) -> float:
        base = master / 100 * 0.5
        vol_contribution = vol_score / 100 * 0.25
        vola_contribution = vola_score / 100 * 0.15
        proximity_boost = 0.10 * max(0.0, 1 - pivot_dist_pct / 10) if pivot_dist_pct >= 0 else 0.10
        return min(1.0, max(0.0, base + vol_contribution + vola_contribution + proximity_boost))
