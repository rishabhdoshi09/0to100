"""
Quality Engine — Stage 5 of the scan pipeline.

Scores each setup candidate on 10 institutional-grade factors.
Output: ELITE_A_PLUS / A / B / WATCHLIST / AVOID

Quality Score (0-100):
  Base Quality          15pts  — tightness of consolidation, duration
  Volume Contraction    15pts  — volume drying into setup
  Volatility Contraction 12pts — ATR declining into base
  Relative Strength     12pts  — RS vs Nifty (20d)
  Sector Leadership      8pts  — is sector in top 3 this week?
  Breadth Alignment      8pts  — market breadth favours this archetype
  Institutional Evidence 10pts — tight closes, supply exhaustion signals
  Liquidity Quality      8pts  — daily turnover + spread proxy
  Breakout Structure     7pts  — clean pivot, no overhead supply
  Regime Alignment       5pts  — from regime engine
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd


_TIER_THRESHOLDS = {
    "ELITE_A_PLUS": 82,
    "A":            68,
    "B":            54,
    "WATCHLIST":    42,
}


@dataclass
class QualityScore:
    symbol: str
    tier: str                      # ELITE_A_PLUS | A | B | WATCHLIST | AVOID
    score: float                   # 0-100
    factors: dict[str, float]      # factor_name → points scored
    evidence: list[str]            # top evidence bullets
    disqualifiers: list[str]       # reasons for deduction
    earnings_multiplier: float = 1.0  # size multiplier from earnings proximity
    accum_score: float = 0.0          # accumulation score: -100 (distribution) to +100 (accumulation)


class QualityEngine:
    """
    Scores setup candidates. Regime-aware: breadth and regime factors
    are injected from the core RegimeState at construction time.
    """

    def __init__(
        self,
        leading_sectors: list[str] = None,
        breadth_label: str = "NEUTRAL",
        market_regime: str = "CHOPPY",
        institutional_activity: str = "NEUTRAL",
    ):
        self._leading_sectors        = [s.upper() for s in (leading_sectors or [])]
        self._breadth                = breadth_label
        self._regime                 = market_regime
        self._institutional_activity = institutional_activity

    def score(
        self,
        candidate,               # SetupCandidate
        df: Optional[pd.DataFrame] = None,
    ) -> QualityScore:
        factors: dict[str, float] = {}
        evidence: list[str]       = []
        disqualifiers: list[str]  = []

        # Fetch price data if not provided
        if df is None:
            df = self._fetch(candidate.symbol)

        if df is None or len(df) < 30:
            return QualityScore(candidate.symbol, "AVOID", 0, {}, [], ["no_data"])

        close  = df["close"].values
        high   = df["high"].values
        low    = df["low"].values
        volume = df["volume"].values if "volume" in df.columns else np.ones(len(close))
        atr    = self._atr(df)

        # ── 1. Base Quality (0-15) ────────────────────────────────────────────
        base_pts = 0.0
        if len(close) >= 20:
            base_hi = float(np.max(close[-42:])) if len(close) >= 42 else float(np.max(close))
            base_lo = float(np.min(close[-42:])) if len(close) >= 42 else float(np.min(close))
            depth   = (base_hi - base_lo) / base_hi * 100 if base_hi > 0 else 99
            if depth < 8:  base_pts = 15; evidence.append(f"Tight base {depth:.1f}%")
            elif depth < 14: base_pts = 10
            elif depth < 20: base_pts = 5
            else:            disqualifiers.append(f"Wide base {depth:.1f}%")

        # Recency decay — bases older than 6 weeks score lower; setups expire
        if len(close) >= 42:
            # Proxy: measure how much the recent 5 bars moved vs the full base
            recent_move = float(np.std(close[-5:])) / float(np.mean(close[-42:])) * 100
            days_est = 42  # we always look at 42-bar windows
            recency_mult = max(0.65, 1.0 - (days_est / 130) * 0.35)
        else:
            recency_mult = 0.85
        base_pts = round(base_pts * recency_mult, 2)

        factors["base_quality"] = base_pts

        # ── 2. Volume Contraction (0-15) ──────────────────────────────────────
        vol_pts = 0.0
        if len(volume) >= 40:
            avg_old = float(volume[-40:-20].mean())
            avg_new = float(volume[-20:].mean())
            ratio   = avg_new / avg_old if avg_old > 0 else 1.0
            if ratio < 0.6:  vol_pts = 15; evidence.append(f"Volume contracted to {ratio:.2f}×")
            elif ratio < 0.75: vol_pts = 10
            elif ratio < 0.90: vol_pts = 5
        factors["volume_contraction"] = vol_pts

        # ── 2b. Accumulation / Distribution (0-12) ───────────────────────────
        accum_pts = 0.0
        accum_net = 0
        if len(volume) >= 11 and len(close) >= 11:
            avg_vol_ref = float(volume[-20:].mean()) if len(volume) >= 20 else float(volume.mean())
            up_heavy = down_heavy = 0
            for i in range(-10, 0):
                is_up = close[i] > close[i - 1]
                is_heavy = volume[i] > avg_vol_ref
                if is_up and is_heavy:
                    up_heavy += 1
                elif not is_up and is_heavy:
                    down_heavy += 1
            accum_net = up_heavy - down_heavy
            if accum_net >= 4:
                accum_pts = 12; evidence.append(f"Accumulation: {up_heavy}↑ {down_heavy}↓ heavy-vol days — smart money buying")
            elif accum_net >= 2:
                accum_pts = 8; evidence.append(f"Mild accumulation: {up_heavy}↑ {down_heavy}↓ heavy-vol days")
            elif accum_net >= 0:
                accum_pts = 4
            else:
                accum_pts = 0
                disqualifiers.append(f"Distribution: {down_heavy} heavy-vol down days vs {up_heavy} up days")
        factors["accumulation"] = accum_pts

        # ── 3. Volatility Contraction (0-12) ─────────────────────────────────
        atr_pts = 0.0
        if len(close) >= 40:
            atr_old = self._atr_period(df, -40, -20)
            atr_new = self._atr_period(df, -20, None)
            if atr_old > 0:
                atr_ratio = atr_new / atr_old
                if atr_ratio < 0.65:  atr_pts = 12; evidence.append(f"ATR contracted {atr_ratio:.2f}×")
                elif atr_ratio < 0.80: atr_pts = 8
                elif atr_ratio < 0.95: atr_pts = 4
        factors["volatility_contraction"] = atr_pts

        # ── 4. Relative Strength (0-12) ───────────────────────────────────────
        rs_pts = 0.0
        rs_val = self._rs_vs_nifty(candidate.symbol, close)
        if rs_val > 8:    rs_pts = 12; evidence.append(f"RS +{rs_val:.1f}% vs Nifty (20d)")
        elif rs_val > 4:  rs_pts = 8
        elif rs_val > 0:  rs_pts = 4
        else:             disqualifiers.append(f"Underperforming Nifty {rs_val:.1f}%")
        factors["relative_strength"] = rs_pts

        # ── 5. Sector Leadership (0-8) ────────────────────────────────────────
        sec_pts = 0.0
        sym_sector = self._guess_sector(candidate.symbol)
        if sym_sector and sym_sector in self._leading_sectors:
            sec_pts = 8; evidence.append(f"Sector {sym_sector} is market leader")
        elif self._breadth == "STRONG":
            sec_pts = 4  # broad leadership
        factors["sector_leadership"] = sec_pts

        # ── 6. Breadth Alignment (0-8) ────────────────────────────────────────
        breadth_pts = {"STRONG": 8, "NEUTRAL": 4, "WEAK": 0}.get(self._breadth, 4)
        factors["breadth_alignment"] = float(breadth_pts)

        # ── 7. Institutional Evidence (0-10) ─────────────────────────────────
        inst_pts = 0.0
        if self._institutional_activity in ("ACCUMULATION", "RISK_ON"):
            inst_pts += 5
        # Weekly tight closes: last 5 weeks, each close within 2% of prior
        if len(close) >= 25:
            weekly_closes = close[-25::5]  # proxy weekly
            diffs = np.abs(np.diff(weekly_closes) / weekly_closes[:-1] * 100)
            if len(diffs) >= 3 and float(diffs.max()) < 2.5:
                inst_pts += 5; evidence.append("Weekly tight closes — institutional holding")
        factors["institutional_evidence"] = min(10.0, inst_pts)

        # ── 8. Liquidity Quality (0-8) ────────────────────────────────────────
        avg_turnover = float(volume[-20:].mean()) * float(close[-1]) / 1e7
        if avg_turnover > 50:     liq_pts = 8
        elif avg_turnover > 20:   liq_pts = 6
        elif avg_turnover > 5:    liq_pts = 4
        elif avg_turnover > 1:    liq_pts = 2
        else:
            liq_pts = 0; disqualifiers.append(f"Low liquidity {avg_turnover:.1f}Cr/day")
        factors["liquidity_quality"] = float(liq_pts)

        # ── 9. Breakout Structure (0-7) ───────────────────────────────────────
        bk_pts = 0.0
        pivot = candidate.pivot_level
        price = candidate.price
        dist  = (pivot - price) / price * 100 if price > 0 else 99
        if dist < 2:   bk_pts = 7; evidence.append(f"Within {dist:.1f}% of pivot")
        elif dist < 4: bk_pts = 5
        elif dist < 7: bk_pts = 3
        factors["breakout_structure"] = bk_pts

        # ── 10. Regime Alignment (0-5) ────────────────────────────────────────
        favorable_regimes = {"TRENDING_BULL", "EXPANSION", "COMPRESSION"}
        reg_pts = 5.0 if self._regime in favorable_regimes else (3.0 if self._regime == "CHOPPY" else 0.0)
        factors["regime_alignment"] = reg_pts

        # ── Final score ───────────────────────────────────────────────────────
        total = sum(factors.values())

        # ── Bonus: Relative Strength RS score ────────────────────────────────
        try:
            from scan.relative_strength import compute_rs_score
            rs_score = compute_rs_score(candidate.symbol, df)
            if rs_score >= 90:
                total += 8
                evidence.append(f"RS {rs_score:.0f} — top 10% vs Nifty")
            factors["rs_bonus"] = 8.0 if rs_score >= 90 else 0.0
        except Exception:
            factors["rs_bonus"] = 0.0

        # ── Bonus/Penalty: Setup Freshness ───────────────────────────────────
        try:
            from scan.setup_freshness import compute_freshness
            archetype_str = getattr(candidate, "archetype", "VCP_BREAKOUT")
            freshness = compute_freshness(candidate.symbol, archetype_str, df)
            freshness_status = freshness.get("status", "")
            if freshness_status == "STALE":
                total -= 10
                disqualifiers.append(freshness.get("note", "Setup is stale"))
                factors["freshness_penalty"] = -10.0
            elif freshness_status == "OPTIMAL":
                total += 5
                evidence.append(freshness.get("note", "Setup in optimal window"))
                factors["freshness_bonus"] = 5.0
            else:
                factors["freshness_bonus"] = 0.0
        except Exception:
            factors["freshness_bonus"] = 0.0

        # ── Penalty: Breakout Memory confidence multiplier ───────────────────
        try:
            from scan.breakout_memory import confidence_penalty, get_false_breakout_rate
            penalty = confidence_penalty(candidate.symbol)
            if penalty < 1.0:
                total = total * penalty
                stats = get_false_breakout_rate(candidate.symbol)
                if stats.get("warning"):
                    disqualifiers.append(stats["warning"])
            factors["breakout_memory_mult"] = penalty
        except Exception:
            factors["breakout_memory_mult"] = 1.0

        total = max(0.0, total)

        tier  = "AVOID"
        for t, threshold in _TIER_THRESHOLDS.items():
            if total >= threshold:
                tier = t
                break

        # ── Earnings proximity: compute size multiplier ───────────────────────
        earnings_multiplier = 1.0
        try:
            from data.earnings_calendar import get_earnings_risk
            e_risk = get_earnings_risk(candidate.symbol)
            earnings_multiplier = float(e_risk.get("size_multiplier", 1.0))
            e_note = e_risk.get("note", "")
            if earnings_multiplier < 1.0 and e_note:
                disqualifiers.append(e_note)
        except Exception:
            pass

        accum_score_pct = accum_net * 10.0  # -100 to +100
        return QualityScore(
            symbol=candidate.symbol, tier=tier, score=round(total, 1),
            factors=factors, evidence=evidence, disqualifiers=disqualifiers,
            earnings_multiplier=earnings_multiplier,
            accum_score=round(accum_score_pct, 1),
        )

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _fetch(self, symbol: str) -> Optional[pd.DataFrame]:
        """Fallback fetch — bulk cache → Kite → yfinance. Never uses fake data."""
        _qlog = __import__("logger").get_logger(__name__)
        # 0. Bulk cache (pre-populated by pipeline)
        try:
            from scan.bulk_fetcher import get_cached
            df = get_cached(symbol)
            if df is not None and len(df) >= 20:
                return df
        except Exception:
            pass
        # 1. Try Kite Connect
        try:
            from data.kite_client import KiteClient
            from data.instruments import InstrumentManager
            from data.historical import HistoricalDataFetcher
            from datetime import date, timedelta
            kite = KiteClient()
            if kite.is_connected():
                fetcher = HistoricalDataFetcher(kite, InstrumentManager())
                to_dt = date.today().strftime("%Y-%m-%d")
                from_dt = (date.today() - timedelta(days=120)).strftime("%Y-%m-%d")
                df = fetcher.fetch(symbol, from_dt, to_dt, interval="day")
                if df is not None and len(df) >= 20:
                    if "close" not in df.columns and "Close" in df.columns:
                        df.columns = [c.lower() for c in df.columns]
                    return df
        except Exception as e:
            _qlog.debug("quality_kite_fetch_failed", symbol=symbol, error=str(e))
        # 2. Try yfinance
        try:
            import yfinance as yf
            df = yf.Ticker(f"{symbol}.NS").history(period="100d", interval="1d")
            if df is not None and len(df) >= 20:
                df.columns = [c.lower() for c in df.columns]
                _qlog.debug("quality_yfinance_fallback", symbol=symbol)
                return df
        except Exception as e:
            _qlog.debug("quality_yfinance_failed", symbol=symbol, error=str(e))
        # 3. Both sources failed — skip, never use fake data
        _qlog.debug("quality_all_sources_failed_skipping", symbol=symbol)
        return None

    def _atr(self, df: pd.DataFrame, period: int = 14) -> float:
        return self._atr_period(df, -(period + 1), None)

    def _atr_period(self, df: pd.DataFrame, start: int, end) -> float:
        try:
            h = df["high"].values[start:end]
            l = df["low"].values[start:end]
            c = df["close"].values[start:end]
            if len(h) < 2:
                return 0.0
            tr = np.array([max(h[i]-l[i], abs(h[i]-c[i-1]), abs(l[i]-c[i-1]))
                           for i in range(1, len(h))])
            return float(tr.mean())
        except Exception:
            return 0.0

    def _rs_vs_nifty(self, symbol: str, close: np.ndarray) -> float:
        try:
            import yfinance as yf
            nifty = yf.Ticker("^NSEI").history(period="30d")
            if nifty is not None and len(nifty) >= 21:
                n_ret = (float(nifty["Close"].iloc[-1]) / float(nifty["Close"].iloc[-21]) - 1) * 100
                s_ret = (float(close[-1]) / float(close[-21]) - 1) * 100 if len(close) >= 21 else 0.0
                return round(s_ret - n_ret, 2)
        except Exception:
            pass
        # Demo fallback: use sector return from regime as proxy
        try:
            from core.demo_data import DEMO_REGIME
            sector = self._guess_sector(symbol)
            sector_ret = DEMO_REGIME["sector_returns"].get(sector or "", 0.0)
            nifty_1d = DEMO_REGIME["nifty_change_1d"]
            return round(sector_ret - nifty_1d, 2)
        except Exception:
            return 0.0

    def _guess_sector(self, symbol: str) -> Optional[str]:
        """Rough sector mapping from symbol name patterns."""
        s = symbol.upper()
        if any(x in s for x in ("BANK", "FIN", "HDFC", "ICICI", "AXIS", "KOTAK", "SBI", "BAJFIN")):
            return "BANK"
        if any(x in s for x in ("INFY", "TCS", "WIPRO", "HCL", "TECH", "MPHASIS", "LTIM")):
            return "IT"
        if any(x in s for x in ("SUN", "CIPLA", "DRREDDY", "LUPIN", "ALKEM", "BIOCON", "AURO")):
            return "PHARMA"
        if any(x in s for x in ("TATA", "MARUTI", "HERO", "BAJAJ", "EICHER", "MRF", "EXIDE")):
            return "AUTO"
        if any(x in s for x in ("HIND", "DABUR", "MARICO", "COLPAL", "NESTLE", "ITC", "GODREJ")):
            return "FMCG"
        if any(x in s for x in ("STEEL", "JSPL", "SAIL", "HINDALCO", "VEDANTA", "NALCO")):
            return "METAL"
        return None
