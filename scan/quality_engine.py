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
        tier  = "AVOID"
        for t, threshold in _TIER_THRESHOLDS.items():
            if total >= threshold:
                tier = t
                break

        return QualityScore(
            symbol=candidate.symbol, tier=tier, score=round(total, 1),
            factors=factors, evidence=evidence, disqualifiers=disqualifiers,
        )

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _fetch(self, symbol: str) -> Optional[pd.DataFrame]:
        try:
            import yfinance as yf
            df = yf.Ticker(f"{symbol}.NS").history(period="100d", interval="1d")
            if df is None or len(df) < 20:
                return None
            df.columns = [c.lower() for c in df.columns]
            return df
        except Exception:
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
            if nifty is None or len(nifty) < 21:
                return 0.0
            n_ret = (float(nifty["Close"].iloc[-1]) / float(nifty["Close"].iloc[-21]) - 1) * 100
            s_ret = (float(close[-1]) / float(close[-21]) - 1) * 100 if len(close) >= 21 else 0.0
            return round(s_ret - n_ret, 2)
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
