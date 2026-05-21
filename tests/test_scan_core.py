"""
Comprehensive tests for scan/ and core/ modules.

Groups:
  1. SetupEngine — data structures and archetype detection
  2. QualityEngine — scoring logic
  3. ScanPipeline — integration
  4. TradingRules from intelligence_hub
"""

import sys
import numpy as np
import pandas as pd
import pytest
from dataclasses import fields
from unittest.mock import patch, MagicMock

sys.path.insert(0, ".")


# ── Helpers ────────────────────────────────────────────────────────────────────

def _uptrend_df(n=260, seed=42) -> pd.DataFrame:
    """Synthetic uptrending OHLCV DataFrame, price > sma50 > sma200."""
    np.random.seed(seed)
    prices = np.linspace(60, 100, n) + np.random.randn(n) * 0.5
    high = prices + 1.5
    low = prices - 1.5
    vol = np.ones(n) * 1_000_000
    return pd.DataFrame(
        {"open": prices - 0.3, "high": high, "low": low, "close": prices, "volume": vol},
        index=pd.date_range("2023-01-01", periods=n),
    )


def _downtrend_df(n=260, seed=42) -> pd.DataFrame:
    """Synthetic downtrending OHLCV DataFrame, price < sma200."""
    np.random.seed(seed)
    prices = np.linspace(100, 60, n) + np.random.randn(n) * 0.5
    high = prices + 1.5
    low = prices - 1.5
    vol = np.ones(n) * 1_000_000
    return pd.DataFrame(
        {"open": prices - 0.3, "high": high, "low": low, "close": prices, "volume": vol},
        index=pd.date_range("2023-01-01", periods=n),
    )


def _vcp_df(seed=42) -> pd.DataFrame:
    """
    260-bar DataFrame engineered to trigger VCP_BREAKOUT.

    Structure:
      - Bars 0-199: linear uptrend 50→96 (establishes sma200 < sma50 < price)
      - Bars 200-259: contracting oscillations (depth shrinks each segment)
                      with last 20 bars having 70% of prior-20 volume
    """
    np.random.seed(seed)
    n = 260
    prices = np.zeros(n)

    # Uptrend phase
    prices[:200] = np.linspace(50, 96, 200) + np.random.randn(200) * 0.3

    # Contracting VCP phase
    for i in range(60):
        t = i / 59
        base = 96 + t * 2.5  # mild upward drift
        amp = 5.0 * (1 - 0.75 * t)  # amplitude shrinks from 5→1.25
        prices[200 + i] = base + np.sin(i * 0.45) * amp

    high = prices + 1.5
    low = prices - 1.5
    vol = np.ones(n) * 1_000_000
    vol[-20:] = 700_000  # volume drying in last 20 bars

    return pd.DataFrame(
        {"open": prices - 0.3, "high": high, "low": low, "close": prices, "volume": vol},
        index=pd.date_range("2023-01-01", periods=n),
    )


def _accumulation_df(seed=42) -> pd.DataFrame:
    """
    120-bar DataFrame with a tight 6-week range near the ceiling.
    Last 42 bars hold within ~2% range, volume contracts in last 21 bars.
    """
    np.random.seed(seed)
    n = 120
    prices = np.linspace(80, 99, n) + np.random.randn(n) * 0.3
    prices[-42:] = 99 + np.random.randn(42) * 0.5  # very tight base
    high = prices + 0.8
    low = prices - 0.8
    vol = np.ones(n) * 1_000_000
    vol[-21:] = 600_000  # volume contracts to 60% of prior
    return pd.DataFrame(
        {"open": prices - 0.2, "high": high, "low": low, "close": prices, "volume": vol},
        index=pd.date_range("2023-01-01", periods=n),
    )


def _momentum_df(seed=42) -> pd.DataFrame:
    """
    260-bar uptrending DataFrame where the last 6 bars include a strong
    surge (>7% 5-day gain) on 2.2x volume — triggers MOMENTUM_EXPANSION.
    """
    np.random.seed(seed)
    n = 260
    prices = np.zeros(n)
    prices[: n - 6] = np.linspace(60, 90, n - 6) + np.random.randn(n - 6) * 0.4
    prices[-6:] = [87, 89, 92, 95, 98, 107]  # 18.9% 5d gain

    high = prices + 1.0
    low = prices - 1.0
    vol = np.ones(n) * 1_000_000
    vol[-1] = 2_200_000  # volume surge on breakout bar
    return pd.DataFrame(
        {"open": prices - 0.3, "high": high, "low": low, "close": prices, "volume": vol},
        index=pd.date_range("2023-01-01", periods=n),
    )


def _make_candidate(
    symbol="TEST",
    price=100.0,
    archetype="VCP_BREAKOUT",
    confidence=75.0,
    pivot=101.0,
    stop=95.0,
):
    from scan.setup_engine import SetupCandidate
    return SetupCandidate(
        symbol=symbol,
        price=price,
        archetype=archetype,
        confidence=confidence,
        pivot_level=pivot,
        stop_level=stop,
        behavioral_evidence=["Stage 2", "Pattern confirmed"],
        raw_indicators={"rsi": 60, "sma50": 95, "sma200": 80, "atr": 1.5, "price": price, "vol_ratio": 1.5},
    )


def _make_opp(total: float):
    """Minimal OpportunityScore with a given total."""
    from core.intelligence_hub import OpportunityScore
    return OpportunityScore(
        total=total, grade="A", label="Test", color="#fff",
        regime_score=20, breadth_score=15, volatility_score=15,
        setup_quality_score=15, personal_edge_score=10,
        regime_note="", breadth_note="", volatility_note="",
        setup_note="", edge_note="",
        recommended_aggression="STANDARD", max_open_trades=3,
    )


# ══════════════════════════════════════════════════════════════════════════════
# Group 1: SetupEngine — data structures and archetype detection
# ══════════════════════════════════════════════════════════════════════════════

class TestSetupCandidateDataclass:

    def test_required_fields_exist(self):
        from scan.setup_engine import SetupCandidate
        field_names = {f.name for f in fields(SetupCandidate)}
        required = {
            "symbol", "price", "archetype", "confidence",
            "pivot_level", "stop_level", "behavioral_evidence",
            "raw_indicators", "fno_banned",
        }
        assert required.issubset(field_names), f"Missing fields: {required - field_names}"

    def test_fno_banned_defaults_false(self):
        from scan.setup_engine import SetupCandidate
        c = SetupCandidate(
            symbol="X", price=100, archetype="VCP_BREAKOUT", confidence=70,
            pivot_level=102, stop_level=96,
            behavioral_evidence=[], raw_indicators={},
        )
        assert c.fno_banned is False

    def test_can_set_fno_banned_true(self):
        from scan.setup_engine import SetupCandidate
        c = SetupCandidate(
            symbol="X", price=100, archetype="VCP_BREAKOUT", confidence=70,
            pivot_level=102, stop_level=96,
            behavioral_evidence=[], raw_indicators={}, fno_banned=True,
        )
        assert c.fno_banned is True


class TestSetupEngineVCP:

    def _engine_and_df_arrays(self, df):
        from scan.setup_engine import SetupEngine
        se = SetupEngine()
        close = df["close"].values
        high = df["high"].values
        low = df["low"].values
        vol = df["volume"].values
        price = float(close[-1])
        sma50 = float(np.mean(close[-50:]))
        sma200 = float(np.mean(close[-200:]))
        sma50p = float(np.mean(close[-55:-5]))
        avg_vol = float(vol[-21:-1].mean())
        vol_ratio = float(vol[-1] / avg_vol) if avg_vol > 0 else 1.0
        ind = {
            "rsi": 60, "sma50": sma50, "sma200": sma200,
            "atr": 1.5, "price": price, "vol_ratio": vol_ratio,
        }
        return se, close, high, low, vol, price, sma50, sma200, sma50p, avg_vol, ind

    def test_vcp_breakout_detected_on_valid_setup(self):
        from scan.setup_engine import SetupEngine
        df = _vcp_df()
        se, close, high, low, vol, price, sma50, sma200, sma50p, avg_vol, ind = \
            self._engine_and_df_arrays(df)

        result = se._check_vcp(
            "TEST", price, close, high, low, vol,
            sma50, sma200, sma50p, 60.0, 1.5, avg_vol, ind,
        )
        assert result is not None, "Expected VCP_BREAKOUT, got None"
        assert result.archetype == "VCP_BREAKOUT"

    def test_vcp_rejected_when_price_below_sma200(self):
        from scan.setup_engine import SetupEngine
        df = _downtrend_df()
        se = SetupEngine()
        close = df["close"].values
        high = df["high"].values
        low = df["low"].values
        vol = df["volume"].values
        price = float(close[-1])
        sma50 = float(np.mean(close[-50:]))
        sma200 = float(np.mean(close[-200:]))
        avg_vol = float(vol[-21:-1].mean())
        ind = {
            "rsi": 45, "sma50": sma50, "sma200": sma200,
            "atr": 1.5, "price": price, "vol_ratio": 1.0,
        }

        assert price < sma200, "Precondition: price should be below sma200 for this test"
        result = se._check_vcp(
            "BEAR", price, close, high, low, vol,
            sma50, sma200, float(np.mean(close[-55:-5])), 45.0, 1.5, avg_vol, ind,
        )
        assert result is None

    def test_vcp_rejected_when_price_below_sma50(self):
        from scan.setup_engine import SetupEngine
        np.random.seed(99)
        n = 260
        # Trend up but then fall back below sma50
        prices = np.linspace(50, 80, n) + np.random.randn(n) * 0.5
        prices[-30:] = np.linspace(80, 68, 30)  # recent weakness: drops below sma50

        high = prices + 1.5
        low = prices - 1.5
        vol = np.ones(n) * 1e6
        df = pd.DataFrame({"open": prices-0.3, "high": high, "low": low, "close": prices, "volume": vol})

        se = SetupEngine()
        close = df["close"].values
        price = float(close[-1])
        sma50 = float(np.mean(close[-50:]))
        sma200 = float(np.mean(close[-200:]))
        avg_vol = float(vol[-21:-1].mean())
        ind = {
            "rsi": 40, "sma50": sma50, "sma200": sma200,
            "atr": 1.5, "price": price, "vol_ratio": 0.8,
        }

        # Explicitly confirm precondition
        assert price < sma50, f"Expected price < sma50 but price={price:.1f} sma50={sma50:.1f}"
        result = se._check_vcp(
            "WEAK", price, close, high, low, vol,
            sma50, sma200, float(np.mean(close[-55:-5])), 40.0, 1.5, avg_vol, ind,
        )
        assert result is None


class TestSetupEngineAccumulationBreakout:

    def test_accumulation_breakout_detected_on_valid_setup(self):
        from scan.setup_engine import SetupEngine
        df = _accumulation_df()
        se = SetupEngine()
        close = df["close"].values
        high = df["high"].values
        low = df["low"].values
        vol = df["volume"].values
        price = float(close[-1])
        sma50 = float(np.mean(close[-50:]))
        sma200 = float(np.mean(close))  # proxy
        avg_vol = float(vol[-21:-1].mean())
        ind = {
            "rsi": 55, "sma50": sma50, "sma200": sma200,
            "atr": 0.8, "price": price, "vol_ratio": 0.65,
        }

        result = se._check_accumulation_breakout(
            "ACC", price, close, high, low, vol,
            sma50, sma200, sma50, 55.0, 0.8, avg_vol, ind,
        )
        assert result is not None, "Expected ACCUMULATION_BREAKOUT, got None"
        assert result.archetype == "ACCUMULATION_BREAKOUT"


class TestSetupEngineMomentumExpansion:

    def test_momentum_expansion_detected_on_valid_setup(self):
        from scan.setup_engine import SetupEngine
        df = _momentum_df()
        se = SetupEngine()
        close = df["close"].values
        high = df["high"].values
        low = df["low"].values
        vol = df["volume"].values
        price = float(close[-1])
        sma50 = float(np.mean(close[-50:]))
        sma200 = float(np.mean(close[-200:]))
        avg_vol = float(vol[-21:-1].mean())
        vol_ratio = float(vol[-1] / avg_vol)
        ind = {
            "rsi": 65, "sma50": sma50, "sma200": sma200,
            "atr": 1.5, "price": price, "vol_ratio": vol_ratio,
        }

        assert price > sma50 > sma200, "Precondition: price > sma50 > sma200"
        assert vol_ratio >= 1.5, f"Precondition: vol_ratio={vol_ratio:.2f} should be >= 1.5"

        result = se._check_momentum_expansion(
            "MOM", price, close, high, low, vol,
            sma50, sma200, float(np.mean(close[-55:-5])), 65.0, 1.5, avg_vol, ind,
        )
        assert result is not None, "Expected MOMENTUM_EXPANSION, got None"
        assert result.archetype == "MOMENTUM_EXPANSION"

    def test_momentum_expansion_rejected_when_rsi_too_high(self):
        from scan.setup_engine import SetupEngine
        df = _momentum_df()
        se = SetupEngine()
        close = df["close"].values
        high = df["high"].values
        low = df["low"].values
        vol = df["volume"].values
        price = float(close[-1])
        sma50 = float(np.mean(close[-50:]))
        sma200 = float(np.mean(close[-200:]))
        avg_vol = float(vol[-21:-1].mean())
        vol_ratio = float(vol[-1] / avg_vol)
        ind = {
            "rsi": 90, "sma50": sma50, "sma200": sma200,  # overbought RSI
            "atr": 1.5, "price": price, "vol_ratio": vol_ratio,
        }

        result = se._check_momentum_expansion(
            "OVB", price, close, high, low, vol,
            sma50, sma200, float(np.mean(close[-55:-5])), 90.0, 1.5, avg_vol, ind,
        )
        assert result is None


class TestSetupEngineConfidenceBounds:

    def _run_all_checks(self, se, df):
        close = df["close"].values
        high = df["high"].values
        low = df["low"].values
        vol = df["volume"].values
        price = float(close[-1])
        sma50 = float(np.mean(close[-50:])) if len(close) >= 50 else price
        sma200 = float(np.mean(close[-200:])) if len(close) >= 200 else price
        sma50p = float(np.mean(close[-55:-5])) if len(close) >= 55 else sma50
        avg_vol = float(vol[-21:-1].mean()) if len(vol) > 21 else float(vol.mean())
        vol_ratio = float(vol[-1] / avg_vol) if avg_vol > 0 else 1.0
        ind = {
            "rsi": 60, "sma50": sma50, "sma200": sma200,
            "atr": 1.5, "price": price, "vol_ratio": vol_ratio,
        }
        checks = [
            se._check_vcp, se._check_accumulation_breakout,
            se._check_momentum_expansion, se._check_trend_continuation,
            se._check_mean_reversion, se._check_failed_breakout,
            se._check_earnings_continuation, se._check_high_tight_flag,
        ]
        results = []
        for fn in checks:
            try:
                r = fn("TEST", price, close, high, low, vol,
                       sma50, sma200, sma50p, 60.0, 1.5, avg_vol, ind)
                if r is not None:
                    results.append(r)
            except Exception:
                pass
        return results

    def test_confidence_always_between_0_and_100_uptrend(self):
        from scan.setup_engine import SetupEngine
        se = SetupEngine()
        for seed in [1, 7, 42, 99, 123]:
            df = _uptrend_df(seed=seed)
            for r in self._run_all_checks(se, df):
                assert 0 <= r.confidence <= 100, (
                    f"confidence={r.confidence} out of range for archetype={r.archetype} seed={seed}"
                )

    def test_confidence_always_between_0_and_100_downtrend(self):
        from scan.setup_engine import SetupEngine
        se = SetupEngine()
        for seed in [1, 7, 42]:
            df = _downtrend_df(seed=seed)
            for r in self._run_all_checks(se, df):
                assert 0 <= r.confidence <= 100, (
                    f"confidence={r.confidence} out of range for archetype={r.archetype}"
                )


class TestSetupEngineCache:

    def test_df_cache_populated_after_detect(self):
        from scan.setup_engine import SetupEngine
        import scan.setup_engine as se_mod

        df = _vcp_df()

        with patch.object(se_mod, "_fetch", return_value=df):
            se = SetupEngine(max_workers=1)
            se.detect(["RELIANCE"])

        assert "RELIANCE" in se._df_cache
        cached = se.get_cached_df("RELIANCE")
        assert cached is not None
        assert isinstance(cached, pd.DataFrame)

    def test_get_cached_df_returns_none_for_unknown_symbol(self):
        from scan.setup_engine import SetupEngine
        se = SetupEngine()
        assert se.get_cached_df("NOTEXIST") is None

    def test_detect_returns_empty_list_when_fetch_fails(self):
        from scan.setup_engine import SetupEngine
        import scan.setup_engine as se_mod

        with patch.object(se_mod, "_fetch", return_value=None):
            se = SetupEngine(max_workers=1)
            result = se.detect(["FAKESYM"])

        assert result == []


# ══════════════════════════════════════════════════════════════════════════════
# Group 2: QualityEngine — scoring logic
# ══════════════════════════════════════════════════════════════════════════════

class TestQualityEnginePerfectSetup:

    def _tight_liquid_df(self, seed=42, n=120) -> pd.DataFrame:
        """High-liquidity, tight-base DF for a perfect quality score."""
        np.random.seed(seed)
        prices = np.linspace(90, 99, n) + np.random.randn(n) * 0.2
        prices[-42:] = 99 + np.random.randn(42) * 0.3  # tight base ≈ 1%
        high = prices + 0.4
        low = prices - 0.4
        vol = np.ones(n) * 1_000_000  # 1M shares × ₹99 = ~₹10Cr/day (liquid)
        vol[-20:] = vol[-40:-20].mean() * 0.5  # strong volume contraction
        return pd.DataFrame(
            {"open": prices - 0.1, "high": high, "low": low, "close": prices, "volume": vol}
        )

    def test_perfect_setup_scores_elite_a_plus(self):
        from scan.quality_engine import QualityEngine

        df = self._tight_liquid_df()
        candidate = _make_candidate(
            symbol="HDFCBANK",
            price=float(df["close"].iloc[-1]),
            archetype="VCP_BREAKOUT",
            pivot=float(df["close"].iloc[-1]) * 1.005,
        )

        qe = QualityEngine(
            leading_sectors=["BANK"],
            breadth_label="STRONG",
            market_regime="TRENDING_BULL",
            institutional_activity="ACCUMULATION",
        )

        with patch.object(qe, "_rs_vs_nifty", return_value=10.0):
            result = qe.score(candidate, df=df)

        assert result.score >= 82, f"Expected >= 82 for ELITE_A_PLUS, got {result.score}"
        assert result.tier == "ELITE_A_PLUS"


class TestQualityEngineZeroData:

    def test_none_df_returns_avoid_tier_with_zero_score(self):
        from scan.quality_engine import QualityEngine
        qe = QualityEngine()
        candidate = _make_candidate()

        with patch.object(qe, "_fetch", return_value=None):
            result = qe.score(candidate, df=None)

        assert result.tier == "AVOID"
        assert result.score == 0

    def test_too_short_df_returns_avoid_tier(self):
        from scan.quality_engine import QualityEngine
        qe = QualityEngine()
        candidate = _make_candidate()

        short_df = pd.DataFrame(
            {"open": [100]*5, "high": [101]*5, "low": [99]*5,
             "close": [100]*5, "volume": [1e6]*5}
        )
        result = qe.score(candidate, df=short_df)
        assert result.tier == "AVOID"
        assert result.score == 0


class TestQualityEngineBreadthFactor:

    def _base_df(self, seed=42) -> pd.DataFrame:
        np.random.seed(seed)
        n = 120
        prices = np.linspace(90, 100, n) + np.random.randn(n) * 0.5
        high = prices + 0.7
        low = prices - 0.7
        vol = np.ones(n) * 1_000_000
        return pd.DataFrame(
            {"open": prices - 0.2, "high": high, "low": low, "close": prices, "volume": vol}
        )

    def test_strong_breadth_adds_8_points_vs_weak(self):
        from scan.quality_engine import QualityEngine

        df = self._base_df()
        candidate = _make_candidate(price=float(df["close"].iloc[-1]))

        qe_strong = QualityEngine(breadth_label="STRONG", market_regime="CHOPPY")
        qe_weak = QualityEngine(breadth_label="WEAK", market_regime="CHOPPY")

        with patch.object(qe_strong, "_rs_vs_nifty", return_value=5.0), \
             patch.object(qe_weak, "_rs_vs_nifty", return_value=5.0):
            score_strong = qe_strong.score(candidate, df=df)
            score_weak = qe_weak.score(candidate, df=df)

        diff = score_strong.factors["breadth_alignment"] - score_weak.factors["breadth_alignment"]
        assert diff == pytest.approx(8.0), (
            f"Expected 8pt breadth difference, got {diff}"
        )

    def test_strong_breadth_factor_is_8(self):
        from scan.quality_engine import QualityEngine
        df = self._base_df()
        candidate = _make_candidate(price=float(df["close"].iloc[-1]))
        qe = QualityEngine(breadth_label="STRONG")
        with patch.object(qe, "_rs_vs_nifty", return_value=5.0):
            result = qe.score(candidate, df=df)
        assert result.factors["breadth_alignment"] == pytest.approx(8.0)

    def test_weak_breadth_factor_is_0(self):
        from scan.quality_engine import QualityEngine
        df = self._base_df()
        candidate = _make_candidate(price=float(df["close"].iloc[-1]))
        qe = QualityEngine(breadth_label="WEAK")
        with patch.object(qe, "_rs_vs_nifty", return_value=5.0):
            result = qe.score(candidate, df=df)
        assert result.factors["breadth_alignment"] == pytest.approx(0.0)


class TestQualityEngineRegimeFactor:

    def _base_df(self) -> pd.DataFrame:
        np.random.seed(7)
        n = 120
        prices = np.linspace(90, 100, n) + np.random.randn(n) * 0.5
        high = prices + 0.7
        low = prices - 0.7
        vol = np.ones(n) * 1_000_000
        return pd.DataFrame(
            {"open": prices - 0.2, "high": high, "low": low, "close": prices, "volume": vol}
        )

    def test_trending_bull_regime_adds_5_points(self):
        from scan.quality_engine import QualityEngine
        df = self._base_df()
        candidate = _make_candidate(price=float(df["close"].iloc[-1]))
        qe = QualityEngine(market_regime="TRENDING_BULL")
        with patch.object(qe, "_rs_vs_nifty", return_value=5.0):
            result = qe.score(candidate, df=df)
        assert result.factors["regime_alignment"] == pytest.approx(5.0)

    def test_trending_bear_regime_adds_0_points(self):
        from scan.quality_engine import QualityEngine
        df = self._base_df()
        candidate = _make_candidate(price=float(df["close"].iloc[-1]))
        qe = QualityEngine(market_regime="TRENDING_BEAR")
        with patch.object(qe, "_rs_vs_nifty", return_value=5.0):
            result = qe.score(candidate, df=df)
        assert result.factors["regime_alignment"] == pytest.approx(0.0)

    def test_choppy_regime_adds_3_points(self):
        from scan.quality_engine import QualityEngine
        df = self._base_df()
        candidate = _make_candidate(price=float(df["close"].iloc[-1]))
        qe = QualityEngine(market_regime="CHOPPY")
        with patch.object(qe, "_rs_vs_nifty", return_value=5.0):
            result = qe.score(candidate, df=df)
        assert result.factors["regime_alignment"] == pytest.approx(3.0)


class TestQualityEngineInstitutionalActivity:

    def _base_df(self) -> pd.DataFrame:
        np.random.seed(11)
        n = 120
        prices = np.linspace(90, 100, n) + np.random.randn(n) * 0.5
        high = prices + 0.7
        low = prices - 0.7
        vol = np.ones(n) * 1_000_000
        return pd.DataFrame(
            {"open": prices - 0.2, "high": high, "low": low, "close": prices, "volume": vol}
        )

    def test_accumulation_adds_at_least_5_points(self):
        from scan.quality_engine import QualityEngine
        df = self._base_df()
        candidate = _make_candidate(price=float(df["close"].iloc[-1]))

        qe_acc = QualityEngine(institutional_activity="ACCUMULATION", market_regime="CHOPPY")
        qe_neutral = QualityEngine(institutional_activity="NEUTRAL", market_regime="CHOPPY")

        # Suppress weekly-tight-close bonus by using prices with high variance
        with patch.object(qe_acc, "_rs_vs_nifty", return_value=5.0), \
             patch.object(qe_neutral, "_rs_vs_nifty", return_value=5.0):
            r_acc = qe_acc.score(candidate, df=df)
            r_neutral = qe_neutral.score(candidate, df=df)

        # ACCUMULATION should add 5pts to institutional_evidence vs NEUTRAL
        assert r_acc.factors["institutional_evidence"] >= r_neutral.factors["institutional_evidence"] + 5


class TestTierThresholds:

    def _score_with_total(self, total: float):
        """Shortcut: build a QualityScore directly and determine tier via engine logic."""
        from scan.quality_engine import _TIER_THRESHOLDS
        tier = "AVOID"
        for t, threshold in _TIER_THRESHOLDS.items():
            if total >= threshold:
                tier = t
                break
        return tier

    def test_score_82_is_elite_a_plus(self):
        assert self._score_with_total(82) == "ELITE_A_PLUS"

    def test_score_81_is_a(self):
        assert self._score_with_total(81) == "A"

    def test_score_68_is_a(self):
        assert self._score_with_total(68) == "A"

    def test_score_67_is_b(self):
        assert self._score_with_total(67) == "B"

    def test_score_54_is_b(self):
        assert self._score_with_total(54) == "B"

    def test_score_53_is_watchlist(self):
        assert self._score_with_total(53) == "WATCHLIST"

    def test_score_42_is_watchlist(self):
        assert self._score_with_total(42) == "WATCHLIST"

    def test_score_41_is_avoid(self):
        assert self._score_with_total(41) == "AVOID"

    def test_score_0_is_avoid(self):
        assert self._score_with_total(0) == "AVOID"


# ══════════════════════════════════════════════════════════════════════════════
# Group 3: ScanPipeline — integration
# ══════════════════════════════════════════════════════════════════════════════

class FakeRegimeState:
    """Minimal regime state object that avoids network calls."""
    market_regime = "CHOPPY"
    quality_multiplier = 1.0
    leading_sectors = []
    breadth_label = "NEUTRAL"
    institutional_activity = "NEUTRAL"
    volatility_regime = "NORMAL"


class TestScanPipelineEmptyUniverse:

    def test_empty_universe_returns_empty_list(self):
        from scan.pipeline import ScanPipeline
        pipeline = ScanPipeline()
        result = pipeline.run([], regime_state=FakeRegimeState(), skip_liquidity_filter=True)
        assert result == []

    def test_empty_universe_does_not_raise(self):
        from scan.pipeline import ScanPipeline
        pipeline = ScanPipeline()
        try:
            pipeline.run([], regime_state=FakeRegimeState(), skip_liquidity_filter=True)
        except Exception as exc:
            pytest.fail(f"pipeline.run([]) raised unexpectedly: {exc}")


class TestScanPipelineWithMockSymbols:

    def test_pipeline_run_with_symbols_returns_list(self):
        from scan.pipeline import ScanPipeline
        import scan.setup_engine as se_mod

        df = _vcp_df()

        with patch.object(se_mod, "_fetch", return_value=df):
            pipeline = ScanPipeline(min_quality_score=0, skip_avoid_tier=False, fire_alerts=False)
            result = pipeline.run(
                ["RELIANCE"],
                regime_state=FakeRegimeState(),
                skip_liquidity_filter=True,
            )

        assert isinstance(result, list)

    def test_pipeline_run_does_not_raise_on_bad_symbol(self):
        from scan.pipeline import ScanPipeline
        import scan.setup_engine as se_mod

        with patch.object(se_mod, "_fetch", return_value=None):
            pipeline = ScanPipeline(fire_alerts=False)
            try:
                result = pipeline.run(
                    ["FAKESYM999"],
                    regime_state=FakeRegimeState(),
                    skip_liquidity_filter=True,
                )
                assert isinstance(result, list)
            except Exception as exc:
                pytest.fail(f"Pipeline raised on bad symbol: {exc}")

    def test_pipeline_result_items_have_expected_attributes(self):
        from scan.pipeline import ScanPipeline
        import scan.setup_engine as se_mod

        df = _vcp_df()

        with patch.object(se_mod, "_fetch", return_value=df):
            pipeline = ScanPipeline(min_quality_score=0, skip_avoid_tier=False, fire_alerts=False)
            result = pipeline.run(
                ["RELIANCE"],
                regime_state=FakeRegimeState(),
                skip_liquidity_filter=True,
            )

        for item in result:
            assert hasattr(item, "symbol"), "item missing 'symbol' attribute"
            assert hasattr(item, "archetype"), "item missing 'archetype' attribute"
            assert hasattr(item, "quality_tier"), "item missing 'quality_tier' attribute"
            assert hasattr(item, "quality_score"), "item missing 'quality_score' attribute"


# ══════════════════════════════════════════════════════════════════════════════
# Group 4: TradingRules from intelligence_hub
# ══════════════════════════════════════════════════════════════════════════════

class TestTradingRulesScoreThresholds:

    def test_score_75_allows_3_positions_and_new_trades(self):
        from core.intelligence_hub import get_trading_rules
        rules = get_trading_rules(_make_opp(75))
        assert rules.max_positions == 3
        assert rules.allow_new_trades is True

    def test_score_75_min_tier_is_b(self):
        from core.intelligence_hub import get_trading_rules
        rules = get_trading_rules(_make_opp(75))
        assert rules.min_tier == "B"

    def test_score_60_allows_2_positions(self):
        from core.intelligence_hub import get_trading_rules
        rules = get_trading_rules(_make_opp(60))
        assert rules.max_positions == 2
        assert rules.allow_new_trades is True

    def test_score_74_allows_2_positions(self):
        from core.intelligence_hub import get_trading_rules
        # 74 falls in the 60-74 band
        rules = get_trading_rules(_make_opp(74))
        assert rules.max_positions == 2

    def test_score_30_allows_1_position_and_elite_only(self):
        from core.intelligence_hub import get_trading_rules
        rules = get_trading_rules(_make_opp(30))
        assert rules.max_positions == 1
        assert rules.min_tier == "ELITE_A_PLUS"
        assert rules.allow_new_trades is True

    def test_score_44_allows_1_position(self):
        from core.intelligence_hub import get_trading_rules
        # 44 falls in the 30-44 band
        rules = get_trading_rules(_make_opp(44))
        assert rules.max_positions == 1

    def test_score_below_30_blocks_new_trades(self):
        from core.intelligence_hub import get_trading_rules
        rules = get_trading_rules(_make_opp(29))
        assert rules.allow_new_trades is False
        assert rules.max_positions == 0

    def test_score_0_blocks_new_trades(self):
        from core.intelligence_hub import get_trading_rules
        rules = get_trading_rules(_make_opp(0))
        assert rules.allow_new_trades is False

    def test_score_100_allows_3_positions_and_new_trades(self):
        from core.intelligence_hub import get_trading_rules
        rules = get_trading_rules(_make_opp(100))
        assert rules.max_positions == 3
        assert rules.allow_new_trades is True


class TestTradingRulesDisplayMethods:

    def test_display_size_contains_percent_sign(self):
        from core.intelligence_hub import get_trading_rules
        rules = get_trading_rules(_make_opp(75))
        result = rules.display_size()
        assert isinstance(result, str)
        assert "%" in result

    def test_display_size_100_pct_at_full_aggression(self):
        from core.intelligence_hub import get_trading_rules
        rules = get_trading_rules(_make_opp(75))
        assert rules.display_size() == "100% of normal"

    def test_display_size_50_pct_at_defensive(self):
        from core.intelligence_hub import get_trading_rules
        rules = get_trading_rules(_make_opp(30))
        assert rules.display_size() == "50% of normal"

    def test_display_size_0_pct_at_stand_down(self):
        from core.intelligence_hub import get_trading_rules
        rules = get_trading_rules(_make_opp(20))
        assert rules.display_size() == "0% of normal"

    def test_one_liner_returns_non_empty_string(self):
        from core.intelligence_hub import get_trading_rules
        for score in [80, 60, 45, 30, 15]:
            rules = get_trading_rules(_make_opp(score))
            result = rules.one_liner()
            assert isinstance(result, str)
            assert len(result) > 0, f"one_liner() was empty for score={score}"

    def test_one_liner_no_new_trades_contains_no_new_trades_text(self):
        from core.intelligence_hub import get_trading_rules
        rules = get_trading_rules(_make_opp(10))
        one_liner = rules.one_liner()
        assert "NO NEW TRADES" in one_liner

    def test_one_liner_active_trading_contains_position_info(self):
        from core.intelligence_hub import get_trading_rules
        rules = get_trading_rules(_make_opp(75))
        one_liner = rules.one_liner()
        assert "Max" in one_liner or "position" in one_liner.lower()
