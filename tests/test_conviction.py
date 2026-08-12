"""Tests for ConvictionScorer, TraderProfile, and TradeSetup."""

import pytest
from signals.profiles import CONSERVATIVE, AGGRESSIVE, TraderProfile
from signals.conviction import ConvictionScorer, _clamp
from signals.trade_setup import compute_trade_setup


# ── Profile tests ──────────────────────────────────────────────────────────────

def test_conservative_weights_sum_to_one():
    assert abs(sum(CONSERVATIVE.weights.values()) - 1.0) < 1e-6


def test_aggressive_weights_sum_to_one():
    assert abs(sum(AGGRESSIVE.weights.values()) - 1.0) < 1e-6


def test_profile_rejects_bad_weights():
    with pytest.raises(ValueError):
        TraderProfile(name="Bad", weights={"trend": 0.5, "rsi": 0.3})


# ── ConvictionScorer — BUY scenario ───────────────────────────────────────────

BUY_INDICATORS = {
    "rsi_14": 35.0,          # oversold → bullish
    "momentum_5d_pct": 0.04, # +4% → strong momentum
    "volume_ratio": 2.0,     # high volume confirmation
    "sma_20": 1050.0,
    "sma_50": 1000.0,        # sma20 > sma50 → uptrend
    "ml_proba": 0.75,        # strong ML signal
}

def test_buy_verdict():
    result = ConvictionScorer(AGGRESSIVE).score(BUY_INDICATORS)
    assert result.verdict == "BUY", f"Expected BUY, got {result.verdict} (score={result.score})"
    assert result.score >= 62


def test_buy_gates_pass():
    result = ConvictionScorer().score(BUY_INDICATORS)
    assert result.gates_passed
    assert result.gate_failures == []


# ── ConvictionScorer — HOLD scenario (weak signals) ───────────────────────────

HOLD_INDICATORS = {
    "rsi_14": 52.0,
    "momentum_5d_pct": 0.001,
    "volume_ratio": 1.0,
    "sma_20": 1001.0,
    "sma_50": 1000.0,
    "ml_proba": 0.50,
}

def test_hold_verdict():
    result = ConvictionScorer().score(HOLD_INDICATORS)
    assert result.verdict == "HOLD"


# ── Gate failures cap score ───────────────────────────────────────────────────

OVERBOUGHT_INDICATORS = {
    "rsi_14": 85.0,          # gate fail: RSI > 80
    "momentum_5d_pct": 0.05,
    "volume_ratio": 2.5,
    "sma_20": 1100.0,
    "sma_50": 1000.0,
    "ml_proba": 0.80,
}

def test_gate_failure_forces_hold():
    result = ConvictionScorer().score(OVERBOUGHT_INDICATORS)
    assert result.verdict == "HOLD"
    assert result.score <= 30
    assert not result.gates_passed
    assert any("RSI" in f for f in result.gate_failures)


def test_low_volume_gate():
    ind = {**BUY_INDICATORS, "volume_ratio": 0.5}
    result = ConvictionScorer().score(ind)
    assert not result.gates_passed
    assert any("volume" in f.lower() for f in result.gate_failures)


# ── TradeSetup ────────────────────────────────────────────────────────────────

def test_buy_setup():
    setup = compute_trade_setup("RELIANCE", "BUY", entry_price=2500.0, atr=25.0, capital=100_000)
    assert setup is not None
    assert setup.stop < setup.entry < setup.target
    assert setup.rr_ratio == pytest.approx(2.0, abs=0.01)
    assert setup.quantity >= 1


def test_sell_setup():
    setup = compute_trade_setup("INFY", "SELL", entry_price=1500.0, atr=20.0, capital=100_000)
    assert setup is not None
    assert setup.target < setup.entry < setup.stop


def test_invalid_atr_returns_none():
    assert compute_trade_setup("TCS", "BUY", entry_price=3000.0, atr=0.0) is None


# ── Utility ───────────────────────────────────────────────────────────────────

def test_clamp():
    assert _clamp(1.5) == 1.0
    assert _clamp(-0.5) == 0.0
    assert _clamp(0.5) == 0.5
