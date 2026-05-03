"""Tests for MetaTrader risk validator."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from sq_ai.backend.meta_trader import MetaTrader, _build_prompt
from sq_ai.backend.llm_clients import DeepSeekClient
from sq_ai.signals.composite_signal import adx


# ─────────────────────────────────────────────────────────────────────────────
# ADX helper
# ─────────────────────────────────────────────────────────────────────────────
def _make_ohlcv(n: int = 60) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    close = 100 + np.cumsum(rng.normal(0, 1, n))
    high = close + rng.uniform(0.5, 2, n)
    low = close - rng.uniform(0.5, 2, n)
    return pd.DataFrame({
        "open": close,
        "high": high,
        "low": low,
        "close": close,
        "volume": rng.integers(100_000, 1_000_000, n).astype(float),
    })


def test_adx_returns_series_same_length():
    df = _make_ohlcv(60)
    result = adx(df, period=14)
    assert len(result) == 60


def test_adx_values_in_range():
    df = _make_ohlcv(100)
    result = adx(df, period=14)
    valid = result.dropna()
    assert (valid >= 0).all()
    assert (valid <= 100).all()


def test_adx_strong_trend():
    """Strongly trending data should produce ADX > 20."""
    n = 100
    close = np.linspace(100, 200, n)
    df = pd.DataFrame({
        "open": close,
        "high": close + 1,
        "low": close - 1,
        "close": close,
        "volume": np.ones(n) * 1_000_000,
    })
    result = adx(df, period=14)
    assert float(result.iloc[-1]) > 20


# ─────────────────────────────────────────────────────────────────────────────
# MetaTrader unit tests
# ─────────────────────────────────────────────────────────────────────────────
def _sample_decision(**kwargs):
    base = {
        "action": "BUY",
        "size_pct": 5.0,
        "stop": 95.0,
        "target": 115.0,
        "confidence": 0.72,
        "reasoning": "Strong uptrend with momentum",
    }
    base.update(kwargs)
    return base


def _sample_features(**kwargs):
    base = {
        "close": 100.0,
        "rsi": 55.0,
        "atr": 2.0,
        "adx": 28.0,
        "regime": 2,
        "momentum_5d": 0.025,
        "zscore_20": 0.3,
        "volatility_20": 0.18,
    }
    base.update(kwargs)
    return base


def _sample_snap(**kwargs):
    base = {
        "equity": 1_000_000.0,
        "cash": 800_000.0,
        "exposure_pct": 20.0,
        "daily_pnl_pct": 0.1,
        "positions": [],
    }
    base.update(kwargs)
    return base


# ── fallback when DeepSeek unavailable ────────────────────────────────────────
def test_meta_trader_fallback_when_unavailable():
    mt = MetaTrader(deepseek=DeepSeekClient(api_key="REPLACE_ME"))
    result = mt.validate("RELIANCE", _sample_decision(), _sample_features(), _sample_snap())
    assert result["verdict"] == "ACCEPT"
    assert result["size_multiplier"] == 1.0
    assert "fallback_accept" in result["reason_codes"]


def test_meta_trader_hold_passthrough():
    mt = MetaTrader(deepseek=DeepSeekClient(api_key="REPLACE_ME"))
    decision = _sample_decision(action="HOLD")
    result = mt.validate("INFY", decision, _sample_features(), _sample_snap())
    assert result["verdict"] == "HOLD"
    assert result["size_multiplier"] == 0.0
    assert "hold_passthrough" in result["reason_codes"]


# ── JSON parsing ──────────────────────────────────────────────────────────────
def test_meta_trader_parse_valid_json():
    mt = MetaTrader(deepseek=DeepSeekClient(api_key="REPLACE_ME"))
    raw = json.dumps({
        "verdict": "SCALE_DOWN",
        "size_multiplier": 0.5,
        "risk_flag": "MEDIUM",
        "reason_codes": ["high_atr", "rsi_extended"],
        "final_confidence": 0.6,
        "reasoning": "ATR too wide — reduce size",
    })
    result = mt._parse(raw)
    assert result["verdict"] == "SCALE_DOWN"
    assert result["size_multiplier"] == 0.5
    assert result["risk_flag"] == "MEDIUM"
    assert "high_atr" in result["reason_codes"]


def test_meta_trader_parse_markdown_fenced_json():
    mt = MetaTrader(deepseek=DeepSeekClient(api_key="REPLACE_ME"))
    raw = "```json\n" + json.dumps({
        "verdict": "REJECT",
        "size_multiplier": 0.0,
        "risk_flag": "CRITICAL",
        "reason_codes": ["regime_down"],
        "final_confidence": 0.2,
        "reasoning": "Downtrend — no BUY",
    }) + "\n```"
    result = mt._parse(raw)
    assert result["verdict"] == "REJECT"
    assert result["risk_flag"] == "CRITICAL"


def test_meta_trader_parse_none_returns_default():
    mt = MetaTrader(deepseek=DeepSeekClient(api_key="REPLACE_ME"))
    result = mt._parse(None)
    assert result["verdict"] == "ACCEPT"


def test_meta_trader_parse_invalid_verdict_normalised():
    mt = MetaTrader(deepseek=DeepSeekClient(api_key="REPLACE_ME"))
    raw = json.dumps({
        "verdict": "SKIP",
        "size_multiplier": 0.8,
        "risk_flag": "LOW",
        "reason_codes": [],
        "final_confidence": 0.5,
        "reasoning": "unknown verdict",
    })
    result = mt._parse(raw)
    assert result["verdict"] == "ACCEPT"


# ── prompt builder ─────────────────────────────────────────────────────────────
def test_build_prompt_contains_key_fields():
    prompt = _build_prompt("TCS", _sample_decision(), _sample_features(), _sample_snap())
    assert "TCS" in prompt
    assert "BUY" in prompt
    assert "ADX" in prompt
    assert "Risk/Reward" in prompt
    assert "PORTFOLIO" in prompt
