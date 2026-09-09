"""Relative strength must not describe a missing Nifty benchmark as underperformance."""
from __future__ import annotations

import numpy as np
import pandas as pd

from scan.quality_engine import QualityEngine


class _Candidate:
    symbol = "TCS"
    price = 108.0
    pivot_level = 109.0
    archetype = "VCP_BREAKOUT"
    stop_level = 100.0


def _close(start: float = 100.0, end: float = 110.0, n: int = 30) -> np.ndarray:
    return np.linspace(start, end, n)


def _frame() -> pd.DataFrame:
    close = _close(100.0, 108.0, 50)
    return pd.DataFrame({
        "open": close,
        "high": close + 1.0,
        "low": close - 1.0,
        "close": close,
        "volume": np.full(50, 5_000_000.0),
    })


def test_rs_positive_when_stock_beats_official_nifty(monkeypatch):
    stock = _close(100.0, 110.0)  # +10%
    nifty = [float(x) for x in _close(100.0, 102.0)]  # +2%

    monkeypatch.setattr("data.index_store.recent_index_closes", lambda *_a, **_k: nifty)
    engine = QualityEngine()
    rs = engine._rs_vs_nifty("TCS", stock)
    assert rs is not None
    assert rs > 0
    monkeypatch.setattr(engine, "_rs_vs_nifty", lambda *_a, **_k: rs)
    monkeypatch.setattr("scan.relative_strength.compute_rs_score", lambda *_a, **_k: 0.0)
    scored = engine.score(_Candidate(), df=_frame())
    assert scored.factors["relative_strength"] > 0
    assert any("vs Nifty" in item for item in scored.evidence)
    assert not any("Underperforming Nifty" in item for item in scored.disqualifiers)
    assert not any("unavailable" in item.lower() for item in scored.disqualifiers)


def test_rs_negative_when_stock_lags_official_nifty(monkeypatch):
    stock = _close(100.0, 101.0)  # +1%
    nifty = [float(x) for x in _close(100.0, 110.0)]  # +10%

    monkeypatch.setattr("data.index_store.recent_index_closes", lambda *_a, **_k: nifty)
    engine = QualityEngine()
    rs = engine._rs_vs_nifty("TCS", stock)
    assert rs is not None
    assert rs < 0
    monkeypatch.setattr(engine, "_rs_vs_nifty", lambda *_a, **_k: rs)
    monkeypatch.setattr("scan.relative_strength.compute_rs_score", lambda *_a, **_k: 0.0)
    scored = engine.score(_Candidate(), df=_frame())
    assert scored.factors["relative_strength"] == 0.0
    assert any("Underperforming Nifty" in item for item in scored.disqualifiers)
    assert not any("unavailable" in item.lower() for item in scored.disqualifiers)


def test_rs_unavailable_does_not_claim_underperformance(monkeypatch):
    monkeypatch.setattr("data.index_store.recent_index_closes", lambda *_a, **_k: [])

    class _Boom:
        def history(self, **_k):
            raise RuntimeError("yahoo blocked")

    monkeypatch.setattr("yfinance.Ticker", lambda *_a, **_k: _Boom())
    monkeypatch.setattr("scan.relative_strength.compute_rs_score", lambda *_a, **_k: 0.0)
    engine = QualityEngine()
    close = _close()
    assert engine._rs_vs_nifty("HDFCBANK", close) is None
    scored = engine.score(_Candidate(), df=_frame())
    assert scored.factors["relative_strength"] == 0.0
    assert any("unavailable" in item.lower() and "nifty" in item.lower() for item in scored.disqualifiers)
    assert not any("Underperforming Nifty" in item for item in scored.disqualifiers)
