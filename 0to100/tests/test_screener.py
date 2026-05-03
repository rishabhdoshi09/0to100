"""Tests for screener prompt building, parsing, fallback, and full run loop."""
from unittest.mock import MagicMock

import numpy as np
import pandas as pd

from sq_ai.backend.screener import (
    Screener,
    build_prompt,
    compute_screener_features,
    fallback_rank,
    parse_json,
)
from sq_ai.portfolio.tracker import PortfolioTracker


def _df(trend=0.001, n=120, seed=0):
    rng = np.random.default_rng(seed)
    rets = rng.normal(trend, 0.012, n)
    close = 1000 * np.cumprod(1 + rets)
    return pd.DataFrame({
        "open": close * 1.001,
        "high": close * 1.005,
        "low":  close * 0.995,
        "close": close,
        "volume": rng.uniform(1e5, 5e5, n),
    }, index=pd.date_range("2023-01-01", periods=n, freq="B"))


def test_compute_screener_features_keys():
    f = compute_screener_features(_df())
    for k in ("price", "sma_20", "sma_50", "rsi", "volume_ratio", "atr"):
        assert k in f
    assert 0 <= f["rsi"] <= 100


def test_build_prompt_contains_symbols_and_top_n():
    feats = [{"symbol": "A", "price": 100, "sma_20": 99, "sma_50": 95,
              "rsi": 55, "volume_ratio": 1.2, "atr": 2.0}]
    p = build_prompt(feats, top_n=10)
    assert "A:" in p
    assert "ranked_tickers" in p
    assert "top_10" in p


def test_parse_json_handles_fences():
    assert parse_json("```json\n{\"ranked_tickers\":[\"X\"]}\n```")["ranked_tickers"] == ["X"]
    assert parse_json("nope") is None
    assert parse_json('{"foo": 1}') is None        # missing ranked_tickers


def test_fallback_rank_returns_top_n_with_marker():
    feats = [
        {"symbol": "UP",  "price": 100, "sma_20": 110, "sma_50": 100,
         "rsi": 55, "volume_ratio": 1.5, "atr": 2.0},
        {"symbol": "DN",  "price": 100, "sma_20": 90,  "sma_50": 100,
         "rsi": 80, "volume_ratio": 0.8, "atr": 2.0},
        {"symbol": "MID", "price": 100, "sma_20": 101, "sma_50": 100,
         "rsi": 50, "volume_ratio": 1.0, "atr": 2.0},
    ]
    out = fallback_rank(feats, top_n=2)
    assert out["_fallback"] is True
    assert len(out["ranked_tickers"]) == 2
    assert out["ranked_tickers"][0] == "UP"


def test_screener_run_with_fallback_persists_to_db(tmp_path, monkeypatch):
    db = tmp_path / "t.db"
    tracker = PortfolioTracker(db_path=str(db))
    # DeepSeek client returns nothing → fallback path
    fake_client = MagicMock()
    fake_client.generate.return_value = None
    cfg = {"max_universe_size": 5, "screener_top_n": 3}

    sc = Screener(tracker=tracker, client=fake_client, config=cfg)

    def gather():
        return [
            {"symbol": "A.NS", "price": 100, "sma_20": 105, "sma_50": 100,
             "rsi": 55, "volume_ratio": 1.4, "atr": 2.0},
            {"symbol": "B.NS", "price": 100, "sma_20": 95, "sma_50": 100,
             "rsi": 25, "volume_ratio": 0.7, "atr": 2.0},
            {"symbol": "C.NS", "price": 100, "sma_20": 102, "sma_50": 100,
             "rsi": 50, "volume_ratio": 1.0, "atr": 2.0},
        ]

    out = sc.run(gather_fn=gather)
    assert out["used_deepseek"] is False
    assert len(out["ranked"]) == 3
    persisted = tracker.latest_screener()
    assert len(persisted) == 3
    assert persisted[0]["rank"] == 1


def test_screener_run_uses_deepseek_response_when_valid(tmp_path):
    tracker = PortfolioTracker(db_path=str(tmp_path / "t.db"))
    fake_client = MagicMock()
    fake_client.generate.return_value = (
        '{"ranked_tickers":["B.NS","A.NS"], '
        '"reasons":{"B.NS":"strong","A.NS":"ok"}}'
    )
    cfg = {"max_universe_size": 5, "screener_top_n": 2}
    sc = Screener(tracker=tracker, client=fake_client, config=cfg)

    def gather():
        return [
            {"symbol": "A.NS", "price": 100, "sma_20": 99, "sma_50": 100,
             "rsi": 50, "volume_ratio": 1.0, "atr": 2.0},
            {"symbol": "B.NS", "price": 100, "sma_20": 110, "sma_50": 100,
             "rsi": 60, "volume_ratio": 1.6, "atr": 2.0},
        ]

    out = sc.run(gather_fn=gather)
    assert out["used_deepseek"] is True
    assert out["ranked"][0]["symbol"] == "B.NS"
    assert "strong" in out["ranked"][0]["reasoning"]
