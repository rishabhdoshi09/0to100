"""Peer ranker must stay local — Stock Intelligence embeds it on every symbol open."""
from __future__ import annotations

import sys
import types

import pandas as pd

from core import peer_ranker as PR


def _frame(start: float = 100.0, n: int = 80) -> pd.DataFrame:
    index = pd.date_range("2025-01-01", periods=n, freq="B")
    close = pd.Series([start + i * 0.4 for i in range(n)], index=index)
    return pd.DataFrame(
        {
            "open": close - 0.5,
            "high": close + 1.0,
            "low": close - 1.0,
            "close": close,
            "volume": [100_000 + i * 500 for i in range(n)],
        },
        index=index,
    )


def _install_fake_bhav(get_ohlcv):
    """Avoid importing data/__init__.py (heavy broker deps) just to stub get_ohlcv."""
    if "data" not in sys.modules:
        pkg = types.ModuleType("data")
        pkg.__path__ = []  # mark as package
        sys.modules["data"] = pkg
    fake = types.ModuleType("data.bhavcopy_runtime")
    fake.get_ohlcv = get_ohlcv
    sys.modules["data.bhavcopy_runtime"] = fake
    return fake


def test_score_peer_uses_local_bhav_not_network():
    frames = {sym: _frame(100 + i) for i, sym in enumerate(PR._SECTOR_PEERS["ENERGY"])}
    _install_fake_bhav(lambda symbol: frames.get(str(symbol).upper()))

    scored = PR._score_peer("GAIL")
    assert not scored.get("error")
    assert scored["symbol"] == "GAIL"
    assert scored["price"] > 0
    assert scored["score"] >= 0

    ranked = PR.rank_vs_peers("GAIL")
    assert ranked is not None
    assert ranked.sector == "ENERGY"
    assert ranked.total_peers >= 1
    assert any(row["symbol"] == "GAIL" for row in ranked.peers_ranked)
    # Source must be local history — agents.tools network helpers are not imported here.
    assert "agents.tools" not in sys.modules


def test_score_peer_missing_history_is_error():
    _install_fake_bhav(lambda _symbol: None)
    assert PR._score_peer("NOSUCH")["error"] is True
