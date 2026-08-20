"""Minervini SEPA trend-template scoring — no invented averages."""
from __future__ import annotations

import pandas as pd

from product.sepa_setup import score_sepa, rank_best_setups
from product.stock_workspace import build_stock_workspace


def _uptrend(periods: int = 280) -> pd.DataFrame:
    index = pd.date_range("2024-01-01", periods=periods, freq="B")
    close = pd.Series([70 + i * 0.55 for i in range(periods)], index=index)
    return pd.DataFrame(
        {
            "open": close - 0.4,
            "high": close + 1.1,
            "low": close - 0.9,
            "close": close,
            "volume": [120000] * periods,
        },
        index=index,
    )


def _downtrend(periods: int = 280) -> pd.DataFrame:
    index = pd.date_range("2024-01-01", periods=periods, freq="B")
    close = pd.Series([220 - i * 0.45 for i in range(periods)], index=index)
    return pd.DataFrame(
        {
            "open": close + 0.3,
            "high": close + 1,
            "low": close - 1,
            "close": close,
            "volume": [110000] * periods,
        },
        index=index,
    )


def test_uptrend_clears_sepa_template():
    sepa = score_sepa(_uptrend())
    assert sepa["available"] is True
    assert sepa["total"] == 7
    assert sepa["score"] >= 80
    assert sepa["passed"] >= 6
    assert sepa["verdict"] == "STRONG"
    by_id = {c["id"]: c for c in sepa["criteria"]}
    assert by_id["near_52w_high"]["passed"] is True
    assert by_id["sma200_rising"]["passed"] is True
    assert sepa["quote"]["close"] > 0
    assert sepa["session"]["label"] in {"MARKET OPEN", "MARKET CLOSED"}
    assert sepa["stage"]["id"] == "stage_2"
    assert sepa["rs"]["label"] in {"LEADER", "IN LINE", "LAGGARD", "UNKNOWN"}


def test_downtrend_is_weak_not_ideal():
    sepa = score_sepa(_downtrend())
    assert sepa["available"] is True
    assert sepa["score"] < 40
    assert sepa["verdict"] == "WEAK"
    assert "NOT IDEAL" in sepa["headline"]


def test_short_history_is_incomplete_not_a_fail():
    sepa = score_sepa(_uptrend(periods=30))
    assert sepa["available"] is False
    assert sepa["verdict"] == "INCOMPLETE"
    assert sepa["score"] == 0


def test_missing_frame_does_not_invent_a_setup():
    sepa = score_sepa(None)
    assert sepa["available"] is False
    assert sepa["score"] == 0
    assert all(c["awarded"] == 0 for c in sepa["criteria"])


def test_rank_best_setups_keeps_only_the_floor():
    frames = {"UP": _uptrend(), "DOWN": _downtrend()}
    ranked, note = rank_best_setups(
        [
            {"symbol": "UP", "score": 90, "verdict": "BUY", "chase_risk": False, "rsi": 55},
            {"symbol": "DOWN", "score": 80, "verdict": "BUY", "chase_risk": False, "rsi": 40},
        ],
        load_frame=lambda symbol: frames[symbol],
        min_score=40,
    )
    assert len(ranked) == 1
    sepa, row = ranked[0]
    assert row["symbol"] == "UP"
    assert sepa["score"] >= 80
    assert "SEPA" in note or "Stage-2" in note


def test_rank_best_setups_respects_time_budget(monkeypatch):
    import time
    from product.sepa_setup import _RANK_CACHE

    monkeypatch.setattr("product.monitor_context.nifty_frame", lambda: None)
    _RANK_CACHE.clear()
    calls: list[str] = []

    def slow(symbol: str):
        calls.append(symbol)
        time.sleep(0.12)
        return _uptrend()

    rows = [
        {"symbol": f"S{i}", "score": 90, "verdict": "BUY", "chase_risk": False, "rsi": 55}
        for i in range(12)
    ]
    t0 = time.monotonic()
    ranked, note = rank_best_setups(
        rows,
        load_frame=slow,
        score_cap=12,
        max_seconds=0.2,
        cache_key="budget-test",
    )
    elapsed = time.monotonic() - t0
    assert elapsed < 1.0
    assert len(calls) < 12
    assert "budget" in note.lower()
    assert "budget-test" not in _RANK_CACHE
    assert ranked is not None


def test_stock_workspace_exposes_sepa_monitor():
    result = build_stock_workspace(
        "TEST",
        scan_payload={"scanned_at": "2026-01-28T10:00:00+00:00", "records": [{"symbol": "TEST"}]},
        long_term_payload={"records": []},
        raw_fundamentals={"available": False, "data": {}, "section_as_of": {}},
        frame=_uptrend(),
        news=[],
        fno_payload={},
    )
    assert result["sepa"]["available"] is True
    assert result["sepa"]["passed"] >= 6
    assert result["technical"]["open"] is not None
    assert result["technical"]["change_pct"] is not None
