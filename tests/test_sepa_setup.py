"""Minervini SEPA trend-template scoring — no invented averages, no order path."""
from __future__ import annotations

import pandas as pd
import pytest

from product.sepa_setup import public_best_setups, rank_best_setups, score_sepa
from ui.desk_board import reco_card_html, sepa_card_row, setup_badge


@pytest.fixture(autouse=True)
def _no_index_network(monkeypatch):
    monkeypatch.setattr("product.monitor_context.nifty_frame", lambda: None)


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
        max_seconds=None,
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


def test_sepa_card_says_qualified_not_buy_order():
    sepa = score_sepa(_uptrend())
    card = sepa_card_row(sepa, {"symbol": "TCS", "company": "TCS", "status": "Ready to trade",
                                "price": 100, "entry": 101, "stop": 95, "target": 120,
                                "why": "scan reason"})
    assert setup_badge(card) == ("SEPA qualified", "buy")
    html = reco_card_html(card)
    assert "SEPA qualified" in html
    assert "SEPA" in html and "/100" in html
    assert "not a buy order" in sepa["disclaimer"].lower() or "not a buy" in sepa["disclaimer"].lower()


def test_persisted_best_setups_skip_rescoring(tmp_path, monkeypatch):
    import json
    from product import sepa_setup as SEPA

    path = tmp_path / "best_setups.json"
    monkeypatch.setattr(SEPA, "best_setups_path", lambda: path)
    cards = [{"symbol": "UP", "sepa_score": 82}]
    path.write_text(json.dumps({
        "scanned_at": "2026-08-24T10:00:00+00:00",
        "cards": cards,
        "note": "cached SEPA rank",
    }), encoding="utf-8")
    again, note = SEPA.public_best_setups(
        {
            "scanned_at": "2026-08-24T10:00:00+00:00",
            "records": [{"symbol": "UP", "score": 90, "verdict": "BUY"}],
        },
        load_frame=lambda symbol: (_ for _ in ()).throw(AssertionError("should use persist")),
    )
    assert again == cards
    assert note == "cached SEPA rank"


def test_public_best_setups_returns_cards_not_pairs():
    frames = {"UP": _uptrend(), "DOWN": _downtrend()}
    cards, note = public_best_setups(
        {
            "scanned_at": "2026-08-24T00:00:00+00:00",
            "records": [
                {"symbol": "UP", "score": 90, "verdict": "BUY", "chase_risk": False, "rsi": 55, "price": 210},
                {"symbol": "DOWN", "score": 80, "verdict": "BUY", "chase_risk": False, "rsi": 40},
            ],
        },
        load_frame=lambda symbol: frames[symbol],
        max_seconds=None,
    )
    assert len(cards) == 1
    assert cards[0]["symbol"] == "UP"
    assert cards[0]["sepa_score"] >= 80
    assert cards[0]["sepa_verdict"] == "STRONG"
    assert "Stage-2" in note or "SEPA" in note
