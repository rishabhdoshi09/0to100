"""Stage + RS vs Nifty from official frames — no invented series."""
from __future__ import annotations

import pandas as pd

from product.monitor_context import (
    attach_context,
    classify_stage,
    index_strip,
    rs_rank,
    rs_vs_benchmark,
)
from product.sepa_setup import sepa_card_fields, score_sepa
from product.recommendations_workspace import build_recommendations_workspace


def _frame(closes, start="2024-01-01") -> pd.DataFrame:
    index = pd.date_range(start, periods=len(closes), freq="B")
    close = pd.Series([float(x) for x in closes], index=index)
    return pd.DataFrame(
        {
            "open": close - 0.4,
            "high": close + 1.0,
            "low": close - 0.9,
            "close": close,
            "volume": [100000] * len(closes),
        },
        index=index,
    )


def _trend(periods: int = 280, start: float = 70.0, step: float = 0.55) -> pd.DataFrame:
    closes = [start + i * step for i in range(periods)]
    return _frame(closes)


def test_stage_2_needs_stacked_averages_and_rising_200():
    stage = classify_stage(
        {"price": 120, "sma50": 110, "sma200": 100},
        sma200_rising=True,
    )
    assert stage["id"] == "stage_2"
    assert stage["label"] == "STAGE 2"


def test_stage_4_is_a_declining_stack():
    stage = classify_stage(
        {"price": 80, "sma50": 90, "sma200": 100},
        sma200_rising=False,
    )
    assert stage["id"] == "stage_4"
    assert "STAGE 4" in stage["label"]


def test_missing_sma_does_not_invent_a_stage():
    stage = classify_stage({"price": 100}, sma200_rising=True)
    assert stage["id"] == "unknown"
    assert stage["label"] == "STAGE ?"


def test_rs_leader_and_laggard_from_excess_return():
    stock = _trend(periods=80, start=100, step=1.0)
    nifty = _trend(periods=80, start=20000, step=2.0)
    rs = rs_vs_benchmark(stock, nifty, lookback=63)
    assert rs["available"] is True
    assert rs["excess_pp"] > 5
    assert rs["label"] == "LEADER"

    weak = _trend(periods=80, start=200, step=-0.8)
    lag = rs_vs_benchmark(weak, nifty, lookback=63)
    assert lag["label"] == "LAGGARD"
    assert lag["excess_pp"] < -5


def test_rs_missing_history_stays_unknown():
    rs = rs_vs_benchmark(_trend(periods=10), _trend(periods=80))
    assert rs["available"] is False
    assert rs["label"] == "UNKNOWN"
    assert rs["excess_pp"] is None


def test_index_strip_uses_official_store_only(monkeypatch):
    nifty = _trend(periods=5, start=22000, step=10)
    vix = _trend(periods=5, start=12, step=0.1)

    def fake_get(ticker: str):
        if ticker == "^NSEI":
            return nifty
        if ticker == "^INDIAVIX":
            return vix
        return None

    monkeypatch.setattr("data.index_store.get_index_ohlcv", fake_get)
    rows = index_strip()
    by_id = {row["id"]: row for row in rows}
    assert by_id["^NSEI"]["available"] is True
    assert by_id["^NSEI"]["close"] == round(float(nifty["close"].iloc[-1]), 2)
    assert by_id["^NSEBANK"]["available"] is False
    assert by_id["^INDIAVIX"]["available"] is True


def test_score_sepa_attaches_stage_and_rs():
    sepa = score_sepa(_trend(), bench_frame=_trend(start=20000, step=4.0))
    assert sepa["stage"]["id"] == "stage_2"
    assert sepa["rs"]["available"] is True
    assert sepa["rs"]["label"] == "LEADER"
    fields = sepa_card_fields(sepa)
    assert fields["stage_label"] == "STAGE 2"
    assert fields["rs_label"] == "LEADER"
    assert rs_rank(sepa) == 2


def test_attach_context_mutates_payload():
    payload = {
        "available": True,
        "levels": {"price": 120, "sma50": 110, "sma200": 100},
        "criteria": [{"id": "sma200_rising", "passed": True}],
    }
    attach_context(payload, _trend(), _trend(start=20000, step=4.0))
    assert payload["stage"]["label"] == "STAGE 2"
    assert payload["rs"]["available"] is True


def test_workspace_exposes_indices_and_rs_on_best_setups(monkeypatch):
    stock = _trend()
    nifty = _trend(start=20000, step=4.0)
    monkeypatch.setattr("product.monitor_context.nifty_frame", lambda: nifty)
    monkeypatch.setattr(
        "product.monitor_context.index_strip",
        lambda: [{
            "id": "^NSEI", "label": "NIFTY 50", "close": 22100.0,
            "change_pct": 0.41, "available": True, "source": "nse_index_store",
        }],
    )
    payload = build_recommendations_workspace(
        scan_payload={
            "scanned_at": "2026-08-19T12:30:00+00:00",
            "records": [{
                "symbol": "LEADER", "score": 88, "signals": ["MOMENTUM"],
                "verdict": "BUY", "status": "Ready to trade", "chase_risk": False,
                "price": float(stock["close"].iloc[-1]), "rsi": 55, "volume_ratio": 1.4,
                "avg_vol20": 1e6,
            }],
        },
        long_term_payload={"records": []},
        refresh_technicals=False,
        compute_sepa=True,
        sepa_load_frame=lambda symbol: stock,
    )
    best = next(c for c in payload["categories"] if c["id"] == "best_setups")
    card = best["cards"][0]
    assert card["stage_label"] == "STAGE 2"
    assert card["rs_label"] == "LEADER"
    assert payload["indices"][0]["label"] == "NIFTY 50"
    assert "Nifty" in payload["cmp_note"] or "Nifty" in payload["index_strip_note"]
