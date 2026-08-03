"""Sniper must arm from product scan_store records, not only legacy PreBreakout."""
from __future__ import annotations

from scan.breakout_sniper import _is_sniper_candidate, build_watch_map, _quality_skip


def test_product_prebreakout_row_is_candidate():
    row = {
        "symbol": "AAA",
        "status": "Watch for breakout",
        "signals": ["PRE_BREAKOUT"],
        "entry": 100.0,
        "price": 98.5,
        "stop": 95.0,
        "target": 110.0,
        "rsi": 55,
        "chase_risk": False,
    }
    assert _is_sniper_candidate(row) is True


def test_product_ready_to_trade_is_candidate():
    row = {
        "symbol": "BBB",
        "status": "Ready to trade",
        "verdict": "BUY",
        "signals": ["MOMENTUM"],
        "entry": 200.0,
        "price": 201.0,
        "rsi": 60,
        "chase_risk": False,
    }
    assert _is_sniper_candidate(row) is True


def test_far_prebreakout_is_not_candidate():
    row = {
        "symbol": "CCC",
        "status": "Watch for breakout",
        "signals": ["PRE_BREAKOUT"],
        "entry": 100.0,
        "price": 90.0,  # 10% away
        "rsi": 50,
        "chase_risk": False,
    }
    assert _is_sniper_candidate(row) is False


def test_build_watch_map_accepts_product_rows(monkeypatch):
    rows = [
        {
            "symbol": "CLEAN",
            "status": "Watch for breakout",
            "signals": ["PRE_BREAKOUT"],
            "entry": 100.0,
            "price": 98.8,
            "stop": 95.0,
            "target": 112.0,
            "avg_vol20": 1_000_000,
            "rsi": 58,
            "chase_risk": False,
        },
        {
            "symbol": "CHASED",
            "status": "Watch for breakout",
            "signals": ["PRE_BREAKOUT"],
            "entry": 50.0,
            "price": 49.5,
            "rsi": 60,
            "chase_risk": True,
        },
    ]

    class FakeIM:
        def tokens_for(self, symbols):
            return {s: 1000 + i for i, s in enumerate(symbols)}

    monkeypatch.setattr("data.instruments.InstrumentManager", FakeIM)
    watch = build_watch_map(rows)
    syms = {v["symbol"] for v in watch.values()}
    assert "CLEAN" in syms
    assert "CHASED" not in syms
    assert _quality_skip(rows[1])


def test_stack_starts_sniper_runtime():
    from pathlib import Path

    text = Path("scripts/run_quantterm.sh").read_text(encoding="utf-8")
    stop = Path("scripts/stop_quantterm.sh").read_text(encoding="utf-8")
    assert "scan.sniper_runtime" in text
    assert "SNIPER_PID" in text
    assert "scan.sniper_runtime" in stop
