"""Confirmed sniper board: durable hits + focused evaluation ranking."""
from __future__ import annotations

from pathlib import Path

from product import sniper_board as SB


def test_append_hits_dedupes_per_session(tmp_path: Path):
    path = tmp_path / "board.json"
    first = SB.append_hits(
        {
            "symbol": "KAYNES",
            "trigger": 4250,
            "ltp": 4268.5,
            "held_s": 48,
            "stop": 4100,
            "target": 4550,
            "cum_vol": 1_200_000,
            "avg_vol": 800_000,
            "session_date": "2026-08-03",
            "confirmed_at": "2026-08-03T05:30:00+00:00",
        },
        path=path,
    )
    assert len(first) == 1
    assert first[0]["symbol"] == "KAYNES"
    assert first[0]["vol_pace"] == 1.5

    again = SB.append_hits(
        {
            "symbol": "kaynes",
            "trigger": 4250,
            "ltp": 4270,
            "held_s": 60,
            "session_date": "2026-08-03",
            "confirmed_at": "2026-08-03T06:00:00+00:00",
        },
        path=path,
    )
    assert again == []

    next_day = SB.append_hits(
        {
            "symbol": "KAYNES",
            "trigger": 4300,
            "ltp": 4310,
            "held_s": 50,
            "session_date": "2026-08-04",
            "confirmed_at": "2026-08-04T05:30:00+00:00",
        },
        path=path,
    )
    assert len(next_day) == 1
    board = SB.load_board(path)
    assert len(board["hits"]) == 2


def test_normalize_hit_rejects_incomplete():
    assert SB.normalize_hit({"symbol": "X", "trigger": 10}) is None
    assert SB.normalize_hit({"symbol": "", "trigger": 10, "ltp": 11}) is None
    ok = SB.normalize_hit({"symbol": "abc", "trigger": 10, "ltp": 10.2, "held_s": 45})
    assert ok is not None
    assert ok["symbol"] == "ABC"


def test_evaluate_board_ranks_with_mocked_inputs(tmp_path: Path, monkeypatch):
    path = tmp_path / "board.json"
    SB.append_hits(
        [
            {
                "symbol": "ALPHA",
                "trigger": 100,
                "ltp": 101.5,
                "held_s": 60,
                "cum_vol": 2_000_000,
                "avg_vol": 1_000_000,
                "session_date": "2026-08-03",
                "confirmed_at": "2026-08-03T05:00:00+00:00",
            },
            {
                "symbol": "BETA",
                "trigger": 50,
                "ltp": 50.8,
                "held_s": 45,
                "cum_vol": 900_000,
                "avg_vol": 1_000_000,
                "session_date": "2026-08-03",
                "confirmed_at": "2026-08-03T05:10:00+00:00",
            },
        ],
        path=path,
    )

    monkeypatch.setattr(
        "product.scan_store.load_scan",
        lambda: {
            "schema_version": 1,
            "records": [
                {
                    "symbol": "ALPHA",
                    "company": "Alpha Ltd",
                    "score": 82,
                    "momentum_5d": 4.2,
                    "verdict": "BUY",
                    "status": "Ready to trade",
                    "signals": ["MOMENTUM", "PRE_BREAKOUT"],
                    "chase_risk": False,
                    "edge_r": 0.12,
                    "price": 101.5,
                },
                {
                    "symbol": "BETA",
                    "company": "Beta Ltd",
                    "score": 55,
                    "momentum_5d": 1.1,
                    "verdict": "WATCH",
                    "status": "Watch",
                    "signals": ["PRE_BREAKOUT"],
                    "chase_risk": True,
                    "edge_r": -0.08,
                    "price": 50.8,
                },
            ],
        },
    )

    class FakeReport:
        status = "SUCCEEDED"
        payload = {
            "records": [
                {
                    "symbol": "ALPHA",
                    "classification": "QUALITY_COMPOUNDER",
                    "technical_score": 70,
                    "fundamental_score": 78,
                    "fundamental_coverage": 0.8,
                    "combined_score": 74,
                    "timing": "CONSTRUCTIVE",
                    "sector": "Capital Goods",
                    "quality_factors": ["ROE 18%"],
                    "risk_flags": [],
                    "price": 101.5,
                },
                {
                    "symbol": "BETA",
                    "classification": "NEEDS_FUNDAMENTALS",
                    "technical_score": 50,
                    "fundamental_score": 30,
                    "fundamental_coverage": 0.2,
                    "combined_score": 40,
                    "timing": "EXTENDED",
                    "sector": "Others",
                    "quality_factors": [],
                    "risk_flags": ["High valuation"],
                    "price": 50.8,
                },
            ]
        }
        error_message = ""

    monkeypatch.setattr(
        "scan.long_term_service.run_long_term_scan",
        lambda **_kwargs: FakeReport(),
    )

    evaluation = SB.evaluate_board(path=path, lookback_days=30, save=True)
    assert evaluation["summary"]["unique_symbols"] == 2
    records = evaluation["records"]
    assert records[0]["symbol"] == "ALPHA"
    assert records[0]["verdict"] == "PRIORITY"
    assert "tomorrow_watch" in records[0]["consider_for"]
    beta = next(r for r in records if r["symbol"] == "BETA")
    assert beta["verdict"] == "AVOID"
    assert beta["edge_r"] == -0.08

    reloaded = SB.load_board(path)
    assert reloaded["evaluation"]["summary"]["priority"] == 1


def test_alert_persists_to_board_without_telegram(tmp_path: Path, monkeypatch):
    path = tmp_path / "board.json"
    monkeypatch.setattr(SB, "DEFAULT_BOARD_PATH", path)

    class FakeEngine:
        def is_configured(self):
            return False

    monkeypatch.setattr(
        "alerts.telegram_alerts.AlertEngine",
        lambda: FakeEngine(),
    )
    monkeypatch.setattr("execution.autopilot.on_breakout", lambda _h: None)

    from scan import breakout_sniper as sniper

    sniper._fired.clear()
    sniper._alert(
        [
            {
                "symbol": "CLEAN",
                "trigger": 200.0,
                "ltp": 201.5,
                "held_s": 50,
                "stop": 190.0,
                "target": 220.0,
                "cum_vol": 500_000,
                "avg_vol": 300_000,
            }
        ]
    )
    board = SB.load_board(path)
    assert any(h["symbol"] == "CLEAN" for h in board["hits"])
