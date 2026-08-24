"""Case memory: ideas become remembered cases; n<30 stays unproven."""
from __future__ import annotations

from product.case_memory import (
    PROVEN_N,
    remember_case,
    setup_memory,
    settle_due_cases,
)


def test_empty_memory_does_not_invent_similar_cases(tmp_path, monkeypatch):
    import product.case_memory as cm

    monkeypatch.setattr(cm, "CASES_DB", tmp_path / "cases.db")
    monkeypatch.setattr(cm, "_live_stats", lambda setup: {"n": 0, "regimes": {}})
    case = remember_case(
        {"symbol": "CARBORUNIV", "category_id": "momentum_breakouts", "setup_label": "Breakout",
         "why_now": ["52-week high breakout"], "what_changes_mind": ["Price closes below stop"]},
        row={"symbol": "CARBORUNIV", "signals": ["BREAKOUT_52W"]},
        persist=False,
    )
    assert case["n_similar"] == 0
    assert case["proven"] is False
    assert case["verdict"] == "unmeasured"
    assert "18" not in case["memory_line"]
    assert "not remembered" in case["memory_line"].lower() or "not remembered similar" in case["memory_line"].lower()
    assert case["places_orders"] is False
    assert case["expectancy_r"] is None


def test_unproven_line_matches_the_breakthrough_example(tmp_path, monkeypatch):
    import product.case_memory as cm

    monkeypatch.setattr(cm, "CASES_DB", tmp_path / "cases.db")
    monkeypatch.setattr(
        cm, "_live_stats",
        lambda setup: {"n": 18, "wins": 10, "win_rate": 55.6, "expectancy_r": 0.4, "regimes": {}},
    )
    monkeypatch.setattr(cm, "_desk_pairs", lambda setup, db_path=None: [])
    case = remember_case(
        {"symbol": "CARBORUNIV", "category_id": "momentum_breakouts", "setup_label": "Ready to trade",
         "why_now": ["52-week high breakout"]},
        row={"symbol": "CARBORUNIV", "signals": ["BREAKOUT_52W"]},
        persist=False,
    )
    assert case["n_similar"] == 18
    assert case["proven"] is False
    assert case["verdict"] == "unproven"
    assert "18 similar" in case["memory_line"]
    assert "not proven" in case["memory_line"].lower()
    assert "made money" not in case["memory_line"].lower()
    assert case["expectancy_r"] is None  # hidden until proven


def test_proven_memory_can_talk_about_regimes_and_costs(tmp_path, monkeypatch):
    import product.case_memory as cm

    monkeypatch.setattr(cm, "CASES_DB", tmp_path / "cases.db")
    monkeypatch.setattr(
        cm, "_live_stats",
        lambda setup: {
            "n": 120, "wins": 70, "win_rate": 58.0, "expectancy_r": 0.22,
            "regimes": {
                "HEALTHY": {"n": 70, "win_rate": 66.0},
                "NARROW": {"n": 50, "win_rate": 38.0},
            },
        },
    )
    monkeypatch.setattr(cm, "_desk_pairs", lambda setup, db_path=None: [])
    case = remember_case(
        {"symbol": "CARBORUNIV", "signals": ["BREAKOUT_52W"]},
        row={"signals": ["BREAKOUT_52W"]},
        persist=False,
    )
    assert case["n_similar"] == 120
    assert case["proven"] is True
    assert "120" in case["memory_line"]
    assert "strong markets" in case["memory_line"]
    assert "weak markets" in case["memory_line"]
    assert "after costs" in case["memory_line"]
    assert case["expectancy_r"] == 0.22


def test_open_and_settle_writes_real_outcomes(tmp_path, monkeypatch):
    import pandas as pd
    import product.case_memory as cm
    from product.case_memory import open_case

    db = tmp_path / "cases.db"
    monkeypatch.setattr(cm, "CASES_DB", db)
    idx = pd.date_range("2026-08-01", periods=20, freq="B")
    df = pd.DataFrame({
        "open": 100.0, "high": 112.0, "low": 99.0, "close": 111.0, "volume": 1e6,
    }, index=idx)
    monkeypatch.setattr("data.bhavcopy_store.get_ohlcv", lambda symbol: df)
    monkeypatch.setattr(cm, "_today", lambda: "2026-08-01")
    case_id = open_case(
        {"symbol": "AAA", "category_id": "momentum_breakouts", "entry": 100, "stop": 95, "target": 110,
         "why_now": ["break"], "what_changes_mind": ["stop"]},
        row={"signals": ["BREAKOUT_52W"], "entry": 100, "stop": 95, "target": 110},
        db_path=db,
    )
    assert case_id
    n = settle_due_cases(db_path=db)
    assert n == 1
    mem = setup_memory("BREAKOUT_52W", db_path=db)
    assert mem["n"] >= 1
    assert mem["desk_n"] == 1


def test_reco_card_carries_a_case(tmp_path, monkeypatch):
    import product.case_memory as cm
    from product.recommendations_workspace import card_from_row

    monkeypatch.setattr(cm, "CASES_DB", tmp_path / "cases.db")
    monkeypatch.setattr(cm, "_live_stats", lambda setup: {"n": 18, "wins": 8, "win_rate": 44.0, "expectancy_r": 0.1, "regimes": {}})
    monkeypatch.setattr(cm, "_desk_pairs", lambda setup, db_path=None: [])
    card = card_from_row(
        {"symbol": "CARBORUNIV", "signals": ["BREAKOUT_52W"], "volume_ratio": 1.6,
         "above_sma50": True, "entry": 100, "stop": 95, "target": 110, "price": 101},
        category_id="momentum_breakouts",
        category_label="Momentum Breakouts",
    )
    assert card["case"]["n_similar"] == 18
    assert card["case"]["verdict"] == "unproven"
    assert "not proven" in card["case"]["memory_line"].lower()
    assert card["case"]["places_orders"] is False


def test_proven_floor_matches_ev_engine():
    from scan.ev_engine import MIN_N
    assert PROVEN_N == MIN_N == 30
