from types import SimpleNamespace

from product.workspace import build_command_center_state, scanner_rows


def test_command_center_projects_real_counts_without_inventing_values():
    scan = {
        "universe_size": 2,
        "records": [
            {"symbol": "AAA", "status": "Ready to trade", "signals": ["MOMENTUM"], "score": 81},
            {"symbol": "BBB", "status": "Watch for breakout", "signals": ["PRE_BREAKOUT"], "score": 70},
        ],
    }
    long_term = {
        "summary": {"coverage_pct": 75},
        "records": [
            {"symbol": "CCC", "classification": "QUALITY_COMPOUNDER", "combined_score": 88, "fundamental_coverage": 0.75},
            {"symbol": "DDD", "classification": "NEEDS_FUNDAMENTALS", "combined_score": 90},
        ],
    }
    paper = SimpleNamespace(
        capital=100_000,
        equity=102_000,
        open_positions=({"symbol": "AAA"},),
        open_risk=1_000,
        enabled=True,
    )
    market = SimpleNamespace(
        health="Healthy",
        summary="Healthy market",
        trade_stance="Selective entries allowed",
        breadth="Strong · 70/100",
        leaders=("IT",),
        laggards=("Energy",),
        nifty_change_1d=1.0,
        vix=12.0,
    )

    state = build_command_center_state(
        scan_payload=scan,
        long_term_payload=long_term,
        paper=paper,
        autonomy={"running": True, "state": "OBSERVING", "plain_state": "Observing"},
        market=market,
    )

    assert state["ready_count"] == 1
    assert state["near_breakout_count"] == 1
    assert state["long_term_count"] == 1
    assert state["paper_return_pct"] == 2.0
    assert state["top_setups"][0]["symbol"] == "AAA"
    assert state["top_long_term"][0]["symbol"] == "CCC"


def test_scanner_modes_use_one_source_without_mixing_long_term_into_momentum():
    scan = {
        "records": [
            {"symbol": "MOM", "signals": ["MOMENTUM"], "score": 80, "fno_available": True},
            {"symbol": "PRE", "signals": ["PRE_BREAKOUT"], "score": 70},
            {"symbol": "EXT", "signals": ["MOMENTUM"], "score": 90, "chase_risk": True},
        ]
    }
    long_term = {
        "records": [
            {"symbol": "LT", "classification": "QUALITY_COMPOUNDER", "combined_score": 91},
            {"symbol": "BAD", "classification": "AVOID_REVIEW", "combined_score": 20},
        ]
    }

    assert [row["symbol"] for row in scanner_rows("Momentum", scan_payload=scan, long_term_payload=long_term)] == ["MOM", "EXT"]
    assert [row["symbol"] for row in scanner_rows("Pre-Breakout", scan_payload=scan, long_term_payload=long_term)] == ["PRE"]
    assert [row["symbol"] for row in scanner_rows("Long-Term", scan_payload=scan, long_term_payload=long_term)] == ["LT", "BAD"]
    avoid = scanner_rows("Avoid", scan_payload=scan, long_term_payload=long_term)
    assert {row["symbol"] for row in avoid} == {"EXT", "BAD"}


def test_breakouts_mode_ranks_best_quality_first():
    scan = {
        "records": [
            {
                "symbol": "WEAK", "signals": ["BREAKOUT_52W"], "score": 88,
                "breakout_grade": "", "breakout_conviction": 40, "chase_risk": False,
                "rsi": 50, "volume_ratio": 1.2, "avg_vol20": 1e6,
                "verdict": "BUY", "status": "Ready to trade",
            },
            {
                "symbol": "HOT", "signals": ["BREAKOUT_52W"], "score": 95,
                "breakout_grade": "A", "breakout_conviction": 90, "chase_risk": False,
                "rsi": 82, "volume_ratio": 2.5, "avg_vol20": 1e6,
                "verdict": "BUY", "status": "Ready to trade",
            },
            {
                "symbol": "STRONG", "signals": ["BREAKOUT_52W"], "score": 70,
                "breakout_grade": "A", "breakout_conviction": 85, "chase_risk": False,
                "rsi": 55, "volume_ratio": 2.0, "avg_vol20": 1e6,
                "verdict": "BUY", "status": "Ready to trade",
            },
        ]
    }
    long_term = {
        "records": [
            {
                "symbol": "STRONG", "classification": "QUALITY_COMPOUNDER",
                "combined_score": 90, "fundamental_coverage": 0.85,
                "fundamental_score": 82,
            },
        ]
    }
    rows = scanner_rows("Breakouts", scan_payload=scan, long_term_payload=long_term)
    assert rows[0]["symbol"] == "STRONG"
    assert rows[0]["fundamental_score"] == 82
    # High RSI and weak names sit below the sniper-quality pick
    assert [r["symbol"] for r in rows][0] != "HOT"


def test_conviction_mode_preserves_conviction_ranking():
    rows = [
        {"symbol": "LOW", "conviction_score": 61},
        {"symbol": "HIGH", "conviction_score": 84},
    ]
    result = scanner_rows("Conviction", scan_payload=None, long_term_payload=None, conviction_rows=rows)
    assert [row["symbol"] for row in result] == ["HIGH", "LOW"]
