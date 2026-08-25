"""Tests for market radar lane projections."""
from __future__ import annotations

from product.radar_workspace import (
    classify_breakout_state,
    classify_momentum_state,
    build_radar_home,
    enrich_scan_row,
)


def test_confirmed_breakout_requires_buy_and_ready_status():
    row = {
        "symbol": "TEST",
        "signals": ["BREAKOUT_52W", "MOMENTUM"],
        "verdict": "BUY",
        "status": "Ready to trade",
        "chase_risk": False,
        "score": 80,
        "volume_ratio": 1.5,
    }
    assert classify_breakout_state(row) == "confirmed_breakout"


def test_extended_breakout_when_chase_risk():
    row = {
        "symbol": "TEST",
        "signals": ["BREAKOUT_52W"],
        "verdict": "BUY",
        "status": "Wait for pullback",
        "chase_risk": True,
    }
    assert classify_breakout_state(row) == "extended_after_breakout"


def test_momentum_extended_state():
    row = {"signals": ["MOMENTUM"], "chase_risk": True, "score": 70, "status": "Watch"}
    assert classify_momentum_state(row) == "strong_but_extended"


def test_radar_home_rejects_quality_without_fundamental_coverage():
    long_term = {
        "scanned_at": "2026-08-01T00:00:00+00:00",
        "records": [
            {"symbol": "THIN", "classification": "QUALITY_COMPOUNDER", "combined_score": 88, "fundamental_coverage": 0.2},
        ],
    }
    market = {"health": "Healthy", "breadth": "60% adv", "trade_stance": "Open", "leaders": [], "laggards": []}
    payload = build_radar_home(scan_payload={"records": []}, long_term_payload=long_term, market=market)
    assert payload["counts"]["long_term_picks"] == 0


def test_radar_home_builds_three_lanes():
    scan = {
        "scanned_at": "2026-08-01T00:00:00+00:00",
        "universe_size": 100,
        "records": [
            {
                "symbol": "AAA",
                "score": 90,
                "verdict": "BUY",
                "status": "Ready to trade",
                "signals": ["MOMENTUM", "BREAKOUT_52W"],
                "chase_risk": False,
                "volume_ratio": 1.2,
                "rsi": 58,
                "reasons": ["Strong volume"],
                "breakout_grade": "B",
                "breakout_conviction": 55,
                "avg_vol20": 1_000_000,
            },
            {
                "symbol": "BEST",
                "score": 78,
                "verdict": "BUY",
                "status": "Ready to trade",
                "signals": ["BREAKOUT_52W"],
                "chase_risk": False,
                "volume_ratio": 2.0,
                "rsi": 55,
                "reasons": ["A-grade break"],
                "breakout_grade": "A",
                "breakout_conviction": 82,
                "avg_vol20": 2_000_000,
            },
            {
                "symbol": "HOT",
                "score": 95,
                "verdict": "BUY",
                "status": "Ready to trade",
                "signals": ["BREAKOUT_52W"],
                "chase_risk": False,
                "volume_ratio": 3.0,
                "rsi": 82,
                "breakout_grade": "A",
                "breakout_conviction": 90,
                "avg_vol20": 2_000_000,
            },
            {
                "symbol": "THIN",
                "score": 88,
                "verdict": "BUY",
                "status": "Ready to trade",
                "signals": ["BREAKOUT_52W"],
                "chase_risk": False,
                "volume_ratio": 0.6,
                "rsi": 50,
                "breakout_grade": "A",
                "breakout_conviction": 80,
                "avg_vol20": 500_000,
            },
            {
                "symbol": "BBB",
                "score": 50,
                "signals": ["MOMENTUM"],
                "chase_risk": True,
                "status": "Wait for pullback",
            },
        ],
    }
    long_term = {
        "scanned_at": "2026-08-01T00:00:00+00:00",
        "records": [
            {"symbol": "QUAL", "classification": "QUALITY_COMPOUNDER", "combined_score": 88, "fundamental_coverage": 0.8},
            {
                "symbol": "BEST",
                "classification": "QUALITY_COMPOUNDER",
                "combined_score": 86,
                "fundamental_coverage": 0.9,
                "fundamental_score": 80,
            },
        ],
    }
    market = {"health": "Healthy", "breadth": "60% adv", "trade_stance": "Open", "leaders": [], "laggards": []}
    payload = build_radar_home(scan_payload=scan, long_term_payload=long_term, market=market)
    assert payload["counts"]["momentum"] >= 1
    assert payload["counts"]["breakouts"] >= 1
    assert payload["counts"]["long_term_picks"] == 2
    assert payload["lanes"]["momentum"][0]["symbol"] == "AAA"
    # Best = sniper pool, RSI≤70 ignored, volume ≥1× preferred
    assert payload["best_breakout"]["symbol"] == "BEST"
    assert payload["best_breakout"]["symbol"] != "HOT"  # RSI 82 ignored
    assert payload["lanes"]["breakouts"][0]["symbol"] == "BEST"
    assert payload["lanes"]["breakouts"][0]["fundamental_score"] == 80
    # Thin volume (<1×) is not a confirmed breakout lane row; HOT (RSI blow-off)
    # is never the best sniper pick even if it appears in the wider lane.
    syms = [r["symbol"] for r in payload["lanes"]["breakouts"]]
    assert "BEST" in syms
    if "HOT" in syms:
        assert syms.index("BEST") < syms.index("HOT")
    assert payload["best_breakout"]["volume_ratio"] >= 1.0
    assert payload["best_breakout"]["rsi"] <= 70
    assert payload["counts"]["sniper_breakouts"] >= 1
    assert any(r["symbol"] == "BEST" for r in payload["sniper_candidates"])
    assert not any(r["symbol"] == "THIN" for r in payload["sniper_candidates"])
    assert payload["lanes"]["breakouts"][0].get("sniper_candidate") is True
    # Thin volume → breakout_without_volume (excluded from the breakouts lane)
    from product.radar_workspace import classify_breakout_state, is_sniper_breakout_candidate
    thin_raw = next(r for r in scan["records"] if r["symbol"] == "THIN")
    assert classify_breakout_state(thin_raw) == "breakout_without_volume"
    assert is_sniper_breakout_candidate(thin_raw) is False
    assert "THIN" not in [r["symbol"] for r in payload["lanes"]["breakouts"]]
    assert payload["best_among_fundamentals"]["symbol"] == "BEST"
    assert payload["best_among_fundamentals"]["funds_used"] is True
    assert payload["best_of_best"][0]["symbol"] == "BEST"
    assert payload["best_of_best"][0]["rank"] == 1
    assert "SEPA" in payload["ranking_legend"]["best_setups"]
    assert payload["sepa_rank_used"] is False
    assert "same saved" in payload["scan_shared_note"].lower() or "same" in payload["scan_shared_note"].lower()


def test_breakout_quality_prefers_grade_and_fundamentals():
    from product.radar_workspace import breakout_quality_score
    weak = {"score": 90, "breakout_grade": "", "breakout_conviction": 40, "edge_r": 0, "rsi": 50}
    strong = {
        "score": 75, "breakout_grade": "A", "breakout_conviction": 80, "edge_r": 0.2,
        "fundamental_score": 78, "fundamental_coverage": 0.8,
        "classification": "QUALITY_COMPOUNDER",
        "rsi": 55, "volume_ratio": 2.0, "avg_vol20": 1e6,
        "signals": ["BREAKOUT_52W"], "verdict": "BUY", "status": "Ready to trade",
    }
    assert breakout_quality_score(strong) > breakout_quality_score(weak)


def test_high_rsi_and_thin_volume_hard_rejected_from_best():
    from product.radar_workspace import (
        breakout_quality_score,
        is_sniper_breakout_candidate,
        pick_best_among_fundamentals,
        pick_best_sniper_breakout,
    )
    base = {
        "verdict": "BUY", "status": "Ready to trade", "signals": ["BREAKOUT_52W"],
        "breakout_grade": "A", "breakout_conviction": 80, "score": 80,
        "avg_vol20": 1e6, "chase_risk": False,
    }
    hot = {**base, "symbol": "HOT", "rsi": 85, "volume_ratio": 2.5}
    thin = {**base, "symbol": "THIN", "rsi": 55, "volume_ratio": 0.5}
    avoid = {
        **base, "symbol": "AVOID", "rsi": 55, "volume_ratio": 2.0,
        "classification": "AVOID_REVIEW", "fundamental_coverage": 0.9,
        "fundamental_score": 20,
    }
    elevated = {**base, "symbol": "WARM", "rsi": 76, "volume_ratio": 1.8}
    solid = {
        **base, "symbol": "SOLID", "rsi": 55, "volume_ratio": 1.8,
        "classification": "QUALITY_COMPOUNDER", "fundamental_coverage": 0.85,
        "fundamental_score": 80,
    }
    assert not is_sniper_breakout_candidate(hot)   # >82 hard
    assert not is_sniper_breakout_candidate(thin)
    assert is_sniper_breakout_candidate(avoid)     # fund no longer blocks sniper
    assert is_sniper_breakout_candidate(elevated)  # 70–82 ok for technical
    assert is_sniper_breakout_candidate(solid)
    assert breakout_quality_score(thin) == -1000.0
    best = pick_best_sniper_breakout([hot, thin, avoid, elevated, solid])
    assert best is not None
    assert best["symbol"] in {"SOLID", "AVOID", "WARM"}
    assert best["quality_ok"] is True
    assert best["quality_gates"]["volume"] == "pass"
    assert "breakout_context" not in best

    fund_best = pick_best_among_fundamentals([hot, thin, avoid, elevated, solid])
    assert fund_best is not None and fund_best["symbol"] == "SOLID"
    assert fund_best.get("best_among_kind") == "second_screen"
    assert fund_best.get("funds_used") is True
    assert fund_best.get("sepa_used") is False


def test_enrich_scan_row_never_fakes_daily_change_label():
    row = enrich_scan_row({"symbol": "X", "momentum_5d": 4.2, "signals": ["MOMENTUM"]}, scanned_at="2026-01-01")
    assert row["change_5d_pct"] == 4.2
    assert "sector" in row
    assert "sniper_candidate" in row
    assert "breakout_quality" in row


def test_enrich_marks_graded_breakout_as_sniper_candidate():
    row = enrich_scan_row(
        {
            "symbol": "ABC",
            "signals": ["BREAKOUT_52W"],
            "verdict": "BUY",
            "status": "Ready to trade",
            "breakout_grade": "A",
            "breakout_conviction": 80,
            "rsi": 55,
            "volume_ratio": 1.5,
            "avg_vol20": 1e6,
            "chase_risk": False,
        },
        scanned_at="2026-01-01",
    )
    assert row["sniper_candidate"] is True
    assert row["breakout_quality"] > 0


def test_best_among_ranks_sepa_overlay_without_inventing_funds():
    from product.radar_workspace import build_radar_home, pick_best_among_fundamentals

    sniper = {
        "symbol": "SEPA1",
        "score": 70,
        "verdict": "BUY",
        "status": "Ready to trade",
        "signals": ["BREAKOUT_52W"],
        "chase_risk": False,
        "volume_ratio": 1.8,
        "rsi": 55,
        "breakout_grade": "A",
        "breakout_conviction": 80,
        "avg_vol20": 1_000_000,
    }
    tape_only = {**sniper, "symbol": "TAPE", "score": 90, "breakout_conviction": 90}
    both = {
        **sniper,
        "symbol": "BOTH",
        "score": 60,
        "classification": "QUALITY_COMPOUNDER",
        "fundamental_coverage": 0.9,
        "fundamental_score": 70,
        "sepa_score": 85,
    }
    sepa_only = pick_best_among_fundamentals([
        {**sniper, "sepa_score": 80},
        tape_only,
    ])
    assert sepa_only is not None and sepa_only["symbol"] == "SEPA1"
    assert sepa_only["sepa_used"] is True
    assert sepa_only["funds_used"] is False
    assert "SEPA 80/100" in sepa_only["best_among_why"]

    dual = pick_best_among_fundamentals([{**sniper, "sepa_score": 80}, both, tape_only])
    assert dual is not None and dual["symbol"] == "BOTH"
    assert dual["second_screens"] == 2

    market = {"health": "Healthy", "breadth": "ok", "trade_stance": "Open", "leaders": [], "laggards": []}
    payload = build_radar_home(
        scan_payload={"scanned_at": "2026-08-01T00:00:00+00:00", "universe_size": 3, "records": [sniper, tape_only]},
        long_term_payload={"records": []},
        market=market,
        sepa_cards=[{"symbol": "SEPA1", "sepa_score": 80, "sepa_passed": 6, "sepa_total": 7, "sepa_verdict": "Stage 2"}],
    )
    assert payload["best_among_fundamentals"]["symbol"] == "SEPA1"
    assert payload["sepa_rank_used"] is True
    assert payload["second_screen_counts"]["sepa_overlay"] == 1
    assert payload["best_breakout"]["symbol"] in {"SEPA1", "TAPE"}
    assert [r["symbol"] for r in payload["best_of_best"]] == ["SEPA1"]
    assert payload["best_of_best"][0]["rank"] == 1
    assert payload["best_of_best"][0]["action_badge"] == "Buy"
    assert payload["best_of_best"][0]["best_of_best_parts"]["funds"] == 0
    assert payload["best_of_best"][0]["best_of_best_parts"]["sepa"] == 80


def test_best_of_best_composite_uses_existing_screens_only():
    from product.radar_workspace import BEST_OF_BEST_WEIGHTS, best_of_best_parts, rank_best_of_best

    base = {
        "verdict": "BUY", "status": "Ready to trade", "signals": ["BREAKOUT_52W"],
        "breakout_grade": "A", "breakout_conviction": 80, "score": 70,
        "avg_vol20": 1e6, "chase_risk": False, "volume_ratio": 2.0, "rsi": 55,
        "price": 100, "entry": 95, "target": 120, "stop": 90, "momentum_5d": 3.2,
    }
    dual = {
        **base, "symbol": "DUAL", "sepa_score": 80,
        "classification": "QUALITY_COMPOUNDER", "fundamental_coverage": 0.9, "fundamental_score": 70,
    }
    sepa = {**base, "symbol": "SEPA", "sepa_score": 100, "score": 90}
    fund = {
        **base, "symbol": "FUND", "score": 88,
        "classification": "QUALITY_COMPOUNDER", "fundamental_coverage": 0.9, "fundamental_score": 90,
    }
    tape = {**base, "symbol": "TAPE", "score": 99, "breakout_conviction": 99}
    ranked = rank_best_of_best([dual, sepa, fund, tape], limit=8)
    assert [r["symbol"] for r in ranked] == ["DUAL", "SEPA", "FUND"]
    assert ranked[0]["rank"] == 1 and ranked[0]["second_screens"] == 2
    parts = ranked[0]["best_of_best_parts"]
    assert parts["weights"] == BEST_OF_BEST_WEIGHTS
    expected = round(0.45 * 80 + 0.30 * 70 + 0.25 * parts["tape"], 2)
    assert parts["composite"] == expected
    sepa_parts = best_of_best_parts(sepa)
    assert sepa_parts["funds"] == 0
    assert sepa_parts["sepa"] == 100
    assert ranked[1]["upside_to_target_pct"] == 20.0
    assert ranked[1]["change_5d_pct"] == 3.2
    assert ranked[1]["risk_tier"] in {"Low", "Medium", "High"}

