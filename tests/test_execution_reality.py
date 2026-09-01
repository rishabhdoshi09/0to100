"""Phase 1 — ExecutionRealityEngine (shadow / analytics-only)."""

from __future__ import annotations

from product.execution_reality import ExecutionRealityEngine, annotate_shadow


def test_perfect_fill_preserves_gross():
    eng = ExecutionRealityEngine()
    r = eng.analyze_round_trip(
        qty=100,
        entry_price=100.0,
        exit_price=110.0,
        bid=100.0,
        ask=100.0,
        slippage_bps=0.0,
        bar_volume=1_000_000,
    )
    assert r.shadow_mode is True
    assert r.affects_paper_orders is False
    assert r.gross_result["pnl"] == 1000.0
    assert r.fill.fill_status in ("PERFECT", "SPREAD", "SLIPPAGE")
    assert r.fill.filled_qty == 100
    # Gross is never overwritten by charges
    assert r.gross_result["entry_price"] == 100.0
    assert r.gross_result["exit_price"] == 110.0
    assert r.execution_adjusted_result["pnl"] < r.gross_result["pnl"]


def test_spread_cost_worsens_entry_for_buy():
    eng = ExecutionRealityEngine()
    r = eng.analyze_round_trip(
        qty=10,
        entry_price=100.0,
        exit_price=100.0,
        bid=99.0,
        ask=101.0,
        slippage_bps=0.0,
        bar_volume=1_000_000,
    )
    spread = next(f for f in r.fill.fields if f.name == "bid_ask_spread")
    assert spread.measured is True
    assert spread.value == 1.0  # half-spread vs mid 100
    assert r.fill.fill_price == 101.0
    assert r.gross_result["entry_price"] == 100.0  # gross preserved
    assert r.execution_adjusted_result["entry_price"] == 101.0


def test_slippage_applied_on_entry():
    eng = ExecutionRealityEngine()
    r = eng.analyze_round_trip(
        qty=10,
        entry_price=100.0,
        exit_price=110.0,
        bid=100.0,
        ask=100.0,
        slippage_bps=50.0,  # 0.50%
        bar_volume=1_000_000,
    )
    slip = next(f for f in r.fill.fields if f.name == "slippage")
    assert slip.measured is True
    assert abs(slip.value - 0.50) < 1e-9
    assert r.fill.fill_price == 100.50
    assert r.gross_result["pnl"] == 100.0  # 10 * 10


def test_gap_through_stop_exits_at_open_not_stop():
    eng = ExecutionRealityEngine()
    r = eng.analyze_round_trip(
        qty=10,
        entry_price=100.0,
        exit_price=95.0,  # intended stop
        stop_price=95.0,
        open_price=90.0,  # gapped through
        bid=100.0,
        ask=100.0,
        slippage_bps=0.0,
        bar_volume=1_000_000,
    )
    assert r.fill.fill_status == "GAP_THROUGH_STOP"
    gap = next(f for f in r.fill.fields if f.name == "gap_through_stop")
    assert gap.value == 90.0
    assert r.gross_result["exit_price"] == 95.0  # intended preserved
    assert r.execution_adjusted_result["exit_price"] == 90.0
    assert r.execution_adjusted_result["pnl"] < r.gross_result["pnl"]


def test_partial_fill_from_volume_participation():
    eng = ExecutionRealityEngine()
    r = eng.analyze_round_trip(
        qty=1000,
        entry_price=50.0,
        exit_price=55.0,
        bid=50.0,
        ask=50.0,
        slippage_bps=0.0,
        bar_volume=2000,  # 10% cap → 200 shares
        participation_cap=0.10,
    )
    assert r.fill.fill_status == "PARTIAL"
    assert r.fill.filled_qty == 200.0
    assert r.gross_result["qty"] == 1000.0
    assert r.execution_adjusted_result["qty"] == 200.0


def test_no_fill():
    eng = ExecutionRealityEngine()
    r = eng.analyze_round_trip(
        qty=100,
        entry_price=10.0,
        exit_price=12.0,
        no_fill=True,
    )
    assert r.fill.fill_status == "NO_FILL"
    assert r.fill.filled_qty == 0.0
    assert r.fill.fill_price is None
    assert r.execution_adjusted_result["pnl"] == 0.0
    assert r.gross_result["pnl"] == 200.0  # gross still computed on intended


def test_illiquid_candidate_no_fill():
    eng = ExecutionRealityEngine()
    r = eng.analyze_round_trip(
        qty=100,
        entry_price=10.0,
        exit_price=12.0,
        bid=10.0,
        ask=10.0,
        bar_volume=0,
    )
    assert r.fill.fill_status == "NO_FILL"
    assert r.fill.reason == "ILLIQUID"
    assert r.execution_adjusted_result["pnl"] == 0.0


def test_costs_reduce_a_profitable_gross_trade():
    eng = ExecutionRealityEngine()
    r = eng.analyze_round_trip(
        qty=100,
        entry_price=100.0,
        exit_price=101.0,  # +₹100 gross
        bid=100.0,
        ask=100.0,
        slippage_bps=0.0,
        bar_volume=1_000_000,
    )
    assert r.gross_result["pnl"] == 100.0
    assert r.execution_adjusted_result["charges_total"] > 0
    assert r.execution_adjusted_result["pnl"] < r.gross_result["pnl"]
    names = {c.name for c in r.charges}
    assert names >= {"brokerage", "stt", "exchange", "sebi", "stamp_duty", "gst", "dp_charges"}
    for c in r.charges:
        assert c.source
        assert c.formula
        assert c.assumptions
        assert c.timestamp
        assert c.confidence
        assert c.estimated is (not c.measured)


def test_gross_result_remains_preserved_when_annotating_trade():
    eng = ExecutionRealityEngine()
    r = eng.analyze_round_trip(
        qty=50,
        entry_price=200.0,
        exit_price=220.0,
        bid=199.0,
        ask=201.0,
        slippage_bps=10.0,
        bar_volume=10_000,
    )
    trade = {"symbol": "AAA", "qty": 50, "entry": 200.0, "gross_pnl": 1000.0}
    annotated = annotate_shadow(trade, r)
    assert trade["gross_pnl"] == 1000.0  # original untouched
    assert annotated["gross_pnl_preserved"] == r.gross_result["pnl"]
    assert annotated["execution_reality"]["affects_paper_orders"] is False
    assert annotated["qty"] == 50  # fill qty not written onto the trade


def test_missing_microstructure_not_invented_as_zero_evidence():
    eng = ExecutionRealityEngine()
    r = eng.analyze_round_trip(qty=10, entry_price=50.0, exit_price=51.0)
    vol = next(f for f in r.fill.fields if f.name == "volume_participation")
    assert vol.value is None
    assert vol.confidence == "none"
    slip = next(f for f in r.fill.fields if f.name == "slippage")
    assert slip.value is None


def test_circuit_blocks_fill():
    eng = ExecutionRealityEngine()
    r = eng.analyze_round_trip(
        qty=10, entry_price=100.0, exit_price=110.0, circuit_hit=True
    )
    assert r.fill.fill_status == "CIRCUIT"
    assert r.fill.filled_qty == 0.0


def test_paper_cycle_fill_unchanged_by_shadow_engine():
    """Phase 1 must not reprice paper order creation."""
    from datetime import datetime, timezone

    from product.paper_autopilot import run_reco_paper_cycle
    from research.auto_research.paper_book import PaperBook

    card = {
        "symbol": "TCS",
        "reco_tier": "high_conviction",
        "entry_state": "ready",
        "entry": 100.0,
        "stop": 94.0,
        "target": 115.0,
        "cmp": 100.0,
        "chase_risk": False,
        "volume_ratio": 1.4,
        "sector": "Technology",
        "score": 82,
        "setup_label": "VCP",
        "allows_recommend": True,
        "methods": [
            {"id": "tape", "status": "pass", "points": 90},
            {"id": "funds", "status": "pass", "points": 80},
        ],
    }
    now = datetime(2026, 9, 1, 10, 0, tzinfo=timezone.utc)
    book = PaperBook(capital=100_000)
    out = run_reco_paper_cycle(
        book=book,
        cards=[card],
        now=now,
        as_of="2026-09-01",
        workspace={
            "schema_version": 4,
            "generated_at": now.isoformat(),
            "scan_scanned_at": now.isoformat(),
            "categories": [{"id": "wealth_builders", "count": 1, "cards": [card]}],
        },
    )
    pos = next(iter(book.open.values()))
    assert pos.entry_price == 100.0
    assert pos.qty > 0
    assert out["taken"][0]["entry_fill"] == 100.0
    assert out["taken"][0]["qty"] == pos.qty
    assert out["execution_reality"]["affects_paper_orders"] is False
    shadow = out["taken"][0]["execution_reality_shadow"]
    assert shadow["affects_paper_orders"] is False
    assert shadow["gross_result"]["entry_price"] == 100.0
    assert shadow["gross_result"]["exit_price"] == 115.0
    # Adjusted may include charges; gross stays the intended round-trip.
    assert shadow["execution_adjusted_result"]["pnl"] <= shadow["gross_result"]["pnl"]
