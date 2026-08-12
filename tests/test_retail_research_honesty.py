"""Retail research honesty: costs, PIT valuations, checklist, book correlation wiring."""
from __future__ import annotations

from datetime import datetime, timezone


def test_outcome_to_net_r_subtracts_costs_and_matches_live_edge():
    from core import costs as C
    from scan.live_edge import _row_r

    entry, stop, outcome = 100.0, 95.0, 5.0  # 5% gain on 5% risk = 1R gross
    net = C.outcome_to_net_r(entry, stop, outcome, product="CNC", clip=(-1.5, 4.0))
    assert net is not None
    assert net < 1.0  # costs reduce the R
    row = {"entry_price": entry, "stop_price": stop, "outcome_pct": outcome}
    assert abs(net - (_row_r(row) or 0)) < 1e-9
    drag = C.cost_drag_r(entry, stop, product="CNC")
    assert drag is not None and abs((1.0 - drag) - net) < 1e-9


def test_outcome_to_net_r_invalid_geometry():
    from core.costs import outcome_to_net_r

    assert outcome_to_net_r(100, 105, 2) is None
    assert outcome_to_net_r(0, 95, 2) is None


def test_pit_valuations_asof_gate(tmp_path, monkeypatch):
    from data import pit_valuations as PV

    dest = tmp_path / "pit.json"
    monkeypatch.setenv("QT_PIT_VALUATIONS_FILE", str(dest))
    PV.write_valuations(
        [
            {"symbol": "AAA", "available_ts": "2024-06-01", "pe": 40.0},
            {"symbol": "AAA", "available_ts": "2024-01-01", "pe": 18.0},
        ],
        path=dest,
        source="unit_test",
    )
    early = PV.get_valuation("AAA", "2024-03-01", path=dest)
    assert early is not None and early["pe"] == 18.0
    late = PV.get_valuation("AAA", "2024-07-01", path=dest)
    assert late is not None and late["pe"] == 40.0
    before = PV.get_valuation("AAA", "2023-12-01", path=dest)
    assert before is None


def test_pit_sample_source_not_research_grade(tmp_path, monkeypatch):
    from data import pit_valuations as PV

    dest = tmp_path / "pit.json"
    monkeypatch.setenv("QT_PIT_VALUATIONS_FILE", str(dest))
    PV.write_valuations(
        [{"symbol": "X", "available_ts": "2024-01-01", "pe": 10}],
        path=dest,
        source="SAMPLE_NOT_FOR_RESEARCH",
    )
    assert PV.ledger_status(dest)["research_grade"] is False


def test_retail_checklist_surfaces_gaps():
    from product.retail_research_checklist import build_retail_research_checklist

    payload = build_retail_research_checklist(
        data={
            "bhavcopy": {"ready": True, "sessions": 500, "symbols": 100},
            "snapshot": {"ready": False},
        },
        ca={"research_grade": False, "events": 0},
        universe={
            "survivorship_complete": True,
            "research_grade": False,
            "source": "bhav_inferred",
        },
        pit_valuations={"research_grade": False, "rows": 0},
        live_edge={"overall": {"n": 5, "expectancy_r": 0.1}},
        book_correlation={"n_positions": 0, "n_bets": 0},
        options_eod={"available": False, "snapshots": 0},
    )
    assert payload["gap_count"] >= 3
    keys = {i["key"] for i in payload["gaps"]}
    assert "corporate_actions" in keys
    assert "verified_snapshot" in keys


def test_product_readiness_includes_retail_checklist():
    from product.product_readiness import build_product_readiness

    now = datetime(2026, 8, 1, 5, 0, tzinfo=timezone.utc)
    payload = build_product_readiness(
        market={},
        scan={},
        long_term={},
        news={},
        fno={},
        data={},
        operations={},
        ca={"research_grade": False, "events": 0},
        universe={"survivorship_complete": False, "research_grade": False},
        now=now,
    )
    assert "retail_research_checklist" in payload
    assert payload["retail_research_checklist"]["gap_count"] >= 1
    assert payload["schema_version"] == 2


def test_trade_plan_payload_passes_open_symbols(monkeypatch):
    import product.trade_plan as TP
    import terminal_product_api as api
    from product.trade_plan import TradePlan

    monkeypatch.setattr(
        api.core,
        "_scan_payload",
        lambda: {
            "records": [{"symbol": "ACME", "entry": 100.0, "stop": 95.0, "target": 115.0}],
        },
    )
    monkeypatch.setattr(api.core, "_market_payload", lambda: {"health": "Healthy"})
    monkeypatch.setattr(api.core, "_json_file", lambda *a, **k: {"capital": 100_000.0})
    monkeypatch.setattr(
        api.core,
        "_paper_payload",
        lambda: {"open_positions": [{"symbol": "OTHER"}]},
    )

    captured = {}

    def fake_plan(record, *, capital, regime_factor=1.0, open_symbols=None, **kwargs):
        captured["open_symbols"] = open_symbols
        return TradePlan(
            symbol="ACME",
            tradeable=True,
            reason="",
            entry=100,
            stop=95,
            target=115,
            qty=200,
            invested=20000,
            rupee_risk=1000,
            capped=False,
            pct_of_capital=20,
            risk_pct_of_capital=1,
            reward_risk=3,
            invalidation_pct=5,
            suggested_risk_pct=0.01,
            open_risk_pct_before=None,
            open_risk_pct_after=None,
            heat_verdict="OK",
            cost_drag_r=0.06,
            round_trip_cost_pct=0.32,
            summary="ok",
        )

    monkeypatch.setattr(TP, "plan_for_candidate", fake_plan)
    body = api._trade_plan_payload("ACME")
    assert body["available"] is True
    assert captured["open_symbols"] == ["OTHER"]
    assert body.get("cost_drag_r") == 0.06


def test_book_correlation_report_shape(monkeypatch):
    import risk.correlation as CORR

    monkeypatch.setattr(
        "risk.position_manager.review_positions",
        lambda: [{"symbol": "A"}, {"symbol": "B"}],
    )
    monkeypatch.setattr(CORR, "pairwise_corr", lambda symbols: {("A", "B"): 0.9})
    monkeypatch.setattr(CORR, "clusters_from_corr", lambda symbols, corr: [["A", "B"]])
    report = CORR.book_correlation_report()
    assert report["available"] is True
    assert report["n_positions"] == 2
    assert report["n_bets"] == 1
    assert report["biggest"] == ["A", "B"]
