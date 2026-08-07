from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

from product.conviction import build_conviction_shortlist
from scan.long_term_service import run_long_term_scan, score_current_fundamentals
from screener.engine import _extract_fundamentals


def _market(health="Healthy", leaders=("Technology",), laggards=()):
    return SimpleNamespace(health=health, leaders=leaders, laggards=laggards)


def test_conviction_shortlist_requires_market_stock_and_entry_alignment():
    payload = {"records": [{
        "symbol": "AAA", "company": "Alpha", "status": "Ready to trade",
        "verdict": "BUY", "score": 85, "price": 100, "entry": 101,
        "stop": 95, "target": 115, "rsi": 64, "volume_ratio": 2.1,
        "chase_risk": False, "reasons": ["Clean base breakout"],
    }]}
    rows = build_conviction_shortlist(payload, _market(),
                                      sector_lookup=lambda _s: "Technology")
    assert rows[0]["classification"] == "HIGH_CONVICTION"
    assert rows[0]["conviction_score"] >= 75
    assert "Leading sector: Technology" in rows[0]["reasons"]

    weak = build_conviction_shortlist(payload, _market("Weak"),
                                      sector_lookup=lambda _s: "Technology")
    assert weak[0]["classification"] != "HIGH_CONVICTION"

    payload["records"][0]["chase_risk"] = True
    chased = build_conviction_shortlist(payload, _market(),
                                        sector_lookup=lambda _s: "Technology")
    assert chased[0]["classification"] == "WAIT_FOR_PULLBACK"


def _deep_snapshot():
    return {
        "key_ratios": [
            {"name": "Stock P/E", "value": "25"},
            {"name": "ROE", "value": "22%"},
            {"name": "ROCE", "value": "25%"},
            {"name": "Debt to equity", "value": "0.20"},
            {"name": "Market Cap", "value": "25000"},
        ],
        "profit_loss": [
            {"": "Sales", "2023": 100, "2024": 120, "2025": 150, "2026": 185},
            {"": "Net Profit", "2023": 10, "2024": 13, "2025": 18, "2026": 25},
        ],
        "cash_flow": [
            {"": "Cash from Operating Activity", "2023": 9, "2024": 14,
             "2025": 20, "2026": 28},
        ],
        "shareholding": [
            {"": "Promoters", "2025": 58, "2026": 59},
            {"": "Pledged percentage", "2025": 0, "2026": 0},
        ],
    }


def test_fundamental_extractor_and_score_are_complete_and_current_only():
    fund = _extract_fundamentals(_deep_snapshot())
    assert fund["roce"] == 25
    assert fund["promoter_holding"] == 59
    assert fund["promoter_pledge"] == 0
    assert fund["sales_growth_3y"] > 20
    assert fund["profit_growth_3y"] > 30
    assert fund["cfo_to_pat"] > 1
    quality = score_current_fundamentals(fund, sector="Technology")
    assert quality["coverage"] >= 0.95
    assert quality["score"] >= 70
    assert not quality["severe_red_flag"]


def _technical_scanner(**_kwargs):
    return [{
        "symbol": "AAA", "score": 76, "verdict": "LONG_TERM_BUY",
        "price": 100, "mom_12m_pct": 24, "mom_6m_pct": 12,
        "from_high_pct": 5, "turnover_cr": 10, "extension_pct": 12,
        "factors": ["above 200-DMA", "200-DMA rising"],
    }]


def test_long_term_service_requires_fundamental_coverage_for_approval():
    good = run_long_term_scan(
        technical_scanner=_technical_scanner,
        fundamental_provider=lambda _s, _r: _deep_snapshot(),
        sector_lookup=lambda _s: "Technology", save=False,
    )
    row = good.payload["records"][0]
    assert good.ok
    assert row["classification"] == "QUALITY_COMPOUNDER"
    assert row["fundamentals_point_in_time"] is False
    assert good.payload["fundamentals_point_in_time"] is False

    missing = run_long_term_scan(
        technical_scanner=_technical_scanner,
        fundamental_provider=lambda _s, _r: None,
        sector_lookup=lambda _s: "Technology", save=False,
    )
    assert missing.payload["records"][0]["classification"] == "NEEDS_FUNDAMENTALS"


def test_long_term_red_flags_cannot_be_promoted():
    snapshot = _deep_snapshot()
    snapshot["key_ratios"].append({"name": "Debt to equity", "value": "3.5"})
    snapshot["shareholding"][1]["2026"] = 35
    report = run_long_term_scan(
        technical_scanner=_technical_scanner,
        fundamental_provider=lambda _s, _r: snapshot,
        sector_lookup=lambda _s: "Industrials", save=False,
    )
    assert report.payload["records"][0]["classification"] == "AVOID_REVIEW"


def test_long_term_controls_jobs_and_weekly_schedule_are_wired():
    from research.autonomy import controls as CTRL, jobs as JOBS, schedules as SCH
    assert CTRL.RUN_LONG_TERM_SCAN_NOW in CTRL.VALID_CONTROLS
    assert CTRL.REFRESH_LONG_TERM_NOW in CTRL.VALID_CONTROLS
    assert CTRL.TRACK_LONG_TERM_IDEA in CTRL.VALID_CONTROLS
    assert SCH.LONG_TERM_SCAN in JOBS.HANDLERS
    assert SCH.LONG_TERM_REFRESH in JOBS.HANDLERS
    assert SCH.long_term_weekly_due(datetime(2026, 7, 31, 18, 10))
    assert not SCH.long_term_weekly_due(datetime(2026, 7, 30, 18, 10))


def test_retail_pages_queue_work_instead_of_scanning_or_starting_workers():
    conviction = Path("ui/conviction_page.py").read_text(encoding="utf-8")
    long_term = Path("ui/long_term_page.py").read_text(encoding="utf-8")
    assert "RUN_SCAN_NOW" in conviction
    assert "RUN_LONG_TERM_SCAN_NOW" in long_term
    assert "REFRESH_LONG_TERM_NOW" in long_term
    for source in (conviction, long_term):
        assert "scan_long_term(" not in source
        assert ".start()" not in source
        assert "place_order" not in source
    assert "record_picks(" not in long_term


def test_track_long_term_control_is_applied_by_supervisor(tmp_path, monkeypatch):
    from research.autonomy import controls as CTRL
    from research.autonomy.supervisor import Supervisor
    import product.long_term_store as STORE
    import core.long_term_tracker as TRACKER

    class Deps:
        def now_ist(self): return datetime(2026, 7, 31, 18, 30)
        def holidays(self): return set()
        def active_snapshot_id(self): return "snap"

    payload = {"records": [{
        "symbol": "AAA", "classification": "QUALITY_COMPOUNDER",
        "combined_score": 80, "quality_factors": ["ROCE 25%"],
        "fundamental_coverage": 0.85,
    }]}
    recorded = []
    monkeypatch.setattr(STORE, "load_long_term_scan", lambda: payload)
    monkeypatch.setattr(TRACKER, "record_picks", lambda rows: recorded.extend(rows) or rows)

    supervisor = Supervisor(tmp_path / "autonomy", deps=Deps())
    supervisor.controls.request(CTRL.TRACK_LONG_TERM_IDEA, value={"symbol": "AAA"},
                                reason="owner test")
    supervisor._process_controls()
    assert recorded and recorded[0]["symbol"] == "AAA"
    assert recorded[0]["score"] == 80
    assert supervisor.controls.recent()[0].status == CTRL.PROCESSED
    supervisor.shutdown()
