"""Phase 9 — deterministic production chaos / reliability."""

from __future__ import annotations

from datetime import datetime, timezone

from product.autopilot_journal import load_journal, why_no_trade
from product.execution_adapter import LiveExecutionAdapter, LiveMoneyLocked
from product.live_readiness import evaluate_live_readiness
from product.paper_autopilot import DUPLICATE_POSITION, run_reco_paper_cycle
from product.system_health_contract import build_system_health_contract
from research.auto_research.paper_book import PaperBook


def _card(symbol="TCS"):
    return {
        "symbol": symbol,
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
        "methods": [{"id": "funds", "status": "pass", "points": 80}],
    }


def _cycle(book, cards):
    now = datetime(2026, 9, 1, 10, 0, tzinfo=timezone.utc)
    return run_reco_paper_cycle(
        book=book,
        cards=cards,
        now=now,
        as_of="2026-09-01",
        workspace={
            "schema_version": 4,
            "generated_at": now.isoformat(),
            "scan_scanned_at": now.isoformat(),
            "categories": [{"id": "w", "count": len(cards), "cards": cards}],
        },
    )


def test_restart_with_open_positions_recovers_stops_and_targets():
    book = PaperBook(capital=100_000)
    _cycle(book, [_card("RELIANCE")])
    pos = next(iter(book.open.values()))
    snap = book.snapshot()
    restored = PaperBook(capital=50_000)
    restored.restore(snap)
    got = next(iter(restored.open.values()))
    assert got.symbol == "RELIANCE"
    assert got.stop_price == pos.stop_price == 94.0
    assert got.target_price == pos.target_price == 115.0
    assert got.qty == pos.qty
    assert restored.open_risk() > 0


def test_corrupt_snapshot_does_not_invent_positions_or_pnl():
    book = PaperBook(capital=100_000)
    book.restore({"open": "not-a-list", "closed": None, "realized_pnl": "NaN"})
    assert book.open == {}
    assert book.closed == []


def test_duplicate_scheduler_or_reco_cycle_cannot_double_fill():
    book = PaperBook(capital=100_000)
    first = _cycle(book, [_card(), _card()])
    second = _cycle(book, [_card()])
    assert len(book.open) == 1
    assert second["rejections"][0]["reason_code"] == DUPLICATE_POSITION
    assert first["taken"][0]["symbol"] == "TCS"


def test_truncated_journal_is_not_fake_green(tmp_path):
    path = tmp_path / "journal.json"
    path.write_text("{partial", encoding="utf-8")
    payload = load_journal(path)
    assert payload["cycles"] == []
    why = why_no_trade(path)
    assert why["available"] is False
    assert why["decision"] == "UNKNOWN"
    assert "NO_CYCLE_RECORDED" in why["reasons"]


def test_health_contract_not_fake_green_when_scan_and_nse_missing():
    payload = build_system_health_contract(
        scan={"available": False, "records": []},
        data={"ready": False, "bhavcopy": {"ready": False}},
        news={"available": False},
        operations={"running": False},
        autonomy={"running": False, "learning_status": "NO_EOD_LEARNING_YET"},
        recommendations_available=False,
        market_report_as_of="",
        paper={"enabled": True},
        execution={"lanes": {
            "scanner": "MISSING",
            "recommendations": "MISSING",
            "paper_execution": "UNKNOWN",
            "autonomy_scheduler": "WAITING",
        }},
    )
    by = {lane["key"]: lane["status"] for lane in payload["lanes"]}
    assert payload["collapsed_status"] is None
    assert by["scanner"] != "HEALTHY"
    assert by["daily_ohlcv"] == "MISSING"
    assert by["paper_execution"] != "HEALTHY"


def test_last_good_fundamentals_survive_provider_outage(tmp_path, monkeypatch):
    monkeypatch.setattr("fundamentals.cache._DB_PATH", tmp_path / "fund.db")
    from fundamentals.cache import FundamentalsCache
    cache = FundamentalsCache()
    cache.set("TCS", {"pe": 24.5, "source_label": "secondary_public"})
    import sqlite3
    import time
    from fundamentals import cache as cache_mod
    with sqlite3.connect(cache_mod._DB_PATH) as conn:
        conn.execute(
            "UPDATE fundamentals_cache SET fetched_at = ? WHERE symbol = ?",
            (time.time() - 5 * 86_400, "TCS"),
        )
        conn.commit()
    assert cache.get("TCS") is None
    last = cache.get("TCS", allow_stale=True)
    assert last["pe"] == 24.5
    assert last["source_label"] == "last_good_snapshot"


def test_malformed_provider_response_does_not_impute_zero():
    from product.due_diligence.moat_layer import company_intelligence_moat
    moat = company_intelligence_moat({"profit_loss": "bad", "cash_flow": None}, framework_id="industrials")
    assert moat["by_id"]["debt"]["value"] is None
    assert moat["dd_effect"] == "NEUTRAL"


def test_live_adapter_and_readiness_remain_fail_closed():
    adapter = LiveExecutionAdapter()
    try:
        adapter.submit(object())
        raise AssertionError("live adapter must raise")
    except LiveMoneyLocked:
        pass
    ready = evaluate_live_readiness(
        settled_trades=10_000,
        trading_days=400,
        expectancy_R=2.0,
        max_drawdown_pct=1.0,
        distinct_regimes=5,
        stops_proven=True,
        critical_lanes_broken=False,
        rules_hash_stable=True,
    )
    assert ready["live_enabled"] is False


def test_stale_data_labelled_not_silently_fresh():
    from product.due_diligence.provenance import resolve_fact
    fact = resolve_fact(last_good={"value": 9.0, "source": "cache", "stale": True})
    assert fact["stale"] is True
    assert fact["value"] == 9.0
