"""Tests for canonical data platform (network-free)."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from data_platform.contracts import QualityStatus
from data_platform.coverage import audit_symbol, remediation_for
from data_platform.import_pipeline import import_fundamentals_json, inspect_file
from data_platform.provider_registry import pick_provider, providers_for
from data_platform.ratios import flatten_screener_snapshot, ratios_from_fundamentals
from data_platform.contracts import DataCapability


def test_provider_priority_order():
    ranked = providers_for(DataCapability.DAILY_PRICES)
    assert ranked[0].name == "nse_bhavcopy"
    assert pick_provider(DataCapability.LIVE_QUOTES).name == "kite"


def test_ratios_missing_inputs_stay_missing():
    rows = ratios_from_fundamentals("TEST", {})
    pe = next(r for r in rows if r["key"] == "pe")
    assert pe["value"] is None
    assert pe["missing_reason"]


def test_ratios_operating_margin_formula():
    raw = {
        "revenue": 1000,
        "operating_profit": 200,
        "period": "FY2024",
        "scope": "consolidated",
    }
    rows = ratios_from_fundamentals("TEST", raw)
    margin = next(r for r in rows if r["key"] == "operating_margin")
    assert margin["value"] == 20.0
    assert margin["formula"] == "operating_profit / revenue"


def test_ratios_from_screener_snapshot():
    snapshot = {
        "key_ratios": [
            {"name": "Stock P/E", "value": "25"},
            {"name": "ROE", "value": "22%"},
            {"name": "Debt to equity", "value": "0.20"},
        ],
        "profit_loss": [
            {"": "Sales", "2024": 100, "2025": 120, "2026": 150},
            {"": "Operating Profit", "2024": 20, "2025": 24, "2026": 30},
            {"": "Net Profit", "2024": 10, "2025": 13, "2026": 18},
        ],
        "balance_sheet": [
            {"": "Borrowings", "2024": 40, "2025": 35, "2026": 30},
            {"": "Equity Capital", "2024": 100, "2025": 110, "2026": 120},
        ],
    }
    flat = flatten_screener_snapshot(snapshot)
    assert flat["revenue"] == 150
    assert flat["_direct_pe"] == 25.0
    assert flat["_direct_roe"] == 22.0
    rows = ratios_from_fundamentals("TEST", snapshot)
    pe = next(r for r in rows if r["key"] == "pe")
    roe = next(r for r in rows if r["key"] == "roe")
    margin = next(r for r in rows if r["key"] == "operating_margin")
    assert pe["value"] == 25.0
    assert roe["value"] == 22.0
    assert margin["value"] == 20.0


def test_peer_average_pe_from_screener_table():
    from data_platform.ratios import compute_peer_average_pe, ratios_from_fundamentals

    snapshot = {
        "key_ratios": [{"name": "Stock P/E", "value": "25"}],
        "peer_comparison": [
            {"": "Peer A", "P/E": "18", "CMP": "400"},
            {"": "Peer B", "P/E": "22", "CMP": "500"},
            {"": "Peer C", "P/E": "20", "CMP": "450"},
        ],
    }
    stats = compute_peer_average_pe("TEST", snapshot, [])
    assert stats["average_pe"] == 20.0
    assert stats["sample_count"] == 3
    assert stats["stock_pe"] == 25.0
    assert stats["pe_vs_peer_avg"] == 1.25
    rows = ratios_from_fundamentals("TEST", snapshot, peer_stats=stats)
    peer_avg = next(r for r in rows if r["key"] == "peer_avg_pe")
    assert peer_avg["value"] == 20.0


def test_peer_average_pe_cache_peers(monkeypatch):
    from data_platform.ratios import compute_peer_average_pe
    from fundamentals.cache import FundamentalsCache

    cache = FundamentalsCache()
    cache.set("PEER1", {"key_ratios": [{"name": "Stock P/E", "value": "30"}]})
    cache.set("PEER2", {"key_ratios": [{"name": "Stock P/E", "value": "10"}]})
    stats = compute_peer_average_pe(
        "TEST",
        {"key_ratios": [{"name": "Stock P/E", "value": "20"}]},
        ["PEER1", "PEER2"],
    )
    assert stats["average_pe"] == 20.0
    assert stats["sample_count"] == 2
    cache.invalidate("PEER1")
    cache.invalidate("PEER2")


def test_coverage_audit_empty_symbol():
    cov = audit_symbol("")
    assert cov.identity == QualityStatus.ERROR


def test_remediation_queue_price_missing():
    from data_platform.contracts import DataCoverage
    cov = DataCoverage(symbol="ABC", price_history=QualityStatus.MISSING, reasons={"price_history": "none"})
    actions = remediation_for(cov)
    assert any(a["action"] == "schedule_price_backfill" for a in actions)


def test_import_fundamentals_json(tmp_path: Path, monkeypatch):
    path = tmp_path / "fund.json"
    path.write_text(json.dumps({"RELIANCE": {"revenue": 100, "net_profit": 10}}), encoding="utf-8")
    monkeypatch_db = tmp_path / "fundamentals_cache.db"
    import fundamentals.cache as fc
    monkeypatch.setattr(fc, "_DB_PATH", monkeypatch_db)
    result = import_fundamentals_json(path)
    assert result["ok"] and result["imported"] == 1
    cached = fc.FundamentalsCache().get("RELIANCE")
    assert cached and cached.get("revenue") == 100


def test_inspect_csv(tmp_path: Path):
    path = tmp_path / "sample.csv"
    path.write_text("symbol,revenue\nRELIANCE,100\n", encoding="utf-8")
    info = inspect_file(path)
    assert info["ok"] and "symbol" in info["columns"]


def test_fundamentals_cache_db_is_repo_absolute():
    from fundamentals.cache import _DB_PATH, _ROOT
    assert _DB_PATH == _ROOT / "data" / "fundamentals_cache.db"
