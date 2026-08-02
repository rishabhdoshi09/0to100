"""Tests for canonical data platform (network-free)."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from data_platform.contracts import QualityStatus
from data_platform.coverage import audit_symbol, remediation_for
from data_platform.import_pipeline import import_fundamentals_json, inspect_file
from data_platform.provider_registry import pick_provider, providers_for
from data_platform.ratios import ratios_from_fundamentals
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
