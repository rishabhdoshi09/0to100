"""Each check must actually catch the wrong state it is named for.

A detector that cannot fire is worse than none: it converts an unexamined risk
into a green tick. So every check here is exercised against a state that should
trip it AND a state that should not, and a check that cannot evaluate reports
UNKNOWN rather than passing quietly.
"""
from __future__ import annotations

import json

import pytest

from product import silent_wrongness_checks as SW


@pytest.fixture(autouse=True)
def _runtime(tmp_path, monkeypatch):
    monkeypatch.setenv("QT_RUNTIME_ROOT", str(tmp_path))
    monkeypatch.setenv("QT_CONDITIONAL_EVIDENCE", str(tmp_path / "cond.json"))
    return tmp_path


def _write(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _audit(tmp_path, summary):
    _write(tmp_path / "logs" / "scan_coverage_audit.json", {"summary": summary})


# ── coverage ───────────────────────────────────────────────────────────────
def test_full_coverage_over_a_thin_scan_is_caught(_runtime):
    _audit(_runtime, {"state": "FULL", "requested": 2000, "checked": 1200,
                      "coverage_pct": 60.0,
                      "universe_provenance": {"state": "ACQUIRED"}})
    finding = SW.check_scan_complete_over_thin_coverage()
    assert finding is not None
    assert finding.severity == SW.CRITICAL


def test_full_coverage_over_a_full_scan_is_fine(_runtime):
    _audit(_runtime, {"state": "FULL", "requested": 2000, "checked": 1990,
                      "coverage_pct": 99.5})
    assert SW.check_scan_complete_over_thin_coverage() is None


def test_a_degraded_scan_is_not_flagged_for_thin_coverage(_runtime):
    """Honest degradation is not the defect; a false FULL is."""
    _audit(_runtime, {"state": "DEGRADED", "coverage_pct": 40.0})
    assert SW.check_scan_complete_over_thin_coverage() is None


def test_a_universe_without_provenance_is_flagged(_runtime):
    _audit(_runtime, {"state": "FULL", "requested": 100, "coverage_pct": 100.0,
                      "universe_provenance": {"state": "UNRECORDED"}})
    finding = SW.check_universe_claimed_without_provenance()
    assert finding is not None
    assert finding.severity == SW.WARNING


def test_a_traced_universe_passes(_runtime):
    _audit(_runtime, {"state": "FULL", "requested": 100, "coverage_pct": 100.0,
                      "universe_provenance": {"state": "ACQUIRED",
                                              "source": "kite_instrument_cache"}})
    assert SW.check_universe_claimed_without_provenance() is None


# ── acquisition ────────────────────────────────────────────────────────────
def test_cache_tier_labelled_as_freshly_acquired_is_caught(_runtime):
    _write(_runtime / "logs" / "provenance" / "prices.json",
           {"dataset": "prices", "state": "ACQUIRED", "tier": 8,
            "source": "bundled_csv", "fallback_level": 2, "record_count": 10})
    finding = SW.check_fallback_data_presented_as_primary()
    assert finding is not None
    assert finding.severity == SW.CRITICAL


def test_a_cache_tier_reporting_last_known_good_is_fine(_runtime):
    _write(_runtime / "logs" / "provenance" / "prices.json",
           {"dataset": "prices", "state": "LAST_KNOWN_GOOD", "tier": 8,
            "source": "bundled_csv", "fallback_level": 2, "record_count": 10})
    assert SW.check_fallback_data_presented_as_primary() is None


def test_a_zero_row_success_is_caught(_runtime):
    _write(_runtime / "logs" / "provenance" / "filings.json",
           {"dataset": "filings", "state": "ACQUIRED", "tier": 1,
            "source": "official", "record_count": 0})
    finding = SW.check_zero_row_successful_acquisition()
    assert finding is not None
    assert finding.severity == SW.CRITICAL


def test_an_acquisition_with_rows_passes(_runtime):
    _write(_runtime / "logs" / "provenance" / "filings.json",
           {"dataset": "filings", "state": "ACQUIRED", "tier": 1,
            "source": "official", "record_count": 42})
    assert SW.check_zero_row_successful_acquisition() is None


# ── evidence class ─────────────────────────────────────────────────────────
def test_a_foreign_class_in_a_paper_cell_is_caught(_runtime, monkeypatch):
    from product import conditional_evidence as CE

    store = {
        "schema_version": 1,
        "cells": {
            "PAPER_FORWARD::setup=VCP": {"evidence_class": "HISTORICAL_REPLAY",
                                         "count": 40},
        },
    }
    monkeypatch.setattr(CE, "load", lambda *a, **k: store)
    finding = SW.check_non_market_evidence_in_paper_cells()
    assert finding is not None
    assert finding.severity == SW.CRITICAL


def test_a_correctly_classed_paper_cell_passes(_runtime, monkeypatch):
    from product import conditional_evidence as CE

    store = {"cells": {"PAPER_FORWARD::setup=VCP":
                       {"evidence_class": "PAPER_FORWARD", "count": 40}}}
    monkeypatch.setattr(CE, "load", lambda *a, **k: store)
    assert SW.check_non_market_evidence_in_paper_cells() is None


# ── attribution ────────────────────────────────────────────────────────────
def test_unattributable_open_positions_are_flagged(_runtime):
    _write(_runtime / "logs" / "intelligence" / "intel_book.json",
           {"open": [{"symbol": "AAA", "decision_id": ""},
                     {"symbol": "BBB", "decision_id": "dec_1"}]})
    finding = SW.check_unattributable_open_positions()
    assert finding is not None
    assert finding.evidence["count"] == 1


def test_fully_attributable_positions_pass(_runtime):
    _write(_runtime / "logs" / "intelligence" / "intel_book.json",
           {"open": [{"symbol": "AAA", "decision_id": "dec_1"}]})
    assert SW.check_unattributable_open_positions() is None


# ── recommendations ────────────────────────────────────────────────────────
def test_recommendations_without_a_scan_timestamp_are_caught(monkeypatch):
    import product.recommendations_store as RS

    monkeypatch.setattr(RS, "load_recommendations",
                        lambda *a, **k: {"schema_version": 4, "categories": [],
                                         "scan_scanned_at": ""})
    finding = SW.check_decisions_published_without_a_scan()
    assert finding is not None
    assert finding.severity == SW.CRITICAL


# ── the runner ─────────────────────────────────────────────────────────────
def test_a_clean_desk_reports_clean(_runtime):
    result = SW.run_silent_wrongness_checks()
    assert result["clean"] is True
    assert result["checks_run"] == len(SW.CHECKS)


def test_a_check_that_raises_becomes_an_unknown_finding(monkeypatch):
    def explodes():
        raise RuntimeError("detector is broken")

    monkeypatch.setattr(SW, "CHECKS", (explodes,))
    result = SW.run_silent_wrongness_checks()
    assert result["clean"] is False
    assert result["unknown"] == 1
    assert "the check itself failed" in result["findings"][0]["summary"]


def test_a_critical_finding_becomes_an_unmet_operating_proof(_runtime):
    from product.daily_operating_report import (
        build_daily_operating_report,
        unmet_operating_proofs,
    )

    _write(_runtime / "logs" / "provenance" / "filings.json",
           {"dataset": "filings", "state": "ACQUIRED", "tier": 1,
            "source": "official", "record_count": 0})
    report = build_daily_operating_report()
    assert report["silent_wrongness"]["critical"] >= 1
    assert any("critical silent-wrongness" in item
               for item in unmet_operating_proofs(report))
