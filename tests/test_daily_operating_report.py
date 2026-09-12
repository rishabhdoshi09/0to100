"""The session report must be able to say it does not know.

A report that renders a missing scan as zero symbols, or a missing regime as a
neutral one, is worse than no report: it converts an operational failure into a
calm-looking number. Every section here carries its own availability, and the
report ends with the proofs it could not evidence rather than an implied
all-clear.
"""
from __future__ import annotations

import json

import pytest

from product.daily_operating_report import (
    build_daily_operating_report,
    render_text,
    unmet_operating_proofs,
    write_daily_operating_report,
)


@pytest.fixture(autouse=True)
def _empty_runtime(tmp_path, monkeypatch):
    """Nothing on disk: the hardest case for an honest report."""
    monkeypatch.setenv("QT_RUNTIME_ROOT", str(tmp_path))
    monkeypatch.setenv("QT_CONDITIONAL_EVIDENCE", str(tmp_path / "cond.json"))


def test_a_cold_desk_reports_absence_everywhere_rather_than_zeros():
    report = build_daily_operating_report()
    assert report["data"]["available"] is False
    assert report["scan"]["available"] is False
    assert report["market"]["available"] is False
    assert report["market"]["regime"] == "UNKNOWN"
    assert report["scan"]["intended_universe"] is None, "not 0 — unknown"
    assert report["scan"]["coverage_pct"] is None
    assert report["paper"]["available"] is False


def test_the_report_lists_the_proofs_it_could_not_evidence():
    unmet = unmet_operating_proofs(build_daily_operating_report())
    assert "no acquisition provenance recorded" in unmet
    assert "market regime not populated" in unmet
    assert "no scan coverage evidence" in unmet


def test_capital_safety_is_always_stated():
    capital = build_daily_operating_report()["capital_safety"]
    assert capital["live_locked"] is True
    assert capital["broker_mutations"] == 0


def test_capital_safety_fails_closed_when_the_interlock_cannot_be_read(monkeypatch):
    """An unreadable interlock reports LOCKED, never an optimistic unknown."""
    import product.daily_operating_report as R

    monkeypatch.setattr(
        R, "_capital_safety_section",
        R._capital_safety_section.__wrapped__ if hasattr(R._capital_safety_section, "__wrapped__")
        else R._capital_safety_section,
    )
    import sys
    monkeypatch.setitem(sys.modules, "product.live_execution_interlock", None)
    section = R._capital_safety_section()
    assert section["live_locked"] is True
    assert section["broker_mutations"] == 0


def test_a_live_lock_failure_is_the_loudest_unmet_proof():
    report = build_daily_operating_report()
    report["capital_safety"] = {"live_locked": False, "broker_mutations": 0}
    assert "LIVE LOCK NOT CONFIRMED" in unmet_operating_proofs(report)


def test_broker_mutations_are_an_unmet_proof():
    report = build_daily_operating_report()
    report["capital_safety"] = {"live_locked": True, "broker_mutations": 3}
    assert "broker mutations recorded" in unmet_operating_proofs(report)


def test_forward_evidence_metrics_are_withheld_below_the_floor():
    """Statistics from four trades are noise wearing the clothes of measurement."""
    section = build_daily_operating_report()["forward_evidence"]
    assert section["settled_sample"] == 0
    assert "metrics_withheld" in section
    assert "by_setup" not in section


def test_data_section_reads_the_acquisition_provenance(tmp_path):
    from data.acquisition import Source, SourceTier, acquire

    acquire("prices", [
        Source("official", SourceTier.OFFICIAL_API,
               lambda: (_ for _ in ()).throw(ConnectionError("down"))),
        Source("cache", SourceTier.LAST_KNOWN_GOOD, lambda: [1, 2, 3]),
    ])
    data = build_daily_operating_report()["data"]
    assert data["available"] is True
    assert data["stale_sources"] == ["prices"]
    assert data["used_a_fallback"] == ["prices"]
    row = data["datasets"][0]
    assert row["selected_source"] == "cache"
    assert [a["outcome"] for a in row["attempts"]] == ["UNREACHABLE", "OK"]


def test_a_parser_change_is_surfaced_even_when_a_lower_rung_covered_it():
    from data.acquisition import PARSER_CHANGED, Source, SourceTier, Validation, acquire

    acquire("filings", [
        Source("scraper", SourceTier.PUBLIC_SCRAPE, lambda: {"wrong": "shape"},
               lambda p: Validation.bad(PARSER_CHANGED, "columns moved")),
        Source("official", SourceTier.OFFICIAL_API, lambda: [1]),
    ])
    assert build_daily_operating_report()["data"]["parser_changes_seen"] == ["filings"]


def test_the_text_form_states_the_sha_and_the_lock():
    text = render_text(build_daily_operating_report())
    assert "PRODUCTION_SHA" in text
    assert "LIVE_LOCKED=TRUE" in text
    assert "BROKER_MUTATIONS=0" in text
    assert "UNMET OPERATING PROOFS" in text


def test_the_report_is_persisted_under_the_runtime_root(tmp_path):
    path = write_daily_operating_report()
    assert path.exists()
    assert str(tmp_path) in str(path)
    assert json.loads(path.read_text())["schema_version"] == 1


def test_the_report_never_runs_a_scan(monkeypatch):
    """Reading the desk's state must not become an action that changes it."""
    import scan.market_scan_service as MSS

    called = []
    for name in dir(MSS):
        if name.startswith("run_") and callable(getattr(MSS, name, None)):
            monkeypatch.setattr(MSS, name, lambda *a, **k: called.append(name))
    build_daily_operating_report()
    assert called == []
