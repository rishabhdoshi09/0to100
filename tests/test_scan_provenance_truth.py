from __future__ import annotations

import json
from contextlib import contextmanager
from datetime import datetime, timezone

from product.scan_store import (
    build_scan_payload,
    load_scan,
    scan_age_hours,
    scan_artifact_is_fresh,
)
from scan.market_scan_service import _coverage_failures, run_whole_market_scan


def _prov(*, session: str = "2026-09-11", current: bool = True) -> dict:
    return {
        "market_session_date": session or None,
        "price_data_as_of": session or None,
        "expected_session_date": "2026-09-11",
        "freshness_state": "CURRENT" if current else "STALE",
        "provenance_reason": "CURRENT" if current else "HISTORY_STALE",
    }


def test_scan_execution_date_is_not_market_session_date():
    completed = datetime(2026, 9, 13, 12, 50, tzinfo=timezone.utc)
    payload = build_scan_payload(
        {"ABC": "ABC Ltd"},
        [],
        scan_completed_at=completed,
        history_provenance=_prov(session="2026-09-11"),
    )

    assert payload["scan_completed_at"] == "2026-09-13T12:50:00+00:00"
    assert payload["scanned_at"] == payload["scan_completed_at"]
    assert payload["market_session_date"] == "2026-09-11"
    assert payload["price_data_as_of"] == "2026-09-11"
    assert payload["scan_duration_s"] is None
    assert payload["scan_duration_status"] == "UNAVAILABLE"
    assert payload["scan_duration_reason"] == "SCAN_START_TIME_UNAVAILABLE"


def test_duration_is_emitted_only_for_ordered_timestamps():
    payload = build_scan_payload(
        {},
        [],
        scan_started_at="2026-09-13T12:00:00+00:00",
        scan_completed_at="2026-09-13T12:00:12.500000+00:00",
        history_provenance=_prov(),
    )

    assert payload["scan_duration_s"] == 12.5
    assert payload["scan_duration_status"] == "AVAILABLE"
    assert payload["scan_duration_reason"] is None


def test_reversed_scan_timestamps_never_become_fake_zero_duration():
    payload = build_scan_payload(
        {},
        [],
        scan_started_at="2026-09-13T12:01:00+00:00",
        scan_completed_at="2026-09-13T12:00:00+00:00",
        history_provenance=_prov(),
    )

    assert payload["scan_duration_s"] is None
    assert payload["scan_duration_status"] == "UNAVAILABLE"
    assert payload["scan_duration_reason"] == "SCAN_TIMESTAMP_ORDER_INVALID"


def test_schema_v1_and_v2_scan_artifacts_both_load(tmp_path):
    path = tmp_path / "scan.json"
    v1 = {"schema_version": 1, "records": [], "scanned_at": "2026-09-11T10:00:00+00:00"}
    path.write_text(json.dumps(v1), encoding="utf-8")
    assert load_scan(path) == v1

    v2 = build_scan_payload(
        {},
        [],
        scan_completed_at="2026-09-13T12:00:00+00:00",
        history_provenance=_prov(),
    )
    path.write_text(json.dumps(v2), encoding="utf-8")
    assert load_scan(path) == v2


def test_future_scan_timestamp_has_unknown_age_and_is_not_fresh(tmp_path):
    now = datetime(2026, 9, 13, 12, 0, tzinfo=timezone.utc)
    payload = {
        "schema_version": 2,
        "records": [],
        "scanned_at": "2026-09-13T12:05:00+00:00",
    }
    path = tmp_path / "future.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    assert scan_age_hours(payload, now=now) is None
    assert scan_artifact_is_fresh(path, max_age_s=3600, now=now) is False


def test_missing_history_provenance_has_explicit_reason():
    payload = build_scan_payload(
        {},
        [],
        scan_completed_at="2026-09-13T12:00:00+00:00",
        history_provenance={
            "market_session_date": None,
            "price_data_as_of": None,
            "expected_session_date": "2026-09-11",
            "freshness_state": "UNAVAILABLE",
            "provenance_reason": "HISTORY_NOT_READY",
        },
    )

    assert payload["market_session_date"] is None
    assert payload["price_data_as_of"] is None
    assert payload["freshness_state"] == "UNAVAILABLE"
    assert payload["provenance_reason"] == "HISTORY_NOT_READY"


def test_universe_failed_excludes_policy_outcomes():
    coverage = {
        "data_unavailable": 3,
        "analysis_errors": 2,
        "analysis_skipped": 1,
        "not_observed": 4,
        "policy_excluded": 19,
        "checked": 700,
    }
    assert _coverage_failures(coverage) == 10
    assert _coverage_failures({"checked": 700, "policy_excluded": 19}) is None


def test_production_scan_service_persists_ordered_real_timing(monkeypatch):
    import product.scan_store as scan_store
    import scan.bulk_fetcher as bulk_fetcher
    import scan.scan_coverage as scan_coverage

    monkeypatch.setattr(scan_store, "_history_provenance", lambda: _prov(session="2026-09-11"))
    monkeypatch.setattr(bulk_fetcher, "cached_symbols", lambda: ["ABC"])

    class Probe:
        def finalize(self, results, *, cached, walked_total):
            return {
                "summary": {
                    "state": "FULL",
                    "requested": 1,
                    "checked": 1,
                    "qualified": 1,
                    "policy_excluded": 0,
                    "data_unavailable": 0,
                    "analysis_errors": 0,
                    "analysis_skipped": 0,
                    "not_observed": 0,
                    "scanner_instrumented": True,
                    "reason_counts": {"QUALIFIED": 1},
                },
                "ledger": [{"symbol": "ABC", "status": "QUALIFIED"}],
            }

    @contextmanager
    def observe_scanner(_scanner, _symbols):
        yield Probe()

    monkeypatch.setattr(scan_coverage, "observe_scanner", observe_scanner)

    class Scanner:
        def scan(self, symbols, *, progress=None, prefetch=False):
            assert symbols == ["ABC"]
            if progress:
                progress(1, 1)
            return [{
                "symbol": "ABC",
                "signals": ["MOMENTUM"],
                "reasons": ["Real test setup"],
                "verdict": "BUY",
                "price": 100.0,
                "entry": 101.0,
                "stop": 97.0,
                "target": 109.0,
                "score": 1.0,
            }]

    report = run_whole_market_scan(
        universe_provider=lambda: {"ABC": "ABC Ltd"},
        prefetch_fn=lambda symbols, progress=None: len(symbols),
        scanner=Scanner(),
        fno_provider=lambda: set(),
        save=False,
        snapshot_id="snapshot-test",
    )

    assert report.ok
    payload = report.payload
    assert payload["market_session_date"] == "2026-09-11"
    assert payload["requested_universe"] == 1
    assert payload["universe_failed"] == 0
    assert payload["scan_duration_status"] == "AVAILABLE"
    assert payload["scan_duration_reason"] is None
    assert payload["scan_duration_s"] is not None
    assert payload["scan_duration_s"] >= 0.0
    started = datetime.fromisoformat(payload["scan_started_at"])
    completed = datetime.fromisoformat(payload["scan_completed_at"])
    assert completed >= started
