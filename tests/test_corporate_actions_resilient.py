from __future__ import annotations

import json
from datetime import date

from data import corporate_actions_resilient as CAR


def _event(symbol="ABC", ex_date="2026-01-15", factor=2.0, kind="bonus", source="nse_csv"):
    return {
        "symbol": symbol,
        "ex_date": ex_date,
        "record_date": ex_date,
        "factor": factor,
        "type": kind,
        "subject": "fixture",
        "source": source,
        "source_url": "fixture://source",
        "source_tier": 1,
        "fetched_at": "2026-08-28T09:00:00+05:30",
        "verification": "official_single_source",
        "provenance": [{"source": source, "source_url": "fixture://source", "fetched_at": "x"}],
    }


def test_share_count_parser_handles_bonus_split_and_consolidation():
    assert CAR.parse_share_count_action("Bonus 1:1") == ("bonus", 2.0)
    assert CAR.parse_share_count_action(
        "Face Value Split (Sub-Division) - From Rs 10/- Per Share To Rs 2/- Per Share"
    ) == ("split", 5.0)
    assert CAR.parse_share_count_action(
        "Consolidation of equity shares from Rs 1 per share to Rs 10 per share"
    ) == ("consolidation", 0.1)
    assert CAR.parse_share_count_action("Dividend - Rs 10 Per Share") is None


def test_csv_fallback_can_complete_all_required_windows(tmp_path):
    events = tmp_path / "ca_events.json"
    coverage = tmp_path / "ca_coverage.json"
    calls = {"json": 0, "csv": 0}

    def json_fail(start, end):
        calls["json"] += 1
        raise RuntimeError("403")

    def csv_ok(start, end):
        calls["csv"] += 1
        # A valid official window may contain zero share-count events.
        return [_event(ex_date=start.isoformat())] if calls["csv"] == 1 else []

    result = CAR.refresh_events_resilient(
        years=1,
        today=date(2026, 8, 28),
        events_path=events,
        coverage_path=coverage,
        nse_json_fetcher=json_fail,
        nse_csv_fetcher=csv_ok,
        bse_fetcher=lambda *_: (_ for _ in ()).throw(AssertionError("BSE should not be needed")),
        sleep_fn=lambda _s: None,
        budget_s=30,
    )

    assert calls["json"] >= 2
    assert calls["csv"] >= 2
    assert result["coverage_complete"] is True
    assert result["n_events"] == 1
    stored = json.loads(events.read_text())
    assert stored[0]["factor"] == 2.0


def test_one_failed_window_does_not_abort_later_windows(tmp_path):
    events = tmp_path / "ca_events.json"
    coverage = tmp_path / "ca_coverage.json"
    seen = []

    def json_fetch(start, end):
        seen.append((start, end))
        if len(seen) == 1:
            raise RuntimeError("first window unavailable")
        return []

    def csv_fetch(start, end):
        if len(seen) == 1:
            raise RuntimeError("csv also unavailable")
        return []

    def bse_fetch(start, end):
        if len(seen) == 1:
            raise RuntimeError("bse also unavailable")
        return []

    result = CAR.refresh_events_resilient(
        years=1,
        today=date(2026, 8, 28),
        events_path=events,
        coverage_path=coverage,
        nse_json_fetcher=json_fetch,
        nse_csv_fetcher=csv_fetch,
        bse_fetcher=bse_fetch,
        sleep_fn=lambda _s: None,
        budget_s=30,
    )

    assert len(seen) > 1, "historical walk must continue after an individual window fails"
    assert result["coverage_complete"] is False
    raw = json.loads(coverage.read_text())
    assert any(not row["success"] for row in raw["windows"].values())
    assert any(row["success"] for row in raw["windows"].values())


def test_conflicting_official_factors_fail_closed():
    existing = [_event(factor=2.0, source="nse_api")]
    incoming = [_event(factor=3.0, source="bse_api")]
    merged, conflicts = CAR.merge_verified_events(existing, incoming)
    assert merged == []
    assert conflicts and conflicts[0]["factors"] == [2.0, 3.0]


def test_duplicate_official_sources_merge_provenance_without_double_adjustment():
    existing = [_event(factor=2.0, source="nse_api")]
    incoming = [_event(factor=2.0, source="bse_api")]
    merged, conflicts = CAR.merge_verified_events(existing, incoming)
    assert not conflicts
    assert len(merged) == 1
    assert merged[0]["factor"] == 2.0
    assert merged[0]["verification"] == "official_cross_verified"
    assert {p["source"] for p in merged[0]["provenance"]} == {"nse_api", "bse_api"}


def test_nonempty_ledger_does_not_claim_complete_coverage(tmp_path):
    events = tmp_path / "ca_events.json"
    coverage = tmp_path / "ca_coverage.json"
    events.write_text(json.dumps([_event()]))
    status = CAR.coverage_status(
        years=1,
        today=date(2026, 8, 28),
        events_path=events,
        coverage_path=coverage,
    )
    assert status["available"] is True
    assert status["coverage_complete"] is False
    assert status["missing_windows"]
