from __future__ import annotations

from datetime import datetime, timezone

from product.due_diligence.coverage import inspect_research_coverage


NOW = datetime(2026, 9, 3, 12, 0, tzinfo=timezone.utc)


def _quarterly_raw() -> dict:
    return {
        "quarterly_results": [
            {
                "row_label": "Sales",
                "Mar 2026": "100",
                "Jun 2026": "110",
            }
        ]
    }


def _dataset(payload: dict, dataset_id: str) -> dict:
    return next(row for row in payload["datasets"] if row["id"] == dataset_id)


def test_failed_refresh_does_not_make_cached_quarterly_data_current() -> None:
    coverage = inspect_research_coverage(
        symbol="ABC",
        raw=_quarterly_raw(),
        autonomy={
            "dataset_meta": {
                "quarterly_results": {
                    "status": "current",
                    "checked_at": "2026-09-03T11:55:00+00:00",
                    "error": "provider refresh failed: HTTP 503",
                }
            }
        },
        framework_id="generic",
        now=NOW,
    )

    row = _dataset(coverage, "quarterly_results")
    assert row["present"] is True
    assert row["status"] == "acquisition_failed"
    assert row["usable_cached"] is True
    assert "cached evidence retained" in row["age_label"]
    assert "quarterly_results" in coverage["to_fetch"]
    assert coverage["needs_acquire"] is True
    assert coverage["latest_data_refresh"] is None


def test_source_outage_keeps_cache_usable_but_retryable() -> None:
    coverage = inspect_research_coverage(
        symbol="ABC",
        raw=_quarterly_raw(),
        autonomy={
            "dataset_meta": {
                "quarterly_results": {
                    "status": "source_unavailable",
                    "checked_at": "2026-09-03T11:55:00+00:00",
                }
            }
        },
        framework_id="generic",
        now=NOW,
    )

    row = _dataset(coverage, "quarterly_results")
    assert row["present"] is True
    assert row["status"] == "source_unavailable"
    assert row["usable_cached"] is True
    assert "quarterly_results" in coverage["to_fetch"]


def test_successful_cached_dataset_can_still_be_current() -> None:
    coverage = inspect_research_coverage(
        symbol="ABC",
        raw=_quarterly_raw(),
        autonomy={
            "dataset_meta": {
                "quarterly_results": {
                    "status": "current",
                    "checked_at": "2026-09-03T11:55:00+00:00",
                }
            }
        },
        framework_id="generic",
        now=NOW,
    )

    row = _dataset(coverage, "quarterly_results")
    assert row["status"] == "current"
    assert row["usable_cached"] is False
    assert coverage["latest_data_refresh"].startswith("2026-09-03T11:55:00")


def test_cache_inspection_does_not_reset_last_good_fetch_age() -> None:
    coverage = inspect_research_coverage(
        symbol="ABC",
        raw=_quarterly_raw(),
        autonomy={
            "dataset_meta": {
                "quarterly_results": {
                    "status": "current",
                    "provider": "cache",
                    "checked_at": "2026-09-03T11:59:00+00:00",
                    "fetched_at": "2026-04-01T09:15:00+00:00",
                }
            }
        },
        framework_id="generic",
        now=NOW,
    )

    row = _dataset(coverage, "quarterly_results")
    assert row["status"] == "stale"
    assert row["usable_cached"] is True
    assert row["checked_at"].startswith("2026-04-01T09:15:00")
    assert "quarterly_results" in coverage["to_fetch"]


def test_present_data_without_trustworthy_age_is_stale_not_fresh() -> None:
    coverage = inspect_research_coverage(
        symbol="ABC",
        raw=_quarterly_raw(),
        autonomy={},
        framework_id="generic",
        now=NOW,
    )

    row = _dataset(coverage, "quarterly_results")
    assert row["present"] is True
    assert row["status"] == "stale"
    assert row["usable_cached"] is True
    assert "source age unknown" in row["age_label"]
