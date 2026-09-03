from __future__ import annotations

from datetime import datetime, timezone

from product.due_diligence import acquire as ACQ
from product.due_diligence import freshness as RF


NOW = datetime(2026, 9, 3, 12, 0, tzinfo=timezone.utc)


def _coverage(status: str = "current", *, checked_at: str | None = None, present: bool = True) -> dict:
    return {
        "coverage_pct": 100.0 if status == "current" else 80.0,
        "summary": "research coverage",
        "datasets": [
            {
                "id": "quarterly_results",
                "label": "Quarterly financials",
                "required": True,
                "present": present,
                "status": status,
                "checked_at": checked_at,
                "age_label": status,
            }
        ],
    }


def _wire(monkeypatch, *, symbols=("ABC",), coverage=None, facts=None):
    monkeypatch.setattr(ACQ, "shortlist_symbols", lambda **_kwargs: list(symbols))
    monkeypatch.setattr(ACQ, "inspect_symbol_coverage", lambda symbol, now=None: dict(coverage or _coverage()))
    monkeypatch.setattr(ACQ, "load_autonomy_facts", lambda symbol: dict(facts or {}))


def test_no_shortlist_is_fresh(monkeypatch):
    _wire(monkeypatch, symbols=())
    state = RF.research_freshness(now=NOW)
    assert state["fresh"] is True
    assert state["retry_due"] is False
    assert state["state"] == "NO_SHORTLIST"


def test_current_dataset_beats_old_attempt_timestamp(monkeypatch):
    _wire(
        monkeypatch,
        coverage=_coverage("current", checked_at="2026-09-03T11:59:00+00:00"),
        facts={"acquired_at": "2020-01-01T00:00:00+00:00"},
    )
    state = RF.research_freshness(now=NOW)
    assert state["fresh"] is True
    assert state["retry_due"] is False
    assert state["state"] == "CURRENT"


def test_recent_provider_failure_is_not_fresh_but_cools_down(monkeypatch):
    _wire(
        monkeypatch,
        coverage=_coverage("acquisition_failed", checked_at="2026-09-03T11:50:00+00:00", present=False),
    )
    state = RF.research_freshness(now=NOW, retry_cooldown_s=20 * 60)
    assert state["fresh"] is False
    assert state["retry_due"] is False
    assert state["state"] == "RETRY_COOLDOWN"
    assert state["unresolved_symbols"] == ["ABC"]
    assert state["unresolved_datasets"][0]["retry_at"].startswith("2026-09-03T12:10:00")


def test_old_provider_failure_becomes_retry_due(monkeypatch):
    _wire(
        monkeypatch,
        coverage=_coverage("source_unavailable", checked_at="2026-09-03T11:00:00+00:00", present=False),
    )
    state = RF.research_freshness(now=NOW, retry_cooldown_s=20 * 60)
    assert state["fresh"] is False
    assert state["retry_due"] is True
    assert state["state"] == "RETRY_DUE"


def test_never_checked_required_dataset_retries_immediately(monkeypatch):
    _wire(monkeypatch, coverage=_coverage("not_yet_acquired", checked_at=None, present=False))
    state = RF.research_freshness(now=NOW)
    assert state["fresh"] is False
    assert state["retry_due"] is True
    assert state["unresolved_datasets"][0]["retry_due"] is True


def test_current_coverage_cannot_hide_recent_failed_refresh_of_cached_data(monkeypatch):
    _wire(
        monkeypatch,
        coverage=_coverage("current", checked_at="2026-09-03T11:50:00+00:00", present=True),
        facts={
            "dataset_meta": {
                "quarterly_results": {
                    "status": "current",
                    "checked_at": "2026-09-03T11:50:00+00:00",
                    "provider": "screener.in",
                    "error": "HTTP 503 while refreshing; cached rows retained",
                }
            }
        },
    )
    state = RF.research_freshness(now=NOW, retry_cooldown_s=20 * 60)
    assert state["fresh"] is False
    assert state["retry_due"] is False
    assert state["state"] == "RETRY_COOLDOWN"
    problem = state["unresolved_datasets"][0]
    assert problem["status"] == "refresh_failed"
    assert problem["cached_data_present"] is True
    assert problem["truth_source"] == "dataset_meta"
    assert "503" in problem["refresh_error"]


def test_cached_refresh_failure_becomes_retry_due_after_cooldown(monkeypatch):
    _wire(
        monkeypatch,
        coverage=_coverage("current", checked_at="2026-09-03T11:00:00+00:00", present=True),
        facts={
            "dataset_meta": {
                "quarterly_results": {
                    "status": "current",
                    "checked_at": "2026-09-03T11:00:00+00:00",
                    "provider": "screener.in",
                    "error": "refresh timed out",
                }
            }
        },
    )
    state = RF.research_freshness(now=NOW, retry_cooldown_s=20 * 60)
    assert state["fresh"] is False
    assert state["retry_due"] is True
    assert state["state"] == "RETRY_DUE"


def test_successful_meta_without_error_remains_current(monkeypatch):
    _wire(
        monkeypatch,
        coverage=_coverage("current", checked_at="2026-09-03T11:55:00+00:00", present=True),
        facts={
            "dataset_meta": {
                "quarterly_results": {
                    "status": "current",
                    "checked_at": "2026-09-03T11:55:00+00:00",
                    "provider": "screener.in",
                }
            }
        },
    )
    state = RF.research_freshness(now=NOW)
    assert state["fresh"] is True
    assert state["state"] == "CURRENT"


def test_local_inspection_failure_fails_closed_without_hot_loop(monkeypatch):
    monkeypatch.setattr(ACQ, "shortlist_symbols", lambda **_kwargs: ["ABC"])
    monkeypatch.setattr(
        ACQ,
        "inspect_symbol_coverage",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("parser broke")),
    )
    monkeypatch.setattr(
        ACQ,
        "load_autonomy_facts",
        lambda _symbol: {"inspected_at": "2026-09-03T11:55:00+00:00"},
    )
    state = RF.research_freshness(now=NOW, retry_cooldown_s=20 * 60)
    assert state["fresh"] is False
    assert state["retry_due"] is False
    assert state["state"] == "RETRY_COOLDOWN"
    assert "RuntimeError" in state["symbols"][0]["inspection_error"]
