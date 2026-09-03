from __future__ import annotations

from datetime import datetime, timezone

from product.due_diligence import acquire
from product.due_diligence.freshness import research_freshness


NOW = datetime(2026, 9, 3, 12, 0, tzinfo=timezone.utc)


def _patch_symbol(monkeypatch, *, facts: dict, datasets: list[dict]) -> None:
    monkeypatch.setattr(acquire, "shortlist_symbols", lambda **_kwargs: ["ABC"])
    monkeypatch.setattr(acquire, "load_autonomy_facts", lambda _symbol: facts)
    monkeypatch.setattr(
        acquire,
        "inspect_symbol_coverage",
        lambda _symbol, now=None: {
            "coverage_pct": 50,
            "summary": "partial",
            "datasets": datasets,
        },
    )


def test_recent_unrelated_dataset_check_does_not_delay_missing_dataset(monkeypatch) -> None:
    _patch_symbol(
        monkeypatch,
        facts={
            "dataset_meta": {
                "recent_news": {
                    "status": "current",
                    "checked_at": "2026-09-03T11:59:00+00:00",
                    "provider": "news_curator",
                }
            },
            "acquired_at": "2026-09-03T11:59:00+00:00",
        },
        datasets=[
            {
                "id": "quarterly_results",
                "label": "Quarterly results",
                "required": True,
                "status": "not_yet_acquired",
                "present": False,
                "checked_at": None,
                "provider": "screener.in",
            }
        ],
    )

    state = research_freshness(now=NOW, retry_cooldown_s=20 * 60)

    assert state["fresh"] is False
    assert state["retry_due"] is True
    assert state["state"] == "RETRY_DUE"
    problem = state["unresolved_datasets"][0]
    assert problem["id"] == "quarterly_results"
    assert problem["retry_due"] is True
    assert problem["cooldown_basis"] == "none"


def test_dataset_uses_its_own_recent_attempt_for_cooldown(monkeypatch) -> None:
    _patch_symbol(
        monkeypatch,
        facts={
            "dataset_meta": {
                "quarterly_results": {
                    "status": "acquisition_failed",
                    "checked_at": "2026-09-03T11:55:00+00:00",
                    "provider": "screener.in",
                    "error": "HTTP 503",
                }
            }
        },
        datasets=[
            {
                "id": "quarterly_results",
                "label": "Quarterly results",
                "required": True,
                "status": "acquisition_failed",
                "present": True,
                "checked_at": "2026-09-03T11:55:00+00:00",
                "provider": "screener.in",
            }
        ],
    )

    state = research_freshness(now=NOW, retry_cooldown_s=20 * 60)

    assert state["fresh"] is False
    assert state["retry_due"] is False
    assert state["state"] == "RETRY_COOLDOWN"
    problem = state["unresolved_datasets"][0]
    assert problem["retry_due"] is False
    assert problem["cooldown_basis"] == "dataset"
    assert problem["provider"] == "screener.in"


def test_coverage_inspection_failure_may_use_symbol_attempt_as_backoff(monkeypatch) -> None:
    monkeypatch.setattr(acquire, "shortlist_symbols", lambda **_kwargs: ["ABC"])
    monkeypatch.setattr(
        acquire,
        "load_autonomy_facts",
        lambda _symbol: {"acquired_at": "2026-09-03T11:55:00+00:00"},
    )

    def broken(_symbol, now=None):
        raise RuntimeError("local coverage store temporarily unreadable")

    monkeypatch.setattr(acquire, "inspect_symbol_coverage", broken)

    state = research_freshness(now=NOW, retry_cooldown_s=20 * 60)

    assert state["fresh"] is False
    assert state["retry_due"] is False
    assert state["state"] == "RETRY_COOLDOWN"
    problem = state["unresolved_datasets"][0]
    assert problem["id"] == "coverage_inspection"
    assert problem["cooldown_basis"] == "symbol_fallback"
