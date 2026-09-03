from __future__ import annotations

from product import desk_pipeline as DP


class _ReadStore:
    def active(self):
        return []

    def latest(self, _kind):
        return None


def _all_other_steps_fresh(monkeypatch):
    monkeypatch.setattr(DP, "prices_kind_due", lambda: None)
    monkeypatch.setattr(DP, "scan_is_fresh", lambda: True)
    monkeypatch.setattr(DP, "long_term_is_fresh", lambda: True)
    monkeypatch.setattr(DP, "news_is_fresh", lambda: True)


def test_incomplete_research_in_cooldown_does_not_requeue(monkeypatch):
    _all_other_steps_fresh(monkeypatch)
    monkeypatch.setattr(
        DP,
        "acquire_freshness",
        lambda: {
            "fresh": False,
            "retry_due": False,
            "state": "RETRY_COOLDOWN",
            "unresolved_symbols": ["ABC"],
            "next_retry_at": "2026-09-03T12:10:00+00:00",
            "reason": "provider cooling down",
        },
    )
    assert DP._kind_for_step("investigate") is None


def test_incomplete_research_requeues_only_when_retry_due(monkeypatch):
    _all_other_steps_fresh(monkeypatch)
    monkeypatch.setattr(
        DP,
        "acquire_freshness",
        lambda: {
            "fresh": False,
            "retry_due": True,
            "state": "RETRY_DUE",
            "unresolved_symbols": ["ABC"],
        },
    )
    assert DP._kind_for_step("investigate") == DP.DUE_DILIGENCE_ACQUIRE


def test_cooldown_is_not_painted_ready_in_pipeline_snapshot(monkeypatch):
    _all_other_steps_fresh(monkeypatch)
    monkeypatch.setattr(
        DP,
        "acquire_freshness",
        lambda: {
            "fresh": False,
            "retry_due": False,
            "state": "RETRY_COOLDOWN",
            "unresolved_symbols": ["ABC", "XYZ"],
            "next_retry_at": "2026-09-03T12:10:00+00:00",
            "reason": "provider cooling down",
        },
    )
    payload = DP._snapshot(_ReadStore(), queued_kind=None, queued_op=None, created=False)
    investigate = next(row for row in payload["steps"] if row["id"] == "investigate")
    assert investigate["state"] == "waiting"
    assert investigate["freshness_state"] == "RETRY_COOLDOWN"
    assert investigate["unresolved_symbols"] == 2
    assert "still incomplete" in payload["message"]
