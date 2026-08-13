"""Retail freshness helpers — scan age must be human, not raw ISO-only."""
from __future__ import annotations


def test_relative_age_js_contract_mirror():
    """Python-side mirror of frontend relativeAge buckets (contract sanity)."""
    from datetime import datetime, timedelta, timezone

    now = datetime(2026, 8, 13, 10, 0, tzinfo=timezone.utc)

    def relative_age(stamp: datetime) -> str:
        sec = max(0, int((now - stamp).total_seconds()))
        if sec < 60:
            return f"{sec}s ago"
        if sec < 3600:
            return f"{round(sec / 60)} min ago"
        if sec < 86400:
            return f"{round(sec / 3600)} hr ago"
        days = round(sec / 86400)
        return "yesterday" if days == 1 else f"{days} days ago"

    assert relative_age(now - timedelta(seconds=20)).endswith("s ago")
    assert relative_age(now - timedelta(minutes=12)) == "12 min ago"
    assert relative_age(now - timedelta(hours=3)) == "3 hr ago"
    assert relative_age(now - timedelta(days=1)) == "yesterday"
