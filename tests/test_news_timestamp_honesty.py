"""news/fetcher.py must never fabricate a timestamp for an RSS entry that
doesn't carry one. Before this fix, an entry with no parseable
published_parsed/updated_parsed field was silently stamped "now" -- always
passing the staleness cutoff and displaying as freshly published. That let a
stale, re-syndicated, timestamp-less article corroborate core/macro_pulse's
theme-detection gate (or appear in any other staleness-gated consumer) as if
it were current news.
"""
from __future__ import annotations

import time
import types
from datetime import datetime, timedelta, timezone

from news.fetcher import NewsFetcher


class _FakeResponse:
    status_code = 200
    content = b""

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return None


def test_parse_entry_time_returns_none_without_a_genuine_timestamp():
    assert NewsFetcher._parse_entry_time({"title": "no dates here"}) is None


def test_parse_entry_time_returns_the_real_published_time():
    struct = time.gmtime(1_700_000_000)
    out = NewsFetcher._parse_entry_time({"published_parsed": struct})
    assert out == datetime.fromtimestamp(1_700_000_000, tz=timezone.utc)


def test_parse_entry_time_falls_back_to_updated_parsed():
    struct = time.gmtime(1_700_000_000)
    out = NewsFetcher._parse_entry_time({"updated_parsed": struct})
    assert out == datetime.fromtimestamp(1_700_000_000, tz=timezone.utc)


def test_fetch_rss_drops_entries_with_no_genuine_timestamp(monkeypatch):
    import feedparser
    import requests

    now = datetime.now(timezone.utc)
    fresh_struct = time.gmtime((now - timedelta(hours=1)).timestamp())
    stale_struct = time.gmtime((now - timedelta(hours=100)).timestamp())
    entries = [
        {"title": "Has a real, fresh timestamp", "published_parsed": fresh_struct, "link": "http://x/1"},
        {"title": "Has a real, STALE timestamp", "published_parsed": stale_struct, "link": "http://x/2"},
        {"title": "No timestamp at all", "link": "http://x/3"},
    ]
    fake_feed = types.SimpleNamespace(entries=entries, feed={"title": "Test Feed"}, bozo=False)

    monkeypatch.setattr(requests, "get", lambda *a, **k: _FakeResponse())
    monkeypatch.setattr(feedparser, "parse", lambda *a, **k: fake_feed)

    articles = NewsFetcher()._fetch_rss("http://example.com/rss", max_age_hours=24)
    headlines = {a.headline for a in articles}

    assert "Has a real, fresh timestamp" in headlines
    assert "Has a real, STALE timestamp" not in headlines
    assert "No timestamp at all" not in headlines, (
        "an entry with no genuine timestamp must never be silently treated as fresh"
    )
