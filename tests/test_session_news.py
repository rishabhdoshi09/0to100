"""The wrap of the day reads the current session's news, never last week's
(owner's Mac 2026-11-16: the 7-day top-by-impact list held 6 Friday-onward
articles of 120, so the daily report led with Tuesday's stories)."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import product.desk_note as DN

IST = ZoneInfo("Asia/Kolkata")


def test_window_starts_at_the_last_completed_session():
    monday_night = datetime(2026, 11, 16, 1, 0, tzinfo=IST)
    assert DN.session_window_start(monday_night) == datetime(2026, 11, 12, 18, 30, tzinfo=timezone.utc)  # Fri 00:00 IST
    monday_after_close = datetime(2026, 11, 16, 16, 0, tzinfo=IST)
    assert DN.session_window_start(monday_after_close) == datetime(2026, 11, 15, 18, 30, tzinfo=timezone.utc)


def test_fallback_list_is_filtered_to_the_window(tmp_path, monkeypatch):
    monkeypatch.setattr(DN, "_SESSION_NEWS_CACHE", {"key": None, "at": 0.0, "rows": None})
    now = datetime(2026, 11, 16, 1, 0, tzinfo=IST)
    articles = [
        {"article_id": "old", "headline": "Tuesday story", "published_at": "2026-11-10T09:00:00+00:00"},
        {"article_id": "fri", "headline": "Friday story", "published_at": "2026-11-13T09:00:00+00:00"},
        {"article_id": "sun", "headline": "Sunday story", "published_at": "2026-11-15T09:00:00Z"},
        {"article_id": "nodate", "headline": "Undated"},
    ]
    got = DN.session_news_articles(articles, now=now, store_path=tmp_path / "missing.sqlite3")
    assert [a["article_id"] for a in got] == ["fri", "sun"]


def test_store_window_reaches_fresh_articles_the_top_120_missed(tmp_path, monkeypatch):
    from news.curator_models import CuratedArticle
    from news.curator_store import NewsCuratorStore

    monkeypatch.setattr(DN, "_SESSION_NEWS_CACHE", {"key": None, "at": 0.0, "rows": None})
    now = datetime.now(IST)
    start = DN.session_window_start(now)

    def art(aid, when, impact):
        return CuratedArticle(article_id=aid, cluster_key=aid, headline=f"h-{aid}", summary="", source="S",
                              source_key="s", source_tier=1, official=False, url=f"https://x/{aid}",
                              published_at=when.isoformat(), fetched_at=when.isoformat(), category="markets",
                              event_type="news", impact_score=impact, direction="neutral", why_it_matters="")

    path = tmp_path / "news.sqlite3"
    store = NewsCuratorStore(path)
    store.upsert_articles([
        art("stale-big", start - timedelta(days=3), 99),
        art("fresh", start + timedelta(minutes=30), 60),
    ])
    store.close()
    stale_only = [{"article_id": "stale-big", "headline": "h", "published_at": (start - timedelta(days=3)).isoformat()}]
    got = DN.session_news_articles(stale_only, now=now, store_path=path)
    assert [a["article_id"] for a in got] == ["fresh"]
    # Cached per window and store identity: a second read does not reopen the store.
    monkeypatch.setattr("news.curator_store.NewsCuratorStore", None)
    assert [a["article_id"] for a in DN.session_news_articles([], now=now, store_path=path)] == ["fresh"]


def test_both_wrap_callers_use_the_session_window():
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    assert "session_news_articles(news.get(\"articles\")" in (root / "terminal_api.py").read_text()
    assert "session_news_articles((news_payload" in (root / "product" / "recommendations_workspace.py").read_text()
