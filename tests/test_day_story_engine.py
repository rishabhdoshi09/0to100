"""Day-story engine — ranks multi-site news into lively wrap bullets."""
from __future__ import annotations

from datetime import datetime, timezone

from news.curator import curate_articles
from news.curator_models import FetchedNews
from news.day_story_engine import (
    build_day_stories,
    select_day_stories,
    wrap_line_from_article,
    wrap_relevance_score,
)
from news.source_catalog import default_sources


def _item(headline, *, source="moneycontrol", tier=2, official=False, summary=""):
    return FetchedNews(
        headline=headline,
        summary=summary or "Detailed market update from the wire",
        source_key=source,
        source_name=source.title(),
        url=f"https://example.com/{source}/{abs(hash(headline)) % 10_000}",
        published_at=datetime.now(timezone.utc),
        category_hint="company",
        source_tier=tier,
        official=official,
        hinted_symbols=(),
    )


def test_retail_media_catalog_includes_moneycontrol_and_ipo_discovery():
    sources = default_sources()
    keys = {s.key for s in sources}
    assert "moneycontrol_markets" in keys
    assert "et_ipo" in keys
    assert "google_ipo" in keys
    assert any("moneycontrol.com" in s.url for s in sources)


def test_listing_and_order_book_outrank_sebi_paperwork():
    rows = curate_articles([
        _item("Manipal Hospitals shares fall after strong listing as investors book profits"),
        _item(
            "SEBI issues master circular on intermediary compliance",
            source="sebi",
            tier=1,
            official=True,
            summary="Regulatory circular",
        ),
        _item("HAL highlights ₹2.55 lakh crore order book; defence manufacturing in focus"),
        _item("Neuland Laboratories reports sharp jump in profit and revenue"),
    ])
    stories = select_day_stories(rows, limit=4)
    heads = " | ".join(s.headline for s in stories)
    assert "Manipal" in heads or "HAL" in heads or "Neuland" in heads
    assert stories[0].wrap_score >= wrap_relevance_score(rows[-1])
    # Paperwork should not lead the wrap when company day stories exist.
    assert "master circular" not in stories[0].headline.lower()


def test_wrap_line_is_newsletter_style_not_robotic():
    article = curate_articles([
        _item(
            "HAL rallies after highlighting a large defence order book",
            summary="Hindustan Aeronautics pointed to a robust order book supporting multi-year visibility.",
        )
    ])[0]
    line = wrap_line_from_article(article)
    assert "HAL" in line or "order" in line.lower()
    assert "In the news:" not in line
    assert line.endswith(".")
    assert len(line) > 40


def test_build_day_stories_from_store(tmp_path, monkeypatch):
    from news.curator_store import NewsCuratorStore

    db = tmp_path / "news.sqlite3"
    store = NewsCuratorStore(db)
    rows = curate_articles([
        _item("US futures edge higher after Dow hits record high"),
        _item("Tata Sons IPO debate returns as NBFC classification stays in focus"),
    ])
    store.upsert_articles(rows)
    store.close()

    payload = build_day_stories(hours=24, limit=3, path=db, refresh_if_stale=False)
    assert payload["available"] is True
    assert payload["count"] >= 1
    assert payload["stories"][0]["wrap_line"]
