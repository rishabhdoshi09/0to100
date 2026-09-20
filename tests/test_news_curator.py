from datetime import datetime, timezone
from pathlib import Path
import time

from news.curator import EntityResolver, curate_articles
from news.curator_models import FetchedNews
from news.curator_store import NewsCuratorStore
from news.source_catalog import default_sources


def _item(headline, *, source="wire", tier=2, official=False, hinted=()):
    return FetchedNews(
        headline=headline,
        summary="Detailed market update",
        source_key=source,
        source_name=source.title(),
        url=f"https://example.com/{source}",
        published_at=datetime.now(timezone.utc),
        category_hint="market",
        source_tier=tier,
        official=official,
        hinted_symbols=tuple(hinted),
    )


def test_entity_resolver_maps_full_symbol_and_fno():
    resolver = EntityResolver(
        {"RELIANCE": "Reliance Industries Limited", "TCS": "Tata Consultancy Services Limited"},
        {"RELIANCE"},
    )
    symbols, fno = resolver.resolve("Reliance Industries wins a large contract; TCS also rises")
    assert symbols == ("RELIANCE", "TCS")
    assert fno == ("RELIANCE",)


def test_entity_resolver_preserves_symbol_boundaries_and_alias_matching():
    resolver = EntityResolver(
        {
            "ABC": "Alpha Beta Corporation Limited",
            "ABCD": "Another Business Company Limited",
            "M&M": "Mahindra Mahindra Limited",
        },
        {"M&M"},
    )
    symbols, fno = resolver.resolve(
        "Alpha Beta Corporation wins an order while ABCD rises; M&M also gains"
    )
    assert symbols == ("ABC", "ABCD", "M&M")
    assert fno == ("M&M",)


def test_entity_resolver_scales_to_production_sized_universe_without_per_symbol_regex_loop():
    symbols = {f"SYM{i:05d}": f"Company {i:05d} Industries Limited" for i in range(10500)}
    symbols["RELIANCE"] = "Reliance Industries Limited"
    resolver = EntityResolver(symbols, {"RELIANCE"})

    started = time.monotonic()
    for _ in range(100):
        found, fno = resolver.resolve("Reliance Industries wins a large order; RELIANCE rises")
        assert "RELIANCE" in found
        assert fno == ("RELIANCE",)
    elapsed = time.monotonic() - started

    # This is deliberately generous and only catches the old O(symbols x articles)
    # implementation.  It is not a microbenchmark; it is a production-scale
    # anti-regression budget for the operator-visible NEWS_REFRESH path.
    assert elapsed < 5.0


def test_curator_deduplicates_and_corroborates_same_story():
    resolver = EntityResolver({"RELIANCE": "Reliance Industries Limited"}, {"RELIANCE"})
    rows = curate_articles([
        _item("Reliance Industries wins major order", source="official", tier=1, official=True),
        _item("Reliance Industries wins major order - Reuters", source="media", tier=2),
    ], resolver)
    assert len(rows) == 1
    assert rows[0].corroboration_count == 2
    assert rows[0].official is True
    assert rows[0].impact_score >= 70
    assert rows[0].fno_symbols == ("RELIANCE",)


def test_macro_regulatory_story_is_ranked_and_explained():
    resolver = EntityResolver({}, set())
    row = curate_articles([
        _item("RBI cuts repo rate as inflation eases", source="rbi", tier=1, official=True)
    ], resolver)[0]
    assert row.category == "economy"
    assert row.event_type == "macro"
    assert "rates" in row.why_it_matters.lower() or "macro" in row.why_it_matters.lower()
    assert row.direction == "likely_positive"


def test_store_is_idempotent_and_filters_fno(tmp_path: Path):
    resolver = EntityResolver({"RELIANCE": "Reliance Industries Limited"}, {"RELIANCE"})
    article = curate_articles([
        _item("RELIANCE wins order", source="nse", tier=1, official=True, hinted=("RELIANCE",))
    ], resolver)[0]
    store = NewsCuratorStore(tmp_path / "news.sqlite3")
    assert store.upsert_articles([article]) == 1
    assert store.upsert_articles([article]) == 1
    assert store.stats(24)["total"] == 1
    rows = store.recent(hours=24, fno_only=True)
    assert len(rows) == 1
    assert rows[0].article_id == article.article_id


def test_source_catalog_contains_official_and_discovery_feeds():
    sources = default_sources(["https://example.com/market.xml"])
    keys = {source.key for source in sources}
    assert {"nse_announcements", "sebi", "rbi_press", "pib_releases"}.issubset(keys)
    assert any(source.key.startswith("google_") for source in sources)
    assert len({source.url for source in sources}) == len(sources)
