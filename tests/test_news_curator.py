from datetime import datetime, timezone
from pathlib import Path
import time

from news.curator import EntityResolver, NewsCurator, curate_articles
from news.curator_models import FetchedNews, SourceSpec
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


def test_entity_resolver_matches_special_nse_symbols():
    resolver = EntityResolver(
        {"M&M": "Mahindra and Mahindra Limited", "RELIANCE": "Reliance Industries Limited"},
        {"M&M"},
    )
    symbols, fno = resolver.resolve("M&M tractor sales beat estimates; RELIANCE also gains")
    assert symbols == ("M&M", "RELIANCE")
    assert fno == ("M&M",)


def test_entity_resolver_scales_to_full_universe():
    names = {f"SYM{i:04d}": f"Company {i} Industries Limited" for i in range(2500)}
    names["RELIANCE"] = "Reliance Industries Limited"
    names["M&M"] = "Mahindra and Mahindra Limited"
    resolver = EntityResolver(names, {"RELIANCE", "M&M"})
    headlines = [
        "RELIANCE wins a large order as markets rally",
        "M&M tractor sales beat estimates this quarter",
        "Unrelated macro update on inflation and the rupee",
    ] * 150
    started = time.monotonic()
    hits = [resolver.resolve(text) for text in headlines]
    assert time.monotonic() - started < 1.5
    assert hits[0] == (("RELIANCE",), ("RELIANCE",))
    assert hits[1][0] == ("M&M",) and hits[1][1] == ("M&M",)
    assert hits[2] == ((), ())


def test_refresh_respects_fetch_budget(tmp_path: Path):
    spec = SourceSpec(key="slow", name="Slow Wire", url="https://example.com/rss", category="market")
    curator = NewsCurator(
        sources=[spec],
        store=NewsCuratorStore(tmp_path / "news.sqlite3"),
        resolver=EntityResolver({}, set()),
    )

    def hang(_source, _max_age_hours):
        time.sleep(1)
        raise AssertionError("hung fetch should have been abandoned")

    curator._fetch_source = hang  # type: ignore[method-assign]
    started = time.monotonic()
    report = curator.refresh(budget_s=0.2)
    assert time.monotonic() - started < 1.5
    assert any("budget" in error.lower() for error in report.errors)
