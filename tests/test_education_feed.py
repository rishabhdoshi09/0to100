"""Education feed — crunch curated news into learnable cards without inventing content."""
from __future__ import annotations

from product.education_feed import build_education_feed, education_from_news_payload


def _article(**kwargs):
    base = {
        "article_id": "a1",
        "headline": "RBI holds repo rate as inflation cools",
        "summary": "Policy pause keeps rate-sensitive sectors in focus.",
        "why_it_matters": "Macro rates context for banks and realty — not a trade tip.",
        "category": "economy",
        "event_type": "macro",
        "tags": ["rbi", "rates"],
        "impact_score": 80,
        "direction": "unclear",
        "source": "RBI",
        "source_tier": 1,
        "official": True,
        "url": "https://example.com/rbi",
        "published_at": "2026-08-01T10:00:00+00:00",
        "mentioned_symbols": [],
        "fno_symbols": [],
        "sectors": [],
        "corroboration_count": 2,
    }
    base.update(kwargs)
    return base


def test_education_feed_includes_concepts_without_news():
    payload = build_education_feed(articles=[], include_concepts=True)
    assert payload["available"] is True
    assert payload["places_orders"] is False
    assert payload["summary"]["concepts"] >= 1
    assert any(c["kind"] == "CONCEPT" for c in payload["cards"])
    assert "never invents" in payload["honesty"].lower()
    assert payload["empty_hint"]


def test_education_classifies_macro_and_micro_lenses():
    articles = [
        _article(
            article_id="macro1",
            headline="RBI cuts repo rate as inflation eases",
            category="economy",
            event_type="macro",
            tags=["rbi"],
        ),
        _article(
            article_id="micro1",
            headline="Reliance Industries wins major order",
            category="company",
            event_type="order",
            tags=["order"],
            impact_score=75,
            official=False,
            mentioned_symbols=["RELIANCE"],
            url="https://example.com/ril",
        ),
        _article(
            article_id="noise",
            headline="Low impact chatter",
            impact_score=10,
            official=False,
            category="market",
        ),
    ]
    payload = build_education_feed(articles=articles, min_impact=40, include_concepts=False)
    lenses = {c["id"]: c["lens"] for c in payload["cards"] if c["kind"] == "NEWS_LESSON"}
    assert lenses["news-macro1"] == "MACRO"
    assert lenses["news-micro1"] == "MICRO"
    assert "news-noise" not in lenses
    micro = next(c for c in payload["cards"] if c["id"] == "news-micro1")
    assert micro["symbols"] == ["RELIANCE"]
    assert micro["url"] == "https://example.com/ril"
    assert micro["is_signal"] is False
    assert micro["places_orders"] is False


def test_education_from_news_payload_shape():
    payload = education_from_news_payload({
        "articles": [
            _article(article_id="sebi1", headline="SEBI tightens disclosure norms", category="regulation",
                     event_type="policy", tags=["sebi"], official=True),
        ],
    })
    assert payload["schema_version"] == 1
    assert payload["summary"]["articles_considered"] == 1
    assert any(c["lens"] == "POLICY" for c in payload["cards"] if c["kind"] == "NEWS_LESSON")
    assert "MACRO" in payload["lenses"]
    assert "MICRO" in payload["lenses"]


def test_education_never_fabricates_article_urls():
    payload = build_education_feed(articles=[], include_concepts=True)
    for card in payload["cards"]:
        if card["kind"] == "CONCEPT":
            assert card["url"] == ""
            assert card["source"] == "QuantTerm concept library"


def test_education_includes_gold_loan_and_mix_shift_concepts():
    payload = build_education_feed(articles=[], include_concepts=True)
    ids = {c["id"] for c in payload["cards"]}
    assert "concept-gold-loan-collateral" in ids
    assert "concept-mix-shift" in ids
