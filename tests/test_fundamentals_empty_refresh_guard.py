"""A zero-row Screener HTTP 200 is not a successful fundamentals refresh.

Protect both existing last-good cache entries and the financial coverage gate.
No network requests are made by these regression tests.
"""
from __future__ import annotations

import pytest

from fundamentals import fetcher as F


class Cache:
    def __init__(self, data=None):
        self.data = dict(data) if data is not None else None
        self.writes = []

    def get(self, symbol, *, allow_stale=False):
        return dict(self.data) if self.data is not None else None

    def set(self, symbol, data):
        self.writes.append((symbol, dict(data)))
        self.data = dict(data)


class Scraper:
    def __init__(self, data):
        self.data = data
        self.calls = 0

    def fetch_all(self, symbol):
        self.calls += 1
        return dict(self.data)


def _wire(monkeypatch, cache, scraper, *, official=None):
    monkeypatch.setattr(F, "_cache", cache)
    monkeypatch.setattr(F, "_scraper", scraper)
    monkeypatch.setattr(F, "_official_warehouse_snapshot", lambda symbol: official)
    monkeypatch.setattr(F, "_try_official_backfill", lambda symbol: None)


def test_forced_empty_refresh_preserves_last_good_without_writing(monkeypatch):
    cache = Cache({"roe": 18.0, "source_label": "secondary_public"})
    scraper = Scraper({"about": "Empty company placeholder",
                       "key_ratios": [], "profit_loss": [],
                       "metadata": {"total_rows_scraped": 0}})
    _wire(monkeypatch, cache, scraper)

    result = F.get_deep_fundamentals("INFY", force_refresh=True)

    assert result["roe"] == 18.0
    assert result["source_label"] == "last_good_snapshot"
    assert result["stale"] is True
    assert cache.writes == []
    assert cache.data["roe"] == 18.0
    assert scraper.calls == 1


def test_empty_scrape_without_financials_is_explicit_failure(monkeypatch):
    cache = Cache()
    scraper = Scraper({"symbol": "INFY", "url": "https://example.test",
                       "metadata": {"total_rows_scraped": 0}})
    _wire(monkeypatch, cache, scraper)

    with pytest.raises(RuntimeError, match="no financial evidence"):
        F.get_deep_fundamentals("INFY", force_refresh=True)
    assert cache.writes == []


def test_previous_poisoned_fresh_cache_does_not_count_as_fundamentals(monkeypatch):
    cache = Cache({"about": "Placeholder", "profit_loss": [],
                   "metadata": {"total_rows_scraped": 0}})
    scraper = Scraper({"key_ratios": [{"name": "ROE", "value": "18%"}]})
    _wire(monkeypatch, cache, scraper)

    result = F.get_deep_fundamentals("INFY")

    assert result["key_ratios"][0]["name"] == "ROE"
    assert len(cache.writes) == 1
    assert scraper.calls == 1


def test_empty_secondary_cannot_erase_official_financial_tables(monkeypatch):
    official = {
        "quarterly_results": [{"row_label": "Sales", "Sep 2026": 100}],
        "source_label": "NSE official XBRL warehouse",
        "source_tier": "official", "official": True,
    }
    cache = Cache({"about": "Old enrichment"})
    scraper = Scraper({"key_ratios": [], "profit_loss": []})
    _wire(monkeypatch, cache, scraper, official=official)

    result = F.get_deep_fundamentals("INFY", force_refresh=True)

    assert result["official"] is True
    assert result["quarterly_results"] == official["quarterly_results"]
    assert result["secondary_refresh_error"]
    assert cache.writes[-1][1]["quarterly_results"] == official["quarterly_results"]


def test_valid_partial_financial_evidence_is_not_rejected(monkeypatch):
    cache = Cache()
    scraper = Scraper({"key_ratios": [{"name": "ROE", "value": "0%"}]})
    _wire(monkeypatch, cache, scraper)

    result = F.get_deep_fundamentals("INFY", force_refresh=True)

    assert result["key_ratios"][0]["value"] == "0%"
    assert len(cache.writes) == 1


def test_long_term_cache_only_path_rejects_legacy_empty_placeholder(monkeypatch):
    from fundamentals.cache import FundamentalsCache
    from scan.long_term_service import _default_fundamental_provider

    monkeypatch.setattr(
        FundamentalsCache, "get",
        lambda self, symbol, **kwargs: {"about": "HTML challenge",
                                        "profit_loss": [],
                                        "metadata": {"total_rows_scraped": 0}},
    )
    assert _default_fundamental_provider("INFY", refresh=False) is None


def test_long_term_cache_only_path_accepts_valid_financial_rows(monkeypatch):
    from fundamentals.cache import FundamentalsCache
    from scan.long_term_service import _default_fundamental_provider

    monkeypatch.setattr(
        FundamentalsCache, "get",
        lambda self, symbol, **kwargs: {
            "key_ratios": [{"name": "Stock P/E", "value": "25"}]
        },
    )
    result = _default_fundamental_provider("INFY", refresh=False)
    assert result["key_ratios"][0]["value"] == "25"
