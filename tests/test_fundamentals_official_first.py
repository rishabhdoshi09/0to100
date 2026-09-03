from __future__ import annotations

import fundamentals.fetcher as fetcher


class FakeCache:
    def __init__(self, fresh=None, stale=None):
        self.fresh = fresh
        self.stale = stale if stale is not None else fresh
        self.saved = None

    def get(self, symbol, allow_stale=False):
        return self.stale if allow_stale else self.fresh

    def set(self, symbol, data):
        self.saved = dict(data)


class FakeScraper:
    def __init__(self, data=None, error=None):
        self.data = data or {}
        self.error = error
        self.calls = 0

    def fetch_all(self, symbol):
        self.calls += 1
        if self.error:
            raise self.error
        return dict(self.data)


def official_pack():
    return {
        "quarterly_results": [{"row_label": "Sales+", "Jun 2026": 100}],
        "profit_loss": [{"row_label": "Sales+", "Mar 2026": 400}],
        "source_label": "NSE official XBRL warehouse",
        "source_tier": "official",
        "official": True,
        "latest_publication": "2026-07-20",
        "debt_to_equity": 0.3,
    }


def test_official_tables_override_secondary_but_keep_enrichment(monkeypatch):
    cache = FakeCache(fresh={
        "about": "Useful company description",
        "quarterly_results": [{"row_label": "Sales+", "Jun 2026": 999}],
        "pe": 25,
    })
    scraper = FakeScraper(error=AssertionError("should not scrape"))
    monkeypatch.setattr(fetcher, "_cache", cache)
    monkeypatch.setattr(fetcher, "_scraper", scraper)
    monkeypatch.setattr(fetcher, "_official_warehouse_snapshot", lambda symbol: official_pack())

    data = fetcher.get_deep_fundamentals("INFY")

    assert data["quarterly_results"][0]["Jun 2026"] == 100
    assert data["about"] == "Useful company description"
    assert data["pe"] == 25
    assert data["official"] is True
    assert data["source_tier"] == "official"
    assert scraper.calls == 0


def test_official_warehouse_avoids_unnecessary_scrape_without_fresh_cache(monkeypatch):
    cache = FakeCache(fresh=None, stale={"about": "Old but useful enrichment"})
    scraper = FakeScraper(error=AssertionError("should not scrape"))
    monkeypatch.setattr(fetcher, "_cache", cache)
    monkeypatch.setattr(fetcher, "_scraper", scraper)
    monkeypatch.setattr(fetcher, "_official_warehouse_snapshot", lambda symbol: official_pack())

    data = fetcher.get_deep_fundamentals("INFY")

    assert data["official"] is True
    assert data["about"] == "Old but useful enrichment"
    assert cache.saved["source_tier"] == "official"
    assert scraper.calls == 0


def test_forced_secondary_refresh_cannot_overwrite_official_financials(monkeypatch):
    cache = FakeCache(fresh=None, stale=None)
    scraper = FakeScraper(data={
        "about": "Fresh description",
        "quarterly_results": [{"row_label": "Sales+", "Jun 2026": 777}],
    })
    monkeypatch.setattr(fetcher, "_cache", cache)
    monkeypatch.setattr(fetcher, "_scraper", scraper)
    monkeypatch.setattr(fetcher, "_official_warehouse_snapshot", lambda symbol: official_pack())

    data = fetcher.get_deep_fundamentals("INFY", force_refresh=True)

    assert scraper.calls == 1
    assert data["about"] == "Fresh description"
    assert data["quarterly_results"][0]["Jun 2026"] == 100
    assert data["official"] is True


def test_secondary_failure_falls_back_to_official(monkeypatch):
    cache = FakeCache(fresh=None, stale=None)
    scraper = FakeScraper(error=RuntimeError("secondary down"))
    monkeypatch.setattr(fetcher, "_cache", cache)
    monkeypatch.setattr(fetcher, "_scraper", scraper)
    monkeypatch.setattr(fetcher, "_official_warehouse_snapshot", lambda symbol: official_pack())

    data = fetcher.get_deep_fundamentals("INFY", force_refresh=True)

    assert data["official"] is True
    assert "secondary down" in data["secondary_refresh_error"]
