from __future__ import annotations

import fundamentals.fetcher as fetcher


class FakeCache:
    def __init__(self, *, fresh=None, last_good=None):
        self.fresh = fresh
        self.last_good = last_good
        self.written = None

    def get(self, symbol):
        return self.fresh

    def get_any(self, symbol):
        return self.last_good

    def set(self, symbol, data):
        self.written = (symbol, data)

    def age_seconds(self, symbol):
        return 172800.0


class FakeScraper:
    def __init__(self, result=None, error=None):
        self.result = result
        self.error = error
        self.calls = 0

    def fetch_all(self, symbol):
        self.calls += 1
        if self.error:
            raise self.error
        return dict(self.result or {})


def test_fresh_cache_does_not_hit_internet(monkeypatch):
    cache = FakeCache(fresh={"symbol": "TCS", "quarterly_results": [1]})
    scraper = FakeScraper(error=RuntimeError("must not run"))
    monkeypatch.setattr(fetcher, "_cache", cache)
    monkeypatch.setattr(fetcher, "_scraper", scraper)
    result = fetcher.get_deep_fundamentals("TCS")
    assert scraper.calls == 0
    assert result["_delivery"]["state"] == "FRESH_CACHE"
    assert result["_delivery"]["source_tier"] == "reputable_secondary"


def test_successful_scrape_replaces_cache(monkeypatch):
    cache = FakeCache(last_good={"symbol": "TCS", "old": True})
    scraper = FakeScraper(result={"symbol": "TCS", "quarterly_results": [{"Sales": 1}]})
    monkeypatch.setattr(fetcher, "_cache", cache)
    monkeypatch.setattr(fetcher, "_scraper", scraper)
    result = fetcher.get_deep_fundamentals("TCS", force_refresh=True)
    assert result["_delivery"]["state"] == "FRESH_SECONDARY"
    assert cache.written is not None
    assert cache.written[0] == "TCS"


def test_scrape_failure_serves_stale_last_good(monkeypatch):
    cache = FakeCache(last_good={"symbol": "TCS", "quarterly_results": [{"Sales": 1}], "_delivery": {"source": "Screener.in", "source_tier": "reputable_secondary"}})
    scraper = FakeScraper(error=RuntimeError("provider down"))
    monkeypatch.setattr(fetcher, "_cache", cache)
    monkeypatch.setattr(fetcher, "_scraper", scraper)
    result = fetcher.get_deep_fundamentals("TCS", force_refresh=True)
    assert result["quarterly_results"]
    assert result["_delivery"]["state"] == "STALE_LAST_GOOD"
    assert result["_delivery"]["stale"] is True
    assert "provider down" in result["_delivery"]["refresh_error"]
    assert result["_delivery"]["cache_age_seconds"] == 172800.0


def test_no_cache_and_scrape_failure_preserves_real_error(monkeypatch):
    cache = FakeCache(last_good=None)
    scraper = FakeScraper(error=RuntimeError("provider down"))
    monkeypatch.setattr(fetcher, "_cache", cache)
    monkeypatch.setattr(fetcher, "_scraper", scraper)
    try:
        fetcher.get_deep_fundamentals("TCS", force_refresh=True)
    except RuntimeError as exc:
        assert "provider down" in str(exc)
    else:
        raise AssertionError("expected provider failure when no last-good evidence exists")
