from __future__ import annotations

import json
import time

from fundamentals.cache import FundamentalsCache


def test_stale_snapshot_is_not_deleted(tmp_path, monkeypatch):
    monkeypatch.setattr("fundamentals.cache._DB_PATH", tmp_path / "fund.db")
    cache = FundamentalsCache()
    cache.set("TCS", {"pe": 22, "source_label": "secondary_public"})
    # Age the row past TTL.
    import sqlite3
    from fundamentals import cache as cache_mod
    with sqlite3.connect(cache_mod._DB_PATH) as conn:
        conn.execute(
            "UPDATE fundamentals_cache SET fetched_at = ? WHERE symbol = ?",
            (time.time() - 10 * 86_400, "TCS"),
        )
        conn.commit()
    assert cache.get("TCS") is None
    last = cache.get("TCS", allow_stale=True)
    assert last is not None
    assert last["pe"] == 22
    assert last["stale"] is True
    assert last["source_label"] == "last_good_snapshot"
    # Housekeeping must not wipe a 10-day-old last-good snapshot.
    deleted = cache.clear_old()
    assert deleted == 0
    assert cache.get("TCS", allow_stale=True)["pe"] == 22


def test_fetcher_falls_back_to_last_good_when_scrape_fails(tmp_path, monkeypatch):
    monkeypatch.setattr("fundamentals.cache._DB_PATH", tmp_path / "fund.db")
    from fundamentals import fetcher as F

    cache = FundamentalsCache()
    cache.set("INFY", {"roe": 18.0, "source_label": "secondary_public"})
    import sqlite3
    from fundamentals import cache as cache_mod
    with sqlite3.connect(cache_mod._DB_PATH) as conn:
        conn.execute(
            "UPDATE fundamentals_cache SET fetched_at = ? WHERE symbol = ?",
            (time.time() - 5 * 86_400, "INFY"),
        )
        conn.commit()

    class Boom:
        def fetch_all(self, symbol):
            raise RuntimeError("screener down")

    monkeypatch.setenv("QT_OFFLINE", "1")
    monkeypatch.setattr(F, "_cache", cache)
    monkeypatch.setattr(F, "_scraper", Boom())
    # Last-good is only reachable when official warehouse/backfill are absent.
    # A live warehouse (or network) must not leak into this unit test.
    monkeypatch.setattr(F, "_official_warehouse_snapshot", lambda symbol: None)
    monkeypatch.setattr(F, "_try_official_backfill", lambda symbol: None)
    data = F.get_deep_fundamentals("INFY", force_refresh=True)
    assert data["roe"] == 18.0
    assert data["source_label"] == "last_good_snapshot"
    assert data["official"] is False
    assert data["stale"] is True


def test_official_warehouse_wins_after_secondary_failure_without_becoming_last_good(tmp_path, monkeypatch):
    monkeypatch.setattr("fundamentals.cache._DB_PATH", tmp_path / "fund.db")
    from fundamentals import fetcher as F

    cache = FundamentalsCache()
    cache.set("INFY", {"roe": 18.0, "source_label": "secondary_public"})

    class Boom:
        def fetch_all(self, symbol):
            raise RuntimeError("screener down")

    official = {
        "roe": 383.95,
        "source_label": "NSE official XBRL warehouse",
        "source_tier": "official",
        "official": True,
        "quarterly_results": [{"period": "FY26Q1"}],
    }
    monkeypatch.setenv("QT_OFFLINE", "1")
    monkeypatch.setattr(F, "_cache", cache)
    monkeypatch.setattr(F, "_scraper", Boom())
    monkeypatch.setattr(F, "_official_warehouse_snapshot", lambda symbol: dict(official))
    monkeypatch.setattr(F, "_try_official_backfill", lambda symbol: None)
    data = F.get_deep_fundamentals("INFY", force_refresh=True)
    assert data["official"] is True
    assert data["source_label"] == "NSE official XBRL warehouse"
    assert data["source_label"] != "last_good_snapshot"
    assert data["roe"] == 383.95
    assert data.get("secondary_refresh_error")
