"""IST calendar-day freshness for fundamentals and scan snapshots."""
from __future__ import annotations

import json
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path


def test_fundamentals_fresh_only_on_ist_today(tmp_path, monkeypatch):
    import fundamentals.cache as cache_mod
    from core.market_clock import IST, today_ist

    db = tmp_path / "fundamentals_cache.db"
    monkeypatch.setattr(cache_mod, "_DB_PATH", db)
    cache = cache_mod.FundamentalsCache()
    cache.set("AAA", {"about": "today co"})
    assert cache.get("AAA") is not None
    assert cache.get("AAA")["_qt_cache_status"] == "TODAY"

    # Backdate fetched_at to yesterday IST → get() must miss; get_any STALE.
    yesterday = (today_ist() - timedelta(days=1))
    y_ts = datetime(yesterday.year, yesterday.month, yesterday.day, 12, 0, tzinfo=IST).timestamp()
    import sqlite3
    with sqlite3.connect(db) as conn:
        conn.execute(
            "UPDATE fundamentals_cache SET fetched_at=? WHERE symbol=?",
            (y_ts, "AAA"),
        )
        conn.commit()
    assert cache.get("AAA") is None
    stale = cache.get_any("AAA")
    assert stale is not None
    assert stale["_qt_cache_status"] == "STALE"
    stats = cache.stats()
    assert stats["freshness"] == "ist_calendar_day"
    assert stats["symbols_fresh"] == 0
    assert stats["symbols_stale"] == 1


def test_load_scan_marks_prior_day_snapshot(tmp_path):
    from product.scan_store import load_scan, save_scan

    path = tmp_path / "scan.json"
    old = (datetime.now(timezone.utc) - timedelta(days=2)).isoformat()
    save_scan(
        {
            "schema_version": 1,
            "scanned_at": old,
            "universe_size": 1,
            "records": [{"symbol": "X", "signals": [], "status": "Watch", "score": 1}],
            "summary": {},
        },
        path,
    )
    payload = load_scan(path)
    assert payload is not None
    assert payload["same_ist_day"] is False
    assert payload["records_status"] == "PRIOR_DAY_SNAPSHOT"


def test_load_scan_marks_current_day(tmp_path):
    from product.scan_store import load_scan, save_scan

    path = tmp_path / "scan.json"
    save_scan(
        {
            "schema_version": 1,
            "scanned_at": datetime.now(timezone.utc).isoformat(),
            "universe_size": 1,
            "records": [{"symbol": "Y", "signals": [], "status": "Watch", "score": 1}],
            "summary": {},
        },
        path,
    )
    payload = load_scan(path)
    assert payload is not None
    assert payload["same_ist_day"] is True
    assert payload["records_status"] == "CURRENT_DAY"


def test_instruments_cache_uses_ist_day(tmp_path, monkeypatch):
    from data import instruments as inst

    cache_file = tmp_path / "instruments_cache.csv"
    cache_file.write_text(
        "instrument_token,exchange_token,tradingsymbol,name,last_price,expiry,strike,"
        "tick_size,lot_size,instrument_type,segment,exchange\n"
        "1,1,RELIANCE,Reliance,0,,,0.05,1,EQ,NSE,NSE\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(inst, "_CACHE_FILE", cache_file)
    # Fresh mtime → today
    assert inst._cache_is_today() is True
    # Backdate mtime to 2 days ago
    old = time.time() - 2 * 86400
    import os
    os.utime(cache_file, (old, old))
    assert inst._cache_is_today() is False
