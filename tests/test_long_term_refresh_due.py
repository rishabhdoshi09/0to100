"""The official-filing acquisition pass (LONG_TERM_REFRESH) must come due.

Audit 2026-11-11: the cache-only long-term scan rewrites
latest_long_term_scan.json after every market scan, and the desk pipeline
judged the refresh by that file's age, so the refresh never ran and the
official warehouse stayed at 375 symbols.
"""
from __future__ import annotations

import json
import os
import time


def _product_dir():
    from core.runtime_paths import logs_dir

    path = logs_dir() / "product"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _touch(path, age_s=0.0):
    path.write_text("{}")
    stamp = time.time() - age_s
    os.utime(path, (stamp, stamp))


def test_fresh_shortlist_without_a_refresh_is_not_fresh():
    from product.desk_pipeline import long_term_is_fresh

    d = _product_dir()
    (d / "long_term_fundamentals_refresh.json").unlink(missing_ok=True)
    _touch(d / "latest_long_term_scan.json")
    assert long_term_is_fresh() is False


def test_recent_refresh_and_shortlist_are_fresh():
    from product.desk_pipeline import long_term_is_fresh

    d = _product_dir()
    _touch(d / "latest_long_term_scan.json")
    _touch(d / "long_term_fundamentals_refresh.json")
    assert long_term_is_fresh() is True


def test_old_refresh_comes_due_again():
    from operations.market_ops import LONG_TERM_FRESH_S
    from product.desk_pipeline import long_term_is_fresh

    d = _product_dir()
    _touch(d / "latest_long_term_scan.json")
    _touch(d / "long_term_fundamentals_refresh.json", age_s=LONG_TERM_FRESH_S + 60)
    assert long_term_is_fresh() is False


def test_refresh_pass_writes_its_marker_and_cache_scan_does_not():
    from scan.long_term_service import fundamentals_refresh_marker, run_long_term_scan

    marker = fundamentals_refresh_marker()
    marker.unlink(missing_ok=True)
    rows = [{"symbol": "ZZTEST", "score": 70.0, "price": 100.0, "signals": []}]
    scanner = lambda **_kw: list(rows)
    run_long_term_scan(symbols=["ZZTEST"], refresh_fundamentals=False,
                       technical_scanner=scanner, save=True)
    assert not marker.exists()
    run_long_term_scan(symbols=["ZZTEST"], refresh_fundamentals=True,
                       technical_scanner=scanner, save=True)
    data = json.loads(marker.read_text())
    assert data["refreshed_at"] and data["candidates"] == 1
