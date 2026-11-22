"""feature002 records WHICH history each shadow row was computed from (Cycle 7).

The relay review of 011dffb0 found that feature002 rows for young listings
matched neither price store: its reader (scan.bulk_fetcher.get_cached) serves
the active Kite snapshot while Kite is authoritative, the official bhavcopy
store otherwise, and the yfinance cache last. New rows now persist their input
source, snapshot id and per-symbol as-of; bars after the session cutoff are
dropped; a row without the record stays 'unknown', never inferred.
"""
from __future__ import annotations

import pandas as pd
import pytest

from research.feature002 import observe as OBS
from research.feature002.provenance import row_input, row_input_source

CARD = {"symbol": "AAA", "score": 80, "verdict": "BUY", "signals": ["52-week high breakout"],
        "entry": 100, "stop": 95, "target": 110, "chase_risk": False}


def _hist(last="2026-08-21", n=280, tag=None, snapshot=None):
    idx = pd.bdate_range(end=pd.Timestamp(last), periods=n)
    close = pd.Series([80.0 + i * 0.4 for i in range(n)], index=idx)
    df = pd.DataFrame({"open": close - 0.2, "high": close + 0.8, "low": close - 0.6,
                       "close": close, "volume": [200000] * n}, index=idx)
    if tag:
        df.attrs["quantterm_source"] = tag
    if snapshot:
        df.attrs["quantterm_snapshot_id"] = snapshot
    return df


@pytest.fixture
def build(monkeypatch, tmp_path):
    monkeypatch.setattr(OBS, "_build_rs_table", lambda *a, **k: None)
    monkeypatch.setattr(OBS, "HOOK_LOG", tmp_path / "hook_log.jsonl")
    monkeypatch.setattr(OBS, "LEDGER_DIR", tmp_path)

    def run(hist, session="2026-08-24"):
        monkeypatch.setattr(OBS, "_hist", lambda symbol: hist)
        _, rows = OBS.build_shadow_records([CARD], session_date=session,
                                           recorded_at=f"{session}T10:30:00+05:30",
                                           scan_cycle_id=f"t-{session}", source="live_scan")
        return rows[0]
    return run


def test_a_kite_snapshot_row_records_its_source_and_snapshot(build):
    row = build(_hist(tag="kite_snapshot", snapshot="snap-123"))
    inp = row_input(row)
    assert row_input_source(row) == "kite_snapshot" and inp["snapshot_id"] == "snap-123"
    assert inp["version"] == OBS.INPUT_PROVENANCE_VERSION
    assert inp["symbol_hist_as_of"] == "2026-08-21" and inp["cutoff"] == "2026-08-24"


def test_an_untagged_frame_is_unknown_not_guessed(build):
    assert row_input_source(build(_hist())) == "unknown"


def test_bars_after_the_session_cutoff_never_reach_a_feature(build):
    """A source that runs ahead of the session (a replay over today's history)."""
    from research.feature001.trend_features import compute_trend_features

    full = _hist(last="2026-09-15", tag="bhavcopy_store")
    row = build(full, session="2026-08-24")
    inp = row_input(row)
    assert inp["bars_after_cutoff_dropped"] == int((full.index > "2026-08-24").sum()) > 0
    assert inp["symbol_hist_as_of"] <= "2026-08-24"
    expected = compute_trend_features(full[full.index <= "2026-08-24"])
    assert row["feature_snapshot"]["trend"]["price"] == expected["price"]
    assert row["feature_snapshot"]["trend"]["price"] != compute_trend_features(full)["price"]


def test_switching_sources_is_recorded_per_row(build):
    a = build(_hist(tag="kite_snapshot", snapshot="snap-1"), session="2026-08-24")
    b = build(_hist(last="2026-08-24", tag="bhavcopy_store"), session="2026-08-25")
    assert (row_input_source(a), row_input_source(b)) == ("kite_snapshot", "bhavcopy_store")
    assert row_input(b)["snapshot_id"] is None


def test_a_legacy_row_stays_unknown_whatever_is_authoritative_today(monkeypatch):
    from scan import bulk_fetcher

    monkeypatch.setattr(bulk_fetcher, "_kite_authoritative", lambda: True)
    legacy = {"feature_snapshot": '{"trend": {"price": 1.0}, "hist_as_of": "2026-10-01"}'}
    assert row_input_source(legacy) == "unknown"
    assert row_input_source({"feature_snapshot": None}) == "unknown"
    assert row_input_source({"feature_snapshot": {"input": {"source": "made-up"}}}) == "unknown"


def test_get_cached_tags_the_served_copy_not_the_cache(monkeypatch):
    from data import bhavcopy_store as BS
    from scan import bulk_fetcher

    stored = _hist()
    monkeypatch.setattr(bulk_fetcher, "_kite_authoritative", lambda: False)
    monkeypatch.setattr(bulk_fetcher, "_bhav_ok", True)
    monkeypatch.setattr(BS, "get_ohlcv", lambda sym: stored.copy())
    served = bulk_fetcher.get_cached("AAA")
    assert served.attrs["quantterm_source"] == "bhavcopy_store"
    assert "quantterm_source" not in stored.attrs

    monkeypatch.setattr(bulk_fetcher, "_bhav_ok", False)
    monkeypatch.setitem(bulk_fetcher._yf_cache, "BBB", stored)
    served = bulk_fetcher.get_cached("BBB")
    assert served.attrs["quantterm_source"] == "yfinance_cache"
    assert "quantterm_source" not in stored.attrs


def test_a_row_without_history_records_none(build):
    assert row_input_source(build(None)) == "unknown"
    assert row_input(build(None))["source"] == "none"
