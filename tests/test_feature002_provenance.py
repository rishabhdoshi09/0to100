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
    # Cycle 9 (item 4): the row is recorded at 10:30 on 2026-08-24, before that session's
    # close, so the official archive's 08-24 bar did not exist yet either. This test used to
    # let it through (expected = bars <= 08-24): a lookahead of one archive bar.
    assert inp["bars_unavailable_at_record_dropped"] == 1 and inp["symbol_hist_as_of"] == "2026-08-21"
    expected = compute_trend_features(full[full.index < "2026-08-24"])
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

    # the offline cache serves whatever was tagged at ACQUISITION (GPT C7-06):
    # an untagged frame there stays unknown; a yfinance download says so
    monkeypatch.setattr(bulk_fetcher, "_bhav_ok", False)
    monkeypatch.setitem(bulk_fetcher._yf_cache, "BBB", stored)
    assert "quantterm_source" not in bulk_fetcher.get_cached("BBB").attrs
    tagged = bulk_fetcher._tag_source(_hist(), "yfinance")
    monkeypatch.setitem(bulk_fetcher._yf_cache, "CCC", tagged)
    assert bulk_fetcher.get_cached("CCC").attrs["quantterm_source"] == "yfinance"
    assert "quantterm_source" not in stored.attrs


def test_a_row_without_history_records_none(build):
    assert row_input_source(build(None)) == "unknown"
    assert row_input(build(None))["source"] == "none"


# ── GPT cycle-7 review (C7-06, C7-07): sketches as written ──

# GPT C7-06 (cycle-7 review sketch)
def test_cycle7_review_official_repair_keeps_its_source(monkeypatch):
    from data import acquisition, bhavcopy_store as BS
    from research.intelligence.data import nse_calendar as cal
    from scan import bulk_fetcher as bf

    official = _hist()
    monkeypatch.setattr(bf, "_kite_authoritative", lambda: False)
    monkeypatch.setattr(bf, "_bhav_ok", False)
    monkeypatch.setattr(bf, "_yf_cache", {})
    monkeypatch.setattr(BS, "_store", {})
    monkeypatch.setattr(BS, "_CA_EVENTS", {})
    monkeypatch.setattr(BS, "is_ready", lambda: True)
    monkeypatch.setattr(BS, "latest_two_eq_sessions", lambda: (None, None, None))
    monkeypatch.setattr(BS, "build_store", lambda: BS._store.update(AAA=official))
    monkeypatch.setattr(cal, "load_holidays", lambda: set())
    monkeypatch.setattr(cal, "latest_required_session", lambda *_: pd.Timestamp("2026-08-24"))
    monkeypatch.setattr(acquisition, "write_provenance", lambda _: None)

    assert bf.backfill_missing(["AAA"])["sources"] == ["nse_bhavcopy"]
    _, inp = OBS._input_provenance(OBS._hist("AAA"), "2026-08-24")
    assert row_input_source({"feature_snapshot": {"input": inp}}) == "bhavcopy_store"

# GPT C7-07 (cycle-7 review sketch)
def test_cycle7_review_rs_uses_the_recorded_frame(monkeypatch):
    from research.sepa.config import DEFAULT_CONFIG
    from research.sepa.rs import score_one

    day = "2026-08-24"
    a = _hist(last=day, tag="kite_snapshot", snapshot="A")
    b = a.copy()
    b.attrs["quantterm_snapshot_id"] = "B"
    b.loc[b.index[-1], "close"] *= 2
    reads = iter([a, b])
    monkeypatch.setattr(OBS, "_hist", lambda _: next(reads))
    monkeypatch.setattr(OBS, "_sector_map", lambda: ({}, "test"))

    _, rows = OBS.build_shadow_records(
        [CARD], session_date=day, recorded_at=f"{day}T10:30:00+05:30",
        scan_cycle_id="test-cycle", source="live_scan",
    )
    assert row_input(rows[0])["snapshot_id"] == "A"
    assert rows[0]["rs_score"] == pytest.approx(score_one(a, DEFAULT_CONFIG)["score"])


def test_the_live_bar_is_dated_by_the_exchange_session_and_kept(monkeypatch, build):
    """GPT C7-08, adapted: the sketch built an already mis-dated frame (the pre-fix
    producer's output). The fix is AT the producer, so this drives the producer with
    a host calendar one day ahead of IST: the live bar lands on the IST session and
    survives feature002's cutoff."""
    from datetime import date as _date
    from core import market_clock
    from data import bhavcopy_store as BS, kite_client, nse_live, nse_universe

    ist_day = _date(2026, 8, 24)
    monkeypatch.setattr(BS, "_store", {"AAA": _hist(last="2026-08-21")})
    monkeypatch.setattr(BS, "_store_last_day", _date(2026, 8, 21))
    monkeypatch.setattr(nse_live, "_is_trading_now", lambda: True)
    monkeypatch.setattr(market_clock, "today_ist", lambda: ist_day)
    monkeypatch.setattr(nse_live, "date", type("HostClock", (), {"today": staticmethod(lambda: _date(2026, 8, 25))}))
    monkeypatch.setattr(nse_universe, "get_nse_universe", lambda: [])
    monkeypatch.setattr(kite_client, "kite_credentials_available", lambda: False)
    monkeypatch.setattr(nse_live, "fetch_live_snapshot", lambda: {
        "AAA": {"open": 190.0, "high": 196.0, "low": 189.0, "close": 195.0, "volume": 1000.0}})
    assert nse_live.apply_live_to_store() == 1
    live = BS._store["AAA"]
    assert live.index[-1] == pd.Timestamp(ist_day)
    assert live.attrs["quantterm_live_overlay_date"] == ist_day.isoformat()
    row = build(live.copy(), session=ist_day.isoformat())
    assert row["feature_snapshot"]["trend"]["price"] == 195.0
    assert row_input(row)["bars_after_cutoff_dropped"] == 0


# ── GPT cycle-7 verification pass (C7V): sketches ──

# GPT C7V-06 (cycle-7 verification sketch)
def test_c7v_partial_official_repair_keeps_precedence(monkeypatch):
    import sys
    from types import SimpleNamespace
    from data import acquisition
    from scan import bulk_fetcher as bf

    official = bf._tag_source(_hist(), "bhavcopy_store")
    yahoo = pd.concat({s + ".NS": _hist(last="2026-08-20")
                       for s in ("AAA", "BBB")}, axis=1)
    monkeypatch.setattr(bf, "_kite_authoritative", lambda: False)
    monkeypatch.setattr(bf, "_bhav_ok", False)
    monkeypatch.setattr(bf, "_bhav_symbols", lambda: set())
    monkeypatch.setattr(bf, "_yf_cache", {})
    monkeypatch.setattr(bf, "_yf_cache_ts", 0)
    monkeypatch.setattr(bf, "_repair_via_bhavcopy", lambda _: {"AAA": official})
    monkeypatch.setattr(acquisition, "write_provenance", lambda _: None)
    monkeypatch.setitem(sys.modules, "yfinance",
                        SimpleNamespace(download=lambda *a, **k: yahoo))

    assert bf.backfill_missing(["AAA", "BBB"])["loaded"] == 2
    assert bf.get_cached("AAA").attrs["quantterm_source"] == "bhavcopy_store"
    assert bf.get_cached("BBB").attrs["quantterm_source"] == "yfinance"

# GPT C7V-07 (cycle-7 verification sketch)
def test_c7v_persisted_input_distinguishes_live_from_official(monkeypatch, build):
    from data import bhavcopy_store as bs
    from scan import bulk_fetcher as bf

    day = "2026-08-24"
    live = _hist(last=day)
    live.attrs["quantterm_live_overlay_date"] = day
    monkeypatch.setattr(bf, "_kite_authoritative", lambda: False)
    monkeypatch.setattr(bf, "_bhav_ok", True)
    monkeypatch.setattr(bs, "get_ohlcv", lambda _: live.copy())

    served = bf.get_cached("AAA")
    assert row_input(build(served)).get("live_overlay_date") == day

# GPT C7V-08 (cycle-7 verification sketch)
def test_c7v_rs_uses_one_snapshot_for_the_candidate_set(monkeypatch):
    day = "2026-08-24"
    aaa_a = _hist(last=day, tag="kite_snapshot", snapshot="A")
    bbb_a = _hist(last=day, tag="kite_snapshot", snapshot="A")
    bbb_a["close"] = 100.0
    bbb_b = bbb_a.copy()
    bbb_b.attrs["quantterm_snapshot_id"] = "B"
    bbb_b.loc[bbb_b.index[-1], "close"] = 300.0
    monkeypatch.setattr(OBS, "_sector_map", lambda: ({}, "test"))
    monkeypatch.setattr(OBS, "_hist", {"AAA": aaa_a, "BBB": bbb_b}.__getitem__)

    _, rows = OBS.build_shadow_records(
        [CARD, dict(CARD, symbol="BBB")], session_date=day,
        recorded_at=f"{day}T10:30:00+05:30", scan_cycle_id="switch", source="live_scan",
    )
    assert row_input(rows[0])["snapshot_id"] == "A"
    coherent = OBS._build_rs_table(["AAA", "BBB"], day, {"AAA": aaa_a, "BBB": bbb_a})
    assert coherent["percentiles"]["AAA"] == 50.0
    assert rows[0]["rs_percentile"] in (None, coherent["percentiles"]["AAA"])

# GPT C7V-09 (cycle-7 verification sketch)
def test_c7v_overlay_does_not_publish_across_ist_sessions(monkeypatch):
    from datetime import date, datetime
    from types import SimpleNamespace
    from core import market_clock
    from data import bhavcopy_store as bs, kite_client, nse_live, nse_universe

    ist = market_clock.IST
    now = [datetime(2026, 8, 24, 15, 59, tzinfo=ist)]

    class Clock:
        @staticmethod
        def now(tz=None):
            return now[0].astimezone(tz) if tz else now[0].replace(tzinfo=None)

    def fetch():
        now[0] = datetime(2026, 8, 25, 9, 16, tzinfo=ist)
        return {"AAA": dict(open=190, high=196, low=189, close=195, volume=1000)}

    monkeypatch.setattr(nse_live, "datetime", Clock)
    monkeypatch.setattr(market_clock, "datetime", Clock)
    monkeypatch.setattr(nse_live, "date", SimpleNamespace(today=lambda: now[0].date()))
    monkeypatch.setattr(bs, "_store", {"AAA": _hist(last="2026-08-21")})
    monkeypatch.setattr(bs, "_store_last_day", date(2026, 8, 21))
    monkeypatch.setattr(nse_universe, "get_nse_universe", lambda: [])
    monkeypatch.setattr(kite_client, "kite_credentials_available", lambda: False)
    monkeypatch.setattr(nse_live, "fetch_live_snapshot", fetch)

    nse_live.apply_live_to_store()
    assert pd.Timestamp("2026-08-24") not in bs._store["AAA"].index


def test_an_overlapping_public_result_never_replaces_an_official_frame(monkeypatch):
    """C7V-06, the second layer: even if the public rung returned a name the
    official rung already supplied, the official frame wins the merge."""
    from data import acquisition
    from scan import bulk_fetcher as bf

    official = bf._tag_source(_hist(), "bhavcopy_store")
    monkeypatch.setattr(bf, "_kite_authoritative", lambda: False)
    monkeypatch.setattr(bf, "_bhav_ok", False)
    monkeypatch.setattr(bf, "_bhav_symbols", lambda: set())
    monkeypatch.setattr(bf, "_yf_cache", {})
    monkeypatch.setattr(bf, "_repair_via_bhavcopy", lambda _: {"AAA": official})
    monkeypatch.setattr(bf, "_repair_via_yfinance", lambda names: {
        s: bf._tag_source(_hist(last="2026-08-20"), "yfinance") for s in ("AAA", "BBB")})
    monkeypatch.setattr(acquisition, "write_provenance", lambda _: None)
    assert bf.backfill_missing(["AAA", "BBB"])["loaded"] == 2
    assert bf.get_cached("AAA").attrs["quantterm_source"] == "bhavcopy_store"
