"""Cycle 9, item 4: feature002 inputs are point-in-time and one generation; equity output
does not depend on F&O eligibility.

Fixed historical fixtures, explicit information time (``recorded_at``), no network.

1. A bar is used only if it EXISTED at recorded_at: never one dated after the recording
   day, and never the official archive's bar of the recording day before that session's
   close (only the live intraday overlay of that day exists then). A Kite snapshot's bar
   of the day is what Kite served at that moment (a partial candle) and stays.
2. A cross-sectional rank is computed over ONE input generation: the Kite snapshot, else
   the official store, else a set of one uncertified source (ranked, marked uncertified).
   A row of another source, or whose history ends before its generation's, is excluded
   with its reason; two uncertified sources and no certified one rank nothing; two Kite
   snapshots rank nothing (C7V-08, unchanged).
3. The F&O universe only orders the scan's walk and labels cards: the published equity
   records (verdicts, scores, order) are identical whatever it holds.
"""
from __future__ import annotations

from contextlib import contextmanager

import pandas as pd
import pytest

from research.feature002 import observe as OBS
from research.feature002.provenance import row_input


def _hist(last="2026-08-24", n=280, tag=None, snapshot=None, overlay=None, slope=0.4):
    idx = pd.bdate_range(end=pd.Timestamp(last), periods=n)
    close = pd.Series([80.0 + i * slope for i in range(n)], index=idx)
    df = pd.DataFrame({"open": close - 0.2, "high": close + 0.8, "low": close - 0.6,
                       "close": close, "volume": [200000] * n}, index=idx)
    if tag:
        df.attrs["quantterm_source"] = tag
    if snapshot:
        df.attrs["quantterm_snapshot_id"] = snapshot
    if overlay:
        df.attrs["quantterm_live_overlay_date"] = overlay
    return df


def _card(symbol, score=80):
    return {"symbol": symbol, "score": score, "verdict": "BUY", "signals": ["52-week high breakout"],
            "entry": 100, "stop": 95, "target": 110, "chase_risk": False}


@pytest.fixture
def build(monkeypatch, tmp_path):
    seen = {}

    def rs_table(symbols, as_of, frames=None):
        seen["frames"] = sorted(frames or {})
        return None

    monkeypatch.setattr(OBS, "_build_rs_table", rs_table)
    monkeypatch.setattr(OBS, "HOOK_LOG", tmp_path / "hook_log.jsonl")
    monkeypatch.setattr(OBS, "LEDGER_DIR", tmp_path)

    def run(frames, *, session="2026-08-24", at="10:30:00"):
        monkeypatch.setattr(OBS, "_hist", lambda symbol: frames[symbol])
        cards = [_card(s, 90 - i) for i, s in enumerate(frames)]
        _, rows = OBS.build_shadow_records(cards, session_date=session,
                                           recorded_at=f"{session}T{at}+05:30",
                                           scan_cycle_id=f"t-{session}", source="live_scan")
        return {r["symbol"]: r for r in rows}

    run.seen = seen
    return run


# ── 1. information time ──

def test_an_archive_bar_of_the_recording_day_does_not_exist_before_the_close(build):
    before = build({"AAA": _hist(tag="bhavcopy_store")}, at="10:30:00")["AAA"]
    after = build({"AAA": _hist(tag="bhavcopy_store")}, at="18:05:00")["AAA"]
    assert row_input(before)["bars_unavailable_at_record_dropped"] == 1
    assert row_input(before)["symbol_hist_as_of"] == "2026-08-21"
    assert row_input(after)["bars_unavailable_at_record_dropped"] == 0
    assert row_input(after)["symbol_hist_as_of"] == "2026-08-24"


def test_the_live_overlay_bar_of_the_recording_day_is_kept(build):
    row = build({"AAA": _hist(tag="bhavcopy_store", overlay="2026-08-24")}, at="10:30:00")["AAA"]
    inp = row_input(row)
    assert inp["bars_unavailable_at_record_dropped"] == 0
    assert inp["symbol_hist_as_of"] == inp["live_overlay_date"] == "2026-08-24"


def test_a_kite_snapshot_bar_of_the_recording_day_is_what_kite_served(build):
    row = build({"AAA": _hist(tag="kite_snapshot", snapshot="s1")}, at="10:30:00")["AAA"]
    assert row_input(row)["bars_unavailable_at_record_dropped"] == 0
    assert row_input(row)["symbol_hist_as_of"] == "2026-08-24"


def test_a_bar_dated_after_the_recording_day_is_dropped_whatever_its_source(build, monkeypatch):
    """A host clock behind the session (the C7-08 class): recorded on 08-21 for session 08-24."""
    from research.feature001.trend_features import compute_trend_features

    frame = _hist(tag="kite_snapshot", snapshot="s1")
    monkeypatch.setattr(OBS, "_hist", lambda symbol: frame)
    _, rows = OBS.build_shadow_records([_card("AAA")], session_date="2026-08-24",
                                       recorded_at="2026-08-21T16:00:00+05:30", scan_cycle_id="t", source="live_scan")
    inp = row_input(rows[0])
    assert inp["bars_after_cutoff_dropped"] == 0 and inp["bars_unavailable_at_record_dropped"] == 1
    assert inp["symbol_hist_as_of"] == "2026-08-21"
    expected = compute_trend_features(frame[frame.index <= "2026-08-21"])
    assert rows[0]["feature_snapshot"]["trend"]["price"] == expected["price"]


@pytest.mark.parametrize("utc, dropped", [("05:30", 1), ("12:30", 0)])
def test_a_recorded_time_in_another_zone_is_judged_in_ist(build, monkeypatch, utc, dropped):
    """05:30 UTC is 11:00 IST (before the close: the archive bar is unavailable); 12:30 UTC is
    18:00 IST (after the close: it exists), although 12:30 is before 15:30 on its own clock."""
    frame = _hist(tag="bhavcopy_store")
    monkeypatch.setattr(OBS, "_hist", lambda symbol: frame)
    _, rows = OBS.build_shadow_records([_card("AAA")], session_date="2026-08-24",
                                       recorded_at=f"2026-08-24T{utc}:00+00:00", scan_cycle_id="t", source="live_scan")
    assert row_input(rows[0])["bars_unavailable_at_record_dropped"] == dropped


# ── 2. one generation per cross-section ──

def test_a_store_fallback_beside_kite_frames_is_excluded_from_the_cross_section(build):
    rows = build({"CCC": _hist(tag="bhavcopy_store", last="2026-08-21", slope=0.5),     # first in card order
                  "AAA": _hist(tag="kite_snapshot", snapshot="s1"),
                  "BBB": _hist(tag="kite_snapshot", snapshot="s1", slope=0.3)})
    ri = rows["AAA"]["feature_snapshot"]["rank_inputs"]
    assert ri["generation"] == {"source": "kite_snapshot", "snapshot_id": "s1", "as_of": "2026-08-24"}
    assert ri["certified"] and ri["excluded"] == {"CCC": "SOURCE_MISMATCH"} and ri["ranked"] == ["AAA", "BBB"]
    assert build.seen["frames"] == ["AAA", "BBB"]                       # the RS table sees one generation
    c = rows["CCC"]
    assert c["rank_excluded"] == "SOURCE_MISMATCH"
    assert all(c[k] is None for k in ("rs_rank", "rs_pctl_in_set", "trend_rank", "trend_pctl_in_set",
                                       "r3_score", "combined_shadow_rank"))
    assert c["feature_snapshot"]["trend"]["available"]                    # its own features are kept
    assert sorted(rows[s]["trend_rank"] for s in ("AAA", "BBB")) == [1, 2]
    assert [rows[s]["production_rank"] for s in ("CCC", "AAA", "BBB")] == [1, 2, 3]     # card order, whole set


def test_a_stale_input_is_excluded_from_its_generation(build):
    rows = build({"AAA": _hist(tag="kite_snapshot", snapshot="s1"),
                  "BBB": _hist(tag="kite_snapshot", snapshot="s1", last="2026-08-20")})
    ri = rows["AAA"]["feature_snapshot"]["rank_inputs"]
    assert ri["excluded"] == {"BBB": "STALE_INPUT"} and ri["generation"]["as_of"] == "2026-08-24"
    assert rows["BBB"]["rank_excluded"] == "STALE_INPUT" and rows["BBB"]["trend_rank"] is None
    assert rows["AAA"]["trend_rank"] == 1


def test_two_uncertified_sources_rank_nothing(build):
    rows = build({"AAA": _hist(), "BBB": _hist(tag="yfinance")})
    ri = rows["AAA"]["feature_snapshot"]["rank_inputs"]
    assert not ri["coherent"] and ri["withheld"] == "MIXED_UNCERTIFIED_SOURCES"
    assert all(r["rs_rank"] is None and r["combined_shadow_rank"] is None for r in rows.values())


def test_a_uniformly_unknown_set_is_ranked_but_marked_uncertified(build):
    rows = build({"AAA": _hist(), "BBB": _hist(slope=0.3)})
    ri = rows["AAA"]["feature_snapshot"]["rank_inputs"]
    assert ri["coherent"] and not ri["certified"] and ri["generation"]["source"] == "unknown"
    assert sorted(r["trend_rank"] for r in rows.values()) == [1, 2]


def test_two_kite_snapshots_still_rank_nothing(build):
    rows = build({"AAA": _hist(tag="kite_snapshot", snapshot="s1"),
                  "BBB": _hist(tag="kite_snapshot", snapshot="s2")})
    ri = rows["AAA"]["feature_snapshot"]["rank_inputs"]
    assert ri["withheld"] == "MIXED_KITE_SNAPSHOTS" and all(r["rs_rank"] is None for r in rows.values())


def test_a_coherent_set_is_ranked_exactly_as_before(build):
    rows = build({"AAA": _hist(tag="kite_snapshot", snapshot="s1"),
                  "BBB": _hist(tag="kite_snapshot", snapshot="s1", slope=0.3)})
    ri = rows["AAA"]["feature_snapshot"]["rank_inputs"]
    assert ri["excluded"] == {} and ri["certified"]
    assert all("rank_excluded" not in r for r in rows.values())


# ── 3. equity output is independent of F&O eligibility ──

class _Scanner:
    def __init__(self):
        self.asked = []

    def scan(self, symbols, *, progress=None, prefetch=False):
        self.asked.append(list(symbols))
        out = []
        for s in symbols:                                       # each result depends on its symbol only
            score = float(40 + (sum(map(ord, s)) % 50))
            out.append({"symbol": s, "score": score, "verdict": "BUY" if score >= 70 else "WATCH",
                        "signals": ["MOMENTUM"] if score >= 60 else ["PRE_BREAKOUT"], "reasons": ["q"],
                        "price": 100.0, "entry": 101.0, "stop": 95.0, "target": 113.0})
        return out


class _Probe:
    def finalize(self, results, *, cached, walked_total):
        return {"summary": {"scanner_instrumented": True, "checked": len(results), "failed": 0,
                            "state": "COMPLETE", "reason_counts": {"QUALIFIED": len(results)}}, "ledger": []}


@contextmanager
def _observe(_scanner, _symbols):
    yield _Probe()


def test_published_equity_records_do_not_depend_on_fno_eligibility(monkeypatch):
    from scan import market_scan_service as S

    universe = {s: s.title() for s in ("AAA", "BBB", "CCC", "DDD", "EEE", "FFF", "GGG", "HHH")}
    monkeypatch.setattr(S, "_saved_priority_inputs", lambda: (None, None, None, []))
    monkeypatch.setattr(S, "_feature002_hook", None)
    monkeypatch.setattr(S, "_active_snapshot_id", lambda: "")
    monkeypatch.setattr("scan.scan_coverage.observe_scanner", _observe)
    monkeypatch.setattr("scan.bulk_fetcher.cached_symbols", lambda: sorted(universe))
    monkeypatch.setattr("product.scan_store._evidence_authority", lambda: None)
    monkeypatch.setattr("product.scan_store._history_freshness", lambda: ({
        "available_session": "2026-09-18", "expected_latest_completed_session": "2026-09-18",
        "reason_code": "HISTORY_CURRENT", "stale_sessions": 0, "current": True}, ""))

    published, walks = [], []
    for fno in (set(), set(universe), {"HHH", "DDD", "BBB"}):
        scanner = _Scanner()
        report = S.run_whole_market_scan(universe_provider=lambda: universe, prefetch_fn=lambda *_a, **_k: None,
                                         scanner=scanner, fno_provider=lambda fno=fno: fno, save=False)
        assert report.ok, report.as_dict()
        records = report.payload["records"]
        assert {r["symbol"] for r in records if r["fno_available"]} == fno     # the label, and only the label
        published.append([{k: v for k, v in r.items() if k != "fno_available"} for r in records])
        walks.append(scanner.asked[0])
    assert published[0] == published[1] == published[2]
    assert sorted(walks[0]) == sorted(walks[1]) == sorted(walks[2]) == sorted(universe)   # every name walked
    assert walks[0] != walks[2]                                                 # in a different order
