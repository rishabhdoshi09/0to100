"""
Deterministic, network-free tests for Zerodha Kite as the DATA-ONLY provider for PAPER_AUTO.

An injected FakeKite/FakeFeed makes this fully offline. Kite is used for instrument master,
historical candles and live ticks only — no order API is reachable from the PAPER_AUTO path.
"""
from __future__ import annotations

import dataclasses
import inspect
from datetime import date, datetime, timedelta

import pytest

from research.intelligence.data import kite_source as KS
from research.intelligence.data import kite_live as KL
from research.intelligence.data import nse_calendar as CAL
from research.intelligence.data.snapshot_store import SnapshotStore
from research.intelligence import data_state as DS


# ── deterministic fake Kite client (DATA ONLY — no order methods) ────────────────

class FakeKite:
    def __init__(self, *, valid=True, with_benchmark=True, fail_first=0):
        self.valid = valid
        self.with_benchmark = with_benchmark
        self.fail_first = fail_first
        self._calls = 0
        self.hist_calls = 0

    def profile(self):
        if not self.valid:
            raise RuntimeError("token expired")
        return {"user_id": "AB1234"}

    def instruments(self, exchange="NSE"):
        out = [
            {"instrument_token": 111, "exchange_token": 1, "tradingsymbol": "WIN", "name": "Winner",
             "isin": "INWIN01", "tick_size": 0.05, "lot_size": 1, "instrument_type": "EQ",
             "segment": "NSE", "exchange": "NSE"},
            {"instrument_token": 222, "exchange_token": 2, "tradingsymbol": "FLAT", "name": "Flatco",
             "isin": "INFLAT1", "tick_size": 0.05, "lot_size": 1, "instrument_type": "EQ",
             "segment": "NSE", "exchange": "NSE"},
            {"instrument_token": 333, "exchange_token": 3, "tradingsymbol": "WEAK", "name": "Weakco",
             "isin": "INWEAK1", "tick_size": 0.05, "lot_size": 1, "instrument_type": "EQ",
             "segment": "NSE", "exchange": "NSE"},
        ]
        if self.with_benchmark:
            out.append({"instrument_token": 999, "exchange_token": 9, "tradingsymbol": "NIFTY50",
                        "name": "NIFTY 50", "isin": "", "instrument_type": "INDEX",
                        "segment": "INDICES", "exchange": "NSE"})
        return out

    def historical(self, token, frm, to, interval="day"):
        self.hist_calls += 1
        if self.hist_calls <= self.fail_first:
            raise RuntimeError("transient 503")
        d0, d1 = date.fromisoformat(frm), date.fromisoformat(to)
        base = {111: (100.0, 0.5), 222: (100.0, 0.0), 333: (100.0, -0.2), 999: (100.0, 0.0)}[token]
        out, i, d = [], 0, d0
        while d <= d1:
            if d.weekday() < 5:                       # sessions only
                c = base[0] + base[1] * i
                out.append({"date": d.isoformat(), "open": c, "high": c + 1.5,
                            "low": c - 1.5, "close": c, "volume": 1000})
                i += 1
            d += timedelta(days=1)
        return out


class FakeFeed:
    def __init__(self):
        self.subscribed = []
        self.connected = False

    def connect(self, subs):
        self.connected = True
        self.subscribed = list(subs)

    def subscribe(self, subs):
        self.subscribed = list(subs)


_NOW = datetime(2024, 6, 3, 18, 30)        # Monday, post publication cutoff


def _ds(tmp_path, **kw):
    return KS.KiteDataSource(
        FakeKite(**kw), SnapshotStore(tmp_path / "snaps"),
        universe={"WIN", "FLAT", "WEAK"}, benchmark_name="NIFTY 50",
        history_dir=tmp_path / "hist", progress_path=tmp_path / "progress.json",
        sleep_fn=lambda s: None)               # no real sleeping


# ── 1–2 session + instrument master ──────────────────────────────────────────────

def test_valid_session_triggers_refresh(tmp_path):
    ds = _ds(tmp_path)
    assert ds.session_valid()
    info = ds.refresh_instruments()
    assert info["n"] >= 3 and info["benchmark_resolvable"]

def test_invalid_session_blocks(tmp_path):
    ds = _ds(tmp_path, valid=False)
    assert not ds.session_valid()
    rep = ds.daily_refresh(now=_NOW)
    assert rep.status == "BLOCKED" and any(i["code"] == "AUTH_INVALID" for i in rep.incidents)

def test_instrument_reconciliation_detects_changes():
    prev = {"isin:INWIN01": {"canonical_id": "isin:INWIN01", "tradingsymbol": "OLDWIN",
                             "instrument_token": 111}}
    new = [{"instrument_token": 900, "tradingsymbol": "WIN", "isin": "INWIN01", "exchange": "NSE"}]
    master, changes = KS.reconcile_instruments(new, prev)
    # canonical identity (ISIN) is stable across symbol AND token change
    assert "isin:INWIN01" in master
    assert changes["symbol_changed"] and changes["token_changed"]

def test_canonical_id_is_not_the_token():
    a = KS.canonical_id({"isin": "INWIN01", "tradingsymbol": "WIN", "instrument_token": 111})
    b = KS.canonical_id({"isin": "INWIN01", "tradingsymbol": "WIN", "instrument_token": 900})
    assert a == b and "111" not in a and "900" not in b


# ── 3–7 historical bootstrap ─────────────────────────────────────────────────────

def test_incremental_download_only_missing_range(tmp_path):
    ds = _ds(tmp_path); ds.refresh_instruments()
    n1 = ds.bootstrap_symbol("isin:INWIN01", 111, "2024-05-01", "2024-05-31")
    n2 = ds.bootstrap_symbol("isin:INWIN01", 111, "2024-05-01", "2024-05-31")   # already have it
    assert n1 > 0 and n2 == 0

def test_rate_limiter_is_used(tmp_path):
    calls = {"n": 0}
    rl = KS.RateLimiter(max_per_sec=5, sleep_fn=lambda s: calls.__setitem__("n", calls["n"] + 1),
                        clock=lambda: 0.0)              # clock frozen → always throttles
    ds = _ds(tmp_path); ds.rl = rl; ds.refresh_instruments()
    ds.bootstrap_symbol("isin:INWIN01", 111, "2024-05-01", "2024-05-10")
    assert rl.calls >= 1

def test_retry_backs_off_then_succeeds(tmp_path):
    slept = {"n": 0}
    ds = KS.KiteDataSource(FakeKite(fail_first=2), SnapshotStore(tmp_path / "s"),
                           universe={"WIN"}, history_dir=tmp_path / "h",
                           progress_path=tmp_path / "p.json",
                           sleep_fn=lambda s: slept.__setitem__("n", slept["n"] + 1))
    ds.refresh_instruments()
    n = ds.bootstrap_symbol("isin:INWIN01", 111, "2024-05-01", "2024-05-10")
    assert n > 0 and slept["n"] >= 2                    # backed off twice, then succeeded

def test_restart_resumes_from_persisted_progress(tmp_path):
    ds1 = _ds(tmp_path); ds1.refresh_instruments()
    ds1.bootstrap_symbol("isin:INWIN01", 111, "2024-05-01", "2024-05-31")
    ds2 = _ds(tmp_path)                                 # new instance, same progress/history paths
    n = ds2.bootstrap_symbol("isin:INWIN01", 111, "2024-05-01", "2024-05-31")
    assert n == 0 and ds2._progress.get("isin:INWIN01")   # resumed, no re-download

def test_invalid_ohlc_is_quarantined(tmp_path):
    class BadKite(FakeKite):
        def historical(self, token, frm, to, interval="day"):
            return [{"date": "2024-05-01", "open": 100, "high": 90, "low": 95, "close": 105, "volume": 1},
                    {"date": "2024-05-02", "open": 100, "high": 110, "low": 95, "close": 105, "volume": 1}]
    ds = KS.KiteDataSource(BadKite(), SnapshotStore(tmp_path / "s"), universe={"WIN"},
                           history_dir=tmp_path / "h", progress_path=tmp_path / "p.json",
                           sleep_fn=lambda s: None)
    ds.refresh_instruments()
    n = ds.bootstrap_symbol("isin:INWIN01", 111, "2024-05-01", "2024-05-02")
    assert n == 1                                       # only the valid bar stored


# ── 8–10 snapshot commit + activation ────────────────────────────────────────────

def test_valid_refresh_activates_new_snapshot(tmp_path, monkeypatch):
    monkeypatch.setattr(CAL, "_now_ist", lambda: _NOW)
    ds = _ds(tmp_path)
    rep = ds.daily_refresh(now=_NOW)
    assert rep.activated and rep.snapshot_id and rep.benchmark_ok
    assert ds.store.get_active_snapshot() == rep.snapshot_id

def test_missing_benchmark_blocks_forward_eligibility(tmp_path, monkeypatch):
    monkeypatch.setattr(CAL, "_now_ist", lambda: _NOW)
    ds = _ds(tmp_path, with_benchmark=False)
    rep = ds.daily_refresh(now=_NOW)
    assert not rep.activated and rep.status == "COMMITTED_NOT_ACTIVATED"

def test_failed_refresh_preserves_previous_active(tmp_path, monkeypatch):
    monkeypatch.setattr(CAL, "_now_ist", lambda: _NOW)
    ds = _ds(tmp_path)
    good = ds.daily_refresh(now=_NOW)
    assert good.activated
    # a subsequent NOT-forward-eligible refresh (benchmark removed) must not deactivate the good one
    ds.client.with_benchmark = False
    ds.master = {}                                      # force re-read of instruments
    bad = ds.daily_refresh(now=_NOW)
    assert not bad.activated
    assert ds.store.get_active_snapshot() == good.snapshot_id


# ── 11–13 live overlay ───────────────────────────────────────────────────────────

def test_reconnect_restores_subscriptions():
    feed = FakeFeed()
    ov = KL.KiteLiveOverlay(feed=feed, sleep_fn=lambda s: None)
    ov.connect(); ov.subscribe(["WIN", "FLAT"])
    feed.subscribed = []                                # simulate drop
    ov.on_reconnect()
    assert set(feed.subscribed) == {"WIN", "FLAT"}

def test_stale_ticks_block_new_entries():
    ov = KL.KiteLiveOverlay(max_stale_s=30, clock=lambda: 1000.0)
    ov.on_tick("WIN", 105.0, ts=900.0)                  # 100s old
    assert ov.is_stale("WIN") and not ov.entry_allowed("WIN")
    ov.on_tick("WIN", 106.0, ts=1000.0)
    assert not ov.is_stale("WIN") and ov.entry_allowed("WIN")

def test_out_of_order_and_future_ticks_rejected():
    ov = KL.KiteLiveOverlay(clock=lambda: 1000.0)
    assert ov.on_tick("WIN", 100.0, ts=1000.0)
    assert not ov.on_tick("WIN", 101.0, ts=990.0)       # out of order
    assert not ov.on_tick("WIN", 102.0, ts=2000.0)      # future
    assert ov.health()["rejected"] == {"out_of_order": 1, "future": 1}

def test_overlay_never_finalizes_a_daily_bar():
    src = inspect.getsource(KL)
    assert "commit_snapshot" not in src and "activate_snapshot" not in src


# ── 14 PAPER_AUTO trades from kite-fed data ──────────────────────────────────────

def test_paper_auto_trades_from_kite_data(tmp_path, monkeypatch):
    monkeypatch.setattr(CAL, "_now_ist", lambda: _NOW)
    from research.auto_research.scheduler import AutoResearchBrain
    from research.intelligence.registry import StrategyRegistry
    from research.strategy_studio import discovery as DISC
    store = SnapshotStore(tmp_path / "snaps")
    ds = KS.KiteDataSource(FakeKite(), store, universe={"WIN", "FLAT", "WEAK"},
                           benchmark_name="NIFTY 50", history_dir=tmp_path / "hist",
                           progress_path=tmp_path / "p.json", sleep_fn=lambda s: None)
    rep = ds.daily_refresh(now=_NOW)
    assert rep.activated
    brain = AutoResearchBrain(event_store_path=tmp_path / "e.jsonl",
                              runtime_state_path=tmp_path / "s.json",
                              intel_book_path=tmp_path / "b.json",
                              paper_config_path=tmp_path / "pc.json",
                              regime_fn=lambda: "RISK_ON")
    brain.snapshot_store = store
    spec = dataclasses.replace(DISC.generate(DISC.DiscoveryBudget())[0],
                               strategy_id="MOM", family="cross_sectional_momentum", max_holding_days=5)
    brain.strategy_registry = StrategyRegistry().build([spec])
    out = brain.run_intelligence_cycle_day(date="2024-06-03")
    assert out["positions_opened"] and out["positions_opened"][0][1] == "WIN"


# ── 15 no Kite order API reachable from PAPER_AUTO ───────────────────────────────

def test_no_kite_order_api_in_paper_auto_path():
    import research.intelligence.runtime.autonomous_loop as L
    import research.auto_research.scheduler as S
    for mod in (L, S, KS, KL):
        src = inspect.getsource(mod)
        for banned in ("place_order", "modify_order", "cancel_order", "place_gtt",
                       ".gtt(", "order_place", "cancel_gtt"):
            assert banned not in src, f"{banned} present in {mod.__name__}"

def test_data_source_uses_only_data_apis(tmp_path):
    # a FakeKite with NO order methods drives the full refresh → proves the path is data-only
    ds = _ds(tmp_path)
    assert not hasattr(ds.client, "place_order")
    assert ds.refresh_instruments()["n"] >= 3


# ── 16–18 restart, no-upload, fallback ───────────────────────────────────────────

def test_restart_restores_downloader_state(tmp_path):
    ds1 = _ds(tmp_path); ds1.refresh_instruments()
    ds1.bootstrap_symbol("isin:INWIN01", 111, "2024-05-01", "2024-05-31")
    ds2 = _ds(tmp_path)
    assert ds2._progress.get("isin:INWIN01") == ds1._progress.get("isin:INWIN01")

def test_no_manual_upload_needed_for_kite_refresh(tmp_path, monkeypatch):
    monkeypatch.setattr(CAL, "_now_ist", lambda: _NOW)
    # daily_refresh produces an active snapshot using ONLY client APIs (no file import)
    ds = _ds(tmp_path)
    assert "data_setup" not in inspect.getsource(KS)     # kite source doesn't touch the file path
    assert ds.daily_refresh(now=_NOW).activated

def test_manual_file_import_remains_available():
    from research.intelligence.data import from_bhav
    assert hasattr(from_bhav, "snapshot_from_bhav_dir")  # offline fallback still present
