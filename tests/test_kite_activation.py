"""
Deterministic, network-free tests for the REAL Zerodha data-activation runner (kite_activation).

Injected FakeKite / FakeTicker make this fully offline. The runner's job is to drive the existing
certified machinery (KiteDataSource / KiteLiveOverlay / the background worker) and report 8 separate
operational states honestly. These tests prove the state logic and the data-only boundary — they do
NOT (and cannot) certify PAPER_AUTO_REAL_DATA_OPERATIONAL on genuine data, which needs a live token.
"""
from __future__ import annotations

import dataclasses
from datetime import datetime

from research.intelligence.data import kite_activation as A
from research.intelligence.data import kite_source as KS
from research.intelligence.data import kite_live as KL
from research.intelligence.data import nse_calendar as CAL
from research.intelligence.data.snapshot_store import SnapshotStore

from tests.test_kite_data import FakeKite, _NOW           # reuse the certified data fakes


# ── a real brain wired to the same store (produces a genuine cycle decision) ─────

def _brain(tmp_path, store):
    from research.auto_research.scheduler import AutoResearchBrain
    from research.intelligence.registry import StrategyRegistry
    from research.strategy_studio import discovery as DISC
    b = AutoResearchBrain(event_store_path=tmp_path / "e.jsonl",
                          runtime_state_path=tmp_path / "s.json",
                          intel_book_path=tmp_path / "b.json",
                          paper_config_path=tmp_path / "pc.json",
                          regime_fn=lambda: "RISK_ON")
    b.snapshot_store = store
    spec = dataclasses.replace(DISC.generate(DISC.DiscoveryBudget())[0],
                               strategy_id="MOM", family="cross_sectional_momentum",
                               max_holding_days=5)
    b.strategy_registry = StrategyRegistry().build([spec])
    return b


def _fresh_overlay(clock_epoch: float):
    ov = KL.KiteLiveOverlay(clock=lambda: clock_epoch)
    ov.on_tick("WIN", 123.0, ts=clock_epoch - 1.0)         # one fresh, valid tick
    return ov


def _activate(tmp_path, monkeypatch, *, market_open=True, with_benchmark=True, ticking=True,
              start_worker=False):
    monkeypatch.setattr(CAL, "_now_ist", lambda: _NOW)
    store = SnapshotStore(tmp_path / "snaps")
    brain = _brain(tmp_path, store)
    ov = _fresh_overlay(1_717_400_000.0) if ticking else None
    rep = A.activate(client=FakeKite(with_benchmark=with_benchmark), store=store, brain=brain,
                     universe={"WIN", "FLAT", "WEAK"}, benchmark_name="NIFTY 50",
                     history_dir=tmp_path / "hist", progress_path=tmp_path / "p.json",
                     overlay=ov, subscribe_symbols=["WIN"], now=_NOW, market_open=market_open,
                     start_worker=start_worker, run_cycle=True)
    if start_worker:
        brain.stop()
    return rep, store


# ── 1. full healthy activation (market open) → all eight states PASS ─────────────

def test_full_activation_all_states_pass(tmp_path, monkeypatch):
    rep, store = _activate(tmp_path, monkeypatch, market_open=True, ticking=True, start_worker=True)
    for k in A._STATE_ORDER:
        assert rep.status(k) == A.PASS, f"{k} = {rep.states[k]}"
    assert rep.tier == "FORWARD_ELIGIBLE"
    assert rep.active_pointer == rep.snapshot_id == store.get_active_snapshot()
    assert rep.latest_cycle.get("eligibility") in ("TRADED", "NO_ELIGIBLE_TRADE")


# ── 2. no session → every state FAIL, previous active snapshot preserved ─────────

def test_invalid_session_fails_all_and_preserves_active(tmp_path, monkeypatch):
    monkeypatch.setattr(CAL, "_now_ist", lambda: _NOW)
    store = SnapshotStore(tmp_path / "snaps")
    # seed a good active snapshot first
    good, _ = _activate(tmp_path, monkeypatch)            # uses its own store; make a fresh explicit one
    store2 = SnapshotStore(tmp_path / "snaps2")
    ds = KS.KiteDataSource(FakeKite(), store2, universe={"WIN", "FLAT", "WEAK"},
                           benchmark_name="NIFTY 50", history_dir=tmp_path / "h2",
                           progress_path=tmp_path / "p2.json", sleep_fn=lambda s: None)
    prior = ds.daily_refresh(now=_NOW).snapshot_id
    assert store2.get_active_snapshot() == prior
    rep = A.activate(client=FakeKite(valid=False), store=store2, brain=_brain(tmp_path, store2),
                     universe={"WIN"}, now=_NOW, start_worker=False, run_cycle=False)
    assert rep.status("KITE_SESSION_CONNECTED") == A.FAIL
    assert all(rep.status(k) == A.FAIL for k in A._STATE_ORDER)
    assert store2.get_active_snapshot() == prior          # untouched
    assert rep.blocker


# ── 3. market closed → live feed PENDING_MARKET_SESSION, data path still PASS ────

def test_market_closed_live_feed_pending(tmp_path, monkeypatch):
    rep, _ = _activate(tmp_path, monkeypatch, market_open=False, ticking=True)
    assert rep.status("GENUINE_SNAPSHOT_ACTIVE") == A.PASS
    assert rep.status("LIVE_FEED_CONNECTED") == A.PENDING_MARKET_SESSION
    # a healthy daily-data cycle still runs and decides
    assert rep.latest_cycle.get("eligibility") in ("TRADED", "NO_ELIGIBLE_TRADE")


# ── 4. no live feed wired → PENDING, everything else operational ─────────────────

def test_no_feed_wired_is_pending(tmp_path, monkeypatch):
    monkeypatch.setattr(CAL, "_now_ist", lambda: _NOW)
    store = SnapshotStore(tmp_path / "snaps")
    brain = _brain(tmp_path, store)
    rep = A.activate(client=FakeKite(), store=store, brain=brain, universe={"WIN", "FLAT", "WEAK"},
                     benchmark_name="NIFTY 50", history_dir=tmp_path / "hist",
                     progress_path=tmp_path / "p.json", now=_NOW, market_open=True,
                     start_worker=False, run_cycle=True)
    assert rep.status("LIVE_FEED_CONNECTED") == A.PENDING_MARKET_SESSION
    assert rep.status("GENUINE_SNAPSHOT_ACTIVE") == A.PASS


# ── 5. not forward-eligible (no benchmark) → snapshot not activated, active preserved ─

def test_not_eligible_snapshot_not_activated(tmp_path, monkeypatch):
    monkeypatch.setattr(CAL, "_now_ist", lambda: _NOW)
    store = SnapshotStore(tmp_path / "snaps")
    # first a good activation
    good_ds = KS.KiteDataSource(FakeKite(), store, universe={"WIN", "FLAT", "WEAK"},
                                benchmark_name="NIFTY 50", history_dir=tmp_path / "h",
                                progress_path=tmp_path / "p.json", sleep_fn=lambda s: None)
    good = good_ds.daily_refresh(now=_NOW).snapshot_id
    # now activate with NO benchmark → committed-not-activated; good stays active
    rep = A.activate(client=FakeKite(with_benchmark=False), store=store, brain=_brain(tmp_path, store),
                     universe={"WIN", "FLAT", "WEAK"}, benchmark_name="NIFTY 50",
                     history_dir=tmp_path / "h2", progress_path=tmp_path / "p2.json",
                     now=_NOW, market_open=True, start_worker=False, run_cycle=True)
    assert rep.status("GENUINE_SNAPSHOT_ACTIVE") == A.FAIL
    assert rep.status("PAPER_AUTO_REAL_DATA_OPERATIONAL") == A.FAIL
    assert store.get_active_snapshot() == good            # previous active preserved
    assert rep.active_pointer == good


# ── 6. worker actually starts (headless) and enables persistent paper-auto ───────

def test_worker_starts_headless(tmp_path, monkeypatch):
    rep, _ = _activate(tmp_path, monkeypatch, market_open=True, start_worker=True)
    assert rep.status("PAPER_AUTO_WORKER_RUNNING") == A.PASS
    assert rep.paper_auto.get("enabled") is True
    assert rep.worker_running is True


# ── 7. data-only client exposes no order surface ─────────────────────────────────

class FakeSession:                                          # mimics the raw SDK (data methods)
    def profile(self):
        return {"user_id": "AB1234", "broker": "ZERODHA"}

    def instruments(self, exchange="NSE"):
        return [{"instrument_token": 111, "tradingsymbol": "WIN", "isin": "INWIN01"}]

    def historical_data(self, token, frm, to, interval="day"):
        return [{"date": "2024-05-01", "open": 100, "high": 101, "low": 99, "close": 100, "volume": 5}]

    # deliberately order-capable on the RAW session
    def place_order(self, *a, **k):
        raise AssertionError("must never be called from the data path")


def test_data_only_client_hides_order_methods():
    c = A.KiteDataClient(FakeSession())
    assert not any(hasattr(c, m) for m in ("place_order", "cancel_order", "modify_order", "place_gtt"))
    assert c.profile()["user_id"] == "AB1234"
    assert c.instruments("NSE")[0]["tradingsymbol"] == "WIN"
    assert c.historical(111, "2024-05-01", "2024-05-01")[0]["close"] == 100
    assert A.ActivationReport().kite_order_reachable is False


# ── 8. KiteTicker→overlay feed bridge: subscribe map, IST ticks, reconnect restore ─

class FakeTicker:
    MODE_LTP = "ltp"
    MODE_QUOTE = "quote"

    def __init__(self):
        self.subscribed, self.mode, self.connected = [], None, False
        self.on_ticks = self.on_connect = self.on_close = self.on_reconnect = None

    def connect(self, threaded=True):
        self.connected = True

    def subscribe(self, toks):
        self.subscribed = list(toks)

    def set_mode(self, mode, toks):
        self.mode = mode


def test_ticker_feed_maps_symbols_and_restores_on_reconnect():
    ov = KL.KiteLiveOverlay(sleep_fn=lambda s: None, clock=lambda: 4_102_511_400.0)
    tk = FakeTicker()
    feed = A.KiteTickerFeed(tk, token_to_symbol={111: "WIN", 222: "FLAT"}, overlay=ov)
    feed.connect()
    feed.subscribe(["WIN", "FLAT"])
    assert set(tk.subscribed) == {111, 222} and tk.mode == "quote"
    # on_connect handler restores approved subscriptions and marks the overlay connected
    tk.subscribed = []
    tk.on_connect(tk, {})
    assert set(tk.subscribed) == {111, 222} and ov.connected is True


def test_ticker_feed_translates_ist_timestamps():
    from datetime import timezone, timedelta
    ist = timezone(timedelta(hours=5, minutes=30))
    naive_ist = datetime(2024, 6, 3, 12, 0, 0)                # exchange stamp: Asia/Kolkata-naive
    epoch = naive_ist.replace(tzinfo=ist).timestamp()
    ov = KL.KiteLiveOverlay(clock=lambda: epoch + 5.0)        # 'now' just after the tick
    tk = FakeTicker()
    feed = A.KiteTickerFeed(tk, token_to_symbol={111: "WIN"}, overlay=ov)
    tk.on_ticks(tk, [{"instrument_token": 111, "last_price": 250.5,
                      "exchange_timestamp": naive_ist}])
    assert ov.price("WIN") == 250.5
    assert abs(ov.last_tick_ts("WIN") - epoch) < 1.0         # interpreted as IST, not server-local
    assert not ov.is_stale("WIN", now=epoch + 5.0)


def test_future_and_out_of_order_ticks_rejected_through_feed():
    ov = KL.KiteLiveOverlay(clock=lambda: 1000.0)
    tk = FakeTicker()
    feed = A.KiteTickerFeed(tk, token_to_symbol={111: "WIN"}, overlay=ov, clock=lambda: 1000.0)
    tk.on_ticks(tk, [{"instrument_token": 111, "last_price": 100.0, "last_trade_time": None}])
    # no exchange ts → clock() = 1000 accepted; a later future ts must be rejected by the overlay
    assert ov.price("WIN") == 100.0
    ov.on_tick("WIN", 200.0, ts=5000.0)                      # future
    assert ov.price("WIN") == 100.0 and ov.health()["rejected"]["future"] == 1
