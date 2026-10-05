"""Session replacement through the production controller/feed/overlay path."""
import sys
from types import SimpleNamespace

import pytest

from research.autonomy.live_feed import LiveFeedController
from research.intelligence.data.kite_live import KiteLiveOverlay


@pytest.fixture
def broker(monkeypatch):
    credentials = {"KITE_API_KEY": "key-one", "KITE_ACCESS_TOKEN": "token-one"}
    tickers = []

    class Ticker:
        MODE_LTP = "ltp"
        def __init__(self, **kwargs):
            self.credentials = kwargs
            self.ws = None
            self.closed = self.retry_stopped = False
            self.tokens = []
            tickers.append(self)
        def connect(self, **kwargs):
            self.ws = object()
            self.on_connect(self, {})
        def subscribe(self, tokens): self.tokens = tokens
        def set_mode(self, *args): pass
        def stop_retry(self): self.retry_stopped = True
        def close(self):
            self.closed = True
            self.on_close(self, 1000, "closed")

    monkeypatch.setitem(sys.modules, "kiteconnect", SimpleNamespace(KiteTicker=Ticker))
    monkeypatch.setitem(sys.modules, "data.kite_client", SimpleNamespace(
        _fresh_env=lambda name: credentials.get(name, "")))
    controller = LiveFeedController()
    monkeypatch.setattr(controller, "_tokens", lambda symbols: {i + 1: s for i, s in enumerate(symbols)})
    monkeypatch.setattr(controller, "_quote_overlay", lambda symbols: 0)
    return controller, credentials, tickers


@pytest.mark.parametrize("field,new_value", [
    ("KITE_ACCESS_TOKEN", "token-two"), ("KITE_API_KEY", "key-two"),
])
def test_changed_session_replaces_socket_and_discards_old_prices(broker, field, new_value):
    ctl, credentials, tickers = broker
    ctl.start(["AAA"])
    old_overlay = ctl.overlay
    old = tickers[0]
    old.on_ticks(old, [{"instrument_token": 1, "last_price": 100}])
    assert ctl.entry_allowed("AAA")
    credentials[field] = new_value
    ctl.start(["BBB"])
    assert len(tickers) == 2
    assert old.closed and old.retry_stopped
    assert tickers[1].credentials == {"api_key": credentials["KITE_API_KEY"],
                                      "access_token": credentials["KITE_ACCESS_TOKEN"]}
    assert ctl.subscribed == {"AAA", "BBB"}
    assert ctl.overlay is not old_overlay
    assert ctl.price("AAA") is None
    old.on_ticks(old, [{"instrument_token": 1, "last_price": 999}])
    old.on_close(old, 403, "late retired callback")
    assert ctl.overlay.connected
    assert ctl.price("AAA") is None
    tickers[1].on_ticks(tickers[1], [{"instrument_token": 2, "last_price": 101}])
    assert ctl.price("AAA") == 101
    assert ctl.entry_allowed("AAA")


def test_unchanged_session_reuses_socket(broker):
    ctl, _, tickers = broker
    ctl.start(["AAA"])
    ctl.start(["AAA", "BBB"])
    assert len(tickers) == 1
    assert not tickers[0].closed
    assert ctl.subscribed == {"AAA", "BBB"}


def test_removed_credentials_retire_socket_without_reopening(broker):
    ctl, credentials, tickers = broker
    ctl.start(["AAA"])
    credentials["KITE_ACCESS_TOKEN"] = ""
    health = ctl.start(["AAA"])
    assert tickers[0].closed and tickers[0].retry_stopped
    assert len(tickers) == 1
    assert not health["connected"]
    assert "credentials" in health["last_error"]


def test_rest_fallback_refreshes_expired_quotes_despite_lifetime_tick_count(monkeypatch):
    now = [1000.0]
    monkeypatch.setattr("research.autonomy.live_feed.time.time", lambda: now[0])
    overlay = KiteLiveOverlay(clock=lambda: now[0])
    ctl = LiveFeedController(overlay=overlay)
    calls = []
    def prices(symbols):
        calls.append(symbols)
        return {"AAA": 100 + len(calls)}
    monkeypatch.setitem(sys.modules, "data.kite_client", SimpleNamespace(
        _fresh_env=lambda name: "configured", KiteClient=lambda: SimpleNamespace(get_ltp=prices)))
    ctl.start(["AAA"])
    assert ctl.price("AAA") == 101
    now[0] += 5
    ctl.start(["AAA"])
    assert len(calls) == 1
    now[0] += 31
    assert overlay.health()["symbols_ticking"] == 1
    assert not ctl.entry_allowed("AAA")
    ctl.start(["AAA"])
    assert len(calls) == 2
    assert ctl.price("AAA") == 102
    assert ctl.entry_allowed("AAA")
