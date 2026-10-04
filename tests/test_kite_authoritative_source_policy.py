from __future__ import annotations

from datetime import date, timedelta

import pandas as pd
import pytest

import data.live_quotes as LQ
import data.market_data as MD
import product.due_diligence.acquire as ACQ
from product.due_diligence.option_chain import summarize_kite_option_chain
import scan.bulk_fetcher as BF


@pytest.fixture(autouse=True)
def _isolate_provider_caches(monkeypatch):
    monkeypatch.setattr(MD, "_provider", None)
    monkeypatch.setattr(MD, "_provider_session", None)
    monkeypatch.setattr(LQ, "_qcache", {})
    monkeypatch.setattr(BF, "_kite_cache", {})
    monkeypatch.setattr(BF, "_kite_snapshot_id", "")


def test_cached_kite_quote_provider_adopts_token_rotation(monkeypatch):
    import data.kite_client as KC

    credentials = {"KITE_API_KEY": "test-key", "KITE_ACCESS_TOKEN": "old-token"}
    monkeypatch.setattr(KC, "_fresh_env", lambda name, default="": credentials.get(name, default))
    clients = []

    class FakeClient:
        def __init__(self):
            self.token = credentials["KITE_ACCESS_TOKEN"]
            clients.append(self)

    monkeypatch.setattr(KC, "KiteClient", FakeClient)
    first = MD.get_provider()
    assert first._kite_client.token == "old-token"
    assert MD.get_provider() is first
    credentials["KITE_ACCESS_TOKEN"] = "fresh-token"
    refreshed = MD.get_provider()
    assert refreshed is not first
    assert refreshed._kite_client.token == "fresh-token"
    assert len(clients) == 2
    credentials["KITE_ACCESS_TOKEN"] = ""
    assert isinstance(MD.get_provider(), MD._GoogleFinanceProvider)


def test_rotated_kite_provider_construction_failure_does_not_return_old_quotes(monkeypatch):
    import data.kite_client as KC

    monkeypatch.setattr(MD, "_kite_available", lambda: True)
    identity = ["old-session"]
    monkeypatch.setattr(KC, "kite_session_identity", lambda: identity[0])
    monkeypatch.setattr(KC, "KiteClient", lambda: object())
    first = MD.get_provider()
    identity[0] = "fresh-session"

    def rejected_client():
        raise RuntimeError("Kite construction failed")

    monkeypatch.setattr(KC, "KiteClient", rejected_client)
    with pytest.raises(RuntimeError, match="Kite construction failed"):
        MD.get_provider()
    assert MD._provider is first
    assert MD._provider_session == "old-session"


@pytest.mark.parametrize("kite_answer", [True, False])
def test_login_rejects_warm_public_quotes_even_when_kite_has_a_gap(monkeypatch, kite_answer):
    import data.kite_client as KC

    connected = [False]
    monkeypatch.setattr(KC, "kite_credentials_available", lambda: connected[0])
    monkeypatch.setattr(LQ, "_nse_quotes", lambda symbols: {
        s: {"price": 99.0, "source": "nse"} for s in symbols
    })
    monkeypatch.setattr(LQ, "_google_quotes", lambda symbols: {})
    calls = []

    def kite_quotes(symbols):
        calls.append(symbols)
        return {s: {"price": 101.0, "source": "kite"} for s in symbols} if kite_answer else {}

    monkeypatch.setattr(LQ, "_kite_quotes", kite_quotes)
    assert LQ.get_live_quotes(["INFY"])["INFY"]["source"] == "nse"
    connected[0] = True

    result = LQ.get_live_quotes(["INFY"])
    assert calls == [["INFY"]]
    if kite_answer:
        assert result["INFY"] == {"price": 101.0, "source": "kite"}
    else:
        assert result == {}


def test_auto_historical_does_not_fallback_when_kite_is_authoritative(monkeypatch):
    called = {"yf": 0}

    monkeypatch.setattr(MD, "_kite_available", lambda: True)
    monkeypatch.setattr(
        MD,
        "get_historical_data_kite",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("kite unavailable")),
    )

    def _yf(*args, **kwargs):
        called["yf"] += 1
        return pd.DataFrame({"close": [1.0]})

    monkeypatch.setattr(MD, "get_historical_data_yfinance", _yf)

    with pytest.raises(RuntimeError, match="kite unavailable"):
        MD.get_historical_data("INFY")

    assert called["yf"] == 0


def test_kite_provider_quotes_preserve_full_kite_market_fields():
    class FakeKite:
        def get_quote(self, keys):
            assert keys == ["NSE:INFY"]
            return {
                "NSE:INFY": {
                    "last_price": 1500.0,
                    "volume": 123456,
                    "average_price": 1492.5,
                    "last_quantity": 7,
                    "buy_quantity": 1000,
                    "sell_quantity": 1200,
                    "oi": 321,
                    "oi_day_high": 400,
                    "oi_day_low": 250,
                    "net_change": 18.5,
                    "instrument_token": 408065,
                    "timestamp": "2026-10-03T10:00:00+05:30",
                    "last_trade_time": "2026-10-03T09:59:59+05:30",
                    "ohlc": {
                        "open": 1480.0,
                        "high": 1510.0,
                        "low": 1470.0,
                        "close": 1481.5,
                    },
                    "depth": {
                        "buy": [{"price": 1499.9, "quantity": 100}],
                        "sell": [{"price": 1500.1, "quantity": 80}],
                    },
                }
            }

    provider = MD._KiteProvider.__new__(MD._KiteProvider)
    provider._kite_client = FakeKite()
    provider._exchange = "NSE"

    row = provider.quotes(["INFY"])["INFY"]

    assert row["source"] == "kite"
    assert row["price"] == 1500.0
    assert row["chg_pct"] == pytest.approx(18.5 / 1481.5 * 100)
    assert row["volume"] == 123456
    assert row["average_price"] == 1492.5
    assert row["last_quantity"] == 7
    assert row["buy_quantity"] == 1000
    assert row["sell_quantity"] == 1200
    assert row["oi"] == 321
    assert row["depth"]["buy"][0]["price"] == 1499.9
    assert row["instrument_token"] == 408065


@pytest.mark.parametrize("price,net_change,expected_pct", [
    (1021.0, 21.0, 2.1), (979.0, -21.0, -2.1), (1000.0, 0.0, 0.0),
])
def test_kite_absolute_change_is_normalized_to_percent_in_every_quote_path(
        monkeypatch, price, net_change, expected_pct):
    import data.kite_client as KC

    payload = {"last_price": price, "net_change": net_change,
               "ohlc": {"close": 1000.0}}

    class FakeKite:
        def quote(self, keys):
            return {key: payload for key in keys}

    monkeypatch.setattr(KC, "KiteConnect", lambda **kwargs: FakeKite())
    client = KC.KiteClient(api_key="test", access_token="", api_secret="")
    monkeypatch.setattr(KC, "KiteClient", lambda: client)
    monkeypatch.setattr(KC, "kite_credentials_available", lambda: True)

    batch = client.batch_quotes(["INFY"])["INFY"]
    assert batch["change"] == pytest.approx(expected_pct)
    assert batch["raw"]["net_change"] == net_change
    assert MD._KiteProvider._normalize_quote(payload)["chg_pct"] == pytest.approx(expected_pct)
    assert LQ.get_live_quotes(["INFY"], ttl=0)["INFY"]["chg_pct"] == pytest.approx(expected_pct)


def test_kite_provider_history_closes_never_calls_yahoo(monkeypatch):
    idx = pd.date_range("2026-09-01", periods=25, freq="B")
    frame = pd.DataFrame({"close": range(100, 125)}, index=idx)

    monkeypatch.setattr(MD, "get_historical_data_kite", lambda *a, **k: frame)
    monkeypatch.setattr(
        MD,
        "_yf_history_closes",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("Yahoo must not be used")),
    )

    provider = MD._KiteProvider.__new__(MD._KiteProvider)
    provider._exchange = "NSE"
    provider._kite_client = object()

    assert provider.history_closes("INFY", days=5) == [120.0, 121.0, 122.0, 123.0, 124.0]


def test_live_quotes_do_not_substitute_nse_or_google_when_kite_is_authoritative(monkeypatch):
    import data.kite_client as KC

    monkeypatch.setattr(KC, "kite_credentials_available", lambda: True)
    monkeypatch.setattr(LQ, "_kite_quotes", lambda symbols: {})
    monkeypatch.setattr(
        LQ,
        "_nse_quotes",
        lambda symbols: (_ for _ in ()).throw(AssertionError("NSE fallback forbidden")),
    )
    monkeypatch.setattr(
        LQ,
        "_google_quotes",
        lambda symbols: (_ for _ in ()).throw(AssertionError("Google fallback forbidden")),
    )

    assert LQ.get_live_quotes(["INFY"], ttl=0) == {}


def test_kite_option_chain_summary_uses_kite_oi_and_derived_iv():
    contracts = [
        {
            "symbol": "ABC26OCT100CE",
            "option_type": "CE",
            "strike": 100.0,
            "expiry": "2026-10-29",
            "oi": 1000,
            "volume": 500,
            "iv": 22.0,
        },
        {
            "symbol": "ABC26OCT100PE",
            "option_type": "PE",
            "strike": 100.0,
            "expiry": "2026-10-29",
            "oi": 1500,
            "volume": 700,
            "iv": 24.0,
        },
        {
            "symbol": "ABC26OCT105CE",
            "option_type": "CE",
            "strike": 105.0,
            "expiry": "2026-10-29",
            "oi": 700,
            "volume": 300,
            "iv": 21.0,
        },
        {
            "symbol": "ABC26OCT105PE",
            "option_type": "PE",
            "strike": 105.0,
            "expiry": "2026-10-29",
            "oi": 900,
            "volume": 350,
            "iv": 25.0,
        },
    ]

    out = summarize_kite_option_chain(contracts, spot=101.0)

    assert out["available"] is True
    assert out["source"] == "ZERODHA_KITE_NFO_READ_ONLY"
    assert out["call_oi"] == 1700
    assert out["put_oi"] == 2400
    assert out["atm_strike"] == 100.0
    assert out["atm_iv"] == 23.0
    assert out["iv_source"] == "IMPLIED_FROM_KITE_MARKET_QUOTES"


def test_due_diligence_option_chain_routes_only_to_kite_when_configured(monkeypatch):
    import data.kite_client as KC

    sentinel = {
        "step": {"id": "option_chain", "ok": True, "source": "kite"},
        "download": {},
        "snapshot": {"available": True, "source": "ZERODHA_KITE_NFO_READ_ONLY"},
    }
    monkeypatch.setattr(KC, "kite_credentials_available", lambda: True)
    monkeypatch.setattr(ACQ, "_fetch_option_chain_kite", lambda symbol: sentinel)
    monkeypatch.setattr(
        ACQ,
        "_fetch_option_chain_nse",
        lambda symbol, session: (_ for _ in ()).throw(
            AssertionError("NSE option-chain fallback forbidden while Kite is configured")
        ),
    )

    assert ACQ._fetch_option_chain("ABC", object()) is sentinel


def test_scan_prefetch_uses_active_kite_snapshot_not_bhavcopy_or_yahoo(monkeypatch, tmp_path):
    from research.intelligence.data import snapshot_store as SS

    store = SS.SnapshotStore(tmp_path / "snapshots")
    start = date(2026, 7, 1)
    rows = []
    for i in range(45):
        d = start + timedelta(days=i)
        rows.append(("INFY", d.isoformat(), 100 + i, 101 + i, 99 + i, 100.5 + i, 1000 + i, "EQ"))
    sid = store.commit_snapshot(
        rows,
        index_rows=[("NIFTY", start.isoformat(), 25000, 25100, 24900, 25050)],
        extra_manifest={
            "source": "kite",
            "has_universe_history": True,
            "adjustment_consistent": True,
            "corporate_action_coverage": 1.0,
            "missing_session_rate": 0.0,
            "validation_errors": 0,
        },
    )
    store.activate_snapshot(sid)

    monkeypatch.setattr(BF, "_kite_authoritative", lambda: True)
    monkeypatch.setattr(SS, "SnapshotStore", lambda *a, **k: store)
    monkeypatch.setattr(
        BF,
        "_repair_via_kite",
        lambda *a, **k: (_ for _ in ()).throw(
            AssertionError("snapshot already covers INFY; no broker repair needed")
        ),
    )
    monkeypatch.setattr(
        BF,
        "adopt_ready_store",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("bhavcopy path forbidden")),
    )
    monkeypatch.setattr(
        BF,
        "_prefetch_yf",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("Yahoo path forbidden")),
    )

    BF._kite_cache.clear()
    BF._kite_snapshot_id = ""

    assert BF.prefetch(["INFY"]) == 1
    frame = BF.get_cached("INFY")
    assert frame is not None
    assert len(frame) == 45
    assert frame.attrs["quantterm_source"] == "kite_snapshot"
    assert frame.attrs["quantterm_snapshot_id"] == sid


@pytest.mark.parametrize("replacement", ["new_snapshot", "removed_pointer"])
def test_history_cache_tracks_active_snapshot_changes(monkeypatch, tmp_path, replacement):
    from research.intelligence.data import snapshot_store as SS

    store = SS.SnapshotStore(tmp_path / "snapshots")
    start = date(2026, 7, 1)
    rows = [("INFY", (start + timedelta(days=i)).isoformat(), 100, 102, 99, 101, 1000, "EQ")
            for i in range(45)]
    sid = store.commit_snapshot(rows, extra_manifest={"source": "kite"})
    store.activate_snapshot(sid)
    monkeypatch.setattr(BF, "_kite_authoritative", lambda: True)
    monkeypatch.setattr(SS, "SnapshotStore", lambda *a, **k: store)
    assert BF._adopt_active_kite_snapshot(["INFY"]) == 1
    assert BF.get_cached("INFY") is not None

    if replacement == "new_snapshot":
        # A new active snapshot covers a different universe. INFY is a real gap.
        newer = store.commit_snapshot([("TCS", *row[1:]) for row in rows],
                                      extra_manifest={"source": "kite"})
        store.activate_snapshot(newer)
    else:
        (store.root / "ACTIVE").unlink()

    assert BF.get_cached("INFY") is None
    assert "INFY" not in BF.cached_symbols()
    assert BF.is_warm() is False
