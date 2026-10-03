from __future__ import annotations

from datetime import date, timedelta

import pandas as pd
import pytest

import data.live_quotes as LQ
import data.market_data as MD
import product.due_diligence.acquire as ACQ
from product.due_diligence.option_chain import summarize_kite_option_chain
import scan.bulk_fetcher as BF


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
                    "change": 1.25,
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
    assert row["volume"] == 123456
    assert row["average_price"] == 1492.5
    assert row["last_quantity"] == 7
    assert row["buy_quantity"] == 1000
    assert row["sell_quantity"] == 1200
    assert row["oi"] == 321
    assert row["depth"]["buy"][0]["price"] == 1499.9
    assert row["instrument_token"] == 408065


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
