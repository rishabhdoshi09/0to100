"""Broker rejection must not turn dashboard reads into a retry storm."""
from concurrent.futures import ThreadPoolExecutor
from datetime import date

import pandas as pd
import pytest

from data import index_store as idx
from data import kite_client as kc
from research.intelligence.data import nse_calendar as cal


@pytest.fixture
def session(monkeypatch, tmp_path):
    credentials = {"KITE_API_KEY": "test-key", "KITE_ACCESS_TOKEN": "rejected-token"}
    monkeypatch.setattr(kc, "_fresh_env", lambda name, default="": credentials.get(name, default))
    monkeypatch.setattr(idx, "_DIR", tmp_path)
    monkeypatch.setattr(idx, "_KITE_PKL", tmp_path / "index_store_kite.pkl")
    monkeypatch.setattr(idx, "_kite_store", {})
    monkeypatch.setattr(idx, "_kite_last_day", None)
    monkeypatch.setattr(idx, "_kite_failed_session", None)
    monkeypatch.setattr(idx, "_kite_retry_after", 0.0)
    monkeypatch.setattr(cal, "latest_required_session", lambda *args: date(2026, 10, 2))
    return credentials


def reject_instruments(monkeypatch):
    attempts = []

    class RejectedKite:
        def get_instruments(self, exchange):
            attempts.append(exchange)
            raise RuntimeError("Incorrect api_key or access_token")

    monkeypatch.setattr(kc, "KiteClient", RejectedKite)
    return attempts


def test_concurrent_index_readers_share_one_rejected_attempt(session, monkeypatch):
    attempts = reject_instruments(monkeypatch)
    monkeypatch.setattr(idx, "_store", {"Nifty 50": pd.DataFrame({"Close": [99.0, 100.0]})})

    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(lambda _: idx.latest_index_print("^NSEI"), range(16)))

    assert results == [None] * 16
    assert attempts == ["NSE"]
    # A warm official cache cannot fill a gap in the configured Kite session.
    assert idx.recent_index_closes("^NSEI") == []
    assert attempts == ["NSE"]


def test_failed_session_retries_after_bounded_cooldown(session, monkeypatch):
    clock = [100.0]
    monkeypatch.setattr(idx.time, "monotonic", lambda: clock[0])
    attempts = reject_instruments(monkeypatch)
    assert idx._build_kite_index_store() == 0
    clock[0] += idx._BUILD_COOLDOWN_S - 0.01
    assert idx._build_kite_index_store() == 0
    assert attempts == ["NSE"]
    clock[0] += 0.01
    assert idx._build_kite_index_store() == 0
    assert attempts == ["NSE", "NSE"]


def test_fresh_login_bypasses_cooldown_and_restores_index_history(session, monkeypatch):
    clock = [100.0]
    monkeypatch.setattr(idx.time, "monotonic", lambda: clock[0])
    attempts = reject_instruments(monkeypatch)
    assert idx._build_kite_index_store() == 0
    assert attempts == ["NSE"]

    frame = pd.DataFrame(
        {"close": [100.0] * 220},
        index=pd.date_range(end="2026-10-02", periods=220),
    )
    monkeypatch.setattr(idx, "TICKER_MAP", {"^NSEI": "Nifty 50"})
    monkeypatch.setattr(idx, "_KITE_INDEX_SYMBOLS", {"^NSEI": "NIFTY 50"})

    class AcceptedKite:
        def get_instruments(self, exchange):
            attempts.append(exchange)
            return [{"tradingsymbol": "NIFTY 50", "instrument_token": 256265}]

        def get_historical(self, *args):
            return frame.copy()

    session["KITE_ACCESS_TOKEN"] = "fresh-token"
    monkeypatch.setattr(kc, "KiteClient", AcceptedKite)
    assert idx._build_kite_index_store(220) == 1
    assert attempts == ["NSE", "NSE"]
    assert idx._kite_failed_session is None
    assert idx._kite_retry_after == 0.0
    result = idx._kite_index_frame("^NSEI")
    assert len(result) == 220
    assert result.attrs["quantterm_source"] == "kite_index_history"


@pytest.mark.parametrize("field", ["KITE_API_KEY", "KITE_ACCESS_TOKEN"])
def test_session_identity_changes_with_current_credentials(session, field):
    before = kc.kite_session_identity()
    assert len(before) == 64
    assert session[field] not in before
    assert kc.kite_session_identity() == before
    session[field] += "-rotated"
    assert kc.kite_session_identity() != before
