from datetime import date, datetime, timedelta
from types import SimpleNamespace

import numpy as np
import pandas as pd

from options.directional_selector import black_scholes
from product import fo_runtime


AS_OF = date(2026, 9, 26)
QUOTE_TIME = datetime(2026, 9, 26, 10, 30, 0)


def _breakout_bars():
    n = 90
    x = np.arange(n, dtype=float)
    close = 1000.0 + x * 1.0 + np.sin(x / 2.4) * 10.0
    high = close + 7.0
    low = close - 7.0
    volume = np.full(n, 100_000.0)
    prior_high = float(high[-21:-1].max())
    close[-1] = prior_high + 7.0
    high[-1] = close[-1] + 2.0
    low[-1] = close[-1] - 13.0
    volume[-1] = 250_000.0
    return pd.DataFrame({
        "open": close - 2.0,
        "high": high,
        "low": low,
        "close": close,
        "volume": volume,
    })


def _quiet_bars():
    frame = _breakout_bars()
    frame.loc[frame.index[-1], "close"] = float(frame["close"].iloc[-2])
    frame.loc[frame.index[-1], "high"] = float(frame["high"].iloc[-2])
    frame.loc[frame.index[-1], "low"] = float(frame["low"].iloc[-2])
    frame.loc[frame.index[-1], "volume"] = 100_000.0
    return frame


def _report(spot):
    return SimpleNamespace(underlyings=(
        SimpleNamespace(
            symbol="TEST",
            future_symbol="TEST26OCTFUT",
            instrument_token=1,
            lot_size=50,
            expiry="2026-10-29",
        ),
    ))


def _instruments(spot):
    strike = round(spot / 10.0) * 10.0
    return [
        {
            "instrument_token": 1, "tradingsymbol": "TEST26OCTFUT",
            "name": "TEST", "expiry": "2026-10-29", "strike": 0,
            "tick_size": 0.05, "lot_size": 50, "instrument_type": "FUT",
            "segment": "NFO-FUT", "exchange": "NFO",
        },
        {
            "instrument_token": 2, "tradingsymbol": "TEST26OCTCE",
            "name": "TEST", "expiry": "2026-10-29", "strike": strike,
            "tick_size": 0.05, "lot_size": 50, "instrument_type": "CE",
            "segment": "NFO-OPT", "exchange": "NFO",
        },
        {
            "instrument_token": 3, "tradingsymbol": "TEST26OCTPE",
            "name": "TEST", "expiry": "2026-10-29", "strike": strike,
            "tick_size": 0.05, "lot_size": 50, "instrument_type": "PE",
            "segment": "NFO-OPT", "exchange": "NFO",
        },
    ]


class _Client:
    def __init__(
        self,
        spot,
        instruments,
        *,
        nifty_quote_time=QUOTE_TIME,
        underlying_quote_time=QUOTE_TIME,
        future_quote_time=QUOTE_TIME,
        option_quote_time=QUOTE_TIME,
    ):
        self.spot = spot
        self.instruments_rows = instruments
        self.quote_calls = []
        self.nifty_quote_time = nifty_quote_time
        self.underlying_quote_time = underlying_quote_time
        self.future_quote_time = future_quote_time
        self.option_quote_time = option_quote_time
        self.quote_now = QUOTE_TIME + timedelta(seconds=30)

    def quote(self, keys):
        self.quote_calls.append(list(keys))
        out = {}
        for key in keys:
            if key == "NSE:NIFTY 50":
                out[key] = {
                    "last_price": 22000.0,
                    "ohlc": {"close": 21900.0},
                    "timestamp": self.nifty_quote_time,
                }
            elif key == "NSE:TEST":
                out[key] = {
                    "last_price": self.spot,
                    "average_price": self.spot - 3.0,
                    "depth": {
                        "buy": [{"price": self.spot - 0.1}],
                        "sell": [{"price": self.spot + 0.1}],
                    },
                    "timestamp": self.underlying_quote_time,
                }
            elif key == "NFO:TEST26OCTFUT":
                out[key] = {
                    "last_price": self.spot + 2.0,
                    "oi": 106_000,
                    "timestamp": self.future_quote_time,
                }
            elif key.startswith("NFO:TEST26OCT"):
                kind = "CE" if key.endswith("CE") else "PE"
                strike = next(
                    float(row["strike"]) for row in self.instruments_rows
                    if row["tradingsymbol"] == key.split(":", 1)[1]
                )
                fair = black_scholes(
                    spot=self.spot, strike=strike, dte=33, iv=0.24, option_type=kind,
                )["price"]
                out[key] = {
                    "last_price": fair,
                    "volume": 8000,
                    "oi": 40000,
                    "depth": {
                        "buy": [{"price": max(0.05, fair - 0.15)}],
                        "sell": [{"price": fair + 0.15}],
                    },
                    "timestamp": self.option_quote_time,
                }
        return out

    def historical_with_oi(self, token, from_date, to_date, interval="day"):
        assert token == 1
        return [{"date": "2026-09-25", "close": self.spot - 10.0, "oi": 100_000}]


class _IvHistory:
    def __init__(self, percentile=35.0, prior_sessions=80):
        self.percentile = percentile
        self.prior_sessions = prior_sessions
        self.calls = []

    def percentile_before(self, **kwargs):
        self.calls.append(dict(kwargs))
        return {
            "available": True,
            "percentile_pct": self.percentile,
            "prior_sessions": self.prior_sessions,
            "minimum_prior_sessions": 60,
            "lookback_sessions": 252,
            "current_iv_pct": kwargs["current_iv_pct"],
            "source": "FORWARD_OBSERVED_ATM_IV_CLOSE",
            "historical_backfill": False,
        }


class _NoQuoteClient:
    def quote(self, keys):
        raise AssertionError("no deep quote should be requested for a valid empty prefilter")


def test_runtime_scans_canonical_universe_and_returns_ranked_paper_candidate(monkeypatch):
    bars = _breakout_bars()
    spot = float(bars["close"].iloc[-1])
    instruments = _instruments(spot)
    client = _Client(spot, instruments)

    monkeypatch.setattr(
        fo_runtime,
        "_index_context",
        lambda: {"return_20d_pct": 1.0, "return_5d_pct": 0.5, "last_close": 21900.0},
    )
    monkeypatch.setattr(fo_runtime, "_sector_strength_map", lambda nifty: {"Tech": 1.2})
    monkeypatch.setattr(fo_runtime, "_sector_for", lambda symbol: "Tech")

    result = fo_runtime.run_fo_directional_scan(
        report=_report(spot),
        instrument_rows=instruments,
        client=client,
        as_of=AS_OF,
        quote_now=client.quote_now,
        history_getter=lambda symbol: bars,
    )
    assert result["available"] is True
    assert result["status"] == "READY"
    assert result["universe_size"] == 1
    assert result["prefilter_passed"] == 1
    assert result["deep_evaluated"] == 1
    assert result["decision"] == "PAPER_CANDIDATES"
    assert result["candidate_count"] == 1
    selected = result["candidates"][0]
    assert selected["direction"] == "LONG"
    assert selected["selected_contract"]["option_type"] == "CE"
    assert selected["selected_contract"]["trade_plan"]["entry"] > selected["selected_contract"]["trade_plan"]["stop"]
    assert selected["paper_only"] is True
    assert selected["live_execution_allowed"] is False


def test_live_fno_scan_before_0930_is_valid_no_action_without_market_work():
    bars = _breakout_bars()
    spot = float(bars["close"].iloc[-1])

    def history_should_not_run(_symbol):
        raise AssertionError("pre-09:30 F&O scan must not touch history")

    result = fo_runtime.run_fo_directional_scan(
        report=_report(spot),
        instrument_rows=_instruments(spot),
        client=_NoQuoteClient(),
        as_of=AS_OF,
        quote_now=datetime(2026, 9, 26, 9, 29, 0),
        history_getter=history_should_not_run,
    )

    assert result["available"] is True
    assert result["status"] == "READY"
    assert result["decision"] == "NO_ELIGIBLE_TRADE"
    assert result["reason"] == "FNO_ENTRY_WINDOW_NOT_OPEN"
    assert result["candidate_count"] == 0
    assert result["quote_scope"]["option_contracts_requested"] == 0


def test_empty_breakout_set_is_valid_no_trade_and_does_not_request_deep_quotes():
    bars = _quiet_bars()
    spot = float(bars["close"].iloc[-1])
    result = fo_runtime.run_fo_directional_scan(
        report=_report(spot),
        instrument_rows=_instruments(spot),
        client=_NoQuoteClient(),
        as_of=AS_OF,
        history_getter=lambda symbol: bars,
    )
    assert result["available"] is True
    assert result["status"] == "READY"
    assert result["decision"] == "NO_ELIGIBLE_TRADE"
    assert result["candidate_count"] == 0
    assert result["quote_scope"]["option_contracts_requested"] == 0


def test_default_history_waits_for_current_session_before_prefilter(monkeypatch):
    bars = _breakout_bars()
    spot = float(bars["close"].iloc[-1])
    instruments = _instruments(spot)
    client = _Client(spot, instruments)
    events = []

    from data import nse_live
    from scan import bulk_fetcher

    def _adopt(*, overlay_live=True):
        events.append(("adopt", overlay_live))
        return 500

    def _live_ready(*, apply=True):
        events.append(("live", apply))
        return {
            "ready": True,
            "symbols": 500,
            "source": "kite_quotes",
            "session_date": AS_OF.isoformat(),
            "sessions": 260,
        }

    def _history(symbol):
        events.append(("history", symbol))
        return bars

    monkeypatch.setattr(bulk_fetcher, "adopt_ready_store", _adopt)
    monkeypatch.setattr(bulk_fetcher, "get_cached", _history)
    monkeypatch.setattr(nse_live, "live_session_ready", _live_ready)
    monkeypatch.setattr(
        fo_runtime,
        "_index_context",
        lambda: {"return_20d_pct": 1.0, "return_5d_pct": 0.5, "last_close": 21900.0},
    )
    monkeypatch.setattr(fo_runtime, "_sector_strength_map", lambda nifty: {"Tech": 1.2})
    monkeypatch.setattr(fo_runtime, "_sector_for", lambda symbol: "Tech")

    result = fo_runtime.run_fo_directional_scan(
        report=_report(spot),
        instrument_rows=instruments,
        client=client,
        as_of=AS_OF,
        quote_now=client.quote_now,
    )

    assert events[0] == ("adopt", False)
    assert events[1] == ("live", True)
    assert events.index(("live", True)) < events.index(("history", "TEST"))
    assert result["history_session"]["source"] == "kite_quotes"
    assert result["candidate_count"] == 1


def test_default_history_fails_closed_when_current_session_is_unavailable(monkeypatch):
    bars = _breakout_bars()
    spot = float(bars["close"].iloc[-1])

    from data import nse_live
    from scan import bulk_fetcher

    monkeypatch.setattr(
        bulk_fetcher, "adopt_ready_store",
        lambda *, overlay_live=True: 500,
    )
    monkeypatch.setattr(
        nse_live, "live_session_ready",
        lambda *, apply=True: {
            "ready": False,
            "symbols": 0,
            "source": "",
            "session_date": "",
            "sessions": 260,
        },
    )

    def _unexpected_history(symbol):
        raise AssertionError("history must not be read before current-session readiness")

    monkeypatch.setattr(bulk_fetcher, "get_cached", _unexpected_history)

    result = fo_runtime.run_fo_directional_scan(
        report=_report(spot),
        instrument_rows=_instruments(spot),
        client=_NoQuoteClient(),
        as_of=AS_OF,
    )

    assert result["available"] is False
    assert result["status"] == "BLOCKED"
    assert result["code"] == "FNO_CURRENT_SESSION_UNAVAILABLE"
    assert result["candidates"] == []
    assert result["live_execution_allowed"] is False


def test_live_nifty_quote_aligns_relative_strength_horizon(monkeypatch):
    bars = _breakout_bars()
    spot = float(bars["close"].iloc[-1])
    instruments = _instruments(spot)
    client = _Client(spot, instruments)
    seen = {}

    monkeypatch.setattr(
        fo_runtime,
        "_index_context",
        lambda: {
            "return_20d_pct": 1.0,
            "return_5d_pct": 0.5,
            "base_20d_close": 21_000.0,
            "base_5d_close": 21_800.0,
            "last_close": 21_900.0,
        },
    )

    def _sector_map(nifty_5d):
        seen["nifty_5d"] = nifty_5d
        return {"Tech": 1.2}

    monkeypatch.setattr(fo_runtime, "_sector_strength_map", _sector_map)
    monkeypatch.setattr(fo_runtime, "_sector_for", lambda symbol: "Tech")

    result = fo_runtime.run_fo_directional_scan(
        report=_report(spot),
        instrument_rows=instruments,
        client=client,
        as_of=AS_OF,
        quote_now=client.quote_now,
        history_getter=lambda symbol: bars,
    )

    expected_20d = (22_000.0 / 21_000.0 - 1.0) * 100.0
    expected_5d = (22_000.0 / 21_800.0 - 1.0) * 100.0
    assert abs(result["market_context"]["nifty_20d_return_pct"] - expected_20d) < 0.001
    assert abs(result["market_context"]["nifty_5d_return_pct"] - expected_5d) < 0.001
    assert abs(seen["nifty_5d"] - expected_5d) < 1e-9



def test_runtime_uses_only_prior_forward_iv_history_for_contract_scoring(monkeypatch):
    bars = _breakout_bars()
    spot = float(bars["close"].iloc[-1])
    instruments = _instruments(spot)
    client = _Client(spot, instruments)
    iv_history = _IvHistory(percentile=35.0, prior_sessions=80)

    monkeypatch.setattr(
        fo_runtime,
        "_index_context",
        lambda: {"return_20d_pct": 1.0, "return_5d_pct": 0.5, "last_close": 21900.0},
    )
    monkeypatch.setattr(fo_runtime, "_sector_strength_map", lambda nifty: {"Tech": 1.2})
    monkeypatch.setattr(fo_runtime, "_sector_for", lambda symbol: "Tech")

    result = fo_runtime.run_fo_directional_scan(
        report=_report(spot),
        instrument_rows=instruments,
        client=client,
        as_of=AS_OF,
        quote_now=client.quote_now,
        history_getter=lambda symbol: bars,
        iv_history_store=iv_history,
    )

    assert result["candidate_count"] == 1
    selected = result["candidates"][0]
    assert selected["selected_contract"]["iv_percentile"] == 35.0
    assert selected["selected_contract"]["iv_percentile_available"] is True
    assert selected["selected_contract"]["components"]["iv"] > 0
    assert selected["iv_history"]["prior_sessions"] == 80
    assert selected["iv_history"]["historical_backfill"] is False
    assert len(iv_history.calls) == 1
    call = iv_history.calls[0]
    assert call["symbol"] == "TEST"
    assert call["session"] == AS_OF.isoformat()
    assert 20.0 <= call["current_iv_pct"] <= 30.0


def test_stale_nifty_quote_blocks_current_market_context(monkeypatch):
    bars = _breakout_bars()
    spot = float(bars["close"].iloc[-1])
    instruments = _instruments(spot)
    client = _Client(
        spot,
        instruments,
        nifty_quote_time=QUOTE_TIME - timedelta(minutes=10),
    )

    monkeypatch.setattr(
        fo_runtime,
        "_index_context",
        lambda: {"return_20d_pct": 1.0, "return_5d_pct": 0.5, "last_close": 21900.0},
    )

    result = fo_runtime.run_fo_directional_scan(
        report=_report(spot),
        instrument_rows=instruments,
        client=client,
        as_of=AS_OF,
        history_getter=lambda symbol: bars,
        quote_now=client.quote_now,
    )

    assert result["available"] is False
    assert result["code"] == "CURRENT_NIFTY_QUOTE_UNTRUSTED"
    assert result["quote_provenance"]["reason"] == "QUOTE_STALE"
    assert result["candidates"] == []


def test_stale_underlying_quote_cannot_create_fno_candidate(monkeypatch):
    bars = _breakout_bars()
    spot = float(bars["close"].iloc[-1])
    instruments = _instruments(spot)
    client = _Client(
        spot,
        instruments,
        underlying_quote_time=QUOTE_TIME - timedelta(minutes=10),
    )

    monkeypatch.setattr(
        fo_runtime,
        "_index_context",
        lambda: {"return_20d_pct": 1.0, "return_5d_pct": 0.5, "last_close": 21900.0},
    )
    monkeypatch.setattr(fo_runtime, "_sector_strength_map", lambda nifty: {"Tech": 1.2})
    monkeypatch.setattr(fo_runtime, "_sector_for", lambda symbol: "Tech")

    result = fo_runtime.run_fo_directional_scan(
        report=_report(spot),
        instrument_rows=instruments,
        client=client,
        as_of=AS_OF,
        history_getter=lambda symbol: bars,
        quote_now=client.quote_now,
    )

    assert result["candidate_count"] == 0
    assert result["deep_evaluated"] == 0
    assert result["deep_failures"][0]["stage"] == "current_quotes"
    assert result["deep_failures"][0]["reason"] == "UNDERLYING_QUOTE_STALE"


def test_stale_option_quotes_are_filtered_before_contract_selection(monkeypatch):
    bars = _breakout_bars()
    spot = float(bars["close"].iloc[-1])
    instruments = _instruments(spot)
    client = _Client(
        spot,
        instruments,
        option_quote_time=QUOTE_TIME - timedelta(minutes=10),
    )

    monkeypatch.setattr(
        fo_runtime,
        "_index_context",
        lambda: {"return_20d_pct": 1.0, "return_5d_pct": 0.5, "last_close": 21900.0},
    )
    monkeypatch.setattr(fo_runtime, "_sector_strength_map", lambda nifty: {"Tech": 1.2})
    monkeypatch.setattr(fo_runtime, "_sector_for", lambda symbol: "Tech")

    result = fo_runtime.run_fo_directional_scan(
        report=_report(spot),
        instrument_rows=instruments,
        client=client,
        as_of=AS_OF,
        history_getter=lambda symbol: bars,
        quote_now=client.quote_now,
    )

    assert result["candidate_count"] == 0
    assert result["deep_evaluated"] == 0
    assert result["deep_failures"][0]["stage"] == "option_quotes"
    assert result["deep_failures"][0]["reason"] == "NO_FRESH_COHERENT_OPTION_QUOTES"
    assert all(
        row["reason"] == "QUOTE_STALE"
        for row in result["deep_failures"][0]["rejected_option_quotes"]
    )


def test_fresh_but_incoherent_underlying_future_quotes_fail_closed(monkeypatch):
    bars = _breakout_bars()
    spot = float(bars["close"].iloc[-1])
    instruments = _instruments(spot)
    client = _Client(
        spot,
        instruments,
        underlying_quote_time=QUOTE_TIME - timedelta(seconds=140),
        future_quote_time=QUOTE_TIME,
    )

    monkeypatch.setattr(
        fo_runtime,
        "_index_context",
        lambda: {"return_20d_pct": 1.0, "return_5d_pct": 0.5, "last_close": 21900.0},
    )
    monkeypatch.setattr(fo_runtime, "_sector_strength_map", lambda nifty: {"Tech": 1.2})
    monkeypatch.setattr(fo_runtime, "_sector_for", lambda symbol: "Tech")

    result = fo_runtime.run_fo_directional_scan(
        report=_report(spot),
        instrument_rows=instruments,
        client=client,
        as_of=AS_OF,
        history_getter=lambda symbol: bars,
        quote_now=client.quote_now,
    )

    assert result["candidate_count"] == 0
    assert result["deep_failures"][0]["stage"] == "current_quotes"
    assert result["deep_failures"][0]["reason"] == "UNDERLYING_FUTURE_QUOTE_SKEW_TOO_WIDE"


def test_fresh_but_incoherent_option_quotes_are_filtered(monkeypatch):
    bars = _breakout_bars()
    spot = float(bars["close"].iloc[-1])
    instruments = _instruments(spot)
    client = _Client(
        spot,
        instruments,
        option_quote_time=QUOTE_TIME - timedelta(seconds=140),
    )

    monkeypatch.setattr(
        fo_runtime,
        "_index_context",
        lambda: {"return_20d_pct": 1.0, "return_5d_pct": 0.5, "last_close": 21900.0},
    )
    monkeypatch.setattr(fo_runtime, "_sector_strength_map", lambda nifty: {"Tech": 1.2})
    monkeypatch.setattr(fo_runtime, "_sector_for", lambda symbol: "Tech")

    result = fo_runtime.run_fo_directional_scan(
        report=_report(spot),
        instrument_rows=instruments,
        client=client,
        as_of=AS_OF,
        history_getter=lambda symbol: bars,
        quote_now=client.quote_now,
    )

    assert result["candidate_count"] == 0
    assert result["deep_failures"][0]["stage"] == "option_quotes"
    assert result["deep_failures"][0]["reason"] == "NO_FRESH_COHERENT_OPTION_QUOTES"
    assert all(
        row["reason"] == "OPTION_QUOTE_SKEW_TOO_WIDE"
        for row in result["deep_failures"][0]["rejected_option_quotes"]
    )
