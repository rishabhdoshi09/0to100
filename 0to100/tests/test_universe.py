"""Tests for the dynamic universe provider."""
from sq_ai.backend.universe import _filter_eq, _yf_suffix, get_active_universe
from sq_ai.portfolio.tracker import PortfolioTracker


def test_yf_suffix_idempotent():
    assert _yf_suffix("RELIANCE") == "RELIANCE.NS"
    assert _yf_suffix("RELIANCE.NS") == "RELIANCE.NS"


def test_filter_eq_keeps_only_nse_eq():
    raw = [
        {"tradingsymbol": "RELIANCE", "instrument_token": 1, "name": "Reliance",
         "segment": "NSE", "instrument_type": "EQ"},
        {"tradingsymbol": "BANKNIFTY24DEC", "segment": "NFO-OPT",
         "instrument_type": "OPT"},
        {"tradingsymbol": "INFY", "instrument_token": 2, "name": "Infy",
         "segment": "NSE", "instrument_type": "EQ"},
        {"tradingsymbol": "GOLDBEES", "segment": "NSE",
         "instrument_type": "ETF"},
    ]
    out = _filter_eq(raw)
    assert {r["trading_symbol"] for r in out} == {"RELIANCE", "INFY"}


def test_get_active_universe_uses_cache(tmp_path):
    tracker = PortfolioTracker(db_path=str(tmp_path / "t.db"))
    tracker.cache_instruments([
        {"trading_symbol": "RELIANCE", "instrument_token": 1, "name": ""},
        {"trading_symbol": "INFY", "instrument_token": 2, "name": ""},
        {"trading_symbol": "TCS", "instrument_token": 3, "name": ""},
    ])
    out = get_active_universe(max_symbols=2, tracker=tracker)
    assert len(out) == 2
    assert all(s.endswith(".NS") for s in out)


def test_get_active_universe_falls_back_to_yaml(tmp_path):
    tracker = PortfolioTracker(db_path=str(tmp_path / "t.db"))
    out = get_active_universe(max_symbols=10, tracker=tracker,
                              fallback_yaml=["A.NS", "B"])
    assert out == ["A.NS", "B.NS"]
