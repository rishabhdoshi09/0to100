"""Tests for the dynamic universe provider."""
from sq_ai.backend.universe import _filter_eq, _normalise, _yf_suffix, get_active_universe
from sq_ai.portfolio.tracker import PortfolioTracker


def test_yf_suffix_idempotent():
    assert _yf_suffix("RELIANCE") == "RELIANCE.NS"
    assert _yf_suffix("RELIANCE.NS") == "RELIANCE.NS"


def test_filter_eq_keeps_only_nse_eq():
    # _filter_eq works on normalised rows (post _normalise)
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
    normalised = [_normalise(r) for r in raw]
    out = _filter_eq(normalised)
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


def test_cache_instruments_stores_extended_fields(tmp_path):
    tracker = PortfolioTracker(db_path=str(tmp_path / "t.db"))
    tracker.cache_instruments([{
        "trading_symbol": "RELIANCE",
        "instrument_token": 738561,
        "name": "Reliance Industries",
        "instrument_type": "EQ",
        "segment": "NSE",
        "lot_size": 1,
        "tick_size": 0.05,
    }])
    rows = tracker.get_cached_instruments()
    assert len(rows) == 1
    r = rows[0]
    assert r["instrument_type"] == "EQ"
    assert r["segment"] == "NSE"
    assert r["lot_size"] == 1
    assert abs(r["tick_size"] - 0.05) < 1e-9


def test_api_universe_search(tmp_path):
    """GET /api/universe?q= filters by symbol/name."""
    import os
    os.environ["SQ_AUTOSTART_SCHEDULER"] = "false"
    db = str(tmp_path / "uni.db")
    os.environ["SQ_DB_PATH"] = db

    from fastapi.testclient import TestClient
    from sq_ai.api.app import app

    with TestClient(app) as cl:
        # seed instruments
        cl.app.state.sched.tracker.cache_instruments([
            {"trading_symbol": "RELIANCE", "instrument_token": 1,
             "name": "Reliance Industries", "instrument_type": "EQ",
             "segment": "NSE", "lot_size": 1, "tick_size": 0.05},
            {"trading_symbol": "TCS", "instrument_token": 2,
             "name": "Tata Consultancy", "instrument_type": "EQ",
             "segment": "NSE", "lot_size": 1, "tick_size": 0.05},
        ])
        r = cl.get("/api/universe", params={"q": "RELI"})
        assert r.status_code == 200
        symbols = [row["trading_symbol"] for row in r.json()]
        assert "RELIANCE" in symbols
        assert "TCS" not in symbols


def test_api_ltp_empty_without_symbols():
    import os
    os.environ["SQ_AUTOSTART_SCHEDULER"] = "false"
    from fastapi.testclient import TestClient
    from sq_ai.api.app import app

    with TestClient(app) as cl:
        r = cl.get("/api/ltp")
        assert r.status_code == 200
        assert r.json() == {}
