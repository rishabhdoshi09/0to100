from datetime import date, timedelta
import inspect

from options.directional_selector import black_scholes
from product.fo_iv_history import (
    FoIvHistoryStore,
    MIN_PRIOR_SESSIONS,
    collect_close_iv_snapshot,
    representative_atm_iv_pct,
)


AS_OF = date(2026, 9, 27)


def _option_rows():
    return [
        {
            "instrument_token": 11,
            "tradingsymbol": "ABC26OCTCE",
            "name": "ABC",
            "expiry": "2026-10-29",
            "strike": 100.0,
            "tick_size": 0.05,
            "lot_size": 100,
            "instrument_type": "CE",
            "segment": "NFO-OPT",
            "exchange": "NFO",
        },
        {
            "instrument_token": 12,
            "tradingsymbol": "ABC26OCTPE",
            "name": "ABC",
            "expiry": "2026-10-29",
            "strike": 100.0,
            "tick_size": 0.05,
            "lot_size": 100,
            "instrument_type": "PE",
            "segment": "NFO-OPT",
            "exchange": "NFO",
        },
        {
            "instrument_token": 13,
            "tradingsymbol": "ABC26NOVCE",
            "name": "ABC",
            "expiry": "2026-11-26",
            "strike": 100.0,
            "tick_size": 0.05,
            "lot_size": 100,
            "instrument_type": "CE",
            "segment": "NFO-OPT",
            "exchange": "NFO",
        },
        {
            "instrument_token": 14,
            "tradingsymbol": "ABC26NOVPE",
            "name": "ABC",
            "expiry": "2026-11-26",
            "strike": 100.0,
            "tick_size": 0.05,
            "lot_size": 100,
            "instrument_type": "PE",
            "segment": "NFO-OPT",
            "exchange": "NFO",
        },
    ]


def _option_quotes(rows, *, spot=100.0, iv=0.25):
    out = {}
    for row in rows:
        dte = (date.fromisoformat(str(row["expiry"])) - AS_OF).days
        fair = black_scholes(
            spot=spot,
            strike=float(row["strike"]),
            dte=dte,
            iv=iv,
            option_type=str(row["instrument_type"]),
        )["price"]
        out[str(row["tradingsymbol"])] = {
            "last_price": fair,
            "volume": 5000,
            "oi": 25000,
            "depth": {
                "buy": [{"price": max(0.05, fair - 0.05)}],
                "sell": [{"price": fair + 0.05}],
            },
        }
    return out


def test_forward_iv_percentile_requires_prior_sessions_and_excludes_current_session(tmp_path):
    path = tmp_path / "iv.sqlite3"
    current_session = date(2026, 4, 1)

    with FoIvHistoryStore(path) as store:
        for i in range(MIN_PRIOR_SESSIONS - 1):
            session = current_session - timedelta(days=MIN_PRIOR_SESSIONS - i)
            store.upsert(symbol="ABC", session=session.isoformat(), iv_pct=20.0 + i)

        thin = store.percentile_before(
            symbol="ABC",
            session=current_session.isoformat(),
            current_iv_pct=40.0,
        )
        assert thin["available"] is False
        assert thin["prior_sessions"] == MIN_PRIOR_SESSIONS - 1

        extra_session = current_session - timedelta(days=1)
        store.upsert(symbol="ABC", session=extra_session.isoformat(), iv_pct=79.0)
        ready = store.percentile_before(
            symbol="ABC",
            session=current_session.isoformat(),
            current_iv_pct=40.0,
        )
        assert ready["available"] is True
        assert ready["prior_sessions"] == MIN_PRIOR_SESSIONS
        assert ready["percentile_pct"] is not None

        # Updating the current session must never enter its own percentile base.
        store.upsert(symbol="ABC", session=current_session.isoformat(), iv_pct=200.0)
        same = store.percentile_before(
            symbol="ABC",
            session=current_session.isoformat(),
            current_iv_pct=40.0,
        )
        assert same["prior_sessions"] == MIN_PRIOR_SESSIONS
        assert same["percentile_pct"] == ready["percentile_pct"]
        assert same["historical_backfill"] is False


def test_representative_atm_iv_requires_nearest_expiry_ce_and_pe():
    rows = _option_rows()
    quotes = _option_quotes(rows, iv=0.25)
    observed = representative_atm_iv_pct(rows, quotes, spot=100.0, as_of=AS_OF)

    assert observed["available"] is True
    assert observed["contracts"] == ["ABC26OCTCE", "ABC26OCTPE"]
    assert 24.0 <= observed["iv_pct"] <= 26.0
    assert observed["source"] == "FORWARD_OBSERVED_ATM_IV_CLOSE"

    missing = dict(quotes)
    missing.pop("ABC26OCTPE")
    unavailable = representative_atm_iv_pct(rows, missing, spot=100.0, as_of=AS_OF)
    assert unavailable["available"] is False


def test_representative_atm_iv_uses_one_common_strike_not_independent_nearest_legs():
    rows = [
        {
            "instrument_token": 21,
            "tradingsymbol": "ABC26OCT100CE",
            "name": "ABC",
            "expiry": "2026-10-29",
            "strike": 100.0,
            "lot_size": 100,
            "instrument_type": "CE",
            "segment": "NFO-OPT",
            "exchange": "NFO",
        },
        {
            "instrument_token": 22,
            "tradingsymbol": "ABC26OCT101PE",
            "name": "ABC",
            "expiry": "2026-10-29",
            "strike": 101.0,
            "lot_size": 100,
            "instrument_type": "PE",
            "segment": "NFO-OPT",
            "exchange": "NFO",
        },
        {
            "instrument_token": 23,
            "tradingsymbol": "ABC26OCT105CE",
            "name": "ABC",
            "expiry": "2026-10-29",
            "strike": 105.0,
            "lot_size": 100,
            "instrument_type": "CE",
            "segment": "NFO-OPT",
            "exchange": "NFO",
        },
        {
            "instrument_token": 24,
            "tradingsymbol": "ABC26OCT105PE",
            "name": "ABC",
            "expiry": "2026-10-29",
            "strike": 105.0,
            "lot_size": 100,
            "instrument_type": "PE",
            "segment": "NFO-OPT",
            "exchange": "NFO",
        },
    ]
    quotes = _option_quotes(rows, spot=100.0, iv=0.25)
    observed = representative_atm_iv_pct(rows, quotes, spot=100.0, as_of=AS_OF)

    assert observed["available"] is True
    assert observed["contracts"] == ["ABC26OCT105CE", "ABC26OCT105PE"]


def test_representative_atm_iv_rejects_nearest_expiry_without_common_strike():
    rows = [
        {
            "instrument_token": 31,
            "tradingsymbol": "ABC26OCT100CE",
            "name": "ABC",
            "expiry": "2026-10-29",
            "strike": 100.0,
            "lot_size": 100,
            "instrument_type": "CE",
            "segment": "NFO-OPT",
            "exchange": "NFO",
        },
        {
            "instrument_token": 32,
            "tradingsymbol": "ABC26OCT101PE",
            "name": "ABC",
            "expiry": "2026-10-29",
            "strike": 101.0,
            "lot_size": 100,
            "instrument_type": "PE",
            "segment": "NFO-OPT",
            "exchange": "NFO",
        },
        {
            "instrument_token": 33,
            "tradingsymbol": "ABC26NOV100CE",
            "name": "ABC",
            "expiry": "2026-11-26",
            "strike": 100.0,
            "lot_size": 100,
            "instrument_type": "CE",
            "segment": "NFO-OPT",
            "exchange": "NFO",
        },
        {
            "instrument_token": 34,
            "tradingsymbol": "ABC26NOV100PE",
            "name": "ABC",
            "expiry": "2026-11-26",
            "strike": 100.0,
            "lot_size": 100,
            "instrument_type": "PE",
            "segment": "NFO-OPT",
            "exchange": "NFO",
        },
    ]
    quotes = _option_quotes(rows, spot=100.0, iv=0.25)
    observed = representative_atm_iv_pct(rows, quotes, spot=100.0, as_of=AS_OF)

    # The close-IV definition is nearest-expiry. Do not silently substitute a
    # later expiry just because it has a clean pair.
    assert observed["available"] is False
    assert observed["reason"] == "ATM_CE_PE_PAIR_UNAVAILABLE"


class _CloseClient:
    def __init__(self, rows):
        self.rows = rows
        self.quotes = _option_quotes(rows, iv=0.30)

    def quote(self, keys):
        out = {}
        for key in keys:
            if key == "NSE:ABC":
                out[key] = {"last_price": 100.0}
            elif key.startswith("NFO:"):
                symbol = key.split(":", 1)[1]
                if symbol in self.quotes:
                    out[key] = self.quotes[symbol]
        return out


def test_close_snapshot_persists_one_forward_observation_per_session(tmp_path):
    rows = _option_rows()
    client = _CloseClient(rows)
    trading_session = date(2026, 9, 25)
    with FoIvHistoryStore(tmp_path / "iv.sqlite3") as store:
        first = collect_close_iv_snapshot(
            symbols=["ABC"],
            instrument_rows=rows,
            client=client,
            as_of=trading_session,
            store=store,
        )
        assert first["persisted_symbols"] == 1
        assert first["store"]["rows"] == 1
        assert first["historical_backfill"] is False

        # Same-session rerun is an UPSERT, not duplicate evidence.
        second = collect_close_iv_snapshot(
            symbols=["ABC"],
            instrument_rows=rows,
            client=client,
            as_of=trading_session,
            store=store,
        )
        assert second["store"]["rows"] == 1
        assert second["store"]["symbols"] == 1


def test_close_snapshot_rejects_non_trading_session_without_writing(tmp_path):
    rows = _option_rows()
    client = _CloseClient(rows)
    path = tmp_path / "iv.sqlite3"
    with FoIvHistoryStore(path) as store:
        result = collect_close_iv_snapshot(
            symbols=["ABC"],
            instrument_rows=rows,
            client=client,
            as_of=AS_OF,  # Sunday 2026-09-27
            store=store,
        )
        assert result["available"] is False
        assert result["status"] == "BLOCKED"
        assert result["reason"] == "NON_TRADING_SESSION"
        assert result["persisted_symbols"] == 0
        assert store.status()["rows"] == 0


def test_market_ops_collects_iv_history_only_on_closing_slot():
    from operations.market_ops import MarketOperationsWorker

    source = inspect.getsource(MarketOperationsWorker._run_fno)
    assert 'operation_slot == "closing-1530"' in source
    assert "collect_close_iv_snapshot" in source
    assert '"CLOSING_1530_SNAPSHOT_ONLY"' in source
