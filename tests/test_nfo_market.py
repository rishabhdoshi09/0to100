from datetime import date

from data.nfo_market import (
    NfoMarketDataClient,
    candidate_option_instruments,
    equity_fo_universe,
    futures_oi_features,
    nearest_future,
    option_instruments,
    previous_future_close_oi,
    quote_to_option_contract,
    read_market_quotes,
    read_nfo_instruments,
    read_nfo_quotes,
)
from options.directional_selector import black_scholes, implied_volatility


def _instruments():
    return [
        {
            "instrument_token": 1, "tradingsymbol": "RELIANCE26OCTFUT",
            "name": "RELIANCE", "expiry": "2026-10-29", "strike": 0,
            "tick_size": 0.05, "lot_size": 250, "instrument_type": "FUT",
            "segment": "NFO-FUT", "exchange": "NFO",
        },
        {
            "instrument_token": 2, "tradingsymbol": "RELIANCE26OCT3000CE",
            "name": "RELIANCE", "expiry": "2026-10-29", "strike": 3000,
            "tick_size": 0.05, "lot_size": 250, "instrument_type": "CE",
            "segment": "NFO-OPT", "exchange": "NFO",
        },
        {
            "instrument_token": 3, "tradingsymbol": "RELIANCE26OCT3000PE",
            "name": "RELIANCE", "expiry": "2026-10-29", "strike": 3000,
            "tick_size": 0.05, "lot_size": 250, "instrument_type": "PE",
            "segment": "NFO-OPT", "exchange": "NFO",
        },
        {
            "instrument_token": 4, "tradingsymbol": "NIFTY26OCTFUT",
            "name": "NIFTY", "expiry": "2026-10-29", "strike": 0,
            "tick_size": 0.05, "lot_size": 50, "instrument_type": "FUT",
            "segment": "NFO-FUT", "exchange": "NFO",
        },
    ]


def test_nfo_universe_requires_stock_future_and_option():
    assert equity_fo_universe(_instruments()) == ["RELIANCE"]


def test_nearest_future_and_option_expiry_filtering():
    as_of = date(2026, 9, 26)
    fut = nearest_future(_instruments(), "RELIANCE", as_of=as_of)
    assert fut and fut["tradingsymbol"] == "RELIANCE26OCTFUT"

    opts = option_instruments(_instruments(), "RELIANCE", as_of=as_of)
    assert {row["instrument_type"] for row in opts} == {"CE", "PE"}


def test_quote_normalization_uses_depth_oi_and_implied_iv():
    as_of = date(2026, 9, 26)
    instrument = _instruments()[1]
    theoretical = black_scholes(
        spot=3050.0, strike=3000.0, dte=33, iv=0.24, option_type="CE",
    )["price"]
    quote = {
        "last_price": theoretical,
        "volume": 5500,
        "oi": 32000,
        "depth": {
            "buy": [{"price": theoretical - 0.25}],
            "sell": [{"price": theoretical + 0.25}],
        },
    }
    row = quote_to_option_contract(instrument, quote, spot=3050.0, as_of=as_of)
    assert row["bid"] > 0 and row["ask"] > row["bid"]
    assert row["oi"] == 32000
    assert row["volume"] == 5500
    assert 20.0 < row["iv"] < 28.0
    assert row["iv_source"] == "IMPLIED_FROM_MARKET_QUOTE"


def test_implied_volatility_recovers_known_volatility():
    price = black_scholes(
        spot=1000.0, strike=1020.0, dte=12, iv=0.31, option_type="CE",
    )["price"]
    recovered = implied_volatility(
        market_price=price,
        spot=1000.0,
        strike=1020.0,
        dte=12,
        option_type="CE",
    )
    assert abs(recovered - 0.31) < 0.01


def test_futures_oi_features_preserve_unknown_baseline():
    known = futures_oi_features(
        current_price=101, previous_price=100, current_oi=1100, previous_oi=1000,
    )
    assert known["futures_price_change_pct"] == 1.0
    assert known["futures_oi_change_pct"] == 10.0

    unknown = futures_oi_features(
        current_price=101, previous_price=0, current_oi=1100, previous_oi=0,
    )
    assert unknown["futures_price_change_pct"] is None
    assert unknown["futures_oi_change_pct"] is None


class _Raw:
    def instruments(self, exchange):
        assert exchange == "NFO"
        return _instruments()

    def quote(self, keys):
        return {
            key: {
                "last_price": 10.0, "volume": 100, "oi": 1000,
                "depth": {"buy": [{"price": 9.9}], "sell": [{"price": 10.1}]},
            }
            for key in keys
        }

    def historical_data(self, token, frm, to, interval, oi=False):
        assert oi is True
        return [
            {"date": "2026-09-24", "close": 100.0, "oi": 1000},
            {"date": "2026-09-25", "close": 101.0, "oi": 1100},
        ]


class _Client:
    raw = _Raw()


def test_read_adapter_uses_only_read_only_market_data_surface():
    instruments = read_nfo_instruments(_Client())
    assert len(instruments) == len(_instruments())
    quotes = read_nfo_quotes(["RELIANCE26OCT3000CE"], _Client())
    assert "RELIANCE26OCT3000CE" in quotes


def test_candidate_options_are_bounded_around_spot():
    rows = _instruments()
    rows.extend([
        {
            "instrument_token": 5, "tradingsymbol": "RELIANCE26OCT3500CE",
            "name": "RELIANCE", "expiry": "2026-10-29", "strike": 3500,
            "tick_size": 0.05, "lot_size": 250, "instrument_type": "CE",
            "segment": "NFO-OPT", "exchange": "NFO",
        },
    ])
    selected = candidate_option_instruments(
        rows, "RELIANCE", spot=3000.0, as_of=date(2026, 9, 26), max_moneyness_pct=8.0,
    )
    assert {row["tradingsymbol"] for row in selected} == {
        "RELIANCE26OCT3000CE", "RELIANCE26OCT3000PE",
    }


def test_read_only_nfo_facade_exposes_data_but_no_order_surface():
    facade = NfoMarketDataClient(_Raw())
    assert facade.is_data_only is True
    assert facade.instruments("NFO")
    assert facade.quote(["NFO:RELIANCE26OCT3000CE"])
    assert facade.historical_with_oi(1, "2026-09-20", "2026-09-25")
    for forbidden in ("place_order", "modify_order", "cancel_order", "place_gtt", "delete_gtt"):
        assert not hasattr(facade, forbidden)


def test_previous_future_snapshot_is_strictly_historical_and_has_oi():
    facade = NfoMarketDataClient(_Raw())
    row = previous_future_close_oi(1, as_of=date(2026, 9, 26), client=facade)
    assert row["date"] == "2026-09-25"
    assert row["close"] == 101.0
    assert row["oi"] == 1100.0


def test_generic_quote_reader_preserves_exchange_qualified_keys():
    facade = NfoMarketDataClient(_Raw())
    rows = read_market_quotes(["NSE:RELIANCE", "NFO:RELIANCE26OCTFUT"], client=facade)
    assert set(rows) == {"NSE:RELIANCE", "NFO:RELIANCE26OCTFUT"}
