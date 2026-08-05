"""Live quote heartbeat — honest about stream vs REST vs closed session."""
from __future__ import annotations


def test_heartbeat_prefers_stream_ticks(monkeypatch):
    from product import quote_heartbeat as QH

    monkeypatch.setattr(QH, "_session_open", lambda: True)
    monkeypatch.setattr(
        QH,
        "_from_ticker",
        lambda symbols: (
            {
                "RELIANCE": {
                    "symbol": "RELIANCE",
                    "price": 2500.0,
                    "chg_pct": 0.5,
                    "age_s": 1.2,
                    "source": "kite_ws",
                    "streaming": True,
                }
            },
            {"streaming": True, "watching": 1},
        ),
    )
    monkeypatch.setattr(QH, "_from_rest", lambda symbols: {})
    monkeypatch.setattr(
        QH,
        "_from_index_quotes",
        lambda symbols: {
            "NIFTY": {
                "symbol": "NIFTY",
                "price": 24500.0,
                "chg_pct": 0.1,
                "age_s": 0,
                "source": "index",
                "streaming": False,
            }
        },
    )

    payload = QH.build_quote_heartbeat(["RELIANCE", "NIFTY"])
    assert payload["available"] is True
    assert payload["streaming"] is True
    assert payload["quotes"]["RELIANCE"]["source"] == "kite_ws"
    assert payload["quotes"]["NIFTY"]["price"] == 24500.0
    assert payload["places_orders"] is False
    assert "EOD" in payload["honesty"]


def test_heartbeat_closed_session_no_stream_but_last_quote_ok(monkeypatch):
    from product import quote_heartbeat as QH

    monkeypatch.setattr(QH, "_session_open", lambda: False)
    monkeypatch.setattr(QH, "_from_ticker", lambda symbols: (_ for _ in ()).throw(AssertionError("no stream off hours")))
    monkeypatch.setattr(
        QH,
        "_from_rest",
        lambda symbols: {
            "RELIANCE": {
                "symbol": "RELIANCE",
                "price": 2490.0,
                "chg_pct": -0.2,
                "age_s": 0,
                "source": "kite",
                "streaming": False,
            }
        },
    )
    monkeypatch.setattr(
        QH,
        "_from_index_quotes",
        lambda symbols: {
            "NIFTY": {"symbol": "NIFTY", "price": 24500.0, "chg_pct": -0.1, "age_s": 0, "source": "index", "streaming": False}
        },
    )

    payload = QH.build_quote_heartbeat(["NIFTY", "RELIANCE"])
    assert payload["session_open"] is False
    assert payload["streaming"] is False
    assert payload["quotes"]["RELIANCE"]["price"] == 2490.0
    assert payload["quotes"]["NIFTY"]["price"] == 24500.0
    assert "Market closed" in payload["honesty"]


def test_clean_symbols_caps_and_dedupes():
    from product.quote_heartbeat import _clean_symbols

    rows = _clean_symbols([" reliance ", "RELIANCE", "tcs", ""], limit=2)
    assert rows == ["RELIANCE", "TCS"]
