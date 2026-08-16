"""Official desk tape must work without Yahoo / yfinance."""
from __future__ import annotations

import pandas as pd
import pytest


def _index_frame(last: float, prev: float, as_of="2026-08-14"):
    idx = pd.to_datetime([as_of]) - pd.Timedelta(days=1)
    idx = list(idx) + [pd.Timestamp(as_of)]
    return pd.DataFrame({"Close": [prev, last]}, index=idx)


def test_kite_quote_health_stale_when_token_rejected(monkeypatch):
    from data import live_quotes as lq

    lq._kite_health_cache = None
    monkeypatch.setattr(lq, "get_index_quotes", lambda names: {
        "NIFTY": {"price": 24366.0, "chg_pct": -0.12, "source": "official_nse"},
    })
    monkeypatch.setattr("data.kite_client._fresh_env", lambda name, default="": "tok" if name == "KITE_ACCESS_TOKEN" else default)
    health = lq.kite_quote_health()
    assert health["ok"] is False
    assert health["status"] == "stale"


def test_kite_quote_health_live_when_kite_answers(monkeypatch):
    from data import live_quotes as lq

    lq._kite_health_cache = None
    monkeypatch.setattr(lq, "get_index_quotes", lambda names: {
        "NIFTY": {"price": 24410.0, "chg_pct": 0.18, "source": "kite"},
    })
    monkeypatch.setattr("data.kite_client._fresh_env", lambda name, default="": "tok" if name == "KITE_ACCESS_TOKEN" else default)
    health = lq.kite_quote_health()
    assert health["ok"] is True
    assert health["nifty"] == 24410.0


def test_regime_engine_imports_without_yfinance():
    import core.regime_engine as re
    assert not hasattr(re, "yf")


def test_official_tape_from_index_store(monkeypatch):
    from product import local_tape as lt

    frames = {
        "Nifty 50": _index_frame(24366.0, 24395.0),
        "India VIX": _index_frame(11.31, 11.42),
        "Nifty Realty": _index_frame(900.0, 880.0),
        "Nifty Pharma": _index_frame(26000.0, 26200.0),
        "Nifty Bank": _index_frame(57500.0, 57400.0),
        "Nifty IT": _index_frame(100.0, 101.0),
        "Nifty Auto": _index_frame(100.0, 100.5),
        "Nifty FMCG": _index_frame(100.0, 99.0),
        "Nifty Metal": _index_frame(100.0, 102.0),
        "Nifty Energy": _index_frame(100.0, 100.2),
    }

    def _frame(name):
        return frames.get(name)

    monkeypatch.setattr(lt, "_index_frame", _frame)
    monkeypatch.setattr(lt, "session_breadth", lambda **k: {
        "n": 400, "advancers": 180, "decliners": 200, "adv_ratio": 0.9,
        "pct_above_50": 44.0, "verdict": "MIXED", "line": "180:200",
    })
    monkeypatch.setattr(lt, "_kite_index_overlay", lambda: {})
    tape = lt.read_official_tape()
    assert tape.usable is True
    assert tape.quote_source == "official_nse"
    assert tape.as_of == "2026-08-14"
    assert tape.nifty_change_1d == pytest.approx(-0.12, abs=0.02)
    assert tape.vix == pytest.approx(11.31)
    assert "REALTY" in tape.leaders


def test_current_market_view_does_not_need_yfinance(monkeypatch):
    from product import market_view as mv
    from product.local_tape import OfficialTape

    tape = OfficialTape(
        as_of="2026-08-14",
        nifty_close=24366.0,
        nifty_change_1d=-0.12,
        nifty_change_5d=-0.8,
        vix=11.31,
        leaders=("REALTY", "PHARMA"),
        laggards=("IT",),
        breadth={"verdict": "MIXED", "pct_above_50": 44, "n": 400, "line": "ok"},
    )
    monkeypatch.setattr("product.local_tape.read_official_tape", lambda: tape)
    view = mv.current_market_view()
    assert view.health in {"Mixed", "Weak", "Healthy"}
    assert view.nifty_change_1d == pytest.approx(-0.12)
    assert view.vix == pytest.approx(11.31)
    assert "2026-08-14" in view.summary
    assert view.technical_details.get("source", "").startswith("official_nse")


def test_kite_index_overlay_wins_nifty_and_vix(monkeypatch):
    from product import local_tape as lt

    frames = {
        "Nifty 50": _index_frame(24366.0, 24395.0),
        "India VIX": _index_frame(11.31, 11.42),
        "Nifty Bank": _index_frame(57500.0, 57400.0),
        "Nifty IT": _index_frame(100.0, 101.0),
        "Nifty Auto": _index_frame(100.0, 100.5),
        "Nifty Pharma": _index_frame(100.0, 101.0),
        "Nifty FMCG": _index_frame(100.0, 99.0),
        "Nifty Metal": _index_frame(100.0, 102.0),
        "Nifty Energy": _index_frame(100.0, 100.2),
        "Nifty Realty": _index_frame(100.0, 99.5),
    }
    monkeypatch.setattr(lt, "_index_frame", lambda name: frames.get(name))
    monkeypatch.setattr(lt, "session_breadth", lambda **k: {
        "n": 400, "advancers": 180, "decliners": 200, "adv_ratio": 0.9,
        "pct_above_50": 44.0, "verdict": "MIXED", "line": "180:200",
    })
    monkeypatch.setattr(lt, "_kite_index_overlay", lambda: {
        "NIFTY": {"price": 24410.0, "chg_pct": 0.18, "source": "kite"},
        "VIX": {"price": 11.05, "chg_pct": -2.3, "source": "kite"},
        "BANK": {"price": 57600.0, "chg_pct": 0.35, "source": "kite"},
    })
    tape = lt.read_official_tape()
    assert tape.quote_source == "kite"
    assert tape.source.startswith("kite")
    assert tape.nifty_close == pytest.approx(24410.0)
    assert tape.nifty_change_1d == pytest.approx(0.18)
    assert tape.vix == pytest.approx(11.05)
    assert tape.sector_changes["BANK"] == pytest.approx(0.35)


def test_market_payload_available_from_local_tape(monkeypatch):
    from product.local_tape import OfficialTape
    from product import market_view as mv

    tape = OfficialTape(
        as_of="2026-08-14",
        nifty_close=24366.0,
        nifty_change_1d=-0.12,
        vix=11.31,
        leaders=("BANK",),
        breadth={"verdict": "MIXED", "pct_above_50": 44, "n": 500},
    )
    monkeypatch.setattr("product.local_tape.read_official_tape", lambda: tape)
    view = mv._view_from_official_tape()
    assert view is not None
    assert view.health != "Unavailable"
