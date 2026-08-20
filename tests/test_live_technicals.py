"""Live technical refresh — RSI/price must not freeze at scan time."""
from __future__ import annotations

import pandas as pd
import pytest


def _frame_with_drop():
    """Synthetic history: rally then drop — live close pulls RSI down vs scan."""
    idx = pd.date_range("2026-07-01", periods=40, freq="B")
    close = list(range(100, 140))  # grind up
    close[-1] = 138
    return pd.DataFrame(
        {
            "open": close,
            "high": [c + 2 for c in close],
            "low": [c - 2 for c in close],
            "close": close,
            "volume": [1_000_000] * len(close),
        },
        index=idx,
    )


def test_refresh_recomputes_rsi_below_stale_scan_value(monkeypatch):
    from product import live_technicals as lt

    frame = _frame_with_drop()
    live_idx = frame.index[-1] + pd.Timedelta(days=1)
    while live_idx.weekday() >= 5:
        live_idx += pd.Timedelta(days=1)
    live_frame = pd.concat([
        frame,
        pd.DataFrame(
            {"open": [130], "high": [131], "low": [118], "close": [120], "volume": [1.5e6]},
            index=[live_idx],
        ),
    ])

    monkeypatch.setattr(lt, "ensure_live_store_overlay", lambda: 0)
    monkeypatch.setattr("data.bhavcopy_runtime.get_ohlcv", lambda sym: live_frame.copy())
    monkeypatch.setattr(
        "data.nse_live.overlay_live_on_frame",
        lambda fr, sym: (fr, {"live": True, "price_tag": "LIVE", "source": "nse", "eod_as_of": str(frame.index[-1].date())}),
    )
    monkeypatch.setattr("core.market_clock.today_ist", lambda: live_idx.date())

    row = {
        "symbol": "YATHARTH",
        "rsi": 66.0,          # stale scan
        "price": 868.0,
        "volume_ratio": 4.9,
        "avg_vol20": 1_000_000,
    }
    out = lt.refresh_row_technicals(row)
    assert out["rsi"] < 66.0
    assert out["price"] == pytest.approx(120.0)
    assert out["tech_source"] in {"live", "eod"}
    assert out["price_tag"] in {"LIVE", "EOD"}
    assert out["pct_below_20d_high"] > 5.0


def test_refresh_fail_open_keeps_scan_fields(monkeypatch):
    from product import live_technicals as lt

    monkeypatch.setattr(lt, "ensure_live_store_overlay", lambda: 0)
    monkeypatch.setattr("data.bhavcopy_runtime.get_ohlcv", lambda sym: None)
    row = {"symbol": "MISSING", "rsi": 55.0, "price": 100.0, "volume_ratio": 1.2}
    out = lt.refresh_row_technicals(row)
    assert out["rsi"] == 55.0
    assert out["price"] == 100.0
    assert "tech_source" not in out


def test_bulk_refresh_skips_per_symbol_network(monkeypatch):
    """Radar must not scrape Google per breakout — that emptied the sniper lane."""
    from product import live_technicals as lt
    import pandas as pd

    idx = pd.date_range("2026-07-01", periods=30, freq="B")
    frame = pd.DataFrame(
        {
            "open": list(range(100, 130)),
            "high": list(range(102, 132)),
            "low": list(range(98, 128)),
            "close": list(range(100, 130)),
            "volume": [1_000_000] * 30,
        },
        index=idx,
    )
    calls = {"overlay": 0}

    def _overlay(fr, sym):
        calls["overlay"] += 1
        return fr, {"live": True, "price_tag": "LIVE", "source": "nse"}

    monkeypatch.setattr(lt, "ensure_live_store_overlay", lambda: 0)
    monkeypatch.setattr("data.bhavcopy_runtime.get_ohlcv", lambda sym: frame.copy())
    monkeypatch.setattr("data.nse_live.overlay_live_on_frame", _overlay)
    monkeypatch.setattr("core.market_clock.today_ist", lambda: idx[-1].date())

    rows = [{"symbol": "AAA", "rsi": 60, "volume_ratio": 1.5, "avg_vol20": 1e6}]
    out = lt.refresh_rows_technicals(rows, bulk_overlay=True)
    assert calls["overlay"] == 0
    assert out[0]["rsi"] is not None


def test_refresh_limit_keeps_unscanned_tail(monkeypatch):
    from product import live_technicals as lt

    monkeypatch.setattr(lt, "ensure_live_store_overlay", lambda: 0)
    monkeypatch.setattr("data.bhavcopy_runtime.get_ohlcv", lambda sym: None)
    monkeypatch.setattr("data.nse_live._is_trading_now", lambda: False)
    rows = [{"symbol": f"S{i}", "volume_ratio": 2.0, "rsi": 50} for i in range(5)]
    out = lt.refresh_rows_technicals(rows, limit=2, bulk_overlay=True)
    assert len(out) == 5
    assert [r["symbol"] for r in out] == [f"S{i}" for i in range(5)]


def test_partial_session_volume_does_not_demote_scan_ratio(monkeypatch):
    """A 10am print at 0.3× avg-day must not wipe a scan-time 2.0× sniper lane."""
    from product import live_technicals as lt
    import pandas as pd

    idx = pd.date_range("2026-07-01", periods=30, freq="B")
    live_idx = idx[-1]
    frame = pd.DataFrame(
        {
            "open": list(range(100, 130)),
            "high": list(range(102, 132)),
            "low": list(range(98, 128)),
            "close": list(range(100, 130)),
            "volume": [1_000_000] * 29 + [300_000],
        },
        index=idx,
    )
    monkeypatch.setattr(lt, "ensure_live_store_overlay", lambda: 0)
    monkeypatch.setattr("data.bhavcopy_runtime.get_ohlcv", lambda sym: frame.copy())
    monkeypatch.setattr("core.market_clock.today_ist", lambda: live_idx.date())
    monkeypatch.setattr("data.nse_live._is_trading_now", lambda: True)
    monkeypatch.setattr(lt, "_session_frac", lambda: 0.20)
    row = {"symbol": "SNIPE", "rsi": 55.0, "price": 129.0, "volume_ratio": 2.0, "avg_vol20": 1_000_000}
    out = lt.refresh_row_technicals(row)
    assert out["volume_ratio"] >= 2.0


def test_today_spike_is_not_the_20d_high(monkeypatch):
    from product import live_technicals as lt
    import pandas as pd

    idx = pd.date_range("2026-07-01", periods=25, freq="B")
    close = [100.0] * 24 + [100.0]
    high = [102.0] * 24 + [130.0]  # today's wick
    frame = pd.DataFrame(
        {"open": close, "high": high, "low": [98] * 25, "close": close, "volume": [1e6] * 25},
        index=idx,
    )
    monkeypatch.setattr("core.market_clock.today_ist", lambda: idx[-1].date())
    out = lt._structure_from_frame(frame, 100.0)
    assert out["high_20d"] == pytest.approx(102.0)
    assert out["pct_below_20d_high"] < 5.0


def test_kite_last_print_overrides_store_close(monkeypatch):
    from product import live_technicals as lt

    monkeypatch.setattr("data.kite_client._fresh_env", lambda name, default="": "token" if name == "KITE_ACCESS_TOKEN" else default)
    monkeypatch.setattr(
        "data.live_quotes._kite_quotes",
        lambda symbols: {"RPEL": {"price": 1440.5, "chg_pct": 0.4, "source": "kite"}},
    )
    monkeypatch.setattr("data.nse_live._is_trading_now", lambda: True)
    rows = [{
        "symbol": "RPEL",
        "price": 1433.9,
        "high_20d": 1473.0,
        "high_52w": 1473.0,
        "pct_below_20d_high": 2.65,
        "pct_below_52w_high": 2.65,
        "price_tag": "EOD",
        "tech_source": "eod",
    }]
    out = lt._apply_kite_last(rows)
    assert out[0]["price"] == pytest.approx(1440.5)
    assert out[0]["quote_source"] == "kite"
    assert out[0]["price_tag"] == "LIVE"
    assert out[0]["pct_below_20d_high"] < 2.65


def test_kite_last_print_skipped_when_market_closed(monkeypatch):
    from product import live_technicals as lt

    monkeypatch.setattr("data.nse_live._is_trading_now", lambda: False)
    monkeypatch.setattr("data.kite_client._fresh_env", lambda name, default="": "token" if name == "KITE_ACCESS_TOKEN" else default)
    monkeypatch.setattr(
        "data.live_quotes._kite_quotes",
        lambda symbols: {"RPEL": {"price": 1440.5, "chg_pct": 0.4, "source": "kite"}},
    )
    rows = [{"symbol": "RPEL", "price": 1433.9, "price_tag": "EOD"}]
    out = lt._apply_kite_last(rows)
    assert out[0]["price"] == pytest.approx(1433.9)
    assert out[0].get("quote_source") != "kite"


def test_nse_fallback_quote_is_not_tagged_kite(monkeypatch):
    from product import live_technicals as lt

    monkeypatch.setattr("data.kite_client._fresh_env", lambda name, default="": "token" if name == "KITE_ACCESS_TOKEN" else default)
    monkeypatch.setattr(
        "data.live_quotes._kite_quotes",
        lambda symbols: {"RPEL": {"price": 1440.5, "chg_pct": 0.4, "source": "nse"}},
    )
    rows = [{"symbol": "RPEL", "price": 1433.9, "price_tag": "EOD", "tech_source": "eod"}]
    out = lt._apply_kite_last(rows)
    assert out[0]["price"] == pytest.approx(1433.9)
    assert out[0].get("quote_source") != "kite"
    assert out[0]["price_tag"] == "EOD"


def test_radar_home_uses_refreshed_rsi(monkeypatch):
    from product.radar_workspace import build_radar_home

    def _refresh(rows, bulk_overlay=True, limit=None):
        out = []
        for r in rows:
            row = dict(r)
            if row.get("symbol") == "YATHARTH":
                row["rsi"] = 49.0
                row["price"] = 848.0
                row["tech_source"] = "live"
                row["price_tag"] = "LIVE"
                row["pct_below_20d_high"] = 8.2
            elif row.get("symbol") == "SOLID":
                row["pct_below_20d_high"] = 1.1
                row["pct_below_52w_high"] = 2.0
            out.append(row)
        return out

    monkeypatch.setattr("product.live_technicals.refresh_rows_technicals", _refresh)

    scan = {
        "scanned_at": "2026-08-13T03:00:00+00:00",
        "universe_size": 2,
        "records": [
            {
                "symbol": "YATHARTH",
                "score": 70,
                "verdict": "BUY",
                "status": "Ready to trade",
                "signals": ["BREAKOUT_52W"],
                "chase_risk": False,
                "volume_ratio": 4.9,
                "rsi": 66.0,
                "breakout_grade": "B",
                "breakout_conviction": 60,
                "avg_vol20": 1e6,
            },
            {
                "symbol": "SOLID",
                "score": 75,
                "verdict": "BUY",
                "status": "Ready to trade",
                "signals": ["BREAKOUT_52W"],
                "chase_risk": False,
                "volume_ratio": 2.0,
                "rsi": 55.0,
                "breakout_grade": "A",
                "breakout_conviction": 80,
                "avg_vol20": 1e6,
            },
        ],
    }
    payload = build_radar_home(
        scan_payload=scan,
        long_term_payload={"records": []},
        market={"health": "Healthy"},
    )
    yath = next(r for r in payload["lanes"]["breakouts"] if r["symbol"] == "YATHARTH")
    assert yath["rsi"] == pytest.approx(49.0)
    assert yath.get("tech_source") == "live"
    assert yath.get("breakout_state") == "faded_breakout"
    assert not any(r["symbol"] == "YATHARTH" for r in payload.get("sniper_candidates") or [])
    assert payload["best_breakout"] is not None
    assert payload["best_breakout"]["symbol"] == "SOLID"
