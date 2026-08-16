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
    assert out[0]["entry"] == pytest.approx(129.0)
    assert out[0]["target"] > out[0]["entry"]
    assert out[0]["stop"] < out[0]["entry"]


def test_refresh_sets_entry_target_from_latest_close(monkeypatch):
    from product import live_technicals as lt

    idx = pd.date_range("2026-07-01", periods=30, freq="B")
    close = list(range(100, 130))
    frame = pd.DataFrame(
        {
            "open": close,
            "high": [c + 3 for c in close],
            "low": [c - 3 for c in close],
            "close": close,
            "volume": [1_000_000] * 30,
        },
        index=idx,
    )
    monkeypatch.setattr(lt, "ensure_live_store_overlay", lambda: 0)
    monkeypatch.setattr("data.bhavcopy_runtime.get_ohlcv", lambda sym: frame.copy())
    monkeypatch.setattr("core.market_clock.today_ist", lambda: idx[-1].date())

    out = lt.refresh_row_technicals({"symbol": "AAA", "price": 50, "entry": 0, "target": 0})
    assert out["price"] == pytest.approx(129.0)
    assert out["entry"] == pytest.approx(129.0)
    assert out["target"] > 129.0
    assert out["stop"] < 129.0
    assert out["levels_source"] == "current_ohlcv"


def test_apply_levels_percent_fallback_when_no_frame():
    from product.live_technicals import apply_current_trade_levels

    row = apply_current_trade_levels({"price": 200.0}, None)
    assert row["entry"] == 200.0
    assert row["target"] == 220.0
    assert row["stop"] == 190.0
    assert row["levels_source"] == "current_pct"


def test_apply_levels_preserves_existing_scanner_zone():
    from product.live_technicals import apply_current_trade_levels

    row = apply_current_trade_levels(
        {"price": 210.0, "entry": 200.0, "stop": 190.0, "target": 240.0},
        None,
    )
    assert row["entry"] == 200.0
    assert row["stop"] == 190.0
    assert row["target"] == 240.0
    assert "levels_source" not in row


def test_apply_levels_fills_missing_target_only():
    from product.live_technicals import apply_current_trade_levels

    row = apply_current_trade_levels({"price": 210.0, "entry": 200.0, "stop": 190.0}, None)
    assert row["entry"] == 200.0
    assert row["stop"] == 190.0
    assert row["target"] == 220.0
    assert row["levels_source"] == "current_pct"


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
    if payload["sniper_candidates"]:
        hit = [r for r in payload["sniper_candidates"] if r["symbol"] == "YATHARTH"]
        if hit:
            assert hit[0]["rsi"] == pytest.approx(49.0)
