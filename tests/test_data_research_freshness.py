"""Data/research freshness: session-aware history + IST live overlay."""
from __future__ import annotations

from datetime import date, datetime

import pandas as pd
import pytest


def test_history_lane_stale_when_completed_session_missing():
    from product.product_readiness import build_product_readiness

    # Wednesday 10:00 IST (UTC 04:30) — prior session Tue must be present.
    now = datetime(2026, 8, 12, 4, 30, tzinfo=__import__("datetime").timezone.utc)
    payload = build_product_readiness(
        market={},
        scan={},
        long_term={},
        news={},
        fno={},
        data={
            "bhavcopy": {
                "ready": True,
                "sessions": 500,
                "symbols": 1800,
                # Stuck on Monday while Tuesday EOD is already required.
                "latest_date": "2026-08-10",
            }
        },
        operations={},
        now=now,
    )
    history = next(item for item in payload["lanes"] if item["key"] == "history")
    assert history["status"] == "STALE"
    assert history["sessions_behind"] and history["sessions_behind"] >= 1
    assert "session" in history["details"].lower()


def test_history_lane_fresh_when_prior_session_present():
    from product.product_readiness import build_product_readiness

    now = datetime(2026, 8, 12, 4, 30, tzinfo=__import__("datetime").timezone.utc)
    payload = build_product_readiness(
        market={},
        scan={},
        long_term={},
        news={},
        fno={},
        data={
            "bhavcopy": {
                "ready": True,
                "sessions": 500,
                "symbols": 1800,
                "latest_date": "2026-08-11",
            }
        },
        operations={},
        now=now,
    )
    history = next(item for item in payload["lanes"] if item["key"] == "history")
    assert history["status"] == "FRESH"
    assert (history.get("sessions_behind") or 0) == 0


def test_is_trading_now_uses_ist_not_machine_local(monkeypatch):
    from data import nse_live as nl

    class _Fixed:
        def __init__(self, hour, minute, weekday=2):
            # Build a fixed IST-like datetime via market_clock stub.
            self.hour = hour
            self.minute = minute
            self.weekday_n = weekday

        def weekday(self):
            return self.weekday_n

    monkeypatch.setattr(
        "core.market_clock.now_ist",
        lambda: _Fixed(10, 0),
    )
    assert nl._is_trading_now() is True

    monkeypatch.setattr(
        "core.market_clock.now_ist",
        lambda: _Fixed(3, 0),  # 03:00 IST — pre-open
    )
    assert nl._is_trading_now() is False


def test_overlay_live_on_frame_appends_today(monkeypatch):
    from data import nse_live as nl

    idx = pd.to_datetime(["2026-08-11", "2026-08-12"])
    frame = pd.DataFrame(
        {
            "open": [100.0, 110.0],
            "high": [105.0, 115.0],
            "low": [99.0, 108.0],
            "close": [104.0, 112.0],
            "volume": [1e5, 1.2e5],
        },
        index=idx,
    )
    monkeypatch.setattr(nl, "_is_trading_now", lambda: True)
    monkeypatch.setattr(nl, "_today_session", lambda: date(2026, 8, 13))
    monkeypatch.setattr(
        nl,
        "live_bar_for",
        lambda symbol: {
            "open": 111.0, "high": 113.0, "low": 109.0, "close": 110.5,
            "volume": 9e4, "source": "nse",
        },
    )
    out, meta = nl.overlay_live_on_frame(frame, "YATHARTH")
    assert meta["live"] is True
    assert meta["price_tag"] == "LIVE"
    assert str(out.index[-1].date()) == "2026-08-13"
    assert float(out.iloc[-1]["close"]) == pytest.approx(110.5)


def test_overlay_without_live_keeps_eod(monkeypatch):
    from data import nse_live as nl

    idx = pd.to_datetime(["2026-08-12"])
    frame = pd.DataFrame(
        {"open": [100.0], "high": [105.0], "low": [99.0], "close": [104.0], "volume": [1e5]},
        index=idx,
    )
    monkeypatch.setattr(nl, "live_bar_for", lambda symbol: None)
    out, meta = nl.overlay_live_on_frame(frame, "YATHARTH")
    assert meta["live"] is False
    assert meta["price_tag"] == "EOD"
    assert len(out) == 1
    assert meta["eod_as_of"] == "2026-08-12"


def test_stock_workspace_marks_stale_history_and_live_tag(monkeypatch):
    from product.stock_workspace import build_stock_workspace

    idx = pd.to_datetime(["2026-08-10", "2026-08-11"])
    frame = pd.DataFrame(
        {
            "open": [800.0, 860.0],
            "high": [820.0, 920.0],
            "low": [790.0, 850.0],
            "close": [810.0, 865.0],
            "volume": [1e5, 2e5],
        },
        index=idx,
    )

    monkeypatch.setattr(
        "data.nse_live.overlay_live_on_frame",
        lambda fr, sym: (
            fr,
            {
                "live": True,
                "price_tag": "LIVE",
                "eod_as_of": "2026-08-11",
                "source": "nse",
                "live_as_of": "2026-08-13",
            },
        ),
    )
    # Force session calendar to demand 2026-08-12 while EOD is 2026-08-11.
    from research.intelligence.data import nse_calendar as CAL

    monkeypatch.setattr(
        CAL,
        "snapshot_freshness",
        lambda latest, now=None, holidays=None, cutoff=None, allowance_sessions=1: {
            "fresh": False,
            "required": "2026-08-12",
            "latest": latest,
            "sessions_behind": 1,
            "reason": "missing",
        },
    )
    now = datetime(2026, 8, 13, 4, 30, tzinfo=__import__("datetime").timezone.utc)
    ws = build_stock_workspace(
        "YATHARTH",
        scan_payload={"scanned_at": "2026-08-13T03:00:00+00:00", "records": []},
        long_term_payload={"scanned_at": "2026-08-10T03:00:00+00:00", "records": []},
        raw_fundamentals={"available": False},
        frame=frame,
        news=[],
        fno_payload={},
        now=now,
    )
    assert ws["technical"]["price_tag"] == "LIVE"
    hist = next(s for s in ws["sources"] if s["name"] == "Official price history")
    assert hist["status"] == "STALE"
    assert hist.get("sessions_behind") == 1
