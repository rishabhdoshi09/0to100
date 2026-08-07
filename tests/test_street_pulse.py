"""Daily Street Pulse — research digest honesty + durable assembly."""
from __future__ import annotations

from reports.street_pulse import (
    SCHEMA_VERSION,
    build_pulse,
    load_pulse,
    pulse_to_telegram,
    save_pulse,
    _stock_card,
)


def test_stock_card_maps_scan_store_prebreakout():
    card = _stock_card(
        {
            "symbol": "HSCL",
            "company": "Himadri",
            "status": "Watch for breakout",
            "signals": ["PRE_BREAKOUT"],
            "price": 100,
            "entry": 105,
            "score": 72,
            "volume_ratio": 1.2,
            "reasons": ["Coil under pivot"],
        }
    )
    assert card["pre_breakout"] is True
    assert card["pivot_distance_pct"] is not None
    assert card["symbol"] == "HSCL"


def test_build_pulse_persists_and_discloses_honesty(tmp_path, monkeypatch):
    monkeypatch.setattr("reports.street_pulse.DEFAULT_PULSE_PATH", tmp_path / "pulse.json")
    monkeypatch.setattr(
        "product.wrap_of_the_day.load_manual_override",
        lambda date=None, path=None: None,
    )
    monkeypatch.setattr(
        "product.wrap_of_the_day.save_wrap",
        lambda *args, **kwargs: {"available": False, "bullets": []},
    )
    monkeypatch.setattr(
        "reports.street_pulse._scan_universe",
        lambda: (
            [
                _stock_card(
                    {
                        "symbol": "ATHERENERG",
                        "change_pct": 8.0,
                        "volume_ratio": 3.5,
                        "price": 500,
                        "score": 80,
                        "status": "Ready to trade",
                        "signals": ["Resistance break on volume"],
                        "reasons": ["Volume breakout"],
                    }
                ),
                _stock_card(
                    {
                        "symbol": "KERNEX",
                        "status": "Watch for breakout",
                        "signals": ["PRE_BREAKOUT"],
                        "price": 90,
                        "entry": 95,
                        "score": 70,
                        "volume_ratio": 1.1,
                        "reasons": ["Pullback hold"],
                    }
                ),
            ],
            2,
            "2026-08-04T10:00:00+00:00",
            "scan_store",
        ),
    )
    monkeypatch.setattr(
        "reports.street_pulse._movers_from_bhav",
        lambda top_n=5: (
            [{"symbol": "AAA", "price": 10, "chg_pct": 5.0}],
            [{"symbol": "BBB", "price": 10, "chg_pct": -4.0}],
        ),
    )
    monkeypatch.setattr(
        "reports.street_pulse._market_snapshot",
        lambda: {
            "indices": [{"name": "NIFTY 50", "price": 24500, "chg_pct": 0.2}],
            "commentary": "Choppy tape",
            "options_stance": {
                "stance": "CAUTION",
                "headline": "Mixed options context",
                "honesty": "Not a buy/sell signal",
            },
        },
    )
    monkeypatch.setattr(
        "reports.street_pulse._sector_heat",
        lambda limit=6: {
            "available": True,
            "leaders": [{"sector": "Pharma", "chg_1d": 1.2, "chg_5d": 3.0, "members": 8}],
            "laggards": [{"sector": "Banks", "chg_1d": -0.8, "chg_5d": -1.0, "members": 10}],
            "message": "",
        },
    )
    monkeypatch.setattr("reports.street_pulse._losing_momentum", lambda: None)
    monkeypatch.setattr("reports.street_pulse._sniper_breakouts", lambda limit=4: [])
    monkeypatch.setattr("reports.street_pulse._headlines", lambda max_n=5: ["RBI holds rates"])
    monkeypatch.setattr(
        "reports.street_pulse._global_cues",
        lambda: [{"name": "S&P 500", "price": 5000, "chg_pct": 0.4, "source": "us_retail"}],
    )

    pulse = build_pulse(persist=True)
    assert pulse["schema_version"] == SCHEMA_VERSION
    assert pulse["available"] is True
    assert pulse["signal_desk"] is False
    assert pulse["places_orders"] is False
    assert "buy ticket" in pulse["honesty"].lower() or "not a buy" in pulse["honesty"].lower()
    assert pulse["buzzing"]["symbol"] == "ATHERENERG"
    assert pulse["strength"]["symbol"] == "KERNEX"
    assert pulse["sectors"]["leaders"][0]["sector"] == "Pharma"
    assert any("CAUTION" in t for t in pulse["takeaways"])
    assert (tmp_path / "pulse.json").exists()
    loaded = load_pulse(tmp_path / "pulse.json")
    assert loaded["buzzing"]["symbol"] == "ATHERENERG"

    tg = pulse_to_telegram(pulse)
    assert "Daily Pulse" in tg
    assert "not a buy" in tg.lower()
    assert "ATHERENERG" in tg


def test_send_pulse_telegram_reports_unconfigured(monkeypatch):
    from reports import street_pulse as SP

    class _Eng:
        def is_configured(self):
            return False

        def send(self, msg):
            raise AssertionError("should not send")

    monkeypatch.setattr("alerts.telegram_alerts.AlertEngine", lambda: _Eng())
    monkeypatch.setattr(SP, "build_pulse", lambda persist=True: {"date": "05 August 2026"})
    out = SP.send_pulse_telegram(force_build=True)
    assert out["sent"] is False
    assert out["configured"] is False


def test_save_load_roundtrip(tmp_path):
    path = tmp_path / "p.json"
    save_pulse({"available": True, "date": "04 August 2026", "takeaways": ["x"]}, path)
    loaded = load_pulse(path)
    assert loaded["date"] == "04 August 2026"
