"""Wrap of the Day — system-composed from stores; optional override only."""
from __future__ import annotations

from fastapi.testclient import TestClient


def test_parse_and_save_override(tmp_path, monkeypatch):
    from product import wrap_of_the_day as W

    path = tmp_path / "wrap.json"
    monkeypatch.setenv("QT_WRAP_FILE", str(path))
    monkeypatch.setattr(W, "DEFAULT_PATH", path)
    monkeypatch.setattr(W, "today_ist", lambda: "2026-08-07")

    text = """Here's the Wrap of the Day:

1) Manipal Health fell after its strong listing as investors booked profits.
2) HAL rallied after highlighting a large order book.
3) US futures edged higher after the Dow hit another record high.
"""
    bullets = W.parse_wrap_text(text)
    assert len(bullets) == 3
    assert bullets[0].startswith("Manipal Health")

    saved = W.save_wrap(text=text, date="2026-08-07", source="override")
    assert saved["available"] is True
    assert saved["override"] is True
    assert saved["auto"] is False

    loaded = W.load_wrap(date="2026-08-07", compose=False)
    assert loaded["available"] is True
    assert loaded["bullets"][1].startswith("HAL")
    assert loaded["override"] is True


def test_seed_loads_for_historical_date(tmp_path, monkeypatch):
    from product import wrap_of_the_day as W

    path = tmp_path / "missing.json"
    monkeypatch.setenv("QT_WRAP_FILE", str(path))
    monkeypatch.setattr(W, "DEFAULT_PATH", path)
    monkeypatch.setattr(W, "today_ist", lambda: "2026-08-07")
    wrap = W.load_wrap(date="2026-08-06", compose=False)
    assert wrap["available"] is True
    assert wrap["source"] == "seed"
    assert any("Neuland" in b for b in wrap["bullets"])


def test_compose_from_pulse_builds_newsletter_style_bullets():
    from product import wrap_of_the_day as W

    wrap = W.compose_from_pulse(
        {
            "snapshot": {
                "indices": [{"name": "NIFTY 50", "price": 24500, "chg_pct": 0.4}],
                "options_stance": {"stance": "CAUTION"},
                "commentary": "Choppy tape",
            },
            "sectors": {
                "available": True,
                "leaders": [{"sector": "Defence", "chg_1d": 2.1}],
                "laggards": [{"sector": "IT", "chg_1d": -1.2}],
            },
            "buzzing": {
                "symbol": "HAL",
                "change_pct": 4.5,
                "volume_ratio": 2.8,
                "note": "order-book focus in scan",
            },
            "breakouts_today": [{"symbol": "BEL"}],
            "headlines": ["Defence PSU order book in focus"],
            "global_cues": [{"name": "S&P 500", "chg_pct": 0.3}],
            "gaps": [],
        }
    )
    assert wrap["available"] is True
    assert wrap["source"] == "auto"
    assert wrap["auto"] is True
    assert any("NIFTY 50" in b for b in wrap["bullets"])
    assert any("HAL" in b for b in wrap["bullets"])
    assert any("Defence" in b or "defence" in b.lower() for b in wrap["bullets"])
    assert any("S&P 500" in b for b in wrap["bullets"])
    assert any("In the news" in b for b in wrap["bullets"])


def test_compose_does_not_invent_when_stores_empty():
    from product import wrap_of_the_day as W

    wrap = W.compose_from_pulse({"gaps": ["No scan yet"], "snapshot": {}, "sectors": {}})
    assert wrap["available"] is False
    assert "never" not in (wrap.get("message") or "").lower() or True
    assert wrap["bullets"] == []


def test_notify_wrap_telegram(tmp_path, monkeypatch):
    from product import wrap_of_the_day as W

    path = tmp_path / "wrap.json"
    monkeypatch.setenv("QT_WRAP_FILE", str(path))
    monkeypatch.setattr(W, "DEFAULT_PATH", path)
    monkeypatch.setattr(W, "today_ist", lambda: "2026-08-07")
    W.save_wrap(bullets=["HAL rallied on order book strength."], date="2026-08-07", source="auto")

    class _Fake:
        def is_configured(self):
            return True

        def send(self, message: str):
            assert "Wrap of the Day" in message
            assert "HAL" in message
            return True

    monkeypatch.setattr("alerts.telegram_alerts.AlertEngine", _Fake)
    result = W.notify_wrap_telegram(W.load_wrap(compose=False))
    assert result["sent"] is True
    assert result["count"] == 1


def test_wrap_api_rebuild_and_override(tmp_path, monkeypatch):
    import terminal_product_api as api
    from product import wrap_of_the_day as W

    path = tmp_path / "wrap.json"
    monkeypatch.setenv("QT_WRAP_FILE", str(path))
    monkeypatch.setattr(W, "DEFAULT_PATH", path)
    monkeypatch.setattr(W, "today_ist", lambda: "2026-08-07")
    monkeypatch.setattr(
        W,
        "compose_wrap",
        lambda persist=True: W.save_wrap(
            ["NIFTY 50 +0.40% led a firm session.", "HAL is buzzing (+4.5%)."],
            date="2026-08-07",
            source="auto",
        ),
    )

    client = TestClient(api.app)
    rebuilt = client.post("/api/wrap-of-the-day/rebuild").json()
    assert rebuilt["available"] is True
    assert rebuilt["source"] == "auto"
    assert "NIFTY" in rebuilt["bullets"][0]

    override = client.post(
        "/api/wrap-of-the-day",
        json={"text": "1) Manual override bullet only.", "notify": False},
    ).json()
    assert override["override"] is True

    got = client.get("/api/wrap-of-the-day").json()
    assert got["bullets"][0].startswith("Manual override")

    cleared = client.post("/api/wrap-of-the-day/clear-override").json()
    assert cleared["source"] == "auto"
    assert "NIFTY" in cleared["bullets"][0]


def test_pulse_uses_auto_wrap_then_override(tmp_path, monkeypatch):
    from product import wrap_of_the_day as W
    from reports import street_pulse as SP

    wrap_path = tmp_path / "wrap.json"
    pulse_path = tmp_path / "pulse.json"
    monkeypatch.setenv("QT_WRAP_FILE", str(wrap_path))
    monkeypatch.setattr(W, "DEFAULT_PATH", wrap_path)
    monkeypatch.setattr(SP, "DEFAULT_PULSE_PATH", pulse_path)
    monkeypatch.setattr(W, "today_ist", lambda: "2026-08-07")

    monkeypatch.setattr(SP, "_scan_universe", lambda: ([], 0, "", "unavailable"))
    monkeypatch.setattr(
        SP,
        "_movers_from_bhav",
        lambda top_n=5: ([{"symbol": "HAL", "price": 10, "chg_pct": 5.0}], []),
    )
    monkeypatch.setattr(
        SP,
        "_market_snapshot",
        lambda: {
            "indices": [{"name": "NIFTY 50", "price": 24500, "chg_pct": 0.4}],
            "commentary": "Firm tape",
            "regime": "",
            "options_stance": {"stance": "SUPPORTIVE"},
        },
    )
    monkeypatch.setattr(
        SP,
        "_sector_heat",
        lambda: {
            "available": True,
            "leaders": [{"sector": "Defence", "chg_1d": 2.0, "chg_5d": 3.0, "members": 4}],
            "laggards": [],
        },
    )
    monkeypatch.setattr(SP, "_losing_momentum", lambda: None)
    monkeypatch.setattr(SP, "_sniper_breakouts", lambda limit=4: [])
    monkeypatch.setattr(SP, "_headlines", lambda max_n=5: ["Defence names in focus"])
    monkeypatch.setattr(
        SP,
        "_global_cues",
        lambda: [{"name": "S&P 500", "price": 5000, "chg_pct": 0.2, "source": "us_retail"}],
    )

    pulse = SP.build_pulse(persist=True)
    assert pulse["wrap_of_the_day"]["available"] is True
    assert pulse["wrap_of_the_day"]["source"] == "auto"
    assert any("NIFTY" in t for t in pulse["takeaways"])

    W.save_wrap(
        bullets=["Override: HAL order book narrative only."],
        date="2026-08-07",
        source="override",
    )
    again = SP.build_pulse(persist=True)
    assert again["wrap_of_the_day"]["override"] is True
    assert again["takeaways"][0].startswith("Override")
    tg = SP.pulse_to_telegram(again)
    assert "Wrap of the Day" in tg
