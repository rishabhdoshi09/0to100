"""Wrap of the Day — user-authored only; never invents bullets."""
from __future__ import annotations

from fastapi.testclient import TestClient


def test_parse_and_save_wrap(tmp_path, monkeypatch):
    from product import wrap_of_the_day as W

    path = tmp_path / "wrap.json"
    monkeypatch.setenv("QT_WRAP_FILE", str(path))
    monkeypatch.setattr(W, "DEFAULT_PATH", path)

    text = """Here's the Wrap of the Day:

1) Manipal Health fell after its strong listing as investors booked profits.
2) HAL rallied after highlighting a large order book.
3) US futures edged higher after the Dow hit another record high.
"""
    bullets = W.parse_wrap_text(text)
    assert len(bullets) == 3
    assert bullets[0].startswith("Manipal Health")
    assert "Here's the Wrap" not in bullets[0]

    saved = W.save_wrap(text=text, date="2026-08-06", source="paste")
    assert saved["available"] is True
    assert saved["date"] == "2026-08-06"
    assert len(saved["bullets"]) == 3

    loaded = W.load_wrap(date="2026-08-06")
    assert loaded["available"] is True
    assert loaded["bullets"][1].startswith("HAL")


def test_seed_loads_when_no_runtime_file(tmp_path, monkeypatch):
    from product import wrap_of_the_day as W

    path = tmp_path / "missing.json"
    monkeypatch.setenv("QT_WRAP_FILE", str(path))
    monkeypatch.setattr(W, "DEFAULT_PATH", path)
    wrap = W.load_wrap(date="2026-08-06")
    assert wrap["available"] is True
    assert wrap["source"] == "seed"
    assert any("Neuland" in b for b in wrap["bullets"])
    assert any("Tata Sons" in b for b in wrap["bullets"])


def test_notify_wrap_telegram(tmp_path, monkeypatch):
    from product import wrap_of_the_day as W

    path = tmp_path / "wrap.json"
    monkeypatch.setenv("QT_WRAP_FILE", str(path))
    monkeypatch.setattr(W, "DEFAULT_PATH", path)
    W.save_wrap(bullets=["HAL rallied on order book strength."], date="2026-08-06")

    class _Fake:
        def is_configured(self):
            return True

        def send(self, message: str):
            assert "Wrap of the Day" in message
            assert "HAL" in message
            return True

    monkeypatch.setattr("alerts.telegram_alerts.AlertEngine", _Fake)
    result = W.notify_wrap_telegram()
    assert result["sent"] is True
    assert result["count"] == 1


def test_wrap_api_save_and_get(tmp_path, monkeypatch):
    import terminal_product_api as api
    from product import wrap_of_the_day as W

    path = tmp_path / "wrap.json"
    monkeypatch.setenv("QT_WRAP_FILE", str(path))
    monkeypatch.setattr(W, "DEFAULT_PATH", path)

    client = TestClient(api.app)
    r = client.post(
        "/api/wrap-of-the-day",
        json={
            "text": "1) Neuland Laboratories reported a massive jump in profit.\n2) US futures edged higher.",
            "date": "2026-08-06",
            "notify": False,
        },
    )
    assert r.status_code == 200
    body = r.json()
    assert body["available"] is True
    assert len(body["bullets"]) == 2

    got = client.get("/api/wrap-of-the-day").json()
    assert got["bullets"][0].startswith("Neuland")


def test_pulse_prefers_wrap_takeaways(tmp_path, monkeypatch):
    from product import wrap_of_the_day as W
    from reports import street_pulse as SP

    wrap_path = tmp_path / "wrap.json"
    pulse_path = tmp_path / "pulse.json"
    monkeypatch.setenv("QT_WRAP_FILE", str(wrap_path))
    monkeypatch.setattr(W, "DEFAULT_PATH", wrap_path)
    monkeypatch.setattr(SP, "DEFAULT_PULSE_PATH", pulse_path)
    W.save_wrap(
        bullets=["Manipal Health corrected after listing.", "HAL order book in focus."],
        date="2026-08-06",
        source="paste",
    )
    monkeypatch.setattr(W, "today_ist", lambda: "2026-08-06")

    monkeypatch.setattr(SP, "_scan_universe", lambda: ([], 0, "", "unavailable"))
    monkeypatch.setattr(SP, "_movers_from_bhav", lambda top_n=5: ([], []))
    monkeypatch.setattr(
        SP,
        "_market_snapshot",
        lambda: {"indices": [], "commentary": "", "regime": "", "options_stance": None},
    )
    monkeypatch.setattr(SP, "_sector_heat", lambda: {"available": False, "leaders": [], "laggards": []})
    monkeypatch.setattr(SP, "_losing_momentum", lambda: None)
    monkeypatch.setattr(SP, "_sniper_breakouts", lambda limit=4: [])
    monkeypatch.setattr(SP, "_headlines", lambda max_n=5: [])
    monkeypatch.setattr(SP, "_global_cues", lambda: [])

    pulse = SP.build_pulse(persist=True)
    assert pulse["wrap_of_the_day"]["available"] is True
    assert pulse["takeaways"][0].startswith("Manipal")
    tg = SP.pulse_to_telegram(pulse)
    assert "Wrap of the Day" in tg
    assert "Manipal" in tg
