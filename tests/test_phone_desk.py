"""Phone desk — Telegram Home for a phone 1000 km away."""
from __future__ import annotations

from alerts.phone_desk import (
    allowed_chat_ids,
    desk_keyboard,
    format_desk,
    format_thesis,
    phone_may_run,
)


def test_stranger_cannot_run_commands():
    assert phone_may_run("999", "/desk", "123", "") is False
    assert phone_may_run("123", "/desk", "123", "") is True


def test_extra_phone_is_read_only():
    extra = "555"
    owner = "123"
    assert phone_may_run(extra, "/desk", owner, extra) is True
    assert phone_may_run(extra, "/thesis RELIANCE", owner, extra) is True
    assert phone_may_run(extra, "/status", owner, extra) is True
    assert phone_may_run(extra, "/trade", owner, extra) is False
    assert phone_may_run(extra, "/resume", owner, extra) is False
    assert phone_may_run(owner, "/trade", owner, extra) is True


def test_allowed_chat_ids_merges_owner_and_extras():
    owner, allowed = allowed_chat_ids("123", "555, 777")
    assert owner == "123"
    assert allowed == {"123", "555", "777"}


def test_format_desk_uses_scan_rows_not_vibes():
    rows = [
        {
            "symbol": "BSE", "company": "BSE", "verdict": "BUY", "score": 90,
            "price": 3447, "entry": 3447, "stop": 3316, "target": 3709,
            "upside_from_buy_pct": 7.6,
        },
        {"symbol": "SKIP", "verdict": "WATCH", "score": 99, "price": 10},
    ]
    text = format_desk(rows)
    assert "BSE" in text
    assert "/thesis BSE" in text
    assert "SKIP" not in text
    kb = desk_keyboard(rows)
    data = kb["inline_keyboard"][0][0]["callback_data"]
    assert data == "th|BSE"
    assert kb["inline_keyboard"][0][1]["callback_data"].startswith("pt|BSE|")


def test_guest_keyboard_has_thesis_not_paper():
    rows = [{"symbol": "HAL", "verdict": "BUY", "price": 100, "entry": 100, "stop": 90, "target": 120}]
    kb = desk_keyboard(rows, thesis_only=True)
    row = kb["inline_keyboard"][0]
    assert len(row) == 1
    assert row[0]["callback_data"] == "th|HAL"


def test_format_thesis_includes_requested_layers_and_stays_honest():
    text = format_thesis({
        "symbol": "EIMCOELECO",
        "company": "EIMCO ELECON (INDIA)",
        "headline": "Clicked name — evidence below.",
        "plan": {"buy": 2018, "stop": 1875.1, "target": 2303.7, "upside_from_buy_pct": 14.2},
        "sector_wave": {
            "headline": "Manufacturing & Capital Goods basket is ahead of Nifty.",
            "bullets": ["Sector: Manufacturing & Capital Goods (from screener)"],
        },
        "smart_money": {
            "headline": "Institutions added to this name recently.",
            "bullets": ["FII holding 3.25% (bought 0.15pp)", "No NSE bulk print"],
        },
        "earnings": {
            "bullets": ["P/E 30.2x", "Operating margin 18.0%"],
        },
        "order_book": {"note": "NSE quote HTTP 403", "source": "nse"},
    })
    assert "Sector wave" in text
    assert "FII / DII" in text
    assert "P/E 30.2x" in text
    assert "HTTP 403" in text
    assert "not an order" in text.lower() or "research" in text.lower()


def test_thesis_tap_does_not_place_a_trade(monkeypatch):
    import requests
    import alerts.telegram_actions as ta

    placed: list[int] = []
    posts: list[dict] = []
    monkeypatch.setattr(ta, "_do_paper_trade", lambda *a, **k: placed.append(1) or "paper")
    monkeypatch.setattr("alerts.phone_desk.load_thesis_text", lambda symbol: f"THESIS {symbol}")
    monkeypatch.setattr(requests, "post", lambda *a, **k: posts.append(k.get("json") or {}) or None)
    monkeypatch.setenv("TELEGRAM_CHAT_ID", "123")
    ta._handle_callback(
        {"id": "cb", "data": "th|BSE", "message": {"chat": {"id": 123}}},
        "tok",
        "123",
    )
    assert placed == []
    assert any("THESIS BSE" in str(item.get("text") or "") for item in posts)
    from alerts.telegram_commands import handle_command
    assert "Aise:" in handle_command("/thesis")
    help_text = handle_command("/start")
    assert "/desk" in help_text and "/thesis" in help_text
