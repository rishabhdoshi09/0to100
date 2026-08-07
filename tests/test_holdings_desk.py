"""Holdings desk: fund → tech → news → research verdict (never places orders)."""
from __future__ import annotations

import json

from product import holdings_desk as HD


def test_compose_verdict_exit_on_critical_tech_and_bad_news():
    stance, conf, thesis, tips = HD._compose_verdict(
        tech_sev="critical",
        fund_sev="warn",
        news_bias="BAD",
        vs_entry_pct=-12.0,
        tech_available=True,
    )
    assert stance == HD.STANCE_EXIT_WATCH
    assert conf >= 0.7
    assert "exit" in thesis.lower() or "risk" in thesis.lower()
    assert tips
    assert all("order" not in t.lower() or "never" in t.lower() or "not" in t.lower() for t in tips)


def test_compose_verdict_add_watch_on_healthy_pullback():
    stance, conf, thesis, tips = HD._compose_verdict(
        tech_sev="good",
        fund_sev="good",
        news_bias="GOOD",
        vs_entry_pct=-2.0,
        tech_available=True,
    )
    assert stance == HD.STANCE_ADD_WATCH
    assert conf >= 0.5
    assert "add" in " ".join(tips).lower() or "hold" in thesis.lower()


def test_compose_verdict_incomplete_without_data():
    stance, conf, _thesis, _tips = HD._compose_verdict(
        tech_sev="unknown",
        fund_sev="unknown",
        news_bias="NONE",
        vs_entry_pct=None,
        tech_available=False,
    )
    assert stance == HD.STANCE_INCOMPLETE
    assert conf < 0.3


def test_run_holdings_desk_empty_book(tmp_path, monkeypatch):
    holdings_file = tmp_path / "holdings.json"
    desk_file = tmp_path / "desk.json"
    holdings_file.write_text(json.dumps({"available": False, "holdings": [], "summary": {"count": 0}}), encoding="utf-8")
    monkeypatch.setenv("QT_HOLDINGS_FILE", str(holdings_file))
    monkeypatch.setenv("QT_HOLDINGS_DESK_FILE", str(desk_file))
    monkeypatch.setattr("product.holdings_book.DEFAULT_PATH", holdings_file)
    monkeypatch.setattr(HD, "DEFAULT_PATH", desk_file)

    desk = HD.run_holdings_desk(persist=True, path=desk_file, holdings_path=holdings_file)
    assert desk["available"] is False
    assert desk["places_orders"] is False
    assert desk["rows"] == []


def test_run_holdings_desk_scores_rows(tmp_path, monkeypatch):
    holdings_file = tmp_path / "holdings.json"
    desk_file = tmp_path / "desk.json"
    book = {
        "available": True,
        "holdings": [
            {
                "tradingsymbol": "RELIANCE",
                "research_symbol": "RELIANCE",
                "quantity": 10,
                "average_price": 1000,
                "last_price": 980,
                "pnl": -200,
                "pnl_pct": -2.0,
            }
        ],
        "summary": {"count": 1, "invested": 10000, "pnl": -200, "pnl_pct": -2.0},
    }
    holdings_file.write_text(json.dumps(book), encoding="utf-8")
    monkeypatch.setenv("QT_HOLDINGS_FILE", str(holdings_file))
    monkeypatch.setenv("QT_HOLDINGS_DESK_FILE", str(desk_file))
    monkeypatch.setattr(HD, "DEFAULT_PATH", desk_file)

    def _fake_eval(symbol, entry_price=None, live_price=None, **_kw):
        return {
            "symbol": symbol,
            "available": True,
            "severity": "good",
            "status_label": "HEALTHY",
            "price": live_price or 980,
            "vs_entry_pct": -2.0,
            "warnings": [],
            "technicals": {
                "available": True,
                "severity": "good",
                "status_label": "HEALTHY",
                "warnings": [],
                "structure": {"chg_1d_pct": -0.5},
            },
            "fundamentals": {
                "available": True,
                "severity": "good",
                "status": "FRESH",
                "ratios": {"pe": 20},
                "flags": [],
            },
        }

    monkeypatch.setattr("product.buy_health.evaluate_symbol", _fake_eval)
    monkeypatch.setattr(HD, "_news_bias", lambda symbol, hours=72: {
        "available": True,
        "bias": "GOOD",
        "label": "News lean positive",
        "positive": 2,
        "negative": 0,
        "mixed": 0,
        "unclear": 0,
        "headlines": [{"title": "Reliance wins order", "tone": "GOOD", "source": "test"}],
        "hours": hours,
    })

    desk = HD.run_holdings_desk(persist=True, path=desk_file, holdings_path=holdings_file)
    assert desk["available"] is True
    assert desk["holdings_count"] == 1
    assert desk["places_orders"] is False
    row = desk["rows"][0]
    assert row["symbol"] == "RELIANCE"
    assert row["stance"] in {HD.STANCE_ADD_WATCH, HD.STANCE_HOLD}
    assert "BUY" in (row["suggestion"] or "") or row["stance"] == HD.STANCE_HOLD
    assert row["news"]["bias"] == "GOOD"
    loaded = HD.load_desk(desk_file)
    assert loaded["available"] is True


def test_notify_holdings_desk_telegram(monkeypatch):
    class _Fake:
        last_error = None
        cred_source = "test"

        def is_configured(self):
            return True

        def send(self, message: str):
            assert "Holdings Desk" in message
            assert "RELIANCE" in message
            return True

    monkeypatch.setattr("alerts.telegram_alerts.AlertEngine", _Fake)
    desk = {
        "available": True,
        "message": "1 holding scored",
        "rows": [
            {
                "symbol": "RELIANCE",
                "suggestion": "HOLD",
                "stance": "HOLD",
                "thesis": "Hold and monitor.",
                "technicals": {"status_label": "HEALTHY"},
                "fundamentals": {"severity": "good"},
                "news": {"bias": "NONE"},
            }
        ],
    }
    result = HD.notify_holdings_desk_telegram(desk)
    assert result["sent"] is True
    assert result["count"] == 1


def test_telegram_credentials_resolve_from_dotenv_file(tmp_path, monkeypatch):
    from alerts import telegram_alerts as TG

    env_file = tmp_path / ".env"
    env_file.write_text("TELEGRAM_BOT_TOKEN=tok123\nTELEGRAM_CHAT_ID=42\n", encoding="utf-8")
    monkeypatch.setattr(TG, "_ROOT", tmp_path)
    monkeypatch.delenv("TELEGRAM_BOT_TOKEN", raising=False)
    monkeypatch.delenv("TELEGRAM_CHAT_ID", raising=False)

    # Avoid Settings picking up a real project .env
    class _S:
        telegram_bot_token = ""
        telegram_chat_id = ""

    monkeypatch.setattr("config.settings", _S(), raising=False)
    token, chat, source = TG.resolve_telegram_credentials()
    assert token == "tok123"
    assert chat == "42"
    assert source in {"dotenv_file", "dotenv_load", "settings", "process_env"}
