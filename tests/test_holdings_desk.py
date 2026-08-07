"""Holdings desk: fund → tech → news → range/target → research verdict."""
from __future__ import annotations

import json

from product import holdings_desk as HD


def test_compose_verdict_exit_talks_fundamentals():
    stance, conf, thesis, tips = HD._compose_verdict(
        symbol="XYZ",
        tech_sev="critical",
        fund_sev="warn",
        news_bias="BAD",
        vs_entry_pct=-12.0,
        tech_available=True,
        fund_brief="Fundamentals (flags on valuation/leverage/growth — respect them): PE 45.0x · D/E 2.10",
        price_plan={"note": "range ₹90–₹110 · stop watch ₹89 · target watch ₹105", "stop_watch": 89, "target_watch": 105},
        flows={"bias": "DISTRIBUTION", "bias_label": "Both selling"},
        long_term=False,
    )
    assert stance == HD.STANCE_EXIT_WATCH
    assert conf >= 0.7
    assert "Fundamentals" in thesis or any("Fundamental" in t or "PE" in t for t in tips)
    assert any("stop" in t.lower() or "Price plan" in t for t in tips)


def test_compose_verdict_bse_long_term_exception():
    stance, conf, thesis, tips = HD._compose_verdict(
        symbol="BSE",
        tech_sev="good",
        fund_sev="good",
        news_bias="GOOD",
        vs_entry_pct=-3.0,
        tech_available=True,
        fund_brief="Fundamentals (constructive on cache): PE 40.0x · ROE 22.0%",
        price_plan={
            "note": "Long-term franchise book",
            "price": 100,
            "range_low": 95,
            "stop_watch": 94,
            "target_watch": 112,
            "reward_risk": 2.0,
        },
        flows={"bias": "SUPPORTIVE", "bias_label": "Both buying"},
        long_term=True,
    )
    assert stance in {HD.STANCE_ADD_WATCH, HD.STANCE_HOLD}
    assert "franchise" in thesis.lower() or any("franchise" in t.lower() or "long-term" in t.lower() for t in tips)
    assert conf >= 0.5


def test_compose_verdict_short_term_thin_rr_holds():
    stance, _conf, thesis, tips = HD._compose_verdict(
        symbol="RELIANCE",
        tech_sev="good",
        fund_sev="info",
        news_bias="NONE",
        vs_entry_pct=5.0,
        tech_available=True,
        fund_brief="Fundamentals (usable): PE 25.0x",
        price_plan={
            "note": "short-term",
            "price": 100,
            "range_low": 90,
            "stop_watch": 96,
            "target_watch": 102,
            "reward_risk": 0.5,
        },
        flows={"bias": "MIXED", "bias_label": "Mixed flows"},
        long_term=False,
    )
    assert stance == HD.STANCE_HOLD
    assert "short-term" in thesis.lower() or any("short-term" in t.lower() or "r:r" in t.lower() for t in tips)


def test_compose_verdict_incomplete_without_data():
    stance, conf, _thesis, tips = HD._compose_verdict(
        symbol="ABC",
        tech_sev="unknown",
        fund_sev="unknown",
        news_bias="NONE",
        vs_entry_pct=None,
        tech_available=False,
        fund_brief="Fundamentals cache missing",
        price_plan={"note": "n/a"},
        flows={},
        long_term=False,
    )
    assert stance == HD.STANCE_INCOMPLETE
    assert conf < 0.3
    assert any("telegram" in t.lower() or "analyse" in t.lower() for t in tips)


def test_humanize_flow_bias_short_label():
    from data.institutional_flows import humanize_flow_bias, parse_fii_dii

    plain = humanize_flow_bias("FII_SELLING_DII_ABSORBING")
    assert plain["bias_label"] == "FII sell · DII absorb"
    assert "ABSO" not in plain["bias_label"]
    parsed = parse_fii_dii(
        [
            {"category": "FII", "netValue": "-2000", "date": "07-Aug-2026"},
            {"category": "DII", "netValue": "1800", "date": "07-Aug-2026"},
        ]
    )
    assert parsed["bias"] == "FII_SELLING_DII_ABSORBING"
    assert parsed["bias_label"] == "FII sell · DII absorb"


def test_telegram_strip_html_hides_tags():
    from alerts.telegram_alerts import AlertEngine

    cleaned = AlertEngine.strip_html("<b>Holdings Desk</b>\n• <b>RELIANCE</b> → HOLD")
    assert "<b>" not in cleaned
    assert "Holdings Desk" in cleaned
    assert "RELIANCE" in cleaned


def test_run_holdings_desk_empty_book(tmp_path, monkeypatch):
    holdings_file = tmp_path / "holdings.json"
    desk_file = tmp_path / "desk.json"
    holdings_file.write_text(json.dumps({"available": False, "holdings": [], "summary": {"count": 0}}), encoding="utf-8")
    monkeypatch.setenv("QT_HOLDINGS_FILE", str(holdings_file))
    monkeypatch.setenv("QT_HOLDINGS_DESK_FILE", str(desk_file))
    monkeypatch.setattr("product.holdings_book.DEFAULT_PATH", holdings_file)
    monkeypatch.setattr(HD, "DEFAULT_PATH", desk_file)
    monkeypatch.setattr(HD, "_market_flows", lambda: {"available": False, "bias_label": "Flows unknown"})

    desk = HD.run_holdings_desk(persist=True, path=desk_file, holdings_path=holdings_file)
    assert desk["available"] is False
    assert desk["places_orders"] is False
    assert desk["rows"] == []


def test_run_holdings_desk_scores_with_price_plan(tmp_path, monkeypatch):
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
            },
            {
                "tradingsymbol": "BSE",
                "research_symbol": "BSE",
                "quantity": 5,
                "average_price": 2000,
                "last_price": 1950,
                "pnl": -250,
                "pnl_pct": -2.5,
            },
        ],
        "summary": {"count": 2},
    }
    holdings_file.write_text(json.dumps(book), encoding="utf-8")
    monkeypatch.setenv("QT_HOLDINGS_FILE", str(holdings_file))
    monkeypatch.setenv("QT_HOLDINGS_DESK_FILE", str(desk_file))
    monkeypatch.setattr(HD, "DEFAULT_PATH", desk_file)
    monkeypatch.setattr(
        HD,
        "_market_flows",
        lambda: {
            "available": True,
            "bias": "FII_SELLING_DII_ABSORBING",
            "bias_label": "FII sell · DII absorb",
            "bias_note": "Foreign selling, domestic absorbing.",
            "as_of": "2026-08-07",
        },
    )

    def _fake_eval(symbol, entry_price=None, live_price=None, **_kw):
        return {
            "symbol": symbol,
            "available": True,
            "severity": "good",
            "status_label": "HEALTHY",
            "price": live_price or 980,
            "vs_entry_pct": -2.0,
            "warnings": [],
            "averages": {"ema20": 990, "ema50": 970, "ema200": 900},
            "supports": {"swing_20d": 950, "swing_60d": 900},
            "resistances": {"swing_20d": 1020, "swing_60d": 1100},
            "technicals": {
                "available": True,
                "severity": "good",
                "status_label": "HEALTHY",
                "warnings": [],
                "structure": {"chg_1d_pct": -0.5},
                "averages": {"ema20": 990, "ema50": 970, "ema200": 900},
                "supports": {"swing_20d": 950, "swing_60d": 900},
                "resistances": {"swing_20d": 1020, "swing_60d": 1100},
            },
            "fundamentals": {
                "available": True,
                "severity": "good",
                "status": "FRESH",
                "ratios": {"pe": 22.0, "roe": 15.0, "debt_to_equity": 0.4},
                "flags": [],
            },
        }

    monkeypatch.setattr("product.buy_health.evaluate_symbol", _fake_eval)
    monkeypatch.setattr(
        HD,
        "_news_bias",
        lambda symbol, hours=72: {
            "available": True,
            "bias": "GOOD",
            "label": "News lean positive",
            "positive": 2,
            "negative": 0,
            "mixed": 0,
            "unclear": 0,
            "headlines": [{"title": f"{symbol} order win", "tone": "GOOD", "source": "test"}],
            "hours": hours,
        },
    )

    desk = HD.run_holdings_desk(persist=True, path=desk_file, holdings_path=holdings_file)
    assert desk["available"] is True
    assert desk["holdings_count"] == 2
    assert desk["market_flows"]["bias_label"] == "FII sell · DII absorb"
    by_sym = {r["symbol"]: r for r in desk["rows"]}
    assert by_sym["RELIANCE"]["horizon"] == "SHORT_TERM"
    assert by_sym["BSE"]["horizon"] == "LONG_TERM"
    assert by_sym["RELIANCE"]["price_plan"]["stop_watch"] is not None
    assert by_sym["RELIANCE"]["price_plan"]["target_watch"] is not None
    assert "PE" in (by_sym["RELIANCE"]["fund_brief"] or "")


def test_notify_refuses_before_analysis(monkeypatch):
    result = HD.notify_holdings_desk_telegram({"available": False, "rows": []})
    assert result["sent"] is False
    assert "analys" in (result.get("reason") or "").lower()


def test_notify_holdings_desk_telegram(monkeypatch):
    from alerts.telegram_alerts import AlertEngine as RealEngine

    class _Fake:
        last_error = None
        cred_source = "test"
        escape = staticmethod(RealEngine.escape)
        strip_html = staticmethod(RealEngine.strip_html)

        def is_configured(self):
            return True

        def send(self, message: str):
            assert "Holdings Desk" in message
            assert "RELIANCE" in message
            assert "Fundamentals" in message or "PE" in message
            return True

    monkeypatch.setattr("alerts.telegram_alerts.AlertEngine", _Fake)
    desk = {
        "available": True,
        "message": "1 holding scored · FII/DII: FII sell · DII absorb",
        "market_flows": {"bias_label": "FII sell · DII absorb", "as_of": "2026-08-07"},
        "rows": [
            {
                "symbol": "RELIANCE",
                "suggestion": "HOLD",
                "stance": "HOLD",
                "horizon": "SHORT_TERM",
                "fund_brief": "Fundamentals (usable): PE 22.0x",
                "thesis": "Hold and monitor.",
                "price_plan": {"range_low": 950, "range_high": 1020, "stop_watch": 945, "target_watch": 1020},
                "technicals": {"status_label": "HEALTHY"},
                "fundamentals": {"severity": "good", "brief": "Fundamentals (usable): PE 22.0x"},
                "news": {"bias": "NONE"},
            }
        ],
    }
    result = HD.notify_holdings_desk_telegram(desk)
    assert result["sent"] is True
    assert result["count"] == 1
