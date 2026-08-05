"""Active Buys book + technical health warnings."""
from __future__ import annotations

import numpy as np
import pandas as pd


def test_add_and_remove_buy_book(tmp_path, monkeypatch):
    from product import buy_book as BB

    monkeypatch.setattr(BB, "DEFAULT_PATH", tmp_path / "buy_book.json")
    item = BB.add_item("RELIANCE", entry_price=2500, stop_price=2400, quantity=10, notes="core buy")
    assert item["symbol"] == "RELIANCE"
    assert item["entry_price"] == 2500
    assert item["quantity"] == 10
    assert BB.symbols() == ["RELIANCE"]
    # Upsert same symbol
    again = BB.add_item("RELIANCE", entry_price=2510, quantity=12)
    assert again["entry_price"] == 2510
    assert again["quantity"] == 12
    assert len(BB.list_active()) == 1
    assert BB.remove_item(item["id"]) is True
    assert BB.list_active() == []


def test_results_summary_up_down_and_est_pnl(tmp_path, monkeypatch):
    from product import buy_book as BB
    from product import buy_health as BH

    monkeypatch.setattr(BB, "DEFAULT_PATH", tmp_path / "buy_book.json")
    BB.add_item("WIN", entry_price=100, quantity=5)
    BB.add_item("LOSE", entry_price=100, quantity=2)
    BB.add_item("NOENTRY")

    def fake_eval(symbol, **kwargs):
        entry = kwargs.get("entry_price")
        price = {"WIN": 110.0, "LOSE": 90.0, "NOENTRY": 50.0}[symbol]
        vs = None if entry is None else round((price / entry - 1.0) * 100.0, 2)
        return {
            "symbol": symbol,
            "available": True,
            "severity": "good",
            "status_label": "HEALTHY",
            "price": price,
            "warnings": [],
            "risk_score": 0,
            "supports": {},
            "averages": {},
            "structure": {"chg_1d_pct": 1.0, "chg_5d_pct": 2.0},
            "vs_entry_pct": vs,
        }

    monkeypatch.setattr(BH, "evaluate_symbol", fake_eval)
    monkeypatch.setattr("data.live_quotes.get_live_quotes", lambda symbols, ttl=8.0: {})
    payload = BH.evaluate_book()
    results = payload["results"]
    assert results["up"] == 1
    assert results["down"] == 1
    assert results["missing_entry"] == 1
    assert results["avg_vs_entry_pct"] == 0.0
    by_sym = {r["symbol"]: r for r in payload["items"]}
    assert by_sym["WIN"]["result_label"] == "UP"
    assert by_sym["WIN"]["est_pnl"] == 50.0
    assert by_sym["LOSE"]["est_pnl"] == -20.0
    assert by_sym["NOENTRY"]["est_pnl"] is None
    assert results["est_pnl_total"] == 30.0


def test_evaluate_symbol_flags_death_stack_and_support(monkeypatch):
    from product import buy_health as BH

    # Synthetic downtrend: price falling, below stacked EMAs and 20d low.
    n = 220
    close = np.linspace(200, 100, n)
    high = close + 2
    low = close - 2
    vol = np.full(n, 1_000_000.0)
    vol[-1] = 3_000_000.0
    idx = pd.date_range("2025-01-01", periods=n, freq="B")
    frame = pd.DataFrame({"close": close, "high": high, "low": low, "volume": vol}, index=idx)

    monkeypatch.setattr("data.bhavcopy_runtime.get_ohlcv", lambda symbol: frame)
    health = BH.evaluate_symbol("TESTCO", entry_price=180.0, stop_price=110.0, live_price=99.0)
    assert health["available"] is True
    assert health["severity"] == "critical"
    codes = {w["code"] for w in health["warnings"]}
    assert "DEATH_STACK" in codes or "BELOW_EMA50" in codes
    assert "STOP_BREACH" in codes or "SUPPORT_20D" in codes
    assert health["price_source"] == "live"


def test_evaluate_healthy_uptrend(monkeypatch):
    from product import buy_health as BH

    n = 220
    close = np.linspace(100, 200, n)
    high = close + 1
    low = close - 1
    vol = np.full(n, 500_000.0)
    idx = pd.date_range("2025-01-01", periods=n, freq="B")
    frame = pd.DataFrame({"close": close, "high": high, "low": low, "volume": vol}, index=idx)
    monkeypatch.setattr("data.bhavcopy_runtime.get_ohlcv", lambda symbol: frame)
    health = BH.evaluate_symbol("GOODCO", live_price=199.0)
    assert health["severity"] == "good"
    assert health["status_label"] == "HEALTHY"


def test_evaluate_book_cache_avoids_repeat_work(tmp_path, monkeypatch):
    from product import buy_book as BB
    from product import buy_health as BH

    monkeypatch.setattr(BB, "DEFAULT_PATH", tmp_path / "buy_book.json")
    BH.invalidate_eval_cache()
    BB.add_item("CACHECO", entry_price=100)

    calls = {"n": 0}

    def fake_eval(symbol, **kwargs):
        calls["n"] += 1
        return {
            "symbol": symbol,
            "available": True,
            "severity": "good",
            "status_label": "HEALTHY",
            "price": 101,
            "warnings": [],
            "risk_score": 0,
            "supports": {},
            "averages": {},
            "structure": {"chg_1d_pct": 0.5, "chg_5d_pct": 1.0},
            "vs_entry_pct": 1.0,
        }

    monkeypatch.setattr(BH, "evaluate_symbol", fake_eval)
    monkeypatch.setattr("data.live_quotes.get_live_quotes", lambda symbols, ttl=8.0: {})
    first = BH.evaluate_book()
    second = BH.evaluate_book()
    assert calls["n"] == 1
    assert first["cached"] is False
    assert second["cached"] is True
    forced = BH.evaluate_book(force=True)
    assert calls["n"] == 2
    assert forced["cached"] is False


def test_evaluate_book_sorts_risk_first(tmp_path, monkeypatch):
    from product import buy_book as BB
    from product import buy_health as BH

    monkeypatch.setattr(BB, "DEFAULT_PATH", tmp_path / "buy_book.json")
    BH.invalidate_eval_cache()
    BB.add_item("AAA")
    BB.add_item("BBB")

    def fake_eval(symbol, **kwargs):
        if symbol == "AAA":
            return {
                "symbol": symbol,
                "available": True,
                "severity": "critical",
                "status_label": "AT RISK",
                "price": 10,
                "warnings": [{"severity": "critical", "code": "X", "text": "bad"}],
                "risk_score": 80,
                "supports": {},
                "averages": {},
                "structure": {},
                "vs_entry_pct": -9,
            }
        return {
            "symbol": symbol,
            "available": True,
            "severity": "good",
            "status_label": "HEALTHY",
            "price": 20,
            "warnings": [{"severity": "good", "code": "OK", "text": "ok"}],
            "risk_score": 0,
            "supports": {},
            "averages": {},
            "structure": {},
            "vs_entry_pct": 2,
        }

    monkeypatch.setattr(BH, "evaluate_symbol", fake_eval)
    monkeypatch.setattr("data.live_quotes.get_live_quotes", lambda symbols, ttl=8.0: {})
    payload = BH.evaluate_book(force=True)
    assert payload["summary"]["critical"] == 1
    assert payload["items"][0]["symbol"] == "AAA"
    assert payload["places_orders"] is False
    assert "results" in payload
