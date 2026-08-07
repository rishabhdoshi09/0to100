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


def test_sync_holdings_into_active_buys(tmp_path, monkeypatch):
    from product import buy_book as BB
    from product import holdings_book as HB

    monkeypatch.setattr(BB, "DEFAULT_PATH", tmp_path / "buy_book.json")
    monkeypatch.setattr(HB, "DEFAULT_PATH", tmp_path / "holdings.json")
    HB.save_holdings(
        [
            {
                "tradingsymbol": "INFY",
                "quantity": 15,
                "average_price": 1400,
                "pnl": 1200,
                "pnl_pct": 5.5,
            },
            {
                "tradingsymbol": "TCS-BE",
                "quantity": 2,
                "average_price": 3200,
                "pnl": -100,
                "pnl_pct": -1.5,
            },
        ],
        source="kite",
    )
    BB.add_item("MANUAL", entry_price=10, notes="keep me", source="manual")

    report = BB.sync_from_holdings(refresh_kite=False)
    assert report["upserted"] == 2
    assert set(report["symbols"]) == {"INFY", "TCS"}
    active = {r["symbol"]: r for r in BB.list_active()}
    assert active["INFY"]["source"] == "zerodha"
    assert active["INFY"]["entry_price"] == 1400
    assert active["INFY"]["quantity"] == 15
    assert active["TCS"]["entry_price"] == 3200
    assert active["MANUAL"]["source"] == "manual"

    # Drop INFY from demat — zerodha row should close; manual stays.
    HB.save_holdings(
        [{"tradingsymbol": "TCS", "quantity": 2, "average_price": 3200}],
        source="kite",
    )
    again = BB.sync_from_holdings(refresh_kite=False)
    assert "INFY" in again["closed_stale_zerodha"]
    symbols = set(BB.symbols())
    assert "INFY" not in symbols
    assert "TCS" in symbols
    assert "MANUAL" in symbols


def test_buy_book_watcher_includes_fund_flags(monkeypatch):
    from risk import buy_book_watcher as W

    monkeypatch.setattr(
        "product.buy_health.evaluate_book",
        lambda: {
            "items": [
                {
                    "symbol": "WEAK",
                    "health": {
                        "price": 100,
                        "warnings": [
                            {"severity": "critical", "code": "DEATH_STACK", "text": "below EMAs"},
                            {"severity": "info", "code": "NOTE", "text": "ignore"},
                        ],
                        "fundamentals": {
                            "flags": [
                                {"severity": "warn", "code": "HIGH_DEBT", "text": "Debt elevated"},
                                {"severity": "good", "code": "SOLID_ROE", "text": "ok"},
                            ]
                        },
                    },
                }
            ]
        },
    )
    events = W.check_buy_book()
    codes = {e["event"] for e in events}
    assert "TECH:DEATH_STACK" in codes
    assert "FUND:HIGH_DEBT" in codes
    assert "TECH:NOTE" not in codes
    assert "FUND:SOLID_ROE" not in codes

    sent_msgs: list[str] = []

    class _FakeEngine:
        def is_configured(self):
            return True

        def send(self, message: str):
            sent_msgs.append(message)
            return True

    monkeypatch.setattr("alerts.telegram_alerts.AlertEngine", _FakeEngine)
    W._alerted.clear()
    n = W.push_buy_book_alerts()
    assert n == 2
    assert sent_msgs and "WEAK" in sent_msgs[0]
    assert W.push_buy_book_alerts() == 0  # once per day


def test_sync_from_holdings_notifies_active_buys(tmp_path, monkeypatch):
    from product import buy_book as BB
    from product import holdings_book as HB

    monkeypatch.setattr(BB, "DEFAULT_PATH", tmp_path / "buy_book.json")
    monkeypatch.setattr(HB, "DEFAULT_PATH", tmp_path / "holdings.json")
    monkeypatch.setenv("QT_HOLDINGS_FILE", str(tmp_path / "holdings.json"))
    HB.save_holdings(
        [{"tradingsymbol": "INFY", "quantity": 5, "average_price": 1400, "last_price": 1450}],
        source="paste",
    )
    sent: list[str] = []

    class _FakeEngine:
        def is_configured(self):
            return True

        def send(self, message: str):
            sent.append(message)
            return True

    monkeypatch.setattr("alerts.telegram_alerts.AlertEngine", _FakeEngine)
    report = BB.sync_from_holdings(refresh_kite=False, notify=True)
    assert report["upserted"] == 1
    assert report["telegram"].get("active_buys_sent") is True
    assert any("Active Buys" in m for m in sent)


def test_sync_from_holdings_fetch_research(tmp_path, monkeypatch):
    from product import buy_book as BB
    from product import holdings_book as HB

    monkeypatch.setattr(BB, "DEFAULT_PATH", tmp_path / "buy_book.json")
    monkeypatch.setattr(HB, "DEFAULT_PATH", tmp_path / "holdings.json")
    monkeypatch.setenv("QT_HOLDINGS_FILE", str(tmp_path / "holdings.json"))
    HB.save_holdings(
        [{"tradingsymbol": "INFY", "quantity": 5, "average_price": 1400, "last_price": 1450}],
        source="paste",
    )

    called: dict[str, object] = {}

    def fake_research(symbols, *, force_fundamentals=False, max_symbols=None):
        called["symbols"] = list(symbols)
        called["force"] = force_fundamentals
        return {
            "accepted": True,
            "message": "Research refresh: fund ok 1/1 · tech ready 1/1",
            "fundamentals": {"ok": 1, "cached": 0, "failed": 0},
            "technicals": {"ok": 1, "thin_or_missing": 0},
            "places_orders": False,
        }

    monkeypatch.setattr("product.buy_health.refresh_book_research", fake_research)
    report = BB.sync_from_holdings(refresh_kite=False, fetch_research=True, force_fundamentals=True)
    assert report["upserted"] == 1
    assert report["fetch_research"] is True
    assert report["research"]["accepted"] is True
    assert called["symbols"] == ["INFY"]
    assert called["force"] is True


def test_refresh_book_research_scores_fund_and_tech(tmp_path, monkeypatch):
    from product import buy_book as BB
    from product import buy_health as BH

    monkeypatch.setattr(BB, "DEFAULT_PATH", tmp_path / "buy_book.json")
    BB.add_item("INFY", entry_price=1400, quantity=5, source="zerodha")

    monkeypatch.setattr(
        "fundamentals.lazy.ensure_deep_fundamentals",
        lambda symbol, force_refresh=False: {"about": "IT", "key_ratios": []},
    )

    class _Cache:
        def has(self, symbol):
            return False

    monkeypatch.setattr("fundamentals.cache.FundamentalsCache", lambda: _Cache())
    monkeypatch.setattr(
        "data.bhavcopy_runtime.ensure_loaded",
        lambda rebuild_from_local=False: {"ready": True, "symbols": 1},
    )
    idx = pd.date_range("2025-01-01", periods=40, freq="B")
    frame = pd.DataFrame(
        {
            "close": np.linspace(100, 120, 40),
            "high": np.linspace(101, 121, 40),
            "low": np.linspace(99, 119, 40),
            "volume": np.full(40, 1_000_000.0),
        },
        index=idx,
    )
    monkeypatch.setattr("data.bhavcopy_runtime.get_ohlcv", lambda symbol: frame)

    out = BH.refresh_book_research(["INFY"], force_fundamentals=False)
    assert out["accepted"] is True
    assert out["fundamentals"]["ok"] == 1
    assert out["technicals"]["ok"] == 1
    assert out["places_orders"] is False
    assert "INFY" in {r["symbol"] for r in out["rows"]}


def test_fundamentals_snapshot_from_cache(monkeypatch):
    from product import buy_health as BH

    monkeypatch.setattr(
        "reporting.evidence_intake.load_raw_fundamentals",
        lambda symbol, auto_fetch=False: {
            "available": True,
            "fetched_at": "2026-08-01T00:00:00+00:00",
            "freshness": "FRESH",
            "data": {
                "about": "IT services company",
                "key_ratios": [
                    {"name": "Stock P/E", "value": "25.5"},
                    {"name": "ROE", "value": "22%"},
                    {"name": "Debt to equity", "value": "0.1"},
                ],
                "profit_loss": [
                    {"row_label": "Sales", "Mar 2024": 100, "Mar 2025": 120},
                    {"row_label": "Net Profit", "Mar 2024": 20, "Mar 2025": 26},
                ],
            },
        },
    )
    snap = BH.fundamentals_snapshot("INFY")
    assert snap["available"] is True
    assert snap["ratios"]["pe"] == 25.5
    assert snap["ratios"]["roe"] == 22.0
    assert snap["ratios"]["sales_growth_pct"] == 20.0
    assert any(f["code"] == "SOLID_ROE" for f in snap["flags"])


def test_evaluate_symbol_exposes_technicals_and_fundamentals(monkeypatch):
    from product import buy_health as BH
    import numpy as np
    import pandas as pd

    n = 220
    close = np.linspace(100, 200, n)
    frame = pd.DataFrame(
        {"close": close, "high": close + 1, "low": close - 1, "volume": np.full(n, 500_000.0)},
        index=pd.date_range("2025-01-01", periods=n, freq="B"),
    )
    monkeypatch.setattr("data.bhavcopy_runtime.get_ohlcv", lambda symbol: frame)
    monkeypatch.setattr(
        BH,
        "fundamentals_snapshot",
        lambda symbol: {
            "available": True,
            "status": "FRESH",
            "severity": "good",
            "risk_score": 0,
            "ratios": {"pe": 18, "roe": 20},
            "flags": [{"severity": "good", "code": "SOLID_ROE", "text": "ROE ok"}],
            "about": "demo",
            "fetched_at": "2026-08-01",
            "freshness": "FRESH",
            "note": "ok",
        },
    )
    health = BH.evaluate_symbol("GOODCO", live_price=199.0)
    assert health["technicals"]["available"] is True
    assert health["fundamentals"]["available"] is True
    assert health["technicals"]["averages"]["ema20"] is not None
