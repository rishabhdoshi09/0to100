"""A corrupt daily premium ledger must block entries, not open-position exits."""
from datetime import datetime, timedelta, timezone

from product.fo_paper_runtime import run_fo_paper_cycle
from product.fo_paper_store import FoPaperStore


IST = timezone(timedelta(hours=5, minutes=30))


class Quotes:
    def quote(self, keys):
        return {
            key: {
                "last_price": 52.0,
                "volume": 5000,
                "oi": 20000,
                "ohlc": {"open": 50.0, "high": 80.0, "low": 30.0, "close": 52.0},
                "depth": {"buy": [{"price": 51.0}], "sell": [{"price": 52.0}]},
            }
            for key in keys
        }


def candidate(symbol, contract):
    return {
        "symbol": symbol, "decision": "PAPER_OPTION_CANDIDATE",
        "setup": {"score": 82.0, "expected_move": {"holding_days": 2}},
        "selected_contract": {
            "symbol": contract, "option_type": "CE", "eligible": True,
            "lot_size": 25, "ask": 50.0, "score": 85.0,
            "trade_plan": {"entry": 50.0, "stop": 40.0, "target": 70.0},
        },
    }


def test_corrupt_closed_trade_still_allows_existing_option_settlement(tmp_path, monkeypatch):
    from risk import portfolio_risk as risk
    monkeypatch.setattr(risk, "equity_books_open_risk", lambda: 0.0)
    monkeypatch.setattr(risk, "_capital", lambda: 200_000.0)
    path = tmp_path / "fo.sqlite3"
    first = datetime(2026, 9, 28, 10, 15, tzinfo=IST)
    with FoPaperStore(path) as store:
        opened = run_fo_paper_cycle(
            {"available": True, "candidates": [candidate("RELIANCE", "RELIANCE26OCTCE")]},
            client=Quotes(), now_ist=first, allow_new_entries=True,
            store=store, capital=200_000,
        )
        assert opened["opened_count"] == 1
        # A previous closed trade was corrupted on disk. Without fail-closed
        # handling this previously vanished from the daily spending calculation.
        store.append_trades([{
            "trade_id": "CORRUPT", "option_symbol": "OLDER26SEPCE",
            "underlying": "OLDER", "entry_price": 10.0, "quantity": 25,
            "opened_at": "2026-09-28T09:30:00+05:30",
            "settled_at": "2026-09-28T10:30:00+05:30",
        }])
        store.conn.execute(
            "UPDATE fo_closed_trades SET payload_json = '{broken' WHERE trade_id = 'CORRUPT'"
        )
        store.conn.commit()
        result = run_fo_paper_cycle(
            {"available": True, "candidates": [candidate("TCS", "TCS26OCTCE")]},
            client=Quotes(), now_ist=first + timedelta(days=1),
            allow_new_entries=True, store=store, capital=200_000,
        )
        assert result["settled_count"] == 1
        assert result["opened_count"] == 0
        assert result["open_count"] == 0
        assert result["new_entries_allowed"] is False
        assert "FNO_DAILY_PREMIUM_LEDGER_UNTRUSTED" in result["blockers"]
        assert result["operator_action_required"] is True
        assert result["daily_premium_ledger_error"]
        assert result["premium_deployed_today"] is None
        assert result["daily_premium_budget"] is None
        assert result["daily_premium_remaining"] is None
        assert result["skipped"][0]["reason"] == "DAILY_PREMIUM_LEDGER_UNTRUSTED"
