from datetime import datetime, timedelta, timezone

from product.fo_paper_runtime import run_fo_paper_cycle
from product.fo_paper_store import FoPaperStore


IST = timezone(timedelta(hours=5, minutes=30))


def _candidate():
    return {
        "symbol": "RELIANCE",
        "direction": "LONG",
        "decision": "PAPER_OPTION_CANDIDATE",
        "setup": {
            "score": 82.0,
            "expected_move": {"holding_days": 2},
        },
        "selected_contract": {
            "symbol": "RELIANCE26OCTCE",
            "option_type": "CE",
            "eligible": True,
            "lot_size": 25,
            "ask": 50.0,
            "score": 86.0,
            "context_key": "LONG|LONG_BUILDUP|CTX",
            "trade_plan": {
                "entry": 50.0,
                "stop": 40.0,
                "target": 70.0,
            },
        },
    }


def _directional():
    return {
        "available": True,
        "status": "READY",
        "decision": "PAPER_CANDIDATES",
        "candidates": [_candidate()],
        "paper_only": True,
        "live_execution_allowed": False,
    }


class _QuoteClient:
    def __init__(self, *, last=52.0, high=80.0, low=30.0, bid=51.0):
        self.last = last
        self.high = high
        self.low = low
        self.bid = bid

    def quote(self, keys):
        return {
            key: {
                "last_price": self.last,
                "volume": 5000,
                "oi": 20000,
                "ohlc": {
                    "open": 50.0,
                    "high": self.high,
                    "low": self.low,
                    "close": 49.0,
                },
                "depth": {
                    "buy": [{"price": self.bid}],
                    "sell": [{"price": self.bid + 1.0}],
                },
            }
            for key in keys
        }


def test_sqlite_store_persists_positions_and_trade_rows(tmp_path):
    store = FoPaperStore(tmp_path / "fo.sqlite3")
    try:
        store.replace_positions([{
            "trade_id": "T1",
            "option_symbol": "ABC26OCTCE",
            "underlying": "ABC",
            "entry_price": 10,
        }])
        assert store.load_positions()[0]["trade_id"] == "T1"
        row = {
            "trade_id": "T1",
            "option_symbol": "ABC26OCTCE",
            "underlying": "ABC",
            "context_key": "CTX",
            "evidence_lane": "FORWARD_PAPER",
            "settled_at": "2026-09-26",
            "production_evidence_eligible": False,
        }
        assert store.append_trades([row]) == 1
        assert store.append_trades([row]) == 0
        assert store.status()["closed_trades"] == 1
    finally:
        store.close()


def test_paper_runtime_survives_restart_and_avoids_pre_entry_daily_range(tmp_path):
    path = tmp_path / "fo.sqlite3"
    opened_at = datetime(2026, 9, 26, 10, 15, tzinfo=IST)

    with FoPaperStore(path) as store:
        first = run_fo_paper_cycle(
            _directional(),
            client=_QuoteClient(),
            now_ist=opened_at,
            allow_new_entries=True,
            store=store,
            capital=200_000,
        )
        assert first["opened_count"] == 1
        assert first["open_count"] == 1
        assert first["production_evidence_enabled"] is False

    # New book/process, same durable DB. Full-day high/low include time before
    # the entry, so same-session marking must not falsely stop/target the trade.
    with FoPaperStore(path) as store:
        same_day = run_fo_paper_cycle(
            {"available": True, "candidates": []},
            client=_QuoteClient(last=52, high=80, low=30, bid=51),
            now_ist=opened_at.replace(hour=14),
            allow_new_entries=False,
            store=store,
            capital=200_000,
        )
        assert same_day["settled_count"] == 0
        assert same_day["open_count"] == 1

    # On the next session the position existed from the open, so the day's
    # range is legitimate evidence; both stop and target hit => stop first.
    with FoPaperStore(path) as store:
        next_day = run_fo_paper_cycle(
            {"available": True, "candidates": []},
            client=_QuoteClient(last=52, high=80, low=30, bid=51),
            now_ist=opened_at + timedelta(days=1),
            allow_new_entries=False,
            store=store,
            capital=200_000,
        )
        assert next_day["settled_count"] == 1
        assert next_day["open_count"] == 0
        assert next_day["settled"][0]["exit_reason"] == "AMBIGUOUS_BAR_STOP_FIRST"
        assert next_day["settled"][0]["production_evidence_eligible"] is False
        assert store.status()["closed_trades"] == 1
