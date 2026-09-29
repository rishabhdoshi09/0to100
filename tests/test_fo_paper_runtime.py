import json
import sqlite3
from datetime import datetime, timedelta, timezone

from product.fo_paper import FoPaperPosition
from product.fo_paper_runtime import _iv_crush_state, run_fo_paper_cycle
from options.directional_selector import black_scholes
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


def _directional_with_token(token: int, *, holding_days: int = 2):
    payload = _directional()
    payload["candidates"][0]["selected_contract"]["instrument_token"] = token
    payload["candidates"][0]["setup"]["expected_move"]["holding_days"] = holding_days
    return payload


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



class _IntradayQuoteClient(_QuoteClient):
    def __init__(self, *, intraday_rows, entry_minute_rows=None, **kwargs):
        super().__init__(**kwargs)
        self.intraday_rows = list(intraday_rows)
        self.entry_minute_rows = (
            list(entry_minute_rows)
            if entry_minute_rows is not None
            else [{"open": 50.0, "high": 55.0, "low": 45.0, "close": 50.0}]
        )
        self.historical_calls = []
        self.entry_minute_calls = []

    def historical(self, token, frm, to, interval):
        start = datetime.fromisoformat(frm)
        end = datetime.fromisoformat(to)
        call = (token, frm, to, interval)
        if (end - start).total_seconds() <= 60:
            self.entry_minute_calls.append(call)
            return list(self.entry_minute_rows)
        self.historical_calls.append(call)
        return list(self.intraday_rows)


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


def test_store_reconstructs_same_day_premium_across_open_and_closed(tmp_path):
    session = "2026-09-29"
    with FoPaperStore(tmp_path / "fo.sqlite3") as store:
        store.replace_positions([{
            "trade_id": "OPEN1",
            "option_symbol": "AAACE",
            "underlying": "AAA",
            "entry_price": 50.0,
            "quantity": 100,
            "opened_at": f"{session}T10:00:00+05:30",
        }])
        store.append_trades([{
            # Legacy/split-state duplicate of the still-open trade must count once.
            "trade_id": "OPEN1",
            "option_symbol": "AAACE",
            "underlying": "AAA",
            "entry_price": 50.0,
            "quantity": 100,
            "opened_at": f"{session}T10:00:00+05:30",
            "settled_at": f"{session}T10:30:00+05:30",
        }, {
            "trade_id": "CLOSED1",
            "option_symbol": "BBBCE",
            "underlying": "BBB",
            "entry_price": 25.0,
            "quantity": 100,
            "opened_at": f"{session}T09:45:00+05:30",
            "settled_at": f"{session}T11:00:00+05:30",
        }, {
            "trade_id": "OLD1",
            "option_symbol": "OLDCE",
            "underlying": "OLD",
            "entry_price": 100.0,
            "quantity": 100,
            "opened_at": "2026-09-26T10:00:00+05:30",
            "settled_at": "2026-09-26T12:00:00+05:30",
        }])

        assert store.premium_deployed_on_session(session) == 7_500.0


def test_runtime_restart_cannot_reset_same_day_premium_budget(tmp_path):
    now_ist = datetime(2026, 9, 29, 12, 0, tzinfo=IST)
    with FoPaperStore(tmp_path / "fo.sqlite3") as store:
        store.append_trades([{
            "trade_id": "EARLIER",
            "option_symbol": "EARLIERCE",
            "underlying": "EARLIER",
            "entry_price": 50.0,
            "quantity": 180,
            "opened_at": "2026-09-29T10:00:00+05:30",
            "settled_at": "2026-09-29T11:00:00+05:30",
            "net_pnl": 0.0,
        }])

        result = run_fo_paper_cycle(
            _directional(),
            client=_QuoteClient(),
            now_ist=now_ist,
            allow_new_entries=True,
            store=store,
            capital=100_000,
        )

        assert result["premium_deployed_today"] == 9_000.0
        assert result["daily_premium_budget"] == 10_000.0
        assert result["daily_premium_remaining"] == 1_000.0
        assert result["opened_count"] == 0
        assert result["skipped"][-1]["reason"] == "DAILY_PREMIUM_BUDGET_EXHAUSTED"


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


def test_paper_store_migrates_v1_net_pnl_and_restores_equity(tmp_path):
    path = tmp_path / "legacy-fo.sqlite3"
    conn = sqlite3.connect(path)
    try:
        conn.execute(
            """
            CREATE TABLE fo_open_positions (
                trade_id TEXT PRIMARY KEY,
                option_symbol TEXT NOT NULL UNIQUE,
                underlying TEXT NOT NULL,
                payload_json TEXT NOT NULL,
                updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE fo_closed_trades (
                trade_id TEXT PRIMARY KEY,
                option_symbol TEXT NOT NULL,
                underlying TEXT NOT NULL,
                context_key TEXT NOT NULL DEFAULT '',
                evidence_lane TEXT NOT NULL DEFAULT 'FORWARD_PAPER',
                settled_at TEXT NOT NULL DEFAULT '',
                production_evidence_eligible INTEGER NOT NULL DEFAULT 0,
                payload_json TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE fo_paper_meta (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            )
            """
        )
        payload = {
            "trade_id": "LEGACY1",
            "option_symbol": "LEGACYCE",
            "underlying": "LEGACY",
            "net_pnl": -750.0,
            "settled": True,
            "evidence_lane": "FORWARD_PAPER",
            "production_evidence_eligible": False,
        }
        conn.execute(
            """
            INSERT INTO fo_closed_trades(
                trade_id, option_symbol, underlying, context_key,
                evidence_lane, settled_at, production_evidence_eligible,
                payload_json
            ) VALUES(?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "LEGACY1", "LEGACYCE", "LEGACY", "CTX", "FORWARD_PAPER",
                "2026-09-25", 0, json.dumps(payload),
            ),
        )
        conn.commit()
    finally:
        conn.close()

    with FoPaperStore(path) as store:
        assert store.status()["schema_version"] == 2
        assert store.realized_pnl() == -750.0
        result = run_fo_paper_cycle(
            {"available": True, "candidates": []},
            client=_QuoteClient(),
            now_ist=datetime(2026, 9, 26, 10, 0, tzinfo=IST),
            allow_new_entries=False,
            store=store,
            capital=200_000,
        )

    assert result["realized_pnl"] == -750.0
    assert result["equity_for_sizing"] == 199_250.0


def test_new_closed_trades_persist_realized_pnl_for_next_process(tmp_path):
    path = tmp_path / "fo.sqlite3"
    opened_at = datetime(2026, 9, 25, 10, 15, tzinfo=IST)

    with FoPaperStore(path) as store:
        opened = run_fo_paper_cycle(
            _directional(),
            client=_QuoteClient(last=50, high=50, low=50, bid=49),
            now_ist=opened_at,
            allow_new_entries=True,
            store=store,
            capital=200_000,
        )
        assert opened["opened_count"] == 1

    with FoPaperStore(path) as store:
        settled = run_fo_paper_cycle(
            {"available": True, "candidates": []},
            client=_QuoteClient(last=72, high=75, low=49, bid=70),
            now_ist=opened_at + timedelta(days=1),
            allow_new_entries=False,
            store=store,
            capital=200_000,
        )
        assert settled["settled_count"] == 1
        realized = settled["realized_pnl"]
        assert realized > 0

    with FoPaperStore(path) as store:
        restarted = run_fo_paper_cycle(
            {"available": True, "candidates": []},
            client=_QuoteClient(),
            now_ist=opened_at + timedelta(days=2),
            allow_new_entries=False,
            store=store,
            capital=200_000,
        )

    assert restarted["realized_pnl"] == realized
    assert restarted["equity_for_sizing"] == 200_000 + realized


def test_same_day_paper_mark_uses_post_entry_intraday_range(tmp_path):
    path = tmp_path / "fo.sqlite3"
    opened_at = datetime(2026, 9, 26, 10, 15, 12, tzinfo=IST)

    with FoPaperStore(path) as store:
        opened = run_fo_paper_cycle(
            _directional_with_token(777),
            client=_QuoteClient(last=50, high=80, low=30, bid=49),
            now_ist=opened_at,
            allow_new_entries=True,
            store=store,
            capital=200_000,
        )
        assert opened["opened_count"] == 1
        assert opened["open_positions"][0]["instrument_token"] == 777

    # Daily high/low are deliberately contaminated by pre-entry prices. The
    # post-entry minute range contains a real stop touch and must be used.
    client = _IntradayQuoteClient(
        last=52,
        high=80,
        low=30,
        bid=51,
        intraday_rows=[
            {"open": 51.0, "high": 55.0, "low": 45.0, "close": 53.0},
            {"open": 53.0, "high": 54.0, "low": 39.0, "close": 52.0},
        ],
    )
    with FoPaperStore(path) as store:
        marked = run_fo_paper_cycle(
            {"available": True, "candidates": []},
            client=client,
            now_ist=opened_at.replace(hour=14, minute=0, second=0),
            allow_new_entries=False,
            store=store,
            capital=200_000,
        )

    assert marked["settled_count"] == 1
    assert marked["settled"][0]["exit_reason"] == "STOP"
    assert marked["same_day_intraday_marks_used"] == 1
    assert marked["same_day_intraday_marks_fallback"] == 0
    assert marked["same_day_intraday_bars_replayed"] == 2
    assert client.historical_calls == [
        (777, "2026-09-26 10:16:00", "2026-09-26 14:00:00", "minute")
    ]


def test_same_day_paper_mark_falls_back_to_ltp_when_intraday_bars_missing(tmp_path):
    path = tmp_path / "fo.sqlite3"
    opened_at = datetime(2026, 9, 26, 10, 15, 12, tzinfo=IST)

    with FoPaperStore(path) as store:
        run_fo_paper_cycle(
            _directional_with_token(778),
            client=_QuoteClient(last=50, high=80, low=30, bid=49),
            now_ist=opened_at,
            allow_new_entries=True,
            store=store,
            capital=200_000,
        )

    client = _IntradayQuoteClient(
        last=52,
        high=80,
        low=30,
        bid=51,
        intraday_rows=[],
    )
    with FoPaperStore(path) as store:
        marked = run_fo_paper_cycle(
            {"available": True, "candidates": []},
            client=client,
            now_ist=opened_at.replace(hour=14, minute=0, second=0),
            allow_new_entries=False,
            store=store,
            capital=200_000,
        )

    # Missing minute history must not make the pre-entry full-day low look like
    # a stop. The existing LTP-only fail-safe remains truthful.
    assert marked["settled_count"] == 0
    assert marked["open_count"] == 1
    assert marked["same_day_intraday_marks_used"] == 0
    assert marked["same_day_intraday_marks_fallback"] == 1


def test_ordered_intraday_replay_respects_target_before_later_stop(tmp_path):
    path = tmp_path / "fo.sqlite3"
    opened_at = datetime(2026, 9, 26, 10, 15, 12, tzinfo=IST)

    with FoPaperStore(path) as store:
        opened = run_fo_paper_cycle(
            _directional_with_token(779),
            client=_QuoteClient(last=50, high=80, low=30, bid=49),
            now_ist=opened_at,
            allow_new_entries=True,
            store=store,
            capital=200_000,
        )
        assert opened["opened_count"] == 1

    # Deliberately return provider rows out of order. Chronological replay must
    # sort them: target is hit at 10:17, while the stop only appears at 10:18.
    # An aggregated high/low range would incorrectly collapse this to the
    # conservative ambiguous STOP-first rule.
    client = _IntradayQuoteClient(
        last=52,
        high=90,
        low=20,
        bid=51,
        intraday_rows=[
            {
                "date": "2026-09-26T10:18:00+05:30",
                "open": 69.0, "high": 69.5, "low": 35.0, "close": 38.0,
            },
            {
                "date": "2026-09-26T10:17:00+05:30",
                "open": 52.0, "high": 72.0, "low": 48.0, "close": 71.0,
            },
        ],
    )
    with FoPaperStore(path) as store:
        marked = run_fo_paper_cycle(
            {"available": True, "candidates": []},
            client=client,
            now_ist=opened_at.replace(hour=14, minute=0, second=0),
            allow_new_entries=False,
            store=store,
            capital=200_000,
        )

    assert marked["settled_count"] == 1
    assert marked["open_count"] == 0
    assert marked["settled"][0]["exit_reason"] == "TARGET"
    assert marked["same_day_intraday_marks_used"] == 1
    # Replay stops as soon as the first chronological exit occurs.
    assert marked["same_day_intraday_bars_replayed"] == 1



def test_overnight_intraday_replay_respects_target_before_later_stop(tmp_path):
    path = tmp_path / "fo.sqlite3"
    opened_at = datetime(2026, 9, 25, 14, 0, tzinfo=IST)

    with FoPaperStore(path) as store:
        opened = run_fo_paper_cycle(
            _directional_with_token(780, holding_days=2),
            client=_QuoteClient(last=50, high=50, low=50, bid=49),
            now_ist=opened_at,
            allow_new_entries=True,
            store=store,
            capital=200_000,
        )
        assert opened["opened_count"] == 1

    # The position existed from the next session's open. Return bars out of
    # order: chronological replay must see the 09:17 target before the 09:18 stop.
    client = _IntradayQuoteClient(
        last=52,
        high=90,
        low=20,
        bid=51,
        intraday_rows=[
            {
                "date": "2026-09-26T09:18:00+05:30",
                "open": 69.0, "high": 69.5, "low": 35.0, "close": 38.0,
            },
            {
                "date": "2026-09-26T09:17:00+05:30",
                "open": 52.0, "high": 72.0, "low": 48.0, "close": 71.0,
            },
        ],
    )
    now = datetime(2026, 9, 26, 10, 0, tzinfo=IST)
    with FoPaperStore(path) as store:
        marked = run_fo_paper_cycle(
            {"available": True, "candidates": []},
            client=client,
            now_ist=now,
            allow_new_entries=False,
            store=store,
            capital=200_000,
        )

    assert marked["settled_count"] == 1
    assert marked["settled"][0]["exit_reason"] == "TARGET"
    assert marked["overnight_intraday_marks_used"] == 1
    assert marked["overnight_intraday_marks_fallback"] == 0
    assert marked["overnight_intraday_bars_replayed"] == 1
    assert client.historical_calls == [
        (780, "2026-09-26 09:15:00", "2026-09-26 10:00:00", "minute")
    ]


def test_overnight_replay_does_not_trigger_max_hold_before_current_mark(tmp_path):
    path = tmp_path / "fo.sqlite3"
    opened_at = datetime(2026, 9, 25, 14, 0, tzinfo=IST)

    with FoPaperStore(path) as store:
        opened = run_fo_paper_cycle(
            _directional_with_token(781, holding_days=1),
            client=_QuoteClient(last=50, high=50, low=50, bid=49),
            now_ist=opened_at,
            allow_new_entries=True,
            store=store,
            capital=200_000,
        )
        assert opened["opened_count"] == 1

    client = _IntradayQuoteClient(
        last=52,
        high=55,
        low=45,
        bid=51,
        intraday_rows=[
            {
                "date": "2026-09-26T09:16:00+05:30",
                "open": 51.0, "high": 54.0, "low": 49.0, "close": 53.0,
            },
            {
                "date": "2026-09-26T09:17:00+05:30",
                "open": 53.0, "high": 55.0, "low": 50.0, "close": 54.0,
            },
        ],
    )
    now = datetime(2026, 9, 26, 10, 0, tzinfo=IST)
    with FoPaperStore(path) as store:
        marked = run_fo_paper_cycle(
            {"available": True, "candidates": []},
            client=client,
            now_ist=now,
            allow_new_entries=False,
            store=store,
            capital=200_000,
        )

    # Historical replay reconstructs path only. Holding-session age advances at
    # the current supervision mark, where MAX_HOLD exits using the current bid.
    assert marked["settled_count"] == 1
    assert marked["settled"][0]["exit_reason"] == "MAX_HOLD"
    assert marked["settled"][0]["settled_at"] == now.isoformat()
    assert marked["settled"][0]["exit_price"] == 51.0
    assert marked["overnight_intraday_marks_used"] == 1
    assert marked["overnight_intraday_bars_replayed"] == 2



def test_same_day_first_complete_minute_preserves_gap_through_stop(tmp_path):
    path = tmp_path / "fo.sqlite3"
    opened_at = datetime(2026, 9, 26, 10, 15, 12, tzinfo=IST)

    with FoPaperStore(path) as store:
        opened = run_fo_paper_cycle(
            _directional_with_token(782),
            client=_QuoteClient(last=50, high=50, low=50, bid=49),
            now_ist=opened_at,
            allow_new_entries=True,
            store=store,
            capital=200_000,
        )
        assert opened["opened_count"] == 1

    # The partial 10:15 minute is excluded. The 10:16 open is fully post-entry
    # and below the stop, so the first trustworthy observation is a gap-through
    # stop rather than a fabricated fill at the ₹40 trigger.
    client = _IntradayQuoteClient(
        last=36,
        high=38,
        low=34,
        bid=35,
        intraday_rows=[
            {
                "date": "2026-09-26T10:16:00+05:30",
                "open": 35.0, "high": 38.0, "low": 34.0, "close": 36.0,
            },
        ],
    )
    with FoPaperStore(path) as store:
        marked = run_fo_paper_cycle(
            {"available": True, "candidates": []},
            client=client,
            now_ist=opened_at.replace(hour=10, minute=20, second=0),
            allow_new_entries=False,
            store=store,
            capital=200_000,
        )

    assert marked["settled_count"] == 1
    assert marked["settled"][0]["exit_reason"] == "GAP_STOP"
    assert marked["settled"][0]["exit_price"] == 34.9825
    assert marked["same_day_intraday_bars_replayed"] == 1
    assert client.historical_calls == [
        (782, "2026-09-26 10:16:00", "2026-09-26 10:20:00", "minute")
    ]


def test_same_day_first_complete_minute_preserves_gap_through_target(tmp_path):
    path = tmp_path / "fo.sqlite3"
    opened_at = datetime(2026, 9, 26, 10, 15, 12, tzinfo=IST)

    with FoPaperStore(path) as store:
        opened = run_fo_paper_cycle(
            _directional_with_token(783),
            client=_QuoteClient(last=50, high=50, low=50, bid=49),
            now_ist=opened_at,
            allow_new_entries=True,
            store=store,
            capital=200_000,
        )
        assert opened["opened_count"] == 1

    client = _IntradayQuoteClient(
        last=76,
        high=78,
        low=74,
        bid=75,
        intraday_rows=[
            {
                "date": "2026-09-26T10:16:00+05:30",
                "open": 75.0, "high": 78.0, "low": 74.0, "close": 76.0,
            },
        ],
    )
    with FoPaperStore(path) as store:
        marked = run_fo_paper_cycle(
            {"available": True, "candidates": []},
            client=client,
            now_ist=opened_at.replace(hour=10, minute=20, second=0),
            allow_new_entries=False,
            store=store,
            capital=200_000,
        )

    assert marked["settled_count"] == 1
    assert marked["settled"][0]["exit_reason"] == "GAP_TARGET"
    assert marked["settled"][0]["exit_price"] == 74.9625
    assert marked["same_day_intraday_bars_replayed"] == 1



def test_entry_minute_clear_path_allows_fully_costed_forward_evidence(tmp_path):
    path = tmp_path / "fo.sqlite3"
    opened_at = datetime(2026, 9, 26, 10, 15, 12, tzinfo=IST)

    with FoPaperStore(path) as store:
        opened = run_fo_paper_cycle(
            _directional_with_token(784),
            client=_QuoteClient(last=50, high=50, low=50, bid=49),
            now_ist=opened_at,
            allow_new_entries=True,
            store=store,
            capital=200_000,
        )
        assert opened["opened_count"] == 1

    client = _IntradayQuoteClient(
        last=72,
        high=75,
        low=49,
        bid=70,
        entry_minute_rows=[
            {
                "date": "2026-09-26T10:15:00+05:30",
                "open": 49.0, "high": 55.0, "low": 45.0, "close": 51.0,
            },
        ],
        intraday_rows=[
            {
                "date": "2026-09-26T10:16:00+05:30",
                "open": 52.0, "high": 72.0, "low": 50.0, "close": 71.0,
            },
        ],
    )
    with FoPaperStore(path) as store:
        marked = run_fo_paper_cycle(
            {"available": True, "candidates": []},
            client=client,
            now_ist=opened_at.replace(hour=10, minute=20, second=0),
            allow_new_entries=False,
            store=store,
            capital=200_000,
            cost_model=lambda entry, exit, qty: 25.0,
            cost_model_name="TEST_COSTS",
        )

    assert marked["settled_count"] == 1
    trade = marked["settled"][0]
    assert trade["entry_minute_status"] == "CLEAR_NO_TRIGGER"
    assert trade["path_observation_complete"] is True
    assert trade["production_evidence_eligible"] is True
    assert marked["entry_minute_clear_count"] == 1
    assert client.entry_minute_calls == [
        (784, "2026-09-26 10:15:00", "2026-09-26 10:15:59", "minute")
    ]


def test_entry_minute_boundary_touch_holds_forward_evidence_even_when_costed(tmp_path):
    path = tmp_path / "fo.sqlite3"
    opened_at = datetime(2026, 9, 26, 10, 15, 12, tzinfo=IST)

    with FoPaperStore(path) as store:
        opened = run_fo_paper_cycle(
            _directional_with_token(785),
            client=_QuoteClient(last=50, high=50, low=50, bid=49),
            now_ist=opened_at,
            allow_new_entries=True,
            store=store,
            capital=200_000,
        )
        assert opened["opened_count"] == 1

    client = _IntradayQuoteClient(
        last=72,
        high=75,
        low=49,
        bid=70,
        # The minute contains time before and after the 10:15:12 paper fill.
        # A stop touch exists somewhere, so timing is unknowable.
        entry_minute_rows=[
            {
                "date": "2026-09-26T10:15:00+05:30",
                "open": 50.0, "high": 55.0, "low": 35.0, "close": 51.0,
            },
        ],
        intraday_rows=[
            {
                "date": "2026-09-26T10:16:00+05:30",
                "open": 52.0, "high": 72.0, "low": 50.0, "close": 71.0,
            },
        ],
    )
    with FoPaperStore(path) as store:
        marked = run_fo_paper_cycle(
            {"available": True, "candidates": []},
            client=client,
            now_ist=opened_at.replace(hour=10, minute=20, second=0),
            allow_new_entries=False,
            store=store,
            capital=200_000,
            cost_model=lambda entry, exit, qty: 25.0,
            cost_model_name="TEST_COSTS",
        )

    assert marked["settled_count"] == 1
    trade = marked["settled"][0]
    assert trade["exit_reason"] == "TARGET"
    assert trade["entry_minute_status"] == "AMBIGUOUS_BOUNDARY_TOUCH"
    assert trade["path_observation_complete"] is False
    assert trade["production_evidence_eligible"] is False
    assert trade["evidence_exclusion_reason"] == "ENTRY_MINUTE_AMBIGUOUS_BOUNDARY_TOUCH"
    assert marked["entry_minute_ambiguous_count"] == 1



def test_current_partial_exit_interval_is_held_out_of_probability_evidence(tmp_path):
    path = tmp_path / "fo.sqlite3"
    opened_at = datetime(2026, 9, 26, 10, 15, 12, tzinfo=IST)

    with FoPaperStore(path) as store:
        opened = run_fo_paper_cycle(
            _directional_with_token(786),
            client=_QuoteClient(last=50, high=50, low=50, bid=49),
            now_ist=opened_at,
            allow_new_entries=True,
            store=store,
            capital=200_000,
        )
        assert opened["opened_count"] == 1

    client = _IntradayQuoteClient(
        last=75,
        high=75,
        low=45,
        bid=74,
        entry_minute_rows=[
            {
                "date": "2026-09-26T10:15:00+05:30",
                "open": 49.0, "high": 55.0, "low": 45.0, "close": 51.0,
            },
        ],
        intraday_rows=[
            {
                "date": "2026-09-26T10:16:00+05:30",
                "open": 52.0, "high": 55.0, "low": 48.0, "close": 53.0,
            },
        ],
    )
    now = opened_at.replace(hour=10, minute=20, second=15)
    with FoPaperStore(path) as store:
        marked = run_fo_paper_cycle(
            {"available": True, "candidates": []},
            client=client,
            now_ist=now,
            allow_new_entries=False,
            store=store,
            capital=200_000,
            cost_model=lambda entry, exit, qty: 25.0,
            cost_model_name="TEST_COSTS",
        )

    assert marked["settled_count"] == 1
    trade = marked["settled"][0]
    assert trade["exit_reason"] == "GAP_TARGET"
    assert trade["entry_minute_status"] == "CLEAR_NO_TRIGGER"
    assert trade["exit_observation_complete"] is False
    assert trade["path_observation_complete"] is False
    assert trade["path_observation_reason"] == "EXIT_PARTIAL_INTERVAL_UNOBSERVED"
    assert trade["production_evidence_eligible"] is False
    assert trade["evidence_exclusion_reason"] == "EXIT_PARTIAL_INTERVAL_UNOBSERVED"
    assert marked["partial_exit_interval_holdout_count"] == 1


def test_iv_crush_runtime_requires_fresh_coherent_observed_quotes():
    now_ist = datetime(2026, 9, 22, 12, 0, tzinfo=IST)
    current_price = black_scholes(
        spot=3050.0,
        strike=3050.0,
        dte=9,
        iv=0.20,
        option_type="CE",
    )["price"]
    pos = FoPaperPosition(
        trade_id="IV1",
        underlying="RELIANCE",
        option_symbol="RELIANCECE",
        option_type="CE",
        context_key="CTX",
        entry_price=100.0,
        stop_price=60.0,
        target_price=140.0,
        lot_size=25,
        lots=1,
        quantity=25,
        opened_at="2026-09-22T10:00:00+05:30",
        max_holding_sessions=2,
        risk_amount=1000.0,
        setup_score=85.0,
        option_score=80.0,
        strike=3050.0,
        expiry="2026-10-01",
        entry_iv_pct=40.0,
        entry_underlying_spot=3050.0,
    )
    option_quote = {
        "timestamp": now_ist.isoformat(),
        "last_price": current_price,
        "depth": {
            "buy": [{"price": current_price - 0.05}],
            "sell": [{"price": current_price + 0.05}],
        },
    }
    underlying_quote = {
        "timestamp": now_ist.isoformat(),
        "last_price": 3050.0,
    }

    triggered, reason, current_iv = _iv_crush_state(
        pos=pos,
        option_quote=option_quote,
        underlying_quote=underlying_quote,
        now_ist=now_ist,
    )
    assert triggered is True
    assert reason == "IV_CRUSH_TRIGGERED"
    assert current_iv is not None and 19.0 <= current_iv <= 21.0

    stale = dict(option_quote)
    stale["timestamp"] = "2026-09-22T11:00:00+05:30"
    triggered, reason, current_iv = _iv_crush_state(
        pos=pos,
        option_quote=stale,
        underlying_quote=underlying_quote,
        now_ist=now_ist,
    )
    assert triggered is False
    assert reason == "IV_CRUSH_QUOTE_UNTRUSTED"
    assert current_iv is None


def test_runtime_blocks_new_intraday_entry_at_eod_cutoff(tmp_path):
    payload = _directional()
    expected = payload["candidates"][0]["setup"]["expected_move"]
    expected.update({"holding_days": 0, "horizon": "INTRADAY", "exit_policy": "EOD"})
    with FoPaperStore(tmp_path / "fo.sqlite3") as store:
        result = run_fo_paper_cycle(
            payload,
            client=_QuoteClient(),
            now_ist=datetime(2026, 9, 29, 15, 35, tzinfo=IST),
            allow_new_entries=True,
            store=store,
            capital=200_000,
        )
    assert result["opened_count"] == 0
    assert result["skipped"][-1]["reason"] == "INTRADAY_EOD_CUTOFF_REACHED"
    assert result["eod_exit_due"] is True
    assert result["eod_exit_cutoff_ist"] == "15:35"


def test_runtime_blocks_all_new_option_entries_after_market_close(tmp_path):
    with FoPaperStore(tmp_path / "fo.sqlite3") as store:
        result = run_fo_paper_cycle(
            _directional(),
            client=_QuoteClient(),
            now_ist=datetime(2026, 9, 29, 15, 40, tzinfo=IST),
            allow_new_entries=True,
            store=store,
            capital=200_000,
        )
    assert result["opened_count"] == 0
    assert result["skipped"][-1]["reason"] == "FNO_MARKET_CLOSED"
    assert result["fno_market_close_ist"] == "15:40"
