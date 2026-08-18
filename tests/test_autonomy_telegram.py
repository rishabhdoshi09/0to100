from datetime import datetime
from types import SimpleNamespace

from research.autonomy.telegram_notifications import TelegramNotifier
from research.autonomy.live_feed import LiveFeedController


class FakeEngine:
    def __init__(self):
        self.messages = []
        self.configured = True

    def is_configured(self):
        return self.configured

    def send(self, message, reply_markup=None):
        self.messages.append((message, reply_markup))
        return True


class FakeLive:
    def __init__(self, price=101.0, fresh=True):
        self._price = price
        self._fresh = fresh

    def entry_allowed(self, symbol):
        return self._fresh

    def price(self, symbol):
        return self._price


def _payload():
    return {
        "universe_size": 2000,
        "summary": {"with_any_setup": 2, "ready_to_trade": 1, "near_breakout": 1,
                    "momentum": 2},
        "records": [
            {"symbol": "AAA", "status": "Ready to trade", "verdict": "BUY",
             "price": 100, "entry": 100.5, "stop": 96, "target": 110,
             "score": 82, "rsi": 61, "volume_ratio": 2.2,
             "signals": ["MOMENTUM"], "reasons": ["Volume confirmed"]},
            {"symbol": "BBB", "status": "Watch for breakout", "verdict": "WATCH",
             "price": 98.5, "entry": 100, "stop": 95, "target": 112,
             "score": 74, "rsi": 58, "volume_ratio": 1.7,
             "signals": ["PRE_BREAKOUT"], "reasons": ["Tight base"]},
        ],
    }


def test_scan_alerts_are_durable_and_deduped(tmp_path):
    engine = FakeEngine()
    now = lambda: datetime(2026, 7, 31, 10, 0)
    n = TelegramNotifier(tmp_path, engine_factory=lambda: engine, now_fn=now)
    first = n.notify_scan(_payload(), phase="intraday")
    assert first["setup"] == 1 and first["prebreakout"] == 1 and first["breakout"] == 0
    assert len(engine.messages) == 1
    assert "AAA" in engine.messages[0][0] and "BBB" in engine.messages[0][0]

    # A process restart must not repeat the same symbol/day alerts.
    n2 = TelegramNotifier(tmp_path, engine_factory=lambda: engine, now_fn=now)
    second = n2.notify_scan(_payload(), phase="intraday")
    assert second["setup"] == 0 and second["prebreakout"] == 0
    assert len(engine.messages) == 1


def test_live_breakout_requires_hold_and_sends_once(tmp_path):
    engine = FakeEngine()
    epoch = [1000.0]
    n = TelegramNotifier(
        tmp_path,
        engine_factory=lambda: engine,
        now_fn=lambda: datetime(2026, 7, 31, 10, 15),
        epoch_fn=lambda: epoch[0],
        breakout_confirmation_s=8,
        breakout_buffer_bps=10,
    )
    payload = _payload()
    live = FakeLive(price=100.25, fresh=True)

    assert n.observe_live_breakouts(payload, live)["confirmed"] == 0
    epoch[0] += 9
    assert n.observe_live_breakouts(payload, live)["confirmed"] == 1
    assert "BREAKOUT CONFIRMED" in engine.messages[-1][0]
    assert "BBB" in engine.messages[-1][0]
    assert "ne ₹" in engine.messages[-1][0]
    epoch[0] += 20
    assert n.observe_live_breakouts(payload, live)["confirmed"] == 0


def test_live_breakout_sends_plain_message_for_empty_scan_row(tmp_path):
    """Missing volume_ratio / funds must not mute a held breakout Telegram."""
    engine = FakeEngine()
    epoch = [1000.0]
    n = TelegramNotifier(
        tmp_path,
        engine_factory=lambda: engine,
        now_fn=lambda: datetime(2026, 7, 31, 10, 15),
        epoch_fn=lambda: epoch[0],
        breakout_confirmation_s=8,
        breakout_buffer_bps=10,
    )
    payload = {"records": [{
        "symbol": "BARE", "status": "Watch for breakout",
        "signals": ["PRE_BREAKOUT"], "entry": 100, "price": 99,
        "score": 70, "rsi": 55,
    }]}
    live = FakeLive(price=101.0, fresh=True)
    assert n.observe_live_breakouts(payload, live)["confirmed"] == 0
    epoch[0] += 9
    assert n.observe_live_breakouts(payload, live)["confirmed"] == 1
    msg = engine.messages[-1][0]
    assert "BREAKOUT CONFIRMED" in msg
    assert "BARE" in msg
    assert "fundamentals n/a" not in msg


def test_scan_sends_plain_breakout_when_price_already_through(tmp_path):
    engine = FakeEngine()
    n = TelegramNotifier(tmp_path, engine_factory=lambda: engine,
                         now_fn=lambda: datetime(2026, 7, 31, 10, 0))
    payload = {"records": [{
        "symbol": "KAYNES", "status": "Watch for breakout",
        "signals": ["PRE_BREAKOUT"], "verdict": "WATCH",
        "price": 4268.4, "entry": 4250, "stop": 4100, "target": 4550,
        "score": 74, "rsi": 58, "volume_ratio": 1.6,
        "reasons": ["Tight base"],
    }]}
    sent = n.notify_scan(payload, phase="intraday")
    assert sent["breakout"] == 1
    assert sent["prebreakout"] == 0
    assert "BREAKOUT CONFIRMED" in engine.messages[-1][0]
    assert "KAYNES" in engine.messages[-1][0]
    assert "ne ₹" in engine.messages[-1][0]
    assert n.notify_scan(payload, phase="intraday")["breakout"] == 0


def test_missing_live_tick_does_not_disarm_breakout(tmp_path):
    engine = FakeEngine()
    epoch = [1000.0]
    n = TelegramNotifier(
        tmp_path,
        engine_factory=lambda: engine,
        now_fn=lambda: datetime(2026, 7, 31, 10, 15),
        epoch_fn=lambda: epoch[0],
        breakout_confirmation_s=8,
        breakout_buffer_bps=10,
    )
    payload = _payload()
    live = FakeLive(price=100.25, fresh=True)
    assert n.observe_live_breakouts(payload, live)["confirmed"] == 0
    assert "BBB" in n.state.get("arms", {})

    class NoTick:
        def entry_allowed(self, symbol):
            return False

        def price(self, symbol):
            return None

    epoch[0] += 9
    assert n.observe_live_breakouts(payload, NoTick())["confirmed"] == 0
    assert "BBB" in n.state.get("arms", {})


def test_paper_open_and_close_alerts_include_ledger_truth(tmp_path):
    engine = FakeEngine()
    n = TelegramNotifier(tmp_path, engine_factory=lambda: engine,
                         now_fn=lambda: datetime(2026, 7, 31, 11, 0))
    open_pos = SimpleNamespace(strategy_id="STR-1", symbol="AAA", qty=10,
                               entry_price=100.0, stop_price=95.0, target_price=110.0)
    closed = SimpleNamespace(strategy_id="STR-2", symbol="BBB", exit_reason="TARGET",
                             exit_date="2026-07-31", entry_price=100.0, exit_price=110.0,
                             pnl=1000.0, realized_R=2.0)
    book = SimpleNamespace(open={("STR-1", "AAA"): open_pos}, closed=[closed])
    result = {"positions_opened": [("STR-1", "AAA")],
              "positions_closed": [("STR-2", "BBB", "TARGET")]}
    counts = n.notify_paper_cycle(result, book=book)
    assert counts == {"opened": 1, "closed": 1}
    joined = "\n".join(m[0] for m in engine.messages)
    assert "LIVE broker order: <b>NO</b>" in joined
    assert "₹+1,000.00" in joined and "+2.00R" in joined


def test_live_feed_controller_exposes_read_only_price():
    overlay = SimpleNamespace(
        price=lambda symbol: 123.45,
        entry_allowed=lambda symbol: True,
        health=lambda: {"connected": True, "subscriptions": 1, "reconnects": 0,
                        "symbols_ticking": 1, "rejected": {}, "last_connect_ts": 1.0},
        is_stale=lambda symbol: False,
        connected=True,
    )
    controller = LiveFeedController(overlay=overlay)
    assert controller.price("AAA") == 123.45


def test_market_scan_job_invokes_supervisor_owned_notifier():
    from research.autonomy import jobs as JOBS

    class Deps:
        def __init__(self):
            self.calls = []
        def active_snapshot_id(self): return "snap-1"
        def run_scan(self):
            return SimpleNamespace(ok=True, payload=_payload(), error_code="", error_message="")
        def now_ist(self): return datetime(2026, 7, 31, 10, 0)
        def holidays(self): return set()
        def notify_scan(self, payload, *, phase=""):
            self.calls.append((payload, phase))
            return {"setup": 1}

    deps = Deps()
    result = JOBS.run_market_scan(JOBS._Ctx(deps))
    assert result.status == "SUCCEEDED"
    assert deps.calls and deps.calls[0][1] == "intraday"
    assert result.metadata["telegram"] == {"setup": 1}


def test_supervisor_telegram_notifier_has_no_execution_or_broker_imports():
    from pathlib import Path
    source = Path("research/autonomy/telegram_notifications.py").read_text(encoding="utf-8")
    assert "execution.autopilot" not in source
    assert "execution.trade_executor" not in source
    assert "place_order" not in source
    assert "place_gtt" not in source


def test_long_term_alerts_are_ranked_and_deduped(tmp_path):
    engine = FakeEngine()
    now = lambda: datetime(2026, 7, 31, 18, 30)
    notifier = TelegramNotifier(tmp_path, engine_factory=lambda: engine, now_fn=now)
    payload = {"records": [{
        "symbol": "AAA", "classification": "QUALITY_COMPOUNDER",
        "combined_score": 82, "technical_score": 78, "fundamental_score": 85,
        "fundamental_coverage": 0.9, "timing": "TECHNICALLY_FAVORABLE",
        "quality_factors": ["ROCE 25%", "3y profit CAGR 20%"], "risk_flags": [],
    }]}
    assert notifier.notify_long_term(payload) == {"sent": 1}
    assert "current long-term shortlist" in engine.messages[-1][0]
    restarted = TelegramNotifier(tmp_path, engine_factory=lambda: engine, now_fn=now)
    assert restarted.notify_long_term(payload) == {"sent": 0}
