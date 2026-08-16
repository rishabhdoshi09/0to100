"""Focused regression tests for autonomy-closure wiring."""
from __future__ import annotations

import dataclasses
from pathlib import Path

from research.autonomy import auth as AUTH
from research.autonomy import challenge as CH
from research.autonomy import controls as CTRL
from research.autonomy import hypotheses as HYP
from research.autonomy import research_loop as RL
from research.autonomy.dialogue import DialogueLog, EXPERIMENT_REGISTRATION, EXPERIMENT_RESULT
from research.autonomy.live_feed import LiveFeedController
from research.auto_research.paper_book import PaperBook
from research.intelligence import strategy_runtime as RT
from research.intelligence.event_store import EventStore
from research.intelligence.registry import StrategyRegistry
from research.intelligence.runtime import run_intelligence_cycle
from research.intelligence.runtime.cycle_context import CycleContext
from research.intelligence.runtime.runtime_state import RuntimeState
from research.strategy_studio import discovery as DISC


class TokenException(Exception):
    pass


class _Client:
    def __init__(self, profile=None, error=None):
        self._profile = profile
        self._error = error

    def profile(self):
        if self._error:
            raise self._error
        return self._profile


def test_auth_health_distinguishes_valid_expired_and_outage():
    valid = AUTH.probe_auth(client_factory=lambda: _Client({"user_id": "AB1", "broker": "ZERODHA"}))
    expired = AUTH.probe_auth(client_factory=lambda: _Client(error=TokenException("invalid token")))
    outage = AUTH.probe_auth(client_factory=lambda: _Client(error=TimeoutError("timed out")))
    assert valid.status == AUTH.SESSION_VALID and valid.user_id == "AB1"
    assert expired.status == AUTH.SESSION_EXPIRED
    assert outage.status == AUTH.PROVIDER_UNAVAILABLE


def test_auth_health_redacts_credential_text():
    health = AUTH.probe_auth(client_factory=lambda: _Client(
        error=RuntimeError("access_token=super-secret authorization=Bearer")))
    assert "super-secret" not in health.reason
    assert health.reason == "RuntimeError"


def test_control_queue_is_durable_and_auditable(tmp_path):
    db = tmp_path / "controls.db"
    first = CTRL.ControlStore(db)
    control = first.request(CTRL.PAUSE_NEW_PAPER_ENTRIES, reason="owner pause")
    first.close()
    second = CTRL.ControlStore(db)
    pending = second.pending()
    assert [c.control_id for c in pending] == [control.control_id]
    second.finish(control.control_id, result="applied")
    recent = second.recent()
    assert recent[0].status == CTRL.PROCESSED and recent[0].reason == "owner pause"
    second.close()


class _Overlay:
    def __init__(self):
        self.connected = False
        self.subscribed = set()
        self.connects = 0
        self.fresh = set()

    def connect(self):
        self.connects += 1
        self.connected = True

    def subscribe(self, symbols):
        self.subscribed |= set(symbols)

    def entry_allowed(self, symbol):
        return symbol in self.fresh

    def is_stale(self, symbol):
        return symbol not in self.fresh

    def health(self):
        return {"connected": self.connected, "subscriptions": len(self.subscribed),
                "reconnects": 0, "symbols_ticking": len(self.fresh), "rejected": {},
                "last_connect_ts": 1.0}


def test_live_feed_controller_reconnects_injected_overlay_and_reports_freshness(tmp_path):
    overlay = _Overlay()
    ctl = LiveFeedController(tmp_path / "live.json", overlay=overlay)
    ctl.start({"AAA"})
    assert overlay.connects == 1 and overlay.connected and "AAA" in overlay.subscribed
    assert not ctl.entry_allowed("AAA")
    overlay.fresh.add("AAA")
    assert ctl.entry_allowed("AAA") and ctl.fresh_symbols() == frozenset({"AAA"})
    ctl.stop()
    ctl.start({"AAA"})
    assert overlay.connects == 2


def _breakout_spec():
    spec = next(s for s in DISC.generate(DISC.DiscoveryBudget()) if s.family == "breakout")
    return dataclasses.replace(spec, strategy_id="LIVE-BO")


def _breakout_history(as_of="d25"):
    bars = [RT.Bar(f"d{i}", 100, 101, 99, 100) for i in range(25)]
    bars.append(RT.Bar(as_of, 100, 112, 100, 111))
    return bars


def test_live_required_strategy_blocks_stale_symbol_but_eod_strategy_does_not():
    base = _breakout_spec()
    live = dataclasses.replace(base, required_data=tuple(base.required_data) + ("live_ticks",))
    book, store, state = PaperBook(), EventStore(), RuntimeState()
    ctx = CycleContext(as_of_date="d25", mode="PAPER_AUTO", data_ok=True,
                       data_snapshot_id="snap", strategies=[live],
                       data={live.strategy_id: {"AAA": _breakout_history()}},
                       fresh_live_symbols=frozenset())
    res = run_intelligence_cycle(ctx, store=store, book=book, runtime_state=state,
                                 backtest_R={live.strategy_id: 0.3},
                                 backtest_trades={live.strategy_id: 40})
    assert not book.open and (live.strategy_id, "LIVE_PRICE_STALE") in res.intents_blocked

    eod = base
    book2, store2, state2 = PaperBook(), EventStore(), RuntimeState()
    ctx2 = CycleContext(as_of_date="d25", mode="PAPER_AUTO", data_ok=True,
                        data_snapshot_id="snap", strategies=[eod],
                        data={eod.strategy_id: {"AAA": _breakout_history()}},
                        fresh_live_symbols=frozenset())
    res2 = run_intelligence_cycle(ctx2, store=store2, book=book2, runtime_state=state2,
                                  backtest_R={eod.strategy_id: 0.3},
                                  backtest_trades={eod.strategy_id: 40})
    assert res2.positions_opened and len(book2.open) == 1


def test_failed_hypothesis_persists_across_memory_instances(tmp_path, monkeypatch):
    import research.scientific_memory as SM
    monkeypatch.setattr(SM, "_DB_PATH", tmp_path / "memory.db")
    parent = _breakout_spec()
    semantic = HYP.hypothesis_hash(parent, {"max_holding_days": 30})
    HYP.ResearchMemory(backend=SM).record_dead(semantic, "failed after costs")
    assert HYP.ResearchMemory(backend=SM).is_known(semantic)


class _Brain:
    def __init__(self, spec, tmp_path):
        self.strategy_registry = StrategyRegistry().build([spec])
        self.runtime_state = RuntimeState(tmp_path / "runtime.json")
        self.intel_book = PaperBook()
        self.evaluate_fn = None


def _passing_evidence():
    return {
        "invalid_data": False, "is_synthetic": False,
        "benchmark_available": True, "n_trades": 120,
        "net_expectancy_R": 0.25, "deflated_sharpe": 0.75,
        "reality_check_p": 0.02, "walk_forward_ok": True,
        "fdr_significant": True, "max_drawdown_pct": 12.0,
        "turnover": 1.0, "max_symbol_share": 0.2,
        "num_trials": 3, "parameter_count": 4, "verdict": "PROMOTE",
    }


def test_research_pipeline_preregisters_then_promotes_successor(tmp_path):
    parent = _breakout_spec()
    brain = _Brain(parent, tmp_path)
    gap = HYP.EvidenceGap("negative_forward_expectancy", parent.strategy_id,
                          "forward expectancy negative", 0.8, 0.9, True, 0.2)
    log = DialogueLog(tmp_path / "dialogue.jsonl")
    result = RL.execute_pipeline(brain, gap=gap, parent=parent, session_date="2026-07-31",
                                 dialogue=log, memory=HYP.ResearchMemory(backend=None),
                                 experiment_runner=lambda child: _passing_evidence())
    assert result["decision"] == CH.PAPER_NOMINATED
    current = brain.strategy_registry.by_id[parent.strategy_id].spec
    assert current.version == parent.version + 1
    assert parent.version == 1 and current.config_hash() != parent.config_hash()
    assert brain.runtime_state.get(parent.strategy_id).lifecycle == "APPROVED_FOR_PAPER"
    types = [r["record_type"] for r in log.all()]
    assert types.index(EXPERIMENT_REGISTRATION) < types.index(EXPERIMENT_RESULT)


def test_missing_institutional_evidence_cannot_promote(tmp_path):
    parent = _breakout_spec()
    brain = _Brain(parent, tmp_path)
    gap = HYP.EvidenceGap("negative_forward_expectancy", parent.strategy_id,
                          "forward expectancy negative", 0.8, 0.9, True, 0.2)
    weak = dict(_passing_evidence())
    weak.pop("walk_forward_ok")
    result = RL.execute_pipeline(brain, gap=gap, parent=parent, session_date="2026-07-31",
                                 memory=HYP.ResearchMemory(backend=None),
                                 experiment_runner=lambda child: weak)
    assert result["decision"] == CH.RETEST_WITH_MORE_DATA
    assert brain.strategy_registry.by_id[parent.strategy_id].spec.version == parent.version


def test_eod_schedule_waits_for_publish_window():
    from datetime import datetime
    from research.autonomy import schedules as SCH
    assert SCH.scan_slot(datetime(2026, 7, 31, 16, 0)) is None
    assert SCH.scan_slot(datetime(2026, 7, 31, 18, 5)) == "eod"


def test_eod_refresh_requires_completed_session():
    from datetime import datetime
    from research.autonomy import jobs as JOBS
    from research.autonomy import job_store as JS

    class Report:
        active_pointer = "snap-old"
        snapshot_id = "snap-old"
        blocker = ""
        quality = {"date_range": ("2026-07-29", "2026-07-30")}
        def status(self, name): return "PASS"

    class Deps:
        def session_valid(self): return True
        def activate(self): return Report()
        def active_snapshot_info(self): return {"latest_date": "2026-07-30"}

    ctx = JOBS._Ctx(Deps())
    ctx.required_session_date = "2026-07-31"
    result = JOBS.run_data_refresh(ctx)
    assert result.status == JS.RETRYABLE_FAILED
    assert result.error_code == "EOD_DATA_PENDING"


def test_outcome_resolution_blocks_old_snapshot():
    from datetime import datetime
    from research.autonomy import jobs as JOBS
    from research.autonomy import job_store as JS

    class Deps:
        def now_ist(self): return datetime(2026, 7, 31, 18, 10)
        def holidays(self): return set()
        def active_snapshot_id(self): return "snap-old"
        def active_snapshot_info(self): return {"latest_date": "2026-07-30"}

    result = JOBS.run_outcome_resolution(JOBS._Ctx(Deps()))
    assert result.status == JS.BLOCKED
    assert result.blocked_on == "EOD_DATA_READY:2026-07-31"


def test_required_completed_session_skips_weekend():
    from datetime import datetime
    from research.autonomy import schedules as SCH

    sunday = datetime(2026, 8, 16, 10, 0)
    friday_eod = datetime(2026, 8, 14, 18, 10)
    monday_morning = datetime(2026, 8, 17, 10, 0)
    assert SCH.required_completed_session(sunday, set()) == "2026-08-14"
    assert SCH.required_completed_session(friday_eod, set()) == "2026-08-14"
    assert SCH.required_completed_session(monday_morning, set()) == "2026-08-14"
    keys = SCH.eod_ready_keys(sunday, set(), latest="2026-08-14")
    assert "EOD_DATA_READY:2026-08-14" in keys
    assert "EOD_DATA_READY:2026-08-16" in keys


def test_outcome_resolution_accepts_friday_tape_on_sunday():
    from datetime import datetime
    from research.autonomy import health as H
    from research.autonomy import jobs as JOBS
    from research.autonomy import job_store as JS

    class Deps:
        resolved = None

        def now_ist(self): return datetime(2026, 8, 16, 10, 0)
        def holidays(self): return set()
        def active_snapshot_id(self): return "snap-fri"
        def active_snapshot_info(self): return {"latest_date": "2026-08-14"}
        def resolve_outcomes(self, session_date, failures=()):
            self.resolved = session_date
            return {"positions_closed": ["A"], "outcomes_recorded": ["A"]}

    deps = Deps()
    result = JOBS.run_outcome_resolution(JOBS._Ctx(deps))
    assert result.status == JS.SUCCEEDED
    assert deps.resolved == "2026-08-14"
    assert H.SNAPSHOT_STALE in result.clears


def test_learning_and_research_use_friday_session_on_sunday():
    from datetime import datetime
    from research.autonomy import jobs as JOBS
    from research.autonomy import job_store as JS
    from research.autonomy import schedules as SCH

    class Deps:
        learned = None
        researched = None

        def now_ist(self): return datetime(2026, 8, 16, 10, 0)
        def holidays(self): return set()
        def run_learning(self, session_date, dialogue=None):
            self.learned = session_date
            return {"diagnostics": 1}
        def run_research(self, session_date, dialogue=None):
            self.researched = session_date
            return {"decision": "NO_RESEARCH_GAP"}

    deps = Deps()
    ctx = JOBS._Ctx(deps)
    learned = JOBS.run_learning_cycle(ctx)
    researched = JOBS.run_research_cycle(ctx)
    assert learned.status == JS.SUCCEEDED
    assert researched.status == JS.SUCCEEDED
    assert deps.learned == "2026-08-14"
    assert deps.researched == "2026-08-14"
    assert SCH.is_expected_eod_wait(
        blocked_on="EOD_DATA_READY:2026-08-16",
        summary="outcomes wait for completed-session data (2026-08-14 < 2026-08-16)",
    )
    assert not SCH.is_expected_eod_wait(blocked_on="AUTH_READY", summary="auth required")


def test_all_ui_entrypoints_are_scheduler_side_effect_free():
    """No browser surface may start or directly mutate the autonomous brain."""
    paths = [Path("app.py"), Path("legacy_app.py"), *Path("ui").glob("*.py")]
    banned = (
        ".start()", "brain.run_once(", "brain.grow_one_day(",
        "brain.engage_paper_autonomy(", "brain.disengage_paper_autonomy(",
        "brain.enable_paper_auto(", "brain.disable_paper_auto(",
        "get_brain().start(", "get_news_curator_service().start(",
    )
    offenders = []
    for path in paths:
        source = path.read_text(encoding="utf-8")
        for token in banned:
            if token in source:
                offenders.append(f"{path}:{token}")
    assert not offenders, offenders
