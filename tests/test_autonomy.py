"""
Acceptance tests for the autonomous quant organisation (deterministic, network-free).

Covers supervisor durability (lock/lease/idempotency/recovery/retry/overdue/state/shutdown), data-job
behaviour, observation gates, the typed research dialogue (preregistration/dedupe/no-self-approval/
failed-memory/successor-versioning), evidence-gated promotion, the safety invariants (no broker, live
locked, constitutional limits immutable, failure≠no-trade) and the read-only product projection.
"""
from __future__ import annotations

import inspect
import io
import threading
import time
import tokenize
from datetime import datetime
from pathlib import Path

import pytest

from research.autonomy import job_store as JS
from research.autonomy import schedules as SCH
from research.autonomy import supervisor_state as ST
from research.autonomy import health as H
from research.autonomy import jobs as JOBS
from research.autonomy import hypotheses as HYP
from research.autonomy import challenge as CH
from research.autonomy import promotion as PR
from research.autonomy.supervisor import Supervisor
from research.autonomy.dialogue import DialogueLog, Record, HYPOTHESIS, OPERATIONAL_INCIDENT


# ── fakes ────────────────────────────────────────────────────────────────────────
_NOW = datetime(2026, 7, 31, 10, 0)          # Friday, in the entry window


class _Report:
    def __init__(self, ok=True, blocker=""):
        self._ok = ok; self.blocker = blocker; self.active_pointer = "snap1"; self.snapshot_id = "snap1"; self.quality = {}
    def status(self, k): return "PASS" if self._ok else "FAIL"


class FakeDeps:
    def __init__(self, *, now=_NOW, authed=True, data_ok=True, scan=None, cycle=None,
                 scan_raises=False):
        self._now = now; self.authed = authed; self.data_ok = data_ok
        self._scan = scan if scan is not None else {"summary": {"with_any_setup": 40, "momentum": 6}}
        self._cycle = cycle if cycle is not None else {"eligibility": "NO_ELIGIBLE_TRADE"}
        self.scan_raises = scan_raises
    def now_ist(self): return self._now
    def holidays(self): return set()
    def session_valid(self): return self.authed
    def activate(self): return _Report(self.data_ok, "" if self.data_ok else "not forward-eligible")
    def active_snapshot_id(self): return "snap1" if self.data_ok else None
    def run_scan(self):
        if self.scan_raises: raise RuntimeError("provider down")
        return self._scan
    def run_paper_cycle(self, entries_allowed): return self._cycle
    def news_health(self): return {"running": True, "error": ""}


def _sup(tmp_path, deps=None, clock=None):
    return Supervisor(tmp_path / "auto", deps=deps or FakeDeps(), clock=clock)


# ══ Supervisor durability ════════════════════════════════════════════════════════
def test_single_instance_lock(tmp_path):
    a = _sup(tmp_path); b = _sup(tmp_path)
    assert a.start() is True
    assert b.start() is False               # second instance refused while first holds the lock
    a.shutdown()
    assert b.start() is True                # released → now available
    b.shutdown()


def test_job_recovery_after_process_death(tmp_path):
    clk = [1000.0]
    store = JS.JobStore(tmp_path / "j.db", clock=lambda: clk[0])
    store.enqueue(SCH.PAPER_CYCLE, idempotency_key="k1")
    leased = store.lease_due("dead-owner", lease_seconds=60)
    assert leased.status == JS.RUNNING       # process now "dies" without completing
    clk[0] += 120                            # lease expires
    assert store.reclaim_expired() == 1
    again = store.lease_due("new-owner")
    assert again is not None and again.job_id == leased.job_id and again.attempt == 2


def test_expired_lease_is_reclaimed_on_lease_due(tmp_path):
    clk = [0.0]
    store = JS.JobStore(tmp_path / "j.db", clock=lambda: clk[0])
    store.enqueue(SCH.MARKET_SCAN, idempotency_key="s1")
    store.lease_due("o1", lease_seconds=10)
    clk[0] = 100
    got = store.lease_due("o2")              # reclaims the dead RUNNING job directly
    assert got is not None and got.lease_owner == "o2"


def test_idempotent_repeated_enqueue(tmp_path):
    store = JS.JobStore(tmp_path / "j.db")
    a = store.enqueue(SCH.DATA_REFRESH, idempotency_key="d1")
    b = store.enqueue(SCH.DATA_REFRESH, idempotency_key="d1")
    assert a.job_id == b.job_id and len(store.list()) == 1


def test_retryable_then_permanent(tmp_path):
    clk = [0.0]
    sup = _sup(tmp_path, deps=FakeDeps(scan_raises=True), clock=lambda: clk[0])
    sup.start()
    j = sup.jobs.enqueue(SCH.MARKET_SCAN, idempotency_key="s1")
    for _ in range(6):
        leased = sup.jobs.lease_due(sup.owner)
        if leased is None: break
        sup._execute(leased); clk[0] += 1000
    final = sup.jobs.get(j.job_id)
    assert final.status == JS.PERMANENT_FAILED and final.attempt >= 5
    sup.shutdown()


def test_critical_overdue_alerts_and_degrades(tmp_path):
    clk = [10_000.0]
    sup = _sup(tmp_path, clock=lambda: clk[0]); sup.start()
    sup.jobs.enqueue(SCH.PAPER_CYCLE, scheduled_for=0.0, idempotency_key="old", critical=True)
    sup._check_overdue()
    assert sup.state.state == ST.DEGRADED
    assert any(r["record_type"] == OPERATIONAL_INCIDENT for r in sup.dialogue.all())
    sup.shutdown()


def test_state_machine_records_transition(tmp_path):
    s = ST.SupervisorState()
    t = s.transition(ST.PAPER_ACTIVE, reason_code="cycle", explanation="trading", trigger="job1",
                     snapshot_id="snapX")
    assert t.from_state == ST.STARTING and t.to_state == ST.PAPER_ACTIVE
    assert t.new_risk_permitted is True and t.positions_manageable is True and t.snapshot_id == "snapX"
    halted = s.transition(ST.HALTED, reason_code="stop", explanation="halt", trigger="job2")
    assert halted.new_risk_permitted is False and halted.positions_manageable is False


def test_graceful_shutdown_persists_and_releases(tmp_path):
    sup = _sup(tmp_path); sup.start()
    sup.tick(_NOW)
    sup.shutdown()
    assert (tmp_path / "auto" / "status.json").exists()
    from product.autonomy_status import read_autonomy_status
    assert read_autonomy_status(root=tmp_path / "auto")["running"] is False
    reopened = _sup(tmp_path)
    assert reopened.start() is True    # lock released
    reopened.shutdown()


# ══ Data jobs ════════════════════════════════════════════════════════════════════
def test_auth_required_when_session_invalid(tmp_path):
    r = JOBS.run_auth_health(JOBS._Ctx(FakeDeps(authed=False)))
    assert r.status == JS.BLOCKED and H.AUTH_MISSING in r.failures and r.state_hint == ST.AUTH_REQUIRED
    ok = JOBS.run_auth_health(JOBS._Ctx(FakeDeps(authed=True)))
    assert ok.status == JS.SUCCEEDED and ok.state_hint == ST.DATA_REFRESHING   # re-probed, not cached


def test_data_refresh_failure_preserves_and_blocks(tmp_path):
    r = JOBS.run_data_refresh(JOBS._Ctx(FakeDeps(data_ok=False)))
    assert r.status == JS.BLOCKED and r.output_snapshot_id is None       # previous active preserved
    assert H.SNAPSHOT_STALE in r.failures and r.state_hint == ST.DATA_BLOCKED
    assert r.new_entries_allowed is False


def test_stale_data_not_shown_ready(tmp_path):
    sup = _sup(tmp_path, deps=FakeDeps(data_ok=False)); sup.start()
    for _ in range(6):
        if sup.tick(_NOW) is None and not sup._bg_threads:
            break
        assert sup.join_workers(2.0)
    from product.autonomy_status import read_autonomy_status
    status = read_autonomy_status(root=tmp_path / "auto")
    assert status["state"] == ST.DATA_BLOCKED and status["new_paper_entries"] == H.BLOCKED
    sup.shutdown()


def test_instrument_reconciliation_persists_mapping_exclusions():
    from data.fno_universe import build_fno_universe
    rep = build_fno_universe([{"exchange": "NFO", "segment": "NFO-FUT", "instrument_type": "FUT",
                               "name": "MISSING", "tradingsymbol": "MISSINGFUT", "expiry": "2026-08-27"}],
                             as_of=None.__class__ and __import__("datetime").date(2026, 7, 31))
    assert rep.exclusions and rep.exclusions[0].stage == "canonical_mapping"


# ══ Observation ══════════════════════════════════════════════════════════════════
def test_scan_runs_without_streamlit(tmp_path):
    r = JOBS.run_market_scan(JOBS._Ctx(FakeDeps()))
    assert r.status == JS.SUCCEEDED and "scan complete" in r.summary   # no streamlit involved


def test_opening_noise_runs_management_only():
    pre = datetime(2026, 7, 31, 9, 20)       # before 09:30 → opening noise
    assert SCH.in_opening_noise(pre) and not SCH.entries_allowed_by_clock(pre)
    # The job succeeds as an observation/management cycle; the authoritative runtime receives
    # the entry gate BEFORE mutation. BLOCKED is reserved for an unmet job dependency.
    seen = {}
    class GateDeps(FakeDeps):
        def run_paper_cycle(self, entries_allowed, reason="", phase="", failures=()):
            seen.update(allowed=entries_allowed, reason=reason, phase=phase)
            return {"eligibility": "BLOCKED_SAFETY"}
    r = JOBS.run_paper_cycle(JOBS._Ctx(GateDeps(now=pre)))
    assert r.status == JS.SUCCEEDED and r.new_entries_allowed is False
    assert seen == {"allowed": False, "reason": "ENTRY_WINDOW_CLOSED", "phase": "opening_noise"}
    assert r.metadata["eligibility"] == "BLOCKED_SAFETY"
    assert SCH.entries_allowed_by_clock(datetime(2026, 7, 31, 10, 0))


def test_provider_failure_is_not_no_opportunity():
    r = JOBS.run_market_scan(JOBS._Ctx(FakeDeps(scan_raises=True)))
    assert r.status == JS.RETRYABLE_FAILED and r.error_code == "SCAN_ERROR"   # a failure, not empty


def test_scan_slot_is_deterministic():
    a = SCH.scan_slot(datetime(2026, 7, 31, 10, 7))
    b = SCH.scan_slot(datetime(2026, 7, 31, 10, 14))
    assert a == b == "intraday-1000"          # same 15-min slot → one immutable scan
    assert not SCH.scan_due(datetime(2026, 7, 31, 10, 14), a)


def test_lease_due_only_types_stays_inside_the_lane(tmp_path):
    store = JS.JobStore(tmp_path / "j.db")
    store.enqueue(SCH.NEWS_REFRESH, idempotency_key="n1")
    store.enqueue(SCH.MARKET_SCAN, idempotency_key="s1")
    store.enqueue(SCH.DATA_REFRESH, idempotency_key="d1", critical=True)
    news = store.lease_due("owner", only_types=SCH.LANE_TYPES[SCH.LANE_NEWS])
    scan = store.lease_due("owner", only_types=SCH.LANE_TYPES[SCH.LANE_SCAN])
    data = store.lease_due("owner", only_types=SCH.LANE_TYPES[SCH.LANE_DATA])
    assert news is not None and news.job_type == SCH.NEWS_REFRESH
    assert scan is not None and scan.job_type == SCH.MARKET_SCAN
    assert data is not None and data.job_type == SCH.DATA_REFRESH
    store.close()


def test_reclaim_expired_does_not_steal_live_job_ids(tmp_path):
    clk = [0.0]
    store = JS.JobStore(tmp_path / "j.db", clock=lambda: clk[0])
    job = store.enqueue(SCH.NEWS_REFRESH, idempotency_key="n1")
    store.lease_due("owner", lease_seconds=10)
    clk[0] = 100
    assert store.reclaim_expired(ignore_ids={job.job_id}) == 0
    assert store.get(job.job_id).status == JS.RUNNING
    assert store.reclaim_expired() == 1
    assert store.get(job.job_id).status == JS.PENDING
    store.close()


def test_every_job_type_has_a_worker_lane():
    assert set(SCH._JOB_LANE) == set(SCH.ALL_JOB_TYPES)
    assert SCH.lane_for(SCH.MARKET_SCAN) == SCH.LANE_SCAN
    assert SCH.lane_for(SCH.NEWS_REFRESH) == SCH.LANE_NEWS
    assert SCH.lane_for(SCH.PAPER_CYCLE) == SCH.LANE_CYCLE


def test_lease_due_runs_scan_before_older_news(tmp_path):
    """News is enqueued first every tick; it must not starve the due market scan."""
    store = JS.JobStore(tmp_path / "j.db")
    store.enqueue(SCH.NEWS_REFRESH, idempotency_key="n1", scheduled_for=10.0)
    store.enqueue(SCH.MARKET_SCAN, idempotency_key="s1", scheduled_for=20.0)
    got = store.lease_due("owner")
    assert got is not None and got.job_type == SCH.MARKET_SCAN
    store.close()


def test_lease_due_still_prefers_critical_over_scan(tmp_path):
    store = JS.JobStore(tmp_path / "j.db")
    store.enqueue(SCH.MARKET_SCAN, idempotency_key="s1")
    store.enqueue(SCH.DATA_REFRESH, idempotency_key="d1", critical=True)
    got = store.lease_due("owner")
    assert got is not None and got.job_type == SCH.DATA_REFRESH
    store.close()


def test_news_still_runs_when_no_scan_is_due(tmp_path):
    clk = [15.0]
    store = JS.JobStore(tmp_path / "j.db", clock=lambda: clk[0])
    store.enqueue(SCH.NEWS_REFRESH, idempotency_key="n1", scheduled_for=10.0)
    store.enqueue(SCH.MARKET_SCAN, idempotency_key="s1", scheduled_for=20.0)
    got = store.lease_due("owner")
    assert got is not None and got.job_type == SCH.NEWS_REFRESH
    store.close()


def test_tick_leases_scan_ahead_of_news(tmp_path):
    sup = _sup(tmp_path)
    sup.start()
    sup.enqueue_due = lambda now_ist=None: None
    sup.jobs.enqueue(SCH.NEWS_REFRESH, idempotency_key="n1")
    sup.jobs.enqueue(SCH.MARKET_SCAN, idempotency_key="s1")
    job = sup.tick(_NOW)
    assert job is not None and job.job_type == SCH.MARKET_SCAN
    assert sup.join_workers(2.0)
    news = next(j for j in sup.jobs.list() if j.job_type == SCH.NEWS_REFRESH)
    assert news.status == JS.SUCCEEDED
    sup.shutdown()


def test_force_lease_skips_queue_order(tmp_path):
    store = JS.JobStore(tmp_path / "j.db")
    news = store.enqueue(SCH.NEWS_REFRESH, idempotency_key="n1")
    scan = store.enqueue(SCH.MARKET_SCAN, idempotency_key="s1")
    got = store.force_lease(scan.job_id, "owner")
    assert got is not None and got.job_id == scan.job_id
    assert store.get(news.job_id).status == JS.PENDING
    store.close()


def test_owner_scan_click_runs_now_ahead_of_news_and_data(tmp_path):
    from research.autonomy import controls as CTRL
    sup = _sup(tmp_path)
    sup.start()
    sup.enqueue_due = lambda now_ist=None: None
    sup.jobs.enqueue(SCH.NEWS_REFRESH, idempotency_key="n1")
    sup.jobs.enqueue(SCH.DATA_REFRESH, idempotency_key="d1", critical=True)
    CTRL.request_control(CTRL.RUN_SCAN_NOW, root=sup.root, reason="close-of-market click")
    job = sup.tick(_NOW)
    assert job is not None and job.job_type == SCH.MARKET_SCAN
    assert bool(job.critical)
    assert sup.join_workers(2.0)
    by_type = {j.job_type: j.status for j in sup.jobs.list()}
    assert by_type[SCH.MARKET_SCAN] == JS.SUCCEEDED
    assert by_type[SCH.NEWS_REFRESH] == JS.SUCCEEDED
    assert by_type[SCH.DATA_REFRESH] in {JS.SUCCEEDED, JS.BLOCKED, JS.RUNNING, JS.PENDING}
    sup.shutdown()


def test_scan_click_runs_while_news_is_in_flight(tmp_path):
    from research.autonomy import controls as CTRL
    hold = threading.Event()

    class BlockingNews(FakeDeps):
        def refresh_news(self):
            hold.wait(timeout=5)
            return {"status": "OK"}

    sup = _sup(tmp_path, deps=BlockingNews())
    sup.start()
    sup.enqueue_due = lambda now_ist=None: None
    sup.jobs.enqueue(SCH.NEWS_REFRESH, idempotency_key="n1")
    first = sup.tick(_NOW)
    assert first is not None and first.job_type == SCH.NEWS_REFRESH
    CTRL.request_control(CTRL.RUN_SCAN_NOW, root=sup.root, reason="click")
    second = sup.tick(_NOW)
    assert second is not None and second.job_type == SCH.MARKET_SCAN
    hold.set()
    assert sup.join_workers(2.0)
    sup.shutdown()


def test_one_tick_runs_scan_and_news_in_parallel(tmp_path):
    news_started = threading.Event()
    scan_started = threading.Event()
    release = threading.Event()

    class Parallel(FakeDeps):
        def refresh_news(self):
            news_started.set()
            assert scan_started.wait(2.0), "scan worker should already be running"
            release.wait(2.0)
            return {"status": "OK"}

        def run_scan(self):
            scan_started.set()
            assert news_started.wait(2.0), "news worker should already be running"
            release.wait(2.0)
            return {"summary": {"with_any_setup": 1, "momentum": 1}}

    sup = _sup(tmp_path, deps=Parallel())
    sup.start()
    sup.enqueue_due = lambda now_ist=None: None
    sup.jobs.enqueue(SCH.NEWS_REFRESH, idempotency_key="n1")
    sup.jobs.enqueue(SCH.MARKET_SCAN, idempotency_key="s1")
    job = sup.tick(_NOW)
    assert job is not None and job.job_type == SCH.MARKET_SCAN
    assert news_started.wait(2.0) and scan_started.wait(2.0)
    running = {j.job_type for j in sup.jobs.list(status=JS.RUNNING)}
    assert SCH.NEWS_REFRESH in running
    assert SCH.MARKET_SCAN in running
    release.set()
    assert sup.join_workers(2.0)
    by_type = {j.job_type: j.status for j in sup.jobs.list()}
    assert by_type[SCH.MARKET_SCAN] == JS.SUCCEEDED
    assert by_type[SCH.NEWS_REFRESH] == JS.SUCCEEDED
    sup.shutdown()


def test_cycle_waits_while_scan_lane_is_busy(tmp_path):
    scan_started = threading.Event()
    release = threading.Event()
    cycle_ran = []

    class HoldScan(FakeDeps):
        def run_scan(self):
            scan_started.set()
            release.wait(3.0)
            return {"summary": {"with_any_setup": 1}}

        def run_paper_cycle(self, entries_allowed, *args, **kwargs):
            cycle_ran.append(True)
            return {"eligibility": "NO_ELIGIBLE_TRADE"}

    sup = _sup(tmp_path, deps=HoldScan())
    sup.start()
    sup.enqueue_due = lambda now_ist=None: None
    sup.jobs.enqueue(SCH.MARKET_SCAN, idempotency_key="s1")
    sup.jobs.enqueue(SCH.PAPER_CYCLE, idempotency_key="c1", critical=True)
    sup.tick(_NOW)
    assert scan_started.wait(2.0)
    cycle = next(j for j in sup.jobs.list() if j.job_type == SCH.PAPER_CYCLE)
    assert cycle.status == JS.PENDING
    assert not cycle_ran
    release.set()
    assert sup.join_workers(2.0)
    sup.tick(_NOW)
    assert sup.join_workers(2.0)
    assert cycle_ran
    sup.shutdown()


def test_news_refresh_budget_releases_worker(monkeypatch):
    monkeypatch.setattr(JOBS, "NEWS_JOB_BUDGET_S", 0.2)

    class Slow:
        def refresh_news(self):
            time.sleep(5)
            return {"status": "OK"}

    started = time.monotonic()
    result = JOBS.run_news_refresh(JOBS._Ctx(Slow()))
    assert time.monotonic() - started < 1.5
    assert result.status == JS.SUCCEEDED
    assert "background" in result.summary
    assert not result.failures


def test_news_refresh_completes_when_fast():
    class Fast:
        def refresh_news(self):
            return {"status": "OK"}

    result = JOBS.run_news_refresh(JOBS._Ctx(Fast()))
    assert result.status == JS.SUCCEEDED and "complete" in result.summary
    assert H.NEWS_UNAVAILABLE in result.clears


def test_headless_scan_service_has_no_ui_dependency():
    import ast
    root = Path(__file__).resolve().parents[1]
    tree = ast.parse((root / "scan/market_scan_service.py").read_text(encoding="utf-8"))
    imports = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imports.append(node.module or "")
    assert not any(name == "streamlit" or name.startswith("ui") for name in imports)


def test_blocked_job_unblocks_without_duplicate(tmp_path):
    store = JS.JobStore(tmp_path / "jobs.db")
    original = store.enqueue(SCH.DATA_REFRESH, idempotency_key="data:2026-07-31")
    store.block(original.job_id, dependency=JOBS.DEP_AUTH, reason="login required")
    assert store.get(original.job_id).status == JS.BLOCKED
    assert store.unblock_dependency(JOBS.DEP_AUTH) == 1
    resumed = store.lease_due("owner")
    assert resumed.job_id == original.job_id and resumed.attempt == 1
    duplicate = store.enqueue(SCH.DATA_REFRESH, idempotency_key="data:2026-07-31")
    assert duplicate.job_id == original.job_id and len(store.list()) == 1
    store.close()


def test_blocked_job_dependency_survives_restart(tmp_path):
    db = tmp_path / "jobs.db"
    first = JS.JobStore(db)
    job = first.enqueue(SCH.DATA_REFRESH, idempotency_key="same-day")
    first.block(job.job_id, dependency=JOBS.DEP_AUTH, reason="daily login")
    first.close()
    second = JS.JobStore(db)
    restored = second.get(job.job_id)
    assert restored.status == JS.BLOCKED and restored.blocked_on == JOBS.DEP_AUTH
    second.unblock_dependency(JOBS.DEP_AUTH)
    assert second.lease_due("owner").job_id == job.job_id
    second.close()


def test_every_declared_job_has_handler():
    assert set(SCH.ALL_JOB_TYPES) == set(JOBS.HANDLERS)


def test_schedule_does_not_scan_at_midnight():
    midnight = datetime(2026, 7, 31, 0, 1)
    assert SCH.scan_slot(midnight) is None
    assert not SCH.scan_due(midnight, None)
    assert SCH.scan_slot(datetime(2026, 7, 31, 9, 5)) == "premarket"


def test_ui_source_does_not_start_workers():
    root = Path(__file__).resolve().parents[1]
    for rel in ("app.py", "product/runtime.py", "ui/retail_home_momentum.py",
                "ui/retail_trade_market.py", "ui/news_curator_page.py",
                "ui/auto_research_page.py", "ui/desk_board.py", "ui/desk_pages.py",
                "product/sepa_setup.py", "product/monitor_context.py",
                "product/paper_learning.py", "product/paper_self_feed.py"):
        text = (root / rel).read_text(encoding="utf-8")
        assert ".start()" not in text, rel
        assert "start_worker=True" not in text, rel


def test_deployment_installs_two_services_and_current_branch():
    root = Path(__file__).resolve().parents[1]
    linux = (root / "deploy/setup_server.sh").read_text(encoding="utf-8")
    mac = (root / "deploy/setup_mac.sh").read_text(encoding="utf-8")
    assert "overhaul/evidence-lab" in linux
    assert "quantterm-ui.service" in linux and "quantterm-autonomy.service" in linux
    assert "com.quantterm.ui" in mac and "com.quantterm.autonomy" in mac
    assert "main.py autonomy" in linux
    assert "<string>autonomy</string>" in mac


# ══ Dialogue & research ══════════════════════════════════════════════════════════
def _spec():
    from research.strategy_studio.spec import StrategySpec
    return StrategySpec(strategy_id="MOM", name="Momentum", version=1, hypothesis="ride momentum",
                        family="cross_sectional_momentum", max_holding_days=10)


def test_missing_data_creates_data_task_not_mutation():
    gaps = HYP.plan_gaps([{"kind": "missing_universe_history", "strategy_id": "MOM",
                           "economic_impact": 0.8, "confidence": 0.9, "data_available": False},
                          {"kind": "regime_negative_expectancy", "strategy_id": "MOM",
                           "economic_impact": 0.6, "confidence": 0.7, "data_available": True}])
    data_gap = next(g for g in gaps if g.kind == "missing_universe_history")
    assert data_gap.recommended_action == "data_task"


def test_hypothesis_preregistered_and_deduped():
    mem = HYP.ResearchMemory(backend=None)
    gap = HYP.EvidenceGap("regime_negative_expectancy", "MOM", "loses in sideways", 0.6, 0.7, True, 0.3)
    prop, child = HYP.propose_hypothesis(_spec(), gap, {"max_holding_days": 5}, memory=mem)
    assert prop is not None and prop.created_before_results is True
    # a semantically identical hypothesis is rejected as a duplicate
    prop2, reason = HYP.propose_hypothesis(_spec(), gap, {"max_holding_days": 5}, memory=mem)
    assert prop2 is None and "duplicate" in reason


def test_hypothesis_only_touches_grammar_dimensions():
    mem = HYP.ResearchMemory(backend=None)
    gap = HYP.EvidenceGap("cost_drag", "MOM", "x", 0.5, 0.5, True, 0.3)
    bad, reason = HYP.propose_hypothesis(_spec(), gap, {"cost_model": "cheat"}, memory=mem)
    assert bad is None and "non-grammar" in reason


def test_successor_gets_new_version_and_hash_parent_unchanged():
    mem = HYP.ResearchMemory(backend=None)
    parent = _spec(); parent_hash = parent.config_hash()
    gap = HYP.EvidenceGap("k", "MOM", "d", 0.5, 0.5, True, 0.3)
    prop, child = HYP.propose_hypothesis(parent, gap, {"max_holding_days": 5}, memory=mem)
    assert child.version == parent.version + 1 and child.config_hash() != parent_hash
    assert parent.config_hash() == parent_hash and parent.max_holding_days == 10   # frozen parent intact


def test_failed_research_is_remembered():
    mem = HYP.ResearchMemory(backend=None)
    h = HYP.hypothesis_hash(_spec(), {"max_holding_days": 5})
    assert not mem.is_known(h)
    mem.record_dead(h, "failed after costs")
    assert mem.is_known(h)                    # an equivalent rule is not rediscovered


def test_researcher_cannot_self_promote():
    ctx = {"forward_eligible": True, "benchmark_available": True, "n_trades": 100,
           "net_expectancy_R": 0.3, "deflated_sharpe": 0.7}
    decision = CH.promotion_committee(ctx, producer="promotion_committee")
    assert decision.decision == CH.REJECT and "cannot approve its own" in decision.rationale


# ══ Evidence & promotion ═════════════════════════════════════════════════════════
def _good_ctx(**over):
    ctx = {"forward_eligible": True, "benchmark_available": True, "n_trades": 120,
           "net_expectancy_R": 0.25, "deflated_sharpe": 0.7, "reality_check_p": 0.02,
           "max_drawdown_pct": 12.0, "walk_forward_ok": True, "num_trials": 3, "parameter_count": 4,
           "top_symbol_weight": 0.2, "max_correlation_to_deployed": 0.3}
    ctx.update(over); return ctx


def test_passing_candidate_nominates_paper_not_live():
    d = CH.promotion_committee(_good_ctx(), producer="researcher")
    assert d.decision == CH.PAPER_NOMINATED
    assert PR.map_to_lifecycle(PR.COMMITTEE_TO_LADDER[d.decision]) == "APPROVED_FOR_PAPER"  # not USER_APPROVED


def test_weak_strategy_cannot_enter_paper():
    d = CH.promotion_committee(_good_ctx(net_expectancy_R=-0.1), producer="researcher")
    assert d.decision == CH.REJECT


def test_multiple_testing_blocks_promotion():
    d = CH.promotion_committee(_good_ctx(num_trials=999), producer="researcher")
    assert d.decision == CH.REJECT and any("multiple testing" in f for v in d.verdicts for f in v.findings)


def test_insufficient_sample_is_inconclusive():
    d = CH.promotion_committee(_good_ctx(n_trades=5), producer="researcher")
    assert d.decision == CH.RETEST_WITH_MORE_DATA


def test_forward_decay_reduces_allocation():
    ch = PR.allocation_action({"strategy_id": "MOM", "forward_expectancy_R": -0.05,
                               "forward_to_backtest": 0.3, "forward_trades": 15, "max_weight": 0.2},
                              current_weight=0.2)
    assert ch.action == "reduce" and ch.after_weight < ch.before_weight


def test_paper_proven_requires_configured_forward_evidence():
    assert not PR.paper_proven({"forward_trades": 10, "forward_lower_bound_R": 0.2, "forward_to_backtest": 0.9})
    assert PR.paper_proven({"forward_trades": 60, "forward_lower_bound_R": 0.2, "forward_to_backtest": 0.8})


# ══ Safety ═══════════════════════════════════════════════════════════════════════
def _code_only(mod) -> str:
    out = []
    for tok in tokenize.generate_tokens(io.StringIO(inspect.getsource(mod)).readline):
        if tok.type not in (tokenize.COMMENT, tokenize.STRING):
            out.append(tok.string)
    return " ".join(out)


def test_no_autonomy_module_reaches_broker_orders():
    import importlib, pkgutil
    import research.autonomy as pkg
    banned = ("place_order", "place_gtt", "modify_order", "cancel_order", "cancel_gtt")
    for m in pkgutil.walk_packages(pkg.__path__, pkg.__name__ + "."):
        code = _code_only(importlib.import_module(m.name))
        for b in banned:
            assert b not in code, f"{b} in {m.name}"


def test_live_execution_locked_and_never_owner_activated():
    assert PR.LIVE_EXECUTION_LOCKED is True
    pkg = {"paper_auto_operational": True, "paper_proven": True, "broker_connected": True,
           "reconciled": True, "data_forward_eligible": True, "risk_governor_healthy": True,
           "protective_exits_proven": True, "restart_recovery_proven": True,
           "order_persistence_healthy": True, "no_unresolved_critical_incident": True}
    rr = PR.live_readiness(pkg)
    assert rr.state == PR.LIMITED_LIVE_ELIGIBLE and rr.owner_activation_present is False
    assert "owner capital-envelope approval absent" in rr.blockers   # system can never self-activate


def test_constitutional_cap_not_exceeded_by_allocation():
    ch = PR.allocation_action({"strategy_id": "MOM", "forward_expectancy_R": 0.3,
                               "forward_to_backtest": 1.2, "forward_trades": 50, "max_weight": 0.2},
                              current_weight=0.2)
    assert ch.after_weight <= 0.2            # never enlarges the constitutional per-strategy cap


def test_event_store_failure_blocks_mutation_and_read_only_ui():
    caps = H.capabilities({H.EVENT_STORE_FAILURE})
    assert caps["new_paper_entries"] == H.BLOCKED and caps["research"] == H.BLOCKED
    assert caps["ui"] == H.READ_ONLY


def test_failure_never_becomes_no_trade():
    # auth missing blocks entries but KEEPS existing management — never a silent "no trade"
    caps = H.capabilities({H.AUTH_MISSING})
    assert caps["new_paper_entries"] == H.BLOCKED and caps["existing_exits"] == H.ALLOWED


# ══ Product projection ═══════════════════════════════════════════════════════════
def test_ui_reads_status_without_starting_supervisor(tmp_path):
    # a supervisor writes a status file, then we read it with NO supervisor instance
    sup = _sup(tmp_path); sup.start(); sup.tick(_NOW); sup.shutdown()
    from product.autonomy_status import read_autonomy_status
    status = read_autonomy_status(root=tmp_path / "auto")
    assert status["state"] in ST.STATES and status["plain_state"]
    assert "running" in status                # pure read; nothing was started


def test_states_are_distinct_in_projection(tmp_path):
    from product import autonomy_status as A
    assert A._STATE_PLAIN["AUTH_REQUIRED"] != A._STATE_PLAIN["DATA_BLOCKED"]
    assert A._STATE_PLAIN["OBSERVING"] != A._STATE_PLAIN["DEGRADED"]
    assert A._STATE_PLAIN["UNKNOWN"] != A._STATE_PLAIN["OBSERVING"]


def test_no_ui_control_implies_live():
    import ui.autonomy_page as page
    src = _code_only(page)
    for banned in ("place_order", "enable_live", "go_live", "USER_APPROVED"):
        assert banned not in src
