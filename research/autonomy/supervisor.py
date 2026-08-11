"""The single durable scheduler and mutation owner for QuantTerm PAPER_AUTO."""
from __future__ import annotations

import json
import os
from pathlib import Path

from research.autonomy import job_store as JS
from research.autonomy import schedules as SCH
from research.autonomy import supervisor_state as ST
from research.autonomy import health as H
from research.autonomy import jobs as JOBS
from research.autonomy import controls as CTRL
from research.autonomy.dialogue import DialogueLog, Record, OPERATIONAL_INCIDENT

_MAX_ATTEMPTS = 5
_BASE_BACKOFF_S = 2.0
_MAX_BACKOFF_S = 300.0


class SingleInstanceLock:
    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        from utils.process_lock import ProcessFileLock

        self._lock = ProcessFileLock(self.path)
        self._fh = None  # compat for older diagnostics

    def acquire(self) -> bool:
        ok = self._lock.acquire()
        self._fh = self._lock._handle
        return ok

    def release(self) -> None:
        self._lock.release()
        self._fh = None


class Supervisor:
    def __init__(self, root, *, deps=None, clock=None, owner="supervisor"):
        import time
        self.clock = clock or time.time
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.owner = owner
        self.deps = deps or JOBS.Deps(self.root)
        from research.autonomy.live_feed import LiveFeedController
        self.live_feed = LiveFeedController(self.root / "live_feed.json")
        if isinstance(self.deps, JOBS.Deps):
            self.deps.live_feed = self.live_feed
        self.jobs = JS.JobStore(self.root / "jobs.db", clock=self.clock)
        self.controls = CTRL.ControlStore(self.root / "controls.db")
        self._state_persist = ST.StatePersistence(self.root / "state.json")
        self.state = self._state_persist.load()
        self.dialogue = DialogueLog(self.root / "dialogue.jsonl")
        self.lock = SingleInstanceLock(self.root / "supervisor.lock")
        self._status_path = self.root / "status.json"
        self._failures_path = self.root / "failures.json"
        self._owner_path = self.root / "owner_state.json"
        self.failures = self._load_failures()
        self.owner_state = self._load_owner_state()
        self._reconcile_expected_research_gaps()
        if self.owner_state.get("new_entries_paused"):
            self.failures.add(H.OWNER_PAUSED)
        else:
            self.failures.discard(H.OWNER_PAUSED)
        self._stop = False
        self._running = False

    def _load_failures(self) -> set:
        try:
            return set(json.loads(self._failures_path.read_text(encoding="utf-8")))
        except Exception:
            return set()

    def _reconcile_expected_research_gaps(self) -> None:
        """Drop sticky CA / universe incompletes for operator-pending research gaps.

        Missing CA or survivorship history is owned by the retail checklist and BLOCKED
        jobs. QuantTerm will not invent those ledgers — they must not permanently red
        the heartbeat as organisational failures.
        """
        self.failures.discard(H.CA_INCOMPLETE)
        self.failures.discard(H.UNIVERSE_INCOMPLETE)

    def _save_failures(self) -> None:
        tmp = self._failures_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(sorted(self.failures)), encoding="utf-8")
        os.replace(tmp, self._failures_path)

    def _load_owner_state(self) -> dict:
        try:
            data = json.loads(self._owner_path.read_text(encoding="utf-8"))
            return {"paper_auto_enabled": bool(data.get("paper_auto_enabled", True)),
                    "new_entries_paused": bool(data.get("new_entries_paused", False)),
                    "halted": bool(data.get("halted", False))}
        except Exception:
            enabled = True
            try:
                repo = Path(__file__).resolve().parents[2]
                cfg = json.loads((repo / "logs" / "intelligence" / "paper_config.json").read_text())
                enabled = bool(cfg.get("enabled", True))
            except Exception:
                pass
            return {"paper_auto_enabled": enabled, "new_entries_paused": not enabled,
                    "halted": False}

    def _save_owner_state(self) -> None:
        tmp = self._owner_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.owner_state, indent=2), encoding="utf-8")
        os.replace(tmp, self._owner_path)

    def start(self) -> bool:
        if not self.lock.acquire():
            return False
        os.environ["QT_AUTONOMY_OWNER"] = "1"
        self._running = True
        self._transition(ST.STARTING, "boot", "Supervisor acquired the single mutation-owner lock.", "start")
        if hasattr(self.deps, "notify_online"):
            try:
                self.deps.notify_online()
            except Exception:
                pass
        return True

    def _transition(self, to_state, reason, explanation, trigger, snapshot_id=None):
        self.state.transition(to_state, reason_code=reason, explanation=explanation, trigger=trigger,
                              snapshot_id=snapshot_id)
        self._state_persist.save(self.state)
        self._write_status()

    def heartbeat(self):
        # Keep operator-pending research gaps out of sticky failures (checklist owns them).
        before = set(self.failures)
        self._reconcile_expected_research_gaps()
        if self.failures != before:
            self._save_failures()
        self._write_status()

    def _write_status(self):
        caps = H.capabilities(self.failures)
        d = self.state.as_dict()
        last_cycle = {}
        if isinstance(self.deps, JOBS.Deps):
            try:
                from research.auto_research.scheduler import get_brain
                last_cycle = dict(get_brain().state.last_intel_cycle or {})
            except Exception:
                last_cycle = {}
        d.update({
            "heartbeat_ist": ST._now_ist_iso(), "active_failures": sorted(self.failures),
            "capabilities": caps, "overdue_critical": [j.job_type for j in self.jobs.overdue_critical()],
            "jobs": self._job_counts(), "owner_state": dict(self.owner_state),
            "scheduler_owner_pid": os.getpid(), "scheduler_of_record": "quantterm-autonomy",
            "process_running": bool(self._running), "last_cycle": last_cycle,
            "live_feed": self.live_feed.health(),
        })
        tmp = self._status_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(d, indent=2, default=str), encoding="utf-8")
        os.replace(tmp, self._status_path)

    def _job_counts(self) -> dict:
        counts = {}
        for job in self.jobs.list(limit=1000):
            counts[job.status] = counts.get(job.status, 0) + 1
        return counts

    def _enqueue_daily_foundation(self, now_ist, session_date):
        # Explicitly scheduled data tasks.  Auth-dependent jobs may become BLOCKED and are requeued
        # with the same identity when AUTH_READY is restored.
        self.jobs.enqueue(SCH.INSTRUMENT_REFRESH, idempotency_key=SCH.instrument_key(session_date))
        self.jobs.enqueue(SCH.DATA_REFRESH, idempotency_key=SCH.data_refresh_key(session_date), critical=True)
        self.jobs.enqueue(SCH.INDEX_WARMUP, idempotency_key=SCH.index_warmup_key(session_date))
        self.jobs.enqueue(SCH.CORPORATE_ACTIONS, idempotency_key=SCH.corporate_actions_key(session_date))
        self.jobs.enqueue(SCH.UNIVERSE_HISTORY, idempotency_key=SCH.universe_history_key(session_date))
        self.jobs.enqueue(SCH.BHAVCOPY_UPDATE, idempotency_key=SCH.bhavcopy_key(session_date))
        self.jobs.enqueue(SCH.OPTIONS_EOD, idempotency_key=SCH.options_eod_key(session_date))

    @staticmethod
    def _news_bucket(now_ist, market_open: bool) -> str:
        size = 5 if market_open else 20
        minute = now_ist.hour * 60 + now_ist.minute
        bucket = minute - minute % size
        return f"{bucket // 60:02d}{bucket % 60:02d}"

    def enqueue_due(self, now_ist=None):
        now_ist = now_ist or self.deps.now_ist()
        holidays = self.deps.holidays()
        if not SCH._is_session_day(now_ist, holidays):
            return
        session_date = now_ist.date().isoformat()
        self.jobs.enqueue(
            SCH.AUTH_HEALTH,
            idempotency_key=f"auth:{session_date}:{SCH.auth_probe_bucket(now_ist)}",
            critical=True,
        )
        # Begin preparation from 07:30 onward; blocked dependency semantics make an early run safe.
        if now_ist.time() >= SCH.AUTH_WINDOW_START:
            self._enqueue_daily_foundation(now_ist, session_date)
        self.jobs.enqueue(
            SCH.NEWS_REFRESH,
            idempotency_key=SCH.news_key(session_date,
                                         self._news_bucket(now_ist, SCH.market_is_open(now_ist, holidays))),
        )

        slot = SCH.scan_slot(now_ist, holidays)
        if slot == "eod":
            self.jobs.enqueue(SCH.BHAVCOPY_UPDATE,
                              idempotency_key=SCH.eod_bhavcopy_key(session_date))
            self.jobs.enqueue(SCH.OPTIONS_EOD,
                              idempotency_key=SCH.eod_options_key(session_date))
            eod_refresh = self.jobs.enqueue(
                SCH.DATA_REFRESH, idempotency_key=SCH.eod_data_refresh_key(session_date), critical=True)
            if eod_refresh.status != JS.SUCCEEDED:
                return
        snap = str(self.deps.active_snapshot_id() or "none")
        if slot:
            skey = SCH.scan_key(snap, slot, session_date)
            scan_job = self.jobs.enqueue(SCH.MARKET_SCAN, idempotency_key=skey,
                                         input_snapshot_id=None if snap == "none" else snap)
            if slot.startswith("intraday") and scan_job.status == JS.SUCCEEDED:
                self.jobs.enqueue(
                    SCH.PAPER_CYCLE,
                    idempotency_key=SCH.paper_cycle_key(snap, f"{session_date}:{slot}"),
                    input_snapshot_id=None if snap == "none" else snap,
                    critical=True,
                )
            if slot == "eod" and scan_job.status == JS.SUCCEEDED:
                if SCH.long_term_weekly_due(now_ist, holidays):
                    self.jobs.enqueue(
                        SCH.LONG_TERM_SCAN,
                        idempotency_key=SCH.long_term_key(session_date),
                    )
                outcome = self.jobs.enqueue(SCH.OUTCOME_RESOLUTION,
                                            idempotency_key=SCH.outcome_key(session_date), critical=True)
                if outcome.status == JS.SUCCEEDED:
                    learning = self.jobs.enqueue(SCH.LEARNING_CYCLE,
                                                 idempotency_key=SCH.learning_key(session_date))
                    if learning.status == JS.SUCCEEDED:
                        self.jobs.enqueue(SCH.RESEARCH_CYCLE,
                                          idempotency_key=SCH.research_key(session_date))

    def _process_controls(self):
        for control in self.controls.pending():
            try:
                ctype = control.control_type
                now = self.deps.now_ist()
                session = now.date().isoformat()
                snap = str(self.deps.active_snapshot_id() or "none")
                if ctype == CTRL.ENABLE_PAPER_AUTO:
                    self.owner_state["paper_auto_enabled"] = True
                    self.owner_state["new_entries_paused"] = False
                    self.failures.discard(H.OWNER_PAUSED)
                    try:
                        from research.auto_research.scheduler import get_brain
                        brain = get_brain(); brain.enable_paper_auto(); brain.engage_paper_autonomy()
                    except Exception:
                        pass
                elif ctype == CTRL.PAUSE_NEW_PAPER_ENTRIES:
                    self.owner_state["new_entries_paused"] = True
                    self.failures.add(H.OWNER_PAUSED)
                elif ctype == CTRL.RESUME_NEW_PAPER_ENTRIES:
                    self.owner_state["new_entries_paused"] = False
                    self.failures.discard(H.OWNER_PAUSED)
                elif ctype == CTRL.REFRESH_DATA_NOW:
                    self.jobs.enqueue(SCH.DATA_REFRESH,
                                      idempotency_key=f"manual:data:{control.control_id}", critical=True)
                elif ctype == CTRL.RUN_SCAN_NOW:
                    self.jobs.enqueue(SCH.MARKET_SCAN,
                                      idempotency_key=f"manual:scan:{snap}:{control.control_id}")
                elif ctype == CTRL.RUN_CYCLE_NOW:
                    self.jobs.enqueue(SCH.PAPER_CYCLE,
                                      idempotency_key=f"manual:cycle:{snap}:{control.control_id}", critical=True)
                elif ctype == CTRL.REFRESH_NEWS_NOW:
                    self.jobs.enqueue(SCH.NEWS_REFRESH,
                                      idempotency_key=f"manual:news:{control.control_id}")
                elif ctype == CTRL.RUN_RESEARCH_NOW:
                    self.jobs.enqueue(SCH.RESEARCH_CYCLE,
                                      idempotency_key=f"manual:research:{control.control_id}")
                elif ctype == CTRL.RUN_LONG_TERM_SCAN_NOW:
                    self.jobs.enqueue(SCH.LONG_TERM_SCAN,
                                      idempotency_key=f"manual:long-term:{control.control_id}")
                elif ctype == CTRL.REFRESH_LONG_TERM_NOW:
                    self.jobs.enqueue(SCH.LONG_TERM_REFRESH,
                                      idempotency_key=f"manual:long-term-refresh:{control.control_id}")
                elif ctype == CTRL.TRACK_LONG_TERM_IDEA:
                    try:
                        value = json.loads(control.value or "{}")
                    except Exception:
                        value = {}
                    symbol = str(value.get("symbol", "") if isinstance(value, dict) else "").upper()
                    from product.long_term_store import load_long_term_scan
                    payload = load_long_term_scan() or {}
                    row = next((dict(r) for r in payload.get("records", [])
                                if str(r.get("symbol", "")).upper() == symbol), None)
                    if not row or row.get("classification") not in (
                            "QUALITY_COMPOUNDER", "GARP_CANDIDATE"):
                        raise ValueError("symbol is not in the eligible current long-term shortlist")
                    if float(row.get("fundamental_coverage") or 0) < 0.50:
                        raise ValueError(
                            "symbol lacks sufficient fundamental coverage for a long-term bet"
                        )
                    from core.long_term_tracker import record_picks
                    record_picks([{**row, "score": row.get("combined_score"),
                                   "thesis": "; ".join(row.get("quality_factors", [])[:3])}])
                elif ctype == CTRL.HALT_AUTONOMY:
                    self.owner_state["halted"] = True
                    self.owner_state["new_entries_paused"] = True
                    self.failures.add(H.OWNER_PAUSED)
                    self._transition(ST.HALTED, "owner_halt", "Owner halted autonomy.", control.control_id)
                elif ctype == CTRL.RESUME_AUTONOMY:
                    self.owner_state["halted"] = False
                    self._transition(ST.STARTING, "owner_resume", "Owner resumed autonomy.", control.control_id)
                self._save_owner_state(); self._save_failures()
                self.controls.finish(control.control_id, result="applied")
            except Exception as exc:
                self.controls.finish(control.control_id, ok=False, result=str(exc))
                self._incident("CONTROL_FAILED", f"{control.control_type}: {exc}")


    def _desired_live_symbols(self) -> set[str]:
        symbols = set()
        try:
            from product.scan_store import load_scan, watchlist_rows
            symbols |= {str(r.get("symbol", "")).upper() for r in watchlist_rows(load_scan(), limit=60)}
        except Exception:
            pass
        try:
            repo = Path(__file__).resolve().parents[2]
            book = json.loads((repo / "logs" / "intelligence" / "intel_book.json").read_text())
            symbols |= {str(p.get("symbol", "")).upper() for p in book.get("open", [])}
        except Exception:
            pass
        return {s for s in symbols if s}

    def _manage_live_feed(self, now_ist) -> None:
        # Only the production dependency set owns a real feed. Injected tests stay network-free.
        if not isinstance(self.deps, JOBS.Deps):
            return
        if SCH.market_is_open(now_ist, self.deps.holidays()) and not (
                {H.AUTH_MISSING, H.AUTH_EXPIRED} & self.failures):
            symbols = self._desired_live_symbols()
            health = self.live_feed.start(symbols) if symbols else self.live_feed.health()
            if symbols and (health.get("last_error") or not health.get("connected")):
                self.failures.add(H.LIVE_FEED_STALE)
            elif health.get("symbols_ticking", 0) > 0:
                self.failures.discard(H.LIVE_FEED_STALE)
            if hasattr(self.deps, "observe_live_breakouts"):
                try:
                    self.deps.observe_live_breakouts()
                except Exception:
                    pass
        elif not SCH.market_is_open(now_ist, self.deps.holidays()):
            self.live_feed.stop()
        self._save_failures()

    def tick(self, now_ist=None):
        self.jobs.reclaim_expired()
        self._process_controls()
        current = now_ist or self.deps.now_ist()
        self._manage_live_feed(current)
        if self.owner_state.get("halted"):
            self.heartbeat()
            return None
        self.enqueue_due(current)
        self._check_overdue()
        job = self.jobs.lease_due(self.owner, lease_seconds=300.0)
        if job is None:
            self.heartbeat()
            return None
        self._execute(job)
        self.heartbeat()
        return job

    def _execute(self, job):
        handler = JOBS.HANDLERS.get(job.job_type)
        if handler is None:
            self.jobs.complete(job.job_id, JS.PERMANENT_FAILED, error_code="NO_HANDLER",
                               error_message=f"no handler for {job.job_type}")
            self._incident("NO_HANDLER", f"No handler for job {job.job_type}", job)
            return
        ctx = JOBS._Ctx(self.deps, active_failures=self.failures,
                        owner_paused=self.owner_state.get("new_entries_paused", False), root=self.root)
        ctx.dialogue = self.dialogue
        if job.job_type == SCH.DATA_REFRESH and str(job.idempotency_key or "").endswith(":eod"):
            ctx.required_session_date = self.deps.now_ist().date().isoformat()
        try:
            result = handler(ctx)
        except Exception as exc:
            self._retry_or_fail(job, error_code="HANDLER_EXCEPTION", error_message=str(exc))
            self._incident("HANDLER_EXCEPTION", f"{job.job_type}: {exc}", job)
            return

        self.failures |= set(result.failures)
        self.failures -= set(result.clears)
        # CA / universe incompletes are never sticky organisational failures.
        self._reconcile_expected_research_gaps()
        self._save_failures()

        if result.status == JS.RETRYABLE_FAILED:
            self._retry_or_fail(job, error_code=result.error_code, error_message=result.error_message,
                                summary=result.summary)
            self._incident(result.error_code or "RETRYABLE", result.summary or result.error_message, job)
        elif result.status == JS.BLOCKED:
            dependency = result.blocked_on or "MANUAL_REVIEW"
            self.jobs.block(job.job_id, dependency=dependency,
                            reason=result.error_message or result.summary,
                            dependency_version=result.dependency_version or None,
                            result_summary=result.summary)
            self._incident("BLOCKED", result.summary, job)
        else:
            self.jobs.complete(job.job_id, result.status, result_summary=result.summary,
                               output_snapshot_id=result.output_snapshot_id,
                               error_code=result.error_code, error_message=result.error_message)
            if result.status == JS.SUCCEEDED:
                for dependency in result.unblocks:
                    self.jobs.unblock_dependency(dependency)

        target = self._gated_state(result.state_hint)
        if target and target != self.state.state:
            self._transition(target, reason=job.job_type,
                             explanation=result.summary or job.job_type, trigger=job.job_id,
                             snapshot_id=result.output_snapshot_id)

    def _gated_state(self, hint):
        if self.owner_state.get("halted"):
            return ST.HALTED
        now = self.deps.now_ist()
        if SCH.kite_login_optional(now, self.deps.holidays()):
            if hint in (ST.AUTH_REQUIRED, ST.DATA_REFRESHING, ST.STARTING, None):
                return ST.OBSERVING
        if H.AUTH_MISSING in self.failures or H.AUTH_EXPIRED in self.failures:
            return ST.AUTH_REQUIRED
        if H.SNAPSHOT_STALE in self.failures:
            return ST.DATA_BLOCKED
        return hint

    def _retry_or_fail(self, job, *, error_code, error_message, summary=""):
        eod_pending = error_code == "EOD_DATA_PENDING"
        max_attempts = 12 if eod_pending else _MAX_ATTEMPTS
        if job.attempt >= max_attempts:
            self.jobs.complete(job.job_id, JS.PERMANENT_FAILED, error_code=error_code,
                               error_message=error_message, result_summary=summary)
            return
        backoff = 300.0 if eod_pending else min(_MAX_BACKOFF_S, _BASE_BACKOFF_S * (2 ** job.attempt))
        self.jobs.reschedule_retry(job.job_id, when=self.clock() + backoff,
                                   error_code=error_code, error_message=error_message)

    def _check_overdue(self):
        overdue = self.jobs.overdue_critical(grace_seconds=3600.0)
        if overdue and self.state.state not in (ST.DEGRADED, ST.HALTED):
            names = ", ".join(sorted({j.job_type for j in overdue}))
            self._incident("CRITICAL_OVERDUE", f"Critical jobs overdue: {names}", overdue[0])
            self._transition(ST.DEGRADED, "critical_overdue",
                             f"A critical job is overdue: {names}.", "overdue_check")

    def _incident(self, code, message, job=None):
        self.dialogue.append(Record(record_type=OPERATIONAL_INCIDENT, producer="supervisor",
                                    claim=message, evidence={"error_code": code,
                                    "job_type": getattr(job, "job_type", ""),
                                    "job_id": getattr(job, "job_id", "")}, decision=code))
        if hasattr(self.deps, "notify_incident"):
            try:
                self.deps.notify_incident(code, message)
            except Exception:
                pass

    def run(self, *, interval_s=15.0, sleep_fn=None, max_iterations=None):
        import time
        sleep_fn = sleep_fn or time.sleep
        count = 0
        while not self._stop:
            self.tick()
            count += 1
            if max_iterations is not None and count >= max_iterations:
                break
            sleep_fn(interval_s)

    def stop(self):
        self._stop = True

    def shutdown(self):
        self._stop = True
        self._running = False
        self._state_persist.save(self.state)
        self._write_status()
        self.live_feed.stop()
        self.controls.close()
        self.jobs.close()
        self.lock.release()
