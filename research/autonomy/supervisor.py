"""The single durable scheduler and mutation owner for QuantTerm PAPER_AUTO."""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

from research.autonomy import job_store as JS
from research.autonomy import schedules as SCH
from research.autonomy import supervisor_state as ST
from research.autonomy import health as H
from research.autonomy import jobs as JOBS
from research.autonomy import controls as CTRL
from research.autonomy.dialogue import DialogueLog, Record, OPERATIONAL_INCIDENT
from core.runtime_paths import logs_path

_MAX_ATTEMPTS = 5
_BASE_BACKOFF_S = 2.0
_MAX_BACKOFF_S = 300.0


class SingleInstanceLock:
    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = None

    def acquire(self) -> bool:
        try:
            import fcntl
            self._fh = open(self.path, "w")
            try:
                fcntl.flock(self._fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError:
                self._fh.close(); self._fh = None
                return False
            self._fh.write(str(os.getpid())); self._fh.flush()
            return True
        except Exception:
            try:
                fd = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                os.write(fd, str(os.getpid()).encode()); os.close(fd)
                return True
            except FileExistsError:
                return False

    def release(self) -> None:
        try:
            if self._fh is not None:
                import fcntl
                fcntl.flock(self._fh, fcntl.LOCK_UN)
                self._fh.close()
            if self.path.exists():
                self.path.unlink()
        except Exception:
            pass


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
        from research.autonomy.incident_store import IncidentStore
        self.incidents = IncidentStore(self.root / "incidents.json")
        self.lock = SingleInstanceLock(self.root / "supervisor.lock")
        self._status_path = self.root / "status.json"
        self._failures_path = self.root / "failures.json"
        self._owner_path = self.root / "owner_state.json"
        self.failures = self._load_failures()
        self.owner_state = self._load_owner_state()
        if self.owner_state.get("new_entries_paused"):
            self.failures.add(H.OWNER_PAUSED)
        else:
            self.failures.discard(H.OWNER_PAUSED)
        self._save_failures()
        self._stop = False
        self._running = False
        self._started_at = None
        self._boot_retained_state = False

    def _load_failures(self) -> set:
        try:
            raw = set(json.loads(self._failures_path.read_text(encoding="utf-8")))
        except Exception:
            raw = set()
        return H.canonicalize_failures(raw)

    def _save_failures(self) -> None:
        tmp = self._failures_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(sorted(self.failures)), encoding="utf-8")
        os.replace(tmp, self._failures_path)

    def _load_owner_state(self) -> dict:
        try:
            data = json.loads(self._owner_path.read_text(encoding="utf-8"))
            return {
                "paper_auto_enabled": bool(data.get("paper_auto_enabled", True)),
                "new_entries_paused": bool(data.get("new_entries_paused", False)),
                "halted": bool(data.get("halted", False)),
                "observe_only_date": str(data.get("observe_only_date") or "")[:10],
                "completed_snapshot_id": str(data.get("completed_snapshot_id") or ""),
                "completed_snapshot_at": str(data.get("completed_snapshot_at") or ""),
                "completed_session_date": str(data.get("completed_session_date") or "")[:10],
            }
        except Exception:
            enabled = True
            try:
                cfg = json.loads(logs_path("intelligence", "paper_config.json").read_text())
                enabled = bool(cfg.get("enabled", True))
            except Exception:
                pass
            return {
                "paper_auto_enabled": enabled,
                "new_entries_paused": not enabled,
                "halted": False,
                "observe_only_date": "",
                "completed_snapshot_id": "",
                "completed_snapshot_at": "",
                "completed_session_date": "",
            }

    def _save_owner_state(self) -> None:
        tmp = self._owner_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.owner_state, indent=2), encoding="utf-8")
        os.replace(tmp, self._owner_path)

    def start(self) -> bool:
        if not self.lock.acquire():
            return False
        os.environ["QT_AUTONOMY_OWNER"] = "1"
        self._running = True
        self._started_at = self.clock()
        try:
            from product.decision_simulation_gate import begin_startup
            begin_startup()
        except Exception:
            pass
        persisted = str(self.state.state or ST.STARTING)
        # Restart must not wipe a durable operational state back to STARTING.
        # AUTH_HEALTH is a broker probe; it is not a reason to forget DATA_READY.
        if persisted in ST.STATES and persisted != ST.STARTING:
            self._boot_retained_state = True
            self._transition(
                persisted,
                "owner_resume",
                "Supervisor re-acquired the lock and retained the last persisted operational state.",
                "start",
            )
        else:
            self._boot_retained_state = False
            self._transition(ST.STARTING, "boot", "Supervisor acquired the single mutation-owner lock.", "start")
        # Leftover BLOCKED CA/universe rows from when the ledger files were
        # missing must retry now that the jobs actually fetch official NSE data.
        try:
            self.jobs.unblock_dependency(JOBS.DEP_CA_SOURCE)
            self.jobs.unblock_dependency(JOBS.DEP_UNIVERSE_SOURCE)
        except Exception:
            pass
        # Legacy recurring rows from the old time-bucket scheduler must not fire
        # after an upgrade. Manual controls use "manual:" keys and are preserved.
        self._retire_legacy_recurring_work()
        self._retire_obsolete_data_refresh_work()
        if hasattr(self.deps, "notify_online"):
            try:
                self.deps.notify_online()
            except Exception:
                pass
        if hasattr(self.deps, "replay_scan_alerts"):
            try:
                self.deps.replay_scan_alerts()
            except Exception:
                pass
        return True

    def _transition(self, to_state, reason, explanation, trigger, snapshot_id=None):
        self.state.transition(to_state, reason_code=reason, explanation=explanation, trigger=trigger,
                              snapshot_id=snapshot_id)
        self._state_persist.save(self.state)
        self._write_status()

    def heartbeat(self):
        self._write_status()

    def _retire_legacy_recurring_work(self) -> None:
        """Retire only rows owned by the superseded recurring scheduler.

        New durable identities survive restart:
          snapshot_*         -> one real forward scan/paper transaction
          forward_*          -> once/session forward settlement/learning/research
          hist_*             -> closed-market historical paper/learning/research
          *:eod              -> once/session official-data recovery

        Manual controls always survive.
        """
        try:
            for job in self.jobs.list(limit=2000):
                if job.status not in {JS.PENDING, JS.BLOCKED}:
                    continue
                key = str(job.idempotency_key or "")
                if key.startswith("manual:"):
                    continue

                retire = False
                if job.job_type == SCH.NEWS_REFRESH:
                    retire = True
                elif job.job_type == SCH.MARKET_SCAN:
                    try:
                        from product.decision_simulation_gate import current_startup_id
                        current_startup = current_startup_id()
                    except Exception:
                        current_startup = ""
                    retire = not (
                        key.startswith("snapshot_scan:")
                        or (
                            current_startup
                            and key.startswith(f"startup_discovery_scan:{current_startup}:")
                        )
                    )
                elif job.job_type == SCH.PAPER_CYCLE:
                    retire = not (
                        key.startswith("snapshot_paper:")
                        or key.startswith("snapshot_manage:")
                    )
                elif job.job_type == SCH.OUTCOME_RESOLUTION:
                    retire = not key.startswith("forward_outcome:")
                elif job.job_type == SCH.LEARNING_CYCLE:
                    retire = not (
                        key.startswith("forward_learning:")
                        or key.startswith("hist_learning:")
                    )
                elif job.job_type == SCH.RESEARCH_CYCLE:
                    retire = not (
                        key.startswith("forward_research:")
                        or key.startswith("hist_research:")
                        or key.startswith("research_replan:")
                    )
                elif job.job_type == SCH.HISTORICAL_PAPER_CYCLE:
                    retire = not key.startswith("hist_paper:")
                elif job.job_type == SCH.DATA_REFRESH:
                    retire = False  # current session + eod refresh are both valid
                elif job.job_type == SCH.BHAVCOPY_UPDATE:
                    retire = not key.endswith(":eod")
                elif job.job_type in {
                    SCH.CORPORATE_ACTIONS,
                    SCH.UNIVERSE_HISTORY,
                }:
                    # start() deliberately unblocks these durable official-data
                    # recovery jobs. Cancelling them here would undo recovery in
                    # the same startup transaction.
                    retire = False
                elif job.job_type in {
                    SCH.LONG_TERM_SCAN,
                    SCH.LONG_TERM_REFRESH,
                    SCH.INSTRUMENT_REFRESH,
                    SCH.INDEX_WARMUP,
                }:
                    retire = True

                if retire:
                    self.jobs.complete(
                        job.job_id,
                        JS.CANCELLED,
                        result_summary="retired legacy recurring scheduler row",
                    )
        except Exception:
            pass


    def _retire_obsolete_data_refresh_work(self) -> None:
        """Keep at most one relevant automatic DATA_REFRESH intent.

        DATA_REFRESH is a canonical snapshot refresh, not an append-only queue.
        Old session rows may survive an unclean shutdown. When official history
        is already usable under the canonical freshness policy, those historical
        intents are obsolete and must not keep the desk in DATA_REFRESHING.

        If data is genuinely stale, preserve one newest recovery intent so the
        safety gate remains fail-closed and the system can recover autonomously.
        Manual refresh controls are never touched.
        """
        try:
            now_ist = self.deps.now_ist()
            today = now_ist.date().isoformat()
            rows = [
                job for job in self.jobs.list(limit=2000)
                if job.job_type == SCH.DATA_REFRESH
                and job.status in {JS.PENDING, JS.BLOCKED}
                and not str(job.idempotency_key or "").startswith("manual:")
            ]
            if not rows:
                return

            def session_key(job):
                key = str(job.idempotency_key or "")
                parts = key.split(":")
                return parts[1] if len(parts) >= 2 and parts[0] == "data_refresh" else ""

            today_rows = [job for job in rows if session_key(job) == today]
            keep_ids = set()

            if today_rows:
                # During the cash session the normal current-session refresh is
                # authoritative. In the EOD publication window prefer the stricter
                # :eod intent. Outside both, keep only the newest current-day row.
                holidays = self.deps.holidays() if hasattr(self.deps, "holidays") else None
                if SCH.in_eod_window(now_ist, holidays):
                    preferred = [j for j in today_rows if str(j.idempotency_key or "").endswith(":eod")]
                elif SCH.market_is_open(now_ist, holidays):
                    preferred = [j for j in today_rows if not str(j.idempotency_key or "").endswith(":eod")]
                else:
                    preferred = []
                candidates = preferred or today_rows
                candidates = sorted(
                    candidates,
                    key=lambda j: (float(j.scheduled_for or 0.0), str(j.job_id)),
                    reverse=True,
                )
                keep_ids.add(candidates[0].job_id)
            else:
                try:
                    from product.readiness import official_history
                    usable = bool((official_history() or {}).get("usable_for_scan"))
                except Exception:
                    usable = False
                if not usable:
                    # Genuine stale/missing data: preserve one newest recovery
                    # intent. We do not cancel the only path that can clear the
                    # freshness blocker.
                    newest = max(
                        rows,
                        key=lambda j: (float(j.scheduled_for or 0.0), str(j.job_id)),
                    )
                    keep_ids.add(newest.job_id)

            for job in rows:
                if job.job_id in keep_ids:
                    continue
                self.jobs.complete(
                    job.job_id,
                    JS.CANCELLED,
                    result_summary="obsolete data-refresh intent retired by canonical freshness truth",
                )
        except Exception:
            # Scheduler cleanup must never weaken the underlying data gate.
            return


    def _activity_truth(self) -> dict:
        try:
            from research.autonomy.runtime_truth import derive_activity
            return derive_activity(
                self.jobs.list(limit=2000),
                now_epoch=float(self.clock()),
            )
        except Exception as exc:
            return {
                "activity": "UNKNOWN",
                "busy": False,
                "current_jobs": [],
                "current_count": 0,
                "primary_job": {},
                "source": "runtime_truth_error",
                "error": f"{type(exc).__name__}: {exc}",
            }

    def _resource_budget(self) -> dict:
        try:
            from research.autonomy.resource_governor import assess
            return assess(self.jobs.list(limit=2000), now_epoch=float(self.clock()))
        except Exception as exc:
            return {
                "historical_replay_allowed": False,
                "decision": "DEFER_HISTORICAL_REPLAY",
                "reason": f"resource governor unavailable: {type(exc).__name__}",
                "learning_allowed": True,
                "research_allowed": True,
                "live_money_unchanged": True,
            }

    def _write_status(self):
        caps = H.capabilities(self.failures)
        d = self.state.as_dict()
        activity_truth = self._activity_truth()
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
            "resource_governor": self._resource_budget(),
            "activity_truth": activity_truth,
            "current_activity": activity_truth.get("activity", "UNKNOWN"),
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
        """Automatic intraday trigger: DATA_REFRESH only.

        Price preparation/scan/paper form one snapshot transaction. Secondary
        foundation/news/research work must not interleave with or repeat it.
        """
        return self.jobs.enqueue(
            SCH.DATA_REFRESH,
            idempotency_key=SCH.data_refresh_key(session_date),
            critical=True,
        )

    def _ensure_decision_simulation_authority(self) -> bool:
        """Authorize simulation only for the current scan/thesis discovery identity.

        Startup approval remains durable, but it is not permission to use a stale
        decision projection after learning changes effective selection behavior or
        a newer scan replaces the approved scan. The current discovery projection
        must exist first. Live-money authority is independent and remains locked.
        """
        try:
            from product.decision_simulation_gate import (
                ensure_autonomous_approval,
                status,
            )

            gate = dict(status() or {})
            if not gate.get("discovery_ready"):
                return False
            if gate.get("approved"):
                return True
            approval = ensure_autonomous_approval()
            return bool(approval.get("accepted") and approval.get("approved"))
        except Exception:
            return False

    def _ensure_startup_trade_discovery(self) -> None:
        """Queue one current official-session scan before autonomous simulation.

        This is discovery only. Its idempotency key is not a snapshot_scan key,
        so scan completion cannot auto-enter PAPER_CYCLE until discovery is
        truthfully current and the simulation authority helper succeeds.
        """
        try:
            from product.decision_simulation_gate import current_startup_id, status

            gate = dict(status() or {})
            if gate.get("discovery_ready"):
                self._ensure_decision_simulation_authority()
                return

            # A fresh persisted market scan already contains the expensive market
            # work. If learning changed the effective selection-policy/thesis
            # identity after that scan, refresh only the deterministic decision
            # projection. Key the durable job to both identities so repeated ticks
            # are idempotent and a later thesis change gets its own projection.
            scan_id = str(gate.get("scan_scanned_at") or "")
            long_term_id = str(gate.get("long_term_scanned_at") or "")
            thesis_hash = str(
                gate.get("current_thesis_hash")
                or gate.get("thesis_hash")
                or ""
            )
            if gate.get("scan_fresh") and scan_id and thesis_hash:
                self.jobs.enqueue(
                    SCH.DISCOVERY_REFRESH,
                    idempotency_key=SCH.discovery_refresh_key(
                        scan_id,
                        long_term_id,
                        thesis_hash,
                    ),
                    input_snapshot_id=scan_id,
                    critical=True,
                )
                return

            startup_id = current_startup_id()
            if not startup_id:
                return
            from product.readiness import official_history

            history = dict(official_history() or {})
            # Discovery may use the latest official session while the next archive is
            # still inside its explicit publication-grace window. Requiring
            # current here deadlocked weekend/off-session startup even though
            # the canonical freshness policy truthfully marked that history as
            # usable_for_scan. Never bypass a genuinely stale history gate.
            if not history.get("usable_for_scan"):
                return
            latest = str(
                history.get("available_session")
                or history.get("latest_date")
                or ""
            )[:10]
            if not latest:
                return
            identity = str(self.deps.active_snapshot_id() or "").strip()
            if not identity:
                identity = f"market:official_nse:{latest}"
            self.jobs.enqueue(
                SCH.MARKET_SCAN,
                idempotency_key=f"startup_discovery_scan:{startup_id}:{identity}",
                input_snapshot_id=identity,
            )
        except Exception:
            return

    @staticmethod
    def _news_bucket(now_ist, market_open: bool) -> str:
        size = 5 if market_open else 20
        minute = now_ist.hour * 60 + now_ist.minute
        bucket = minute - minute % size
        return f"{bucket // 60:02d}{bucket % 60:02d}"

    def _enqueue_closed_market_replan(
        self,
        *,
        stage: dict | None,
        next_batch: dict | None,
        session_date: str,
    ):
        """Schedule one bounded research replan for an exhausted/stalled evidence state.

        The goal is not to manufacture busy work. When historical replay has no
        runnable batch, QuantTerm gets exactly one research pass for that durable
        evidence state. If nothing material changes, the idempotency key prevents
        an infinite loop. Only a changed request/evidence state or failure reason
        creates another pass; thesis changes produced by research do not.
        """
        stage = dict(stage or {})
        nxt = dict(next_batch or {})
        reason = str(nxt.get("reason") or "historical_scheduler_stalled")
        thesis_hash = str(stage.get("thesis_hash") or "")
        if not thesis_hash:
            try:
                from product.trading_thesis import manifest as thesis_manifest

                thesis_hash = str((thesis_manifest() or {}).get("thesis_hash") or "")
            except Exception:
                thesis_hash = ""
        processed = list(stage.get("processed_sessions") or [])
        request = {}
        try:
            from research.autonomy.evidence_acquisition import open_request_for_lane

            request = dict(open_request_for_lane("HISTORICAL_REPLAY") or {})
        except Exception:
            request = {}
        request_id = str(request.get("request_id") or "")

        # A request that asks historical replay for more samples cannot remain
        # OPEN once every currently settleable historical session is consumed.
        # Close it as PLATEAUED and hand the question back to research planning.
        if request and reason == "historical_backlog_caught_up":
            try:
                from research.autonomy.evidence_progress import mark_historical_source_exhausted

                mark_historical_source_exhausted(
                    request,
                    reason=reason,
                    eligible_sessions=nxt.get("eligible_sessions"),
                    processed_sessions=nxt.get("processed_sessions", len(processed)),
                )
            except Exception as exc:
                self._incident(
                    "EVIDENCE_EXHAUSTION_WRITE_FAILED",
                    f"Could not persist historical evidence exhaustion: {type(exc).__name__}: {exc}",
                )

        state_token = json.dumps(
            {
                "phase": str(stage.get("phase") or ""),
                "batch_id": str(stage.get("batch_id") or ""),
                "processed": len(processed),
                "last_error": str(stage.get("last_error") or ""),
                "reason": reason,
                "eligible": nxt.get("eligible_sessions"),
                "sessions_total": nxt.get("sessions_total"),
            },
            sort_keys=True,
            default=str,
        )
        return self.jobs.enqueue(
            SCH.RESEARCH_CYCLE,
            idempotency_key=SCH.historical_replan_key(
                request_id=request_id,
                reason=reason,
                state_token=state_token,
            ),
            input_snapshot_id=request_id or thesis_hash or str(session_date or ""),
        )

    def enqueue_due(self, now_ist=None):
        """Keep QuantTerm productive in both live and closed-market regimes.

        Cash market / entry window:
            AUTH -> DATA_REFRESH -> MARKET_SCAN -> real PAPER_CYCLE.

        Market closed (including weekends/holidays):
            current forward-paper settlement -> learning -> research, once/session;
            plus durable historical batches:
            PIT replay -> virtual historical paper -> learning -> research -> next batch.

        Historical evidence is explicitly HISTORICAL_REPLAY and cannot satisfy
        real-forward promotion requirements.
        """
        now_ist = now_ist or self.deps.now_ist()
        holidays = self.deps.holidays()
        self._release_stale_official_blocks()

        # Never let historical backfill compete with the live cash session.
        if SCH.market_is_open(now_ist, holidays):
            if not SCH.in_scan_window(now_ist, holidays):
                return
            session_date = now_ist.date().isoformat()
            self.jobs.enqueue(
                SCH.AUTH_HEALTH,
                idempotency_key=f"auth:{session_date}",
                critical=True,
            )
            data_job = self._enqueue_daily_foundation(now_ist, session_date)
            if getattr(data_job, "status", None) == JS.SUCCEEDED:
                snap = self._snapshot_token(
                    getattr(data_job, "output_snapshot_id", None),
                    None,
                )
                self._ensure_snapshot_pipeline(snap)

            # Discovery repair is independent of broker auth. A fresh persisted
            # scan may already exist while a long-term refresh or learned-thesis
            # change invalidates only its immutable decision projection. The
            # intraday branch used to return before scheduling that repair, so a
            # broker-auth failure could leave Product Acceptance (and the real
            # desk) in SEARCHING_BEST_TRADES with an empty queue indefinitely.
            # Do not start a second intraday scan here: only repair/authorize when
            # the current persisted market scan is already fresh.
            try:
                from product.decision_simulation_gate import status as decision_status

                gate = dict(decision_status() or {})
            except Exception:
                gate = {}
            if gate.get("discovery_ready") or (
                gate.get("scan_fresh") and gate.get("scan_scanned_at")
            ):
                self._ensure_startup_trade_discovery()
            return

        # Closed market starts by finding today's best available trades from the
        # latest completed official session. Once fresh discovery is ready, the
        # supervisor automatically authorizes paper/history simulation.
        self._ensure_startup_trade_discovery()

        # Closed market: keep official completed-session data current once the
        # exchange publication window opens. These jobs never auto-chain a scan.
        if SCH.in_eod_window(now_ist, holidays):
            session_date = now_ist.date().isoformat()
            self.jobs.enqueue(
                SCH.BHAVCOPY_UPDATE,
                idempotency_key=SCH.eod_bhavcopy_key(session_date),
            )
            self.jobs.enqueue(
                SCH.DATA_REFRESH,
                idempotency_key=SCH.eod_data_refresh_key(session_date),
                critical=True,
            )

        # First preserve the genuine forward-paper learning loop once per session.
        # A BLOCKED outcome job does not prevent historical backfill from running.
        last_session = SCH.last_completed_session_date(now_ist, holidays)
        if last_session:
            self._enqueue_post_market_grind(now_ist, session_date=last_session)

        # Historical/present decision simulation is automatically authorized
        # only after fresh best-trade discovery. Existing forward positions may
        # still settle above, but no new historical learning work starts before
        # that freshness/discovery contract succeeds.
        if not self._ensure_decision_simulation_authority():
            return

        # Then run historical virtual-paper batches whenever the cash market is
        # closed. Each batch has a durable cursor and cannot silently repeat.
        try:
            from product.historical_paper_loop import pending_stage, peek_next_batch

            stage = pending_stage()
            phase = str(stage.get("phase") or "IDLE")
            batch_id = str(stage.get("batch_id") or "")

            if phase == "AWAITING_LEARNING" and batch_id:
                # The worker may have finished and persisted this phase just
                # before a process restart, before the polling job got one last
                # chance to record SUCCEEDED. State is authoritative: retire
                # that stale poll row so it can never re-run an already-finished
                # historical batch after the cursor moves on.
                try:
                    completed_poll = self.jobs.find_by_type_and_key(
                        SCH.HISTORICAL_PAPER_CYCLE,
                        SCH.historical_paper_key(batch_id),
                    )
                    if (
                        completed_poll is not None
                        and completed_poll.status in {JS.PENDING, JS.BLOCKED}
                    ):
                        self.jobs.complete(
                            completed_poll.job_id,
                            JS.SKIPPED_IDEMPOTENT,
                            result_summary="historical worker completed before supervisor poll reconciliation",
                        )
                except Exception:
                    pass
                self.jobs.enqueue(
                    SCH.LEARNING_CYCLE,
                    idempotency_key=SCH.historical_learning_key(batch_id),
                    input_snapshot_id=batch_id,
                )
                return
            if phase == "AWAITING_RESEARCH" and batch_id:
                self.jobs.enqueue(
                    SCH.RESEARCH_CYCLE,
                    idempotency_key=SCH.historical_research_key(batch_id),
                    input_snapshot_id=batch_id,
                )
                return

            # Learning/research may consume already-produced evidence, but heavy
            # historical replay yields to due/running current-market work.
            budget = self._resource_budget()
            if not budget.get("historical_replay_allowed"):
                return

            if phase == "RUNNING" and batch_id:
                self.jobs.enqueue(
                    SCH.HISTORICAL_PAPER_CYCLE,
                    idempotency_key=SCH.historical_paper_key(batch_id),
                    input_snapshot_id=batch_id,
                )
                return
            if phase == "FAILED":
                # Do not skip the failed batch or advance its cursor. But also do
                # not leave a healthy supervisor silently IDLE for hours: surface
                # the failure and allow one bounded research replan on the evidence
                # that already exists.
                self._incident(
                    "HISTORICAL_PIPELINE_FAILED",
                    f"Historical replay is failed: {stage.get('last_error') or 'unknown error'}",
                )
                self._enqueue_closed_market_replan(
                    stage=stage,
                    next_batch={
                        "available": False,
                        "reason": "historical_phase_failed",
                    },
                    session_date=last_session or now_ist.date().isoformat(),
                )
                return


            nxt = peek_next_batch()
            if nxt.get("available"):
                bid = str(nxt.get("batch_id") or "")
                if bid:
                    self.jobs.enqueue(
                        SCH.HISTORICAL_PAPER_CYCLE,
                        idempotency_key=SCH.historical_paper_key(bid),
                        input_snapshot_id=bid,
                    )
                    return

            # No runnable historical batch is still a meaningful scheduler
            # state. Give the Research Director one idempotent chance to close,
            # reframe or redirect the evidence request instead of emitting hours
            # of DATA_READY/IDLE heartbeats with no work due.
            self._enqueue_closed_market_replan(
                stage=stage,
                next_batch=nxt,
                session_date=last_session or now_ist.date().isoformat(),
            )
        except Exception as exc:
            self._incident(
                "HISTORICAL_SCHEDULER_ERROR",
                f"Closed-market historical scheduler: {type(exc).__name__}: {exc}",
            )


    _OFFICIAL_BLOCKERS = frozenset({
        JOBS.DEP_DATA, JOBS.DEP_OFFICIAL, JOBS.DEP_OUTCOME_DATA, "DATA_READY",
    })

    def _release_stale_official_blocks(self) -> None:
        """Old jobs blocked on generic DATA_READY can proceed on official bars."""
        try:
            from product.readiness import official_history

            hist = official_history()
        except Exception:
            hist = {}
        if not hist.get("current"):
            return
        for dep in self._OFFICIAL_BLOCKERS:
            try:
                self.jobs.unblock_dependency(dep)
            except Exception:
                continue

    def _official_retry_ready(self) -> bool:
        """True when official completed-session data could plausibly have landed.

        Official-data blockers are structural: the exchange has not published
        the session yet. Requeueing them on every 15-second tick re-ran the job
        thousands of times a day against a condition that cannot change until
        the next publication window.
        """
        until = float(getattr(self, "_official_retry_not_before", 0.0) or 0.0)
        return time.time() >= until

    def _defer_official_retry(self) -> None:
        try:
            delay = float(
                SCH.seconds_until_official_data_boundary(
                    self.deps.now_ist(),
                    self.deps.holidays() if hasattr(self.deps, "holidays") else None,
                )
            )
        except Exception:
            delay = 300.0
        self._official_retry_not_before = time.time() + max(60.0, delay)

    def _requeue_if_official_blocked(self, job):
        if job is None:
            return job
        if getattr(job, "status", None) == JS.BLOCKED and str(getattr(job, "blocked_on", "") or "") in self._OFFICIAL_BLOCKERS:
            if not self._official_retry_ready():
                # Still BLOCKED, truthfully, until the next publication window.
                return job
            try:
                self.jobs.requeue(job.job_id)
                self._defer_official_retry()
                return self.jobs.get(job.job_id)
            except Exception:
                return job
        return job

    def _enqueue_post_market_grind(self, now_ist=None, session_date: str | None = None) -> None:
        """Settle / learn / research after the cash session without a second scan."""
        now_ist = now_ist or self.deps.now_ist()
        holidays = self.deps.holidays()
        session_date = session_date or SCH.last_completed_session_date(now_ist, holidays)
        if not session_date:
            return
        self._release_stale_official_blocks()

        # Do not manufacture a failure while the exchange's completed-session
        # bar is not available yet. That is a structural publication wait, not
        # stale/corrupt runtime data. EOD refresh and historical learning may
        # continue; forward settlement starts only when its required session is
        # actually present.
        try:
            from product.readiness import official_history

            hist = official_history()
            available = str(
                hist.get("available_session")
                or hist.get("latest_date")
                or ""
            )[:10]
            if not available or available < str(session_date)[:10]:
                return
        except Exception:
            return

        outcome = self._requeue_if_official_blocked(self.jobs.enqueue(
            SCH.OUTCOME_RESOLUTION,
            idempotency_key=SCH.forward_outcome_key(session_date),
            critical=True,
        ))
        if getattr(outcome, "status", None) != JS.SUCCEEDED:
            return
        learning = self.jobs.enqueue(
            SCH.LEARNING_CYCLE,
            idempotency_key=SCH.forward_learning_key(session_date),
        )
        if getattr(learning, "status", None) == JS.SUCCEEDED:
            self.jobs.enqueue(
                SCH.RESEARCH_CYCLE,
                idempotency_key=SCH.forward_research_key(session_date),
            )

    def _snapshot_token(
        self,
        snapshot_id: str | None = None,
        metadata: dict | None = None,
    ) -> str:
        """Stable identity for one authoritative data state.

        Prefer the genuine snapshot id. If the canonical refresh succeeded from
        an authoritative session without a broker snapshot, derive a namespaced
        token from its latest official/live session date. This token is identity
        only; it never pretends an official session is a broker snapshot.
        """
        sid = str(snapshot_id or self.deps.active_snapshot_id() or "").strip()
        if sid:
            return sid
        meta = dict(metadata or {})
        latest = str(
            meta.get("latest_date")
            or meta.get("session_date")
            or meta.get("available_session")
            or ""
        )[:10]
        if not latest:
            return ""
        source = str(meta.get("source") or "official_market").strip().lower()
        safe_source = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in source)
        return f"market:{safe_source}:{latest}"

    def _pipeline_complete(self, snapshot_id: str) -> bool:
        snap = str(snapshot_id or "")
        owner_state = getattr(self, "owner_state", {}) or {}
        return bool(
            snap
            and snap == str(owner_state.get("completed_snapshot_id") or "")
        )

    def _mark_snapshot_complete(self, snapshot_id: str) -> None:
        snap = str(snapshot_id or "")
        if not snap:
            return
        if not hasattr(self, "owner_state") or self.owner_state is None:
            self.owner_state = {}
        now = self.deps.now_ist()
        self.owner_state["completed_snapshot_id"] = snap
        self.owner_state["completed_snapshot_at"] = (
            now.isoformat() if hasattr(now, "isoformat") else str(now)
        )
        self.owner_state["completed_session_date"] = str(now.date().isoformat())
        if hasattr(self, "_save_owner_state"):
            self._save_owner_state()
        # Retire only automatic rows. Manual controls have manual:* identities.
        try:
            self.jobs.cancel_pending_by_prefix(
                "news_refresh:",
                "market_scan:",
                "paper_cycle:",
                f"snapshot_scan:{snap}",
                f"snapshot_paper:{snap}",
                summary=f"snapshot {snap} terminal paper cycle completed",
            )
        except Exception:
            pass

    def _has_open_paper_positions(self) -> bool:
        try:
            payload = json.loads(
                logs_path("intelligence", "intel_book.json").read_text(encoding="utf-8")
            )
            return bool(payload.get("open") or payload.get("open_positions"))
        except Exception:
            return False

    def _paper_for_snapshot(self, snapshot_id: str) -> None:
        """Enqueue the terminal paper pass for an already-successful scan."""
        snap = str(snapshot_id or "")
        if not snap or self._pipeline_complete(snap):
            return
        now_ist = self.deps.now_ist()
        holidays = self.deps.holidays()
        if not SCH.entries_allowed_by_clock(now_ist, holidays):
            return
        approved = self._ensure_decision_simulation_authority()
        if not approved:
            # Existing paper positions must still be managed truthfully. This
            # distinct idempotency key can never become a new-entry pass and
            # does not mark the snapshot transaction complete.
            if self._has_open_paper_positions():
                self.jobs.enqueue(
                    SCH.PAPER_CYCLE,
                    idempotency_key=f"snapshot_manage:{snap}",
                    input_snapshot_id=snap,
                    critical=True,
                )
            return
        paper = self.jobs.enqueue(
            SCH.PAPER_CYCLE,
            idempotency_key=SCH.snapshot_paper_key(snap),
            input_snapshot_id=snap,
            critical=True,
        )
        if getattr(paper, "status", None) == JS.SUCCEEDED:
            self._mark_snapshot_complete(snap)

    def _ensure_snapshot_pipeline(self, snapshot_id: str) -> None:
        """Enqueue exactly one MARKET_SCAN for a fresh data identity."""
        snap = str(snapshot_id or "")
        if not snap or self._pipeline_complete(snap):
            return
        now_ist = self.deps.now_ist()
        holidays = self.deps.holidays()
        if not SCH.entries_allowed_by_clock(now_ist, holidays):
            return

        scan = self.jobs.enqueue(
            SCH.MARKET_SCAN,
            idempotency_key=SCH.snapshot_scan_key(snap),
            input_snapshot_id=snap,
        )
        # A recovered/reused scan may already be terminal; continue directly to
        # paper without scheduling another scan.
        if getattr(scan, "status", None) == JS.SUCCEEDED:
            self._paper_for_snapshot(snap)

    def _enqueue_paper_after_scan(self, scan_job) -> None:
        if str(getattr(scan_job, "job_type", "") or "") != SCH.MARKET_SCAN:
            return
        if not str(getattr(scan_job, "idempotency_key", "") or "").startswith("snapshot_scan:"):
            # Manual RUN_SCAN_NOW is scan-only. Only the automatic snapshot
            # transaction is authorised to continue into PAPER_CYCLE.
            return
        active_fn = getattr(self.deps, "active_snapshot_id", None)
        active = active_fn() if callable(active_fn) else ""
        snap = str(
            getattr(scan_job, "input_snapshot_id", None)
            or active
            or ""
        )
        self._paper_for_snapshot(snap)

    def _enqueue_scan_after_refresh(
        self,
        refresh_job,
        output_snapshot_id: str | None = None,
        metadata: dict | None = None,
    ) -> None:
        if str(getattr(refresh_job, "job_type", "") or "") != SCH.DATA_REFRESH:
            return
        key = str(getattr(refresh_job, "idempotency_key", "") or "")
        if not key.startswith("data_refresh:") or key.endswith(":eod"):
            # Manual REFRESH_DATA_NOW is data-only. The automatic per-session
            # DATA_REFRESH identity is the only source of the auto scan->paper chain.
            return
        snap = self._snapshot_token(
            output_snapshot_id or getattr(refresh_job, "output_snapshot_id", None),
            metadata,
        )
        self._ensure_snapshot_pipeline(snap)


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
                elif ctype == CTRL.OBSERVE_ONLY_TODAY:
                    # Operator intent only. Paper decisions and learning continue.
                    # Live money stays locked regardless.
                    today = now.date().isoformat()
                    current = str(self.owner_state.get("observe_only_date") or "")
                    self.owner_state["observe_only_date"] = "" if current == today else today
                elif ctype == CTRL.CLEAR_OBSERVE_ONLY:
                    self.owner_state["observe_only_date"] = ""
                elif ctype == CTRL.REFRESH_DATA_NOW:
                    self.jobs.enqueue(SCH.DATA_REFRESH,
                                      idempotency_key=f"manual:data:{session}", critical=True)
                elif ctype == CTRL.RUN_SCAN_NOW:
                    self.jobs.enqueue(SCH.MARKET_SCAN,
                                      idempotency_key=f"manual:scan:{snap}:{control.control_id}")
                elif ctype == CTRL.RUN_CYCLE_NOW:
                    try:
                        from product.decision_simulation_gate import approve
                        approval = dict(approve() or {})
                    except Exception as exc:
                        raise ValueError(f"decision simulation approval failed: {exc}")
                    if not approval.get("accepted"):
                        # Discovery can legitimately be between immutable identities
                        # after a new scan or an effective thesis-policy change. Keep
                        # the already-accepted operator control durable and pending;
                        # queue the prerequisite repair and consume this control only
                        # after the canonical projection is current. This prevents a
                        # one-tick race from turning an accepted request into a lost
                        # paper cycle, while still failing closed on execution.
                        self._ensure_startup_trade_discovery()
                        continue
                    self.jobs.enqueue(
                        SCH.PAPER_CYCLE,
                        idempotency_key=f"manual:cycle:{snap}:{control.control_id}",
                        critical=True,
                    )
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
                    from core.long_term_tracker import record_picks
                    record_picks([{**row, "score": row.get("combined_score"),
                                   "thesis": "; ".join(row.get("quality_factors", [])[:3])}])
                elif ctype == CTRL.RUN_LEARNING_NOW:
                    self.jobs.enqueue(SCH.LEARNING_CYCLE,
                                      idempotency_key=f"manual:learning:{control.control_id}")
                elif ctype == CTRL.RUN_HISTORICAL_REPLAY:
                    from product.autonomous_learning import maybe_run_closed_market_replay
                    maybe_run_closed_market_replay(now=now, force=True)
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


    def _desired_live_symbols(self) -> list[str]:
        ordered: list[str] = []
        seen: set[str] = set()

        def add(items) -> None:
            for raw in items or []:
                symbol = str(raw or "").upper()
                if symbol and symbol not in seen:
                    seen.add(symbol)
                    ordered.append(symbol)

        try:
            from product.scan_store import load_scan, watchlist_rows
            payload = load_scan()
            try:
                from research.autonomy.telegram_notifications import live_sniper_symbols
                add(live_sniper_symbols(payload, limit=40))
            except Exception:
                pass
            add(str(r.get("symbol", "")).upper() for r in watchlist_rows(payload, limit=60))
        except Exception:
            pass
        try:
            book = json.loads(logs_path("intelligence", "intel_book.json").read_text())
            add(str(p.get("symbol", "")).upper() for p in book.get("open", []))
        except Exception:
            pass
        return ordered[:80]

    def _manage_live_feed(self, now_ist) -> None:
        # Only the production dependency set owns a real feed. Injected tests stay network-free.
        if not isinstance(self.deps, JOBS.Deps):
            return
        if SCH.market_is_open(now_ist, self.deps.holidays()):
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
        # A dead old worker may have left RUNNING recurring work. Reclaim first,
        # then retire those rows before anything can lease them again.
        self._retire_legacy_recurring_work()
        self._retire_obsolete_data_refresh_work()
        self._process_controls()
        current = now_ist or self.deps.now_ist()
        self._manage_live_feed(current)
        if hasattr(self.deps, "drain_telegram_alerts"):
            try:
                self.deps.drain_telegram_alerts(min_interval_s=45.0)
            except Exception:
                pass
        if self.owner_state.get("halted"):
            self.heartbeat()
            return None
        self.enqueue_due(current)
        job = self.jobs.lease_due(self.owner, lease_seconds=300.0)
        self._check_overdue()
        if job is None:
            self._reconcile_idle_state()
            self.heartbeat()
            return None
        self._execute(job)
        # Non-data jobs must not leave a stale DATA_REFRESHING activity label.
        if job.job_type != SCH.DATA_REFRESH:
            self._reconcile_idle_state()
        self.heartbeat()
        return job

    def _refresh_activity_active(self) -> bool:
        """True only when a data refresh is running or actually due now.

        A future-scheduled DATA_REFRESH is the normal recurring schedule. Treating
        those PENDING rows as live activity latched DATA_REFRESHING forever.
        """
        now = float(self.clock())
        for job in self.jobs.list(limit=1000):
            if job.job_type != SCH.DATA_REFRESH:
                continue
            if job.status == JS.RUNNING:
                return True
            if job.status == JS.PENDING and float(job.scheduled_for or 0.0) <= now:
                return True
        return False

    def _reconcile_idle_state(self) -> None:
        """Repair transient labels from durable queue/failure truth.

        PAPER_ACTIVE remains a policy/capability state and is not collapsed merely
        because no job runs this instant. DATA_REFRESHING, STARTING and RESEARCHING
        are transient activity labels: once their authoritative work disappears,
        the supervisor must converge instead of leaving a stale UI/runtime claim.
        """
        if self.owner_state.get("halted"):
            return
        current = self.state.state
        activity_truth = self._activity_truth()
        activity = str(activity_truth.get("activity") or "IDLE")

        if current == ST.RESEARCHING:
            try:
                from research.autonomy.runtime_truth import research_activities
                research_active = activity in research_activities()
            except Exception:
                research_active = activity in {"HISTORICAL_REPLAY", "LEARNING", "RESEARCH"}
            if research_active:
                return
            if activity == "DATA_REFRESH":
                self._transition(
                    ST.DATA_REFRESHING,
                    "activity_reconcile",
                    "Research work ended; official-data refresh is now the authoritative active work.",
                    "activity_truth",
                )
                return
            if activity == "FORWARD_PAPER":
                self._transition(
                    ST.PAPER_ACTIVE,
                    "activity_reconcile",
                    "Research work ended; a real forward-paper cycle is now the authoritative active work.",
                    "activity_truth",
                )
                return
            self._transition(
                ST.OBSERVING,
                "activity_reconcile",
                "Research work ended; no due or running research job remains.",
                "activity_truth",
            )
            return

        if (
            activity == "FORWARD_PAPER"
            and current in (ST.OBSERVING, ST.DATA_READY, ST.DATA_REFRESHING, ST.STARTING)
        ):
            self._transition(
                ST.PAPER_ACTIVE,
                "activity_reconcile",
                "A real forward-paper cycle is the authoritative active work.",
                "activity_truth",
            )
            return

        if current not in (ST.DATA_REFRESHING, ST.STARTING):
            return
        if self._refresh_activity_active():
            if current == ST.STARTING:
                self._transition(
                    ST.DATA_REFRESHING,
                    "idle_reconcile",
                    "Boot finished and a data refresh is running or due.",
                    "idle_tick",
                )
            return
        if H.SNAPSHOT_STALE in self.failures:
            self._transition(
                ST.DATA_BLOCKED,
                "idle_reconcile",
                "No data refresh is active and the accepted market snapshot is stale.",
                "idle_tick",
            )
            return
        self._transition(
            ST.DATA_READY,
            "idle_reconcile",
            "No data refresh is active; the last accepted market data remains ready.",
            "idle_tick",
        )

    def _execute(self, job):
        handler = JOBS.HANDLERS.get(job.job_type)
        if handler is None:
            self.jobs.complete(job.job_id, JS.PERMANENT_FAILED, error_code="NO_HANDLER",
                               error_message=f"no handler for {job.job_type}")
            self._incident("NO_HANDLER", f"No handler for job {job.job_type}", job)
            return
        ctx = JOBS._Ctx(
            self.deps,
            active_failures=self.failures,
            owner_paused=self.owner_state.get("new_entries_paused", False),
            root=self.root,
            job=job,
        )
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
            if result.status in {JS.SUCCEEDED, JS.SKIPPED_IDEMPOTENT}:
                try:
                    self.incidents.recover_for_job(
                        job,
                        note=f"{job.job_type} completed with {result.status}",
                    )
                except Exception:
                    pass
            if result.status == JS.SUCCEEDED:
                for dependency in result.unblocks:
                    self.jobs.unblock_dependency(dependency)
                self._enqueue_scan_after_refresh(
                    job,
                    result.output_snapshot_id,
                    result.metadata,
                )
                self._enqueue_paper_after_scan(job)
                if (
                    job.job_type == SCH.PAPER_CYCLE
                    and str(job.idempotency_key or "").startswith("snapshot_paper:")
                ):
                    snap = str(
                        getattr(job, "input_snapshot_id", None)
                        or self.deps.active_snapshot_id()
                        or ""
                    )
                    self._mark_snapshot_complete(snap)
                key = str(job.idempotency_key or "")
                if job.job_type == SCH.LEARNING_CYCLE and key.startswith("hist_learning:"):
                    batch_id = str(getattr(job, "input_snapshot_id", None) or key.split(":", 1)[1])
                    from product.historical_paper_loop import mark_learning_complete

                    mark_learning_complete(batch_id)
                elif job.job_type == SCH.RESEARCH_CYCLE and key.startswith("hist_research:"):
                    batch_id = str(getattr(job, "input_snapshot_id", None) or key.split(":", 1)[1])
                    from product.historical_paper_loop import mark_research_complete

                    mark_research_complete(batch_id)

        target = self._gated_state(result.state_hint)
        if (
            job.job_type == SCH.PAPER_CYCLE
            and result.status == JS.SUCCEEDED
            and str(job.idempotency_key or "").startswith("snapshot_paper:")
        ):
            # The automatic snapshot transaction is terminal after one paper
            # decision pass. Keep durable positions intact, but the scheduler
            # returns to OBSERVING and waits for new data.
            target = ST.OBSERVING
        # A successful auth probe proves only broker-session health. It must not
        # move an already productive desk into OBSERVING or a fake data-refresh.
        if (
            job.job_type == SCH.AUTH_HEALTH
            and result.status == JS.SUCCEEDED
            and self.state.state in (
                ST.DATA_READY, ST.OBSERVING, ST.PAPER_ACTIVE, ST.RESEARCHING,
                ST.DEGRADED, ST.DATA_REFRESHING,
            )
        ):
            target = self.state.state
        if target and target != self.state.state:
            self._transition(target, reason=job.job_type,
                             explanation=result.summary or job.job_type, trigger=job.job_id,
                             snapshot_id=result.output_snapshot_id)

    def _gated_state(self, hint):
        if self.owner_state.get("halted"):
            return ST.HALTED
        # Official post-market work is not a broker outage. Productive hints stand.
        if hint in (ST.OBSERVING, ST.RESEARCHING, ST.PAPER_ACTIVE, ST.DATA_READY, ST.DATA_REFRESHING):
            return hint
        # Broker login is an execution exception card, not overall autonomy health.
        if hint == ST.AUTH_REQUIRED or H.AUTH_MISSING in self.failures or H.AUTH_EXPIRED in self.failures:
            current = getattr(self.state, "state", None) or ST.STARTING
            if current in (ST.HALTED, ST.DATA_BLOCKED, ST.DEGRADED):
                return current
            if current in (ST.OBSERVING, ST.RESEARCHING, ST.PAPER_ACTIVE, ST.DATA_READY, ST.DATA_REFRESHING):
                return current
            return ST.OBSERVING
        if H.SNAPSHOT_STALE in self.failures:
            return ST.DATA_BLOCKED
        return hint

    def _retry_or_fail(self, job, *, error_code, error_message, summary=""):
        if (
            job.job_type == SCH.PAPER_CYCLE
            and str(job.idempotency_key or "").startswith("snapshot_paper:")
        ):
            # Automatic paper execution is a once-per-data-identity mutation.
            # Never replay it after an exception; surface the failure and wait
            # for a new data identity or an explicit manual control.
            self.jobs.complete(
                job.job_id,
                JS.PERMANENT_FAILED,
                result_summary=summary or "automatic paper cycle failed; not retried",
                error_code=error_code,
                error_message=error_message,
            )
            return
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
        # Historical PENDING rows from while the desk was down are queue, not an
        # outage. Give this process one grace window to run them before paging.
        started = self._started_at
        if started is not None and (self.clock() - float(started)) < 3600.0:
            return
        overdue = self.jobs.overdue_critical(grace_seconds=3600.0)
        if overdue and self.state.state not in (ST.DEGRADED, ST.HALTED):
            names = ", ".join(sorted({j.job_type for j in overdue}))
            self._incident("CRITICAL_OVERDUE", f"Critical jobs overdue: {names}", overdue[0])
            self._transition(ST.DEGRADED, "critical_overdue",
                             f"A critical job is overdue: {names}.", "overdue_check")

    def _incident(self, code, message, job=None):
        """Persist one deduplicated incident dossier and append audit dialogue only on change."""
        try:
            dossier = self.incidents.upsert(
                code=str(code or ""),
                message=str(message or ""),
                job=job,
                activity_truth=self._activity_truth(),
                resource_governor=self._resource_budget(),
                active_failures=sorted(self.failures),
            )
        except Exception:
            dossier = {
                "incident_id": "",
                "occurrence_count": 1,
                "materially_changed": True,
                "recovery_action": "",
                "job": {
                    "job_type": getattr(job, "job_type", ""),
                    "job_id": getattr(job, "job_id", ""),
                },
            }

        if dossier.get("materially_changed"):
            evidence = {
                "incident_id": dossier.get("incident_id", ""),
                "error_code": code,
                "job_type": (dossier.get("job") or {}).get("job_type", ""),
                "job_id": (dossier.get("job") or {}).get("job_id", ""),
                "occurrence_count": dossier.get("occurrence_count", 1),
                "progress": dossier.get("progress") or {},
                "recovery_action": dossier.get("recovery_action", ""),
                "activity": (dossier.get("activity_truth") or {}).get("activity", ""),
                "resource_decision": (dossier.get("resource_governor") or {}).get("decision", ""),
            }
            self.dialogue.append(
                Record(
                    record_type=OPERATIONAL_INCIDENT,
                    producer="supervisor",
                    claim=message,
                    evidence=evidence,
                    decision=code,
                )
            )
            if hasattr(self.deps, "notify_incident"):
                try:
                    self.deps.notify_incident(code, message)
                except Exception:
                    pass
        return dossier

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
