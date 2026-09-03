"""First-class autonomy job handlers over QuantTerm's canonical subsystems.

Handlers are dependency-injected for deterministic tests.  Production wiring is Streamlit-free,
paper-only and never imports a broker execution path.  BLOCKED jobs name the dependency that can
requeue them; provider failure is never translated into "no opportunity".
"""
from __future__ import annotations

import csv
import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path

from research.autonomy import job_store as JS
from research.autonomy import schedules as SCH
from research.autonomy import supervisor_state as ST
from research.autonomy import health as H
from research.autonomy import auth as AUTH

DEP_AUTH = "AUTH_READY"
DEP_DATA = "DATA_READY"
DEP_OFFICIAL = "OFFICIAL_MARKET_DATA_READY"
DEP_BROKER = "BROKER_LIVE_DATA_READY"
DEP_OUTCOME_DATA = "OUTCOME_DATA_READY"
DEP_SCAN = "SCAN_READY"
DEP_OUTCOMES = "OUTCOMES_RESOLVED"
DEP_LEARNING = "LEARNING_READY"
DEP_CA_SOURCE = "CORPORATE_ACTIONS_SOURCE"
DEP_UNIVERSE_SOURCE = "UNIVERSE_HISTORY_SOURCE"


@dataclass
class JobResult:
    status: str
    summary: str = ""
    output_snapshot_id: str | None = None
    error_code: str = ""
    error_message: str = ""
    failures: set = field(default_factory=set)
    clears: set = field(default_factory=set)
    state_hint: str | None = None
    new_entries_allowed: bool = True
    blocked_on: str = ""
    dependency_version: str = ""
    unblocks: tuple[str, ...] = ()
    metadata: dict = field(default_factory=dict)


class _Ctx:
    def __init__(self, deps, *, active_failures=(), owner_paused=False, root=None):
        self.deps = deps
        self.active_failures = set(active_failures or ())
        self.owner_paused = bool(owner_paused)
        self.root = Path(root) if root else None


class Deps:
    """Production dependency wiring.  Tests inject a smaller duck-typed object."""

    def __init__(self, root=None, live_feed=None):
        from research.autonomy import default_root
        self.root = Path(root or default_root())
        self.live_feed = live_feed
        self.repo_root = Path(__file__).resolve().parents[2]
        self.logs = self.repo_root / "logs"
        self.logs.mkdir(parents=True, exist_ok=True)
        from research.autonomy.telegram_notifications import TelegramNotifier
        self.telegram = TelegramNotifier(self.root)

    def now_ist(self):
        from research.intelligence.data import nse_calendar as CAL
        return CAL._now_ist()

    def holidays(self):
        try:
            from research.intelligence.data import nse_calendar as CAL
            return CAL.load_holidays()
        except Exception:
            return set()

    def auth_health(self):
        return AUTH.probe_auth()

    def session_valid(self) -> bool:  # compatibility for existing tests/callers
        return self.auth_health().valid

    def activate(self):
        from research.intelligence.data.kite_activation import activate
        from research.intelligence.data.snapshot_store import SnapshotStore
        return activate(
            store=SnapshotStore(self.logs / "snapshots"),
            history_dir=self.logs / "kite_history",
            progress_path=self.logs / "kite_history" / "progress.json",
            start_worker=False,
            run_cycle=False,
        )

    def active_snapshot_id(self):
        try:
            from research.intelligence.data.snapshot_store import SnapshotStore
            return SnapshotStore(self.logs / "snapshots").get_active_snapshot()
        except Exception:
            return None

    def active_snapshot_info(self) -> dict:
        try:
            from research.intelligence.data.snapshot_store import SnapshotStore
            store = SnapshotStore(self.logs / "snapshots")
            sid = store.get_active_snapshot()
            if not sid:
                return {}
            manifest = json.loads((Path(store.root) / sid / "manifest.json").read_text(encoding="utf-8"))
            return {"snapshot_id": sid, "latest_date": str(manifest.get("last_trading_date") or ""),
                    "source": str(manifest.get("source") or "")}
        except Exception:
            return {}

    def refresh_instruments(self):
        from research.intelligence.data.kite_activation import KiteDataClient
        from data.fno_universe import build_fno_universe
        client = KiteDataClient.from_config()
        rows = [dict(r) for r in (list(client.instruments("NSE")) + list(client.instruments("NFO")))]
        if not rows:
            raise RuntimeError("instrument master returned no rows")
        cache = self.logs / "instruments_cache.csv"
        cache.parent.mkdir(parents=True, exist_ok=True)
        fields = sorted({str(k) for row in rows for k in row})
        tmp = cache.with_suffix(".tmp")
        with open(tmp, "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            for row in rows:
                writer.writerow({k: row.get(k, "") for k in fields})
        os.replace(tmp, cache)
        report = build_fno_universe(rows, as_of=self.now_ist().date(), source="zerodha_kite")
        exclusions_path = self.root / "instrument_exclusions.json"
        exclusions_path.write_text(json.dumps([
            {"underlying": x.underlying, "stage": x.stage, "reason": x.reason,
             "observed_at": self.now_ist().isoformat()}
            for x in report.exclusions
        ], indent=2), encoding="utf-8")
        return {"rows": len(rows), "fno_underlyings": report.mapped_underlyings,
                "exclusions": len(report.exclusions), "source": report.source}

    def update_bhavcopy(self):
        from data import bhavcopy_store as BS
        symbols = BS.build_store()
        return {"symbols": int(symbols), "ready": bool(BS.is_ready()), "source": "official_nse"}

    def corporate_actions_status(self):
        from data import corporate_actions as CA
        return CA.refresh_events()

    def universe_history_status(self):
        from data.nse_universe import refresh_universe_history
        return refresh_universe_history(as_of=self.now_ist().date())

    def live_market_ready(self):
        from data.nse_live import live_session_ready
        info = live_session_ready()
        if info.get("ready"):
            print(
                f"[KITE] latest session ready · {int(info.get('symbols') or 0)} symbols · "
                f"{info.get('source') or 'kite'} · {info.get('session_date') or ''}",
                flush=True,
            )
        return info

    def warm_indices(self):
        from data.index_store import build_index_store
        return {"indices": int(build_index_store()), "source": "official_nse"}

    def run_scan(self):
        from product.scan_progress import finish_progress, write_progress
        from scan.market_scan_service import run_whole_market_scan
        last_print = 0.0

        def progress(current, total=0, **_kw):
            nonlocal last_print
            payload = write_progress(
                current=int(current or 0),
                total=int(total or 0),
                stage="SCANNING",
                source="autonomy",
            )
            now = time.time()
            if int(current or 0) in (0, 1) or int(current or 0) == int(total or 0) or now - last_print >= 5:
                remain = str(payload.get("eta_label") or "").strip()
                extra = f" · {remain} left" if remain else ""
                print(
                    f"[SCAN] {int(current or 0)}/{int(total or 0)} · "
                    f"{payload.get('pct') or 0:.0f}%{extra}",
                    flush=True,
                )
                last_print = now

        write_progress(current=0, total=0, stage="STARTING", source="autonomy")
        try:
            report = run_whole_market_scan(
                progress_callback=progress,
                snapshot_id=str(self.active_snapshot_id() or ""),
                save=True,
            )
            payload = dict(getattr(report, "payload", {}) or {})
            summary = dict(payload.get("summary", {}) or {})
            finish_progress(
                records=len(payload.get("records") or []),
                setups=int(summary.get("with_any_setup") or 0),
            )
            return report
        except Exception:
            finish_progress(error="scan_failed")
            raise

    def notify_scan(self, payload, *, phase=""):
        sent = self.telegram.notify_scan(payload, phase=phase) or {}
        try:
            sent = dict(sent)
            sent["desk"] = self.telegram.drain_desk_alerts()
        except Exception:
            pass
        return sent

    def run_long_term_scan(self, *, refresh_fundamentals=False):
        from scan.long_term_service import run_long_term_scan
        return run_long_term_scan(refresh_fundamentals=refresh_fundamentals, save=True)

    def notify_long_term(self, payload):
        return self.telegram.notify_long_term(payload)

    def observe_live_breakouts(self):
        try:
            from product.scan_store import load_scan
            out = self.telegram.observe_live_breakouts(load_scan(), self.live_feed) or {}
            confirmed = int(out.get("confirmed") or 0)
            if confirmed > 0:
                print(f"[SNIPER] Telegram sent {confirmed} breakout alert(s)", flush=True)
            self._log_sniper(out)
            return out
        except Exception as exc:
            print(f"[SNIPER] observe failed: {type(exc).__name__}: {exc}", flush=True)
            return {"confirmed": 0}

    def _log_sniper(self, out: dict) -> None:
        now = time.time()
        last = float(getattr(self, "_sniper_log_at", 0.0) or 0.0)
        if now - last < 60.0 and int(out.get("confirmed") or 0) == 0:
            return
        self._sniper_log_at = now
        health = {}
        try:
            health = self.live_feed.health() if self.live_feed is not None else {}
        except Exception:
            health = {}
        telegram = "ON" if self.telegram.configured() else "OFF"
        err = str(health.get("last_error") or "").strip()
        if "sendMessage" in err:
            err = "websocket connecting"
        extra = f" · feed={err}" if err else ""
        reason = str(out.get("reason") or "idle")
        if reason == "no_live_ticks" and telegram == "ON":
            reason = "no_live_ticks · waiting for Kite LTP or websocket"
        print(
            f"[SNIPER] telegram {telegram} · watching {int(out.get('watching') or 0)} · "
            f"ticks {int(out.get('fresh') or health.get('symbols_ticking') or 0)} · "
            f"armed {int(out.get('armed') or 0)} · {reason}{extra}",
            flush=True,
        )

    def drain_telegram_alerts(self, *, min_interval_s: float = 45.0):
        from product.scan_store import load_scan
        from research.autonomy.telegram_notifications import sniper_symbols
        if not self.telegram.configured():
            now = time.time()
            last = float(getattr(self, "_tg_off_log_at", 0.0) or 0.0)
            if now - last >= 60.0:
                self._tg_off_log_at = now
                print("[TELEGRAM] OFF · set TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID in .env", flush=True)
            return {"setup": 0, "prebreakout": 0, "reason": "not_configured"}
        payload = load_scan()
        desk: dict = {}
        try:
            desk = self.telegram.drain_desk_alerts() or {}
        except Exception:
            desk = {}
        if not payload:
            print("[TELEGRAM] no saved scan to alert yet — keep autonomy running", flush=True)
            return {"setup": 0, "prebreakout": 0, "reason": "no_scan", "desk": desk}
        watch = sorted(sniper_symbols(payload))
        sent = self.telegram.drain_last_scan(payload, min_interval_s=min_interval_s) or {}
        sent = dict(sent)
        sent["desk"] = desk
        reason = str(sent.get("reason") or "").strip()
        quiet = reason in {"retry_wait", "already_sent", "in_progress", "no_candidates"}
        if min_interval_s == 0 or not quiet:
            print(
                f"[TELEGRAM] last-scan alerts · setups={int(sent.get('setup') or 0)} · "
                f"near-breakout={int(sent.get('prebreakout') or 0)} · "
                f"sniper-watch {len(watch)}"
                + (f" · {', '.join(watch[:8])}" if watch else " · no Watch-for-breakout/Ready names with an entry")
                + (f" · {reason}" if reason else ""),
                flush=True,
            )
        return sent

    def replay_scan_alerts(self):
        return self.drain_telegram_alerts(min_interval_s=0.0)

    def notify_online(self):
        result = self.telegram.notify_online()
        if result == "sent":
            print("[TELEGRAM] autonomy-online message sent", flush=True)
            return True
        if result == "already_sent":
            print("[TELEGRAM] autonomy-online already sent today", flush=True)
            return True
        if not self.telegram.configured():
            print("[TELEGRAM] OFF · set TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID in .env", flush=True)
            return False
        err = ""
        try:
            err = str(getattr(self.telegram._engine(), "last_error", "") or "").strip()
        except Exception:
            err = ""
        print("[TELEGRAM] autonomy-online send failed" + (f" · {err}" if err else ""), flush=True)
        return False

    def notify_incident(self, code, message):
        return self.telegram.notify_incident(code, message)

    def run_paper_cycle(self, entries_allowed: bool, entry_block_reason="",
                        session_phase="intraday", capability_failures=()):
        from research.auto_research.scheduler import get_brain
        live = self.live_feed
        brain = get_brain()
        # Intelligence cycle still manages exits / evidence. New paper entries are
        # taken from the recommendation selection authority, not strategy-registry signals.
        result = brain.run_intelligence_cycle_day(
            new_entries_allowed=False,
            entry_block_reason=("RECO_SELECTION_AUTHORITY" if entries_allowed else entry_block_reason),
            session_phase=session_phase,
            capability_failures=capability_failures,
            fresh_live_symbols=(live.fresh_symbols() if live is not None else ()),
        )
        reco = {}
        try:
            from product.paper_autopilot import run_reco_paper_cycle
            paper_on = True
            try:
                paper_on = bool(brain.is_paper_auto_enabled())
            except Exception:
                paper_on = True
            reco = run_reco_paper_cycle(
                book=brain.intel_book,
                as_of=str((result or {}).get("as_of_date") or ""),
                entries_allowed=bool(entries_allowed),
                entry_block_reason=entry_block_reason,
                session_phase=session_phase,
                paper_enabled=paper_on,
            )
            if isinstance(result, dict):
                opened = list(result.get("positions_opened") or [])
                opened.extend(list(reco.get("positions_opened") or []))
                result["positions_opened"] = opened
                result["reco_autopilot"] = {
                    "taken": reco.get("taken") or [],
                    "rejections": reco.get("rejections") or [],
                    "waits": reco.get("waits") or [],
                    "final_decision": reco.get("final_decision"),
                    "eligibility": reco.get("eligibility"),
                    "cycle_reasons": reco.get("cycle_reasons") or [],
                    "summary": reco.get("summary") or "",
                }
                if reco.get("positions_opened"):
                    result["eligibility"] = "TRADED"
                elif result.get("eligibility") in {"NO_ELIGIBLE_TRADE", "", None}:
                    result["eligibility"] = reco.get("eligibility") or result.get("eligibility")
            try:
                brain._save_intel_book()
            except Exception:
                pass
        except Exception as exc:
            if isinstance(result, dict):
                result.setdefault("reco_autopilot", {})
                result["reco_autopilot"]["error"] = str(exc)[:300]
        self.telegram.notify_paper_cycle(result, book=brain.intel_book)
        return result

    def resolve_outcomes(self, session_date: str, capability_failures=()):
        from research.auto_research.scheduler import get_brain
        brain = get_brain()
        result = brain.run_intelligence_cycle_day(
            date=session_date, new_entries_allowed=False,
            entry_block_reason="EOD_MANAGEMENT_ONLY", session_phase="eod",
            capability_failures=capability_failures,
        )
        try:
            from product.paper_learning_loop import ingest_closed_book
            learned = ingest_closed_book(brain.intel_book)
            if isinstance(result, dict):
                result["paper_learning"] = learned
        except Exception as exc:
            if isinstance(result, dict):
                result["paper_learning"] = {"error": str(exc)[:200]}
        try:
            from product.forward_soak import settle_and_report
            soak = settle_and_report(str(session_date), book=brain.intel_book)
            if isinstance(result, dict):
                result["forward_soak"] = soak
        except Exception as exc:
            if isinstance(result, dict):
                result["forward_soak"] = {"error": str(exc)[:200]}
        self.telegram.notify_paper_cycle(result, book=brain.intel_book)
        return result

    def run_learning(self, session_date: str, dialogue=None):
        from research.autonomy.research_loop import run_learning
        from research.auto_research.scheduler import get_brain
        return run_learning(get_brain(), session_date=session_date, dialogue=dialogue)

    def run_research(self, session_date: str, dialogue=None):
        from research.autonomy.research_loop import run_research_cycle
        from research.auto_research.scheduler import get_brain
        return run_research_cycle(get_brain(), session_date=session_date, dialogue=dialogue)

    def refresh_news(self):
        from news.curator_service import get_news_curator_service
        return get_news_curator_service().refresh_now()

    def news_health(self):  # legacy compatibility
        try:
            from news.curator_service import get_news_curator_service
            service = get_news_curator_service()
            return {"running": service.running, "error": service.last_error}
        except Exception as exc:
            return {"running": False, "error": str(exc)}


def _auth_health(deps):
    if hasattr(deps, "auth_health"):
        return deps.auth_health()
    valid = bool(deps.session_valid())
    return AUTH.AuthHealth(AUTH.SESSION_VALID if valid else AUTH.TOKEN_MISSING, "test")


def run_auth_health(ctx) -> JobResult:
    health = _auth_health(ctx.deps)
    if health.status == AUTH.SESSION_VALID:
        return JobResult(JS.SUCCEEDED, "AUTH_READY", clears={H.AUTH_MISSING, H.AUTH_EXPIRED,
                         H.PROVIDER_UNAVAILABLE}, state_hint=ST.DATA_REFRESHING,
                         unblocks=(DEP_AUTH,), metadata=health.as_dict())
    if health.status == AUTH.PROVIDER_UNAVAILABLE:
        return JobResult(JS.RETRYABLE_FAILED, "Zerodha provider temporarily unavailable",
                         error_code=health.error_code or "PROVIDER_UNAVAILABLE",
                         error_message=health.reason, failures={H.PROVIDER_UNAVAILABLE},
                         state_hint=ST.DEGRADED, new_entries_allowed=False)
    failure = H.AUTH_MISSING if health.status == AUTH.TOKEN_MISSING else H.AUTH_EXPIRED
    return JobResult(JS.BLOCKED, health.reason or "daily Zerodha login required",
                     error_code=health.error_code, failures={failure},
                     clears={H.PROVIDER_UNAVAILABLE}, state_hint=ST.AUTH_REQUIRED,
                     new_entries_allowed=False, blocked_on="CREDENTIAL_UPDATE",
                     metadata=health.as_dict())


def run_instrument_refresh(ctx) -> JobResult:
    if not _auth_health(ctx.deps).valid:
        return JobResult(JS.BLOCKED, "auth required before instrument refresh",
                         failures={H.AUTH_MISSING}, state_hint=ST.AUTH_REQUIRED,
                         blocked_on=DEP_AUTH, new_entries_allowed=False)
    try:
        info = ctx.deps.refresh_instruments()
        return JobResult(JS.SUCCEEDED,
                         f"instrument master current · {info.get('rows', 0)} rows · "
                         f"{info.get('fno_underlyings', 0)} F&O underlyings",
                         clears={H.AUTH_MISSING, H.AUTH_EXPIRED}, unblocks=(DEP_AUTH,), metadata=info)
    except Exception as exc:
        return JobResult(JS.RETRYABLE_FAILED, "instrument refresh failed",
                         error_code="INSTRUMENT_REFRESH_ERROR", error_message=str(exc))


def _live_market(ctx) -> dict:
    if hasattr(ctx.deps, "live_market_ready"):
        try:
            return dict(ctx.deps.live_market_ready() or {})
        except Exception:
            return {}
    return {}


def _kite_live_ready_result(ctx, *, sid=None, quality=None, live=None) -> JobResult | None:
    live = dict(live or _live_market(ctx) or {})
    if not live.get("ready"):
        return None
    quality = dict(quality or {})
    latest = str(live.get("session_date") or quality.get("latest_date") or "")
    unblocks = [DEP_DATA, DEP_CA_SOURCE, DEP_UNIVERSE_SOURCE]
    if latest:
        unblocks.append(f"EOD_DATA_READY:{latest}")
    return JobResult(
        JS.SUCCEEDED,
        f"Kite latest session ready · {int(live.get('symbols') or 0)} symbols · "
        f"{live.get('source') or 'kite_quotes'}",
        output_snapshot_id=sid,
        clears={H.SNAPSHOT_STALE, H.AUTH_MISSING, H.AUTH_EXPIRED, H.PROVIDER_UNAVAILABLE,
                H.OPTIONS_HISTORY_INCOMPLETE},
        state_hint=ST.DATA_READY,
        unblocks=tuple(unblocks),
        metadata={**quality, **live, "latest_date": latest or live.get("session_date", "")},
    )


def run_data_refresh(ctx) -> JobResult:
    if not _auth_health(ctx.deps).valid:
        return JobResult(JS.BLOCKED, "auth required before data refresh",
                         failures={H.AUTH_MISSING}, state_hint=ST.AUTH_REQUIRED,
                         blocked_on=DEP_AUTH, new_entries_allowed=False)
    try:
        report = ctx.deps.activate()
    except Exception as exc:
        kite = _kite_live_ready_result(ctx)
        if kite is not None:
            return kite
        return JobResult(JS.RETRYABLE_FAILED, "data refresh error", error_code="ACTIVATE_ERROR",
                         error_message=str(exc), failures={H.SNAPSHOT_STALE}, state_hint=ST.DATA_BLOCKED)
    ok = report.status("GENUINE_SNAPSHOT_ACTIVE") == "PASS" if hasattr(report, "status") else False
    if ok:
        sid = getattr(report, "active_pointer", None) or getattr(report, "snapshot_id", None)
        quality = dict(getattr(report, "quality", {}) or {})
        latest = ""
        date_range = quality.get("date_range")
        if isinstance(date_range, (list, tuple)) and date_range:
            latest = str(date_range[-1])
        if not latest and hasattr(ctx.deps, "active_snapshot_info"):
            latest = str((ctx.deps.active_snapshot_info() or {}).get("latest_date") or "")
        required = str(getattr(ctx, "required_session_date", "") or "")
        if required and latest < required:
            live = _live_market(ctx)
            if live.get("ready") and str(live.get("session_date") or "") >= required:
                kite = _kite_live_ready_result(ctx, sid=sid, quality={**quality, "latest_date": latest}, live=live)
                if kite is not None:
                    return kite
            return JobResult(JS.RETRYABLE_FAILED,
                             f"EOD data pending · active snapshot latest={latest or 'unknown'} · required={required}",
                             output_snapshot_id=sid, error_code="EOD_DATA_PENDING",
                             error_message="the completed session is not yet present in the active snapshot",
                             failures={H.SNAPSHOT_STALE}, state_hint=ST.DATA_REFRESHING,
                             new_entries_allowed=False, metadata={**quality, "latest_date": latest,
                                                                 "required_date": required})
        unblocks = [DEP_DATA, DEP_CA_SOURCE, DEP_UNIVERSE_SOURCE]
        if latest:
            unblocks.append(f"EOD_DATA_READY:{latest}")
        return JobResult(JS.SUCCEEDED, "genuine snapshot active", output_snapshot_id=sid,
                         clears={H.SNAPSHOT_STALE, H.AUTH_MISSING, H.AUTH_EXPIRED,
                                 H.PROVIDER_UNAVAILABLE, H.OPTIONS_HISTORY_INCOMPLETE}, state_hint=ST.DATA_READY,
                         unblocks=tuple(unblocks), metadata={**quality, "latest_date": latest})
    kite = _kite_live_ready_result(
        ctx,
        sid=getattr(report, "active_pointer", None) or getattr(report, "snapshot_id", None),
        quality=dict(getattr(report, "quality", {}) or {}),
    )
    if kite is not None:
        return kite
    blocker = getattr(report, "blocker", "") or "snapshot not forward-eligible"
    return JobResult(JS.BLOCKED, f"data not ready: {blocker}", failures={H.SNAPSHOT_STALE},
                     state_hint=ST.DATA_BLOCKED, new_entries_allowed=False,
                     blocked_on=DEP_AUTH if "session" in blocker.lower() else "DATA_SOURCE")


def run_bhavcopy_update(ctx) -> JobResult:
    try:
        info = ctx.deps.update_bhavcopy()
    except Exception as exc:
        return JobResult(JS.RETRYABLE_FAILED, "official bhavcopy update failed",
                         error_code="BHAVCOPY_ERROR", error_message=str(exc))
    if not info.get("ready"):
        return JobResult(JS.BLOCKED, "official bhavcopy history is not yet sufficient",
                         blocked_on="BHAVCOPY_SOURCE", metadata=info)
    universe = {}
    if hasattr(ctx.deps, "universe_history_status"):
        try:
            universe = dict(ctx.deps.universe_history_status() or {})
        except Exception:
            universe = {}
    clears = set()
    if universe.get("survivorship_complete"):
        clears.add(H.UNIVERSE_INCOMPLETE)
    return JobResult(JS.SUCCEEDED, f"official bhavcopy ready · {info.get('symbols', 0)} symbols",
                     clears=clears, unblocks=(DEP_UNIVERSE_SOURCE,),
                     metadata={**info, "universe": universe})


def run_corporate_actions(ctx) -> JobResult:
    try:
        info = ctx.deps.corporate_actions_status()
    except Exception as exc:
        return JobResult(JS.RETRYABLE_FAILED, "corporate-action validation failed",
                         error_code="CA_VALIDATION_ERROR", error_message=str(exc),
                         failures={H.CA_INCOMPLETE})
    if not info.get("available"):
        return JobResult(JS.BLOCKED,
                         "official corporate-action table unavailable; affected historical research remains blocked",
                         failures={H.CA_INCOMPLETE}, blocked_on=DEP_CA_SOURCE, metadata=info)
    return JobResult(JS.SUCCEEDED, f"corporate actions loaded · {info.get('symbols', 0)} symbols",
                     clears={H.CA_INCOMPLETE, H.OPTIONS_HISTORY_INCOMPLETE}, metadata=info)


def run_universe_history(ctx) -> JobResult:
    try:
        info = ctx.deps.universe_history_status()
    except Exception as exc:
        return JobResult(JS.RETRYABLE_FAILED, "universe-history validation failed",
                         error_code="UNIVERSE_HISTORY_ERROR", error_message=str(exc),
                         failures={H.UNIVERSE_INCOMPLETE})
    if not info.get("survivorship_complete"):
        return JobResult(JS.BLOCKED, info.get("note") or "point-in-time universe unavailable",
                         failures={H.UNIVERSE_INCOMPLETE}, blocked_on=DEP_UNIVERSE_SOURCE,
                         metadata=info)
    return JobResult(JS.SUCCEEDED, f"point-in-time universe ready · {len(info.get('symbols', []))} symbols",
                     clears={H.UNIVERSE_INCOMPLETE}, metadata=info)


def run_index_warmup(ctx) -> JobResult:
    try:
        info = ctx.deps.warm_indices()
    except Exception as exc:
        return JobResult(JS.RETRYABLE_FAILED, "index warm-up failed", error_code="INDEX_WARMUP_ERROR",
                         error_message=str(exc))
    if int(info.get("indices", 0)) <= 0:
        return JobResult(JS.RETRYABLE_FAILED, "official index store unavailable",
                         error_code="INDEX_DATA_UNAVAILABLE")
    return JobResult(JS.SUCCEEDED, f"index store warm · {info.get('indices')} indices", metadata=info)


def _run_long_term(ctx, *, refresh_fundamentals: bool) -> JobResult:
    try:
        report = ctx.deps.run_long_term_scan(refresh_fundamentals=refresh_fundamentals)
    except Exception as exc:
        return JobResult(JS.RETRYABLE_FAILED, "long-term scan failed",
                         error_code="LONG_TERM_SCAN_ERROR", error_message=str(exc))
    if hasattr(report, "ok") and not report.ok:
        if getattr(report, "status", "") == "NO_CANDIDATES":
            payload = getattr(report, "payload", {}) or {}
        else:
            return JobResult(JS.RETRYABLE_FAILED, "long-term scan unavailable",
                             error_code=getattr(report, "error_code", "LONG_TERM_SCAN_ERROR"),
                             error_message=getattr(report, "error_message", ""))
    else:
        payload = getattr(report, "payload", report or {}) or {}
    summary = dict(payload.get("summary", {}) or {})
    telegram = {}
    if hasattr(ctx.deps, "notify_long_term"):
        try:
            telegram = ctx.deps.notify_long_term(payload) or {}
        except Exception:
            telegram = {"error": "notification_failed"}
    return JobResult(
        JS.SUCCEEDED,
        (f"long-term scan complete · {summary.get('quality_compounder', 0)} compounders · "
         f"{summary.get('garp_candidate', 0)} GARP · {summary.get('coverage_pct', 0)}% covered"),
        metadata={**summary, "telegram": telegram,
                  "fundamentals_refreshed": bool(refresh_fundamentals)},
    )


def run_long_term_scan_job(ctx) -> JobResult:
    return _run_long_term(ctx, refresh_fundamentals=False)


def run_long_term_refresh_job(ctx) -> JobResult:
    return _run_long_term(ctx, refresh_fundamentals=True)


def _official_ready(ctx=None) -> dict:
    if ctx is not None and hasattr(ctx.deps, "official_history"):
        try:
            return dict(ctx.deps.official_history() or {})
        except Exception:
            return {"current": False}
    try:
        from product.readiness import official_history

        return official_history()
    except Exception:
        return {"current": False}


def run_market_scan(ctx) -> JobResult:
    snap = ctx.deps.active_snapshot_id()
    official = _official_ready(ctx)
    live = {} if snap or official.get("current") else _live_market(ctx)
    if not snap and not official.get("current") and not live.get("ready"):
        return JobResult(JS.BLOCKED, "official completed-session history required before market scan",
                         failures={H.SNAPSHOT_STALE}, blocked_on=DEP_OFFICIAL,
                         state_hint=ST.DATA_BLOCKED, new_entries_allowed=False)
    if not snap and live.get("ready"):
        print(
            f"[SCAN] using latest Kite session · {int(live.get('symbols') or 0)} symbols · "
            f"{live.get('source') or 'kite_quotes'}",
            flush=True,
        )
    try:
        report = ctx.deps.run_scan()
    except Exception as exc:
        return JobResult(JS.RETRYABLE_FAILED, "scan failed to load market data",
                         error_code="SCAN_ERROR", error_message=str(exc), state_hint=ST.OBSERVING)
    if hasattr(report, "ok"):
        if not report.ok:
            return JobResult(JS.RETRYABLE_FAILED, "whole-market scan failed",
                             error_code=report.error_code or "SCAN_ERROR",
                             error_message=report.error_message, state_hint=ST.OBSERVING)
        payload = report.payload
    else:
        payload = report or {}
    summary = dict(payload.get("summary", {}))
    n = int(summary.get("with_any_setup", 0) or 0)
    telegram = {}
    if hasattr(ctx.deps, "notify_scan"):
        try:
            phase = SCH.session_phase(ctx.deps.now_ist(), ctx.deps.holidays())
            telegram = ctx.deps.notify_scan(payload, phase=phase) or {}
            print(
                f"[TELEGRAM] scan alerts · setups={int(telegram.get('setup') or 0)} · "
                f"near-breakout={int(telegram.get('prebreakout') or 0)}"
                + (f" · {telegram.get('reason')}" if telegram.get("reason") else ""),
                flush=True,
            )
        except Exception:
            telegram = {"error": "notification_failed"}
            print("[TELEGRAM] scan alert send failed", flush=True)
    return JobResult(JS.SUCCEEDED,
                     f"scan complete · {n} setups · {summary.get('momentum', 0)} momentum",
                     state_hint=ST.OBSERVING, unblocks=(DEP_SCAN,),
                     metadata={**summary, "telegram": telegram})


def _entry_reason(now, holidays, ctx) -> tuple[bool, str, str]:
    phase = SCH.session_phase(now, holidays)
    if not SCH.entries_allowed_by_clock(now, holidays):
        return False, "ENTRY_WINDOW_CLOSED", phase
    caps = H.capabilities(ctx.active_failures | ({H.OWNER_PAUSED} if ctx.owner_paused else set()))
    if caps["new_paper_entries"] == H.BLOCKED:
        return False, "CAPABILITY_BLOCKED", phase
    return True, "", phase


def run_paper_cycle(ctx) -> JobResult:
    now = ctx.deps.now_ist()
    holidays = ctx.deps.holidays()
    entries_ok, reason, phase = _entry_reason(now, holidays, ctx)
    if not ctx.deps.active_snapshot_id():
        # Still consume recommendations and persist BLOCKED_BROKER intents.
        entries_ok = False
        reason = reason or "BROKER_LOGIN_REQUIRED"
    try:
        try:
            result = ctx.deps.run_paper_cycle(entries_ok, reason, phase, ctx.active_failures)
        except TypeError:  # legacy injected fakes
            result = ctx.deps.run_paper_cycle(entries_ok)
    except Exception as exc:
        return JobResult(JS.RETRYABLE_FAILED, "paper cycle error", error_code="CYCLE_ERROR",
                         error_message=str(exc))
    eligibility = (result or {}).get("eligibility", "")
    hint = ST.PAPER_ACTIVE if entries_ok else ST.OBSERVING
    metadata = {"eligibility": eligibility, "entry_block_reason": reason,
                "session_phase": phase}
    if not os.environ.get("PYTEST_CURRENT_TEST"):
        try:
            from product.paper_self_feed import ingest_paper_cycle

            feed = ingest_paper_cycle(
                result or {},
                as_of=now.date().isoformat(),
                slot=phase,
            )
            metadata["self_feed"] = {
                "taken": len(feed.get("taken") or []),
                "skipped": len(feed.get("skipped") or []),
                "sepa_best": len(feed.get("sepa_best") or []),
                "candidate_tests": len(feed.get("candidate_tests") or []),
            }
        except Exception:
            pass
    return JobResult(JS.SUCCEEDED, f"paper cycle: {eligibility or 'no-op'}",
                     state_hint=hint, new_entries_allowed=entries_ok,
                     metadata=metadata)


def run_outcome_resolution(ctx) -> JobResult:
    now = ctx.deps.now_ist()
    holidays = ctx.deps.holidays() if hasattr(ctx.deps, "holidays") else None
    session_date = SCH.last_completed_session_date(now, holidays) or now.date().isoformat()
    official = _official_ready(ctx)
    available = str(official.get("available_session") or official.get("latest_date") or "")[:10]
    if not official.get("current") and (not available or available < session_date):
        return JobResult(
            JS.BLOCKED,
            f"official completed-session bars required before outcome resolution ({available or 'none'} < {session_date})",
            blocked_on=DEP_OUTCOME_DATA,
            failures={H.SNAPSHOT_STALE},
        )
    result: dict = {}
    try:
        if hasattr(ctx.deps, "resolve_outcomes"):
            result = ctx.deps.resolve_outcomes(session_date, ctx.active_failures) or {}
        else:
            result = ctx.deps.run_paper_cycle(False) or {}
    except Exception as exc:
        result = {"paper_book_error": str(exc)[:240]}
    official_settle: dict = {}
    if not os.environ.get("PYTEST_CURRENT_TEST"):
        try:
            from product.autonomous_loop import advance_loop

            official_settle = advance_loop(trigger="outcome_resolution")
        except Exception as exc:
            official_settle = {"error": str(exc)[:240]}
        try:
            from product.paper_self_feed import ingest_paper_cycle

            ingest_paper_cycle(result or {}, as_of=session_date, slot="eod")
        except Exception:
            pass
    else:
        try:
            from product.autonomous_loop import settle_official_outcomes

            official_settle = settle_official_outcomes(session_date)
        except Exception as exc:
            official_settle = {"error": str(exc)[:240]}
    if isinstance(result, dict):
        result["official_settlement"] = official_settle
    closed = len((result or {}).get("positions_closed", []))
    recorded = len((result or {}).get("outcomes_recorded", []))
    matured = int((official_settle or {}).get("n_settled") or 0)
    return JobResult(
        JS.SUCCEEDED,
        f"outcomes resolved · {closed} book closes · {recorded} decoded · {matured} official",
        unblocks=(f"{DEP_OUTCOMES}:{session_date}",),
        metadata=result or {},
    )


def run_learning_cycle(ctx) -> JobResult:
    now = ctx.deps.now_ist()
    holidays = ctx.deps.holidays() if hasattr(ctx.deps, "holidays") else None
    session_date = SCH.last_completed_session_date(now, holidays) or now.date().isoformat()
    try:
        result = ctx.deps.run_learning(session_date, getattr(ctx, "dialogue", None)) or {}
    except Exception as exc:
        return JobResult(JS.RETRYABLE_FAILED, "learning cycle failed", error_code="LEARNING_ERROR",
                         error_message=str(exc), failures={H.LEARNING_FAILED}, state_hint=ST.DEGRADED)
    if not os.environ.get("PYTEST_CURRENT_TEST"):
        try:
            from product.autonomous_loop import consume_learning_memory

            result["settled_memory"] = consume_learning_memory(session_date)
        except Exception as exc:
            result["settled_memory"] = {"error": str(exc)[:200]}
    return JobResult(JS.SUCCEEDED,
                     f"learning complete · {result.get('diagnostics', 0)} diagnostics · "
                     f"{result.get('paper_closed', 0)} paper trades · "
                     f"{result.get('paper_cooldown', 0)} cooldown · "
                     f"{result.get('paper_prefer', 0)} preferred",
                     clears={H.LEARNING_FAILED}, state_hint=ST.RESEARCHING,
                     unblocks=(f"{DEP_LEARNING}:{session_date}",), metadata=result)


def run_research_cycle(ctx) -> JobResult:
    session_date = ctx.deps.now_ist().date().isoformat()
    try:
        result = ctx.deps.run_research(session_date, getattr(ctx, "dialogue", None))
    except Exception as exc:
        return JobResult(JS.RETRYABLE_FAILED, "research cycle failed", error_code="RESEARCH_ERROR",
                         error_message=str(exc), failures={H.LEARNING_FAILED}, state_hint=ST.DEGRADED)
    return JobResult(JS.SUCCEEDED, f"research cycle: {result.get('decision', 'no action')}",
                     clears={H.LEARNING_FAILED}, state_hint=ST.OBSERVING, metadata=result)


def run_news_refresh(ctx) -> JobResult:
    try:
        if hasattr(ctx.deps, "refresh_news"):
            report = ctx.deps.refresh_news() or {}
            failed = str(report.get("status", "")).upper() == "ERROR"
            error = report.get("error", "")
        else:
            health = ctx.deps.news_health()
            failed = not health.get("running")
            error = health.get("error", "")
    except Exception as exc:
        failed, error = True, str(exc)
    if failed:
        return JobResult(JS.SUCCEEDED, f"news unavailable: {error}", failures={H.NEWS_UNAVAILABLE})
    return JobResult(JS.SUCCEEDED, "news refresh complete", clears={H.NEWS_UNAVAILABLE})


# Compatibility name retained for older tests.
run_news_health = run_news_refresh

HANDLERS = {
    SCH.AUTH_HEALTH: run_auth_health,
    SCH.INSTRUMENT_REFRESH: run_instrument_refresh,
    SCH.DATA_REFRESH: run_data_refresh,
    SCH.BHAVCOPY_UPDATE: run_bhavcopy_update,
    SCH.CORPORATE_ACTIONS: run_corporate_actions,
    SCH.UNIVERSE_HISTORY: run_universe_history,
    SCH.INDEX_WARMUP: run_index_warmup,
    SCH.MARKET_SCAN: run_market_scan,
    SCH.NEWS_REFRESH: run_news_refresh,
    SCH.PAPER_CYCLE: run_paper_cycle,
    SCH.OUTCOME_RESOLUTION: run_outcome_resolution,
    SCH.LEARNING_CYCLE: run_learning_cycle,
    SCH.RESEARCH_CYCLE: run_research_cycle,
    SCH.LONG_TERM_SCAN: run_long_term_scan_job,
    SCH.LONG_TERM_REFRESH: run_long_term_refresh_job,
}
