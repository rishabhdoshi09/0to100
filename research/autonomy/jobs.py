"""First-class autonomy job handlers over QuantTerm's canonical subsystems.

Handlers are dependency-injected for deterministic tests.  Production wiring is Streamlit-free,
paper-only and never imports a broker execution path.  BLOCKED jobs name the dependency that can
requeue them; provider failure is never translated into "no opportunity".
"""
from __future__ import annotations

import csv
import json
import os
from dataclasses import dataclass, field
from pathlib import Path

from research.autonomy import job_store as JS
from research.autonomy import schedules as SCH
from research.autonomy import supervisor_state as ST
from research.autonomy import health as H
from research.autonomy import auth as AUTH

DEP_AUTH = "AUTH_READY"
DEP_DATA = "DATA_READY"
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

    def certify_bhav_snapshot(self, *, activate: bool = True):
        """Pin an immutable snapshot from the official bhav store (Kite-independent)."""
        from research.intelligence.data.from_bhav import snapshot_from_bhav_store
        from research.intelligence.data.snapshot_store import SnapshotStore

        store = SnapshotStore(self.logs / "snapshots")
        sid, report = snapshot_from_bhav_store(
            store,
            activate=activate,
            actor="autonomy",
            reason="bhav store certification",
        )
        latest = ""
        if sid:
            try:
                latest = str(store.open_snapshot(sid).manifest.get("last_trading_date") or "")
            except Exception:
                latest = ""
        return {
            "snapshot_id": sid,
            "activated": bool(report.get("activated")),
            "accepted": int(report.get("accepted") or 0),
            "symbols": int(report.get("symbols") or 0),
            "result": report.get("result"),
            "source": "bhav_store",
            "latest_date": latest,
        }

    def corporate_actions_status(self):
        from data import corporate_actions as CA

        return CA.ledger_status()

    def ensure_corporate_actions(self):
        """Validate existing ledger or ingest an operator-supplied source.

        Never invents CA events from price gaps. Looks for:
          1. existing logs/ca_events.json
          2. QT_CA_SOURCE_FILE
          3. logs/ca_events.incoming.json / .csv
        """
        from data import corporate_actions as CA

        status = CA.ledger_status()
        if status.get("available") and int(status.get("events") or 0) > 0:
            status["ingested"] = False
            status["reason"] = "ledger_present"
            return status

        candidates = []
        env_src = os.getenv("QT_CA_SOURCE_FILE", "").strip()
        if env_src:
            candidates.append(Path(env_src))
        candidates.extend([
            self.logs / "ca_events.incoming.json",
            self.logs / "ca_events.incoming.csv",
        ])
        for src in candidates:
            if src.exists():
                info = CA.ingest_from_path(src)
                info["ingested"] = True
                info["reason"] = f"ingested:{src.name}"
                return info
        status["ingested"] = False
        status["reason"] = "source_missing"
        return status

    def universe_history_status(self):
        from data.nse_universe import point_in_time_universe

        return point_in_time_universe(self.now_ist().date())

    def ensure_universe_history(self):
        """Prefer an official/operator archive; only then bootstrap from bhav.

        Never invents membership. Never overwrites a research-grade ledger with
        bhav-inferred rows. Looks for:
          1. existing research-grade ledger
          2. QT_UNIVERSE_SOURCE_FILE
          3. logs/universe_history.incoming.json / .csv
          4. existing inferred ledger (keep)
          5. bhav bootstrap
        """
        from data import universe_history as UH

        status = UH.ledger_status()
        if status.get("research_grade"):
            status["built"] = False
            status["ingested"] = False
            status["reason"] = "ledger_present_research_grade"
            return status

        candidates = []
        env_src = os.getenv("QT_UNIVERSE_SOURCE_FILE", "").strip()
        if env_src:
            candidates.append(Path(env_src))
        candidates.extend([
            self.logs / "universe_history.incoming.json",
            self.logs / "universe_history.incoming.csv",
        ])
        for src in candidates:
            if src.exists():
                info = UH.ingest_from_path(src)
                info["built"] = False
                info["ingested"] = True
                info["reason"] = f"ingested:{src.name}"
                return info

        if status.get("survivorship_complete"):
            status["built"] = False
            status["ingested"] = False
            status["reason"] = status.get("reason") or "ledger_present_inferred"
            return status
        return UH.build_from_bhav(force=False)

    def options_eod_status(self):
        from options.eod_store import store_status

        return store_status()

    def capture_options_eod(self, symbols=None):
        from options.eod_snapshot import capture_universe

        return capture_universe(symbols)

    def warm_indices(self):
        from data.index_store import build_index_store
        return {"indices": int(build_index_store()), "source": "official_nse"}

    def run_scan(self):
        from scan.market_scan_service import run_whole_market_scan
        return run_whole_market_scan(snapshot_id=str(self.active_snapshot_id() or ""))

    def notify_scan(self, payload, *, phase=""):
        return self.telegram.notify_scan(payload, phase=phase)

    def run_long_term_scan(self, *, refresh_fundamentals=False):
        from scan.long_term_service import run_long_term_scan
        return run_long_term_scan(refresh_fundamentals=refresh_fundamentals, save=True)

    def notify_long_term(self, payload):
        return self.telegram.notify_long_term(payload)

    def observe_live_breakouts(self):
        try:
            from product.scan_store import load_scan
            return self.telegram.observe_live_breakouts(load_scan(), self.live_feed)
        except Exception:
            return {"confirmed": 0}

    def notify_online(self):
        return self.telegram.notify_online()

    def notify_incident(self, code, message):
        return self.telegram.notify_incident(code, message)

    def run_paper_cycle(self, entries_allowed: bool, entry_block_reason="",
                        session_phase="intraday", capability_failures=()):
        from research.auto_research.scheduler import get_brain
        live = self.live_feed
        brain = get_brain()
        result = brain.run_intelligence_cycle_day(
            new_entries_allowed=entries_allowed,
            entry_block_reason=entry_block_reason,
            session_phase=session_phase,
            capability_failures=capability_failures,
            fresh_live_symbols=(live.fresh_symbols() if live is not None else ()),
        )
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


def _kite_login_optional(ctx) -> bool:
    now = ctx.deps.now_ist()
    holidays = ctx.deps.holidays()
    return SCH.kite_login_optional(now, holidays)


def run_auth_health(ctx) -> JobResult:
    health = _auth_health(ctx.deps)
    if health.status == AUTH.SESSION_VALID:
        return JobResult(JS.SUCCEEDED, "AUTH_READY", clears={H.AUTH_MISSING, H.AUTH_EXPIRED,
                         H.PROVIDER_UNAVAILABLE}, state_hint=ST.DATA_REFRESHING,
                         unblocks=(DEP_AUTH,), metadata=health.as_dict())
    if _kite_login_optional(ctx):
        return JobResult(
            JS.SUCCEEDED,
            "Zerodha login optional outside session window · bhavcopy scans continue",
            clears={H.AUTH_MISSING, H.AUTH_EXPIRED, H.PROVIDER_UNAVAILABLE},
            state_hint=ST.OBSERVING,
            metadata={**health.as_dict(), "kite_deferred": True},
        )
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
        if _kite_login_optional(ctx):
            return JobResult(
                JS.SUCCEEDED,
                "instrument refresh deferred until auth window",
                clears={H.AUTH_MISSING, H.AUTH_EXPIRED},
                metadata={"kite_deferred": True},
            )
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


def _low_power_mode() -> bool:
    return str(os.getenv("QT_LOW_POWER", "") or "").strip() in {"1", "true", "TRUE", "yes"}


def _reuse_existing_snapshot(ctx) -> JobResult | None:
    """Avoid multi-minute Kite daily_refresh when a usable snapshot already exists."""
    info = {}
    if hasattr(ctx.deps, "active_snapshot_info"):
        try:
            info = dict(ctx.deps.active_snapshot_info() or {})
        except Exception:
            info = {}
    sid = str(info.get("snapshot_id") or "") or None
    if not sid and hasattr(ctx.deps, "active_snapshot_id"):
        try:
            sid = ctx.deps.active_snapshot_id()
        except Exception:
            sid = None
    if not sid:
        return None
    latest = str(info.get("latest_date") or "")
    unblocks = [DEP_DATA]
    if latest:
        unblocks.append(f"EOD_DATA_READY:{latest}")
    return JobResult(
        JS.SUCCEEDED,
        f"existing snapshot reused · {sid}"
        + (" · low-power mode skips full Kite history refresh" if _low_power_mode() else ""),
        output_snapshot_id=str(sid),
        metadata={**info, "reused_existing": True, "low_power": _low_power_mode()},
        state_hint=ST.DATA_READY,
        clears={H.PROVIDER_UNAVAILABLE, H.SNAPSHOT_STALE, H.AUTH_MISSING, H.AUTH_EXPIRED},
        unblocks=tuple(unblocks),
    )


def run_data_refresh(ctx) -> JobResult:
    # Low-power Macs: never block the supervisor for 10–20 minutes on Kite
    # daily_refresh. Reuse an active snapshot, else fall through to bhav/activate.
    if _low_power_mode():
        reused = _reuse_existing_snapshot(ctx)
        if reused is not None:
            return reused

    if not _auth_health(ctx.deps).valid:
        if _kite_login_optional(ctx):
            try:
                info = ctx.deps.update_bhavcopy()
            except Exception as exc:
                return JobResult(
                    JS.RETRYABLE_FAILED,
                    "bhavcopy refresh failed while Kite deferred",
                    error_code="BHAVCOPY_ERROR",
                    error_message=str(exc),
                )
            if info.get("ready"):
                snap = {}
                certify = getattr(ctx.deps, "certify_bhav_snapshot", None)
                if callable(certify):
                    try:
                        snap = certify(activate=True) or {}
                    except Exception as exc:
                        snap = {"error": str(exc), "snapshot_id": None}
                sid = snap.get("snapshot_id")
                if sid:
                    latest = str(snap.get("latest_date") or "")
                    unblocks = [DEP_DATA]
                    if latest:
                        unblocks.append(f"EOD_DATA_READY:{latest}")
                    return JobResult(
                        JS.SUCCEEDED,
                        "official bhavcopy ready · verified snapshot certified from bhav "
                        f"(Kite login still deferred) · {sid}",
                        output_snapshot_id=sid,
                        metadata={**info, **snap, "kite_deferred": True},
                        state_hint=ST.DATA_READY,
                        clears={H.PROVIDER_UNAVAILABLE, H.SNAPSHOT_STALE},
                        unblocks=tuple(unblocks),
                    )
                return JobResult(
                    JS.SUCCEEDED,
                    "official bhavcopy ready · Kite snapshot deferred until login window "
                    "(bhav certification pending)",
                    metadata={**info, **snap, "kite_deferred": True},
                    state_hint=ST.OBSERVING,
                    clears={H.PROVIDER_UNAVAILABLE},
                )
            return JobResult(
                JS.BLOCKED,
                "bhavcopy not ready; Kite login deferred until next session window",
                blocked_on="BHAVCOPY_SOURCE",
                metadata=info,
                state_hint=ST.DATA_BLOCKED,
            )
        return JobResult(JS.BLOCKED, "auth required before data refresh",
                         failures={H.AUTH_MISSING}, state_hint=ST.AUTH_REQUIRED,
                         blocked_on=DEP_AUTH, new_entries_allowed=False)
    try:
        report = ctx.deps.activate()
    except Exception as exc:
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
            return JobResult(JS.RETRYABLE_FAILED,
                             f"EOD data pending · active snapshot latest={latest or 'unknown'} · required={required}",
                             output_snapshot_id=sid, error_code="EOD_DATA_PENDING",
                             error_message="the completed session is not yet present in the active snapshot",
                             failures={H.SNAPSHOT_STALE}, state_hint=ST.DATA_REFRESHING,
                             new_entries_allowed=False, metadata={**quality, "latest_date": latest,
                                                                 "required_date": required})
        unblocks = [DEP_DATA]
        if latest:
            unblocks.append(f"EOD_DATA_READY:{latest}")
        return JobResult(JS.SUCCEEDED, "genuine snapshot active", output_snapshot_id=sid,
                         clears={H.SNAPSHOT_STALE, H.AUTH_MISSING, H.AUTH_EXPIRED,
                                 H.PROVIDER_UNAVAILABLE}, state_hint=ST.DATA_READY,
                         unblocks=tuple(unblocks), metadata={**quality, "latest_date": latest})
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
    snap = {}
    certify = getattr(ctx.deps, "certify_bhav_snapshot", None)
    if callable(certify):
        try:
            snap = certify(activate=True) or {}
        except Exception as exc:
            snap = {"error": str(exc)}
    sid = snap.get("snapshot_id")
    if sid:
        latest = str(snap.get("latest_date") or "")
        unblocks = [DEP_DATA]
        if latest:
            unblocks.append(f"EOD_DATA_READY:{latest}")
        return JobResult(
            JS.SUCCEEDED,
            f"official bhavcopy ready · {info.get('symbols', 0)} symbols · snapshot {sid}",
            output_snapshot_id=sid,
            metadata={**info, **snap},
            clears={H.SNAPSHOT_STALE},
            unblocks=tuple(unblocks),
            state_hint=ST.DATA_READY,
        )
    return JobResult(JS.SUCCEEDED, f"official bhavcopy ready · {info.get('symbols', 0)} symbols",
                     metadata={**info, **snap})


def run_corporate_actions(ctx) -> JobResult:
    try:
        ensure = getattr(ctx.deps, "ensure_corporate_actions", None)
        info = ensure() if callable(ensure) else ctx.deps.corporate_actions_status()
    except Exception as exc:
        return JobResult(JS.RETRYABLE_FAILED, "corporate-action validation failed",
                         error_code="CA_VALIDATION_ERROR", error_message=str(exc),
                         failures={H.CA_INCOMPLETE})
    if not info.get("available"):
        # Operator-pending source — research checklist tracks the gap. Do NOT sticky-fail
        # the heartbeat: QuantTerm refuses to invent CA events, so BLOCKED is expected.
        return JobResult(JS.BLOCKED,
                         "official corporate-action table unavailable; drop "
                         "logs/ca_events.incoming.json (or set QT_CA_SOURCE_FILE) "
                         "with real NSE CA events — QuantTerm will not invent them. "
                         "Or: python main.py ca-ingest --from-gaps",
                         clears={H.CA_INCOMPLETE}, blocked_on=DEP_CA_SOURCE, metadata=info)
    if int(info.get("events") or info.get("symbols") or 0) <= 0:
        return JobResult(JS.BLOCKED,
                         "corporate-action ledger is present but empty; research stays RAW. "
                         "Fill logs/ca_events.todo.csv from NSE filings, then ca-ingest.",
                         clears={H.CA_INCOMPLETE}, blocked_on=DEP_CA_SOURCE, metadata=info)
    return JobResult(JS.SUCCEEDED,
                     f"corporate actions loaded · {info.get('symbols', 0)} symbols / "
                     f"{info.get('events', info.get('symbols', 0))} events",
                     clears={H.CA_INCOMPLETE}, metadata=info)


def run_universe_history(ctx) -> JobResult:
    try:
        ensure = getattr(ctx.deps, "ensure_universe_history", None)
        info = ensure() if callable(ensure) else ctx.deps.universe_history_status()
        # Re-read PIT view so job metadata matches consumers.
        status = ctx.deps.universe_history_status()
        info = {**status, **{k: v for k, v in info.items() if k not in status}}
    except Exception as exc:
        return JobResult(JS.RETRYABLE_FAILED, "universe-history validation failed",
                         error_code="UNIVERSE_HISTORY_ERROR", error_message=str(exc),
                         failures={H.UNIVERSE_INCOMPLETE})
    if not info.get("survivorship_complete"):
        # Survivors-only / empty bhav bootstrap is an acknowledged research limit,
        # not a sticky organisational failure. Checklist + research_grade=false remain honest.
        return JobResult(
            JS.BLOCKED,
            info.get("note") or "point-in-time universe unavailable",
            clears={H.UNIVERSE_INCOMPLETE},
            blocked_on=DEP_UNIVERSE_SOURCE,
            metadata=info,
        )
    source = info.get("source") or "operator"
    grade = "research-grade" if info.get("research_grade") else "inferred/bootstrap"
    return JobResult(
        JS.SUCCEEDED,
        f"point-in-time universe ready · {len(info.get('symbols', []))} symbols "
        f"({source}, {grade})",
        clears={H.UNIVERSE_INCOMPLETE},
        metadata=info,
    )


def run_options_eod(ctx) -> JobResult:
    try:
        capture = getattr(ctx.deps, "capture_options_eod", None)
        info = capture() if callable(capture) else {"saved": 0}
        status = getattr(ctx.deps, "options_eod_status", lambda: {})()
        info = {**(status or {}), **(info or {})}
    except Exception as exc:
        return JobResult(
            JS.RETRYABLE_FAILED,
            "options EOD capture failed",
            error_code="OPTIONS_EOD_ERROR",
            error_message=str(exc),
            failures={H.OPTIONS_HISTORY_INCOMPLETE},
        )
    saved = int(info.get("saved") or 0)
    if saved <= 0 and not info.get("available"):
        return JobResult(
            JS.RETRYABLE_FAILED,
            "options EOD chain unavailable for NIFTY/BANKNIFTY/FINNIFTY — will retry",
            error_code="OPTIONS_CHAIN_UNAVAILABLE",
            failures={H.OPTIONS_HISTORY_INCOMPLETE},
            metadata=info,
        )
    if saved <= 0:
        return JobResult(
            JS.RETRYABLE_FAILED,
            "options EOD capture saved no underlyings — will retry",
            error_code="OPTIONS_EOD_EMPTY",
            failures={H.OPTIONS_HISTORY_INCOMPLETE},
            metadata=info,
        )
    return JobResult(
        JS.SUCCEEDED,
        f"options EOD history saved · {saved}/{info.get('requested', saved)} underlyings · "
        f"store latest={info.get('latest_as_of') or info.get('as_of') or '?'}",
        clears={H.OPTIONS_HISTORY_INCOMPLETE},
        metadata=info,
    )


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


def run_market_scan(ctx) -> JobResult:
    if not ctx.deps.active_snapshot_id():
        return JobResult(JS.BLOCKED, "verified active snapshot required before market scan",
                         failures={H.SNAPSHOT_STALE}, blocked_on=DEP_DATA,
                         state_hint=ST.DATA_BLOCKED, new_entries_allowed=False)
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
        except Exception:
            telegram = {"error": "notification_failed"}
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
    if not ctx.deps.active_snapshot_id():
        return JobResult(JS.BLOCKED, "verified active snapshot required before paper cycle",
                         failures={H.SNAPSHOT_STALE}, blocked_on=DEP_DATA,
                         state_hint=ST.DATA_BLOCKED, new_entries_allowed=False)
    now = ctx.deps.now_ist()
    holidays = ctx.deps.holidays()
    entries_ok, reason, phase = _entry_reason(now, holidays, ctx)
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
    return JobResult(JS.SUCCEEDED, f"paper cycle: {eligibility or 'no-op'}",
                     state_hint=hint, new_entries_allowed=entries_ok,
                     metadata={"eligibility": eligibility, "entry_block_reason": reason,
                               "session_phase": phase})


def run_outcome_resolution(ctx) -> JobResult:
    session_date = ctx.deps.now_ist().date().isoformat()
    if not ctx.deps.active_snapshot_id():
        return JobResult(JS.BLOCKED, "verified EOD data required before outcome resolution",
                         blocked_on=DEP_DATA, failures={H.SNAPSHOT_STALE})
    if hasattr(ctx.deps, "active_snapshot_info"):
        latest = str((ctx.deps.active_snapshot_info() or {}).get("latest_date") or "")
        if latest < session_date:
            return JobResult(JS.BLOCKED,
                             f"outcomes wait for completed-session data ({latest or 'unknown'} < {session_date})",
                             blocked_on=f"EOD_DATA_READY:{session_date}", failures={H.SNAPSHOT_STALE})
    try:
        if hasattr(ctx.deps, "resolve_outcomes"):
            result = ctx.deps.resolve_outcomes(session_date, ctx.active_failures)
        else:
            result = ctx.deps.run_paper_cycle(False)
    except Exception as exc:
        return JobResult(JS.RETRYABLE_FAILED, "outcome resolution failed",
                         error_code="OUTCOME_ERROR", error_message=str(exc))
    closed = len((result or {}).get("positions_closed", []))
    recorded = len((result or {}).get("outcomes_recorded", []))
    return JobResult(JS.SUCCEEDED, f"outcomes resolved · {closed} positions closed · {recorded} decoded",
                     unblocks=(f"{DEP_OUTCOMES}:{session_date}",), metadata=result or {})


def run_learning_cycle(ctx) -> JobResult:
    session_date = ctx.deps.now_ist().date().isoformat()
    try:
        result = ctx.deps.run_learning(session_date, getattr(ctx, "dialogue", None))
    except Exception as exc:
        return JobResult(JS.RETRYABLE_FAILED, "learning cycle failed", error_code="LEARNING_ERROR",
                         error_message=str(exc), failures={H.LEARNING_FAILED}, state_hint=ST.DEGRADED)
    return JobResult(JS.SUCCEEDED, f"learning complete · {result.get('diagnostics', 0)} diagnostics",
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
    SCH.OPTIONS_EOD: run_options_eod,
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
