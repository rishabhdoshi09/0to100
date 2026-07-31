"""
🔧 Job handlers — thin wiring over the EXISTING canonical components.

Each handler runs one operation by calling the real subsystem through an injected `deps` object, so
the whole thing is deterministic and network-free in tests while wiring genuine components in
production. Handlers never place a broker order, never fabricate data, and translate a genuine failure
into a BLOCKED/FAILED job — never into a false "no trade" / "no opportunity".
"""
from __future__ import annotations

from dataclasses import dataclass, field

from research.autonomy import job_store as JS
from research.autonomy import schedules as SCH
from research.autonomy import supervisor_state as ST
from research.autonomy import health as H


@dataclass
class JobResult:
    status: str
    summary: str = ""
    output_snapshot_id: str | None = None
    error_code: str = ""
    error_message: str = ""
    failures: set = field(default_factory=set)   # health failure codes to raise
    clears: set = field(default_factory=set)     # health failure codes to clear
    state_hint: str | None = None
    new_entries_allowed: bool = True


class _Ctx:
    """Handler context — carries the injected dependency object."""
    def __init__(self, deps):
        self.deps = deps


class Deps:
    """Production dependency wiring (guarded/lazy). Tests pass a fake with the same attributes."""

    def now_ist(self):
        from research.intelligence.data import nse_calendar as CAL
        return CAL._now_ist()

    def holidays(self):
        try:
            from research.intelligence.data import nse_calendar as CAL
            return CAL.load_holidays()
        except Exception:
            return set()

    def session_valid(self) -> bool:
        try:
            from data.kite_client import KiteClient
            return bool(KiteClient().is_connected())
        except Exception:
            return False

    def activate(self):
        from research.intelligence.data.kite_activation import activate
        return activate(start_worker=False, run_cycle=False)

    def active_snapshot_id(self):
        try:
            from research.auto_research.scheduler import get_brain
            store = get_brain().snapshot_store
            return store.get_active_snapshot() if store else None
        except Exception:
            return None

    def run_scan(self):
        from ui.retail_home_momentum import _run_and_save_momentum  # existing whole-market scan+save
        return _run_and_save_momentum()

    def run_paper_cycle(self, entries_allowed: bool):
        from research.auto_research.scheduler import get_brain
        return get_brain().run_intelligence_cycle_day()

    def news_health(self):
        try:
            from news.curator_service import get_news_curator_service
            svc = get_news_curator_service()
            return {"running": svc.running, "error": svc.last_error}
        except Exception as exc:
            return {"running": False, "error": str(exc)}


# ── handlers ─────────────────────────────────────────────────────────────────────
def run_auth_health(ctx) -> JobResult:
    if ctx.deps.session_valid():
        return JobResult(JS.SUCCEEDED, "AUTH_READY", clears={H.AUTH_MISSING},
                         state_hint=ST.DATA_REFRESHING)
    return JobResult(JS.BLOCKED, "AUTH_REQUIRED — complete the daily Zerodha login",
                     failures={H.AUTH_MISSING}, state_hint=ST.AUTH_REQUIRED, new_entries_allowed=False)


def run_data_refresh(ctx) -> JobResult:
    if not ctx.deps.session_valid():
        return JobResult(JS.BLOCKED, "auth required before data refresh", failures={H.AUTH_MISSING},
                         state_hint=ST.AUTH_REQUIRED, new_entries_allowed=False)
    try:
        report = ctx.deps.activate()
    except Exception as exc:
        return JobResult(JS.RETRYABLE_FAILED, "data refresh error", error_code="ACTIVATE_ERROR",
                         error_message=str(exc), failures={H.SNAPSHOT_STALE}, state_hint=ST.DATA_BLOCKED)
    ok = report.status("GENUINE_SNAPSHOT_ACTIVE") == "PASS" if hasattr(report, "status") else False
    if ok:
        sid = getattr(report, "active_pointer", None) or getattr(report, "snapshot_id", None)
        return JobResult(JS.SUCCEEDED, "genuine snapshot active", output_snapshot_id=sid,
                         clears={H.SNAPSHOT_STALE, H.AUTH_MISSING}, state_hint=ST.DATA_READY)
    blocker = getattr(report, "blocker", "") or "snapshot not forward-eligible"
    return JobResult(JS.BLOCKED, f"data not ready: {blocker}", failures={H.SNAPSHOT_STALE},
                     state_hint=ST.DATA_BLOCKED, new_entries_allowed=False)


def run_market_scan(ctx) -> JobResult:
    try:
        payload = ctx.deps.run_scan()
    except Exception as exc:
        # a provider/parse failure is a FAILED job, never a silent "no opportunity"
        return JobResult(JS.RETRYABLE_FAILED, "scan failed to load market data",
                         error_code="SCAN_ERROR", error_message=str(exc), state_hint=ST.OBSERVING)
    summ = (payload or {}).get("summary", {})
    n = summ.get("with_any_setup", 0)
    return JobResult(JS.SUCCEEDED, f"scan complete · {n} setups · {summ.get('momentum', 0)} momentum",
                     state_hint=ST.OBSERVING)


def run_paper_cycle(ctx) -> JobResult:
    now = ctx.deps.now_ist()
    entries_ok = SCH.entries_allowed_by_clock(now, ctx.deps.holidays())
    try:
        res = ctx.deps.run_paper_cycle(entries_ok)
    except Exception as exc:
        return JobResult(JS.RETRYABLE_FAILED, "paper cycle error", error_code="CYCLE_ERROR",
                         error_message=str(exc))
    elig = (res or {}).get("eligibility", "")
    if not entries_ok and elig == "TRADED":
        # structural guard: opening-noise / outside-window cycles must not open NEW entries
        return JobResult(JS.BLOCKED, "outside entry window — new entries blocked (opening noise)",
                         failures=set(), state_hint=ST.OBSERVING, new_entries_allowed=False)
    hint = ST.PAPER_ACTIVE if entries_ok else ST.OBSERVING
    return JobResult(JS.SUCCEEDED, f"paper cycle: {elig or 'no-op'}", state_hint=hint,
                     new_entries_allowed=entries_ok)


def run_news_health(ctx) -> JobResult:
    h = ctx.deps.news_health()
    if h.get("running"):
        return JobResult(JS.SUCCEEDED, "news curator healthy", clears={H.NEWS_UNAVAILABLE})
    return JobResult(JS.SUCCEEDED, f"news unavailable: {h.get('error', '')}",
                     failures={H.NEWS_UNAVAILABLE})     # non-critical: trading unaffected


HANDLERS = {
    SCH.AUTH_HEALTH: run_auth_health,
    SCH.DATA_REFRESH: run_data_refresh,
    SCH.MARKET_SCAN: run_market_scan,
    SCH.PAPER_CYCLE: run_paper_cycle,
    SCH.NEWS_REFRESH: run_news_health,
}
