"""Canonical operator-facing capability registry.

Metadata / control-plane only. Trading stays in Selection Authority, paper
autopilot, operations store, and autonomy jobs.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[1]
STATE_PATH = ROOT / "logs" / "product" / "capability_runtime.json"

AUTOMATIC = "AUTOMATIC"
HOME_ACTION = "HOME_ACTION"
ADVANCED_ACTION = "ADVANCED_ACTION"
DEVELOPER_ONLY = "DEVELOPER_ONLY"

MODES = {AUTOMATIC, HOME_ACTION, ADVANCED_ACTION, DEVELOPER_ONLY}


def _cap(
    capability_id: str,
    *,
    name: str,
    plain_name: str,
    description: str,
    mode: str,
    operation_kind: str = "",
    control: str = "",
    old_invocation: str = "",
    automatic_trigger: str = "",
    canonical_owner: str,
    persisted_state: str,
    failure_behavior: str,
    still_requires_terminal: bool,
    safe_to_retry: bool = True,
    idempotent: bool = True,
    requires_market_open: bool = False,
    requires_zerodha_auth: bool = False,
    requires_network: bool = False,
    read_only: bool = False,
    affects_paper_entries: bool = False,
    affects_live_money: bool = False,
    homepage_visible: bool = False,
) -> dict[str, Any]:
    if mode not in MODES:
        raise ValueError(f"invalid mode {mode}")
    if affects_live_money:
        raise ValueError(f"{capability_id} must not affect live money")
    return {
        "capability_id": capability_id,
        "name": name,
        "plain_name": plain_name,
        "description": description,
        "mode": mode,
        "operation_kind": operation_kind,
        "control": control,
        "old_invocation": old_invocation,
        "automatic_trigger": automatic_trigger,
        "canonical_owner": canonical_owner,
        "persisted_state": persisted_state,
        "failure_behavior": failure_behavior,
        "still_requires_terminal": still_requires_terminal,
        "safe_to_retry": safe_to_retry,
        "idempotent": idempotent,
        "requires_market_open": requires_market_open,
        "requires_zerodha_auth": requires_zerodha_auth,
        "requires_network": requires_network,
        "read_only": read_only,
        "affects_paper_entries": affects_paper_entries,
        "affects_live_money": False,
        "homepage_visible": homepage_visible,
        "home_action": control or ("instruction" if mode == HOME_ACTION else ""),
    }


CAPABILITIES: tuple[dict[str, Any], ...] = (
    _cap(
        "service_startup",
        name="Service startup / recovery",
        plain_name="Start QuantTerm",
        description="Bring up API, desk, reports, autonomy, and market operations.",
        mode=AUTOMATIC,
        old_invocation="bash scripts/run_quantterm_complete.sh",
        automatic_trigger="launcher start + supervise loop",
        canonical_owner="scripts/run_quantterm_complete.sh → run_quantterm.sh",
        persisted_state="logs/market_ops/runtime.json, logs/autonomy/status.json",
        failure_behavior="Restart the dead child. Do not start a second owner.",
        still_requires_terminal=True,
        homepage_visible=True,
    ),
    _cap(
        "config_validation",
        name="Configuration validation",
        plain_name="Check settings",
        description="Require .env and Kite API key/secret before the stack stays up.",
        mode=AUTOMATIC,
        old_invocation="manual .env edit",
        automatic_trigger="launcher start",
        canonical_owner="scripts/run_quantterm_complete.sh",
        persisted_state=".env",
        failure_behavior="Stop with a plain instruction. Never fake credentials.",
        still_requires_terminal=False,
        requires_network=False,
        read_only=True,
    ),
    _cap(
        "zerodha_login_detect",
        name="Zerodha login requirement",
        plain_name="Zerodha login",
        description="Detect missing/expired Kite session. Interactive launch may open login.",
        mode=HOME_ACTION,
        old_invocation="python main.py login",
        automatic_trigger="launcher auth probe; Home LOGIN_REQUIRED",
        canonical_owner="data.kite_client + research.autonomy.auth",
        persisted_state="KITE_ACCESS_TOKEN",
        failure_behavior="WAITING FOR ZERODHA LOGIN — not failed, not fake green.",
        still_requires_terminal=False,
        requires_zerodha_auth=True,
        requires_network=True,
        homepage_visible=True,
        control="",
    ),
    _cap(
        "data_freshness",
        name="Data freshness check",
        plain_name="Is today's data ready?",
        description="Read official bhavcopy readiness. Missing stays missing.",
        mode=AUTOMATIC,
        old_invocation="manual data page",
        automatic_trigger="desk pipeline + autonomy DATA_REFRESH",
        canonical_owner="product.desk_pipeline + autonomy DATA_REFRESH",
        persisted_state="bhavcopy store / dashboard.data",
        failure_behavior="Home shows Preparing or Retry. No invented bars.",
        still_requires_terminal=False,
        read_only=True,
        homepage_visible=True,
    ),
    _cap(
        "data_prepare",
        name="Official data preparation",
        plain_name="Get today's prices",
        description="Prepare official NSE history used by scan and paper.",
        mode=AUTOMATIC,
        operation_kind="DATA_PREPARE",
        control="REFRESH_DATA_NOW",
        old_invocation="POST /api/controls/REFRESH_DATA_NOW",
        automatic_trigger="desk pipeline first step; autonomy data job",
        canonical_owner="operations.market_ops DATA_PREPARE",
        persisted_state="operations store + bhavcopy",
        failure_behavior="Home primary Retry → same control. Deduped.",
        still_requires_terminal=False,
        requires_network=True,
        homepage_visible=True,
        safe_to_retry=True,
    ),
    _cap(
        "market_scan",
        name="Shared market scan",
        plain_name="Look at the whole market",
        description="One canonical whole-market scan. Home, Scanner, Recos share the file.",
        mode=AUTOMATIC,
        operation_kind="MARKET_SCAN",
        control="RUN_SCAN_NOW",
        old_invocation="POST /api/controls/RUN_SCAN_NOW or scripts/local_stack.py scan",
        automatic_trigger="desk kick + autonomy scan slots; Home Scan now",
        canonical_owner="operations.market_ops → scan.market_scan_service",
        persisted_state="logs/product/latest_momentum_scan.json",
        failure_behavior="Reuse lock; no duplicate scan. Home can retry.",
        still_requires_terminal=False,
        requires_network=True,
        homepage_visible=True,
    ),
    _cap(
        "recommendations",
        name="Recommendation refresh",
        plain_name="Write today's research list",
        description="Build recommendation cards from the saved scan. Empty list is valid.",
        mode=AUTOMATIC,
        old_invocation="scan completion write",
        automatic_trigger="after MARKET_SCAN succeeds",
        canonical_owner="product.recommendations_store",
        persisted_state="logs/product/latest_recommendations.json",
        failure_behavior="Missing stays missing. No invented BUY.",
        still_requires_terminal=False,
        read_only=False,
        homepage_visible=True,
    ),
    _cap(
        "paper_selection_cycle",
        name="Paper selection cycle",
        plain_name="Decide whether to take a paper trade",
        description="Selection Authority consumes saved recos. Eligible tiers only.",
        mode=AUTOMATIC,
        control="RUN_CYCLE_NOW",
        old_invocation="POST /api/controls/RUN_CYCLE_NOW",
        automatic_trigger="autonomy PAPER_CYCLE after scan slot",
        canonical_owner="product.paper_autopilot.run_reco_paper_cycle",
        persisted_state="paper_autopilot_journal + intel_book",
        failure_behavior="No trade is valid. Reasons stay machine-readable.",
        still_requires_terminal=False,
        requires_market_open=True,
        affects_paper_entries=True,
        homepage_visible=True,
    ),
    _cap(
        "open_position_supervision",
        name="Open-position supervision",
        plain_name="Watch open paper trades",
        description="Mark stops/targets on official bars. Resume after restart.",
        mode=AUTOMATIC,
        old_invocation="autonomy intelligence cycle",
        automatic_trigger="intraday + EOD management cycles",
        canonical_owner="research.auto_research.paper_book.PaperBook.mark",
        persisted_state="logs/intelligence/intel_book.json",
        failure_behavior="Missing bar skips that name. No invented fill.",
        still_requires_terminal=False,
        homepage_visible=True,
    ),
    _cap(
        "eod_settlement",
        name="EOD settlement",
        plain_name="Finish the day",
        description="Close management, ingest closed book, settle rejects, write daily report.",
        mode=AUTOMATIC,
        old_invocation="manual settle scripts (removed from normal use)",
        automatic_trigger="autonomy OUTCOME_RESOLUTION → settle_and_report",
        canonical_owner="research.autonomy.jobs.resolve_outcomes + product.forward_soak.settle_and_report",
        persisted_state="intel_book + forward_evidence.jsonl + forward_daily/",
        failure_behavior="Pending stays pending. Duplicate settlement is a no-op.",
        still_requires_terminal=False,
        homepage_visible=True,
    ),
    _cap(
        "counterfactual_settlement",
        name="Counterfactual settlement",
        plain_name="Grade skipped names later",
        description="Rejected/waited/not-surfaced get a classification when later bars exist.",
        mode=AUTOMATIC,
        old_invocation="settle_pending_counterfactuals by hand",
        automatic_trigger="settle_and_report / official later bars",
        canonical_owner="product.forward_soak.settle_pending_from_market",
        persisted_state="forward_evidence.jsonl",
        failure_behavior="No bars → PENDING. Never booked as P&L.",
        still_requires_terminal=False,
        read_only=False,
    ),
    _cap(
        "learning_ingestion",
        name="Learning ingestion",
        plain_name="Remember what happened",
        description="Closed trades and classified rejects update next-cycle policies.",
        mode=AUTOMATIC,
        old_invocation="ingest_closed_book by hand",
        automatic_trigger="EOD ingest_closed_book then next cycle evaluate_policies",
        canonical_owner="product.paper_learning_loop",
        persisted_state="learning_policies.json + learning_ingested.json",
        failure_behavior="Idempotent keys. Hard DD/chase/risk stay hard.",
        still_requires_terminal=False,
        homepage_visible=True,
    ),
    _cap(
        "forward_evidence",
        name="Forward evidence persistence",
        plain_name="Write today's decisions down",
        description="Freeze each cycle into the append-only ledger with provenance.",
        mode=AUTOMATIC,
        old_invocation="implicit journal only",
        automatic_trigger="run_reco_paper_cycle → record_cycle_evidence",
        canonical_owner="product.forward_evidence",
        persisted_state="logs/product/forward_evidence.jsonl",
        failure_behavior="Fail-open. Never overwrite PIT fields.",
        still_requires_terminal=False,
        homepage_visible=True,
    ),
    _cap(
        "daily_forward_report",
        name="Daily forward report",
        plain_name="Yesterday's one-page summary",
        description="Machine + markdown daily soak report.",
        mode=AUTOMATIC,
        old_invocation="none",
        automatic_trigger="settle_and_report",
        canonical_owner="product.forward_soak.write_daily_report",
        persisted_state="logs/product/forward_daily/YYYY-MM-DD.json",
        failure_behavior="Write what is known. Missing stays missing.",
        still_requires_terminal=False,
        homepage_visible=True,
        read_only=True,
    ),
    _cap(
        "forward_soak_verify",
        name="Forward soak verification",
        plain_name="Check today's evidence trail",
        description="Read-only verify_persisted_soak. CLI and Home share this function.",
        mode=AUTOMATIC,
        control="VERIFY_FORWARD_SOAK",
        old_invocation="python scripts/verify_forward_soak.py",
        automatic_trigger="startup + after settlement + throttled after paper cycle",
        canonical_owner="product.forward_soak.verify_persisted_soak",
        persisted_state="logs/product/forward_soak_verify.json",
        failure_behavior="Valid no-trade is not failure. Live money must stay LOCKED.",
        still_requires_terminal=False,
        read_only=True,
        homepage_visible=True,
        safe_to_retry=True,
        idempotent=True,
    ),
    _cap(
        "health_observation",
        name="Read-only health observation",
        plain_name="Is anything broken?",
        description="Independent health lanes. No collapsed green light.",
        mode=AUTOMATIC,
        old_invocation="GET /api/system-health-contract",
        automatic_trigger="dashboard poll",
        canonical_owner="product.system_health_contract",
        persisted_state="computed from artifacts",
        failure_behavior="Lane stays Problem/Waiting. Telegram optional.",
        still_requires_terminal=False,
        read_only=True,
        homepage_visible=True,
    ),
    _cap(
        "zerodha_observation",
        name="Zerodha reconciliation observation",
        plain_name="Watch the broker book (read-only)",
        description="Capture orders/positions/margins. Never place or change an order.",
        mode=AUTOMATIC,
        old_invocation="scripts/run_zerodha_observation.py",
        automatic_trigger="operations.zerodha_observer when Kite session is valid",
        canonical_owner="operations.zerodha_observer + execution.reconciliation",
        persisted_state="logs/reconciliation/",
        failure_behavior="No auth → WAITING FOR ZERODHA LOGIN. broker_mutations_enabled=False.",
        still_requires_terminal=False,
        requires_zerodha_auth=True,
        requires_network=True,
        read_only=True,
    ),
    _cap(
        "scan_now",
        name="Scan now",
        plain_name="Scan now",
        description="User click jumps the existing market-scan queue.",
        mode=HOME_ACTION,
        operation_kind="MARKET_SCAN",
        control="RUN_SCAN_NOW",
        old_invocation="Home Scan now button",
        automatic_trigger="",
        canonical_owner="operations.store.enqueue",
        persisted_state="operations store",
        failure_behavior="Deduped if already running.",
        still_requires_terminal=False,
        homepage_visible=True,
        requires_network=True,
    ),
    _cap(
        "refresh_data_now",
        name="Refresh data",
        plain_name="Retry market data",
        description="User retry of official data preparation.",
        mode=HOME_ACTION,
        operation_kind="DATA_PREPARE",
        control="REFRESH_DATA_NOW",
        old_invocation="POST /api/controls/REFRESH_DATA_NOW",
        automatic_trigger="",
        canonical_owner="operations.store.enqueue",
        persisted_state="operations store",
        failure_behavior="Deduped. Home shows progress.",
        still_requires_terminal=False,
        homepage_visible=True,
        requires_network=True,
    ),
    _cap(
        "refresh_funds",
        name="Refresh fundamentals",
        plain_name="Refresh company facts",
        description="Long-term funds refresh.",
        mode=HOME_ACTION,
        operation_kind="LONG_TERM_REFRESH",
        control="REFRESH_LONG_TERM_NOW",
        old_invocation="POST /api/controls/REFRESH_LONG_TERM_NOW",
        automatic_trigger="desk pipeline after scan",
        canonical_owner="operations.market_ops",
        persisted_state="long-term overlay",
        failure_behavior="Last-good facts stay labelled stale.",
        still_requires_terminal=False,
        requires_network=True,
    ),
    _cap(
        "refresh_news_report",
        name="Refresh news / market report",
        plain_name="Refresh the daily note",
        description="News curator and market pulse.",
        mode=HOME_ACTION,
        operation_kind="NEWS_REFRESH",
        control="REFRESH_NEWS_NOW",
        old_invocation="POST /api/controls/REFRESH_NEWS_NOW",
        automatic_trigger="desk kick",
        canonical_owner="operations.market_ops",
        persisted_state="news store / market reports",
        failure_behavior="Empty articles stay empty.",
        still_requires_terminal=False,
        requires_network=True,
    ),
    _cap(
        "run_paper_cycle_now",
        name="Run paper cycle now",
        plain_name="Decide now",
        description="Ask autonomy to run one paper selection cycle.",
        mode=HOME_ACTION,
        control="RUN_CYCLE_NOW",
        old_invocation="POST /api/controls/RUN_CYCLE_NOW",
        automatic_trigger="",
        canonical_owner="research.autonomy.controls",
        persisted_state="autonomy controls.db",
        failure_behavior="Stale reco cannot enter. Duplicate position refused.",
        still_requires_terminal=False,
        requires_market_open=True,
        affects_paper_entries=True,
        homepage_visible=True,
    ),
    _cap(
        "pause_paper_entries",
        name="Pause new paper entries",
        plain_name="Pause the paper bot",
        description="Stop new entries. Open positions stay supervised.",
        mode=HOME_ACTION,
        control="PAUSE_NEW_PAPER_ENTRIES",
        old_invocation="POST /api/controls/PAUSE_NEW_PAPER_ENTRIES",
        automatic_trigger="",
        canonical_owner="research.autonomy.controls",
        persisted_state="autonomy owner_state",
        failure_behavior="Fail closed to no new entries if unclear.",
        still_requires_terminal=False,
        affects_paper_entries=True,
        homepage_visible=True,
    ),
    _cap(
        "resume_paper_entries",
        name="Resume new paper entries",
        plain_name="Resume the paper bot",
        description="Allow new paper entries again.",
        mode=HOME_ACTION,
        control="RESUME_NEW_PAPER_ENTRIES",
        old_invocation="POST /api/controls/RESUME_NEW_PAPER_ENTRIES",
        automatic_trigger="",
        canonical_owner="research.autonomy.controls",
        persisted_state="autonomy owner_state",
        failure_behavior="Still cannot invent BUY or unlock live money.",
        still_requires_terminal=False,
        affects_paper_entries=True,
        homepage_visible=True,
    ),
    _cap(
        "refresh_fno",
        name="Refresh F&O universe",
        plain_name="Refresh futures list",
        description="Optional F&O universe refresh.",
        mode=ADVANCED_ACTION,
        operation_kind="FNO_REFRESH",
        control="REFRESH_FNO_NOW",
        old_invocation="POST /api/controls/REFRESH_FNO_NOW",
        automatic_trigger="",
        canonical_owner="operations.market_ops",
        persisted_state="fno store",
        failure_behavior="Desk stays usable without F&O.",
        still_requires_terminal=False,
        requires_network=True,
    ),
    _cap(
        "refresh_market_report",
        name="Refresh market report now",
        plain_name="Rebuild today's pulse",
        description="Force a market-report rebuild.",
        mode=ADVANCED_ACTION,
        operation_kind="MARKET_REPORT",
        control="REFRESH_MARKET_REPORT_NOW",
        old_invocation="POST /api/controls/REFRESH_MARKET_REPORT_NOW",
        automatic_trigger="",
        canonical_owner="operations.market_ops",
        persisted_state="market reports dir",
        failure_behavior="Needs_refresh stays honest.",
        still_requires_terminal=False,
        requires_network=True,
    ),
    _cap(
        "pytest_suite",
        name="pytest",
        plain_name="Engineering tests",
        description="Full regression suite.",
        mode=DEVELOPER_ONLY,
        old_invocation="python -m pytest",
        automatic_trigger="",
        canonical_owner="tests/",
        persisted_state="none",
        failure_behavior="CI red. Never a Home button.",
        still_requires_terminal=True,
        read_only=True,
    ),
    _cap(
        "issue92_dod",
        name="Historical DoD verifier",
        plain_name="Old issue checklist",
        description="Issue #92 live Definition of Done script.",
        mode=DEVELOPER_ONLY,
        old_invocation="python scripts/verify_issue92_dod.py",
        automatic_trigger="",
        canonical_owner="scripts/verify_issue92_dod.py",
        persisted_state="docs/issue92_live_dod_proof.md",
        failure_behavior="Engineering only.",
        still_requires_terminal=True,
        read_only=True,
    ),
    _cap(
        "synthetic_fixtures",
        name="Synthetic fixtures / backfill",
        plain_name="Test data",
        description="Pytest isolation and one-off backfills. Never production evidence.",
        mode=DEVELOPER_ONLY,
        old_invocation="tests/ + QT_* temp paths",
        automatic_trigger="",
        canonical_owner="tests/conftest.py",
        persisted_state="tmp",
        failure_behavior="TEST_FIXTURE cannot enter promotion stats.",
        still_requires_terminal=True,
    ),
)


def all_capabilities() -> list[dict[str, Any]]:
    return [dict(row) for row in CAPABILITIES]


def by_id(capability_id: str) -> dict[str, Any] | None:
    for row in CAPABILITIES:
        if row["capability_id"] == capability_id:
            return dict(row)
    return None


def home_actions() -> list[dict[str, Any]]:
    return [dict(row) for row in CAPABILITIES if row["mode"] == HOME_ACTION]


def automatic() -> list[dict[str, Any]]:
    return [dict(row) for row in CAPABILITIES if row["mode"] == AUTOMATIC]


def audit_rows() -> list[dict[str, Any]]:
    """Table rows for the Phase 13 automation audit."""
    rows = []
    for row in CAPABILITIES:
        rows.append({
            "Capability": row["plain_name"],
            "Old invocation": row["old_invocation"],
            "New mode": row["mode"],
            "Automatic trigger": row["automatic_trigger"] or "—",
            "Home action": row["home_action"] or "—",
            "Canonical owner": row["canonical_owner"],
            "Persisted state": row["persisted_state"],
            "Failure behavior": row["failure_behavior"],
            "Still requires terminal?": "yes" if row["still_requires_terminal"] else "no",
        })
    return rows


def _state_path() -> Path:
    override = os.environ.get("QT_CAPABILITY_STATE")
    return Path(override) if override else STATE_PATH


def load_runtime_state() -> dict[str, Any]:
    path = _state_path()
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        payload = {}
    return payload if isinstance(payload, dict) else {}


def note_run(capability_id: str, *, status: str, reason: str = "") -> dict[str, Any]:
    state = load_runtime_state()
    items = dict(state.get("items") or {})
    items[capability_id] = {
        "last_run": datetime.now(timezone.utc).isoformat(),
        "last_status": status,
        "last_reason": reason,
    }
    state = {"schema_version": 1, "updated_at": datetime.now(timezone.utc).isoformat(), "items": items}
    path = _state_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(state, indent=2), encoding="utf-8")
    os.replace(tmp, path)
    return items[capability_id]


def attach_runtime(rows: list[Mapping[str, Any]] | None = None) -> list[dict[str, Any]]:
    state = load_runtime_state().get("items") or {}
    out = []
    for row in rows or all_capabilities():
        item = dict(row)
        extra = dict(state.get(item["capability_id"]) or {})
        item["last_run"] = extra.get("last_run") or ""
        item["last_status"] = extra.get("last_status") or ""
        item["last_reason"] = extra.get("last_reason") or ""
        item["next_due"] = extra.get("next_due") or ""
        out.append(item)
    return out
