"""Canonical terminal API with user-first operation routing and recovery.

The launcher normally owns Market Operations. The API remains an observer while
that worker is healthy, but an explicit user action must never disappear behind a
stale worker. If the launcher-owned worker is dead/stale, this product wrapper
cleans up only the verified stale ``operations.market_ops`` PID, invokes the base
bounded recovery path, and refuses the command if a healthy worker still does not
appear. This gives the one-terminal product a second safety net without creating a
second market-data or scan architecture.
"""
from __future__ import annotations

import os
import signal
import subprocess
import time

import terminal_api as core
import terminal_product_api as product
from operations.store import pid_is_alive
from product.operator_health import enrich_autonomy_payload

# Keep terminal_api's canonical control registry untouched. One whole-market scan
# fills all setup families; the UI's separate funds action uses LONG_TERM_REFRESH.
_base_ensure_ops_worker = core._ensure_ops_worker


def _healthy_runtime() -> dict:
    runtime = core._ops_runtime_payload()
    if runtime.get("running") and pid_is_alive(runtime.get("worker_pid")):
        return runtime
    return runtime


def _market_ops_command(pid: int) -> str:
    try:
        return subprocess.check_output(
            ["ps", "-p", str(pid), "-o", "command="],
            text=True,
            stderr=subprocess.DEVNULL,
            timeout=0.5,
        ).strip()
    except Exception:
        return ""


def _stop_stale_owner(runtime: dict) -> bool:
    """Terminate only a verified stale market-ops process."""
    try:
        pid = int(runtime.get("worker_pid") or 0)
    except (TypeError, ValueError):
        return False
    if pid <= 1 or not pid_is_alive(pid):
        return False
    command = _market_ops_command(pid)
    if "operations.market_ops" not in command:
        return False
    try:
        os.kill(pid, signal.SIGTERM)
    except OSError:
        return True
    deadline = time.time() + 1.5
    while time.time() < deadline:
        if not pid_is_alive(pid):
            return True
        time.sleep(0.05)
    if "operations.market_ops" in _market_ops_command(pid):
        try:
            os.kill(pid, signal.SIGKILL)
        except OSError:
            pass
    return True


def _ensure_ops_worker_strict(*, wait: bool = True) -> dict:
    """Return only when Market Operations is healthy or fail loudly."""
    runtime = _healthy_runtime()
    if runtime.get("running") and pid_is_alive(runtime.get("worker_pid")):
        return runtime

    _stop_stale_owner(runtime)
    recovered = _base_ensure_ops_worker(wait=True)
    if recovered.get("running") and pid_is_alive(recovered.get("worker_pid")):
        return recovered

    deadline = time.time() + (1.5 if wait else 0.5)
    while time.time() < deadline:
        time.sleep(0.1)
        recovered = _healthy_runtime()
        if recovered.get("running") and pid_is_alive(recovered.get("worker_pid")):
            return recovered

    raise RuntimeError(
        "Market operations worker did not become ready after bounded recovery; "
        "the command was not silently accepted. The launcher watchdog owns recovery "
        "after this attempt. Check System Health for the worker blocker."
    )


core._ensure_ops_worker = _ensure_ops_worker_strict

_base_autonomy_payload = core._autonomy_payload


def _operator_autonomy_payload() -> dict:
    return enrich_autonomy_payload(_base_autonomy_payload())


core._autonomy_payload = _operator_autonomy_payload

# Preserve stock-by-stock coverage accounting through the compact dashboard API.
_base_scan_payload = core._scan_payload


def _scan_payload_with_coverage() -> dict:
    projected = dict(_base_scan_payload() or {})
    try:
        from product.scan_store import load_scan
        raw = load_scan() or {}
    except Exception:
        raw = {}
    projected["requested_universe"] = int(raw.get("requested_universe", projected.get("universe_size", 0)) or 0)
    projected["coverage_state"] = str(raw.get("coverage_state") or "UNKNOWN")
    projected["coverage_warning"] = str(raw.get("coverage_warning") or "")
    projected["coverage"] = dict(raw.get("coverage") or {})
    return projected


core._scan_payload = _scan_payload_with_coverage


def _attach_authority(payload: dict) -> dict:
    """Decorate recommendations with explanatory evidence; never change ranking/gates."""
    from product.evidence_authority import build_authority_contract, evidence_scorecard

    for category in payload.get("categories") or []:
        if not isinstance(category, dict):
            continue
        for card in category.get("cards") or []:
            if isinstance(card, dict):
                card["evidence_scorecard"] = evidence_scorecard(card)
    payload["authority"] = build_authority_contract(core._scan_payload())
    return payload


@product.app.get("/api/operator-health")
def operator_health() -> dict:
    """Current-session health plus historical-ledger separation for diagnostics."""
    return core._autonomy_payload()


@product.app.get("/api/recommendations-workspace")
def recommendations_workspace() -> dict:
    """Canonical Recommendations projection plus read-only evidence explanations."""
    from product.recommendations_workspace import (
        build_recommendations_workspace,
        slim_workspace_for_desk,
    )

    payload = build_recommendations_workspace(
        scan_payload=core._scan_payload(),
        long_term_payload=core._long_term_payload(),
        refresh_technicals=False,
        settle_cases=False,
        deep_confirm=False,
        persist_ledger=False,
    )
    return slim_workspace_for_desk(_attach_authority(payload))


@product.app.get("/api/evidence-authority")
def evidence_authority() -> dict:
    """Methodology, scan coverage, and measured outcome performance."""
    from product.evidence_authority import build_authority_contract
    return build_authority_contract(core._scan_payload())


@product.app.get("/api/decision-journal")
def decision_journal(symbol: str = "", limit: int = 120) -> dict:
    """Public audit trail for surfaced and non-surfaced market decisions."""
    from product.evidence_authority import build_decision_journal
    return build_decision_journal(symbol=symbol, limit=max(1, min(int(limit or 120), 1000)))


@product.app.get("/api/market-reports-workspace")
def market_reports_workspace() -> dict:
    """Canonical read projection for Market Reports."""
    from product.recommendations_workspace import build_market_reports_workspace

    payload = build_market_reports_workspace(
        persist_today=True,
        news_payload=core._news_payload(),
        scan_payload=core._scan_payload(),
        rebuild=False,
    )
    if payload.get("needs_refresh") and not payload.get("empty_detail"):
        payload["empty_detail"] = (
            "Today's sourced market report is incomplete. Missing scan/news evidence "
            "stays empty; QuantTerm does not invent headlines, prices, or market facts."
        )
    return payload


@product.app.get("/api/scan-audit")
def scan_audit(symbol: str = "", limit: int = 250) -> dict:
    """Explain exactly what happened to each symbol in the latest market scan."""
    from scan.scan_coverage import load_audit, lookup_symbol

    payload = load_audit()
    summary = dict(payload.get("summary") or {})
    clean = str(symbol or "").strip().upper()
    if clean:
        row = lookup_symbol(clean, payload)
        return {
            "generated_at": payload.get("generated_at"),
            "summary": summary,
            "symbol": clean,
            "found": row is not None,
            "result": row,
        }
    ledger = list(payload.get("ledger") or [])
    cap = max(1, min(int(limit or 250), 2500))
    return {
        "generated_at": payload.get("generated_at"),
        "summary": summary,
        "total": len(ledger),
        "rows": ledger[:cap],
        "truncated": len(ledger) > cap,
    }


def _registered_paths() -> set[str]:
    return {
        str(getattr(route, "path", ""))
        for route in product.app.routes
        if getattr(route, "path", None)
    }


@product.app.get("/api/product-contract")
def product_contract() -> dict:
    """Machine-readable proof that the primary desk surfaces are actually wired."""
    paths = _registered_paths()
    scan = core._scan_payload()
    long_term = core._long_term_payload()
    operations = core._operations_payload()
    autonomy = core._autonomy_payload()
    checks = {
        "market_scan": {
            "route_registered": "/api/controls/{control_name}" in paths,
            "audit_route_registered": "/api/scan-audit" in paths,
            "trigger": "RUN_SCAN_NOW",
            "worker_running": bool(operations.get("running")),
            "data_available": bool(scan.get("available")),
            "requested_universe": int(scan.get("requested_universe") or 0),
            "checked_universe": int(scan.get("universe_size") or 0),
            "coverage_state": str(scan.get("coverage_state") or "UNKNOWN"),
            "coverage": dict(scan.get("coverage") or {}),
        },
        "recommendations": {
            "route_registered": "/api/recommendations-workspace" in paths,
            "authority_route_registered": "/api/evidence-authority" in paths,
            "journal_route_registered": "/api/decision-journal" in paths,
            "depends_on": ["market_scan", "long_term_scan"],
            "data_available": bool(scan.get("available") or long_term.get("available")),
        },
        "market_reports": {
            "route_registered": "/api/market-reports-workspace" in paths,
            "trigger": "REFRESH_MARKET_REPORT_NOW",
            "worker_running": bool(operations.get("running")),
        },
        "stock_intelligence": {
            "route_registered": "/api/stock-intelligence/{symbol}" in paths,
            "acquire_route_registered": "/api/due-diligence/{symbol}/acquire" in paths,
        },
        "learning": {
            "operator_health_route_registered": "/api/operator-health" in paths,
            "status": str(autonomy.get("learning_status") or "UNKNOWN"),
            "supervisor_running": bool(autonomy.get("running")),
        },
    }
    wired = all(
        bool(item.get("route_registered", item.get("operator_health_route_registered", False)))
        for item in checks.values()
    )
    return {
        "wired": wired,
        "checks": checks,
        "note": (
            "wired=true proves the canonical API paths and triggers are registered. "
            "Data availability and provider health are reported separately and are never fabricated."
        ),
    }


app = product.app
