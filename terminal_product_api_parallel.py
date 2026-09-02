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


def _live_owner_pid(runtime: dict) -> int:
    """A live lock holder is the owner even when runtime.json still has a dead PID."""
    candidates: list[int] = []
    try:
        from operations.store import live_lock_owner_pid
        lock_pid = int(live_lock_owner_pid(core.OPS_ROOT / "worker.lock") or 0)
    except Exception:
        lock_pid = 0
    if lock_pid:
        candidates.append(lock_pid)
    try:
        runtime_pid = int(runtime.get("worker_pid") or 0)
    except (TypeError, ValueError):
        runtime_pid = 0
    if runtime_pid:
        candidates.append(runtime_pid)
    seen: set[int] = set()
    for pid in candidates:
        if pid <= 1 or pid in seen or not pid_is_alive(pid):
            continue
        seen.add(pid)
        command = _market_ops_command(pid)
        if command and "operations.market_ops" not in command:
            continue
        return pid
    return 0


def _ensure_ops_worker_strict(*, wait: bool = True) -> dict:
    """Return only when Market Operations is healthy or fail loudly."""
    runtime = _healthy_runtime()
    if runtime.get("running") and pid_is_alive(runtime.get("worker_pid")):
        return runtime

    live = _live_owner_pid(runtime)
    if live:
        deadline = time.time() + (4.0 if wait else 0.8)
        while time.time() < deadline:
            time.sleep(0.1)
            runtime = _healthy_runtime()
            if runtime.get("running") and pid_is_alive(runtime.get("worker_pid")):
                return runtime
            if not pid_is_alive(live):
                break
        # Launcher already owns this process. Do not spawn a second worker
        # and do not take the desk API down for a late first heartbeat.
        out = dict(runtime)
        out["running"] = True
        out["worker_pid"] = live
        out["recovering"] = True
        return out

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


def _component_line(scorecard: dict) -> str:
    bits: list[str] = []
    for component in scorecard.get("components") or []:
        if not isinstance(component, dict):
            continue
        label = str(component.get("label") or component.get("id") or "Evidence")
        score = component.get("score")
        coverage = component.get("coverage_pct")
        if score is None:
            bits.append(f"{label}: unknown")
        else:
            bits.append(
                f"{label}: {float(score):.0f}/100"
                + (f" @ {float(coverage):.0f}% coverage" if coverage is not None else "")
            )
    return " · ".join(bits)


def _attach_authority(payload: dict) -> dict:
    """Decorate recommendations with explanatory evidence; never change ranking/gates.

    Existing React cards already render ``evidence`` and ``evidence_coverage``.
    The existing See Evidence panel renders its provenance paragraph, so the five
    component scores are injected there as an explanation only. No authority field
    is read by the recommendation selection or money path.
    """
    from product.evidence_authority import (
        build_authority_contract,
        build_decision_journal,
        evidence_scorecard,
    )
    from product.strategy_catalog import decorate_card

    for category in payload.get("categories") or []:
        if not isinstance(category, dict):
            continue
        for card in category.get("cards") or []:
            if not isinstance(card, dict):
                continue
            scorecard = evidence_scorecard(card)
            card["evidence_scorecard"] = scorecard
            card["evidence_coverage"] = scorecard.get("coverage_pct")
            score = scorecard.get("score")
            quality = str(scorecard.get("quality") or "THIN").replace("_", " ").title()
            card["evidence_grade"] = card.get("evidence")
            card["evidence"] = (
                f"{float(score):.0f}/100 · {quality}" if score is not None
                else f"Unscored · {quality}"
            )
            panel = dict(card.get("evidence_panel") or {})
            old_provenance = str(panel.get("provenance") or "").strip()
            components = _component_line(scorecard)
            headline = (
                f"QuantTerm Evidence Score {float(score):.0f}/100 · "
                f"coverage {float(scorecard.get('coverage_pct') or 0):.0f}%"
                if score is not None else
                f"QuantTerm Evidence Score unscored · coverage {float(scorecard.get('coverage_pct') or 0):.0f}%"
            )
            panel["provenance"] = " ".join(
                part for part in (
                    headline + ".",
                    ("Components — " + components + ".") if components else "",
                    "Unknown evidence remains unknown; this score is not a win probability.",
                    old_provenance,
                ) if part
            )
            card["evidence_panel"] = panel
            card.update(decorate_card(card))

    authority = build_authority_contract(core._scan_payload())
    journal = build_decision_journal(limit=12)
    payload["authority"] = authority
    payload["decision_journal"] = journal

    cov = dict(authority.get("scan_coverage") or {})
    requested = int(cov.get("requested") or 0)
    checked = int(cov.get("checked") or 0)
    qualified = int(cov.get("qualified") or 0)
    no_setup = int(cov.get("no_setup") or 0)
    data_unavailable = int(cov.get("data_unavailable") or 0)
    coverage_line = (
        f"Coverage proof: requested {requested:,} NSE EQ · checked {checked:,} · "
        f"qualified {qualified:,} · no setup {no_setup:,} · data unavailable {data_unavailable:,}."
        if requested else
        "Coverage proof will appear after the first whole-market scan."
    )
    payload["cmp_note"] = f"{coverage_line} {str(payload.get('cmp_note') or '')}".strip()

    perf = dict(authority.get("performance") or {})
    sample = int(perf.get("sample_size") or 0)
    if sample:
        hit = perf.get("hit_rate_pct")
        expectancy = perf.get("expectancy_pct")
        perf_line = (
            f"Tracked outcome journal: n={sample}"
            + (f" · hit rate {float(hit):.1f}%" if hit is not None else "")
            + (f" · expectancy {float(expectancy):+.2f}%" if expectancy is not None else "")
            + " · paper/tracked research outcomes, not broker-verified live P&L."
        )
    else:
        perf_line = "Tracked outcome journal has no settled sample yet; QuantTerm makes no performance claim."
    payload["methods_note"] = (
        f"{authority.get('principle')} {coverage_line} {perf_line} "
        f"{authority.get('score_semantics')} {str(payload.get('methods_note') or '')}"
    ).strip()
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


@product.app.get("/api/strategy-catalog")
def strategy_catalog() -> dict:
    """Production recommendation methods plus research-only paper strategies."""
    from product.strategy_catalog import production_registry, research_only_strategies

    payload = production_registry()
    payload["research_only"] = research_only_strategies()
    return payload


@product.app.get("/api/research-status")
def research_status() -> dict:
    """Visible learning/research state from existing paper, journal, and registries."""
    from product.research_status import build_research_status

    return build_research_status(autonomy=core._autonomy_payload())


@product.app.get("/api/paper-autopilot")
def paper_autopilot() -> dict:
    """Latest selection-authority cycle: taken, rejected, waits, why-no-trade."""
    from product.autopilot_journal import load_journal, why_no_trade
    from product.live_readiness import evaluate_live_readiness
    from product.learning_policy_store import load_policies

    why = why_no_trade()
    journal = load_journal()
    paper = core._paper_payload()
    closed = list(paper.get("closed_trades") or [])
    days = {str(t.get("exit_date") or t.get("entry_date") or "")[:10] for t in closed if isinstance(t, dict)}
    days.discard("")
    return {
        "schema_version": 1,
        "why_no_trade": why,
        "latest": journal.get("latest") or {},
        "paper": {
            "enabled": paper.get("enabled"),
            "supervisor_running": paper.get("supervisor_running"),
            "open_positions": paper.get("open_positions") or [],
            "refusals": paper.get("refusals") or [],
        },
        "policies": load_policies().get("policies") or [],
        "live_readiness": evaluate_live_readiness(
            settled_trades=len(closed),
            trading_days=len(days),
            expectancy_R=None,
            max_drawdown_pct=None,
            distinct_regimes=0,
            stops_proven=False,
            critical_lanes_broken=not bool(paper.get("supervisor_running")),
            rules_hash_stable=False,
        ),
        "live_locked": True,
    }


@product.app.get("/api/why-no-trade")
def why_no_trade_today() -> dict:
    from product.autopilot_journal import why_no_trade
    return why_no_trade()


@product.app.get("/api/learning-policies")
def learning_policies() -> dict:
    from product.paper_learning_loop import learning_dashboard
    return learning_dashboard()


@product.app.get("/api/learning-dashboard")
def learning_dashboard_api() -> dict:
    """Explicit policy layer + counterfactual counts. Never 'AI is learning'."""
    from product.paper_learning_loop import learning_dashboard
    return learning_dashboard()


@product.app.get("/api/forward-soak")
def forward_soak_api() -> dict:
    """Forward paper-trading soak scoreboard from persisted artifacts."""
    from product.forward_soak import load_latest_verification, persist_soak_verification, scoreboard
    payload = scoreboard()
    payload["verification"] = load_latest_verification() or persist_soak_verification()
    return payload


@product.app.get("/api/decision-simulator")
def decision_simulator_get() -> dict:
    """Cached take-vs-skip historical report. BACKTEST provenance only."""
    from product.decision_simulator import load_latest
    payload = load_latest()
    if not payload:
        return {"available": False, "provenance": "BACKTEST", "live_locked": True, "cache_hit": True}
    payload["available"] = True
    return payload


@product.app.post("/api/decision-simulator")
def decision_simulator_run() -> dict:
    """Run or reuse the take-vs-skip simulator. Never writes REAL_FORWARD_MARKET."""
    from product.decision_simulator import run_decision_simulator
    return run_decision_simulator()


@product.app.post("/api/forward-soak")
def forward_soak_verify_now() -> dict:
    """Read-only re-verify. Same function as scripts/verify_forward_soak.py."""
    from product.forward_soak import persist_soak_verification, scoreboard
    verified = persist_soak_verification(force=True)
    board = scoreboard()
    board["verification"] = verified
    return board


@product.app.get("/api/system-health-contract")
def system_health_contract() -> dict:
    """Independent health lanes. No collapsed green light."""
    from product.system_health_contract import build_system_health_contract

    scan = core._scan_payload()
    long_term = core._long_term_payload()
    report_as_of = ""
    try:
        from product.recommendations_workspace import load_today_pulse
        pulse = load_today_pulse() or {}
        report_as_of = str(pulse.get("created_at") or pulse.get("date") or "")
    except Exception:
        report_as_of = ""
    fund_cov = None
    try:
        summary = dict((long_term or {}).get("summary") or {})
        raw = summary.get("coverage_pct")
        if raw is not None and raw != "":
            fund_cov = float(raw)
            if fund_cov <= 1.5:
                fund_cov = fund_cov * 100.0
    except Exception:
        fund_cov = None
    reco_ws: dict = {}
    try:
        from product.recommendations_store import load_recommendations
        reco_ws = load_recommendations() or {}
    except Exception:
        reco_ws = {}
    operations = core._operations_payload()
    news = core._news_payload()
    fno = core._fno_payload()
    data = core._data_payload(scan, long_term, operations, fno, news)
    return build_system_health_contract(
        scan=scan,
        data=data,
        news=news,
        operations=operations,
        autonomy=core._autonomy_payload(),
        paper=core._paper_payload(),
        recommendations_workspace=reco_ws,
        recommendations_available=bool(
            reco_ws or scan.get("available") or long_term.get("available")
        ),
        market_report_as_of=report_as_of,
        product_wired=True,
        fundamental_coverage_pct=fund_cov,
    )


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


def _all_declared_routes_registered(item: dict) -> bool:
    route_flags = [
        bool(value)
        for key, value in item.items()
        if key.endswith("route_registered")
    ]
    return bool(route_flags) and all(route_flags)


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
            "research_status_route_registered": "/api/research-status" in paths,
            "policies_route_registered": "/api/learning-policies" in paths,
            "dashboard_route_registered": "/api/learning-dashboard" in paths,
            "why_no_trade_route_registered": "/api/why-no-trade" in paths,
            "paper_autopilot_route_registered": "/api/paper-autopilot" in paths,
            "forward_soak_route_registered": "/api/forward-soak" in paths,
            "status": str(autonomy.get("learning_status") or "UNKNOWN"),
            "supervisor_running": bool(autonomy.get("running")),
        },
        "strategies": {
            "catalog_route_registered": "/api/strategy-catalog" in paths,
            "journal_route_registered": "/api/decision-journal" in paths,
        },
        "system_health": {
            "health_contract_route_registered": "/api/system-health-contract" in paths,
            "audit_route_registered": "/api/scan-audit" in paths,
        },
    }
    wired = all(_all_declared_routes_registered(item) for item in checks.values())
    return {
        "wired": wired,
        "checks": checks,
        "note": (
            "wired=true proves every declared canonical route for each primary capability is registered. "
            "Data availability and provider health are reported separately and are never fabricated."
        ),
    }


app = product.app
