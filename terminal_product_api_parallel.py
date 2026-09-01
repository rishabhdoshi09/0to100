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

# Stock Intelligence remains one canonical workspace. Add the already-existing
# cache-only StockResearchEngine as a deep fundamentals projection rather than
# forcing users to discover a separate hidden due-diligence product.
_base_build_stock_workspace = product.build_stock_workspace


def _stock_workspace_with_fundamental_intelligence(symbol: str) -> dict:
    base = dict(_base_build_stock_workspace(symbol) or {})
    try:
        from product.due_diligence import build_due_diligence
        deep = build_due_diligence(symbol) or {}
    except Exception as exc:
        deep = {"error": str(exc)[:200]}
    from product.fundamental_intelligence import build_fundamental_intelligence
    if deep.get("error") and not deep.get("symbol"):
        dossier = build_fundamental_intelligence(base, {})
        dossier["error"] = deep["error"]
    else:
        dossier = build_fundamental_intelligence(base, deep)
    base["fundamental_intelligence"] = dossier
    return base


product.build_stock_workspace = _stock_workspace_with_fundamental_intelligence


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
    """Decorate recommendations with explanatory evidence; never change ranking/gates."""
    from product.evidence_authority import (
        build_authority_contract,
        build_decision_journal,
        evidence_scorecard,
    )
    from product.strategy_contract import strategy_reference_for_category

    for category in payload.get("categories") or []:
        if not isinstance(category, dict):
            continue
        category_id = str(category.get("id") or "")
        strategy_ref = strategy_reference_for_category(category_id)
        category["strategy_ref"] = strategy_ref
        for card in category.get("cards") or []:
            if not isinstance(card, dict):
                continue
            scorecard = evidence_scorecard(card)
            card["evidence_scorecard"] = scorecard
            card["evidence_coverage"] = scorecard.get("coverage_pct")
            card["strategy_ref"] = dict(strategy_ref)
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
            strategy_line = (
                f"Production strategy {strategy_ref.get('strategy_id')} v{strategy_ref.get('strategy_version')} · "
                f"rules {strategy_ref.get('rules_hash')} · BACKTEST PARITY: {strategy_ref.get('backtest_parity')}."
                if strategy_ref.get("strategy_id") else
                "Production strategy identity unavailable · BACKTEST PARITY: UNVERIFIED."
            )
            panel["provenance"] = " ".join(
                part for part in (
                    headline + ".",
                    ("Components — " + components + ".") if components else "",
                    strategy_line,
                    str(strategy_ref.get("parity_reason") or ""),
                    "Unknown evidence remains unknown; this score is not a win probability.",
                    old_provenance,
                ) if part
            )
            card["evidence_panel"] = panel

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


@product.app.get("/api/strategy-registry")
def strategy_registry() -> dict:
    """Exact production strategy identities and fail-closed backtest parity."""
    from product.strategy_contract import strategy_registry_contract
    return strategy_registry_contract()


@product.app.get("/api/research-status")
def research_status() -> dict:
    """Readable proof of what Research/Learning is actually doing."""
    from product.research_status import build_research_status
    return build_research_status()


@product.app.get("/api/fundamental-intelligence/{symbol}")
def fundamental_intelligence(symbol: str) -> dict:
    """Deep, framework-specific fundamental dossier from cached research evidence."""
    clean = product.clean_symbol(symbol)
    workspace = _stock_workspace_with_fundamental_intelligence(clean)
    return dict(workspace.get("fundamental_intelligence") or {})


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
            "strategy_route_registered": "/api/strategy-registry" in paths,
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
            "fundamental_intelligence_route_registered": "/api/fundamental-intelligence/{symbol}" in paths,
        },
        "learning": {
            "operator_health_route_registered": "/api/operator-health" in paths,
            "research_status_route_registered": "/api/research-status" in paths,
            "status": str(autonomy.get("learning_status") or "UNKNOWN"),
            "supervisor_running": bool(autonomy.get("running")),
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
