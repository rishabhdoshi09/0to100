"""Data platform API routes for terminal_product_api."""
from __future__ import annotations

from typing import Any

from fastapi import HTTPException, Query


def providers_workspace() -> dict[str, Any]:
    from data_platform.provider_registry import registry_payload
    return registry_payload()


def coverage_workspace(symbol: str = Query("", description="Optional single symbol")) -> dict[str, Any]:
    from data_platform.coverage import audit_symbol, audit_universe
    from data_platform.security_master import supported_symbols
    if symbol.strip():
        row = audit_symbol(symbol.strip().upper())
        return {
            "generated_at": row.symbol,
            "symbol": row.symbol,
            "coverage": {
                "identity": row.identity.value,
                "price_history": row.price_history.value,
                "fundamentals": row.fundamentals.value,
                "ratios": row.ratios.value,
                "long_term_eligible": row.long_term_eligible.value,
                "reasons": row.reasons,
            },
        }
    return audit_universe(supported_symbols(limit=200), limit=120)


def jobs_workspace() -> dict[str, Any]:
    from data_platform.jobs import jobs_payload
    return jobs_payload()


def security_master_workspace(limit: int = Query(100, ge=1, le=500)) -> dict[str, Any]:
    from data_platform.security_master import security_master_payload
    return security_master_payload(limit=limit)


def symbol_ratios_workspace(symbol: str) -> dict[str, Any]:
    sym = str(symbol or "").strip().upper()
    if not sym:
        raise HTTPException(status_code=400, detail="symbol required")
    from fundamentals.cache import FundamentalsCache
    from data_platform.ratios import (
        compute_peer_average_pe,
        flatten_screener_snapshot,
        peer_pe_fundamental_metrics,
        ratios_from_fundamentals,
    )
    cache = FundamentalsCache()
    raw = cache.get(sym) or cache.get_any(sym) or {}
    flat = flatten_screener_snapshot(raw) if raw else {}

    peer_symbols: list[str] = []
    try:
        from product.scan_store import load_scan
        from product.stock_workspace import _sector_peers_from_scan

        scan = load_scan() or {}
        sector = ""
        for row in scan.get("records", []) or []:
            if str(row.get("symbol", "")).upper() == sym:
                sector = str(row.get("sector") or "")
                break
        peer_symbols = [
            str(p.get("symbol", "")).upper()
            for p in _sector_peers_from_scan(sym, sector, scan)
        ]
    except Exception:
        peer_symbols = []

    peer_stats = compute_peer_average_pe(sym, raw, peer_symbols)
    return {
        "symbol": sym,
        "ratios": ratios_from_fundamentals(sym, raw, peer_stats=peer_stats),
        "source": "fundamentals_cache+data_platform.ratios",
        "fundamentals_cached": bool(raw),
        "inputs_available": sorted(k for k, v in flat.items() if v is not None and not str(k).startswith("_")),
        "peer_average_pe": peer_stats,
    }


def fundamentals_backfill_status_workspace() -> dict[str, Any]:
    from fundamentals.lazy import cache_status

    return cache_status()


def fundamentals_backfill_run(
    scope: str = Query("nse", description="nse | nifty500 | bhav (optional maintenance)"),
    limit: int = Query(5, ge=1, le=50),
    force: bool = Query(False),
) -> dict[str, Any]:
    """Optional maintenance batch — normal use is lazy per-symbol on Stock Intelligence."""
    from fundamentals.backfill import run_fundamentals_backfill

    return run_fundamentals_backfill(scope=scope, force=force, limit=limit, resume=True)


def run_job_workspace(job_id: str) -> dict[str, Any]:
    from data_platform.jobs import run_job

    clean = str(job_id or "").strip()
    if not clean:
        raise HTTPException(status_code=400, detail="job_id required")
    return run_job(clean)


def install_data_routes(app) -> None:
    app.add_api_route("/api/data/providers", providers_workspace, methods=["GET"], name="data_providers")
    app.add_api_route("/api/data/coverage", coverage_workspace, methods=["GET"], name="data_coverage")
    app.add_api_route("/api/data/jobs", jobs_workspace, methods=["GET"], name="data_jobs")
    app.add_api_route(
        "/api/data/jobs/{job_id}/run",
        run_job_workspace,
        methods=["POST"],
        name="data_job_run",
    )
    app.add_api_route(
        "/api/data/security-master",
        security_master_workspace,
        methods=["GET"],
        name="data_security_master",
    )
    app.add_api_route(
        "/api/data/ratios/{symbol}",
        symbol_ratios_workspace,
        methods=["GET"],
        name="data_symbol_ratios",
    )
    app.add_api_route(
        "/api/data/fundamentals-backfill",
        fundamentals_backfill_status_workspace,
        methods=["GET"],
        name="fundamentals_backfill_status",
    )
    app.add_api_route(
        "/api/data/fundamentals-backfill",
        fundamentals_backfill_run,
        methods=["POST"],
        name="fundamentals_backfill_run",
    )
