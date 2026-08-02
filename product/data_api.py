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
    from data_platform.ratios import ratios_from_fundamentals
    raw = FundamentalsCache().get(sym) or {}
    return {
        "symbol": sym,
        "ratios": ratios_from_fundamentals(sym, raw),
        "source": "fundamentals_cache+data_platform.ratios",
    }


def fundamentals_backfill_status_workspace() -> dict[str, Any]:
    from fundamentals.backfill import backfill_status
    return backfill_status()


def fundamentals_backfill_run(
    scope: str = Query("nse", description="nse | nifty500 | bhav"),
    limit: int = Query(50, ge=1, le=500),
    force: bool = Query(False),
) -> dict[str, Any]:
    """Optional bounded batch (maintenance). Per-symbol fundamentals load on Stock Intelligence refresh."""
    from fundamentals.backfill import run_fundamentals_backfill
    return run_fundamentals_backfill(scope=scope, force=force, limit=limit, resume=True)


def install_data_routes(app) -> None:
    app.add_api_route("/api/data/providers", providers_workspace, methods=["GET"], name="data_providers")
    app.add_api_route("/api/data/coverage", coverage_workspace, methods=["GET"], name="data_coverage")
    app.add_api_route("/api/data/jobs", jobs_workspace, methods=["GET"], name="data_jobs")
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
