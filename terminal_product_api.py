"""Product-hardening API extensions for the QuantTerm terminal.

This module reuses the authoritative terminal API and adds product-level readiness,
one-click data bootstrap, institutional deployment gates, and explainable single-stock
intelligence. Run Uvicorn with ``terminal_product_api:app`` so the original endpoints
remain unchanged.
"""
from __future__ import annotations

from typing import Any

from fastapi import HTTPException

import terminal_api as core
from product.institutional_readiness import build_institutional_readiness
from product.product_readiness import build_product_readiness
from product.stock_workspace import build_stock_workspace, clean_symbol

app = core.app
app.version = "0.6.0"

INSTITUTIONAL_CERTIFICATIONS = core.ROOT / "logs" / "institutional_readiness" / "certifications.json"


def _current_product_payloads() -> dict[str, dict[str, Any]]:
    market = core._market_payload()
    scan = core._scan_payload()
    long_term = core._long_term_payload()
    operations = core._operations_payload()
    news = core._news_payload()
    fno = core._fno_payload()
    data = core._data_payload(scan, long_term, operations, fno, news)
    return {
        "market": market,
        "scan": scan,
        "long_term": long_term,
        "operations": operations,
        "news": news,
        "fno": fno,
        "data": data,
    }


def _institutional_capabilities() -> dict[str, bool]:
    """Read explicit capability certifications; absent or malformed entries fail closed."""
    raw = core._json_file(INSTITUTIONAL_CERTIFICATIONS, {})
    certifications = dict(raw.get("certifications", {}) or {}) if isinstance(raw, dict) else {}
    return {
        str(key): True
        for key, value in certifications.items()
        if isinstance(value, dict)
        and value.get("certified") is True
        and bool(value.get("certified_at"))
        and bool(value.get("evidence"))
    }


@app.get("/api/product-readiness")
def product_readiness() -> dict[str, Any]:
    payloads = _current_product_payloads()
    return build_product_readiness(**payloads)


@app.get("/api/institutional-readiness")
def institutional_readiness() -> dict[str, Any]:
    """Expose independent production gates without converting them into one score."""
    payloads = _current_product_payloads()
    return build_institutional_readiness(
        data=payloads["data"],
        market=payloads["market"],
        scan=payloads["scan"],
        paper=core._paper_payload(),
        autonomy=core._autonomy_payload(),
        operations=payloads["operations"],
        capabilities=_institutional_capabilities(),
    )


@app.post("/api/product-bootstrap")
def product_bootstrap() -> dict[str, Any]:
    """Queue the independent operations that make the retail product usable."""
    try:
        from operations.market_ops import (
            DATA_PREPARE,
            LANES,
            LONG_TERM_REFRESH,
            MARKET_SCAN,
            NEWS_REFRESH,
        )
        from operations.store import OperationStore

        core._ensure_ops_worker()
        store = OperationStore(core.OPS_DB)
        operations = []
        for kind in (DATA_PREPARE, NEWS_REFRESH, MARKET_SCAN, LONG_TERM_REFRESH):
            item, created = store.enqueue(
                kind,
                lane=LANES[kind],
                requested_by="product_bootstrap",
            )
            operations.append({
                "kind": kind,
                "operation_id": item.get("operation_id"),
                "status": item.get("status"),
                "created": created,
            })
        return {
            "accepted": True,
            "message": "QuantTerm preparation queued across independent data, news, scan and long-term lanes.",
            "operations": operations,
            "readiness": product_readiness(),
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Product bootstrap failed: {exc}") from exc


@app.get("/api/stock-intelligence/{symbol}")
def stock_intelligence(symbol: str) -> dict[str, Any]:
    try:
        return build_stock_workspace(clean_symbol(symbol))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Stock intelligence failed: {exc}") from exc


@app.post("/api/stock-intelligence/{symbol}/refresh-fundamentals")
def refresh_stock_fundamentals(symbol: str) -> dict[str, Any]:
    try:
        from fundamentals.fetcher import get_deep_fundamentals

        clean = clean_symbol(symbol)
        data = get_deep_fundamentals(clean, force_refresh=True)
        return {
            "accepted": True,
            "symbol": clean,
            "sections": {
                "about": bool(data.get("about")),
                "quarterly_results": len(data.get("quarterly_results", []) or []),
                "profit_loss": len(data.get("profit_loss", []) or []),
                "balance_sheet": len(data.get("balance_sheet", []) or []),
                "cash_flow": len(data.get("cash_flow", []) or []),
                "shareholding": len(data.get("shareholding", []) or []),
                "peer_comparison": len(data.get("peer_comparison", []) or []),
            },
            "workspace": build_stock_workspace(clean),
        }
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Fundamental refresh failed: {exc}") from exc
