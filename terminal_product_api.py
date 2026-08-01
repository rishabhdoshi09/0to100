"""Product-hardening API extensions for the QuantTerm terminal.

This module reuses the authoritative terminal API and adds product-level readiness,
one-click data bootstrap, institutional deployment gates, canonical target-portfolio,
broker-neutral execution evidence projections, and explainable single-stock intelligence.
Run Uvicorn with ``terminal_product_api:app`` so the original endpoints remain unchanged.
"""
from __future__ import annotations

from typing import Any

from fastapi import HTTPException

import terminal_api as core
from product.institutional_readiness import build_institutional_readiness
from product.observer_api import install as install_observer_api, observer_payload
from product.product_readiness import build_product_readiness
from product.stock_workspace import build_stock_workspace, clean_symbol

app = core.app
app.version = "0.11.0"
install_observer_api(app)

INSTITUTIONAL_CERTIFICATIONS = (
    core.ROOT / "logs" / "institutional_readiness" / "certifications.json"
)
TARGET_EVENT_STORE = core.ROOT / "logs" / "intelligence" / "events.jsonl"
OMS_DB = core.ROOT / "logs" / "oms" / "orders.db"
RISK_DB = core.ROOT / "logs" / "risk" / "decisions.db"
RECONCILIATION_DB = core.ROOT / "logs" / "reconciliation" / "reports.db"
PROTECTION_DB = core.ROOT / "logs" / "protection" / "plans.db"
TCA_DB = core.ROOT / "logs" / "tca" / "assessments.db"


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


def _target_portfolio_payload() -> dict[str, Any]:
    """Project the latest canonical target portfolio without inventing missing state."""
    try:
        from research.intelligence.event_store import EventStore

        if not TARGET_EVENT_STORE.exists():
            return {
                "available": False,
                "portfolio": {},
                "positions": [],
                "message": "No target portfolio has been persisted yet.",
            }
        store = EventStore(TARGET_EVENT_STORE)
        portfolio = store.latest_target_portfolio()
        if portfolio is None:
            return {
                "available": False,
                "portfolio": {},
                "positions": [],
                "message": "No target portfolio has been persisted yet.",
            }
        positions = store.target_positions_for(portfolio)
        return {
            "available": True,
            "portfolio": portfolio.as_dict(),
            "positions": [position.as_dict() for position in positions],
            "summary": {
                "current_positions": portfolio.current_position_count,
                "target_positions": portfolio.target_position_count,
                "executable_changes": len(portfolio.executable_position_ids),
                "blocked_changes": len(portfolio.blocked_position_ids),
                "current_open_risk_pct": portfolio.current_open_risk_pct,
                "pending_open_risk_pct": portfolio.pending_open_risk_pct,
                "target_open_risk_pct": portfolio.target_open_risk_pct,
                "available_cash": portfolio.available_cash,
            },
        }
    except Exception as exc:
        return {
            "available": False,
            "portfolio": {},
            "positions": [],
            "message": "Target portfolio projection is unavailable.",
            "error": str(exc),
        }


def _oms_payload() -> dict[str, Any]:
    """Project durable OMS state. This endpoint never creates or advances an order."""
    try:
        if not OMS_DB.exists():
            return {
                "available": False,
                "summary": {"orders": 0, "by_status": {}, "recovery_required": []},
                "orders": [],
                "message": "No durable OMS database exists yet.",
            }
        from execution.oms.store import OmsStore

        store = OmsStore(OMS_DB)
        orders = store.list_orders()
        return {
            "available": True,
            "summary": store.summary(),
            "orders": [order.as_dict() for order in orders[-100:]],
            "broker_connected": False,
            "submission_enabled": False,
            "message": (
                "OMS is in broker-neutral shadow mode; no broker submission path is connected."
            ),
        }
    except Exception as exc:
        return {
            "available": False,
            "summary": {"orders": 0, "by_status": {}, "recovery_required": []},
            "orders": [],
            "message": "OMS projection is unavailable.",
            "error": str(exc),
        }


def _risk_governor_payload() -> dict[str, Any]:
    """Project persisted governor decisions without claiming reconciled live risk state."""
    try:
        if not RISK_DB.exists():
            return {
                "available": False,
                "summary": {"decisions": 0, "by_action": {}},
                "mode": "SHADOW",
                "authoritative_state_connected": False,
                "message": "No independent Risk Governor decisions have been persisted yet.",
            }
        from risk.governor_store import RiskDecisionStore

        store = RiskDecisionStore(RISK_DB)
        return {
            "available": True,
            "summary": store.summary(),
            "mode": "SHADOW",
            "authoritative_state_connected": False,
            "certified_for_live": False,
            "message": (
                "The governor is deterministic and independent, but remains shadow-only until "
                "broker positions, orders, cash, margin and protection are continuously reconciled."
            ),
        }
    except Exception as exc:
        return {
            "available": False,
            "summary": {"decisions": 0, "by_action": {}},
            "mode": "SHADOW",
            "authoritative_state_connected": False,
            "message": "Risk Governor projection is unavailable.",
            "error": str(exc),
        }


def _reconciliation_payload() -> dict[str, Any]:
    """Project persisted reconciliation evidence without opening a broker connection."""
    try:
        if not RECONCILIATION_DB.exists():
            return {
                "available": False,
                "summary": {
                    "reports": 0,
                    "by_status": {},
                    "latest_status": "",
                    "entry_freeze_required": True,
                },
                "latest": {},
                "certified_for_live": False,
                "broker_snapshot_connected": False,
                "message": "No reconciliation report database exists yet.",
            }
        from execution.reconciliation.store import ReconciliationReportStore

        store = ReconciliationReportStore(RECONCILIATION_DB)
        latest = store.latest()
        if latest is None:
            return {
                "available": False,
                "summary": store.summary(),
                "latest": {},
                "certified_for_live": False,
                "broker_snapshot_connected": False,
                "message": "No reconciliation report has been persisted yet.",
            }
        return {
            "available": True,
            "summary": store.summary(),
            "latest": latest,
            "certified_for_live": False,
            "broker_snapshot_connected": False,
            "message": (
                "Reconciliation evidence is persisted, but continuous authoritative broker "
                "observation is not certified for live operation."
            ),
        }
    except Exception as exc:
        return {
            "available": False,
            "summary": {
                "reports": 0,
                "by_status": {},
                "latest_status": "",
                "entry_freeze_required": True,
            },
            "latest": {},
            "certified_for_live": False,
            "broker_snapshot_connected": False,
            "message": "Reconciliation projection is unavailable.",
            "error": str(exc),
        }


def _protection_payload() -> dict[str, Any]:
    """Project exact quantity-aware protection plans without touching exchange state."""
    try:
        if not PROTECTION_DB.exists():
            return {
                "available": False,
                "summary": {
                    "plans": 0,
                    "by_status": {},
                    "fully_protected": 0,
                    "unsafe_plan_ids": [],
                    "entry_freeze_required": True,
                },
                "plans": [],
                "exchange_adapter_connected": False,
                "certified_for_live": False,
                "message": "No protection plan database exists yet.",
            }
        from execution.protection.store import ProtectionStore

        store = ProtectionStore(PROTECTION_DB)
        plans = store.list_plans()
        return {
            "available": True,
            "summary": store.summary(),
            "plans": [plan.as_dict() for plan in plans[-100:]],
            "exchange_adapter_connected": False,
            "certified_for_live": False,
            "message": (
                "Protection plans are durable and quantity-aware, but exchange-side mutation "
                "and continuous verification remain disabled."
            ),
        }
    except Exception as exc:
        return {
            "available": False,
            "summary": {
                "plans": 0,
                "by_status": {},
                "fully_protected": 0,
                "unsafe_plan_ids": [],
                "entry_freeze_required": True,
            },
            "plans": [],
            "exchange_adapter_connected": False,
            "certified_for_live": False,
            "message": "Protection projection is unavailable.",
            "error": str(exc),
        }


def _tca_payload() -> dict[str, Any]:
    """Project only persisted transaction-cost evidence; never estimate missing fills."""
    try:
        if not TCA_DB.exists():
            return {
                "available": False,
                "summary": {
                    "assessments": 0,
                    "complete_assessments": 0,
                    "total_implementation_shortfall": 0.0,
                    "average_implementation_shortfall_bps": 0.0,
                },
                "assessments": [],
                "live_fill_feed_connected": False,
                "message": "No transaction-cost assessment database exists yet.",
            }
        from execution.tca.store import TcaStore

        store = TcaStore(TCA_DB)
        assessments = store.list_assessments()
        return {
            "available": True,
            "summary": store.summary(),
            "assessments": assessments[-100:],
            "live_fill_feed_connected": False,
            "message": (
                "Only persisted assessments are shown; missing timestamps, fills or fees are "
                "not invented."
            ),
        }
    except Exception as exc:
        return {
            "available": False,
            "summary": {
                "assessments": 0,
                "complete_assessments": 0,
                "total_implementation_shortfall": 0.0,
                "average_implementation_shortfall_bps": 0.0,
            },
            "assessments": [],
            "live_fill_feed_connected": False,
            "message": "Transaction-cost projection is unavailable.",
            "error": str(exc),
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


@app.get("/api/target-portfolio")
def target_portfolio() -> dict[str, Any]:
    """Return the latest immutable target-versus-current portfolio projection."""
    return _target_portfolio_payload()


@app.get("/api/oms")
def oms_status() -> dict[str, Any]:
    """Return read-only durable order lifecycle state."""
    return _oms_payload()


@app.get("/api/risk-governor")
def risk_governor_status() -> dict[str, Any]:
    """Return read-only independent risk-decision state."""
    return _risk_governor_payload()


@app.get("/api/reconciliation")
def reconciliation_status() -> dict[str, Any]:
    """Return the latest persisted broker reconciliation evidence."""
    return _reconciliation_payload()


@app.get("/api/protection")
def protection_status() -> dict[str, Any]:
    """Return durable exchange-protection requirements and verification state."""
    return _protection_payload()


@app.get("/api/tca")
def tca_status() -> dict[str, Any]:
    """Return persisted transaction-cost and execution-latency assessments."""
    return _tca_payload()


@app.get("/api/product-dashboard")
def product_dashboard() -> dict[str, Any]:
    """Compose existing read-only dashboard state with institutional projections."""
    dashboard_payload = core.dashboard()
    dashboard_payload["product_readiness"] = product_readiness()
    dashboard_payload["institutional_readiness"] = institutional_readiness()
    dashboard_payload["target_portfolio"] = target_portfolio()
    dashboard_payload["oms"] = oms_status()
    dashboard_payload["risk_governor"] = risk_governor_status()
    dashboard_payload["reconciliation"] = reconciliation_status()
    dashboard_payload["protection"] = protection_status()
    dashboard_payload["tca"] = tca_status()
    dashboard_payload["broker_observer"] = observer_payload()
    return dashboard_payload


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
            operations.append(
                {
                    "kind": kind,
                    "operation_id": item.get("operation_id"),
                    "status": item.get("status"),
                    "created": created,
                }
            )
        return {
            "accepted": True,
            "message": (
                "QuantTerm preparation queued across independent data, news, scan and "
                "long-term lanes."
            ),
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
