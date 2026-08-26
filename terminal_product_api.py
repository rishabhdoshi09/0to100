"""Product-hardening API extensions for the QuantTerm terminal.

This module reuses the authoritative terminal API and adds product-level readiness,
one-click data bootstrap, institutional deployment gates, canonical target-portfolio,
broker-neutral execution evidence projections, and explainable single-stock intelligence.
Run Uvicorn with ``terminal_product_api:app`` so the original endpoints remain unchanged.
"""
from __future__ import annotations

import os
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


def _trade_plan_payload(symbol: str) -> dict[str, Any]:
    """Risk-first plan for a scanned candidate, composed from the authoritative sizer + book-risk +
    market-health functions. Read-only; correlation stays 'unknown' here (no synchronous data fetch)."""
    from product.trade_plan import plan_for_candidate
    sym = str(symbol or "").strip().upper()
    scan = core._scan_payload()
    record = next((r for r in scan.get("records", []) if str(r.get("symbol", "")).upper() == sym), None)
    if record is None:
        return {"available": False, "symbol": sym,
                "message": "No current scan setup for this symbol. Run a fresh Momentum scan first."}
    if not (record.get("entry") and record.get("stop")):
        return {"available": False, "symbol": sym,
                "message": "This candidate has no entry/stop yet — no risk plan can be computed."}
    book = core._json_file(core.ROOT / "logs" / "intelligence" / "intel_book.json", {})
    try:
        capital = float(book.get("capital") or 0.0)
    except Exception:
        capital = 0.0
    if capital <= 0:
        try:
            from config import settings
            capital = float(settings.trading_capital)
        except Exception:
            capital = 100_000.0
    health = str(core._market_payload().get("health", "")).strip().lower()
    regime_factor = {"healthy": 1.0, "mixed": 0.75, "weak": 0.5}.get(health, 1.0)
    plan = plan_for_candidate(record, capital=capital, regime_factor=regime_factor)
    payload = plan.as_dict()
    payload.update({"available": True, "symbol": sym, "capital": round(capital, 0),
                    "market_health": health or "unknown", "market_risk_factor": regime_factor})
    return payload


@app.get("/api/trade-plan/{symbol}")
def trade_plan(symbol: str) -> dict[str, Any]:
    """Read-only risk-first plan for a scanned candidate: exact shares, ₹ risk, reward:risk,
    invalidation, book open-risk before/after and a market-throttled risk suggestion. No order is
    ever placed."""
    return _trade_plan_payload(symbol)


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


def _market_scan_is_fresh() -> bool:
    """Skip a second whole-market scan when the canonical artifact is still fresh."""
    try:
        from product.desk_pipeline import scan_is_fresh

        return bool(scan_is_fresh())
    except Exception:
        return False


def queue_product_bootstrap(*, requested_by: str = "product_bootstrap") -> dict[str, Any]:
    """Start the next due desk download only. Later steps wait until this one finishes."""
    from operations.store import OperationStore
    from product.desk_pipeline import advance_desk_pipeline

    core._ensure_ops_worker()
    store = OperationStore(core.OPS_DB)
    pipeline = advance_desk_pipeline(store, requested_by=requested_by)
    queued = pipeline.get("queued_kind")
    if queued:
        message = str(pipeline.get("message") or f"Queued {queued} — next desk step starts after this finishes.")
    else:
        message = str(pipeline.get("message") or "Desk data is current.")
    return {
        "accepted": True,
        "message": message,
        "scan_reused": bool(pipeline.get("scan_reused")),
        "sequential": True,
        "queued_kind": queued,
        "operations": list(pipeline.get("operations") or []),
        "pipeline": pipeline,
    }


def _startup_prepare_product() -> None:
    """Put data lanes to work as soon as the API process starts. Fail-open. Skip pytest."""
    if os.environ.get("PYTEST_CURRENT_TEST"):
        return
    if os.environ.get("QUANTTERM_SKIP_STARTUP_BOOTSTRAP") == "1":
        return
    try:
        queue_product_bootstrap(requested_by="api_startup")
    except Exception:
        return


@app.on_event("startup")
def _product_startup() -> None:
    _startup_prepare_product()


@app.get("/api/desk-pipeline")
def desk_pipeline_status() -> dict[str, Any]:
    """Read-only viewing-order snapshot. Does not enqueue downloads."""
    from operations.store import OperationStore
    from product.desk_pipeline import describe_desk_pipeline

    return describe_desk_pipeline(OperationStore(core.OPS_DB))


@app.post("/api/product-bootstrap")
def product_bootstrap() -> dict[str, Any]:
    """Start the next due desk download only. Later steps wait until this one finishes."""
    try:
        payload = queue_product_bootstrap(requested_by="product_bootstrap")
        payload["readiness"] = product_readiness()
        return payload
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


@app.get("/api/due-diligence/{symbol}")
def due_diligence(symbol: str) -> dict[str, Any]:
    """Second-stage research on a scanner candidate. Cache and files only — never scrapes."""
    try:
        from product.due_diligence import build_due_diligence

        return build_due_diligence(clean_symbol(symbol))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Due diligence failed: {exc}") from exc


@app.get("/api/stock-research/{symbol}")
def stock_research(symbol: str) -> dict[str, Any]:
    """Alias of due-diligence. Same StockResearchEngine, still cache-only — never scrapes."""
    return due_diligence(symbol)


@app.get("/api/stock-investigator/suggest")
def stock_investigator_suggest(q: str = "", limit: int = 8) -> dict[str, Any]:
    """Typeahead for the manual Stock Investigator. Does not fetch fundamentals."""
    try:
        from product.due_diligence import suggest_tickers

        cap = max(1, min(int(limit or 8), 20))
        matches = suggest_tickers(q, limit=cap)
        return {
            "query": q,
            "matches": matches,
            "engine": "StockResearchEngine",
            "places_orders": False,
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Investigator suggest failed: {exc}") from exc


@app.post("/api/due-diligence/{symbol}/acquire")
def acquire_due_diligence(symbol: str) -> dict[str, Any]:
    """Download official sources for this symbol, persist them, then rebuild Investigate from files."""
    try:
        from product.due_diligence import build_due_diligence
        from product.due_diligence.acquire import acquire_symbol

        clean = clean_symbol(symbol)
        acquired = acquire_symbol(clean, force=True)
        return {
            "accepted": True,
            "symbol": clean,
            "acquire": acquired,
            "report": build_due_diligence(clean),
            "places_orders": False,
        }
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Investigate acquire failed: {exc}") from exc


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
