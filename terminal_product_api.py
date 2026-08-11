"""Product-hardening API extensions for the QuantTerm terminal.

This module reuses the authoritative terminal API and adds product-level readiness,
one-click data bootstrap, institutional deployment gates, canonical target-portfolio,
broker-neutral execution evidence projections, and explainable single-stock intelligence.
Run Uvicorn with ``terminal_product_api:app`` so the original endpoints remain unchanged.
"""
from __future__ import annotations

from typing import Any

from fastapi import Body, HTTPException

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
    extras: dict[str, Any] = {}
    try:
        from data import corporate_actions as CA

        extras["ca"] = CA.ledger_status()
    except Exception:
        extras["ca"] = {}
    try:
        from data import universe_history as UH

        extras["universe"] = UH.ledger_status()
    except Exception:
        extras["universe"] = {}
    try:
        from data import pit_valuations as PV

        extras["pit_valuations"] = PV.ledger_status()
    except Exception:
        extras["pit_valuations"] = {}
    try:
        from scan.live_edge import profile_edge

        extras["live_edge"] = profile_edge()
    except Exception:
        extras["live_edge"] = {}
    try:
        from risk.correlation import book_correlation_report

        extras["book_correlation"] = book_correlation_report()
    except Exception:
        extras["book_correlation"] = {}
    try:
        from product.full_universe_backtest import backtest_status

        extras["signal_backtest"] = backtest_status()
    except Exception:
        extras["signal_backtest"] = {}
    return build_product_readiness(**payloads, **extras)


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
    market-health functions. Read-only; uses open paper symbols for correlation when available."""
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
    open_symbols: list[str] = []
    try:
        paper = core._paper_payload()
        open_symbols = [
            str(p.get("symbol") or "").upper()
            for p in (paper.get("open_positions") or [])
            if p.get("symbol")
        ]
    except Exception:
        open_symbols = []
    if not open_symbols:
        try:
            from risk.position_manager import review_positions

            open_symbols = [
                str(p.get("symbol") or "").upper()
                for p in (review_positions() or [])
                if p.get("symbol")
            ]
        except Exception:
            open_symbols = []
    plan = plan_for_candidate(
        record,
        capital=capital,
        regime_factor=regime_factor,
        open_symbols=open_symbols or None,
    )
    payload = plan.as_dict()
    payload.update({
        "available": True,
        "symbol": sym,
        "capital": round(capital, 0),
        "market_health": health or "unknown",
        "market_risk_factor": regime_factor,
        "open_symbols_considered": open_symbols,
    })
    return payload


@app.get("/api/trade-plan/{symbol}")
def trade_plan(symbol: str) -> dict[str, Any]:
    """Read-only risk-first plan for a scanned candidate: exact shares, ₹ risk, reward:risk,
    invalidation, book open-risk before/after and a market-throttled risk suggestion. No order is
    ever placed."""
    return _trade_plan_payload(symbol)


def _pre_trade_payload(symbol: str) -> dict[str, Any]:
    """Compose plan + market + readiness + book into an honest GO/CAUTION/NO_GO cockpit."""
    from product.pre_trade import build_pre_trade

    sym = str(symbol or "").strip().upper()
    plan = _trade_plan_payload(sym)
    market = core._market_payload()
    scan = core._scan_payload()
    record = next(
        (r for r in scan.get("records", []) if str(r.get("symbol", "")).upper() == sym),
        None,
    )
    try:
        readiness = product_readiness()
    except Exception:
        readiness = {}
    try:
        book = book_correlation()
    except Exception:
        book = {}
    try:
        paper = core._paper_payload()
    except Exception:
        paper = {}
    return build_pre_trade(
        symbol=sym,
        plan=plan,
        market=market,
        scan_record=record,
        readiness=readiness,
        book_correlation=book,
        paper=paper,
    )


@app.get("/api/pre-trade/{symbol}")
def pre_trade(symbol: str) -> dict[str, Any]:
    """Read-only pre-trade cockpit. Never places orders; GO is not a buy signal."""
    return _pre_trade_payload(symbol)


@app.get("/api/signal-backtest")
def signal_backtest_status() -> dict[str, Any]:
    """Status of the last full-universe signal backtest (research only)."""
    from product.full_universe_backtest import backtest_status

    return backtest_status()


@app.get("/api/corporate-actions")
def corporate_actions_status() -> dict[str, Any]:
    """Corporate-action ledger status. Never invents events from price gaps."""
    from data import corporate_actions as CA

    return CA.ledger_status(verify=False)


@app.post("/api/corporate-actions/from-gaps")
def corporate_actions_from_gaps(sample: int = 400) -> dict[str, Any]:
    """Export phantom-gap TODO CSV (factor/type blank). Never invents factors."""
    from data import corporate_actions as CA

    return CA.export_gap_todo(sample=max(20, min(int(sample or 400), 2000)))


@app.post("/api/corporate-actions/verify")
def corporate_actions_verify(sample: int = 80) -> dict[str, Any]:
    """Verify adjust-on-read collapsed phantom gaps. Fails closed without a ledger."""
    from data import corporate_actions as CA

    verify = CA.refresh_adjustment_verify(sample=max(20, min(int(sample or 80), 400)))
    status = CA.ledger_status(verify=False)
    status["verify"] = verify
    return status


@app.get("/api/symbols")
def symbols_directory(q: str = "", limit: int = 0) -> dict[str, Any]:
    """Full NSE equity directory for search — A→Z, not limited to scan setups.

    ``limit=0`` (default) returns the complete universe on empty query so letters
    after M/N are not truncated. Pass ``q=N`` for prefix search.
    """
    from product.symbol_directory import build_symbol_directory

    return build_symbol_directory(query=q, limit=limit)


@app.get("/api/holdings")
def holdings_book() -> dict[str, Any]:
    """Your demat holdings book (Zerodha sync or manual import). Never places orders."""
    from product.holdings_book import build_holdings_payload

    return build_holdings_payload()


@app.post("/api/holdings/sync")
def holdings_sync() -> dict[str, Any]:
    """Pull CNC holdings from Zerodha Kite when connected."""
    from product.holdings_book import sync_from_kite

    return sync_from_kite()


@app.post("/api/holdings/import")
def holdings_import(body: dict[str, Any] = Body(default_factory=dict)) -> dict[str, Any]:
    """Replace the local holdings book from a pasted/imported row list.

    Body: ``{"holdings": [{"tradingsymbol"|"symbol", "quantity", "average_price", ...}, ...]}``
    """
    from product.holdings_book import enrich_ltp, save_holdings

    rows = list(body.get("holdings") or body.get("rows") or [])
    if not rows:
        raise HTTPException(status_code=400, detail="Provide holdings: [{symbol, quantity, average_price, ...}]")
    book = save_holdings(rows, source=str(body.get("source") or "import"))
    return enrich_ltp(book)


@app.get("/api/book-correlation")
def book_correlation() -> dict[str, Any]:
    """Read-only concentration lens: how many open positions vs independent bets."""
    try:
        from risk.correlation import book_correlation_report

        return book_correlation_report()
    except Exception as exc:
        return {
            "available": False,
            "n_positions": 0,
            "n_bets": 0,
            "clusters": [],
            "biggest": None,
            "message": "Book correlation unavailable.",
            "error": str(exc),
        }


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
    """Read-only workspace (fundamentals from cache only — use fetch-fundamentals from UI)."""
    try:
        return build_stock_workspace(clean_symbol(symbol))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Stock intelligence failed: {exc}") from exc


def _fundamentals_fetch_payload(symbol: str, data: dict[str, Any], *, steps: list | None = None) -> dict[str, Any]:
    from fundamentals.resolver import coverage_score, next_actions

    return {
        "accepted": True,
        "symbol": symbol,
        "outcome": "READY",
        "source": str(data.get("_source") or "unknown"),
        "coverage": coverage_score(data),
        "sections": {
            "about": bool(data.get("about")),
            "quarterly_results": len(data.get("quarterly_results", []) or []),
            "profit_loss": len(data.get("profit_loss", []) or []),
            "balance_sheet": len(data.get("balance_sheet", []) or []),
            "cash_flow": len(data.get("cash_flow", []) or []),
            "shareholding": len(data.get("shareholding", []) or []),
            "peer_comparison": len(data.get("peer_comparison", []) or []),
        },
        "steps": list(steps or []),
        "next_actions": next_actions(symbol),
        "workspace": build_stock_workspace(symbol),
    }


@app.post("/api/stock-intelligence/{symbol}/fetch-fundamentals")
def fetch_stock_fundamentals(symbol: str, force: bool = False) -> dict[str, Any]:
    """Resolve fundamentals with per-step yield trail (Screener → Yahoo → cache → uploads)."""
    try:
        clean = clean_symbol(symbol)
        from fundamentals.lazy import ensure_deep_fundamentals, last_resolve_trail
        from fundamentals.resolver import next_actions, resolve

        if force:
            data, steps = resolve(clean, force_refresh=True, write_cache=True)
        else:
            data = ensure_deep_fundamentals(clean, force_refresh=False)
            steps = last_resolve_trail(clean)
        if data is None:
            return {
                "accepted": False,
                "symbol": clean,
                "outcome": "MISSING",
                "source": "",
                "coverage": 0,
                "sections": {},
                "steps": steps,
                "next_actions": next_actions(clean),
                "message": steps[-1]["message"] if steps else "All fundamentals sources exhausted",
            }
        return _fundamentals_fetch_payload(clean, data, steps=steps)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        # Still yield what we know so the UI can show the trail + next actions
        try:
            from fundamentals.lazy import last_resolve_trail
            from fundamentals.resolver import next_actions

            clean = clean_symbol(symbol)
            steps = last_resolve_trail(clean)
            return {
                "accepted": False,
                "symbol": clean,
                "outcome": "ERROR",
                "source": "",
                "coverage": 0,
                "sections": {},
                "steps": steps,
                "next_actions": next_actions(clean),
                "message": str(exc),
            }
        except Exception:
            raise HTTPException(status_code=502, detail=f"Fundamentals fetch failed: {exc}") from exc


@app.post("/api/stock-intelligence/{symbol}/resolve-fundamentals")
def resolve_stock_fundamentals(symbol: str, force: bool = True) -> dict[str, Any]:
    """Explicit resolve endpoint — always returns the full step trail."""
    return fetch_stock_fundamentals(symbol, force=force)


@app.post("/api/stock-intelligence/{symbol}/refresh-fundamentals")
def refresh_stock_fundamentals(symbol: str) -> dict[str, Any]:
    """Force-refresh alias for retry from the UI."""
    return fetch_stock_fundamentals(symbol, force=True)


@app.get("/api/autopilot/status")
def autopilot_status() -> dict[str, Any]:
    """Scanner autopilot arm/mode/pool + today's funnel counts."""
    try:
        from execution.autopilot import get_status, reject_funnel

        status = get_status()
        funnel = reject_funnel()
        return {
            "available": True,
            "armed": bool(status.get("armed")),
            "mode": status.get("mode"),
            "disarmed_reason": status.get("disarmed_reason") or "",
            "allocation": status.get("allocation"),
            "pool": status.get("pool"),
            "trades_today": status.get("trades_today_count"),
            "open_trades": len(status.get("open_trades") or []),
            "funnel": funnel,
            "activity": list(status.get("activity") or [])[:10],
            "places_orders": False,
            "live_enabled": status.get("live_enabled"),
        }
    except Exception as exc:
        return {"available": False, "error": str(exc), "places_orders": False}


@app.get("/api/autopilot/diagnose")
def autopilot_diagnose() -> dict[str, Any]:
    """Why autopilot is not taking trades — durable state only."""
    try:
        from execution.autopilot import diagnose_silence

        return diagnose_silence()
    except Exception as exc:
        return {
            "available": False,
            "headline": f"Diagnose failed: {exc}",
            "blockers": [str(exc)],
            "places_orders": False,
        }
