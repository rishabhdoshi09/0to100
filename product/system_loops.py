"""Read-only system loops: decision journal + holdings rotation advice."""
from __future__ import annotations

from typing import Any


def decision_journal_workspace() -> dict[str, Any]:
    """Taken vs rejected + calibration. Thin samples stay claim-free."""
    try:
        from core.decision_journal import calibration_report, decision_report
    except Exception as exc:
        return {
            "available": False,
            "message": f"Decision journal unread ({exc})",
            "decision": {},
            "calibration": {},
        }
    decision = decision_report()
    calibration = calibration_report()
    taken_n = int((decision.get("taken") or {}).get("n") or 0)
    rejected_n = int((decision.get("rejected") or {}).get("n") or 0)
    thin = not bool(decision.get("verdict"))
    return {
        "available": True,
        "advice_only": True,
        "thin": thin,
        "taken_n": taken_n,
        "rejected_n": rejected_n,
        "decision": decision,
        "calibration": calibration,
        "message": (
            decision.get("verdict")
            or "No claim yet — taken and rejected each need ≥10 resolved outcomes. Logging continues."
        ),
        "calibration_message": calibration.get("verdict") or "",
    }


def portfolio_intel_workspace() -> dict[str, Any]:
    """Advice-only rotation. Never places an order. Thin EV → no swap claim."""
    try:
        from core.portfolio_intel import MIN_EV_GAP_PCT, daily_review
    except Exception as exc:
        return {
            "available": False,
            "advice_only": True,
            "message": f"Portfolio intelligence unread ({exc})",
            "swap": None,
        }
    review = daily_review() or {}
    review["available"] = True
    review["advice_only"] = True
    review["min_ev_gap_pct"] = MIN_EV_GAP_PCT
    swap = review.get("swap")
    n_holdings = int(review.get("n_holdings") or 0)
    if n_holdings <= 0:
        review["message"] = (
            "No open book to compare. Sync holdings or wait for a paper position — "
            "QuantTerm will not invent a rotation."
        )
    elif swap:
        review["message"] = (
            f"{swap.get('note') or 'Opportunity-cost gap vs a stronger idea.'} "
            "Advice only — this never rotates capital."
        )
    else:
        review["message"] = (
            f"No swap claim. A replacement needs an EV gap of at least {MIN_EV_GAP_PCT}pp "
            "after costs, and both sides need a real EV (n ≥ 30)."
        )
    return review


def install_system_loop_routes(app) -> None:
    app.add_api_route(
        "/api/decision-journal",
        decision_journal_workspace,
        methods=["GET"],
        name="decision_journal",
    )
    app.add_api_route(
        "/api/portfolio-intel",
        portfolio_intel_workspace,
        methods=["GET"],
        name="portfolio_intel",
    )
    from product.session_honesty import session_payload

    app.add_api_route(
        "/api/session-honesty",
        session_payload,
        methods=["GET"],
        name="session_honesty",
    )
