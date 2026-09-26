"""Read-only US market/PAPER status for the canonical QuantTerm product."""
from __future__ import annotations

from typing import Any


def status() -> dict[str, Any]:
    scan: dict[str, Any] = {}
    try:
        from scan.us_scanner import persisted_us_scan
        scan = dict(persisted_us_scan() or {})
    except Exception:
        scan = {}

    paper: dict[str, Any] = {}
    report: dict[str, Any] = {}
    funnel: dict[str, Any] = {}
    try:
        from execution.us_autopilot import get_status, report_card, reject_funnel
        paper = dict(get_status() or {})
        report = dict(report_card() or {})
        funnel = dict(reject_funnel() or {})
    except Exception:
        paper, report, funnel = {}, {}, {}

    learning: dict[str, Any] = {}
    try:
        from product.us_learning import dashboard
        learning = dict(dashboard() or {})
    except Exception:
        learning = {}

    market_open = False
    try:
        from data.us_data import us_market_open
        market_open = bool(us_market_open())
    except Exception:
        market_open = False

    records = [
        dict(r) for r in (scan.get("records") or [])
        if isinstance(r, dict)
    ]
    top = sorted(
        records,
        key=lambda r: float(
            r.get("learned_rank_score")
            or r.get("conviction_rank")
            or r.get("score")
            or 0.0
        ),
        reverse=True,
    )[:5]

    pit_snapshots: list[str] = []
    try:
        from data.us_universe import available_us_universe_snapshots
        pit_snapshots = list(available_us_universe_snapshots() or [])
    except Exception:
        pit_snapshots = []

    return {
        "schema_version": 1,
        "market": "US",
        "market_open": market_open,
        "scan": {
            "status": str(scan.get("status") or "idle"),
            "scope": str(scan.get("scope") or "All"),
            "scanned_at": str(scan.get("scanned_at") or ""),
            "count": int(scan.get("count") or len(records)),
        },
        "top_setups": top,
        "paper": {
            "armed": bool(paper.get("armed")),
            "allocation": paper.get("allocation"),
            "pool": paper.get("pool"),
            "available": paper.get("available"),
            "open_trades": list(paper.get("open_trades") or []),
            "trades_today": int(paper.get("trades_today_count") or 0),
            "preset": paper.get("preset"),
            "disarmed_reason": paper.get("disarmed_reason"),
            "report": report,
            "rejection_funnel": funnel,
        },
        "learning": learning,
        "parity": {
            "state": "FORWARD_PAPER_PARITY",
            "autonomous_scan": True,
            "paper_auto_execution": True,
            "forward_outcome_settlement": True,
            "forward_counterfactuals": True,
            "forward_learning": True,
            "learning_can_reorder_paper_only": True,
            "historical_pit_replay": False,
            "pit_universe_archive_started": bool(pit_snapshots),
            "pit_universe_archive_snapshots": len(pit_snapshots),
            "pit_universe_archive_first": pit_snapshots[0] if pit_snapshots else "",
            "pit_universe_archive_latest": pit_snapshots[-1] if pit_snapshots else "",
            "historical_pit_blocker": (
                "Current US sources provide present-day listed/index membership and "
                "daily OHLCV, but not a trustworthy point-in-time historical universe "
                "including delistings. Historical replay is intentionally not promoted "
                "until a PIT US universe/history source is connected."
            ),
            "survivorship_biased_backtest_allowed": False,
            "live_money_parity": False,
        },
        "paper_only": True,
        "live_locked": True,
        "live_execution_available": False,
    }
