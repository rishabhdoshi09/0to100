"""Read-only, cost-adjusted F&O forward-paper evidence by setup and contract.

This report never rewrites outcomes, changes rankings, promotes live-money
authority or treats counterfactual trades as real evidence.
"""
from __future__ import annotations

from collections import defaultdict
import math
from statistics import mean, median, stdev
from typing import Any, Mapping, Sequence


MIN_SEGMENT_SAMPLE = 40


def _num(value: Any) -> float | None:
    try:
        x = float(value)
        return x if math.isfinite(x) else None
    except (TypeError, ValueError, OverflowError):
        return None


def _bucket_delta(delta: float | None) -> str:
    if delta is None or delta == 0:
        return "DELTA_UNKNOWN"
    value = abs(delta)
    if value < 0.5:
        return "DELTA_LT_050"
    if value <= 0.7:
        return "DELTA_050_TO_070"
    return "DELTA_GT_070"


def _bucket_expiry(dte: float | None) -> str:
    if dte is None or dte <= 0:
        return "DTE_UNKNOWN"
    if dte <= 7:
        return "DTE_1_TO_7"
    if dte <= 21:
        return "DTE_8_TO_21"
    return "DTE_GT_21"


def _statistics(rows: Sequence[Mapping[str, float]]) -> dict[str, Any]:
    values = [float(row["r"]) for row in rows]
    n = len(values)
    if not n:
        return {
            "n": 0, "verdict": "INSUFFICIENT_FORWARD_EVIDENCE",
            "net_expectancy_R": None, "approx_95pct_lower_R": None,
            "win_rate_pct": None, "profit_factor": None,
        }
    avg = mean(values)
    stderr = stdev(values) / math.sqrt(n) if n > 1 else float("inf")
    lo = avg - 1.96 * stderr
    gains = sum(max(0.0, float(row["net_pnl"])) for row in rows)
    losses = -sum(min(0.0, float(row["net_pnl"])) for row in rows)
    # Confidence interval is an approximate descriptive diagnostic, not a
    # validated multiple-comparison or strategy promotion test.
    return {
        "n": n,
        "net_expectancy_R": round(avg, 5),
        "median_net_R": round(median(values), 5),
        "approx_95pct_lower_R": round(lo, 5) if math.isfinite(lo) else None,
        "win_rate_pct": round(100 * sum(v > 0 for v in values) / n, 2),
        "gross_pnl": round(sum(float(row["gross_pnl"]) for row in rows), 2),
        "total_costs": round(sum(float(row["costs"]) for row in rows), 2),
        "net_pnl": round(sum(float(row["net_pnl"]) for row in rows), 2),
        "profit_factor": round(gains / losses, 4) if losses > 0 else None,
        "verdict": (
            "PAPER_EDGE_CANDIDATE_REQUIRES_INDEPENDENT_REVIEW"
            if n >= MIN_SEGMENT_SAMPLE and lo > 0
            else "INSUFFICIENT_FORWARD_EVIDENCE" if n < MIN_SEGMENT_SAMPLE
            else "NO_DEMONSTRATED_POSITIVE_FORWARD_EDGE"
        ),
    }


def forward_scorecard(trades: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Include only distinct, costed, path-complete PAPER_FORWARD settlements."""
    valid: list[dict[str, float | str]] = []
    seen: set[str] = set()
    excluded: dict[str, int] = defaultdict(int)
    for row in trades:
        if not isinstance(row, Mapping):
            excluded["INVALID_ROW"] += 1
            continue
        trade_id = str(row.get("trade_id") or "")
        if not trade_id or trade_id in seen:
            excluded["MISSING_OR_DUPLICATE_TRADE_ID"] += 1
            continue
        seen.add(trade_id)
        if (row.get("evidence_lane") != "FORWARD_PAPER"
                or not row.get("production_evidence_eligible")
                or not str(row.get("cost_model_status") or "").startswith("CONFIGURED")):
            excluded["NOT_COSTED_FULLY_OBSERVED_FORWARD"] += 1
            continue
        e, stop, qty = (_num(row.get(k)) for k in ("entry_price", "stop_price", "quantity"))
        gross, costs, net = (_num(row.get(k)) for k in ("gross_pnl", "costs", "net_pnl"))
        kind = str(row.get("option_type") or "")
        if (None in (e, stop, qty, gross, costs, net)
                or e <= stop or stop <= 0 or qty <= 0 or costs < 0
                or kind not in {"CE", "PE"}):
            excluded["INVALID_COST_OR_RISK"] += 1
            continue
        if not math.isclose(gross - costs, net, abs_tol=max(0.05, abs(net) * 0.00001)):
            excluded["COST_RECONCILIATION_FAILED"] += 1
            continue
        risk = (e - stop) * qty
        valid.append({
            "r": net / risk, "gross_pnl": gross, "costs": costs, "net_pnl": net,
            "setup": "20D_BREAKOUT_LONG" if kind == "CE" else "20D_BREAKDOWN_SHORT",
            "contract": f"{kind}|{_bucket_delta(_num(row.get('entry_delta')))}|{_bucket_expiry(_num(row.get('entry_dte')))}",
        })

    by_setup: dict[str, list[dict]] = defaultdict(list)
    by_contract: dict[str, list[dict]] = defaultdict(list)
    for row in valid:
        by_setup[str(row["setup"])].append(row)
        by_contract[str(row["contract"])].append(row)
    return {
        "evidence_class": "PAPER_FORWARD_COSTED_ONLY",
        "paper_only": True, "live_execution_allowed": False,
        "score_is_win_probability": False,
        "total_input_trades": len(trades),
        "eligible_trades": len(valid),
        "excluded": dict(sorted(excluded.items())),
        "overall": _statistics(valid),
        "by_setup_type": {name: _statistics(rows) for name, rows in sorted(by_setup.items())},
        "by_contract_shape": {name: _statistics(rows) for name, rows in sorted(by_contract.items())},
        "limitations": [
            "Grouping is descriptive and cannot establish causal trading edge.",
            "Approximate confidence intervals are not multiple-testing corrected.",
            "Only observed, costed, path-complete PAPER_FORWARD trades qualify.",
            "Fill and quote realism and out-of-sample stability require independent host review.",
        ],
    }


def read_forward_scorecard(*, limit: int = 5000, store=None) -> dict[str, Any]:
    """Read the existing ledger without creating tables or triggering paper work."""
    if store is not None:
        return forward_scorecard(store.load_trades(limit=limit))
    from product.fo_paper_store import FoPaperStore
    try:
        with FoPaperStore(read_only=True) as ledger:
            return forward_scorecard(ledger.load_trades(limit=limit))
    except Exception as exc:
        # Missing/unmounted/corrupt ledger is NOT a clean zero-trade session.
        return {
            "available": False, "reason": f"PAPER_LEDGER_UNAVAILABLE:{type(exc).__name__}",
            "paper_only": True, "live_execution_allowed": False,
        }
