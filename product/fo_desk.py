"""Shared read-only F&O Home/Telegram projection; never creates a trade."""
from __future__ import annotations

from collections import Counter
from datetime import datetime
import math
from typing import Any, Mapping
from zoneinfo import ZoneInfo


def _positive(value: Any) -> bool:
    try:
        return math.isfinite(float(value)) and float(value) > 0
    except (TypeError, ValueError):
        return False


def build_fo_desk(
    directional: Mapping[str, Any] | None,
    paper: Mapping[str, Any] | None,
    *,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Keep saved candidates distinct from actual durable PAPER positions."""
    scan = dict(directional or {})
    book = dict(paper or {})
    now = now or datetime.now(ZoneInfo("Asia/Kolkata"))
    if now.tzinfo is not None:
        now = now.astimezone(ZoneInfo("Asia/Kolkata"))
    day = now.date().isoformat()
    as_of = str(scan.get("as_of") or "")
    status = str(scan.get("status") or "NOT_RUN")
    reason = str(scan.get("reason") or scan.get("code") or scan.get("decision") or status)
    fresh = bool(scan.get("available") and status == "READY" and as_of == day)
    if scan and as_of and as_of != day:
        status, reason = "STALE", "SAVED_FNO_SCAN_IS_FROM_ANOTHER_SESSION"
    elif status == "READY" and not as_of:
        status, reason = "UNAVAILABLE", "FNO_SCAN_SESSION_UNAVAILABLE"
    generated = scan.get("generated_at") or scan.get("cache_mtime")
    if fresh and not _positive(generated):
        fresh = False
        status, reason = "UNAVAILABLE", "FNO_SCAN_TIMESTAMP_UNAVAILABLE"
    elif fresh:
        age = now.replace(tzinfo=ZoneInfo("Asia/Kolkata")).timestamp() - float(generated)
        if age < -30 or age > 1800:
            fresh = False
            status, reason = "STALE", "SAVED_FNO_SCAN_EXPIRED"
    candidates = []
    if fresh:
        for row in scan.get("candidates") or []:
            if not isinstance(row, Mapping) or row.get("decision") != "PAPER_OPTION_CANDIDATE":
                continue
            contract = row.get("selected_contract") or {}
            if not isinstance(contract, Mapping):
                continue
            plan = contract.get("trade_plan") or {}
            if not isinstance(plan, Mapping):
                continue
            expiry = str(contract.get("expiry") or "")
            if (not row.get("symbol") or not contract.get("symbol")
                    or contract.get("option_type") not in {"CE", "PE"}
                    or not contract.get("eligible") or not _positive(contract.get("lot_size"))
                    or len(expiry) != 10 or expiry < day
                    or not all(_positive(v) for v in (
                        contract.get("premium"), plan.get("entry"),
                        plan.get("stop"), plan.get("target"),
                    ))
                    or not (float(plan["stop"]) < float(plan["entry"]) < float(plan["target"]))):
                continue
            candidates.append({**row, "paper_only": True, "live_execution_allowed": False})
    blockers = Counter()
    for row in scan.get("considered") or []:
        if isinstance(row, Mapping) and not row.get("eligible"):
            blockers[str(row.get("reason") or "PREFILTER_NOT_PASSED")] += 1
    for row in (scan.get("decisions") or []) + (scan.get("deep_failures") or []):
        if isinstance(row, Mapping) and row.get("decision") != "PAPER_OPTION_CANDIDATE":
            blockers[str(row.get("reason") or row.get("code") or "OPTION_GATES_NOT_PASSED")] += 1
    if fresh and not candidates and scan.get("candidates"):
        reason = "NO_COMPLETE_CURRENT_OPTION_PLAN"
    return {
        "status": status, "reason": reason, "as_of": as_of,
        "generated_at": generated,
        "candidates": candidates[:5], "candidate_count": len(candidates),
        "blockers": [{"reason": key, "count": count} for key, count in blockers.most_common(5)],
        "paper_available": bool(book.get("available")),
        "paper_error": str(book.get("error") or book.get("reason") or ""),
        "open_positions": list(book.get("open_positions") or []),
        "recent_closed_trades": list(book.get("recent_closed_trades") or book.get("settled") or [])[:5],
        "paper_only": True, "live_execution_allowed": False,
    }
