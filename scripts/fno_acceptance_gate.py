#!/usr/bin/env python3
"""Local, read-only F&O acceptance diagnostics; NEVER starts a scan or order.

This is evidence collection on the installed Mac, NOT self-certification.
HOLD is the correct result without a real current-session scan, settled and
fully costed trades, and observable Telegram delivery.
"""
from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path
import sys
from zoneinfo import ZoneInfo

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core.runtime_paths import logs_path
from product.fno_forward_scorecard import forward_scorecard
from product.fo_paper_store import FoPaperStore
from research.autonomy import default_root


def _read(path: Path) -> dict:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


def audit() -> dict:
    today = datetime.now(ZoneInfo("Asia/Kolkata")).date().isoformat()
    scan = _read(logs_path("product", "fo_directional.json"))
    telegram = _read(default_root() / "telegram_notifications.json")
    sent_today = set((telegram.get("sent") or {}).get(today) or [])
    findings = []
    scan_ok = (
        str(scan.get("as_of") or "") == today
        and str(scan.get("status") or "") == "READY"
        and bool(scan.get("available"))
        and isinstance(scan.get("considered"), list)
    )
    findings.append({"name": "current_fno_scan", "passed": scan_ok,
                     "reason": str(scan.get("code") or scan.get("reason") or "")})

    try:
        with FoPaperStore(read_only=True) as store:
            positions = store.load_positions()
            trades = store.load_trades(limit=5000)
        ledger_ok = True
    except Exception as exc:
        positions, trades = [], []
        ledger_ok = False
        findings.append({"name": "paper_ledger_readable", "passed": False,
                         "reason": type(exc).__name__})
    if ledger_ok:
        findings.append({"name": "paper_ledger_readable", "passed": True})
    scorecard = forward_scorecard(trades)

    # Closed trades are only proof when real entries and exits have persisted.
    settled_ok = scorecard["eligible_trades"] > 0
    findings.append({"name": "fully_costed_observed_settlement", "passed": settled_ok,
                     "count": scorecard["eligible_trades"]})

    events = [row for row in trades if str(row.get("settled_at") or "")[:10] == today]
    observed = all(
        f"fno_paper_closed:{row['trade_id']}" in sent_today
        for row in events if row.get("trade_id")
    ) and bool(events)
    findings.append({"name": "today_telegram_close_keys", "passed": observed,
                     "settled_today": len(events)})
    # Equal-day OPEN/CLOSE is a strict acceptance check; an overnight trade may
    # be valid but does not prove same-day delivery from a pruned notifier file.
    entry_sent = bool(
        [key for key in sent_today if str(key).startswith("fno_paper_opened:")]
    )
    findings.append({"name": "today_telegram_open_keys", "passed": entry_sent})

    try:
        from risk.portfolio_risk import portfolio_risk_report
        report = portfolio_risk_report()
        risk_ok = report["open_risk_pct"] <= 5.0
        findings.append({"name": "combined_equity_fno_risk_under_5pct",
                         "passed": risk_ok,
                         "open_risk_pct": report["open_risk_pct"],
                         "positions": report["n_positions"]})
    except Exception as exc:
        findings.append({"name": "combined_equity_fno_risk_under_5pct",
                         "passed": False, "reason": type(exc).__name__})

    return {
        "verdict": "PASS_DIAGNOSTICS_ONLY" if all(x["passed"] for x in findings) else "HOLD",
        "as_of": today, "paper_only": True, "live_execution_allowed": False,
        "open_fno_paper_positions": len(positions),
        "findings": findings, "forward_evidence": scorecard,
        "requires_manual_host_soak_and_order_trace_review": True,
    }


if __name__ == "__main__":
    payload = audit()
    print(json.dumps(payload, indent=2, sort_keys=True))
    sys.exit(0 if payload["verdict"] == "PASS_DIAGNOSTICS_ONLY" else 2)
