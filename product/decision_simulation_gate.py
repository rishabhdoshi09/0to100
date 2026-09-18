"""One-time startup approval gate for QuantTerm decision simulation.

Startup order is explicit:
    discover/rank current best trades -> ask once -> simulate/learn.

The approval is scoped to both the complete-stack startup id and the exact
production thesis hash. A material thesis change invalidates approval and
requires another explicit operator action.
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from core.runtime_paths import logs_path

SCHEMA_VERSION = 1
DEFAULT_PATH = logs_path("product/decision_simulation_gate.json")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _path(path: str | Path | None = None) -> Path:
    return Path(path) if path is not None else DEFAULT_PATH


def _read(path: str | Path | None = None) -> dict[str, Any]:
    target = _path(path)
    try:
        payload = json.loads(target.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _write(payload: Mapping[str, Any], path: str | Path | None = None) -> dict[str, Any]:
    target = _path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    data = dict(payload)
    data["schema_version"] = SCHEMA_VERSION
    data["updated_at"] = _now()
    tmp = target.with_suffix(target.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")
    os.replace(tmp, target)
    return data


def current_startup_id() -> str:
    return str(os.environ.get("QT_STARTUP_ID") or "").strip()


def begin_startup(startup_id: str | None = None, *, path: str | Path | None = None) -> dict[str, Any]:
    sid = str(startup_id or current_startup_id() or "").strip()
    if not sid:
        return status(path=path)

    from product.trading_thesis import manifest

    thesis = manifest()
    thesis_hash = str(thesis.get("thesis_hash") or "")
    state = _read(path)
    same_startup = str(state.get("startup_id") or "") == sid
    same_thesis = str(state.get("approved_thesis_hash") or "") == thesis_hash
    approved = bool(state.get("approved")) and same_startup and same_thesis
    return _write({
        "startup_id": sid,
        "startup_started_at": state.get("startup_started_at") if same_startup else _now(),
        "approved": approved,
        "approved_at": state.get("approved_at") if approved else "",
        "approved_thesis_hash": thesis_hash if approved else "",
        "approved_scan_id": state.get("approved_scan_id") if approved else "",
        "approved_symbols": list(state.get("approved_symbols") or []) if approved else [],
        "thesis_hash": thesis_hash,
    }, path)


def _board() -> dict[str, Any]:
    try:
        from product.decision_service import decision_board
        return dict(decision_board(limit=40) or {})
    except Exception as exc:
        return {
            "available": False,
            "state": "DISCOVERY_ERROR",
            "reason": str(exc)[:240],
            "scan_scanned_at": "",
            "best_trades": [],
            "thesis": {},
        }


def status(*, path: str | Path | None = None) -> dict[str, Any]:
    from product.trading_thesis import manifest

    state = _read(path)
    startup_id = current_startup_id() or str(state.get("startup_id") or "")
    thesis = manifest()
    thesis_hash = str(thesis.get("thesis_hash") or "")
    board = _board()
    scan_id = str(board.get("scan_scanned_at") or "")
    discovery_ready = bool(board.get("available")) and bool(scan_id)
    approved = bool(
        state.get("approved")
        and str(state.get("startup_id") or "") == startup_id
        and str(state.get("approved_thesis_hash") or "") == thesis_hash
    )
    if approved:
        phase = "APPROVED"
        message = "Decision Simulation approved for this startup and thesis."
    elif discovery_ready:
        phase = "AWAITING_APPROVAL"
        message = (
            "Current best-trade search is complete. Review the shortlist, then "
            "approve Decision Simulation once."
        )
    else:
        phase = "SEARCHING_BEST_TRADES"
        message = str(board.get("reason") or "Searching and ranking current trade candidates first.")

    return {
        "schema_version": SCHEMA_VERSION,
        "phase": phase,
        "startup_id": startup_id,
        "approved": approved,
        "approved_at": str(state.get("approved_at") or "") if approved else "",
        "approval_required": not approved,
        "discovery_ready": discovery_ready,
        "scan_scanned_at": scan_id,
        "best_trades": list(board.get("best_trades") or []),
        "decision_count": len(list(board.get("decisions") or [])),
        "actionable": int(board.get("actionable") or 0),
        "thesis": thesis,
        "thesis_hash": thesis_hash,
        "message": message,
        "live_locked": True,
        "simulation_scope": ["PAPER_FORWARD", "HISTORICAL_REPLAY"],
    }


def approve(*, path: str | Path | None = None) -> dict[str, Any]:
    current = status(path=path)
    if not current.get("discovery_ready"):
        return {
            **current,
            "accepted": False,
            "reason": "BEST_TRADE_DISCOVERY_NOT_READY",
        }

    startup_id = str(current.get("startup_id") or current_startup_id() or "")
    if not startup_id:
        return {
            **current,
            "accepted": False,
            "reason": "STARTUP_ID_UNAVAILABLE",
        }

    best = list(current.get("best_trades") or [])
    _write({
        "startup_id": startup_id,
        "startup_started_at": _read(path).get("startup_started_at") or _now(),
        "approved": True,
        "approved_at": _now(),
        "approved_thesis_hash": str(current.get("thesis_hash") or ""),
        "approved_scan_id": str(current.get("scan_scanned_at") or ""),
        "approved_symbols": [str(row.get("symbol") or "") for row in best[:5]],
        "thesis_hash": str(current.get("thesis_hash") or ""),
    }, path)
    return {**status(path=path), "accepted": True, "reason": "APPROVED"}


def is_approved(*, path: str | Path | None = None) -> bool:
    return bool(status(path=path).get("approved"))
