"""One-time startup approval gate for QuantTerm decision simulation.

Startup order is explicit:
    discover/rank current best trades -> ask once -> simulate/learn.

The approval is scoped to the complete-stack startup id. The operator approves
the autonomous learning process once per startup; thesis versions remain strict
evidence/batch identities but may evolve inside that approved process without
stopping for another click.
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
    approved = bool(state.get("approved")) and same_startup
    return _write({
        "startup_id": sid,
        "startup_started_at": state.get("startup_started_at") if same_startup else _now(),
        "approved": approved,
        "approval_required": not approved,
        "approved_at": state.get("approved_at") if approved else "",
        "approved_thesis_hash": str(state.get("approved_thesis_hash") or "") if approved else "",
        "approved_scan_id": state.get("approved_scan_id") if approved else "",
        "approved_symbols": list(state.get("approved_symbols") or []) if approved else [],
        "thesis_hash": thesis_hash,
    }, path)


def _board() -> dict[str, Any]:
    try:
        from product.decision_discovery_store import (
            load as load_discovery,
            save as save_discovery,
        )
        from product.decision_service import decision_board
        from product.recommendations_store import (
            load_recommendations,
            reco_matches_scan,
        )
        from product.recommendations_workspace import build_recommendations_workspace
        from product.scan_store import load_scan
        from product.trading_thesis import manifest as thesis_manifest

        scan = dict(load_scan() or {})
        long_term: dict[str, Any] = {}
        try:
            from product.long_term_store import load_long_term_scan
            long_term = dict(load_long_term_scan() or {})
        except Exception:
            long_term = {}

        if not scan or not list(scan.get("records") or []):
            return {
                "available": False,
                "state": "SEARCHING_BEST_TRADES",
                "reason": "Whole-market scan has not produced current candidates yet.",
                "scan_scanned_at": str(scan.get("scanned_at") or ""),
                "best_trades": [],
                "thesis": {},
            }

        scan_at = str(scan.get("scanned_at") or "")
        long_term_at = str(long_term.get("scanned_at") or "")
        thesis_hash = str(thesis_manifest().get("thesis_hash") or "")

        cached = load_discovery(
            scan_scanned_at=scan_at,
            long_term_scanned_at=long_term_at,
            thesis_hash=thesis_hash,
        )
        if cached is not None:
            return cached

        # Compatibility/recovery path for older runtimes whose latest scan
        # predates discovery persistence. Normal startup discovery never pays
        # this cost in the HTTP path because run_market_scan writes the board.
        workspace = load_recommendations()
        if not reco_matches_scan(
            workspace,
            scan_scanned_at=scan_at,
            long_term_scanned_at=long_term_at,
        ):
            workspace = build_recommendations_workspace(
                scan_payload=scan,
                long_term_payload=long_term,
                refresh_technicals=False,
                settle_cases=False,
                deep_confirm=False,
                persist_ledger=False,
            )
        board = dict(decision_board(workspace=workspace, limit=40) or {})
        try:
            save_discovery(
                board,
                scan_scanned_at=scan_at,
                long_term_scanned_at=long_term_at,
                thesis_hash=thesis_hash,
            )
        except Exception:
            pass
        return board
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
    try:
        from product.desk_pipeline import scan_is_fresh
        scan_fresh = bool(scan_is_fresh())
    except Exception:
        scan_fresh = False
    discovery_ready = bool(board.get("available")) and bool(scan_id) and scan_fresh
    approved = bool(
        state.get("approved")
        and startup_id
        and str(state.get("startup_id") or "") == startup_id
    )
    approved_thesis_hash = str(state.get("approved_thesis_hash") or "")
    thesis_changed_since_approval = bool(
        approved and approved_thesis_hash and approved_thesis_hash != thesis_hash
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
        if board.get("available") and scan_id and not scan_fresh:
            message = (
                "A scan artifact exists, but authoritative market history is not "
                "current enough to present it as today's best trades. Refreshing "
                "data/search remains required before approval."
            )
        else:
            message = str(board.get("reason") or "Searching and ranking current trade candidates first.")

    visible_best_trades = list(board.get("best_trades") or []) if scan_fresh else []

    return {
        "schema_version": SCHEMA_VERSION,
        "phase": phase,
        "startup_id": startup_id,
        "approved": approved,
        "approved_at": str(state.get("approved_at") or "") if approved else "",
        "approved_thesis_hash": approved_thesis_hash if approved else "",
        "current_thesis_hash": thesis_hash,
        "thesis_changed_since_approval": thesis_changed_since_approval,
        "approval_required": not approved,
        "discovery_ready": discovery_ready,
        "scan_scanned_at": scan_id,
        "scan_fresh": scan_fresh,
        "best_trades": visible_best_trades,
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
    try:
        from product.historical_paper_loop import load_state as load_historical_state, reset_for_thesis

        hist = load_historical_state()
        if str(hist.get("thesis_hash") or "") != str(current.get("thesis_hash") or ""):
            reset_for_thesis(str(current.get("thesis_hash") or ""))
    except Exception:
        pass

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
    """Cheap hot-path startup approval check; thesis may evolve autonomously."""
    state = _read(path)
    startup_id = current_startup_id()
    if not startup_id or str(state.get("startup_id") or "") != startup_id:
        return False
    return bool(state.get("approved"))
