"""Persisted-first Recommendations API liveness.

Opening Recommendations must remain a read path.  Expensive projection work is rebuilt in the
background from already-persisted scan/funds artifacts and never from broker/network calls.

The rebuild claim is protected by an OS file lock, not merely a Python ``threading.Lock``.  That
matters when Uvicorn is started with more than one worker: only one process may build a given
artifact at a time.  The lock is released automatically by the OS if the process dies; a small
persisted state file tells every API worker whether the current target is RUNNING, FAILED or
SUCCEEDED.  A stale build re-checks its input generation before writing so an older projection
cannot overwrite a newer scan.
"""
from __future__ import annotations

import json
import os
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

_INSTALLED = False
_BUILD_LOCK = threading.Lock()
_BUILD_THREAD: threading.Thread | None = None
_RETRY_AFTER_FAILURE_S = 30.0
_ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_LOCK_PATH = _ROOT / "logs" / "product" / "recommendations_rebuild.lock"
_DEFAULT_STATE_PATH = _ROOT / "logs" / "product" / "recommendations_rebuild_state.json"


def _lock_path() -> Path:
    return Path(os.environ.get("QT_RECOMMENDATIONS_REBUILD_LOCK", str(_DEFAULT_LOCK_PATH)))


def _state_path() -> Path:
    return Path(os.environ.get("QT_RECOMMENDATIONS_REBUILD_STATE", str(_DEFAULT_STATE_PATH)))


def _target(scan: Mapping[str, Any], long_term: Mapping[str, Any]) -> dict[str, str]:
    return {
        "scan_scanned_at": str(scan.get("scanned_at") or ""),
        "long_term_scanned_at": str(long_term.get("scanned_at") or ""),
    }


def _same_target(state: Mapping[str, Any] | None, scan: Mapping[str, Any], long_term: Mapping[str, Any]) -> bool:
    wanted = _target(scan, long_term)
    return bool(state) and all(str(state.get(k) or "") == v for k, v in wanted.items())


def _read_state() -> dict[str, Any]:
    try:
        payload = json.loads(_state_path().read_text(encoding="utf-8"))
        return payload if isinstance(payload, dict) else {}
    except Exception:
        return {}


def _write_state(payload: Mapping[str, Any]) -> None:
    target = _state_path()
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(f".{target.name}.{os.getpid()}.{time.time_ns()}.tmp")
    try:
        tmp.write_text(json.dumps(dict(payload), indent=2, default=str), encoding="utf-8")
        os.replace(tmp, target)
    finally:
        try:
            tmp.unlink(missing_ok=True)
        except Exception:
            pass


def _claim_rebuild(scan: Mapping[str, Any], long_term: Mapping[str, Any]):
    """Acquire one cross-process rebuild claim; return the locked handle or ``None``."""
    path = _lock_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        import fcntl

        handle = path.open("a+", encoding="utf-8")
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            handle.close()
            return None
        handle.seek(0)
        handle.truncate()
        handle.write(str(os.getpid()))
        handle.flush()
        _write_state({
            "schema_version": 1,
            **_target(scan, long_term),
            "status": "RUNNING",
            "pid": os.getpid(),
            "started_at": time.time(),
            "finished_at": None,
            "error": "",
        })
        return handle
    except Exception:
        return None


def _release_claim(handle) -> None:
    if handle is None:
        return
    try:
        import fcntl

        fcntl.flock(handle, fcntl.LOCK_UN)
    except Exception:
        pass
    try:
        handle.close()
    except Exception:
        pass
    # Deliberately keep the lock file. Unlinking a flock file can create two inodes and
    # allow two future processes to believe they own the same logical lock.


def _empty_workspace(
    scan_at: str = "",
    long_at: str = "",
    *,
    status: str = "REFRESHING",
    error: str = "",
) -> dict[str, Any]:
    try:
        from product.recommendations_workspace import CATEGORIES
        categories = [
            {
                **dict(meta),
                "count": 0,
                "cards": [],
                "empty_detail": "Recommendations are being rebuilt from the latest completed scan.",
            }
            for meta in CATEGORIES
        ]
    except Exception:
        categories = []
    failed = status == "FAILED"
    return {
        "schema_version": 4,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "scan_scanned_at": scan_at,
        "long_term_scanned_at": long_at,
        "records_status": status,
        "same_ist_day": False,
        "cmp_note": (
            "Latest recommendation projection failed to rebuild; no recommendation claim is being made."
            if failed
            else "Latest scan is available; recommendation projection is rebuilding in the background."
        ),
        "methods_note": "No recommendation claim is made until the persisted projection is ready.",
        "ensemble": {
            "high_conviction_count": 0,
            "good_setup_count": 0,
            "watch_count": 0,
            "avoid_count": 0,
            "empty_high_conviction": True,
            "empty_line": "Recommendation projection is unavailable." if failed else "Recommendation projection is rebuilding.",
        },
        "categories": categories,
        "lifecycle": {"active": [], "closed": [], "active_count": 0, "closed_count": 0},
        "disclaimer": "Persisted evidence only; no recommendation was fabricated while rebuilding.",
        "error": error,
        "from_saved_market_scan": True,
        "rebuilding": not failed,
    }


def _matches(saved: Mapping[str, Any] | None, scan: Mapping[str, Any], long_term: Mapping[str, Any]) -> bool:
    if not saved:
        return False
    try:
        from product.recommendations_store import reco_matches_scan

        return reco_matches_scan(
            saved,
            scan_scanned_at=str(scan.get("scanned_at") or ""),
            long_term_scanned_at=str(long_term.get("scanned_at") or ""),
        )
    except Exception:
        return False


def _current_generation_matches(scan: Mapping[str, Any], long_term: Mapping[str, Any]) -> bool:
    """Refuse last-writer-wins corruption if a newer scan arrives during a rebuild."""
    try:
        from product.long_term_store import load_long_term_scan
        from product.scan_store import load_scan

        current_scan = load_scan() or {}
        current_long = load_long_term_scan() or {}
        return _target(current_scan, current_long) == _target(scan, long_term)
    except Exception:
        return False


def _build_and_persist(scan: dict[str, Any], long_term: dict[str, Any], claim) -> None:
    global _BUILD_THREAD
    target = _target(scan, long_term)
    started = time.time()
    try:
        from product.recommendations_store import save_recommendations
        from product.recommendations_workspace import build_recommendations_workspace

        payload = build_recommendations_workspace(
            scan_payload=scan,
            long_term_payload=long_term,
            refresh_technicals=False,
            settle_cases=False,
            deep_confirm=False,
            persist_ledger=False,
        )
        if not _current_generation_matches(scan, long_term):
            _write_state({
                "schema_version": 1,
                **target,
                "status": "SUPERSEDED",
                "pid": os.getpid(),
                "started_at": started,
                "finished_at": time.time(),
                "error": "A newer scan/funds generation arrived before this rebuild could commit.",
            })
            return
        save_recommendations(payload)
        _write_state({
            "schema_version": 1,
            **target,
            "status": "SUCCEEDED",
            "pid": os.getpid(),
            "started_at": started,
            "finished_at": time.time(),
            "error": "",
        })
    except Exception as exc:
        message = f"{type(exc).__name__}: {exc}"[:400]
        _write_state({
            "schema_version": 1,
            **target,
            "status": "FAILED",
            "pid": os.getpid(),
            "started_at": started,
            "finished_at": time.time(),
            "error": message,
        })
        print(f"[API] recommendation background rebuild failed: {message}", flush=True)
    finally:
        _release_claim(claim)
        with _BUILD_LOCK:
            _BUILD_THREAD = None


def _recent_failed_target(scan: Mapping[str, Any], long_term: Mapping[str, Any]) -> bool:
    state = _read_state()
    if not _same_target(state, scan, long_term) or str(state.get("status") or "") != "FAILED":
        return False
    try:
        finished = float(state.get("finished_at") or 0.0)
    except (TypeError, ValueError):
        return False
    return finished > 0 and (time.time() - finished) < _RETRY_AFTER_FAILURE_S


def _ensure_rebuild(scan: dict[str, Any], long_term: dict[str, Any]) -> bool:
    global _BUILD_THREAD
    with _BUILD_LOCK:
        if _BUILD_THREAD is not None and _BUILD_THREAD.is_alive():
            return False
        if _recent_failed_target(scan, long_term):
            return False
        claim = _claim_rebuild(scan, long_term)
        if claim is None:
            return False
        thread = threading.Thread(
            target=_build_and_persist,
            args=(dict(scan), dict(long_term), claim),
            name="quantterm-recommendations-rebuild",
            daemon=True,
        )
        _BUILD_THREAD = thread
        thread.start()
        return True


def build_fast_response(core, attach_authority=None) -> dict[str, Any]:
    """Return immediately from persisted state; schedule expensive projection if needed."""
    from product.recommendations_store import load_recommendations
    from product.recommendations_workspace import slim_workspace_for_desk

    scan = dict(core._scan_payload() or {})
    long_term = dict(core._long_term_payload() or {})
    saved = load_recommendations()
    current = _matches(saved, scan, long_term)

    if not current:
        _ensure_rebuild(scan, long_term)

    state = _read_state()
    target_state = state if _same_target(state, scan, long_term) else {}
    state_status = str(target_state.get("status") or "")
    state_error = str(target_state.get("error") or "")[:300]

    if saved:
        payload = dict(saved)
        if not current:
            failed = state_status == "FAILED"
            payload["records_status"] = "FAILED" if failed else "REFRESHING"
            payload["rebuilding"] = not failed
            payload["rebuild_state"] = target_state
            payload["rebuild_error"] = state_error if failed else ""
            payload["cmp_note"] = (
                (
                    "A newer scan/funds artifact exists, but its recommendation rebuild failed. "
                    "Cards below are the last persisted projection and are NOT current. "
                )
                if failed
                else (
                    "A newer scan/funds artifact exists. QuantTerm is rebuilding the recommendation "
                    "projection in the background; cards below are the last persisted projection. "
                )
            ) + str(payload.get("cmp_note") or "")
        if callable(attach_authority):
            try:
                payload = attach_authority(payload)
            except Exception as exc:
                payload["authority_warning"] = str(exc)[:200]
        return slim_workspace_for_desk(payload)

    status = "FAILED" if state_status == "FAILED" else "REFRESHING"
    payload = _empty_workspace(
        str(scan.get("scanned_at") or ""),
        str(long_term.get("scanned_at") or ""),
        status=status,
        error=state_error if status == "FAILED" else "",
    )
    payload["rebuild_state"] = target_state
    if callable(attach_authority):
        try:
            payload = attach_authority(payload)
        except Exception as exc:
            payload["authority_warning"] = str(exc)[:200]
    return payload


def _replace_route() -> None:
    core = sys.modules.get("terminal_api")
    parallel = sys.modules.get("terminal_product_api_parallel")
    if core is None or parallel is None:
        return

    app = core.app
    path = "/api/recommendations-workspace"
    kept = []
    for route in list(app.router.routes):
        methods = set(getattr(route, "methods", set()) or set())
        if getattr(route, "path", None) == path and "GET" in methods:
            continue
        kept.append(route)
    app.router.routes[:] = kept

    def _fast_recommendations_workspace() -> dict[str, Any]:
        return build_fast_response(core, getattr(parallel, "_attach_authority", None))

    app.add_api_route(
        path,
        _fast_recommendations_workspace,
        methods=["GET"],
        name="recommendations_workspace_persisted_first",
    )


def register_terminal_recommendations_liveness() -> None:
    """Register only when terminal_api is already being assembled; never import it as a side effect."""
    global _INSTALLED
    if _INSTALLED:
        return
    core = sys.modules.get("terminal_api")
    if core is None or not hasattr(core, "app"):
        return

    @core.app.on_event("startup")
    def _install_fast_recommendations_route() -> None:
        _replace_route()

    _INSTALLED = True
