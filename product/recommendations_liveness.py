"""Persisted-first Recommendations API liveness.

Opening Recommendations must remain a read path.  Expensive projection work is rebuilt in the
background from already-persisted scan/funds artifacts and never from broker/network calls.

``api.app`` registers the one canonical GET route for this module at import time; no
startup hook or launch-module detection decides which handler serves the desk.

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
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping
from core.runtime_paths import logs_dir

_BUILD_LOCK = threading.Lock()
_BUILD_THREAD: threading.Thread | None = None
_RETRY_AFTER_FAILURE_S = 30.0
_MAX_RETRY_AFTER_FAILURE_S = 900.0
# A pipeline publication holds the lock while it builds and saves one
# generation; the rebuild waits that long at most, then reports FAILED.
_PUBLISH_LOCK_TIMEOUT_S = 300.0
_ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_LOCK_PATH = logs_dir() / "product" / "recommendations_rebuild.lock"
_DEFAULT_STATE_PATH = logs_dir() / "product" / "recommendations_rebuild_state.json"


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
        previous = _read_state()
        carried = 0
        if _same_target(previous, scan, long_term):
            try:
                carried = int(previous.get("failed_attempts") or 0)
            except (TypeError, ValueError):
                carried = 0
        _write_state({
            "schema_version": 1,
            **_target(scan, long_term),
            "status": "RUNNING",
            "pid": os.getpid(),
            "started_at": time.time(),
            "finished_at": None,
            "error": "",
            # Retry backoff survives the claim; a new generation starts at 0.
            "failed_attempts": carried,
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
    failed = status == "FAILED"
    withheld = status == "STALE"
    detail = (
        "No current verified NSE scan for this session; cards are withheld until one is published."
        if withheld
        else "Recommendations are being rebuilt from the latest completed scan."
    )
    try:
        from product.recommendations_workspace import CATEGORIES
        categories = [
            {
                **dict(meta),
                "count": 0,
                "cards": [],
                "empty_detail": detail,
            }
            for meta in CATEGORIES
        ]
    except Exception:
        categories = []
    if failed:
        cmp_note = "Latest recommendation projection failed to rebuild; no recommendation claim is being made."
        empty_line = "Recommendation projection is unavailable."
    elif withheld:
        cmp_note = (
            "The saved scan is not verified as current for today's NSE session; "
            "recommendation cards are withheld and nothing is rebuilt from it."
        )
        empty_line = "No current verified NSE scan."
    else:
        cmp_note = "Latest scan is available; recommendation projection is rebuilding in the background."
        empty_line = "Recommendation projection is rebuilding."
    return {
        "schema_version": 4,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "scan_scanned_at": scan_at,
        "long_term_scanned_at": long_at,
        "records_status": status,
        "same_ist_day": False,
        "cmp_note": cmp_note,
        "methods_note": "No recommendation claim is made until the persisted projection is ready.",
        "ensemble": {
            "high_conviction_count": 0,
            "good_setup_count": 0,
            "watch_count": 0,
            "avoid_count": 0,
            "empty_high_conviction": True,
            "empty_line": empty_line,
        },
        "categories": categories,
        "lifecycle": {"active": [], "closed": [], "active_count": 0, "closed_count": 0},
        "disclaimer": "Persisted evidence only; no recommendation was fabricated while rebuilding.",
        "error": error,
        "from_saved_market_scan": True,
        "rebuilding": not (failed or withheld),
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
        from product.recommendations_store import publication_lock

        # The generation check and the save are one critical section with the
        # scan pipeline's publication: a newer generation published while this
        # rebuild ran is never overwritten by this older projection.
        with publication_lock(timeout_s=_PUBLISH_LOCK_TIMEOUT_S):
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
        previous = _read_state()
        attempts = 1
        if _same_target(previous, scan, long_term) and str(previous.get("status") or "") in {"FAILED", "RUNNING"}:
            try:
                attempts = int(previous.get("failed_attempts") or 0) + 1
            except (TypeError, ValueError):
                attempts = 1
        _write_state({
            "schema_version": 1,
            **target,
            "status": "FAILED",
            "pid": os.getpid(),
            "started_at": started,
            "finished_at": time.time(),
            "error": message,
            "failed_attempts": attempts,
        })
        print(f"[API] recommendation background rebuild failed: {message}", flush=True)
    finally:
        _release_claim(claim)
        with _BUILD_LOCK:
            _BUILD_THREAD = None


def _retry_after_s(failed_attempts: int) -> float:
    """30 s, 60 s, 120 s ... capped: a deterministic failure must not rebuild
    the whole projection every 30 s for as long as the page is polled."""
    n = max(1, int(failed_attempts or 1))
    return min(_RETRY_AFTER_FAILURE_S * (2 ** (n - 1)), _MAX_RETRY_AFTER_FAILURE_S)


def _recent_failed_target(scan: Mapping[str, Any], long_term: Mapping[str, Any]) -> bool:
    state = _read_state()
    if not _same_target(state, scan, long_term) or str(state.get("status") or "") != "FAILED":
        return False
    try:
        finished = float(state.get("finished_at") or 0.0)
        attempts = int(state.get("failed_attempts") or 1)
    except (TypeError, ValueError):
        return False
    return finished > 0 and (time.time() - finished) < _retry_after_s(attempts)


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


def display_authority(
    payload: Mapping[str, Any],
    scan: Mapping[str, Any],
    *,
    projection: str,
    scan_current: bool | None = None,
) -> dict[str, Any]:
    """Whether the desk may present this workspace as today's recommendations.

    Computed here, at read time, from the canonical scan identity and the
    canonical scan-freshness authority (``product.desk_pipeline.scan_is_fresh``),
    so the browser never infers it from a separate, short-lived dashboard cache.
    ``CURRENT`` requires the projection to be the one published for the current
    scan generation AND that scan to be fresh. Anything else is shown as its own
    explicit state, never as current recommendations. ``scan_current`` is the
    freshness verdict this read already obtained; ``None`` asks the authority.
    """
    scan_at = str(scan.get("scanned_at") or "")
    if scan_current is None:
        try:
            from product.desk_pipeline import scan_is_fresh
            scan_current = bool(scan_is_fresh())
        except Exception:
            scan_current = False
    scan_current = bool(scan_at) and bool(scan_current)
    if projection == "CURRENT" and not scan_current:
        status = "SCAN_NOT_CURRENT"
    else:
        status = projection
    overlays = dict(scan.get("desk_overlays") or {})
    publication = {
        "recommendations": str(overlays.get("recommendations") or "UNKNOWN"),
        "recommendation_cards": overlays.get("recommendation_cards"),
        "error": overlays.get("recommendations_error"),
        "scan_scanned_at": str(overlays.get("scan_scanned_at") or ""),
    }
    provenance = scan.get("provenance") if isinstance(scan.get("provenance"), Mapping) else {}
    return {
        "status": status,
        "projection": projection,
        "publication": publication,
        "scan_current": scan_current,
        "scan_scanned_at": scan_at,
        # Which artifact and which market session this verdict is about.
        "scan_id": str(provenance.get("scan_id") or ""),
        "scan_market_session": str(
            provenance.get("market_session_date") or provenance.get("price_data_as_of") or ""
        )[:10],
        "projection_scan_scanned_at": str(payload.get("scan_scanned_at") or ""),
        "evaluated_at": datetime.now(timezone.utc).isoformat(),
        "source": "product.recommendations_liveness",
    }


def _withheld(scan: Mapping[str, Any], *, error: str, scan_current: bool | None) -> dict[str, Any]:
    """Empty workspace for a scan that is not current (or absent): no cards, no rebuild."""
    payload = _empty_workspace(status="STALE", error=error)
    projection = "SCAN_UNAVAILABLE" if scan.get("available") is False else "SCAN_NOT_CURRENT"
    payload["display_authority"] = display_authority(
        payload, scan, projection=projection, scan_current=scan_current,
    )
    return payload


def build_fast_response(
    core,
    attach_authority=None,
    *,
    scan_current: bool | None = None,
    scan: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Return immediately from persisted state; schedule expensive projection if needed.

    ``attach_authority(payload, scan)`` receives the scan payload this read
    already loaded, so the scan artifact is parsed once per request.
    ``scan_current`` is the freshness verdict a caller already obtained for
    ``scan`` -- the same parsed artifact, never a re-read of the file.
    """
    from product.recommendations_store import load_recommendations
    from product.recommendations_workspace import slim_workspace_for_desk

    scan = dict(scan if scan is not None else (core._scan_payload() or {}))
    long_term = dict(core._long_term_payload() or {})
    saved = load_recommendations()
    # A persisted recommendation is not actionable when its source scan has
    # disappeared from the validated read path. Do not serve stale BUY cards
    # while the corrected Kite snapshot is being scanned.
    if scan.get("available") is False:
        payload = _empty_workspace(
            status="FAILED",
            error="Verified market scan is missing or stale; recommendations withheld.",
        )
        payload["display_authority"] = display_authority(
            payload, scan, projection="SCAN_UNAVAILABLE", scan_current=scan_current,
        )
        return payload
    # The same saved scan identity is not enough when the scan's own data
    # provenance has been invalidated: withhold, and do not rebuild from it.
    provenance = scan.get("provenance") or {}
    if isinstance(provenance, Mapping) and provenance.get("data_current") is False:
        return _withheld(
            scan,
            error="Current scan provenance is stale; recommendation cards withheld.",
            scan_current=False,
        )
    current = _matches(saved, scan, long_term)

    if not current:
        _ensure_rebuild(scan, long_term)

    state = _read_state()
    target_state = state if _same_target(state, scan, long_term) else {}
    state_status = str(target_state.get("status") or "")
    state_error = str(target_state.get("error") or "")[:300]

    if saved and current:
        payload = dict(saved)
        try:
            from product.win_probability import overlay_cards

            payload = overlay_cards(payload)
        except Exception:
            pass
        if callable(attach_authority):
            try:
                payload = attach_authority(payload, scan)
            except Exception as exc:
                payload["authority_warning"] = str(exc)[:200]
        payload = slim_workspace_for_desk(payload)
        payload["display_authority"] = display_authority(
            payload, scan, projection="CURRENT", scan_current=scan_current,
        )
        return payload

    # Last-generation BUY cards must not remain on an actionable recommendation
    # surface after a scan or long-term artifact changes. Even a clear "stale"
    # label is insufficient: consumers might still rank/display those cards.
    # Retain refresh/error status, never the previous generation's picks.
    status = "FAILED" if state_status == "FAILED" else "REFRESHING"
    payload = _empty_workspace(
        str(scan.get("scanned_at") or ""),
        str(long_term.get("scanned_at") or ""),
        status=status,
        error=state_error if status == "FAILED" else "",
    )
    if saved:
        payload["cmp_note"] = (
            "Saved recommendations are NOT current: they belong to an older scan/funds generation and "
            "have been withheld. "
            + str(payload.get("cmp_note") or "")
        )
    payload["rebuild_state"] = target_state
    payload["rebuild_error"] = state_error if status == "FAILED" else ""
    if callable(attach_authority):
        try:
            payload = attach_authority(payload, scan)
        except Exception as exc:
            payload["authority_warning"] = str(exc)[:200]
    payload = slim_workspace_for_desk(payload)
    payload["display_authority"] = display_authority(
        payload, scan, projection=status, scan_current=scan_current,
    )
    return payload


def build_current_scan_response(core, attach_authority=None) -> dict[str, Any]:
    """The one Opportunities read: fail closed unless today's NSE scan is verified.

    A persisted ``data_current=True`` flag describes the original scan, not
    today's market. The canonical scan-freshness authority independently
    verifies the active Kite snapshot identity and exchange-calendar date (or
    the current official EOD archive) on every read. An unverified scan is
    neither served nor rebuilt from: its saved BUY cards are withheld.

    The scan is parsed ONCE: the freshness verdict and the served payload are
    the same artifact, so a scan replaced mid-read cannot inherit a verdict.
    """
    from product.desk_pipeline import scan_payload_is_fresh
    from product.scan_store import load_scan, resolved_scan_path

    try:
        raw = load_scan(resolved_scan_path())
    except Exception:
        raw = None
    try:
        fresh = bool(scan_payload_is_fresh(raw))
    except Exception:
        fresh = False
    view = getattr(core, "_scan_payload_from", None)
    try:
        scan = dict(view(raw) if callable(view) else (core._scan_payload() or {}))
    except Exception:
        scan = {}
    if not fresh:
        return _withheld(
            scan,
            error="No current verified NSE scan; recommendation cards withheld.",
            scan_current=False,
        )
    return build_fast_response(core, attach_authority, scan_current=True, scan=scan)
