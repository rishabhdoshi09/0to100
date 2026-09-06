"""Persisted-first Recommendations API liveness.

The whole-market scan can contain thousands of rows. Rebuilding every expert/method layer
inside a GET request makes the browser wait behind CPU-heavy work and can trip the desk's
request timeout while the actual scan worker is healthy.

This module installs a single replacement GET route at FastAPI startup:
- serve the recommendation artifact matching the current saved scan immediately;
- when that artifact is missing/stale, rebuild it once in a daemon thread;
- serve the last honest artifact (explicitly marked REFRESHING) while rebuilding;
- never start a scan, scrape, broker call or order from the request path.
"""
from __future__ import annotations

import threading
from datetime import datetime, timezone
from typing import Any, Mapping

_INSTALLED = False
_BUILD_LOCK = threading.Lock()
_BUILD_THREAD: threading.Thread | None = None


def _empty_workspace(scan_at: str = "", long_at: str = "") -> dict[str, Any]:
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
    return {
        "schema_version": 4,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "scan_scanned_at": scan_at,
        "long_term_scanned_at": long_at,
        "records_status": "REFRESHING",
        "same_ist_day": False,
        "cmp_note": "Latest scan is available; recommendation projection is rebuilding in the background.",
        "methods_note": "No recommendation claim is made until the persisted projection is ready.",
        "ensemble": {
            "high_conviction_count": 0,
            "good_setup_count": 0,
            "watch_count": 0,
            "avoid_count": 0,
            "empty_high_conviction": True,
            "empty_line": "Recommendation projection is rebuilding.",
        },
        "categories": categories,
        "lifecycle": {"active": [], "closed": [], "active_count": 0, "closed_count": 0},
        "disclaimer": "Persisted evidence only; no recommendation was fabricated while rebuilding.",
        "error": "",
        "from_saved_market_scan": True,
        "rebuilding": True,
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


def _build_and_persist(scan: dict[str, Any], long_term: dict[str, Any]) -> None:
    global _BUILD_THREAD
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
        save_recommendations(payload)
    except Exception as exc:
        # Request path remains healthy even when the background projection fails.
        print(f"[API] recommendation background rebuild failed: {type(exc).__name__}: {exc}", flush=True)
    finally:
        with _BUILD_LOCK:
            _BUILD_THREAD = None


def _ensure_rebuild(scan: dict[str, Any], long_term: dict[str, Any]) -> bool:
    global _BUILD_THREAD
    with _BUILD_LOCK:
        if _BUILD_THREAD is not None and _BUILD_THREAD.is_alive():
            return False
        thread = threading.Thread(
            target=_build_and_persist,
            args=(dict(scan), dict(long_term)),
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

    if saved:
        payload = dict(saved)
        if not current:
            payload["records_status"] = "REFRESHING"
            payload["rebuilding"] = True
            payload["cmp_note"] = (
                "A newer scan/funds artifact exists. QuantTerm is rebuilding the recommendation "
                "projection in the background; cards below are the last persisted projection. "
                + str(payload.get("cmp_note") or "")
            ).strip()
        if callable(attach_authority):
            try:
                payload = attach_authority(payload)
            except Exception as exc:
                payload["authority_warning"] = str(exc)[:200]
        return slim_workspace_for_desk(payload)

    payload = _empty_workspace(
        str(scan.get("scanned_at") or ""),
        str(long_term.get("scanned_at") or ""),
    )
    if callable(attach_authority):
        try:
            payload = attach_authority(payload)
        except Exception as exc:
            payload["authority_warning"] = str(exc)[:200]
    return payload


def _replace_route() -> None:
    import terminal_api as core
    import terminal_product_api_parallel as parallel

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
    """Register once; actual route replacement runs after all API modules are imported."""
    global _INSTALLED
    if _INSTALLED:
        return
    try:
        import terminal_api as core
    except Exception:
        return

    @core.app.on_event("startup")
    def _install_fast_recommendations_route() -> None:
        _replace_route()

    _INSTALLED = True
