#!/usr/bin/env python3
"""Strict façade for QuantTerm product acceptance.

The large operation-grading implementation lives in ``_product_acceptance_core``.
This façade keeps the acceptance boundary small and explicit: broker safety must
be verified, locked, and unauthorized, and Paper acceptance reads the real
``/api/dashboard`` Paper projection plus canonical ``/api/health`` safety truth.
"""
from __future__ import annotations

import importlib.util
import sys
import urllib.parse
from pathlib import Path
from typing import Any, Mapping

_CORE_PATH = Path(__file__).with_name("_product_acceptance_core.py")
_SPEC = importlib.util.spec_from_file_location("_quantterm_product_acceptance_core", _CORE_PATH)
if _SPEC is None or _SPEC.loader is None:  # pragma: no cover - import failure is fatal
    raise RuntimeError(f"Unable to load product-acceptance core from {_CORE_PATH}")
_core = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_core)

# Re-export the established acceptance helpers; strict helpers below intentionally
# override the corresponding names.
for _name in dir(_core):
    if not _name.startswith("__"):
        globals()[_name] = getattr(_core, _name)

_original_request_json = _core._request_json
_original_row = _core._row


def grade_learning_dashboard(payload: Mapping[str, Any] | None) -> dict[str, Any]:
    data = _core._as_dict(payload)
    if not data:
        return {"status": "FAIL", "blocker_reason": "empty learning dashboard", "count": 0}
    if data.get("schema_version") in {None, ""}:
        return {"status": "FAIL", "blocker_reason": "missing schema_version", "count": 0}
    if data.get("live_lock_verified") is not True:
        return {"status": "FAIL", "blocker_reason": "learning dashboard live lock is unverified", "count": 0}
    if data.get("live_locked") is not True:
        return {"status": "FAIL", "blocker_reason": "learning dashboard did not prove live money locked", "count": 0}
    if data.get("live_execution_authorized") is not False:
        return {"status": "FAIL", "blocker_reason": "learning dashboard did not prove live execution unauthorized", "count": 0}
    if "policies" not in data or "counterfactuals" not in data:
        return {"status": "FAIL", "blocker_reason": "missing policies/counterfactuals contract", "count": 0}
    return {"status": "PASS", "blocker_reason": "", "count": len(_core._as_list(data.get("policies")))}


def grade_forward_soak(payload: Mapping[str, Any] | None) -> dict[str, str]:
    data = _core._as_dict(payload)
    verification = data.get("verification")
    if not isinstance(verification, Mapping):
        return {"status": "FAIL", "blocker_reason": "missing verification contract"}
    lanes = verification.get("lanes") if isinstance(verification.get("lanes"), Mapping) else data.get("lanes")
    if not isinstance(lanes, Mapping):
        return {"status": "FAIL", "blocker_reason": "verification missing lanes"}
    verified = verification.get("live_lock_verified")
    if verified is None:
        verified = data.get("live_lock_verified")
    if verified is not True:
        return {"status": "FAIL", "blocker_reason": "forward soak live lock is unverified"}
    locked = verification.get("live_locked")
    if locked is None:
        locked = data.get("live_locked")
    if locked is not True:
        return {"status": "FAIL", "blocker_reason": "forward soak did not prove live_locked"}
    authorized = verification.get("live_execution_authorized")
    if authorized is None:
        authorized = data.get("live_execution_authorized")
    if authorized is not False:
        return {"status": "FAIL", "blocker_reason": "forward soak did not prove live execution unauthorized"}
    return {"status": "PASS", "blocker_reason": ""}


def grade_paper_status(
    payload: Mapping[str, Any] | None,
    *,
    live_locked: bool,
    live_lock_verified: bool,
    live_execution_authorized: bool | None = None,
) -> dict[str, Any]:
    """Grade persisted Paper state while keeping broker safety a separate authority."""
    data = _core._as_dict(payload)
    if not data:
        return {"status": "FAIL", "blocker_reason": "empty Paper projection", "count": 0}
    nested = _core._as_dict(data.get("paper"))
    positions = nested.get("open_positions") if "open_positions" in nested else data.get("open_positions")
    has_cycle = "last_cycle" in data or "latest" in data or "why_no_trade" in data
    if positions is None or not has_cycle:
        return {"status": "FAIL", "blocker_reason": "missing paper positions/cycle contract", "count": 0}
    if live_lock_verified is not True:
        return {"status": "FAIL", "blocker_reason": "canonical paper-execution live lock is unverified", "count": 0}
    if live_locked is not True:
        return {"status": "FAIL", "blocker_reason": "canonical paper-execution boundary is not locked", "count": 0}
    authorized = live_execution_authorized
    if authorized is None:
        authorized = data.get("live_execution_authorized")
    if authorized is not False:
        return {"status": "FAIL", "blocker_reason": "canonical live execution was not proven unauthorized", "count": 0}
    return {"status": "PASS", "blocker_reason": "", "count": len(_core._as_list(positions))}


def _request_json(
    url: str,
    *,
    method: str = "GET",
    timeout: float = 12.0,
    body: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Route the obsolete Paper-only probe to the real dashboard projection.

    No safety value is invented: canonical safety fields are copied from the
    same terminal's /api/health response solely for acceptance grading.
    """
    parsed = urllib.parse.urlsplit(url)
    if parsed.path.rstrip("/") == "/api/paper-autopilot":
        base = urllib.parse.urlunsplit((parsed.scheme, parsed.netloc, "", "", "")).rstrip("/")
        dashboard = _original_request_json(f"{base}/api/dashboard", timeout=timeout)
        paper = _core._as_dict(dashboard.get("paper"))
        health = _original_request_json(f"{base}/api/health", timeout=timeout)
        for key in ("live_locked", "live_lock_verified", "live_execution_authorized"):
            if key in health:
                paper[key] = health[key]
        return paper
    return _original_request_json(url, method=method, timeout=timeout, body=body)


def _row(**kwargs: Any) -> dict[str, Any]:
    if kwargs.get("feature") == "Paper execution status":
        kwargs["trigger_tested"] = "GET /api/dashboard[paper] + GET /api/health"
        kwargs["backend_path"] = "terminal_api._paper_payload + product.runtime_lifecycle"
    return _original_row(**kwargs)


def run(args) -> int:
    # The core's run function resolves helpers in the core module namespace.
    # Install the strict boundary functions before executing it.
    _core.grade_learning_dashboard = grade_learning_dashboard
    _core.grade_forward_soak = grade_forward_soak
    _core.grade_paper_status = grade_paper_status
    _core._request_json = _request_json
    _core._row = _row
    return _core.run(args)


if __name__ == "__main__":
    try:
        raise SystemExit(run(build_parser().parse_args()))
    except KeyboardInterrupt:
        raise SystemExit(130)
    except Exception as exc:
        print(f"NOT WORKING: {exc}", file=sys.stderr)
        raise SystemExit(1)
