#!/usr/bin/env python3
"""Strict façade for QuantTerm product acceptance.

The established operation-grading implementation lives in
``_product_acceptance_core``. This façade keeps the acceptance boundary small
and explicit: broker safety must be verified, locked, and unauthorized. Paper
acceptance exercises the real Paper Autopilot route, Forward Soak retains its
POST verify-now probe, and the FINAL product contract must prove the operator
Control Center and Forward Evidence console remain genuinely wired.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Any, Mapping

_CORE_PATH = Path(__file__).with_name("_product_acceptance_core.py")
_SPEC = importlib.util.spec_from_file_location("_quantterm_product_acceptance_core", _CORE_PATH)
if _SPEC is None or _SPEC.loader is None:  # pragma: no cover - import failure is fatal
    raise RuntimeError(f"Unable to load product-acceptance core from {_CORE_PATH}")
_core = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_core)

# Re-export established acceptance helpers; strict helpers below intentionally
# override the corresponding names.
for _name in dir(_core):
    if not _name.startswith("__"):
        globals()[_name] = getattr(_core, _name)


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
    """Grade persisted Paper state and independently require canonical safety truth."""
    data = _core._as_dict(payload)
    if not data:
        return {"status": "FAIL", "blocker_reason": "empty Paper Autopilot projection", "count": 0}
    nested = _core._as_dict(data.get("paper"))
    positions = nested.get("open_positions") if "open_positions" in nested else data.get("open_positions")
    has_cycle = "last_cycle" in data or "latest" in data or "why_no_trade" in data
    if positions is None or not has_cycle:
        return {"status": "FAIL", "blocker_reason": "missing paper positions/cycle contract", "count": 0}
    if live_lock_verified is not True or data.get("live_lock_verified") is not True:
        return {"status": "FAIL", "blocker_reason": "paper execution live lock is unverified", "count": 0}
    if live_locked is not True or data.get("live_locked") is not True:
        return {"status": "FAIL", "blocker_reason": "paper execution did not prove live money locked", "count": 0}
    authorized = live_execution_authorized
    if authorized is None:
        authorized = data.get("live_execution_authorized")
    if authorized is not False or data.get("live_execution_authorized") is not False:
        return {"status": "FAIL", "blocker_reason": "paper execution did not prove live execution unauthorized", "count": 0}
    return {"status": "PASS", "blocker_reason": "", "count": len(_core._as_list(positions))}


def grade_final_product_contract(payload: Mapping[str, Any] | None) -> dict[str, str]:
    data = _core._as_dict(payload)
    if data.get("wired") is not True:
        return {"status": "FAIL", "blocker_reason": "product-contract wired is not true"}
    checks = _core._as_dict(data.get("checks"))
    operator = _core._as_dict(checks.get("operator_control_center"))
    evidence = _core._as_dict(checks.get("forward_evidence_console"))
    if operator.get("route_registered") is not True:
        return {"status": "FAIL", "blocker_reason": "Control Center control route is not registered"}
    if operator.get("all_required_controls_available") is not True:
        return {"status": "FAIL", "blocker_reason": "Control Center is missing required safe controls"}
    if _core._as_list(operator.get("live_money_controls_exposed")):
        return {"status": "FAIL", "blocker_reason": "Control Center exposes live-money mutation controls"}
    for key in (
        "forward_evidence_route_registered",
        "forward_soak_route_registered",
        "decision_simulator_route_registered",
    ):
        if evidence.get(key) is not True:
            return {"status": "FAIL", "blocker_reason": f"Forward Evidence contract missing {key}"}
    if evidence.get("simulation_evidence_class") != "HISTORICAL_REPLAY":
        return {"status": "FAIL", "blocker_reason": "simulator evidence is not isolated as HISTORICAL_REPLAY"}
    if evidence.get("forward_evidence_class") != "PAPER_FORWARD":
        return {"status": "FAIL", "blocker_reason": "forward evidence is not isolated as PAPER_FORWARD"}
    return {"status": "PASS", "blocker_reason": ""}


def _append_final_contract_evidence(args, core_exit: int) -> int:
    """Make the final operator contract part of the persisted acceptance verdict."""
    try:
        contract = _core._request_json(
            _core._url(args.api, "/api/product-contract"),
            timeout=args.request_timeout,
        )
        grade = grade_final_product_contract(contract)
    except Exception as exc:
        contract = {}
        grade = {"status": "FAIL", "blocker_reason": str(exc)[:300]}

    path = Path(args.output)
    if not path.exists():
        return 1
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = [row for row in payload.get("features", []) if isinstance(row, dict)]
    rows.append(_core._row(
        feature="Final product contract",
        trigger_tested="GET /api/product-contract",
        backend_path="terminal_product_api_parallel.product_contract",
        durable_artifact="registered API routes + canonical control allow-list",
        result_count=len(_core._as_dict(contract.get("checks"))),
        status=grade["status"],
        blocker_reason=grade["blocker_reason"],
        start_timestamp=_core._now(),
        finish_timestamp=_core._now(),
        code_sha=_core._sha(),
    ))
    payload["features"] = rows
    if grade["status"] != "PASS":
        payload["verdict"] = "PRODUCT ACCEPTANCE HOLD"
        payload["verdict_reason"] = "final product contract failed: " + grade["blocker_reason"]
        core_exit = 1
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return int(core_exit)


def run(args) -> int:
    # The core's run function resolves helpers in the core module namespace.
    # Install strict boundary functions before executing it. Its original
    # /api/forward-soak POST is intentionally preserved to force fresh proof.
    _core.grade_learning_dashboard = grade_learning_dashboard
    _core.grade_forward_soak = grade_forward_soak
    _core.grade_paper_status = grade_paper_status
    core_exit = _core.run(args)
    return _append_final_contract_evidence(args, core_exit)


if __name__ == "__main__":
    try:
        raise SystemExit(run(build_parser().parse_args()))
    except KeyboardInterrupt:
        raise SystemExit(130)
    except Exception as exc:
        print(f"NOT WORKING: {exc}", file=sys.stderr)
        raise SystemExit(1)
