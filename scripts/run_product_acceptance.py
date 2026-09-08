#!/usr/bin/env python3
"""Live product-acceptance matrix against a running canonical QuantTerm stack.

Writes logs/product/product_acceptance.json. Never unlocks live money.

Classification inspects the durable operation *result*, not only operation.status.
A non-empty JSON object is not proof that a feature works.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
TERMINAL = {"SUCCEEDED", "FAILED", "BLOCKED", "CANCELLED"}
PAPER_CYCLE_DONE = {"TRADED", "NO_ELIGIBLE_TRADE", "BLOCKED_SAFETY", "NO_DATA", "BLOCKED_BROKER"}
EXPECTED_EXTERNAL_TOKENS = (
    "FNO_UNIVERSE_UNAVAILABLE",
    "LOGIN REQUIRED",
    "LOGIN_REQUIRED",
    "BROKER_LOGIN_REQUIRED",
)
EXPECTED_EXTERNAL_FEATURES = frozenset({"Data refresh", "F&O refresh"})
EXPLICIT_BLOCKER_CODES = frozenset({
    "FNO_UNIVERSE_UNAVAILABLE",
    "BROKER_LOGIN_REQUIRED",
    "LOGIN_REQUIRED",
    "LOGIN REQUIRED",
    "HISTORY_TOO_SHALLOW",
    "HISTORY_STALE",
    "HISTORY_NOT_READY",
    "SNAPSHOT_STALE",
    "OFFICIAL_HISTORY_UNAVAILABLE",
})


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception:
        return ""


def _as_dict(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _as_list(value: Any) -> list[Any]:
    return list(value) if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)) else []


def _join_reasons(*parts: Any) -> str:
    seen: list[str] = []
    for part in parts:
        text = str(part or "").strip()
        if text and text not in seen:
            seen.append(text)
    return "; ".join(seen)


def collect_nested_blockers(result: Mapping[str, Any] | None) -> list[str]:
    """Surface nested blocked lanes (e.g. DATA_PREPARE F&O) from a durable result."""
    reasons: list[str] = []
    payload = _as_dict(result)

    def walk(node: Any, path: str) -> None:
        if isinstance(node, Mapping):
            blocked = node.get("blocked") is True
            code = str(node.get("code") or node.get("error_code") or "").strip()
            error = str(node.get("error") or node.get("error_message") or node.get("message") or "").strip()
            if blocked:
                label = path or "lane"
                detail = _join_reasons(code, error) or "blocked"
                reasons.append(f"{label} blocked: {detail}")
            elif path and code and str(node.get("status") or "").upper() == "BLOCKED":
                reasons.append(f"{path} blocked: {code}")
            for key, child in node.items():
                if key in {"history", "payload", "artifact"}:
                    continue
                child_path = str(key)
                walk(child, child_path)
        elif isinstance(node, Sequence) and not isinstance(node, (str, bytes, bytearray)):
            for item in node:
                walk(item, path)

    walk(payload, "")
    missing = [str(item) for item in _as_list(payload.get("missing_lanes")) if str(item).strip()]
    for lane in missing:
        token = f"missing_lanes={lane}"
        if token not in reasons and not any(lane in item for item in reasons):
            reasons.append(token)
    # Deduplicate while preserving order.
    out: list[str] = []
    for item in reasons:
        if item not in out:
            out.append(item)
    return out


def _has_explicit_blocker(op: Mapping[str, Any], nested: Sequence[str]) -> bool:
    code = str(op.get("error_code") or "").strip()
    blocked_on = str(op.get("blocked_on") or "").strip()
    message = str(op.get("error_message") or op.get("message") or op.get("blocked_reason") or "")
    blob = " ".join([code, blocked_on, message, " ".join(nested)]).upper()
    if code and (code.upper() in {item.upper() for item in EXPLICIT_BLOCKER_CODES} or "UNAVAILABLE" in code.upper() or "LOGIN" in code.upper() or "BLOCK" in code.upper()):
        return True
    if blocked_on:
        return True
    if nested:
        return True
    for token in EXPLICIT_BLOCKER_CODES:
        if token.upper() in blob:
            return True
    return False


def classify_operation(op: Mapping[str, Any] | None) -> dict[str, str]:
    """Grade a durable market-ops (or similar) record from status *and* result.

    Contract:
      SUCCEEDED + clean result → PASS
      SUCCEEDED + result.degraded=true → DEGRADED
      nested blocked lanes appear in blocker_reason
      FAILED → FAIL
      CANCELLED → FAIL
      BLOCKED → BLOCKED only with an explicit blocker; otherwise FAIL
      Internal exceptions are never silently turned into DEGRADED/BLOCKED.
    """
    payload = _as_dict(op)
    status = str(payload.get("status") or "").upper()
    result = payload.get("result")
    if isinstance(result, str):
        try:
            parsed = json.loads(result)
            result = parsed if isinstance(parsed, Mapping) else {}
        except Exception:
            result = {}
    result_d = _as_dict(result)
    nested = collect_nested_blockers(result_d)
    degraded = result_d.get("degraded") is True
    error_code = str(payload.get("error_code") or "").strip()
    error_message = str(payload.get("error_message") or payload.get("message") or "").strip()
    nested_reason = _join_reasons(*nested)

    if status in {"FAILED", "CANCELLED"}:
        return {
            "status": "FAIL",
            "blocker_reason": _join_reasons(error_code, error_message, nested_reason) or status,
        }
    if status == "BLOCKED":
        reason = _join_reasons(error_code, error_message, nested_reason) or status
        if _has_explicit_blocker(payload, nested):
            return {"status": "BLOCKED", "blocker_reason": reason}
        return {
            "status": "FAIL",
            "blocker_reason": _join_reasons("BLOCKED without explicit blocker", reason),
        }
    if status == "SUCCEEDED":
        if degraded or nested:
            return {
                "status": "DEGRADED",
                "blocker_reason": _join_reasons(
                    "degraded=true" if degraded else "",
                    nested_reason,
                    error_code,
                    error_message,
                ) or "degraded result",
            }
        return {"status": "PASS", "blocker_reason": ""}
    return {
        "status": "FAIL",
        "blocker_reason": _join_reasons(status or "UNKNOWN", error_code, error_message) or "non-terminal operation",
    }


def is_expected_external_blocker(row: Mapping[str, Any]) -> bool:
    """LOGIN REQUIRED / FNO_UNIVERSE_UNAVAILABLE on data/F&O lanes only."""
    feature = str(row.get("feature") or "")
    status = str(row.get("status") or "").upper()
    reason = str(row.get("blocker_reason") or "").upper()
    if feature not in EXPECTED_EXTERNAL_FEATURES:
        return False
    if status not in {"DEGRADED", "BLOCKED"}:
        return False
    return any(token in reason for token in EXPECTED_EXTERNAL_TOKENS)


def grade_canonical_health(payload: Mapping[str, Any] | None) -> dict[str, Any]:
    """Fail-closed canonical /api/health gate. Missing safety fields are never PASS."""
    data = _as_dict(payload)
    reasons: list[str] = []
    if "ok" not in data or data.get("ok") is not True:
        reasons.append("health.ok is not True")
    if "live_locked" not in data:
        reasons.append("missing live_locked")
    elif data.get("live_locked") is not True:
        reasons.append("live_locked is not True")
    if "operational_ready" not in data:
        reasons.append("missing operational_ready")
    elif data.get("operational_ready") is not True:
        reasons.append("operational_ready is not True")
    if "evidence_ready" not in data:
        reasons.append("missing evidence_ready")
    elif data.get("evidence_ready") is not True:
        reasons.append("evidence_ready is not True")
    if "lifecycle" not in data:
        reasons.append("missing lifecycle")
    elif data.get("lifecycle") != "READY":
        reasons.append(f"lifecycle {data.get('lifecycle')!r} is not READY")
    live_locked = data.get("live_locked") is True if "live_locked" in data else False
    if reasons:
        return {
            "status": "FAIL",
            "blocker_reason": _join_reasons(*reasons),
            "live_locked": live_locked,
        }
    return {"status": "PASS", "blocker_reason": "", "live_locked": True}


def product_acceptance_verdict(
    rows: Sequence[Mapping[str, Any]],
    *,
    live_locked: bool,
) -> dict[str, Any]:
    """Overall PASS is allowed with the known F&O login blocker, not arbitrary DEGRADED."""
    if not live_locked:
        return {
            "verdict": "PRODUCT ACCEPTANCE HOLD",
            "exit_code": 1,
            "reason": "live money was not locked",
        }
    fails = [str(row.get("feature") or "") for row in rows if str(row.get("status") or "").upper() == "FAIL"]
    if fails:
        return {
            "verdict": "PRODUCT ACCEPTANCE HOLD",
            "exit_code": 1,
            "reason": "internal FAILED feature: " + ", ".join(fails),
        }
    unexpected: list[str] = []
    expected: list[str] = []
    for row in rows:
        status = str(row.get("status") or "").upper()
        feature = str(row.get("feature") or "")
        if status == "PASS":
            continue
        if is_expected_external_blocker(row):
            expected.append(f"{feature}={status}")
            continue
        unexpected.append(f"{feature}={status}")
    if unexpected:
        return {
            "verdict": "PRODUCT ACCEPTANCE HOLD",
            "exit_code": 1,
            "reason": "non-pass feature is not an expected external blocker: " + ", ".join(unexpected),
        }
    return {
        "verdict": "PRODUCT ACCEPTANCE PASS",
        "exit_code": 0,
        "reason": (
            "required features passed; expected external blocker: " + ", ".join(expected)
            if expected
            else "required features passed; live money locked"
        ),
    }


def _request_json(
    url: str,
    *,
    method: str = "GET",
    timeout: float = 12.0,
    body: dict[str, Any] | None = None,
) -> dict[str, Any]:
    encoded = json.dumps(body).encode("utf-8") if body is not None else (b"" if method != "GET" else None)
    request = urllib.request.Request(
        url,
        method=method,
        headers={"Accept": "application/json", "Content-Type": "application/json"},
        data=encoded,
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            raw = response.read()
            payload = json.loads(raw.decode("utf-8")) if raw else {}
            if not isinstance(payload, dict):
                raise RuntimeError(f"{url} returned non-object JSON")
            payload["_http_status"] = int(response.status)
            return payload
    except urllib.error.HTTPError as exc:
        response_body = exc.read().decode("utf-8", errors="replace")[:800]
        raise RuntimeError(f"HTTP {exc.code} {url}: {response_body}") from exc


def _url(base: str, path: str) -> str:
    return base.rstrip("/") + path


def _row(**kwargs: Any) -> dict[str, Any]:
    base = {
        "feature": "",
        "trigger_tested": "",
        "operation_id": "",
        "start_timestamp": "",
        "finish_timestamp": "",
        "backend_path": "",
        "durable_artifact": "",
        "freshness": "",
        "result_count": 0,
        "status": "FAIL",
        "blocker_reason": "",
        "code_sha": _sha(),
    }
    base.update(kwargs)
    return base


def _wait_operation(api: str, operation_id: str, *, timeout: float, request_timeout: float) -> dict[str, Any]:
    deadline = time.monotonic() + timeout
    last = ""
    while time.monotonic() < deadline:
        payload = _request_json(_url(api, f"/api/operations/{urllib.parse.quote(operation_id, safe='')}"), timeout=request_timeout)
        status = str(payload.get("status") or "UNKNOWN").upper()
        line = f"{status} · {payload.get('stage') or ''} · {payload.get('progress_current') or 0}/{payload.get('progress_total') or '?'}"
        if line != last:
            print(f"    {line}")
            last = line
        if status in TERMINAL:
            return payload
        time.sleep(1.25)
    raise TimeoutError(f"operation {operation_id} did not finish within {int(timeout)} seconds")


def _start_control(api: str, control: str, timeout: float) -> dict[str, Any]:
    payload = _request_json(_url(api, f"/api/controls/{urllib.parse.quote(control, safe='')}"), method="POST", timeout=timeout)
    if payload.get("accepted") is not True:
        raise RuntimeError(f"{control} was not accepted: {payload}")
    if not payload.get("operation_id") and not payload.get("control_id"):
        raise RuntimeError(f"{control} accepted without operation_id/control_id: {payload}")
    return payload


def grade_stock_intelligence(payload: Mapping[str, Any] | None, symbol: str) -> dict[str, str]:
    data = _as_dict(payload)
    if not data:
        return {"status": "FAIL", "blocker_reason": "empty stock-intelligence payload"}
    got = str(data.get("symbol") or data.get("ticker") or "").upper()
    if got != str(symbol or "").upper():
        return {"status": "FAIL", "blocker_reason": f"symbol missing or mismatched ({got or 'none'})"}
    if data.get("schema_version") in {None, ""}:
        return {"status": "FAIL", "blocker_reason": "missing schema_version"}
    sources = _as_list(data.get("sources"))
    if not sources or not all(isinstance(item, Mapping) for item in sources):
        return {"status": "FAIL", "blocker_reason": "missing sources contract"}
    if not isinstance(data.get("technical"), Mapping) or not isinstance(data.get("fundamentals"), Mapping):
        return {"status": "FAIL", "blocker_reason": "missing technical/fundamentals structure"}
    return {"status": "PASS", "blocker_reason": ""}


def grade_due_diligence(payload: Mapping[str, Any] | None, symbol: str) -> dict[str, str]:
    data = _as_dict(payload)
    if not data:
        return {"status": "FAIL", "blocker_reason": "empty due-diligence payload"}
    got = str(data.get("symbol") or "").upper()
    if got != str(symbol or "").upper():
        return {"status": "FAIL", "blocker_reason": f"symbol missing or mismatched ({got or 'none'})"}
    if data.get("schema_version") in {None, ""}:
        return {"status": "FAIL", "blocker_reason": "missing schema_version"}
    kpis = data.get("kpis") if isinstance(data.get("kpis"), list) else data.get("findings")
    if not isinstance(kpis, list):
        return {"status": "FAIL", "blocker_reason": "missing kpis/findings"}
    coverage = data.get("research_coverage") if isinstance(data.get("research_coverage"), Mapping) else data.get("decision_coverage")
    if not isinstance(coverage, Mapping) and data.get("decision_coverage_pct") is None:
        return {"status": "FAIL", "blocker_reason": "missing coverage contract"}
    verdict = data.get("vs_technical_setup") or data.get("fundamental_confirmation") or data.get("thesis")
    if verdict in {None, ""}:
        return {"status": "FAIL", "blocker_reason": "missing Investigate verdict"}
    return {"status": "PASS", "blocker_reason": ""}


def grade_recommendations(payload: Mapping[str, Any] | None) -> dict[str, Any]:
    data = _as_dict(payload)
    categories = data.get("categories")
    if not isinstance(categories, list):
        return {"status": "FAIL", "blocker_reason": "missing categories contract", "cards": 0}
    if not categories or not all(isinstance(item, Mapping) and "cards" in item for item in categories):
        return {"status": "FAIL", "blocker_reason": "categories are not structured recommendation buckets", "cards": 0}
    rec_status = str(data.get("records_status") or "")
    if rec_status == "FAILED":
        return {"status": "FAIL", "blocker_reason": "records_status=FAILED", "cards": 0}
    if data.get("rebuilding"):
        return {"status": "DEGRADED", "blocker_reason": "projection rebuilding from persisted scan", "cards": 0}
    if not (data.get("generated_at") or data.get("scan_scanned_at")):
        return {"status": "FAIL", "blocker_reason": "missing generation timestamp", "cards": 0}
    cards = 0
    for cat in categories:
        cards += len(_as_list(cat.get("cards")))
    return {"status": "PASS", "blocker_reason": "", "cards": cards}


def grade_market_reports(payload: Mapping[str, Any] | None) -> dict[str, Any]:
    data = _as_dict(payload)
    reports = [item for item in _as_list(data.get("reports")) if isinstance(item, Mapping)]
    missing = [str(item) for item in _as_list(data.get("missing_lanes")) if str(item).strip()]
    if data.get("error") and not reports:
        return {"status": "FAIL", "blocker_reason": str(data.get("error")), "count": 0}
    if not reports:
        return {"status": "FAIL", "blocker_reason": "no durable market reports", "count": 0}
    structured = 0
    for item in reports:
        if item.get("id") or item.get("title") or item.get("as_of") or item.get("as_of_ist") or item.get("body") or item.get("sections"):
            structured += 1
    if structured <= 0:
        return {"status": "FAIL", "blocker_reason": "reports present but unstructured", "count": len(reports)}
    reason = ",".join(missing)
    return {"status": "PASS", "blocker_reason": reason, "count": len(reports)}


def grade_simulator(payload: Mapping[str, Any] | None) -> dict[str, str]:
    data = _as_dict(payload)
    sim_status = str(data.get("status") or "").upper()
    if sim_status in {"UNAVAILABLE", "HISTORICAL_DECISION_UNAVAILABLE", "AMBIGUOUS_HISTORICAL_DECISION"}:
        return {"status": "DEGRADED", "blocker_reason": sim_status}
    if sim_status == "RUNNING":
        return {"status": "FAIL", "blocker_reason": "simulator still RUNNING; not treated as success"}
    if sim_status != "SUCCEEDED":
        return {"status": "FAIL", "blocker_reason": sim_status or "missing simulator status"}
    original = _as_dict(data.get("original") or data.get("decision"))
    action = str(original.get("action") or data.get("decision") or "").strip()
    if not action or action.upper() in {"UNAVAILABLE", "NONE", "NULL"}:
        return {"status": "FAIL", "blocker_reason": "SUCCEEDED without historical decision/reasons"}
    if data.get("kind") not in {"PAST_DECISION_SIMULATION"} and data.get("schema_version") in {None, ""}:
        return {"status": "FAIL", "blocker_reason": "missing simulator schema"}
    return {"status": "PASS", "blocker_reason": ""}


def grade_learning_dashboard(payload: Mapping[str, Any] | None) -> dict[str, Any]:
    data = _as_dict(payload)
    if not data:
        return {"status": "FAIL", "blocker_reason": "empty learning dashboard", "count": 0}
    if data.get("schema_version") in {None, ""}:
        return {"status": "FAIL", "blocker_reason": "missing schema_version", "count": 0}
    if data.get("live_locked") is not True:
        return {"status": "FAIL", "blocker_reason": "learning dashboard did not lock live money", "count": 0}
    if "policies" not in data or "counterfactuals" not in data:
        return {"status": "FAIL", "blocker_reason": "missing policies/counterfactuals contract", "count": 0}
    count = len(_as_list(data.get("policies")))
    return {"status": "PASS", "blocker_reason": "", "count": count}


def grade_forward_soak(payload: Mapping[str, Any] | None) -> dict[str, str]:
    data = _as_dict(payload)
    verification = data.get("verification")
    if not isinstance(verification, Mapping):
        return {"status": "FAIL", "blocker_reason": "missing verification contract"}
    lanes = verification.get("lanes") if isinstance(verification.get("lanes"), Mapping) else data.get("lanes")
    if not isinstance(lanes, Mapping):
        return {"status": "FAIL", "blocker_reason": "verification missing lanes"}
    live = verification.get("live_locked")
    if live is None:
        live = data.get("live_locked")
    if live is not True:
        return {"status": "FAIL", "blocker_reason": "forward soak did not prove live_locked"}
    return {"status": "PASS", "blocker_reason": ""}


def grade_paper_status(payload: Mapping[str, Any] | None, *, live_locked: bool) -> dict[str, Any]:
    data = _as_dict(payload)
    if not data:
        return {"status": "FAIL", "blocker_reason": "empty paper-autopilot payload", "count": 0}
    nested = _as_dict(data.get("paper"))
    positions = nested.get("open_positions") if "open_positions" in nested else data.get("open_positions")
    has_cycle = "last_cycle" in data or "latest" in data or "why_no_trade" in data
    if positions is None or not has_cycle:
        return {"status": "FAIL", "blocker_reason": "missing paper positions/cycle contract", "count": 0}
    if data.get("live_locked") is False or not live_locked:
        return {"status": "FAIL", "blocker_reason": "live money unlocked", "count": 0}
    count = len(_as_list(positions))
    return {"status": "PASS", "blocker_reason": "", "count": count}


def _persisted_decision(symbol: str, as_of: str) -> dict[str, Any]:
    """Pick a real journal row. Never invents a decision."""
    try:
        from product.decision_journal import hydrate, list_for_session, list_for_symbol
    except Exception:
        return {}
    rows = []
    name = str(symbol or "").strip().upper()
    session = str(as_of or "")[:10]
    if name:
        if session:
            rows = list_for_symbol(name, as_of=session, limit=8)
        if not rows:
            rows = list_for_symbol(name, limit=8)
    if not rows and session:
        rows = list_for_session(session, limit=8)
    if not rows:
        try:
            from product.decision_journal import _connect
            con = _connect()
            raw = con.execute(
                "SELECT * FROM decisions ORDER BY decision_time DESC LIMIT 1"
            ).fetchone()
            con.close()
            if raw is not None:
                rows = [hydrate(dict(raw))]
        except Exception:
            rows = []
    row = next((dict(item) for item in rows if isinstance(item, Mapping) and item.get("decision_id")), {})
    return row


def _cycle_identity(cycle: Mapping[str, Any] | None) -> str:
    data = _as_dict(cycle)
    return "|".join(
        str(data.get(key) or "")
        for key in ("cycle_id", "as_of_date", "as_of", "eligibility", "status", "finished_at")
    )


def _paper_jobs(autonomy: Mapping[str, Any] | None) -> list[dict[str, Any]]:
    jobs = [
        dict(item)
        for item in _as_list(_as_dict(autonomy).get("jobs_recent"))
        if isinstance(item, Mapping) and str(item.get("job_type") or "") == "paper_cycle"
    ]
    return jobs


def grade_paper_cycle_execution(
    *,
    job: Mapping[str, Any] | None = None,
    last_cycle: Mapping[str, Any] | None = None,
    observed: bool = False,
) -> dict[str, str]:
    job_d = _as_dict(job)
    cycle = _as_dict(last_cycle)
    job_status = str(job_d.get("status") or "").upper()
    eligibility = str(
        cycle.get("eligibility")
        or cycle.get("decision")
        or job_d.get("result_summary")
        or ""
    ).upper()
    if job_status in {"FAILED", "CANCELLED"}:
        return {
            "status": "FAIL",
            "blocker_reason": _join_reasons(job_d.get("error_code"), job_d.get("error_message"), job_status),
        }
    if job_status == "BLOCKED":
        reason = _join_reasons(job_d.get("error_code"), job_d.get("blocked_reason"), job_d.get("blocked_on"))
        if _has_explicit_blocker(job_d, []):
            return {"status": "BLOCKED", "blocker_reason": reason or "paper_cycle BLOCKED"}
        return {"status": "FAIL", "blocker_reason": _join_reasons("paper_cycle BLOCKED without explicit blocker", reason)}
    if observed and (job_status == "SUCCEEDED" or any(token in eligibility for token in PAPER_CYCLE_DONE)):
        if "NO_ELIGIBLE_TRADE" in eligibility:
            return {"status": "PASS", "blocker_reason": "NO_ELIGIBLE_TRADE after completed cycle"}
        if "TRADED" in eligibility:
            return {"status": "PASS", "blocker_reason": ""}
        if job_status == "SUCCEEDED":
            return {
                "status": "PASS",
                "blocker_reason": str(job_d.get("result_summary") or eligibility or "paper_cycle SUCCEEDED"),
            }
        return {"status": "PASS", "blocker_reason": eligibility}
    return {
        "status": "DEGRADED",
        "blocker_reason": "control accepted; durable cycle completion not observed",
    }


def run(args: argparse.Namespace) -> int:
    sha = _sha()
    rows: list[dict[str, Any]] = []
    api = args.api
    print(f"QuantTerm real product acceptance\nSHA {sha}\nAPI {api}\n")

    health = _request_json(_url(api, "/api/health"), timeout=args.request_timeout)
    health_grade = grade_canonical_health(health)
    live_locked = bool(health_grade.get("live_locked") is True)
    rows.append(_row(
        feature="Canonical stack / readiness",
        trigger_tested="GET /api/health",
        backend_path="product.runtime_lifecycle.inspect_runtime + terminal_api",
        durable_artifact="logs/market_ops/runtime.json",
        freshness=str(health.get("lifecycle") if "lifecycle" in health else ""),
        result_count=len(health.get("components") or []),
        status=health_grade["status"],
        blocker_reason=health_grade["blocker_reason"],
        start_timestamp=_now(),
        finish_timestamp=_now(),
        code_sha=sha,
    ))
    if not live_locked:
        print("NOT WORKING: live money lock was not explicitly proven")
        dest = Path(args.output)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(json.dumps({
            "schema_version": 2,
            "generated_at": _now(),
            "code_sha": sha,
            "api": api,
            "live_locked": False,
            "verdict": "PRODUCT ACCEPTANCE HOLD",
            "verdict_reason": health_grade["blocker_reason"] or "live money was not locked",
            "features": rows,
        }, indent=2), encoding="utf-8")
        print(f"Wrote {dest}")
        return 1

    ui_ok = False
    try:
        urllib.request.urlopen("http://127.0.0.1:5173/", timeout=5).read(32)
        ui_ok = True
    except Exception:
        ui_ok = False
    report_ok = False
    report: dict[str, Any] = {}
    try:
        report = _request_json("http://127.0.0.1:8766/health", timeout=5)
        report_ok = bool(report.get("ok"))
    except Exception as exc:
        report = {"error": str(exc)}
    rows.append(_row(
        feature="UI + report endpoints",
        trigger_tested="GET :5173/ and GET :8766/health",
        backend_path="vite + report_api",
        status="PASS" if ui_ok and report_ok else "FAIL",
        blocker_reason="" if ui_ok and report_ok else f"ui={ui_ok} report={report_ok} {report.get('error') or ''}",
        start_timestamp=_now(),
        finish_timestamp=_now(),
        result_count=int(ui_ok) + int(report_ok),
        code_sha=sha,
    ))

    market = dict((_request_json(_url(api, "/api/dashboard"), timeout=args.request_timeout).get("market") or {}))
    if market.get("available") is True and str(market.get("health") or "") == "Unavailable":
        raise RuntimeError("dashboard claimed available=true for Unavailable regime")
    if str(market.get("health") or "").lower() in {"healthy", "mixed", "weak"} and market.get("available") is False:
        raise RuntimeError("dashboard invented a market stance while available=false")

    controls = [
        ("Market Scan", "RUN_SCAN_NOW", "logs/product/latest_momentum_scan.json"),
        ("News refresh", "REFRESH_NEWS_NOW", "logs/news_curator.sqlite3"),
        ("Market Report", "REFRESH_MARKET_REPORT_NOW", "logs/product/market_reports/"),
        ("Data refresh", "REFRESH_DATA_NOW", "official history / DATA_PREPARE"),
        ("F&O refresh", "REFRESH_FNO_NOW", "logs/product/fno_universe.json"),
    ]
    if args.include_long_term:
        controls.insert(1, ("Long-term fundamentals", "REFRESH_LONG_TERM_NOW", "logs/product/latest_long_term_scan.json"))

    for label, control, artifact in controls:
        started = _now()
        print(f"[RUN] {label} · {control}")
        try:
            queued = _start_control(api, control, args.request_timeout)
            op = _wait_operation(api, str(queued["operation_id"]), timeout=args.operation_timeout, request_timeout=args.request_timeout)
            graded = classify_operation(op)
            count = int(op.get("progress_current") or op.get("progress_total") or 0)
            rows.append(_row(
                feature=label,
                trigger_tested=f"POST /api/controls/{control}",
                operation_id=str(queued.get("operation_id") or ""),
                start_timestamp=started,
                finish_timestamp=_now(),
                backend_path="operations.market_ops",
                durable_artifact=artifact,
                freshness=str(op.get("status") or ""),
                result_count=count,
                status=graded["status"],
                blocker_reason=graded["blocker_reason"],
                code_sha=sha,
            ))
            print(f"[{graded['status']}] {label} · {op.get('status')} · {graded['blocker_reason']}\n")
        except Exception as exc:
            rows.append(_row(
                feature=label,
                trigger_tested=f"POST /api/controls/{control}",
                start_timestamp=started,
                finish_timestamp=_now(),
                backend_path="operations.market_ops",
                durable_artifact=artifact,
                status="FAIL",
                blocker_reason=str(exc)[:300],
                code_sha=sha,
            ))
            print(f"[FAIL] {label}: {exc}\n")

    rec = {}
    rec_deadline = time.monotonic() + 90
    while time.monotonic() < rec_deadline:
        rec = _request_json(_url(api, "/api/recommendations-workspace"), timeout=max(args.request_timeout, 30))
        if not rec.get("rebuilding"):
            break
        time.sleep(1.5)
    rec_grade = grade_recommendations(rec)
    rows.append(_row(
        feature="Recommendations",
        trigger_tested="GET /api/recommendations-workspace",
        backend_path="product.recommendations_liveness",
        durable_artifact="logs/product/latest_recommendations.json",
        freshness=str(rec.get("scan_scanned_at") or rec.get("records_status") or ""),
        result_count=int(rec_grade.get("cards") or 0),
        status=rec_grade["status"],
        blocker_reason=rec_grade["blocker_reason"],
        start_timestamp=_now(),
        finish_timestamp=_now(),
        code_sha=sha,
    ))

    reports = _request_json(_url(api, "/api/market-reports-workspace"), timeout=args.request_timeout)
    report_grade = grade_market_reports(reports)
    rows.append(_row(
        feature="Market Report workspace",
        trigger_tested="GET /api/market-reports-workspace",
        backend_path="product.recommendations_workspace.build_market_reports_workspace",
        durable_artifact="logs/product/market_reports/",
        freshness=str(reports.get("as_of_ist") or ""),
        result_count=int(report_grade.get("count") or 0),
        status=report_grade["status"],
        blocker_reason=report_grade["blocker_reason"],
        start_timestamp=_now(),
        finish_timestamp=_now(),
        code_sha=sha,
    ))

    symbol = args.symbol.strip().upper() or "TCS"
    intel = _request_json(_url(api, f"/api/stock-intelligence/{urllib.parse.quote(symbol)}"), timeout=args.request_timeout)
    intel_grade = grade_stock_intelligence(intel, symbol)
    rows.append(_row(
        feature="Stock Intelligence",
        trigger_tested=f"GET /api/stock-intelligence/{symbol}",
        backend_path="product.stock_workspace.build_stock_workspace",
        durable_artifact=f"research/fundamentals cache · {symbol}",
        result_count=1 if intel_grade["status"] == "PASS" else 0,
        status=intel_grade["status"],
        blocker_reason=intel_grade["blocker_reason"],
        start_timestamp=_now(),
        finish_timestamp=_now(),
        code_sha=sha,
    ))

    dd = _request_json(_url(api, f"/api/due-diligence/{urllib.parse.quote(symbol)}"), timeout=args.request_timeout)
    dd_grade = grade_due_diligence(dd, symbol)
    rows.append(_row(
        feature="Investigate / Due Diligence read",
        trigger_tested=f"GET /api/due-diligence/{symbol}",
        backend_path="product.due_diligence.build_due_diligence",
        durable_artifact=f"logs/research_evidence/{symbol}/",
        result_count=len(_as_list(dd.get("kpis") or dd.get("findings"))),
        status=dd_grade["status"],
        freshness=str((dd.get("freshness") or _as_dict(dd.get("as_of")).get("generated_at") or dd.get("status") or "") if isinstance(dd, dict) else ""),
        blocker_reason=dd_grade["blocker_reason"],
        start_timestamp=_now(),
        finish_timestamp=_now(),
        code_sha=sha,
    ))

    if args.acquire:
        started = _now()
        print(f"[RUN] Due diligence acquire · {symbol}")
        try:
            queued = _request_json(
                _url(api, f"/api/due-diligence/{urllib.parse.quote(symbol)}/acquire?mode=missing_or_stale&async_job=true"),
                method="POST",
                timeout=args.request_timeout,
            )
            if queued.get("report"):
                status, op_id, reason = "PASS", "", ""
            else:
                op_id = str(queued.get("operation_id") or "")
                op = _wait_operation(api, op_id, timeout=args.acquire_timeout, request_timeout=args.request_timeout)
                graded = classify_operation(op)
                status, reason = graded["status"], graded["blocker_reason"]
            rows.append(_row(
                feature="Investigate acquire",
                trigger_tested=f"POST /api/due-diligence/{symbol}/acquire",
                operation_id=op_id,
                start_timestamp=started,
                finish_timestamp=_now(),
                backend_path="operations.market_ops DUE_DILIGENCE_ACQUIRE",
                durable_artifact=f"logs/research_evidence/{symbol}/autonomy/",
                status=status,
                blocker_reason=reason,
                code_sha=sha,
            ))
        except Exception as exc:
            rows.append(_row(
                feature="Investigate acquire",
                trigger_tested=f"POST /api/due-diligence/{symbol}/acquire",
                start_timestamp=started,
                finish_timestamp=_now(),
                status="FAIL",
                blocker_reason=str(exc)[:300],
                code_sha=sha,
            ))

    started = _now()
    try:
        query = urllib.parse.urlencode({"symbol": symbol, "as_of": args.as_of})
        sim = _request_json(_url(api, f"/api/decision-simulator?{query}"), timeout=max(args.request_timeout, 60))
        if str(sim.get("status") or "").upper() in {
            "HISTORICAL_DECISION_UNAVAILABLE",
            "AMBIGUOUS_HISTORICAL_DECISION",
            "UNAVAILABLE",
        }:
            persisted = _persisted_decision(symbol, str(args.as_of))
            if persisted.get("decision_id"):
                query = urllib.parse.urlencode({
                    "symbol": str(persisted.get("symbol") or symbol),
                    "as_of": str(persisted.get("market_as_of") or args.as_of)[:10],
                    "decision_id": str(persisted.get("decision_id")),
                })
                sim = _request_json(
                    _url(api, f"/api/decision-simulator?{query}"),
                    timeout=max(args.request_timeout, 60),
                )
        sim_grade = grade_simulator(sim)
        rows.append(_row(
            feature="Simulate Past Decision",
            trigger_tested="GET /api/decision-simulator?symbol=&as_of=&decision_id=",
            start_timestamp=started,
            finish_timestamp=_now(),
            backend_path="product.decision_simulator",
            durable_artifact="logs/product/decisions.db",
            freshness=str(sim.get("status") or ""),
            status=sim_grade["status"],
            blocker_reason=sim_grade["blocker_reason"] or str(sim.get("symbol") or ""),
            result_count=1 if sim_grade["status"] == "PASS" else 0,
            operation_id=str(sim.get("decision_id") or ""),
            code_sha=sha,
        ))
    except Exception as exc:
        rows.append(_row(
            feature="Simulate Past Decision",
            trigger_tested="GET /api/decision-simulator",
            start_timestamp=started,
            finish_timestamp=_now(),
            backend_path="product.decision_simulator",
            status="FAIL",
            blocker_reason=str(exc)[:300],
            code_sha=sha,
        ))

    paper = _request_json(_url(api, "/api/paper-autopilot"), timeout=args.request_timeout)
    paper_grade = grade_paper_status(paper, live_locked=live_locked)
    rows.append(_row(
        feature="Paper execution status",
        trigger_tested="GET /api/paper-autopilot",
        backend_path="product.paper_autopilot",
        durable_artifact="logs/product paper book / journal",
        result_count=int(paper_grade.get("count") or 0),
        status=paper_grade["status"],
        freshness=str(paper.get("status") or paper.get("state") or ""),
        blocker_reason=paper_grade["blocker_reason"],
        start_timestamp=_now(),
        finish_timestamp=_now(),
        code_sha=sha,
    ))

    before_cycle = _cycle_identity(paper.get("last_cycle") if isinstance(paper, dict) else {})
    before_jobs: list[dict[str, Any]] = []
    try:
        before_dash = _request_json(_url(api, "/api/dashboard"), timeout=max(args.request_timeout, 20))
        before_jobs = _paper_jobs(_as_dict(before_dash.get("autonomy")))
    except Exception:
        before_jobs = []
    before_job_ids = {str(job.get("job_id") or "") for job in before_jobs if job.get("job_id")}
    before_finished = max((float(job.get("finished_at") or 0) for job in before_jobs), default=0.0)

    started = _now()
    try:
        cycle = _start_control(api, "RUN_CYCLE_NOW", args.request_timeout)
        control_id = str(cycle.get("control_id") or cycle.get("operation_id") or "")
        rows.append(_row(
            feature="Paper cycle request",
            trigger_tested="POST /api/controls/RUN_CYCLE_NOW",
            operation_id=control_id,
            start_timestamp=started,
            finish_timestamp=_now(),
            backend_path="research.autonomy.controls → paper_cycle",
            durable_artifact="logs/autonomy/",
            status="PASS" if cycle.get("accepted") and control_id else "FAIL",
            blocker_reason="" if cycle.get("accepted") else str(cycle)[:200],
            freshness="request accepted" if cycle.get("accepted") else "",
            code_sha=sha,
        ))
        observed_job: dict[str, Any] = {}
        observed_cycle: dict[str, Any] = {}
        observed = False
        deadline = time.monotonic() + float(args.cycle_timeout)
        while time.monotonic() < deadline:
            try:
                paper_now = _request_json(_url(api, "/api/paper-autopilot"), timeout=args.request_timeout)
            except Exception:
                paper_now = {}
            last_cycle = _as_dict(paper_now.get("last_cycle") or paper_now.get("latest"))
            if not last_cycle:
                last_cycle = _as_dict(_as_dict(paper_now.get("paper")).get("last_cycle"))
            identity = _cycle_identity(last_cycle)
            if last_cycle and identity and identity != before_cycle:
                observed_cycle = last_cycle
                observed = True
            try:
                dash = _request_json(_url(api, "/api/dashboard"), timeout=max(args.request_timeout, 20))
                jobs = _paper_jobs(_as_dict(dash.get("autonomy")))
            except Exception:
                jobs = []
            for job in jobs:
                job_id = str(job.get("job_id") or "")
                finished_at = float(job.get("finished_at") or 0)
                job_status = str(job.get("status") or "").upper()
                newer = (job_id and job_id not in before_job_ids) or (finished_at > before_finished > 0 and finished_at >= before_finished)
                if newer and job_status in TERMINAL:
                    observed_job = job
                    observed = True
                    break
            if observed and (observed_job or any(token in str(observed_cycle.get("eligibility") or "").upper() for token in PAPER_CYCLE_DONE)):
                break
            time.sleep(1.5)
        exec_grade = grade_paper_cycle_execution(
            job=observed_job,
            last_cycle=observed_cycle or _as_dict((paper.get("last_cycle") if isinstance(paper, dict) else {})),
            observed=observed,
        )
        rows.append(_row(
            feature="Paper cycle execution",
            trigger_tested="durable last_cycle / autonomy jobs_recent",
            operation_id=str(observed_job.get("job_id") or control_id),
            start_timestamp=started,
            finish_timestamp=_now(),
            backend_path="research.autonomy.jobs.run_paper_cycle",
            durable_artifact="logs/autonomy/status.json last_cycle",
            status=exec_grade["status"],
            blocker_reason=exec_grade["blocker_reason"],
            freshness=str(observed_job.get("status") or observed_cycle.get("eligibility") or "unobserved"),
            code_sha=sha,
        ))
    except Exception as exc:
        rows.append(_row(
            feature="Paper cycle request",
            trigger_tested="POST /api/controls/RUN_CYCLE_NOW",
            start_timestamp=started,
            finish_timestamp=_now(),
            status="FAIL",
            blocker_reason=str(exc)[:300],
            code_sha=sha,
        ))

    learning = _request_json(_url(api, "/api/learning-dashboard"), timeout=args.request_timeout)
    learn_grade = grade_learning_dashboard(learning)
    rows.append(_row(
        feature="Learning loop dashboard",
        trigger_tested="GET /api/learning-dashboard",
        backend_path="product learning ledger",
        durable_artifact="logs/product/taken_evidence.jsonl",
        result_count=int(learn_grade.get("count") or 0),
        status=learn_grade["status"],
        blocker_reason=learn_grade["blocker_reason"],
        start_timestamp=_now(),
        finish_timestamp=_now(),
        code_sha=sha,
    ))

    soak = _request_json(_url(api, "/api/forward-soak"), method="POST", timeout=args.request_timeout)
    soak_grade = grade_forward_soak(soak)
    rows.append(_row(
        feature="Forward soak verification",
        trigger_tested="POST /api/forward-soak",
        backend_path="product.forward_soak",
        durable_artifact="forward soak ledger",
        status=soak_grade["status"],
        blocker_reason=soak_grade["blocker_reason"],
        start_timestamp=_now(),
        finish_timestamp=_now(),
        code_sha=sha,
    ))

    verdict = product_acceptance_verdict(rows, live_locked=live_locked)
    out = {
        "schema_version": 2,
        "generated_at": _now(),
        "code_sha": sha,
        "api": api,
        "live_locked": live_locked,
        "verdict": verdict["verdict"],
        "verdict_reason": verdict["reason"],
        "features": rows,
    }
    dest = Path(args.output)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nWrote {dest}")
    for row in rows:
        print(f"  {row['status']:9} {row['feature']}: {row.get('blocker_reason') or row.get('freshness') or ''}")
    print(f"\n{verdict['verdict']}")
    print(verdict["reason"])
    return int(verdict["exit_code"])


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run live QuantTerm product acceptance.")
    parser.add_argument("--api", default="http://127.0.0.1:8765")
    parser.add_argument("--request-timeout", type=float, default=12.0)
    parser.add_argument("--operation-timeout", type=float, default=1200.0)
    parser.add_argument("--acquire-timeout", type=float, default=720.0)
    parser.add_argument("--cycle-timeout", type=float, default=180.0)
    parser.add_argument("--symbol", default="TCS")
    parser.add_argument("--as-of", dest="as_of", default="2024-01-15")
    parser.add_argument("--acquire", action="store_true")
    parser.add_argument("--include-long-term", action="store_true")
    parser.add_argument("--output", default=str(ROOT / "logs" / "product" / "product_acceptance.json"))
    return parser


if __name__ == "__main__":
    try:
        raise SystemExit(run(build_parser().parse_args()))
    except KeyboardInterrupt:
        raise SystemExit(130)
    except Exception as exc:
        print(f"NOT WORKING: {exc}", file=sys.stderr)
        raise SystemExit(1)
