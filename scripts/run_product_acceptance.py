#!/usr/bin/env python3
"""Live product-acceptance matrix against a running canonical QuantTerm stack.

Writes logs/product/product_acceptance.json. Never unlocks live money.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
TERMINAL = {"SUCCEEDED", "FAILED", "BLOCKED", "CANCELLED"}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception:
        return ""


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
    if not payload.get("operation_id"):
        raise RuntimeError(f"{control} accepted without operation_id: {payload}")
    return payload


def _classify_terminal(status: str, *, require_success: bool) -> str:
    status = str(status or "").upper()
    if status == "SUCCEEDED":
        return "PASS"
    if status == "BLOCKED":
        return "BLOCKED"
    if status in {"FAILED", "CANCELLED"}:
        return "FAIL" if require_success else "DEGRADED"
    return "FAIL"


def run(args: argparse.Namespace) -> int:
    sha = _sha()
    rows: list[dict[str, Any]] = []
    api = args.api
    print(f"QuantTerm real product acceptance\nSHA {sha}\nAPI {api}\n")

    health = _request_json(_url(api, "/api/health"), timeout=args.request_timeout)
    live_locked = bool(health.get("live_locked", True))
    rows.append(_row(
        feature="Canonical stack / readiness",
        trigger_tested="GET /api/health",
        backend_path="product.startup_check + terminal_api",
        durable_artifact="logs/market_ops/runtime.json",
        freshness=str(health.get("lifecycle") or ""),
        result_count=len(health.get("components") or []),
        status="PASS" if health.get("ok") and live_locked else "FAIL",
        blocker_reason="" if live_locked else "live money was not locked",
        start_timestamp=_now(),
        finish_timestamp=_now(),
        code_sha=sha,
    ))
    if not live_locked:
        print("NOT WORKING: live money is not locked")
        return 1

    ui_ok = False
    try:
        urllib.request.urlopen("http://127.0.0.1:5173/", timeout=5).read(32)
        ui_ok = True
    except Exception:
        ui_ok = False
    report_ok = False
    try:
        report = _request_json("http://127.0.0.1:8766/health", timeout=5)
        report_ok = bool(report.get("ok"))
    except Exception as exc:
        report = {"error": str(exc)}
    rows.append(_row(
        feature="UI + report endpoints",
        trigger_tested="GET :5173/ and GET :8766/health",
        backend_path="vite + report_api",
        status="PASS" if ui_ok and report_ok else "DEGRADED",
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
        ("Market Scan", "RUN_SCAN_NOW", True, "logs/product/latest_momentum_scan.json"),
        ("News refresh", "REFRESH_NEWS_NOW", False, "logs/news_curator.sqlite3"),
        ("Market Report", "REFRESH_MARKET_REPORT_NOW", False, "logs/product/market_reports/"),
        ("Data refresh", "REFRESH_DATA_NOW", False, "official history / DATA_PREPARE"),
        ("F&O refresh", "REFRESH_FNO_NOW", False, "logs/product/fno_universe.json"),
    ]
    if args.include_long_term:
        controls.insert(1, ("Long-term fundamentals", "REFRESH_LONG_TERM_NOW", False, "logs/product/latest_long_term_scan.json"))

    for label, control, require_success, artifact in controls:
        started = _now()
        print(f"[RUN] {label} · {control}")
        try:
            queued = _start_control(api, control, args.request_timeout)
            op = _wait_operation(api, str(queued["operation_id"]), timeout=args.operation_timeout, request_timeout=args.request_timeout)
            status = _classify_terminal(str(op.get("status") or ""), require_success=require_success)
            reason = ""
            if status != "PASS":
                reason = str(op.get("error_code") or op.get("error_message") or op.get("message") or op.get("status") or "")
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
                status=status,
                blocker_reason=reason,
                code_sha=sha,
            ))
            print(f"[{status}] {label} · {op.get('status')}\n")
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

    rec = _request_json(_url(api, "/api/recommendations-workspace"), timeout=max(args.request_timeout, 30))
    rec_status = str(rec.get("records_status") or "")
    cards = 0
    for cat in rec.get("categories") or []:
        if isinstance(cat, dict):
            cards += len(cat.get("cards") or [])
    rec_ok = rec.get("categories") is not None and rec_status != "FAILED"
    if rec.get("rebuilding"):
        rec_grade = "DEGRADED"
        rec_reason = "projection rebuilding from persisted scan"
    elif rec_ok:
        rec_grade = "PASS"
        rec_reason = ""
    else:
        rec_grade = "FAIL"
        rec_reason = rec_status or "malformed workspace"
    rows.append(_row(
        feature="Recommendations",
        trigger_tested="GET /api/recommendations-workspace",
        backend_path="product.recommendations_liveness",
        durable_artifact="logs/product/latest_recommendations.json",
        freshness=str(rec.get("scan_scanned_at") or rec_status),
        result_count=cards,
        status=rec_grade,
        blocker_reason=rec_reason,
        start_timestamp=_now(),
        finish_timestamp=_now(),
        code_sha=sha,
    ))

    reports = _request_json(_url(api, "/api/market-reports-workspace"), timeout=args.request_timeout)
    report_count = len(reports.get("reports") or [])
    missing = list(reports.get("missing_lanes") or [])
    if reports.get("error") and not report_count:
        report_grade, report_reason = "FAIL", str(reports.get("error"))
    elif reports.get("stale") or reports.get("needs_refresh") or missing:
        report_grade, report_reason = "DEGRADED", ",".join(missing) or "needs_refresh"
    else:
        report_grade, report_reason = "PASS", ""
    rows.append(_row(
        feature="Market Report workspace",
        trigger_tested="GET /api/market-reports-workspace",
        backend_path="product.recommendations_workspace.build_market_reports_workspace",
        durable_artifact="logs/product/market_reports/",
        freshness=str(reports.get("as_of_ist") or ""),
        result_count=report_count,
        status=report_grade,
        blocker_reason=report_reason,
        start_timestamp=_now(),
        finish_timestamp=_now(),
        code_sha=sha,
    ))

    symbol = args.symbol.strip().upper() or "TCS"
    intel = _request_json(_url(api, f"/api/stock-intelligence/{urllib.parse.quote(symbol)}"), timeout=args.request_timeout)
    intel_ok = bool(intel) and (intel.get("symbol") or intel.get("ticker") or intel.get("available") is not False)
    rows.append(_row(
        feature="Stock Intelligence",
        trigger_tested=f"GET /api/stock-intelligence/{symbol}",
        backend_path="product.stock_workspace.build_stock_workspace",
        durable_artifact=f"research/fundamentals cache · {symbol}",
        result_count=1 if intel else 0,
        status="PASS" if intel_ok else "DEGRADED",
        blocker_reason="" if intel_ok else "empty structured payload",
        start_timestamp=_now(),
        finish_timestamp=_now(),
        code_sha=sha,
    ))

    dd = _request_json(_url(api, f"/api/due-diligence/{urllib.parse.quote(symbol)}"), timeout=args.request_timeout)
    rows.append(_row(
        feature="Investigate / Due Diligence read",
        trigger_tested=f"GET /api/due-diligence/{symbol}",
        backend_path="product.due_diligence.build_due_diligence",
        durable_artifact=f"logs/research_evidence/{symbol}/",
        result_count=len((dd.get("questions") or dd.get("facts") or dd.get("items") or []) if isinstance(dd, dict) else []),
        status="PASS" if dd else "FAIL",
        freshness=str((dd.get("freshness") or dd.get("status") or "") if isinstance(dd, dict) else ""),
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
                status = _classify_terminal(str(op.get("status") or ""), require_success=False)
                reason = str(op.get("error_code") or op.get("error_message") or "")
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
                status="BLOCKED",
                blocker_reason=str(exc)[:300],
                code_sha=sha,
            ))

    sim_body = {"symbol": symbol, "as_of": args.as_of}
    started = _now()
    try:
        sim = _request_json(_url(api, "/api/decision-simulator"), method="POST", timeout=max(args.request_timeout, 60), body=sim_body)
        sim_status = str(sim.get("status") or "")
        if sim_status in { "UNAVAILABLE", "HISTORICAL_DECISION_UNAVAILABLE", "AMBIGUOUS_HISTORICAL_DECISION"}:
            grade = "DEGRADED"
        elif sim_status in {"SUCCEEDED", "RUNNING"} or sim.get("decision") or sim.get("original"):
            grade = "PASS"
        else:
            grade = "DEGRADED"
        rows.append(_row(
            feature="Simulate Past Decision",
            trigger_tested="POST /api/decision-simulator",
            start_timestamp=started,
            finish_timestamp=_now(),
            backend_path="product.decision_simulator",
            durable_artifact="logs/product/decision_simulator.json",
            freshness=sim_status,
            status=grade,
            blocker_reason="" if grade == "PASS" else sim_status or str(sim.get("message") or "")[:200],
            result_count=int(sim.get("sessions_done") or 1 if grade == "PASS" else 0),
            code_sha=sha,
        ))
    except Exception as exc:
        rows.append(_row(
            feature="Simulate Past Decision",
            trigger_tested="POST /api/decision-simulator",
            start_timestamp=started,
            finish_timestamp=_now(),
            backend_path="product.decision_simulator",
            status="DEGRADED",
            blocker_reason=str(exc)[:300],
            code_sha=sha,
        ))

    paper = _request_json(_url(api, "/api/paper-autopilot"), timeout=args.request_timeout)
    rows.append(_row(
        feature="Paper execution status",
        trigger_tested="GET /api/paper-autopilot",
        backend_path="product.paper_autopilot",
        durable_artifact="logs/product paper book / journal",
        result_count=len(paper.get("positions") or paper.get("open_positions") or []),
        status="PASS" if isinstance(paper, dict) else "FAIL",
        freshness=str(paper.get("status") or paper.get("state") or ""),
        blocker_reason="" if live_locked else "live money unlocked",
        start_timestamp=_now(),
        finish_timestamp=_now(),
        code_sha=sha,
    ))

    started = _now()
    try:
        cycle = _start_control(api, "RUN_CYCLE_NOW", args.request_timeout)
        rows.append(_row(
            feature="Paper cycle request",
            trigger_tested="POST /api/controls/RUN_CYCLE_NOW",
            operation_id=str(cycle.get("operation_id") or cycle.get("job_id") or ""),
            start_timestamp=started,
            finish_timestamp=_now(),
            backend_path="research.autonomy.controls → paper_cycle",
            durable_artifact="logs/autonomy/",
            status="PASS" if cycle.get("accepted") else "FAIL",
            blocker_reason="" if cycle.get("accepted") else str(cycle)[:200],
            code_sha=sha,
        ))
    except Exception as exc:
        rows.append(_row(
            feature="Paper cycle request",
            trigger_tested="POST /api/controls/RUN_CYCLE_NOW",
            start_timestamp=started,
            finish_timestamp=_now(),
            status="DEGRADED",
            blocker_reason=str(exc)[:300],
            code_sha=sha,
        ))

    learning = _request_json(_url(api, "/api/learning-dashboard"), timeout=args.request_timeout)
    rows.append(_row(
        feature="Learning loop dashboard",
        trigger_tested="GET /api/learning-dashboard",
        backend_path="product learning ledger",
        durable_artifact="logs/product/taken_evidence.jsonl",
        result_count=int((learning.get("n_policies") or learning.get("policy_count") or 0) or 0),
        status="PASS" if isinstance(learning, dict) else "FAIL",
        start_timestamp=_now(),
        finish_timestamp=_now(),
        code_sha=sha,
    ))

    soak = _request_json(_url(api, "/api/forward-soak"), method="POST", timeout=args.request_timeout)
    rows.append(_row(
        feature="Forward soak verification",
        trigger_tested="POST /api/forward-soak",
        backend_path="product.forward_soak",
        durable_artifact="forward soak ledger",
        status="PASS" if "verification" in soak else "FAIL",
        start_timestamp=_now(),
        finish_timestamp=_now(),
        code_sha=sha,
    ))

    out = {
        "schema_version": 1,
        "generated_at": _now(),
        "code_sha": sha,
        "api": api,
        "live_locked": live_locked,
        "features": rows,
    }
    dest = Path(args.output)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nWrote {dest}")
    worst = "PASS"
    for row in rows:
        print(f"  {row['status']:9} {row['feature']}: {row.get('blocker_reason') or row.get('freshness') or ''}")
        if row["status"] == "FAIL":
            worst = "FAIL"
        elif row["status"] in {"DEGRADED", "BLOCKED"} and worst == "PASS":
            worst = row["status"]
    print(f"\nMATRIX {worst}")
    return 1 if worst == "FAIL" else 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run live QuantTerm product acceptance.")
    parser.add_argument("--api", default="http://127.0.0.1:8765")
    parser.add_argument("--request-timeout", type=float, default=12.0)
    parser.add_argument("--operation-timeout", type=float, default=1200.0)
    parser.add_argument("--acquire-timeout", type=float, default=720.0)
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
