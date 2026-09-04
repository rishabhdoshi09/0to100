#!/usr/bin/env python3
"""Exercise QuantTerm's real safe product actions against a running local stack.

Unlike verify_quantterm_stack.py, this script intentionally queues real research
operations and waits for their durable operation records to finish.  It never
submits a broker order, never unlocks live money, and never bypasses risk gates.

Default sequence:
  RUN_SCAN_NOW -> REFRESH_LONG_TERM_NOW -> REFRESH_MARKET_REPORT_NOW

Optionally, --symbol SYMBOL also exercises the durable due-diligence acquire for
that stock after the shared desk pipeline has finished.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Any


TERMINAL = {"SUCCEEDED", "FAILED", "BLOCKED", "CANCELLED"}
SAFE_CONTROLS = (
    ("Market scan", "RUN_SCAN_NOW"),
    ("Long-term fundamentals", "REFRESH_LONG_TERM_NOW"),
    ("Market report", "REFRESH_MARKET_REPORT_NOW"),
)


def _request_json(url: str, *, method: str = "GET", timeout: float = 10.0) -> dict[str, Any]:
    request = urllib.request.Request(
        url,
        method=method,
        headers={"Accept": "application/json", "Content-Type": "application/json"},
        data=b"" if method != "GET" else None,
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            raw = response.read()
            payload = json.loads(raw.decode("utf-8")) if raw else {}
            if not isinstance(payload, dict):
                raise RuntimeError(f"{url} returned non-object JSON")
            return payload
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")[:500]
        raise RuntimeError(f"HTTP {exc.code} {url}: {body}") from exc


def _url(base: str, path: str) -> str:
    return base.rstrip("/") + path


def _fd(health: dict[str, Any], lane: str) -> int | None:
    try:
        value = ((health.get("resources") or {}).get(lane) or {}).get("fd_count")
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _resource_state(health: dict[str, Any]) -> str:
    return str((health.get("resources") or {}).get("state") or "UNKNOWN")


def _start_control(api: str, control: str, timeout: float) -> dict[str, Any]:
    payload = _request_json(_url(api, f"/api/controls/{urllib.parse.quote(control, safe='')}"), method="POST", timeout=timeout)
    if payload.get("accepted") is not True:
        raise RuntimeError(f"{control} was not accepted: {payload}")
    operation_id = str(payload.get("operation_id") or "")
    if not operation_id:
        raise RuntimeError(f"{control} was accepted without an operation_id")
    return payload


def _wait_operation(api: str, operation_id: str, *, timeout: float, request_timeout: float) -> dict[str, Any]:
    deadline = time.monotonic() + timeout
    last_line = ""
    while time.monotonic() < deadline:
        payload = _request_json(
            _url(api, f"/api/operations/{urllib.parse.quote(operation_id, safe='')}"),
            timeout=request_timeout,
        )
        status = str(payload.get("status") or "UNKNOWN").upper()
        stage = str(payload.get("stage") or "")
        current = payload.get("progress_current")
        total = payload.get("progress_total")
        message = str(payload.get("message") or "")
        line = f"{status} · {stage or 'working'}"
        if current is not None or total is not None:
            line += f" · {current or 0}/{total or '?'}"
        if message:
            line += f" · {message[:120]}"
        if line != last_line:
            print(f"    {line}")
            last_line = line
        if status in TERMINAL:
            return payload
        time.sleep(1.25)
    raise TimeoutError(f"operation {operation_id} did not finish within {int(timeout)} seconds")


def _assert_succeeded(label: str, operation: dict[str, Any]) -> None:
    status = str(operation.get("status") or "UNKNOWN").upper()
    if status != "SUCCEEDED":
        code = str(operation.get("error_code") or "")
        detail = str(operation.get("error_message") or operation.get("message") or "")
        raise RuntimeError(f"{label} ended {status}{f' [{code}]' if code else ''}: {detail}")


def _verify_outputs(api: str, timeout: float) -> None:
    dashboard = _request_json(_url(api, "/api/dashboard"), timeout=timeout)
    scan = dict(dashboard.get("scan") or {})
    data = dict(dashboard.get("data") or {})
    if not scan.get("scanned_at") and not scan.get("records"):
        raise RuntimeError("Market scan reported success but dashboard has no saved scan artifact")
    if int(scan.get("universe_size") or 0) <= 0:
        raise RuntimeError("Market scan reported success but universe_size is zero")
    if not data.get("scan_saved") and not scan.get("scanned_at"):
        raise RuntimeError("Dashboard does not acknowledge a saved scan")

    reports = _request_json(_url(api, "/api/market-reports-workspace"), timeout=timeout)
    if "reports" not in reports:
        raise RuntimeError("Market Reports workspace contract is malformed after report refresh")

    recommendations = _request_json(_url(api, "/api/recommendations-workspace"), timeout=timeout)
    if not isinstance(recommendations.get("categories"), list):
        raise RuntimeError("Recommendations workspace contract is malformed after scan")


def _acquire_symbol(api: str, symbol: str, *, op_timeout: float, request_timeout: float) -> None:
    clean = symbol.strip().upper()
    encoded = urllib.parse.quote(clean, safe="")
    print(f"\n[RUN] Due diligence acquire · {clean}")
    payload = _request_json(
        _url(api, f"/api/due-diligence/{encoded}/acquire?mode=missing_or_stale&async_job=true"),
        method="POST",
        timeout=request_timeout,
    )
    if payload.get("report"):
        print("    SUCCEEDED · backend returned refreshed report directly")
    else:
        operation_id = str(payload.get("operation_id") or "")
        if not operation_id:
            raise RuntimeError("due-diligence acquire returned neither report nor operation_id")
        operation = _wait_operation(api, operation_id, timeout=op_timeout, request_timeout=request_timeout)
        _assert_succeeded(f"Due diligence {clean}", operation)
    report = _request_json(_url(api, f"/api/due-diligence/{encoded}"), timeout=request_timeout)
    if not report:
        raise RuntimeError(f"Due diligence {clean} is empty after successful acquire")
    intelligence = _request_json(_url(api, f"/api/stock-intelligence/{encoded}"), timeout=request_timeout)
    if not intelligence:
        raise RuntimeError(f"Stock Intelligence {clean} is empty after successful acquire")
    print(f"[PASS] Due diligence + Stock Intelligence · {clean}")


def run(args: argparse.Namespace) -> int:
    print("QuantTerm real safe-action verification")
    print("Live-money mutation: DISABLED by design\n")

    before = _request_json(_url(args.api, "/api/health"), timeout=args.request_timeout)
    state = _resource_state(before)
    if state in {"RESOURCE_PRESSURE", "RESOURCE_EXHAUSTED"}:
        raise RuntimeError(f"Refusing integration run while API resources are {state}")
    before_api = _fd(before, "api")
    before_ops = _fd(before, "market_ops")

    for label, control in SAFE_CONTROLS:
        print(f"[RUN] {label} · {control}")
        started = _start_control(args.api, control, args.request_timeout)
        operation_id = str(started["operation_id"])
        operation = _wait_operation(
            args.api,
            operation_id,
            timeout=args.operation_timeout,
            request_timeout=args.request_timeout,
        )
        _assert_succeeded(label, operation)
        print(f"[PASS] {label} · {operation_id[:8]}\n")

    _verify_outputs(args.api, args.request_timeout)
    print("[PASS] Dashboard, Recommendations and Market Reports reflect persisted outputs")

    if args.symbol:
        _acquire_symbol(
            args.api,
            args.symbol,
            op_timeout=args.acquire_timeout,
            request_timeout=args.request_timeout,
        )

    after = _request_json(_url(args.api, "/api/health"), timeout=args.request_timeout)
    after_state = _resource_state(after)
    after_api = _fd(after, "api")
    after_ops = _fd(after, "market_ops")
    api_growth = None if before_api is None or after_api is None else after_api - before_api
    ops_growth = None if before_ops is None or after_ops is None else after_ops - before_ops
    print(
        f"[CHECK] resources {state}→{after_state} · "
        f"api_fd {before_api}→{after_api} (Δ{api_growth}) · "
        f"market_ops_fd {before_ops}→{after_ops} (Δ{ops_growth})"
    )
    if after_state in {"RESOURCE_PRESSURE", "RESOURCE_EXHAUSTED"}:
        raise RuntimeError(f"Product actions finished but resource state degraded to {after_state}")
    if api_growth is not None and api_growth > args.fd_growth_limit:
        raise RuntimeError(f"API descriptors grew by {api_growth}; limit is {args.fd_growth_limit}")
    if ops_growth is not None and ops_growth > args.fd_growth_limit:
        raise RuntimeError(f"Market-ops descriptors grew by {ops_growth}; limit is {args.fd_growth_limit}")

    print("\nWORKING: real scan, long-term refresh and market-report operations completed end-to-end.")
    if args.symbol:
        print(f"WORKING: {args.symbol.strip().upper()} acquisition and Stock Intelligence also completed end-to-end.")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run real non-money QuantTerm actions end-to-end.")
    parser.add_argument("--api", default="http://127.0.0.1:8765")
    parser.add_argument("--request-timeout", type=float, default=8.0)
    parser.add_argument("--operation-timeout", type=float, default=1200.0, help="seconds per scan/report operation")
    parser.add_argument("--acquire-timeout", type=float, default=420.0)
    parser.add_argument("--fd-growth-limit", type=int, default=8)
    parser.add_argument("--symbol", default="", help="also test real due-diligence acquire for this NSE symbol")
    return parser


if __name__ == "__main__":
    try:
        raise SystemExit(run(build_parser().parse_args()))
    except KeyboardInterrupt:
        print("\nStopped by user.", file=sys.stderr)
        raise SystemExit(130)
    except Exception as exc:
        print(f"\nNOT WORKING: {exc}", file=sys.stderr)
        raise SystemExit(1)
