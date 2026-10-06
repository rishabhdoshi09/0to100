#!/usr/bin/env python3
"""Fail-closed preflight for exposing QuantTerm through a public reverse proxy.

The mutation probes intentionally POST to nonexistent paths. When the public
middleware is active they are rejected with 403 before routing; when it is not
active they fall through to 404. No real QuantTerm operation is triggered.
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from typing import Any
from urllib.error import HTTPError
from urllib.request import Request, urlopen


@dataclass(frozen=True)
class Probe:
    status: int
    body: dict[str, Any]


def _json_get(url: str, timeout: float = 3.0) -> dict[str, Any]:
    with urlopen(url, timeout=timeout) as response:
        return dict(json.load(response) or {})


def _mutation_probe(url: str, timeout: float = 3.0) -> Probe:
    request = Request(
        url,
        data=b"",
        method="POST",
        headers={"Content-Type": "application/json", "Accept": "application/json"},
    )
    try:
        with urlopen(request, timeout=timeout) as response:
            raw = response.read().decode("utf-8", errors="replace")
            try:
                body = json.loads(raw) if raw else {}
            except json.JSONDecodeError:
                body = {}
            return Probe(int(response.status), dict(body or {}))
    except HTTPError as exc:
        raw = exc.read().decode("utf-8", errors="replace")
        try:
            body = json.loads(raw) if raw else {}
        except json.JSONDecodeError:
            body = {}
        return Probe(int(exc.code), dict(body or {}))


def evaluate(
    main_access: dict[str, Any],
    report_access: dict[str, Any],
    main_health: dict[str, Any],
    main_mutation: Probe,
    report_mutation: Probe,
) -> tuple[bool, list[str]]:
    checks: list[tuple[bool, str]] = [
        (
            main_access.get("public_read_only") is True,
            "Main API reports public_read_only=true",
        ),
        (
            report_access.get("public_read_only") is True,
            "Report API reports public_read_only=true",
        ),
        (
            main_mutation.status == 403
            and main_mutation.body.get("code") in {"PUBLIC_READ_ONLY", "OPERATOR_AUTH_REQUIRED"},
            "Main API rejects unauthenticated mutation before routing",
        ),
        (
            report_mutation.status == 403
            and report_mutation.body.get("code") in {"PUBLIC_READ_ONLY", "OPERATOR_AUTH_REQUIRED"},
            "Report API rejects unauthenticated mutation before routing",
        ),
        (
            main_health.get("live_lock_verified") is True,
            "Canonical live-money interlock is verified",
        ),
        (
            main_health.get("live_locked") is True,
            "Live money is locked",
        ),
        (
            main_health.get("live_execution_authorized") is False,
            "Live execution is not authorized",
        ),
    ]
    lines = [f"{'PASS' if ok else 'FAIL'} · {label}" for ok, label in checks]
    return all(ok for ok, _ in checks), lines


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Verify QuantTerm public launch safety")
    parser.add_argument("--api", default="http://127.0.0.1:8765", help="Local terminal API base URL")
    parser.add_argument("--reports", default="http://127.0.0.1:8766", help="Local report API base URL")
    parser.add_argument("--timeout", type=float, default=3.0)
    args = parser.parse_args(argv)

    api = args.api.rstrip("/")
    reports = args.reports.rstrip("/")
    try:
        main_access = _json_get(f"{api}/api/access", args.timeout)
        report_access = _json_get(f"{reports}/access", args.timeout)
        main_health = _json_get(f"{api}/api/health", args.timeout)
        main_mutation = _mutation_probe(f"{api}/api/__public_launch_probe__", args.timeout)
        report_mutation = _mutation_probe(f"{reports}/__public_launch_probe__", args.timeout)
    except Exception as exc:
        print(f"HOLD · public launch preflight could not complete: {exc}", file=sys.stderr)
        return 2

    ok, lines = evaluate(
        main_access,
        report_access,
        main_health,
        main_mutation,
        report_mutation,
    )
    print("QuantTerm Public Launch Preflight")
    for line in lines:
        print(line)
    print("PUBLIC LAUNCH PASS" if ok else "PUBLIC LAUNCH HOLD")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
