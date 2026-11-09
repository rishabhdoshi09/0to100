#!/usr/bin/env python3
"""Read-only, loopback-only QuantTerm host release evidence probe.

This is a measurement aid, not a substitute for real detach/restore/soak or
browser-level release validation. It never calls a state-changing HTTP method.
"""
from __future__ import annotations

import argparse
import json
import math
import platform
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlsplit
from urllib.request import HTTPRedirectHandler, ProxyHandler, Request, build_opener


MAX_RESPONSE_BYTES = 8 * 1024 * 1024
SAFE_HOSTS = frozenset({"localhost", "127.0.0.1", "::1"})
ENDPOINTS = ("/api/health", "/api/dashboard")
FULL_SHA = re.compile(r"[0-9a-f]{40}\Z")


def _origin(raw: str) -> str:
    """Do not accidentally send host diagnostics or credentials to an outside URL."""
    parsed = urlsplit(str(raw or ""))
    if (parsed.scheme != "http" or parsed.hostname not in SAFE_HOSTS
            or parsed.username or parsed.password or parsed.query
            or parsed.fragment or parsed.path not in {"", "/"}):
        raise ValueError("API must be a plain http:// loopback origin with no credentials/path")
    if not parsed.port:
        raise ValueError("API port must be explicit")
    return raw.rstrip("/")


class _NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def _fetch_json(url: str, timeout: float) -> tuple[float, dict]:
    request = Request(url, method="GET", headers={"Accept": "application/json"})
    started = time.perf_counter()
    # Disable environment proxies and HTTP redirects: a local endpoint must
    # never cause this probe to contact an arbitrary remote destination.
    opener = build_opener(ProxyHandler({}), _NoRedirect())
    with opener.open(request, timeout=timeout) as response:
        if response.status != 200:
            raise RuntimeError("non-200 HTTP response")
        raw = response.read(MAX_RESPONSE_BYTES + 1)
    elapsed_ms = (time.perf_counter() - started) * 1000
    if len(raw) > MAX_RESPONSE_BYTES:
        raise ValueError("API JSON response exceeded the bounded measurement budget")
    payload = json.loads(raw.decode("utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("API response is not an object")
    return elapsed_ms, payload


def _quantile_ms(samples: list[float], percent: float) -> float | None:
    """Use a documented linear interpolation, never claim unsampled latency."""
    if not samples:
        return None
    ordered = sorted(samples)
    where = (len(ordered) - 1) * percent
    lo = math.floor(where)
    hi = math.ceil(where)
    value = ordered[lo] + (ordered[hi] - ordered[lo]) * (where - lo)
    return round(value, 2)


def _summary(latencies: list[float], failures: list[str]) -> dict:
    return {
        "successes": len(latencies),
        "failures": len(failures),
        "failure_types": failures[:8],
        "p50_ms": _quantile_ms(latencies, 0.50),
        "p95_ms": _quantile_ms(latencies, 0.95),
        "p99_ms": _quantile_ms(latencies, 0.99),
        "worst_ms": round(max(latencies), 2) if latencies else None,
        "note": "Sample interpolation only, not a statistically certified percentile.",
    }


def _host_snapshot() -> dict:
    result = {"platform": platform.platform(), "swap": "UNAVAILABLE"}
    if sys.platform == "darwin":
        try:
            output = subprocess.run(
                ["sysctl", "vm.swapusage"], check=True, capture_output=True,
                text=True, timeout=3,
            )
            result["swap"] = output.stdout.strip()[:400]
        except (OSError, subprocess.SubprocessError):
            pass
    return result



def _verify_release_identity(
    expected_sha: str, checkout_sha: str, host_report: dict,
) -> tuple[bool, dict]:
    """Require checkout, deployed manifest and running supervisor to be identical.

    Unlike a green remote CI run, the evidence must come from the actual local
    service. A missing or unhealthy host never passes as a release identity.
    """
    if FULL_SHA.fullmatch(str(expected_sha or "")) is None:
        raise ValueError("expected SHA must be a full 40-character lowercase git hash")
    host = host_report if isinstance(host_report, dict) else {}
    deployment = host.get("deployment") or {}
    supervisor = host.get("supervisor") or {}
    service = host.get("service") or {}
    live = host.get("live_execution") or {}
    checks = {
        "checkout_matches": checkout_sha == expected_sha,
        "deployment_matches": deployment.get("build_sha") == expected_sha,
        "supervisor_matches": supervisor.get("production_sha") == expected_sha,
        "supervisor_running": supervisor.get("state") == "RUNNING",
        "heartbeat_fresh": host.get("heartbeat_fresh") is True,
        "service_launchd_running": (
            service.get("manager") == "launchd"
            and service.get("returncode") == 0
        ),
        "children_alive": host.get("all_children_alive") is True,
        "children_healthy": host.get("all_children_healthy") is True,
        "live_money_locked": (
            live.get("safe") is True
            and live.get("verified") is True
            and live.get("locked") is True
            and live.get("authorized") is False
        ),
        "host_healthy": host.get("healthy") is True,
    }
    verified = all(checks.values())
    return verified, {
        "result": "VERIFIED" if verified else "BLOCKED",
        "expected_sha": expected_sha,
        "checkout_sha": checkout_sha or "UNKNOWN",
        "deployed_sha": str(deployment.get("build_sha") or "UNKNOWN"),
        "running_sha": str(supervisor.get("production_sha") or "UNKNOWN"),
        "checks": checks,
        "release_certified": False,
    }


def _host_identity(expected_sha: str) -> tuple[bool, dict]:
    """Read the host install's existing status; do not restart or mutate it."""
    try:
        checkout = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parents[1],
            check=True, capture_output=True, text=True, timeout=8,
        ).stdout.strip()
        from product.host_install import host_status
        return _verify_release_identity(
            expected_sha, checkout, host_status(manager="launchd"),
        )
    except Exception as exc:
        return False, {
            "result": "BLOCKED",
            "reason": "host-identity-unavailable:" + type(exc).__name__,
            "release_certified": False,
        }


def collect(
    origin: str,
    *,
    samples: int,
    timeout: float,
    pause: float,
    fetch=_fetch_json,
    sleeper=time.sleep,
    expected_sha: str | None = None,
    identity_probe=_host_identity,
) -> tuple[int, dict]:
    """Verify live-money lock BEFORE timing the read-only endpoints."""
    root = _origin(origin)
    report = {
        "schema_version": 1,
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "origin": root,
        "samples_requested_per_endpoint": samples,
        "live_execution_gate": "BLOCKED_UNVERIFIED",
        "endpoints": {},
        "release_certified": False,
    }
    if expected_sha is not None:
        if FULL_SHA.fullmatch(expected_sha) is None:
            raise ValueError("expected SHA must be a full 40-character lowercase git hash")
        verified, identity = identity_probe(expected_sha)
        report["host_identity_before"] = identity
        if not verified:
            report["reason"] = "release-sha-host-identity-not-verified"
            return 2, report
    try:
        _, health = fetch(root + "/api/health", timeout)
    except Exception as exc:
        report["reason"] = "health-check-unavailable:" + type(exc).__name__
        return 2, report
    if not (
        health.get("live_lock_verified") is True
        and health.get("live_locked") is True
        and health.get("live_execution_authorized") is False
    ):
        report["reason"] = "live-money-lock-not-verified"
        return 2, report
    report["live_execution_gate"] = "LOCKED_VERIFIED"

    status = 0
    for path in ENDPOINTS:
        times: list[float] = []
        failures: list[str] = []
        for i in range(samples):
            if i and pause:
                sleeper(pause)
            try:
                elapsed, _ = fetch(root + path, timeout)
                if not math.isfinite(elapsed) or elapsed < 0:
                    raise ValueError("invalid timing")
                times.append(elapsed)
            except Exception as exc:
                failures.append(type(exc).__name__)
        report["endpoints"][path] = _summary(times, failures)
        if failures:
            status = 1
    if expected_sha is not None:
        verified, identity = identity_probe(expected_sha)
        report["host_identity_after"] = identity
        if not verified:
            report["reason"] = "release-sha-host-identity-changed-during-probe"
            status = 2
    report["host"] = _host_snapshot()
    report["result"] = (
        "BLOCKED_SHA_IDENTITY" if status == 2 else
        "MEASURED_WITH_FAILURES" if status else "MEASURED_ONLY"
    )
    report["release_certified"] = False
    report["limitations"] = (
        "GET-only, one process, serial samples. No disk detach, recovery, "
        "backup restore, rollback, browser E2E, soak or production authorization."
    )
    return status, report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--api", default="http://127.0.0.1:8765")
    parser.add_argument("--samples", type=int, default=8)
    parser.add_argument("--timeout", type=float, default=10.0)
    parser.add_argument("--pause", type=float, default=0.25)
    parser.add_argument(
        "--expected-sha", help="full release SHA; verify host checkout, service and supervisor before/after sampling",
    )
    args = parser.parse_args(argv)
    if not (1 <= args.samples <= 100 and 0 < args.timeout <= 120
            and 0 <= args.pause <= 5):
        parser.error("samples 1..100; timeout (0,120]; pause [0,5]")
    try:
        status, report = collect(
            args.api, samples=args.samples, timeout=args.timeout, pause=args.pause,
            expected_sha=args.expected_sha,
        )
    except ValueError as exc:
        parser.error(str(exc))
    print(json.dumps(report, indent=2, sort_keys=True))
    return status


if __name__ == "__main__":
    raise SystemExit(main())
