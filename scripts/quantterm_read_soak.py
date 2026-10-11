#!/usr/bin/env python3
"""Read-only latency/freshness soak for QuantTerm's local desk API.

Run ONLY after deploying a reviewed exact SHA to the Mac. No broker calls,
orders, POSTs, database mutations, or service restarts; HTTP GET only.
This is one narrow latency gate, not approval for public/live-money trading.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections import Counter
from typing import Any, Mapping
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import HTTPRedirectHandler, ProxyHandler, Request, build_opener

ENDPOINTS = ("health", "access", "dashboard", "decision-simulation-gate")
MAX_RESPONSE_BYTES = 6 * 1024 * 1024


def percentile_nearest_rank(values: list[float], pct: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    return round(ordered[max(0, math.ceil(len(ordered) * pct / 100) - 1)], 4)


def assess_samples(
    samples: list[Mapping[str, Any]], *, min_samples: int = 20,
    max_p95_s: float = 3.0,
) -> dict[str, Any]:
    """No payloads/secrets in reports; observations are request metadata."""
    blockers: list[str] = []
    endpoints: dict[str, Any] = {}
    if len(samples) < min_samples:
        blockers.append(f"Only {len(samples)} cycles; require {min_samples}")
    for name in ENDPOINTS:
        rows = [row.get(name) if isinstance(row, dict) else None for row in samples]
        latency = [
            float(x["seconds"]) for x in rows
            if isinstance(x, dict) and isinstance(x.get("seconds"), (int, float))
            and math.isfinite(float(x["seconds"]))
        ]
        valid = [
            x for x in rows if isinstance(x, dict)
            and x.get("ok") is True
        ]
        p95 = percentile_nearest_rank(latency, 95)
        states = dict(Counter(
            str(x.get("cache_status") or "N/A")
            for x in rows if isinstance(x, dict)
        ))
        endpoints[name] = {
            "samples": len(rows),
            "successes": len(valid),
            "p95_seconds": p95,
            "max_seconds": round(max(latency), 4) if latency else None,
            "cache_status_counts": states if name in (
                "dashboard", "decision-simulation-gate",
            ) else {},
        }
        if len(valid) != len(samples):
            blockers.append(f"/api/{name} had {len(samples) - len(valid)} failed/unsafe reads")
        if p95 is None or p95 > max_p95_s:
            blockers.append(f"/api/{name} p95 exceeds {max_p95_s:g}s")
        if name in ("dashboard", "decision-simulation-gate") and (
            states.get("FRESH", 0) != len(samples)
        ):
            blockers.append(f"/api/{name} did not serve FRESH status for every sample")
    return {
        "verdict": "HOLD" if blockers else "READ_SOAK_PASS",
        "sample_cycles": len(samples),
        "minimum_cycles": min_samples,
        "p95_budget_seconds": max_p95_s,
        "endpoints": endpoints,
        "blockers": blockers,
        "scope": "GET-only local latency. No external security/market data/paper execution approval.",
    }


def _safe_origin(raw: str) -> str:
    parsed = urlsplit(raw)
    if parsed.scheme != "http" or parsed.hostname not in ("127.0.0.1", "localhost", "::1"):
        raise ValueError("Only local loopback HTTP origins are supported")
    if parsed.username or parsed.password or parsed.path not in ("", "/") or parsed.query or parsed.fragment:
        raise ValueError("Only a simple local loopback origin is supported")
    if parsed.port is None:
        raise ValueError("Specify the local API port explicitly")
    return raw.rstrip("/")


def access_policy_safe(access: Mapping[str, Any]) -> bool:
    """Recognize the three explicit deployment modes, never unknown access."""
    policy = str(access.get("mutation_policy") or "")
    public = access.get("public_read_only")
    token = access.get("operator_token_configured") is True
    if policy == "READ_ONLY":
        return public is True and not token
    if policy == "OPERATOR_TOKEN_REQUIRED":
        return public is True and token
    # This GET-only soak runs against loopback and must not make a private
    # owner-operated Mac report insecure merely because public mode is off.
    # This DOES NOT establish internet-facing security or permit proxying it.
    if policy == "PRIVATE_OPERATOR":
        return public is False
    return False


class _RejectRedirects(HTTPRedirectHandler):
    """Never let a local test be redirected to an external endpoint."""

    def redirect_request(self, request, fp, code, msg, headers, newurl):
        return None


# Ignore HTTP_PROXY/HTTPS_PROXY even if the host machine has them set.
# A local-only test must never send API reads or readiness state to a proxy.
_LOCAL_OPENER = build_opener(ProxyHandler({}), _RejectRedirects())


def sample_one(origin: str, path: str, timeout_s: float) -> dict[str, Any]:
    started = time.monotonic()
    try:
        request = Request(
            origin + "/api/" + path,
            headers={"Accept": "application/json"},
            method="GET",
        )
        with _LOCAL_OPENER.open(request, timeout=timeout_s) as response:
            if response.status != 200:
                raise ValueError("Unexpected HTTP response")
            raw = response.read(MAX_RESPONSE_BYTES + 1)
            if len(raw) > MAX_RESPONSE_BYTES:
                raise ValueError("API JSON response exceeded size bound")
        payload = json.loads(raw)
        if not isinstance(payload, dict):
            raise ValueError("API returned a non-object JSON payload")
        safe = bool(payload.get("ok") is True) if path == "health" else True
        if path == "health":
            safe = safe and payload.get("live_locked") is True
            safe = safe and payload.get("live_lock_verified") is True
            safe = safe and payload.get("live_execution_authorized") is False
        elif path == "access":
            safe = access_policy_safe(payload)
        elif path == "dashboard":
            safe = (payload.get("dashboard_cache") or {}).get("status") == "FRESH"
        else:
            safe = (payload.get("status_cache") or {}).get("status") == "FRESH"
        cache_key = "dashboard_cache" if path == "dashboard" else "status_cache"
        return {
            "ok": safe,
            "seconds": round(time.monotonic() - started, 4),
            "cache_status": str((payload.get(cache_key) or {}).get("status") or "")
                if path in ("dashboard", "decision-simulation-gate") else "",
        }
    except (HTTPError, URLError, OSError, ValueError, json.JSONDecodeError) as exc:
        return {
            "ok": False,
            "seconds": round(time.monotonic() - started, 4),
            "error_type": type(exc).__name__,
        }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--origin", default="http://127.0.0.1:8765")
    parser.add_argument("--samples", type=int, default=20)
    parser.add_argument("--interval-seconds", type=float, default=1.0)
    parser.add_argument("--timeout-seconds", type=float, default=3.0)
    parser.add_argument("--p95-budget-seconds", type=float, default=3.0)
    args = parser.parse_args(argv)
    try:
        origin = _safe_origin(args.origin)
    except ValueError as exc:
        parser.error(str(exc))
    if not 1 <= args.samples <= 500:
        parser.error("--samples must be between 1 and 500")
    if not 0 <= args.interval_seconds <= 60:
        parser.error("--interval-seconds must be between 0 and 60")
    if not 0.5 <= args.timeout_seconds <= 120:
        parser.error("--timeout-seconds must be between 0.5 and 120")
    results: list[dict[str, Any]] = []
    for index in range(args.samples):
        results.append({
            endpoint: sample_one(origin, endpoint, args.timeout_seconds)
            for endpoint in ENDPOINTS
        })
        if index + 1 < args.samples:
            time.sleep(args.interval_seconds)
    result = assess_samples(
        results,
        min_samples=args.samples,
        max_p95_s=args.p95_budget_seconds,
    )
    print(json.dumps(result, indent=2))
    return 0 if result["verdict"] == "READ_SOAK_PASS" else 2


if __name__ == "__main__":
    sys.exit(main())
