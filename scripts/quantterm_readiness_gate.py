#!/usr/bin/env python3
"""Read-only production acceptance probe for QuantTerm's canonical local API.

No POST, no broker request, no background task and no changes to runtime state.
A PASS is a *local evidence* gate, not permission to unlock live money or
proof that a public reverse proxy is secure.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from typing import Any, Mapping
from urllib.error import URLError
from urllib.request import Request, urlopen


def evaluate(
    responses: Mapping[str, Mapping[str, Any]],
    *,
    timings_s: Mapping[str, float],
    public: bool = False,
    max_read_s: float = 3.0,
    min_coverage_pct: float = 95.0,
) -> dict[str, Any]:
    blockers: list[str] = []
    warnings: list[str] = []
    health = responses.get("health") or {}
    access = responses.get("access") or {}
    dashboard = responses.get("dashboard") or {}
    gate = responses.get("decision-simulation-gate") or {}

    if health.get("ok") is not True:
        blockers.append("Canonical API liveness not verified")
    if health.get("live_locked") is not True or health.get("live_lock_verified") is not True:
        blockers.append("Live-money interlock is not positively verified LOCKED")
    if health.get("live_execution_authorized") is not False:
        blockers.append("Live execution is not positively verified unauthorized")
    if public:
        if access.get("public_read_only") is not True:
            blockers.append("Public read-only access is not enabled")
        if access.get("operator_token_configured") is not True:
            blockers.append("Public operator mutation token is not configured")
        if str(access.get("mutation_policy") or "") != "OPERATOR_TOKEN_REQUIRED":
            blockers.append("Public mutation authorization policy is not enforced")
    for endpoint in ("health", "access", "dashboard", "decision-simulation-gate"):
        if endpoint not in responses:
            blockers.append(f"Missing GET /api/{endpoint}")
            continue
        seconds = timings_s.get(endpoint)
        if seconds is None or seconds > max_read_s:
            blockers.append(f"/api/{endpoint} exceeded the {max_read_s:g}s read budget")

    cache = dashboard.get("dashboard_cache") or {}
    if cache.get("status") != "FRESH":
        blockers.append("Dashboard read snapshot is not FRESH")
    if not dashboard.get("generated_at"):
        blockers.append("Dashboard lacks source snapshot timestamp")

    data = dashboard.get("data") or {}
    if data.get("history_current") is not True:
        blockers.append("Official market-session history is not verified current")
    scan = dashboard.get("scan") or {}
    if not scan.get("available") or not scan.get("scanned_at"):
        blockers.append("Completed persisted market scan is not readable")
    provenance = scan.get("provenance") or {}
    if not provenance.get("data_current") or not provenance.get("price_data_as_of"):
        blockers.append("Scanner has no verified current-session price provenance")
    coverage = scan.get("coverage") or {}
    requested = int(scan.get("requested_universe") or 0)
    checked = int(coverage.get("checked") or scan.get("universe_size") or 0)
    if requested <= 0 or checked <= 0 or checked > requested:
        blockers.append("Scan universe coverage counters are unverified")
        coverage_pct = 0.0
    else:
        coverage_pct = 100.0 * checked / requested
        if coverage_pct < min_coverage_pct:
            blockers.append(
                f"Approved NSE universe coverage {coverage_pct:.1f}% below {min_coverage_pct:.1f}%"
            )
    if int(coverage.get("analysis_errors") or 0) > 0:
        blockers.append("Scanner recorded analysis errors; check exact coverage ledger")

    status_cache = gate.get("status_cache") or {}
    if status_cache.get("status") != "FRESH":
        blockers.append("Decision Simulation status read snapshot is not FRESH")
    if gate.get("scan_fresh") is not True:
        blockers.append("Decision Simulation has not independently verified current scan freshness")

    paper = dashboard.get("paper") or {}
    if paper.get("available") is not True:
        warnings.append("Paper-book read unavailable; execution lifecycle not certified")
    if not (paper.get("last_cycle") or {}):
        warnings.append("No paper decision cycle observed in dashboard")

    if dashboard.get("long_term", {}).get("available"):
        lt = dashboard["long_term"]
        lt_summary = lt.get("summary") or {}
        if int(lt_summary.get("fundamentals_covered") or 0) == 0:
            warnings.append("Long-term fundamentals coverage requires separate validation")

    return {
        "verdict": "HOLD" if blockers else "LOCAL_READ_GATE_PASS",
        "blockers": blockers,
        "warnings": warnings,
        "metrics": {
            "scan_coverage_pct": round(coverage_pct, 2),
            "endpoint_seconds": {
                name: round(seconds, 3) for name, seconds in timings_s.items()
            },
        },
        "scope": (
            "Read-only local API check. Not a public-network penetration test, "
            "broker execution approval, forward-trade evidence audit, or "
            "authorization to enable live trading."
        ),
    }


def fetch_json(origin: str, endpoint: str, timeout_s: float) -> tuple[dict, float]:
    url = f"{origin.rstrip('/')}/api/{endpoint}"
    started = time.monotonic()
    request = Request(url, headers={"Accept": "application/json"}, method="GET")
    with urlopen(request, timeout=timeout_s) as response:
        if response.status != 200:
            raise ValueError(f"/api/{endpoint} returned HTTP {response.status}")
        raw = response.read(6 * 1024 * 1024 + 1)
        if len(raw) > 6 * 1024 * 1024:
            raise ValueError(f"/api/{endpoint} exceeded safe JSON response size")
    result = json.loads(raw)
    if not isinstance(result, dict):
        raise ValueError(f"/api/{endpoint} returned non-object JSON")
    return result, time.monotonic() - started


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--origin", default="http://127.0.0.1:8765")
    parser.add_argument("--public", action="store_true", help="Require public read-only operator security")
    parser.add_argument("--max-read-seconds", type=float, default=3.0)
    parser.add_argument("--min-coverage-pct", type=float, default=95.0)
    args = parser.parse_args(argv)
    origin = args.origin.rstrip("/")
    if not origin.startswith(("http://127.0.0.1:", "http://localhost:")):
        parser.error("Only local loopback URLs are supported by this safety probe")
    responses: dict[str, dict] = {}
    timings: dict[str, float] = {}
    failed: list[str] = []
    for path in ("health", "access", "dashboard", "decision-simulation-gate"):
        try:
            result, elapsed = fetch_json(origin, path, max(0.5, args.max_read_seconds))
            responses[path] = result
            timings[path] = elapsed
        except (URLError, OSError, ValueError, json.JSONDecodeError) as exc:
            failed.append(f"/api/{path}: {type(exc).__name__}")
    report = evaluate(
        responses, timings_s=timings, public=args.public,
        max_read_s=args.max_read_seconds, min_coverage_pct=args.min_coverage_pct,
    )
    if failed:
        report["blockers"].extend(failed)
        report["verdict"] = "HOLD"
    print(json.dumps(report, indent=2))
    return 0 if report["verdict"] == "LOCAL_READ_GATE_PASS" else 2


if __name__ == "__main__":
    sys.exit(main())
