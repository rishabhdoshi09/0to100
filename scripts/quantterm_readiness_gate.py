#!/usr/bin/env python3
"""Read-only production acceptance probe for QuantTerm's canonical local API.

No POST, no broker request, no background task and no changes to runtime state.
A PASS is a *local evidence* gate, not permission to unlock live money or
proof that a public reverse proxy is secure.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from typing import Any, Mapping
from urllib.error import URLError
from urllib.parse import urlsplit
from urllib.request import HTTPRedirectHandler, ProxyHandler, Request, build_opener


def evaluate(
    responses: Mapping[str, Mapping[str, Any]],
    *,
    timings_s: Mapping[str, float],
    public: bool = False,
    max_read_s: float = 3.0,
    min_coverage_pct: float = 95.0,
    expected_sha: str = "",
) -> dict[str, Any]:
    blockers: list[str] = []
    warnings: list[str] = []
    # This command is a release gate, not a benchmark knob: NaN, infinity,
    # relaxed read budgets, or relaxed coverage may not produce a PASS.
    if not math.isfinite(max_read_s) or not 0 < max_read_s <= 3.0:
        blockers.append("Release read budget must be finite, positive and <=3s")
    if not math.isfinite(min_coverage_pct) or not 95.0 <= min_coverage_pct <= 100.0:
        blockers.append("Release coverage target must be finite and >=95%")
    health = responses.get("health") or {}
    access = responses.get("access") or {}
    dashboard = responses.get("dashboard") or {}
    gate = responses.get("decision-simulation-gate") or {}

    if health.get("ok") is not True:
        blockers.append("Canonical API liveness not verified")
    if expected_sha:
        # A healthy old process is NOT evidence that the proposed release is
        # running. Require the full exact 40-digit SHA, not a short prefix.
        actual_sha = str(health.get("build_sha") or "").strip().lower()
        if (health.get("service") != "quantterm-terminal-api"
                or actual_sha != expected_sha.lower()):
            blockers.append("Running QuantTerm API build SHA does not match expected release")
    if health.get("live_locked") is not True or health.get("live_lock_verified") is not True:
        blockers.append("Live-money interlock is not positively verified LOCKED")
    if health.get("live_execution_authorized") is not False:
        blockers.append("Live execution is not positively verified unauthorized")
    if public:
        if access.get("public_read_only") is not True:
            blockers.append("Public read-only access is not enabled")
        policy = str(access.get("mutation_policy") or "")
        token_present = access.get("operator_token_configured") is True
        # Two safe spectator deployments:
        # 1) No token => every public mutation returns 403 (READ_ONLY).
        # 2) Token present => operator-only mutation (OPERATOR_TOKEN_REQUIRED).
        # Never require an operator credential for a zero-mutation kiosk.
        valid_policy = (
            (policy == "READ_ONLY" and not token_present)
            or (policy == "OPERATOR_TOKEN_REQUIRED" and token_present)
        )
        if not valid_policy:
            blockers.append("Public mutation policy/token configuration is inconsistent or unsafe")
    for endpoint in ("health", "access", "dashboard", "decision-simulation-gate"):
        if endpoint not in responses:
            blockers.append(f"Missing GET /api/{endpoint}")
            continue
        seconds = timings_s.get(endpoint)
        if (seconds is None or not isinstance(seconds, (int, float))
                or not math.isfinite(float(seconds)) or seconds < 0
                or seconds > max_read_s):
            blockers.append(f"/api/{endpoint} exceeded or lacked a valid {max_read_s:g}s read budget")

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

    desk_overlays = scan.get("desk_overlays") or {}
    if desk_overlays.get("recommendations") != "saved":
        blockers.append("Recommendation cards were not durably published for the scan")
    if desk_overlays.get("decision_discovery") != "saved":
        blockers.append("Decision discovery was not durably published for the scan")

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
            "build_sha": str(health.get("build_sha") or ""),
            "expected_sha": expected_sha,
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


class _RejectRedirects(HTTPRedirectHandler):
    """A localhost safety check must never follow external redirects."""

    def redirect_request(self, request, fp, code, msg, headers, newurl):
        return None


_LOCAL_OPENER = build_opener(ProxyHandler({}), _RejectRedirects())


def _safe_origin(raw: str) -> str:
    """Only direct local API origins; never proxy, credentials or URL suffix."""
    parsed = urlsplit(raw)
    if parsed.scheme != "http" or parsed.hostname not in ("127.0.0.1", "localhost", "::1"):
        raise ValueError("Only local loopback HTTP origins are supported")
    if (parsed.username or parsed.password or parsed.path not in ("", "/")
            or parsed.query or parsed.fragment or parsed.port is None):
        raise ValueError("Only a plain local HTTP origin with an explicit port is supported")
    return raw.rstrip("/")


def fetch_json(origin: str, endpoint: str, timeout_s: float) -> tuple[dict, float]:
    url = f"{_safe_origin(origin)}/api/{endpoint}"
    started = time.monotonic()
    request = Request(url, headers={"Accept": "application/json"}, method="GET")
    with _LOCAL_OPENER.open(request, timeout=timeout_s) as response:
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
    parser.add_argument("--expected-sha", default="", help="Require the exact full 40-character Git SHA served by the running API")
    parser.add_argument("--max-read-seconds", type=float, default=3.0)
    parser.add_argument("--min-coverage-pct", type=float, default=95.0)
    args = parser.parse_args(argv)
    try:
        origin = _safe_origin(args.origin)
    except ValueError as exc:
        parser.error(str(exc))
    if args.expected_sha and (
        len(args.expected_sha) != 40
        or any(ch not in "0123456789abcdefABCDEF" for ch in args.expected_sha)
    ):
        parser.error("--expected-sha must be a full 40-character hexadecimal commit SHA")
    if not math.isfinite(args.max_read_seconds) or not 0 < args.max_read_seconds <= 3.0:
        parser.error("--max-read-seconds must be finite, positive and at most 3")
    if (not math.isfinite(args.min_coverage_pct)
            or not 95.0 <= args.min_coverage_pct <= 100.0):
        parser.error("--min-coverage-pct must be finite and at least 95")
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
        expected_sha=args.expected_sha,
    )
    if failed:
        report["blockers"].extend(failed)
        report["verdict"] = "HOLD"
    print(json.dumps(report, indent=2))
    return 0 if report["verdict"] == "LOCAL_READ_GATE_PASS" else 2


if __name__ == "__main__":
    sys.exit(main())
