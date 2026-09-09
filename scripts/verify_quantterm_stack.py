#!/usr/bin/env python3
"""Read-only end-to-end verifier for the locally running QuantTerm desk.

This is deliberately stricter than "the ports are open". It verifies the UI,
report service and the canonical API surfaces that power the visible product,
then performs a small status-endpoint soak and checks that API/worker file
descriptors stay essentially flat.

It never creates an order, never unlocks live money and never invents data.
Empty recommendation/report results are valid; missing routes, malformed
contracts, timeouts and resource exhaustion are not.
"""
from __future__ import annotations

import argparse
import json
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Any, Iterable


DEFAULT_API = "http://127.0.0.1:8765"
DEFAULT_UI = "http://127.0.0.1:5173"
DEFAULT_REPORTS = "http://127.0.0.1:8766"
SCANNER_MODES = ("Momentum", "Conviction", "Breakouts", "Pre-Breakout", "Long-Term", "F&O", "Avoid")


@dataclass
class Probe:
    name: str
    ok: bool
    detail: str
    elapsed_ms: int
    payload: dict[str, Any] | None = None


def _get(url: str, timeout: float) -> tuple[int, bytes]:
    request = urllib.request.Request(
        url,
        method="GET",
        headers={"Accept": "application/json,text/html;q=0.9,*/*;q=0.8"},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return int(response.status), response.read()


def probe_text(name: str, url: str, timeout: float) -> Probe:
    started = time.monotonic()
    try:
        status, body = _get(url, timeout)
        elapsed = int((time.monotonic() - started) * 1000)
        ok = status == 200 and bool(body)
        return Probe(name, ok, f"HTTP {status} · {len(body):,} bytes", elapsed)
    except Exception as exc:
        elapsed = int((time.monotonic() - started) * 1000)
        return Probe(name, False, f"{type(exc).__name__}: {exc}", elapsed)


def probe_json(
    name: str,
    url: str,
    timeout: float,
    required_keys: Iterable[str] = (),
) -> Probe:
    started = time.monotonic()
    try:
        status, raw = _get(url, timeout)
        elapsed = int((time.monotonic() - started) * 1000)
        payload = json.loads(raw.decode("utf-8"))
        if not isinstance(payload, dict):
            return Probe(name, False, "response is not a JSON object", elapsed)
        missing = [key for key in required_keys if key not in payload]
        if status != 200:
            return Probe(name, False, f"HTTP {status}", elapsed, payload)
        if missing:
            return Probe(name, False, f"missing keys: {', '.join(missing)}", elapsed, payload)
        return Probe(name, True, "contract present", elapsed, payload)
    except urllib.error.HTTPError as exc:
        elapsed = int((time.monotonic() - started) * 1000)
        try:
            body = exc.read().decode("utf-8", errors="replace")[:240]
        except Exception:
            body = ""
        return Probe(name, False, f"HTTP {exc.code}: {body}", elapsed)
    except Exception as exc:
        elapsed = int((time.monotonic() - started) * 1000)
        return Probe(name, False, f"{type(exc).__name__}: {exc}", elapsed)


def _join(base: str, path: str) -> str:
    return base.rstrip("/") + path


def _fd_counts(health: dict[str, Any] | None) -> tuple[int | None, int | None, str]:
    resources = dict((health or {}).get("resources") or {})
    api = dict(resources.get("api") or {})
    ops = dict(resources.get("market_ops") or {})

    def integer(value: Any) -> int | None:
        try:
            return int(value) if value is not None else None
        except (TypeError, ValueError):
            return None

    return (
        integer(api.get("fd_count")),
        integer(ops.get("fd_count")),
        str(resources.get("state") or "UNKNOWN"),
    )


def _pick_symbols(dashboard: dict[str, Any], explicit: str = "") -> list[str]:
    out: list[str] = []
    if explicit.strip():
        out.append(explicit.strip().upper())
    for section in ("scan", "long_term"):
        rows = ((dashboard.get(section) or {}).get("records") or []) if isinstance(dashboard.get(section), dict) else []
        for row in rows:
            if not isinstance(row, dict):
                continue
            symbol = str(row.get("symbol") or "").strip().upper()
            if symbol and symbol not in out:
                out.append(symbol)
            if len(out) >= 2:
                return out
    return out


def _pick_symbol(dashboard: dict[str, Any], explicit: str) -> str:
    symbols = _pick_symbols(dashboard, explicit)
    return symbols[0] if symbols else ""


def _print(probe: Probe) -> None:
    mark = "PASS" if probe.ok else "FAIL"
    print(f"[{mark}] {probe.name:<32} {probe.elapsed_ms:>5} ms · {probe.detail}")


def run(args: argparse.Namespace) -> int:
    failures: list[str] = []
    probes: list[Probe] = []

    probes.append(probe_text("Frontend", args.ui.rstrip("/") + "/", args.timeout))
    probes.append(probe_json("Report API", _join(args.reports, "/health"), args.timeout))

    # One row per visible/product-critical surface. Market Overview is projected
    # from /api/dashboard and is therefore covered by the Dashboard probe rather
    # than inventing a second endpoint just for the page.
    canonical = [
        ("API health", "/api/health", ("resources",)),
        ("Dashboard / Market overview", "/api/dashboard", ("market", "scan", "long_term", "operations", "data")),
        ("Operations", "/api/operations", ("running",)),
        ("Desk pipeline", "/api/desk-pipeline", ("steps",)),
        ("Product contract", "/api/product-contract", ("wired", "checks")),
        ("Product readiness", "/api/product-readiness", ("state", "score", "lanes")),
        ("Radar home", "/api/radar-home", ("lanes",)),
        ("Recommendations", "/api/recommendations-workspace", ("categories",)),
        ("Market reports", "/api/market-reports-workspace", ("reports",)),
        ("System health", "/api/system-health-contract", ()),
        ("Operator health", "/api/operator-health", ()),
        ("Research status", "/api/research-status", ()),
        ("Strategies", "/api/strategy-catalog", ()),
        ("Learning dashboard", "/api/learning-dashboard", ()),
        ("Paper autopilot", "/api/paper-autopilot", ("live_locked",)),
        ("Why no trade", "/api/why-no-trade", ()),
        ("Decision journal", "/api/decision-journal?limit=1", ()),
        ("Decision simulator", "/api/decision-simulator", ("live_locked",)),
        ("Forward soak", "/api/forward-soak", ()),
        ("Scan audit / Coverage", "/api/scan-audit", ()),
        ("Watchlist", "/api/watchlist", ("items", "count")),
        ("Education", "/api/education?min_impact=0&limit=1", ("cards",)),
        ("Data readiness", "/api/data-readiness", ("ready",)),
        ("News", "/api/news", ("articles",)),
        ("F&O", "/api/fno", ()),
    ]
    by_name: dict[str, Probe] = {}
    for name, path, keys in canonical:
        result = probe_json(name, _join(args.api, path), args.timeout, keys)
        probes.append(result)
        by_name[name] = result

    for mode in SCANNER_MODES:
        encoded = urllib.parse.quote(mode, safe="")
        probes.append(
            probe_json(
                f"Scanner · {mode}",
                _join(args.api, f"/api/scanner-workspace/{encoded}"),
                args.timeout,
                ("mode", "rows"),
            )
        )

    contract = by_name.get("Product contract")
    if contract and contract.ok and contract.payload and contract.payload.get("wired") is not True:
        contract.ok = False
        contract.detail = "wired=false — a primary backend route contract is missing"

    health = by_name.get("API health")
    if health and health.ok and health.payload:
        api_fd, ops_fd, resource_state = _fd_counts(health.payload)
        if resource_state in {"RESOURCE_PRESSURE", "RESOURCE_EXHAUSTED"}:
            health.ok = False
            health.detail = f"resource state {resource_state} · api_fd={api_fd} · market_ops_fd={ops_fd}"
        else:
            health.detail = f"resources {resource_state} · api_fd={api_fd} · market_ops_fd={ops_fd}"

    dashboard_probe = by_name.get("Dashboard / Market overview")
    dashboard = dashboard_probe.payload if dashboard_probe and dashboard_probe.ok and dashboard_probe.payload else {}
    symbols = _pick_symbols(dashboard, args.symbol)
    if symbols:
        symbol = symbols[0]
        encoded = urllib.parse.quote(symbol, safe="")
        symbol_probes = [
            (f"Chart · {symbol}", f"/api/chart/{encoded}", ("symbol", "bars")),
            (f"Stock intelligence · {symbol}", f"/api/stock-intelligence/{encoded}", ()),
            (f"Due diligence · {symbol}", f"/api/due-diligence/{encoded}", ()),
            (f"Trade plan · {symbol}", f"/api/trade-plan/{encoded}", ()),
            (f"Ratios · {symbol}", f"/api/data/ratios/{encoded}", ()),
        ]
        for name, path, keys in symbol_probes:
            probes.append(probe_json(name, _join(args.api, path), args.timeout, keys))
        compare_symbols = symbols[:2]
        if len(compare_symbols) == 2:
            value = urllib.parse.quote(",".join(compare_symbols), safe="")
            probes.append(
                probe_json(
                    "Compare workspace",
                    _join(args.api, f"/api/compare?symbols={value}"),
                    args.timeout,
                    ("rows", "symbols"),
                )
            )
    else:
        print("[INFO] No saved scan symbol available; per-stock probes skipped. Pass --symbol RELIANCE to force them.")

    print("\nQuantTerm full-stack contract")
    print("-" * 82)
    for result in probes:
        _print(result)
        if not result.ok:
            failures.append(result.name)

    if args.soak > 0 and not failures:
        before_probe = probe_json("Soak baseline health", _join(args.api, "/api/health"), args.timeout, ("resources",))
        before_api, before_ops, _ = _fd_counts(before_probe.payload)
        soak_paths = (
            "/api/health",
            "/api/operations",
            "/api/data-readiness",
            "/api/product-contract",
            "/api/radar-home",
        )
        soak_error = ""
        started = time.monotonic()
        for _ in range(args.soak):
            for path in soak_paths:
                result = probe_json("soak", _join(args.api, path), args.timeout)
                if not result.ok:
                    soak_error = f"{path}: {result.detail}"
                    break
            if soak_error:
                break
        after_probe = probe_json("Soak final health", _join(args.api, "/api/health"), args.timeout, ("resources",))
        after_api, after_ops, after_state = _fd_counts(after_probe.payload)
        elapsed = int((time.monotonic() - started) * 1000)
        api_growth = None if before_api is None or after_api is None else after_api - before_api
        ops_growth = None if before_ops is None or after_ops is None else after_ops - before_ops
        stable = (
            not soak_error
            and after_state not in {"RESOURCE_PRESSURE", "RESOURCE_EXHAUSTED"}
            and (api_growth is None or api_growth <= args.fd_growth_limit)
            and (ops_growth is None or ops_growth <= args.fd_growth_limit)
        )
        detail = (
            soak_error
            or f"{args.soak * len(soak_paths)} requests · api_fd {before_api}→{after_api} (Δ{api_growth}) · "
            f"ops_fd {before_ops}→{after_ops} (Δ{ops_growth}) · {after_state}"
        )
        soak_probe = Probe("Descriptor/status soak", stable, detail, elapsed)
        _print(soak_probe)
        if not stable:
            failures.append(soak_probe.name)

    print("-" * 82)
    if failures:
        print("NOT WORKING: " + ", ".join(failures))
        return 1
    print("WORKING: frontend, APIs and all probed visible desk surfaces answered their contracts.")
    if args.soak > 0:
        print("RESOURCE CHECK: repeated status traffic did not show material descriptor growth.")
    print("NOTE: This verifier is read-only. Run verify_quantterm_actions.py for real non-money action proof.")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Verify the running QuantTerm frontend/backend contract.")
    parser.add_argument("--api", default=DEFAULT_API)
    parser.add_argument("--ui", default=DEFAULT_UI)
    parser.add_argument("--reports", default=DEFAULT_REPORTS)
    parser.add_argument("--timeout", type=float, default=5.0)
    parser.add_argument("--soak", type=int, default=12, help="status soak rounds; 0 disables")
    parser.add_argument("--fd-growth-limit", type=int, default=6)
    parser.add_argument("--symbol", default="", help="optional NSE symbol for stock-level probes")
    return parser


if __name__ == "__main__":
    raise SystemExit(run(build_parser().parse_args()))
