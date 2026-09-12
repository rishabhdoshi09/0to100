"""Is this host actually able to run QuantTerm for paper operation?

One command, one verdict, no green without real-market access:

    python -m product.host_preflight

The rule that shapes everything here is that a preflight which passes on a
machine that cannot reach NSE is worse than no preflight. It would let an
operator install the desk, walk away, and come back to a week of NO_TRADE days
that look like a quiet market and are actually a firewall.

So market access is a REQUIRED check, and the verdict is BLOCKED without it.

The second rule comes from watching this exact failure: when NSE, BSE and
Yahoo all fail at the same tunnel, that is one fact about the machine and not
three provider outages. The report says ENVIRONMENT_EGRESS_BLOCKED once,
rather than listing every provider as independently broken.

Probes are injectable so the hermetic test suite can exercise every verdict
without a socket. Nothing here writes trading state or starts a workflow.
"""
from __future__ import annotations

import json
import os
import shutil
import sqlite3
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from core.runtime_paths import ENV_VAR, REPO_ROOT, logs_dir, runtime_root
from data.egress import (
    ProbeResult,
    classify_environment,
    classify_failure,
)

SCHEMA_VERSION = 1

PASS = "PASS"
FAIL = "FAIL"
WARN = "WARN"
UNKNOWN = "UNKNOWN"

READY = "READY_FOR_PAPER_OPERATION"
BLOCKED = "BLOCKED"

MIN_PYTHON = (3, 11)
MIN_FREE_BYTES = 2 * 1024 * 1024 * 1024      # a session writes history and logs
DESK_PORTS = (8765, 8766, 5173)

#: Filesystems that are commonly cleared on reboot. A runtime root here holds
#: market evidence that may not be there tomorrow.
EPHEMERAL_PREFIXES = ("/tmp/", "/var/tmp/", "/dev/shm/")

#: Hosts the desk genuinely needs, and one unrelated control. The control is
#: what makes ENVIRONMENT_EGRESS_BLOCKED decidable: without something outside
#: the exchange to compare against, "NSE is down" and "we are unplugged" look
#: identical.
MARKET_ENDPOINTS: tuple[tuple[str, str, str], ...] = (
    ("nse_official", "https://www.nseindia.com", "NSE website"),
    ("nse_archive", "https://nsearchives.nseindia.com", "NSE archives (bhavcopy)"),
    ("zerodha", "https://api.kite.trade", "Zerodha Kite API"),
    ("control", "https://pypi.org", "unrelated control endpoint"),
)

#: Secret names only. Values are never read into the report.
REQUIRED_SECRETS = ("KITE_API_KEY", "KITE_API_SECRET")
OPTIONAL_SECRETS = ("KITE_ACCESS_TOKEN", "TELEGRAM_BOT_TOKEN", "TELEGRAM_CHAT_ID",
                    "DEEPSEEK_API_KEY")


@dataclass(frozen=True)
class Check:
    name: str
    status: str
    detail: str = ""
    required: bool = True
    evidence: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _ok(name, detail="", **evidence) -> Check:
    return Check(name, PASS, detail, True, dict(evidence))


def _bad(name, detail, *, required=True, **evidence) -> Check:
    return Check(name, FAIL, detail, required, dict(evidence))


def _warn(name, detail, **evidence) -> Check:
    return Check(name, WARN, detail, False, dict(evidence))


def _unknown(name, detail, *, required=True, **evidence) -> Check:
    return Check(name, UNKNOWN, detail, required, dict(evidence))


# ── local checks ───────────────────────────────────────────────────────────
def check_python_runtime() -> Check:
    version = tuple(sys.version_info[:2])
    if version < MIN_PYTHON:
        return _bad("python_runtime",
                    f"Python {version[0]}.{version[1]} is below the required "
                    f"{MIN_PYTHON[0]}.{MIN_PYTHON[1]}",
                    found=".".join(str(v) for v in version))
    return _ok("python_runtime", f"Python {sys.version.split()[0]}")


def check_repository_sha() -> Check:
    try:
        sha = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT,
                             capture_output=True, text=True, timeout=10,
                             check=True).stdout.strip()
    except Exception as exc:
        return _unknown("repository_sha",
                        f"could not establish the deployed SHA: {type(exc).__name__}",
                        required=False)
    dirty = ""
    try:
        status = subprocess.run(["git", "status", "--porcelain"], cwd=REPO_ROOT,
                                capture_output=True, text=True, timeout=10,
                                check=True).stdout.strip()
        dirty = " (working tree has uncommitted changes)" if status else ""
    except Exception:
        dirty = ""
    if dirty:
        return _warn("repository_sha", f"{sha}{dirty}", sha=sha, dirty=True)
    return _ok("repository_sha", sha, sha=sha)


def check_runtime_root() -> Check:
    root = runtime_root()
    configured = bool(os.environ.get(ENV_VAR, "").strip())
    try:
        root.mkdir(parents=True, exist_ok=True)
        probe = root / ".preflight_write_probe"
        probe.write_text("ok", encoding="utf-8")
        probe.unlink()
    except Exception as exc:
        return _bad("runtime_root", f"{root} is not writable: {type(exc).__name__}",
                    path=str(root))
    if not configured:
        return _warn(
            "runtime_root",
            f"{root} is inside the source checkout; a clean checkout or branch "
            f"change would take the accumulated market evidence with it. Set "
            f"{ENV_VAR} to a persistent path.",
            path=str(root), persistent=False,
        )
    if str(root).startswith(EPHEMERAL_PREFIXES):
        return _warn("runtime_root",
                     f"{root} is under a temp filesystem and may not survive a reboot",
                     path=str(root), persistent=False)
    return _ok("runtime_root", f"{root} (persistent, writable)",
               path=str(root), persistent=True)


def check_disk_space() -> Check:
    root = runtime_root()
    try:
        usage = shutil.disk_usage(root if root.exists() else root.parent)
    except Exception as exc:
        return _unknown("disk_space", f"could not measure: {type(exc).__name__}")
    free_gb = usage.free / (1024 ** 3)
    if usage.free < MIN_FREE_BYTES:
        return _bad("disk_space", f"{free_gb:.1f} GB free is below the "
                                  f"{MIN_FREE_BYTES / 1024 ** 3:.0f} GB a session needs",
                    free_bytes=usage.free)
    return _ok("disk_space", f"{free_gb:.1f} GB free", free_bytes=usage.free)


def check_database_writable() -> Check:
    target = runtime_root() / "db" / "preflight.sqlite3"
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(target) as conn:
            conn.execute("CREATE TABLE IF NOT EXISTS probe (id INTEGER PRIMARY KEY)")
            conn.execute("INSERT INTO probe DEFAULT VALUES")
            conn.commit()
        target.unlink(missing_ok=True)
        for suffix in ("-wal", "-shm"):
            Path(str(target) + suffix).unlink(missing_ok=True)
    except Exception as exc:
        return _bad("database_writable",
                    f"sqlite could not write under the runtime root: "
                    f"{type(exc).__name__}: {exc}"[:160], path=str(target.parent))
    return _ok("database_writable", f"sqlite writes under {target.parent}")


def check_clock() -> Check:
    """The desk gates on IST. A host whose clock is wrong trades the wrong day."""
    try:
        from core.market_clock import now_ist

        ist = now_ist()
    except Exception as exc:
        return _bad("clock_timezone", f"IST clock unavailable: {type(exc).__name__}")
    utc = datetime.now(timezone.utc)
    if not (2024 <= utc.year <= 2100):
        return _bad("clock_timezone", f"system clock reads {utc.isoformat()}, "
                                      "which is not a plausible date")
    offset = ist.utcoffset()
    expected_minutes = 330
    if offset is None or int(offset.total_seconds() // 60) != expected_minutes:
        return _bad("clock_timezone", f"IST offset resolved to {offset}, expected +05:30")
    return _ok("clock_timezone",
               f"UTC {utc.strftime('%Y-%m-%d %H:%M')} · IST {ist.strftime('%H:%M')}",
               utc=utc.isoformat(), ist=ist.isoformat())


def check_ports() -> Check:
    """A port already in use usually means a second copy of the desk."""
    import socket as _socket

    busy = []
    for port in DESK_PORTS:
        sock = _socket.socket()
        try:
            sock.settimeout(1.0)
            if sock.connect_ex(("127.0.0.1", port)) == 0:
                busy.append(port)
        finally:
            sock.close()
    if busy:
        return _warn("ports_available",
                     f"already serving on {busy} — is the desk already running?",
                     busy=busy)
    return _ok("ports_available", f"{list(DESK_PORTS)} free")


def check_secrets() -> Check:
    """Presence only. A preflight that prints a key is a preflight that leaks one."""
    missing = [name for name in REQUIRED_SECRETS if not os.environ.get(name, "").strip()]
    absent_optional = [name for name in OPTIONAL_SECRETS
                       if not os.environ.get(name, "").strip()]
    if missing:
        return _bad("secrets_present",
                    f"missing required credentials: {', '.join(missing)}",
                    missing=missing, optional_absent=absent_optional)
    if absent_optional:
        return _warn("secrets_present",
                     f"present: {', '.join(REQUIRED_SECRETS)} · not set: "
                     f"{', '.join(absent_optional)}",
                     optional_absent=absent_optional)
    return _ok("secrets_present", "all configured credentials are present")


def check_live_lock() -> Check:
    """The one check whose failure must stop the install, not warn about it."""
    try:
        from product.live_execution_interlock import get_live_execution_state

        state = get_live_execution_state()
    except Exception as exc:
        return _bad("live_execution_locked",
                    f"the interlock could not be read: {type(exc).__name__}; "
                    "trading workflows must not start")
    if not (state.locked and state.verified) or state.authorized:
        return _bad("live_execution_locked",
                    f"live execution is not verified-locked: status={state.status}")
    return _ok("live_execution_locked", f"{state.status} · {state.reason}"[:160])


def check_service_installation() -> Check:
    """Whether anything will restart the desk after a reboot."""
    systemctl = shutil.which("systemctl")
    if systemctl:
        try:
            enabled = subprocess.run([systemctl, "is-enabled", "quantterm.service"],
                                     capture_output=True, text=True, timeout=10)
            active = subprocess.run([systemctl, "is-active", "quantterm.service"],
                                    capture_output=True, text=True, timeout=10)
        except Exception as exc:
            return _unknown("service_installation",
                            f"systemd query failed: {type(exc).__name__}",
                            required=False)
        state = enabled.stdout.strip() or enabled.stderr.strip() or "unknown"
        running = active.stdout.strip() or "unknown"
        if state == "enabled":
            return _ok("service_installation",
                       f"systemd unit enabled (currently {running})",
                       manager="systemd", enabled=True, active=running)
        return _warn("service_installation",
                     "systemd is available but quantterm.service is not enabled; "
                     "the desk will not come back after a reboot",
                     manager="systemd", enabled=False)
    if sys.platform == "darwin" and shutil.which("launchctl"):
        return _warn("service_installation",
                     "launchd is available; no QuantTerm agent is installed",
                     manager="launchd", enabled=False)
    return _warn("service_installation",
                 "no supported service manager found; the desk will not survive a reboot",
                 manager="", enabled=False)


# ── network checks ─────────────────────────────────────────────────────────
def _default_probe(url: str, timeout: float = 8.0) -> ProbeResult:
    """One bounded HEAD/GET. Never raises; classifies instead."""
    from urllib.parse import urlsplit

    host = urlsplit(url).hostname or url
    try:
        import requests

        response = requests.get(url, timeout=timeout,
                                headers={"User-Agent": "QuantTerm-preflight/1"})
        if response.status_code >= 400:
            return ProbeResult(host, host, False,
                               classify_failure(f"{response.status_code} response",
                                                status_code=response.status_code),
                               f"HTTP {response.status_code}")
        return ProbeResult(host, host, True, "", f"HTTP {response.status_code}")
    except Exception as exc:
        return ProbeResult(host, host, False,
                           classify_failure(f"{type(exc).__name__}: {exc}"),
                           f"{type(exc).__name__}: {exc}"[:200])


def probe_market_access(
    probe: Callable[[str], ProbeResult] | None = None,
    endpoints: Sequence[tuple[str, str, str]] = MARKET_ENDPOINTS,
) -> tuple[list[Check], dict[str, Any]]:
    """Probe every endpoint, then decide whether the machine is the problem."""
    runner = probe or _default_probe
    results: list[ProbeResult] = []
    labels: dict[str, tuple[str, str]] = {}
    for name, url, description in endpoints:
        result = runner(url)
        results.append(result)
        labels[result.host] = (name, description)

    controls = [host for host, (name, _desc) in labels.items() if name == "control"]
    verdict = classify_environment(results, control_hosts=controls)
    checks: list[Check] = []

    if verdict.market_blocked:
        # ONE finding. Listing four providers as independently broken would be
        # four wrong statements about the world.
        # ONE finding. Listing every provider as independently broken would be
        # several wrong statements about the world.
        checks.append(Check(
            "market_access", FAIL, verdict.reason, True,
            {"failure_class": verdict.failure_class,
             "blocked_hosts": list(verdict.blocked_hosts),
             "reachable_hosts": list(verdict.reachable_hosts),
             "note": "provider retries are futile until the environment changes"},
        ))
        return checks, verdict.as_dict()

    for result in results:
        name, description = labels.get(result.host, (result.host, result.host))
        required = name != "control"
        if result.ok:
            checks.append(Check(name, PASS, f"{description} reachable", required,
                                {"host": result.host}))
        else:
            checks.append(Check(
                name, FAIL if required else WARN,
                f"{description} unreachable: {result.detail}"[:200], required,
                {"host": result.host, "failure_class": result.failure_class},
            ))
    return checks, verdict.as_dict()


# ── the verdict ────────────────────────────────────────────────────────────
def run_host_preflight(
    probe: Callable[[str], ProbeResult] | None = None,
    *,
    skip_network: bool = False,
) -> dict[str, Any]:
    checks: list[Check] = [
        check_python_runtime(),
        check_repository_sha(),
        check_runtime_root(),
        check_disk_space(),
        check_database_writable(),
        check_clock(),
        check_ports(),
        check_secrets(),
        check_live_lock(),
        check_service_installation(),
    ]
    environment: dict[str, Any] = {}
    if skip_network:
        checks.append(Check(
            "market_access", UNKNOWN,
            "network probes were skipped, so real-market access is unproven",
            True, {"skipped": True},
        ))
    else:
        network_checks, environment = probe_market_access(probe)
        checks.extend(network_checks)

    blockers = [c for c in checks if c.required and c.status in (FAIL, UNKNOWN)]
    warnings = [c for c in checks if c.status == WARN]
    verdict = BLOCKED if blockers else READY

    return {
        "schema_version": SCHEMA_VERSION,
        "checked_at": datetime.now(timezone.utc).isoformat(),
        "verdict": verdict,
        "runtime_root": str(runtime_root()),
        "checks": [c.to_dict() for c in checks],
        "blockers": [{"check": c.name, "detail": c.detail} for c in blockers],
        "warnings": [{"check": c.name, "detail": c.detail} for c in warnings],
        "environment": environment,
    }


def render_text(report: Mapping[str, Any]) -> str:
    lines = [f"HOST PREFLIGHT  {report['verdict']}",
             f"runtime root    {report['runtime_root']}",
             f"checked         {report['checked_at']}", ""]
    width = max((len(c["name"]) for c in report["checks"]), default=10)
    for check in report["checks"]:
        marker = {PASS: "ok  ", FAIL: "FAIL", WARN: "warn", UNKNOWN: "????"}[check["status"]]
        required = "" if check["required"] else " (optional)"
        lines.append(f"  [{marker}] {check['name']:<{width}}  {check['detail']}{required}")
    if report["blockers"]:
        lines += ["", "BLOCKERS"]
        lines += [f"  - {b['check']}: {b['detail']}" for b in report["blockers"]]
    else:
        lines += ["", "No blockers. This host can run paper operation."]
    return "\n".join(lines)


def write_preflight(report: Mapping[str, Any]) -> Path:
    target = logs_dir() / "product" / "host_preflight.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(".tmp")
    tmp.write_text(json.dumps(dict(report), indent=2, default=str), encoding="utf-8")
    os.replace(tmp, target)
    return target


def main(argv: list[str] | None = None) -> int:
    """0 when the host is ready for paper operation, 1 when it is blocked."""
    import argparse

    parser = argparse.ArgumentParser(description="QuantTerm host preflight")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--skip-network", action="store_true",
                        help="local checks only; the verdict can never be READY")
    parser.add_argument("--no-write", action="store_true")
    args = parser.parse_args(argv)

    report = run_host_preflight(skip_network=args.skip_network)
    if not args.no_write:
        try:
            write_preflight(report)
        except Exception as exc:
            print(f"# could not persist the preflight: {type(exc).__name__}: {exc}")
    print(json.dumps(report, indent=2, default=str) if args.json else render_text(report))
    return 0 if report["verdict"] == READY else 1


if __name__ == "__main__":  # pragma: no cover - exercised through main()
    raise SystemExit(main())
