"""Truthful host preflight for QuantTerm PAPER/SHADOW operation.

A host is never READY without real market access and a verified locked live
execution interlock. Operational constraints such as memory pressure and macOS
sleep are surfaced explicitly rather than silently treated as healthy.
"""
from __future__ import annotations

import getpass
import json
import os
import re
import shutil
import sqlite3
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from core.runtime_paths import REPO_ROOT, logs_dir, runtime_root
from data.egress import ProbeResult, classify_environment, classify_failure

SCHEMA_VERSION = 1
PASS = "PASS"
FAIL = "FAIL"
WARN = "WARN"
UNKNOWN = "UNKNOWN"
READY = "READY_FOR_PAPER_OPERATION"
BLOCKED = "BLOCKED"
MIN_PYTHON = (3, 11)
MIN_FREE_BYTES = 2 * 1024 * 1024 * 1024
MIN_AVAILABLE_MEMORY_BYTES = 4 * 1024 * 1024 * 1024
DESK_PORTS = (8765, 8766, 5173)
EPHEMERAL_PREFIXES = ("/tmp/", "/var/tmp/", "/dev/shm/")
# A probe answers two different questions and conflating them is how a working
# host gets declared unreachable. TRANSPORT asks whether this machine can reach
# the provider at all; CAPABILITY asks whether QuantTerm can actually obtain the
# data it needs. NSE answers a bare root GET with 403 (www) and 404 (archives)
# even when both production data routes serve perfectly, so a root status can
# never be the verdict.
CAPABILITY_USABLE = "USABLE"
CAPABILITY_REJECTED = "REJECTED"
CAPABILITY_ENDPOINT_INVALID = "ENDPOINT_INVALID"
CAPABILITY_UNREACHABLE = "UNREACHABLE"

#: How far back the archive probe will walk looking for a published bhavcopy.
#: 404 means holiday/weekend, not a broken provider, so it keeps walking; a long
#: NSE break plus the current day is covered well inside this bound.
ARCHIVE_PROBE_LOOKBACK_DAYS = 6

#: Only the control endpoint is optional -- it exists to separate "unplugged"
#: from "selective egress", not to gate anything. Every market provider stays
#: REQUIRED exactly as before: this change fixes how capability is determined,
#: never what the install is allowed to proceed without.
OPTIONAL_CAPABILITIES = frozenset({"control"})
REQUIRED_SECRETS = ("KITE_API_KEY", "KITE_API_SECRET")
OPTIONAL_SECRETS = (
    "KITE_ACCESS_TOKEN", "TELEGRAM_BOT_TOKEN", "TELEGRAM_CHAT_ID", "DEEPSEEK_API_KEY",
)


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


def check_python_runtime() -> Check:
    version = tuple(sys.version_info[:2])
    if version < MIN_PYTHON:
        return _bad(
            "python_runtime",
            f"Python {version[0]}.{version[1]} is below the required {MIN_PYTHON[0]}.{MIN_PYTHON[1]}",
            found=".".join(str(v) for v in version),
        )
    return _ok("python_runtime", f"Python {sys.version.split()[0]}")


def check_repository_sha() -> Check:
    try:
        sha = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT,
            capture_output=True, text=True, timeout=10, check=True,
        ).stdout.strip()
    except Exception as exc:
        return _unknown(
            "repository_sha", f"could not establish the deployed SHA: {type(exc).__name__}",
            required=False,
        )
    dirty = ""
    try:
        status = subprocess.run(
            ["git", "status", "--porcelain"], cwd=REPO_ROOT,
            capture_output=True, text=True, timeout=10, check=True,
        ).stdout.strip()
        dirty = " (working tree has uncommitted changes)" if status else ""
    except Exception:
        pass
    if dirty:
        return _warn("repository_sha", f"{sha}{dirty}", sha=sha, dirty=True)
    return _ok("repository_sha", sha, sha=sha)


def check_runtime_root() -> Check:
    try:
        root = runtime_root().expanduser().resolve()
    except Exception as exc:
        return _bad("runtime_root", f"runtime root cannot be resolved: {type(exc).__name__}: {exc}"[:180])
    try:
        root.mkdir(parents=True, exist_ok=True)
        probe = root / ".preflight_write_probe"
        probe.write_text("ok", encoding="utf-8")
        probe.unlink()
    except Exception as exc:
        return _bad("runtime_root", f"{root} is not writable: {type(exc).__name__}", path=str(root))
    if root == REPO_ROOT.resolve() or REPO_ROOT.resolve() in root.parents:
        return _warn(
            "runtime_root",
            f"{root} is inside the source checkout; accumulated market evidence is not durable across checkout changes",
            path=str(root), persistent=False,
        )
    if (str(root) + "/").startswith(EPHEMERAL_PREFIXES):
        return _warn(
            "runtime_root", f"{root} is under a temp filesystem and may not survive a reboot",
            path=str(root), persistent=False,
        )
    return _ok("runtime_root", f"{root} (persistent, writable)", path=str(root), persistent=True)


def check_disk_space() -> Check:
    root = runtime_root()
    try:
        usage = shutil.disk_usage(root if root.exists() else root.parent)
    except Exception as exc:
        return _unknown("disk_space", f"could not measure: {type(exc).__name__}")
    free_gb = usage.free / (1024 ** 3)
    if usage.free < MIN_FREE_BYTES:
        return _bad(
            "disk_space",
            f"{free_gb:.1f} GB free is below the {MIN_FREE_BYTES / 1024 ** 3:.0f} GB a session needs",
            free_bytes=usage.free,
        )
    return _ok("disk_space", f"{free_gb:.1f} GB free", free_bytes=usage.free)


def _memory_snapshot() -> dict[str, int]:
    """Return total/available RAM without making psutil a hard dependency."""
    try:
        import psutil  # type: ignore
        vm = psutil.virtual_memory()
        return {"total": int(vm.total), "available": int(vm.available)}
    except Exception:
        pass

    if sys.platform.startswith("linux"):
        try:
            rows: dict[str, int] = {}
            for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
                key, value = line.split(":", 1)
                if key in {"MemTotal", "MemAvailable"}:
                    rows[key] = int(value.strip().split()[0]) * 1024
            if rows.get("MemTotal") and rows.get("MemAvailable") is not None:
                return {"total": rows["MemTotal"], "available": rows["MemAvailable"]}
        except Exception:
            pass

    if sys.platform == "darwin":
        try:
            sysctl = subprocess.run(
                ["sysctl", "-n", "hw.memsize"], capture_output=True, text=True, timeout=5, check=True,
            )
            total = int(sysctl.stdout.strip())
            vm = subprocess.run(["vm_stat"], capture_output=True, text=True, timeout=5, check=True).stdout
            match = re.search(r"page size of (\d+) bytes", vm)
            page_size = int(match.group(1)) if match else 4096
            pages: dict[str, int] = {}
            for line in vm.splitlines():
                m = re.match(r"([^:]+):\s+(\d+)\.?", line)
                if m:
                    pages[m.group(1).strip()] = int(m.group(2))
            available_pages = sum(
                pages.get(name, 0)
                for name in ("Pages free", "Pages inactive", "Pages speculative", "Pages purgeable")
            )
            return {"total": total, "available": available_pages * page_size}
        except Exception:
            pass
    return {}


def check_memory() -> Check:
    snapshot = _memory_snapshot()
    total = int(snapshot.get("total") or 0)
    available = int(snapshot.get("available") or 0)
    if total <= 0 or available <= 0:
        return _unknown("memory_capacity", "RAM availability could not be measured", required=False)
    total_gb = total / (1024 ** 3)
    available_gb = available / (1024 ** 3)
    if available < MIN_AVAILABLE_MEMORY_BYTES:
        return _warn(
            "memory_capacity",
            f"{available_gb:.1f} GB RAM available of {total_gb:.1f} GB; low-memory hosts may swap during whole-market scans",
            total_bytes=total, available_bytes=available,
        )
    return _ok(
        "memory_capacity", f"{available_gb:.1f} GB RAM available of {total_gb:.1f} GB",
        total_bytes=total, available_bytes=available,
    )


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
        return _bad(
            "database_writable",
            f"sqlite could not write under the runtime root: {type(exc).__name__}: {exc}"[:160],
            path=str(target.parent),
        )
    return _ok("database_writable", f"sqlite writes under {target.parent}")


def check_clock() -> Check:
    try:
        from core.market_clock import now_ist
        ist = now_ist()
    except Exception as exc:
        return _bad("clock_timezone", f"IST clock unavailable: {type(exc).__name__}")
    utc = datetime.now(timezone.utc)
    if not (2024 <= utc.year <= 2100):
        return _bad("clock_timezone", f"system clock reads {utc.isoformat()}, which is not a plausible date")
    offset = ist.utcoffset()
    if offset is None or int(offset.total_seconds() // 60) != 330:
        return _bad("clock_timezone", f"IST offset resolved to {offset}, expected +05:30")
    return _ok(
        "clock_timezone", f"UTC {utc.strftime('%Y-%m-%d %H:%M')} · IST {ist.strftime('%H:%M')}",
        utc=utc.isoformat(), ist=ist.isoformat(),
    )


def check_ports() -> Check:
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
        return _warn("ports_available", f"already serving on {busy} — is the desk already running?", busy=busy)
    return _ok("ports_available", f"{list(DESK_PORTS)} free")


def check_secrets() -> Check:
    missing = [name for name in REQUIRED_SECRETS if not os.environ.get(name, "").strip()]
    absent_optional = [name for name in OPTIONAL_SECRETS if not os.environ.get(name, "").strip()]
    if missing:
        return _bad(
            "secrets_present", f"missing required credentials: {', '.join(missing)}",
            missing=missing, optional_absent=absent_optional,
        )
    if absent_optional:
        return _warn(
            "secrets_present",
            f"present: {', '.join(REQUIRED_SECRETS)} · not set: {', '.join(absent_optional)}",
            optional_absent=absent_optional,
        )
    return _ok("secrets_present", "all configured credentials are present")


def check_live_lock() -> Check:
    try:
        from product.live_execution_interlock import get_live_execution_state
        state = get_live_execution_state()
    except Exception as exc:
        return _bad(
            "live_execution_locked",
            f"the interlock could not be read: {type(exc).__name__}; trading workflows must not start",
        )
    if not (state.locked and state.verified) or state.authorized:
        return _bad("live_execution_locked", f"live execution is not verified-locked: status={state.status}")
    return _ok("live_execution_locked", f"{state.status} · {state.reason}"[:160])


def _linger_state() -> tuple[str, str]:
    loginctl = shutil.which("loginctl")
    if not loginctl:
        return "unknown", "loginctl unavailable"
    user = getpass.getuser()
    try:
        proc = subprocess.run(
            [loginctl, "show-user", user, "--property=Linger", "--value"],
            capture_output=True, text=True, timeout=10,
        )
    except Exception as exc:
        return "unknown", f"{type(exc).__name__}"
    if proc.returncode != 0:
        return "unknown", (proc.stderr.strip() or f"rc={proc.returncode}")[:120]
    return proc.stdout.strip().lower(), ""


def check_service_installation() -> Check:
    systemctl = shutil.which("systemctl")
    if systemctl:
        try:
            enabled = subprocess.run(
                [systemctl, "--user", "is-enabled", "quantterm.service"],
                capture_output=True, text=True, timeout=10,
            )
            active = subprocess.run(
                [systemctl, "--user", "is-active", "quantterm.service"],
                capture_output=True, text=True, timeout=10,
            )
        except Exception as exc:
            return _unknown(
                "service_installation", f"systemd user query failed: {type(exc).__name__}", required=False,
            )
        state = enabled.stdout.strip() or enabled.stderr.strip() or "unknown"
        running = active.stdout.strip() or "unknown"
        linger, linger_error = _linger_state()
        if state == "enabled" and linger == "yes":
            return _ok(
                "service_installation", f"systemd user unit enabled, linger=yes (currently {running})",
                manager="systemd", enabled=True, active=running, linger=True,
            )
        if state == "enabled":
            return _warn(
                "service_installation",
                "systemd user unit is enabled but linger is not verified; it may not restart after reboot without login",
                manager="systemd", enabled=True, active=running, linger=linger, linger_error=linger_error,
            )
        return _warn(
            "service_installation", "systemd user unit is not enabled; the desk will not come back after reboot/login",
            manager="systemd", enabled=False, active=running, linger=linger,
        )
    if sys.platform == "darwin" and shutil.which("launchctl"):
        plist = Path.home() / "Library" / "LaunchAgents" / "com.quantterm.desk.plist"
        if not plist.exists():
            return _warn(
                "service_installation", "launchd is available; no QuantTerm agent is installed",
                manager="launchd", enabled=False,
            )
        target = f"gui/{os.getuid()}/com.quantterm.desk"
        try:
            active = subprocess.run(
                ["launchctl", "print", target], capture_output=True, text=True, timeout=10,
            )
            running = active.returncode == 0
        except Exception:
            running = False
        return _ok(
            "service_installation", f"launchd agent installed ({'loaded' if running else 'not currently loaded'})",
            manager="launchd", enabled=True, active=running,
        )
    return _warn(
        "service_installation", "no supported service manager found; the desk will not survive a reboot",
        manager="", enabled=False,
    )


def check_power_management() -> Check:
    """Surface macOS idle/lid sleep risk for an unattended laptop host."""
    if sys.platform != "darwin":
        return _ok("power_management", "non-macOS host; macOS sleep check not applicable", platform=sys.platform)
    pmset = shutil.which("pmset") or "/usr/bin/pmset"
    if not Path(pmset).exists() and shutil.which("pmset") is None:
        return _unknown("power_management", "pmset unavailable; macOS sleep state is unverified", required=False)
    try:
        proc = subprocess.run([pmset, "-g"], capture_output=True, text=True, timeout=10)
    except Exception as exc:
        return _unknown("power_management", f"pmset query failed: {type(exc).__name__}", required=False)
    if proc.returncode != 0:
        return _unknown("power_management", f"pmset returned rc={proc.returncode}", required=False)
    sleep_minutes: int | None = None
    for line in proc.stdout.splitlines():
        match = re.match(r"^\s*sleep\s+(\d+)\b", line)
        if match:
            sleep_minutes = int(match.group(1))
            break
    caffeinate = Path("/usr/bin/caffeinate").exists() or bool(shutil.which("caffeinate"))
    if sleep_minutes is None:
        return _unknown(
            "power_management", "pmset output did not expose the active idle-sleep setting", required=False,
            raw=proc.stdout[-1000:],
        )
    if sleep_minutes > 0:
        detail = (
            f"macOS idle sleep is enabled ({sleep_minutes} min). Installed launchd service uses caffeinate -i "
            "to prevent idle sleep while running, but closing the laptop lid can still suspend market-hour work."
        )
        return _warn(
            "power_management", detail,
            sleep_minutes=sleep_minutes, caffeinate_available=caffeinate, lid_close_risk=True,
        )
    return _ok(
        "power_management",
        "macOS idle sleep is disabled; keep the laptop powered and do not close the lid unless clamshell wake is configured",
        sleep_minutes=0, caffeinate_available=caffeinate, lid_close_risk=True,
    )


@dataclass(frozen=True)
class CapabilityProbe:
    """One endpoint's answer to both questions.

    ``result.ok`` is TRANSPORT only -- any HTTP response at all proves this
    machine can reach the provider, which is what the egress classifier needs.
    ``state`` is the capability: whether QuantTerm could actually obtain data.
    """

    result: ProbeResult
    state: str
    evidence: dict[str, Any] = field(default_factory=dict)

    @property
    def usable(self) -> bool:
        return self.state == CAPABILITY_USABLE


def _transport_ok(host: str, detail: str) -> ProbeResult:
    """A provider that answered is reachable, whatever it answered.

    403 and 404 are the provider talking to us. Only DNS/TLS/timeout/proxy
    failures mean this machine cannot get out, and only those may feed the
    selective-egress verdict.
    """
    return ProbeResult(host, host, True, "", detail[:200])


def _transport_failed(host: str, exc: BaseException) -> ProbeResult:
    detail = f"{type(exc).__name__}: {exc}"
    return ProbeResult(host, host, False, classify_failure(detail), detail[:200])


def _capability_from_status(status: int) -> str:
    if status in (401, 403, 429):
        return CAPABILITY_REJECTED
    if status == 404:
        return CAPABILITY_ENDPOINT_INVALID
    if status >= 400:
        return CAPABILITY_REJECTED
    return CAPABILITY_USABLE


def _reachability_probe(url: str, timeout: float = 8.0) -> CapabilityProbe:
    """For non-NSE endpoints, where reaching the host IS the whole question.

    The control endpoint exists to separate "unplugged" from "the firewall
    allows the internet but not the exchanges", and Zerodha credentials are
    proven by check_secrets, not here. So any HTTP response is a usable answer;
    only a 5xx says the provider itself is down.
    """
    from urllib.parse import urlsplit
    host = urlsplit(url).hostname or url
    try:
        import requests
        response = requests.get(url, timeout=timeout, headers={"User-Agent": "QuantTerm-preflight/1"})
    except Exception as exc:
        return CapabilityProbe(
            _transport_failed(host, exc), CAPABILITY_UNREACHABLE,
            {"url": url, "transport": "FAILED"},
        )
    status = int(response.status_code)
    state = CAPABILITY_REJECTED if status >= 500 else CAPABILITY_USABLE
    return CapabilityProbe(
        _transport_ok(host, f"HTTP {status}"), state,
        {"url": url, "http_status": status, "transport": "OK"},
    )


def _nse_live_probe(url: str, timeout: float = 12.0) -> CapabilityProbe:
    """Exercise the real live route: cookie-primed session, production headers.

    ``data.nse_live.fetch_live_snapshot`` primes cookies with a root GET and
    then reads the equity-stockIndices API. The root GET is a cookie step, never
    a verdict -- NSE answers it 403 for plenty of clients that then get a
    perfectly good 200 from the API. Headers come from the acquisition module
    itself so the probe cannot drift away from what production sends.
    """
    from urllib.parse import urlsplit
    host = urlsplit(url).hostname or url
    try:
        import requests
        from data.nse_live import _HEADERS as NSE_LIVE_HEADERS
    except Exception as exc:
        return CapabilityProbe(
            _transport_failed(host, exc), CAPABILITY_UNREACHABLE,
            {"url": url, "transport": "NOT_ATTEMPTED",
             "note": "QuantTerm's own NSE live acquisition module could not be imported"},
        )
    session = requests.Session()
    priming_status: int | str = "not-attempted"
    try:
        session.headers.update(NSE_LIVE_HEADERS)
        try:
            priming_status = int(session.get(
                "https://www.nseindia.com", timeout=timeout).status_code)
        except Exception as exc:                      # priming is best-effort
            priming_status = f"{type(exc).__name__}"
        response = session.get(url, timeout=timeout)
    except Exception as exc:
        return CapabilityProbe(
            _transport_failed(host, exc), CAPABILITY_UNREACHABLE,
            {"url": url, "cookie_priming_status": priming_status, "transport": "FAILED"},
        )
    finally:
        try:
            session.close()
        except Exception:
            pass

    status = int(response.status_code)
    evidence: dict[str, Any] = {
        "url": url, "http_status": status, "transport": "OK",
        "cookie_priming_status": priming_status,
    }
    state = _capability_from_status(status)
    detail = f"HTTP {status}"
    if state == CAPABILITY_USABLE:
        try:
            rows = response.json().get("data") or []
        except Exception as exc:
            state, detail = CAPABILITY_REJECTED, f"HTTP 200 but unparseable: {type(exc).__name__}"
            rows = []
        else:
            evidence["rows"] = len(rows)
            if rows:
                detail = f"HTTP 200 with {len(rows)} live rows"
            else:
                state, detail = CAPABILITY_REJECTED, "HTTP 200 but no live rows returned"
    evidence["capability"] = state
    return CapabilityProbe(_transport_ok(host, detail), state, evidence)


def _archive_candidate_days(lookback: int = ARCHIVE_PROBE_LOOKBACK_DAYS) -> list[date]:
    """Recent candidate sessions, newest first, excluding today.

    Today's bhavcopy is not published until the evening, so asking for it would
    read as a holiday. Dates come from the IST clock like every other NSE gate.
    """
    from datetime import timedelta
    from core.market_clock import today_ist
    today = today_ist()
    return [today - timedelta(days=offset) for offset in range(1, lookback + 1)]


def _nse_archive_probe(url: str, timeout: float = 12.0) -> CapabilityProbe:
    """Exercise the real bhavcopy route: the exact URL and headers production uses.

    ``data.bhavcopy_store._download_day`` treats 404 as holiday/weekend and 200
    with >= 1000 bytes as a real file. The archive ROOT is not a route at all --
    it 404s permanently -- so this walks recent sessions instead and only calls
    the capability invalid when every candidate day 404s.
    """
    from urllib.parse import urlsplit
    host = urlsplit(url).hostname or url
    try:
        import requests
        from data.bhavcopy_store import _HEADERS as BHAV_HEADERS, _URL as BHAV_URL
    except Exception as exc:
        return CapabilityProbe(
            _transport_failed(host, exc), CAPABILITY_UNREACHABLE,
            {"transport": "NOT_ATTEMPTED",
             "note": "QuantTerm's own bhavcopy acquisition module could not be imported"},
        )

    attempts: list[dict[str, Any]] = []
    holidays = 0
    last_exc: BaseException | None = None
    for day in _archive_candidate_days():
        day_url = BHAV_URL.format(d=day.strftime("%d%m%Y"))
        try:
            response = requests.get(day_url, headers=BHAV_HEADERS, timeout=timeout)
        except Exception as exc:
            # Transport failure says nothing about this particular date, so
            # walking further would only multiply the timeout on a host that
            # cannot reach NSE at all.
            last_exc = exc
            attempts.append({"day": str(day), "error": f"{type(exc).__name__}"})
            break
        status = int(response.status_code)
        size = len(response.content or b"")
        attempts.append({"day": str(day), "http_status": status, "bytes": size})
        if status == 404:                              # holiday/weekend, keep walking
            holidays += 1
            continue
        if status == 200 and size >= 1000:
            return CapabilityProbe(
                _transport_ok(host, f"HTTP 200, {size} bytes"),
                CAPABILITY_USABLE,
                {"session": str(day), "url": day_url, "bytes": size,
                 "http_status": 200, "transport": "OK", "attempts": attempts,
                 "capability": CAPABILITY_USABLE},
            )
        state = CAPABILITY_REJECTED
        detail = (f"HTTP {status}" if status != 200
                  else f"HTTP 200 but only {size} bytes, not a bhavcopy")
        return CapabilityProbe(
            _transport_ok(host, detail), state,
            {"session": str(day), "url": day_url, "http_status": status,
             "bytes": size, "transport": "OK", "attempts": attempts, "capability": state},
        )

    if holidays and last_exc is None:
        # Every candidate answered, every answer was 404. The host is reachable;
        # the route did not serve a file. Truthfully invalid, not unreachable.
        return CapabilityProbe(
            _transport_ok(host, f"no bhavcopy published in the last {len(attempts)} days"),
            CAPABILITY_ENDPOINT_INVALID,
            {"transport": "OK", "attempts": attempts,
             "capability": CAPABILITY_ENDPOINT_INVALID},
        )
    exc = last_exc or RuntimeError("no bhavcopy candidate could be requested")
    return CapabilityProbe(
        _transport_failed(host, exc), CAPABILITY_UNREACHABLE,
        {"transport": "FAILED", "attempts": attempts, "capability": CAPABILITY_UNREACHABLE},
    )


#: Production data routes, not liveness pings. Each is the endpoint QuantTerm
#: actually reads in production, requested the way production requests it.
def market_endpoints() -> tuple[tuple[str, str, str], ...]:
    from urllib.parse import quote
    return (
        ("nse_official",
         f"https://www.nseindia.com/api/equity-stockIndices?index={quote('NIFTY TOTAL MARKET')}",
         "NSE live equity API (intraday overlay)"),
        ("nse_archive",
         "https://nsearchives.nseindia.com/products/content/sec_bhavdata_full_<session>.csv",
         "NSE bhavcopy archive (primary history)"),
        ("zerodha", "https://api.kite.trade", "Zerodha Kite API"),
        ("control", "https://pypi.org", "unrelated control endpoint"),
    )


_CAPABILITY_RUNNERS: dict[str, Callable[[str], CapabilityProbe]] = {
    "nse_official": _nse_live_probe,
    "nse_archive": _nse_archive_probe,
}


def _default_probe(url: str, timeout: float = 8.0) -> ProbeResult:
    """Back-compatible single-URL probe.

    Kept because callers and tests may still hand :func:`probe_market_access` a
    plain ``probe(url) -> ProbeResult``. Root-status semantics are gone here
    too: an answered request means the host is reachable.
    """
    return _reachability_probe(url, timeout).result


def _run_capability(
    name: str, url: str, probe: Callable[[str], ProbeResult] | None
) -> CapabilityProbe:
    """One endpoint's probe, honouring an injected single-URL probe if given."""
    if probe is not None:
        result = probe(url)
        if isinstance(result, CapabilityProbe):
            return result
        return CapabilityProbe(
            result,
            CAPABILITY_USABLE if result.ok else CAPABILITY_UNREACHABLE,
            {"url": url, "injected": True},
        )
    return _CAPABILITY_RUNNERS.get(name, _reachability_probe)(url)


def probe_market_access(
    probe: Callable[[str], ProbeResult] | None = None,
    endpoints: Sequence[tuple[str, str, str]] | None = None,
) -> tuple[list[Check], dict[str, Any]]:
    """Prove QuantTerm can obtain real market data, not that a root URL is up.

    Each endpoint reports transport and capability separately. The egress
    classifier only ever sees transport, so a provider that answers 403 or 404
    can no longer masquerade as a firewall. Capability then decides the check
    status, and a required capability that is not USABLE still blocks.
    """
    resolved = tuple(endpoints) if endpoints is not None else market_endpoints()
    results: list[ProbeResult] = []
    labels: dict[str, tuple[str, str]] = {}
    probes: dict[str, CapabilityProbe] = {}
    for name, url, description in resolved:
        capability = _run_capability(name, url, probe)
        results.append(capability.result)
        labels[capability.result.host] = (name, description)
        probes[name] = capability
    controls = [host for host, (name, _desc) in labels.items() if name == "control"]
    verdict = classify_environment(results, control_hosts=controls)
    checks: list[Check] = []
    if verdict.market_blocked:
        checks.append(Check(
            "market_access", FAIL, verdict.reason, True,
            {
                "failure_class": verdict.failure_class,
                "blocked_hosts": list(verdict.blocked_hosts),
                "reachable_hosts": list(verdict.reachable_hosts),
                "note": "provider retries are futile until the environment changes",
            },
        ))
        return checks, verdict.as_dict()
    for name, _url, description in resolved:
        capability = probes[name]
        required = name not in OPTIONAL_CAPABILITIES
        evidence = {
            "host": capability.result.host,
            "capability": capability.state,
            **capability.evidence,
        }
        if capability.usable:
            checks.append(Check(
                name, PASS, f"{description} usable: {capability.result.detail}"[:200],
                required, evidence,
            ))
            continue
        evidence["failure_class"] = capability.result.failure_class
        checks.append(Check(
            name, FAIL if required else WARN,
            f"{description} {capability.state}: {capability.result.detail}"[:200],
            required, evidence,
        ))
    return checks, verdict.as_dict()


def run_host_preflight(
    probe: Callable[[str], ProbeResult] | None = None, *, skip_network: bool = False,
) -> dict[str, Any]:
    checks: list[Check] = [
        check_python_runtime(), check_repository_sha(), check_runtime_root(), check_disk_space(),
        check_memory(), check_database_writable(), check_clock(), check_ports(), check_secrets(),
        check_live_lock(), check_service_installation(), check_power_management(),
    ]
    environment: dict[str, Any] = {}
    if skip_network:
        checks.append(Check(
            "market_access", UNKNOWN,
            "network probes were skipped, so real-market access is unproven", True, {"skipped": True},
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
    lines = [
        f"HOST PREFLIGHT  {report['verdict']}", f"runtime root    {report['runtime_root']}",
        f"checked         {report['checked_at']}", "",
    ]
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
    import argparse
    parser = argparse.ArgumentParser(description="QuantTerm host preflight")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--skip-network", action="store_true", help="local checks only; the verdict can never be READY")
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


if __name__ == "__main__":
    raise SystemExit(main())
