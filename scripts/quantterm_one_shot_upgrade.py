"""One-command, fail-closed update and diagnosis for installed QuantTerm.

Fetched from the approved Git branch and run with the existing Python interpreter.
No secrets printed, trading enabled or runtime deleted. The canonical host
installer remains the only authority for runtime migration and service restart.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import pwd
import re
import signal
import subprocess
import sys
import time
from urllib.error import HTTPError, URLError
from urllib.request import urlopen

BRANCH = "integration/quantterm-india-universal-20261109"
RUNTIME = Path("/Volumes/QuantTermStorage/QuantTerm/runtime")
REPO = Path.home() / "0to100"
EXPECTED_OWNER = "com.quantterm.desk"
PORTS = (8765, 8766, 5173)


class Blocked(RuntimeError):
    """Fail closed with a specific operator-visible blocker."""


def run(argv: list[str], *, cwd: Path | None = None, timeout: int = 60,
        check: bool = True) -> subprocess.CompletedProcess[str]:
    try:
        p = subprocess.run(argv, cwd=cwd, text=True, capture_output=True,
                           timeout=timeout, check=False)
    except Exception as exc:
        raise Blocked(f"{argv[0]} could not run: {type(exc).__name__}") from exc
    if check and p.returncode:
        raise Blocked(f"{argv[0]} failed (exit {p.returncode}); check local service logs")
    return p


def say(section: str) -> None:
    print(f"\n=== {section} ===", flush=True)


def git(repo: Path, *args: str) -> str:
    return run(["git", "-C", str(repo), *args], timeout=50).stdout.strip()


def verify_preflight(repo: Path, runtime: Path, branch: str) -> tuple[str, str]:
    if sys.platform != "darwin":
        raise Blocked("macOS launchd host required; no changes made")
    if not (repo / ".git").exists() or not (repo / ".env").is_file():
        raise Blocked("repository or existing .env missing; no changes made")
    if not (runtime / "state" / "host_runtime_manifest.json").is_file():
        raise Blocked("canonical persistent runtime manifest absent")
    mount = Path("/Volumes/QuantTermStorage")
    if not mount.is_mount() or not runtime.is_dir():
        raise Blocked("QuantTermStorage volume is not mounted; refusing substitute runtime")
    if "QuantTermStorage_WORKING_20261107.sparsebundle" not in run(
        ["hdiutil", "info"], timeout=30).stdout:
        raise Blocked("verified WORKING sparsebundle not attached; refusing old image")
    if git(repo, "branch", "--show-current") != branch:
        raise Blocked("checkout branch mismatch; preserve work before upgrade")
    if git(repo, "status", "--porcelain"):
        raise Blocked("checkout contains uncommitted changes; upgrade declined")
    current = git(repo, "rev-parse", "HEAD")
    latest = git(repo, "rev-parse", "FETCH_HEAD")
    remote = git(repo, "rev-parse", "origin/" + branch)
    if latest != remote or len(latest) != 40:
        raise Blocked("fetched commit and remote tracking ref disagree")
    if run(["git", "-C", str(repo), "merge-base", "--is-ancestor",
            current, latest], check=False, timeout=30).returncode:
        raise Blocked("remote is not a fast-forward descendant of checkout")
    if not (repo / "venv" / "bin" / "python").is_file():
        raise Blocked("expected existing virtualenv missing")
    print(f"SOURCE {current[:12]} -> {latest[:12]}")
    print("PERSISTENT_STORAGE verified · .env preserved")
    return current, latest


def launchd_unloaded() -> bool:
    p = run(["launchctl", "print", f"gui/{os.getuid()}/{EXPECTED_OWNER}"],
            timeout=15, check=False)
    if p.returncode == 0:
        return False
    output = (p.stdout + "\n" + p.stderr).lower()
    if any(m in output for m in (
            "could not find service", "service not found", "no such process", "not found in domain")):
        return True
    raise Blocked("launchd state ambiguous; not safe to touch worker processes")


def owned_command(command: str) -> bool:
    """Only known QuantTerm host children, not arbitrary Python/node jobs."""
    c = command.lower()
    patterns = (
        r"(?:^|\s)main\.py\s+autonomy\s*$",
        r"(?:^|\s)-m\s+operations\.market_ops(?:\s|$)",
        r"(?:^|\s)-m\s+product\.host_entrypoint(?:\s|$)",
        r"(?:^|\s)-m\s+product\.host_supervisor(?:\s|$)",
        r"(?:^|\s)-m\s+uvicorn\s+(?:terminal_api|terminal_product_api_parallel|report_api):app(?:\s|$)",
        r"(?:^|\s)uvicorn\s+(?:terminal_api|terminal_product_api_parallel|report_api):app(?:\s|$)",
        r"(?:^|\s)\S*node\s+.*?(?:node_modules/\.bin/vite|vite\s+--host)",
        r"(?:^|\s)\S*npm\s+--prefix\s+\S+/frontend\s+run\s+dev(?:\s|$)",
        r"(?:^|\s)\S*sh\s+-c\s+vite\s+--host\s+127\.0\.0\.1",
    )
    return any(re.search(pat, c) for pat in patterns)


def parse_owned_processes(ps_output: str) -> list[tuple[int, str, str]]:
    rows = []
    for line in ps_output.splitlines():
        fields = line.strip().split(None, 4)
        if len(fields) != 5 or not fields[0].isdigit():
            continue
        pid, _ppid, user, state, command = fields
        if owned_command(command) and "Z" not in state:
            rows.append((int(pid), user, command))
    return rows


def process_cwd(pid: int) -> Path | None:
    p = run(["lsof", "-nP", "-a", "-p", str(pid), "-d", "cwd", "-Fn"],
            timeout=12, check=False)
    if p.returncode:
        return None
    entries = [line[1:] for line in p.stdout.splitlines() if line.startswith("n/")]
    if len(entries) != 1:
        return None
    return Path(entries[0]).resolve()


def pid_present(pid: int) -> bool:
    p = run(["ps", "-p", str(pid), "-o", "stat="], check=False, timeout=6)
    return p.returncode == 0 and bool(p.stdout.strip()) and "Z" not in p.stdout.strip()


def remaining_processes() -> list[tuple[int, str, str]]:
    ps = run(["ps", "-axo", "pid=,ppid=,user=,stat=,command="], timeout=12).stdout
    return parse_owned_processes(ps)


def stop_orphans(repo: Path) -> None:
    if not launchd_unloaded():
        raise Blocked("launchd owner still loaded after canonical stop")
    candidates = remaining_processes()
    if not candidates:
        print("No orphan workers")
        return
    owner = pwd.getpwuid(os.getuid()).pw_name
    for pid, user, command in candidates:
        if user != owner:
            raise Blocked(f"unrecognized process owner pid={pid}; not terminating")
        cwd = process_cwd(pid)
        if cwd is None or not (cwd == repo or repo in cwd.parents):
            raise Blocked(f"could not establish QuantTerm cwd for pid={pid}")
        print(f"Verified orphan PID {pid} · {command.split()[-2:]} · cwd={cwd}")
    for pid, user, _ in candidates:
        if not pid_present(pid):
            continue
        # Verify PID again immediately before sending a signal (PID reuse guard).
        row = run(["ps", "-p", str(pid), "-o", "user=,stat=,command="],
                  timeout=6, check=False)
        fields = row.stdout.strip().split(None, 2)
        if row.returncode or len(fields) != 3 or fields[0] != user or not owned_command(fields[2]):
            raise Blocked(f"pid={pid} changed identity before termination")
        cwd = process_cwd(pid)
        if cwd is None or not (cwd == repo or repo in cwd.parents):
            raise Blocked(f"pid={pid} cwd changed before termination")
        try:
            os.kill(pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
    deadline = time.monotonic() + 75
    while time.monotonic() < deadline:
        if not any(pid_present(pid) for pid, _, _ in candidates):
            print("Previously verified orphan workers exited")
            return
        time.sleep(1)
    raise Blocked("orphan did not exit after graceful SIGTERM; no forced kill")


def ports_free() -> bool:
    p = run(["lsof", "-nP", *[f"-iTCP:{port}" for port in PORTS],
            "-sTCP:LISTEN"], timeout=12, check=False)
    return p.returncode == 1 and not p.stdout.strip()


def upgrade(repo: Path, runtime: Path, branch: str, current: str, latest: str) -> None:
    if current == latest:
        print(f"Already at target source {latest[:12]}; no service restart needed")
        return
    say("STOP VERIFIED HOST")
    python = str(repo / "venv" / "bin" / "python")
    p = run([python, "-m", "product.launchd_control", "stop"], cwd=repo, timeout=65)
    if p.stdout.strip():
        print(p.stdout.strip()[:500])
    stop_orphans(repo)
    if remaining_processes() or not ports_free():
        raise Blocked("workers or service ports still occupied; checkout untouched")
    if not launchd_unloaded():
        raise Blocked("launchd unexpectedly reloaded; checkout untouched")
    say("FAST-FORWARD VERIFIED SOURCE")
    run(["git", "-C", str(repo), "merge", "--ff-only", "origin/" + branch], timeout=65)
    if git(repo, "rev-parse", "HEAD") != latest or git(repo, "status", "--porcelain"):
        raise Blocked("checkout did not match target commit; installation stopped")
    say("CANONICAL HOST INSTALL")
    p = subprocess.run([python, "-m", "product.host_install", "install",
                        "--runtime-root", str(runtime), "--env-file", str(repo / ".env"),
                        "--manager", "launchd"], cwd=repo, text=True, timeout=2100,
                       check=False)
    if p.returncode:
        raise Blocked(f"canonical installer failed rc={p.returncode}; do not retry blindly")


def read_json(url: str, timeout: int = 12) -> tuple[int, dict]:
    try:
        with urlopen(url, timeout=timeout) as response:
            obj = json.load(response)
            return response.status, obj if isinstance(obj, dict) else {}
    except HTTPError as exc:
        return exc.code, {}
    except (URLError, TimeoutError, OSError, ValueError):
        return 0, {}


def post_checks(repo: Path, runtime: Path, latest: str) -> int:
    say("POST-INSTALL HOST AND PAPER-ONLY GUARD")
    python = str(repo / "venv" / "bin" / "python")
    st = run([python, "-m", "product.host_install", "status",
              "--runtime-root", str(runtime), "--manager", "launchd", "--json"],
             cwd=repo, timeout=75, check=False)
    try:
        status = json.loads(st.stdout)
    except ValueError:
        raise Blocked("host status JSON could not be parsed")
    dep = status.get("deployment") or {}
    live = status.get("live_execution") or {}
    children = status.get("children") or {}
    installed = dep.get("build_sha")
    safe = live.get("locked") is True and live.get("verified") is True and live.get("authorized") is False
    healthy = st.returncode == 0 and status.get("healthy") is True and installed == latest
    print("DEPLOYED_SHA:", installed)
    print("HOST:", "HEALTHY" if healthy else "NOT_READY")
    print("CHILDREN:", ", ".join(f"{name}={'UP' if row.get('healthy') else 'DOWN'}" for name,row in children.items()))
    print("LIVE_MONEY_LOCK:", "VERIFIED_LOCKED" if safe else "NOT_VERIFIED")
    if not healthy or not safe:
        raise Blocked("host services or live-money lock not verified after installation")

    say("KITE AUTH SESSION")
    auth_code = "from research.autonomy.auth import probe_auth; p=probe_auth(); print(p.status); print(p.error_code or 'NONE')"
    auth = run([python, "-c", auth_code], cwd=repo, timeout=35, check=False)
    auth_lines = auth.stdout.strip().splitlines()
    auth_state = auth_lines[-2] if len(auth_lines)>=2 else "UNKNOWN"
    print("AUTH:", auth_state)
    print("AUTH_REASON_CODE:", auth_lines[-1] if auth_lines else "UNKNOWN")

    say("API AND NSE SESSION FRESHNESS")
    code, health = read_json("http://127.0.0.1:8765/api/health", timeout=15)
    print("API_HEALTH_HTTP:", code)
    history = health.get("history") or {}
    print("AVAILABLE_NSE_SESSION:", history.get("available_session") or "UNKNOWN")
    print("EXPECTED_NSE_SESSION:", history.get("expected_session") or history.get("expected_latest_completed_session") or "UNKNOWN")
    print("NSE_CURRENT:", history.get("current"))
    print("NSE_REASON:", history.get("reason_code") or "UNKNOWN")

    say("DASHBOARD RESPONSIVENESS")
    deadline = time.monotonic() + 140
    current = False
    while True:
        t0 = time.monotonic()
        http, payload = read_json("http://127.0.0.1:8765/api/dashboard", timeout=15)
        elapsed = round(time.monotonic() - t0, 2)
        meta = payload.get("response_meta") or {}
        state = str(meta.get("status") or ("HTTP_" + str(http)))
        print(f"DASHBOARD http={http} elapsed={elapsed}s projection={state} stale={meta.get('stale')}")
        if http == 200 and state == "CURRENT" and meta.get("stale") is False:
            current = True
            break
        if time.monotonic() >= deadline:
            break
        time.sleep(min(12, max(1, deadline - time.monotonic())))
    say("ONE-PASS VERDICT")
    print("DASHBOARD:", "CURRENT read completed" if current else "WARMING/BLOCKED; heavy read not yet verified")
    if auth_state != "SESSION_VALID":
        print("KITE: session not validated; BUY blocked")
    if history.get("current") is not True:
        print("DATA: latest official NSE session not verified; BUY blocked")
    if auth_state == "SESSION_VALID" and history.get("current") is True and current:
        print("RESULT: HOST_AND_READS_VERIFIED (scan evidence still required)")
    else:
        print("RESULT: HOST_UPGRADED; missing research/data gates are reported above")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="One-pass QuantTerm recovery (paper only)")
    parser.add_argument("--repo", type=Path, default=REPO)
    parser.add_argument("--runtime", type=Path, default=RUNTIME)
    parser.add_argument("--branch", default=BRANCH)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    repo = args.repo.expanduser().resolve()
    runtime = args.runtime.expanduser().resolve()
    try:
        say("PREFLIGHT: STORAGE, GIT, HOST")
        before, after = verify_preflight(repo, runtime, args.branch)
        if args.check_only:
            print("CHECK_ONLY: no services changed")
            return 0
        upgrade(repo, runtime, args.branch, before, after)
        return post_checks(repo, runtime, after)
    except Blocked as exc:
        print("\nBLOCKED:", str(exc), file=sys.stderr)
        print("No forced kill, git reset or data deletion was performed.", file=sys.stderr)
        return 2
    except Exception as exc:
        print(f"\nBLOCKED: unexpected {type(exc).__name__}; manual review required", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
