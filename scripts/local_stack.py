"""Stop the local QuantTerm listeners by PID, or queue a market scan.

``bash scripts/run_quantterm_complete.sh`` is the one-terminal start. The
machine-wide supervisor lock lives outside any git checkout so a second
clone cannot acquire ownership and kill a healthy first desk. Only the
process that holds that lock may stop :5173/:8765/:8766. ``scan`` POSTs
``RUN_SCAN_NOW`` to the local API. It never uses ``pkill -f``; it only
signals the PIDs it resolved from TCP listen tables, ``lsof`` on macOS,
and autonomy / market-ops runtime files.
"""
from __future__ import annotations

import argparse
import json
import os
import signal
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

MACHINE_LOCK_NAME = "quantterm.supervisor.lock"
MACHINE_OWNER_NAME = "quantterm.supervisor.owner.json"


def machine_lock_path() -> Path:
    """One lock for the whole machine, not one lock per checkout."""
    runtime = os.environ.get("XDG_RUNTIME_DIR") or os.environ.get("TMPDIR") or "/tmp"
    return Path(runtime) / MACHINE_LOCK_NAME


def machine_owner_path() -> Path:
    return machine_lock_path().with_name(MACHINE_OWNER_NAME)


def _git_sha(root: str) -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            text=True,
            timeout=2,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return ""


def write_machine_owner(*, pid: int, root: str) -> dict:
    """Record which checkout holds the machine-wide supervisor lock."""
    payload = {
        "pid": int(pid),
        "root": str(Path(root).resolve()),
        "sha": _git_sha(root),
        "started_at": datetime.now(timezone.utc).isoformat(),
    }
    path = machine_owner_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return payload


def read_machine_owner() -> dict:
    path = machine_owner_path()
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return raw if isinstance(raw, dict) else {}


def port_open(port: int, host: str = "127.0.0.1", timeout_s: float = 0.4) -> bool:
    sock = socket.socket()
    sock.settimeout(max(0.1, float(timeout_s)))
    try:
        return sock.connect_ex((host, int(port))) == 0
    except OSError:
        return False
    finally:
        sock.close()


def desk_ports_healthy(ports: tuple[int, ...] = (5173, 8765)) -> bool:
    """True when every requested local desk port is accepting connections."""
    return all(port_open(port) for port in ports)


def _listen_inodes(port: int) -> set[str]:
    hex_port = f"{port:04X}"
    inodes: set[str] = set()
    for table in ("/proc/net/tcp", "/proc/net/tcp6"):
        path = Path(table)
        if not path.exists():
            continue
        try:
            lines = path.read_text(encoding="utf-8", errors="replace").splitlines()[1:]
        except OSError:
            continue
        for line in lines:
            parts = line.split()
            if len(parts) < 10:
                continue
            local = parts[1]
            if ":" not in local:
                continue
            _ip, local_port = local.rsplit(":", 1)
            if local_port.upper() != hex_port:
                continue
            if parts[3] != "0A":  # TCP_LISTEN
                continue
            inode = parts[9]
            if inode and inode != "0":
                inodes.add(inode)
    return inodes


def _pids_on_port_proc(port: int) -> list[int]:
    inodes = _listen_inodes(port)
    if not inodes:
        return []
    found: set[int] = set()
    proc = Path("/proc")
    try:
        entries = list(proc.iterdir())
    except OSError:
        return []
    for entry in entries:
        if not entry.name.isdigit():
            continue
        fd_dir = entry / "fd"
        try:
            fds = list(fd_dir.iterdir())
        except OSError:
            continue
        for fd in fds:
            try:
                target = os.readlink(fd)
            except OSError:
                continue
            if target.startswith("socket:[") and target[8:-1] in inodes:
                found.add(int(entry.name))
                break
    return sorted(found)


def _pids_on_port_lsof(port: int) -> list[int]:
    """macOS / BSD: /proc/net/tcp is absent, so use lsof listen PIDs."""
    port = int(port)
    binaries = ("lsof", "/usr/sbin/lsof", "/usr/bin/lsof")
    arg_sets = (
        ["-nP", f"-iTCP:{port}", "-sTCP:LISTEN", "-t"],
        ["-nP", f"-iTCP:{port}", "-t"],
        ["-nP", f"-i:{port}", "-t"],
    )
    for binary in binaries:
        for extra in arg_sets:
            try:
                out = subprocess.check_output(
                    [binary, *extra],
                    text=True,
                    stderr=subprocess.DEVNULL,
                )
            except (OSError, subprocess.CalledProcessError):
                continue
            found: set[int] = set()
            for token in out.split():
                try:
                    pid = int(token)
                except ValueError:
                    continue
                if pid > 1:
                    found.add(pid)
            if found:
                return sorted(found)
    return []


def pids_on_port(port: int) -> list[int]:
    """PIDs with a TCP LISTEN socket on ``port`` (Linux /proc, else lsof)."""
    return _pids_on_port_proc(port) or _pids_on_port_lsof(port)


def _pid_from_json(path: Path, *keys: str) -> int | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict):
        return None
    for key in keys:
        try:
            pid = int(payload.get(key) or 0)
        except (TypeError, ValueError):
            continue
        if pid > 1:
            return pid
    return None


def autonomy_pid(root: Path | None = None) -> int | None:
    repo = Path(root) if root else Path(__file__).resolve().parents[1]
    base = repo / "logs" / "autonomy"
    for name, keys in (
        ("runtime.json", ("scheduler_owner_pid", "worker_pid")),
        ("status.json", ("scheduler_owner_pid",)),
    ):
        pid = _pid_from_json(base / name, *keys)
        if pid:
            return pid
    lock = base / "supervisor.lock"
    try:
        raw = lock.read_text(encoding="utf-8").strip()
        pid = int(raw.split()[0])
    except (OSError, TypeError, ValueError, IndexError):
        pid = 0
    return pid if pid > 1 else None


def market_ops_pid(root: Path | None = None) -> int | None:
    repo = Path(root) if root else Path(__file__).resolve().parents[1]
    return _pid_from_json(repo / "logs" / "market_ops" / "runtime.json", "worker_pid")


def stop_pid(pid: int, *, timeout_s: float = 2.0) -> bool:
    if pid <= 1 or pid == os.getpid():
        return False
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    try:
        os.kill(pid, signal.SIGTERM)
    except OSError:
        return False
    deadline = time.monotonic() + max(0.2, timeout_s)
    while time.monotonic() < deadline:
        try:
            os.kill(pid, 0)
        except OSError:
            return True
        time.sleep(0.05)
    try:
        os.kill(pid, signal.SIGKILL)
    except OSError:
        return True
    return True


def stop_local_stack(*, ports: tuple[int, ...] = (5173, 8765, 8766), autonomy: bool = True) -> list[int]:
    stopped: list[int] = []
    seen: set[int] = set()

    def _stop(pid: int | None) -> None:
        if not pid or pid in seen:
            return
        seen.add(pid)
        if stop_pid(pid):
            stopped.append(pid)

    if autonomy:
        _stop(autonomy_pid())
        _stop(market_ops_pid())
    for port in ports:
        for pid in pids_on_port(port):
            _stop(pid)
    return stopped


def queue_control(control: str, *, origin: str = "http://127.0.0.1:8765", timeout_s: float = 8.0) -> dict:
    """POST one owner control to the local market API."""
    name = str(control or "").strip().upper()
    url = origin.rstrip("/") + f"/api/controls/{name}"
    req = urllib.request.Request(url, data=b"", method="POST")
    try:
        with urllib.request.urlopen(req, timeout=max(1.0, float(timeout_s))) as response:
            raw = response.read().decode("utf-8", "replace")
            status = int(response.status)
    except urllib.error.HTTPError as exc:
        raise RuntimeError(f"{name} HTTP {exc.code}") from exc
    except Exception as exc:
        raise RuntimeError(str(exc)) from exc
    if status >= 400:
        raise RuntimeError(f"{name} HTTP {status}")
    payload = json.loads(raw) if raw.strip() else {}
    if not isinstance(payload, dict):
        payload = {"raw": raw}
    payload.setdefault("control", name)
    return payload


def queue_scan_now(*, origin: str = "http://127.0.0.1:8765", timeout_s: float = 8.0) -> dict:
    """POST RUN_SCAN_NOW to the local market API. Used by the one-terminal stack."""
    return queue_control("RUN_SCAN_NOW", origin=origin, timeout_s=timeout_s)


DESK_START_CONTROLS = ("RUN_SCAN_NOW", "REFRESH_NEWS_NOW", "REFRESH_LONG_TERM_NOW")


def queue_desk_jobs(*, origin: str = "http://127.0.0.1:8765", timeout_s: float = 8.0) -> dict:
    """Queue scan, news and long-term funds together so one terminal fills the desk."""
    jobs: list[dict] = []
    accepted = 0
    for name in DESK_START_CONTROLS:
        try:
            item = queue_control(name, origin=origin, timeout_s=timeout_s)
            if item.get("accepted"):
                accepted += 1
            jobs.append(item)
        except Exception as exc:
            jobs.append({"accepted": False, "control": name, "error": str(exc)})
    return {"accepted": accepted > 0, "queued": accepted, "jobs": jobs}


def _parse_ports(raw: str) -> tuple[int, ...]:
    return tuple(int(part) for part in str(raw or "").split(",") if part.strip())


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "action",
        choices=(
            "stop",
            "pids",
            "scan",
            "machine-lock-path",
            "ports-healthy",
            "write-owner",
            "owner-status",
        ),
    )
    parser.add_argument("--ports", default="5173,8765,8766")
    parser.add_argument("--no-autonomy", action="store_true")
    parser.add_argument("--origin", default="http://127.0.0.1:8765")
    parser.add_argument("--pid", type=int, default=0)
    parser.add_argument("--root", default="")
    args = parser.parse_args(argv)
    if args.action == "machine-lock-path":
        print(machine_lock_path())
        return 0
    if args.action == "write-owner":
        root = args.root or os.getcwd()
        pid = args.pid or os.getpid()
        print(json.dumps(write_machine_owner(pid=pid, root=root)))
        return 0
    if args.action == "owner-status":
        print(json.dumps(read_machine_owner()))
        return 0
    if args.action == "ports-healthy":
        ports = _parse_ports(args.ports) or (5173, 8765)
        ok = desk_ports_healthy(ports)
        print(json.dumps({"healthy": ok, "ports": {str(port): port_open(port) for port in ports}}))
        return 0 if ok else 1
    if args.action == "scan":
        try:
            payload = queue_desk_jobs(origin=args.origin)
        except Exception as exc:
            print(f"[STACK] Desk jobs not queued yet: {exc}", file=sys.stderr)
            return 1
        print(json.dumps(payload))
        return 0 if payload.get("accepted") else 1
    ports = tuple(int(part) for part in args.ports.split(",") if part.strip())
    if args.action == "pids":
        payload = {str(port): pids_on_port(port) for port in ports}
        if not args.no_autonomy:
            payload["autonomy"] = [autonomy_pid()] if autonomy_pid() else []
            payload["market_ops"] = [market_ops_pid()] if market_ops_pid() else []
        print(json.dumps(payload))
        return 0
    stopped = stop_local_stack(ports=ports, autonomy=not args.no_autonomy)
    print(json.dumps({"stopped": stopped}))
    if not stopped and args.action == "stop":
        print(json.dumps({
            "autonomy_pid": autonomy_pid(),
            "market_ops_pid": market_ops_pid(),
            "ports": {str(port): pids_on_port(port) for port in ports},
        }))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
