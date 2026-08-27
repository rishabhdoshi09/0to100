"""Stop the local QuantTerm listeners by PID so a re-run loads current code.

``bash scripts/run_quantterm_complete.sh`` reuses a healthy API, desk and
autonomy process by default. After git pull, operators pass ``--restart``
which calls this module. It never uses ``pkill -f``; it only signals the
PIDs it resolved from TCP listen tables and autonomy ``status.json``.
"""
from __future__ import annotations

import argparse
import json
import os
import signal
import time
from pathlib import Path


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


def pids_on_port(port: int) -> list[int]:
    """Linux PIDs with a TCP LISTEN socket on ``port``."""
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


def autonomy_pid(root: Path | None = None) -> int | None:
    base = Path(root) if root else Path(__file__).resolve().parents[1] / "logs" / "autonomy"
    for name in ("status.json", "supervisor.lock"):
        path = base / name
        try:
            raw = path.read_text(encoding="utf-8").strip()
        except OSError:
            continue
        if name == "supervisor.lock":
            try:
                pid = int(raw.split()[0])
            except (TypeError, ValueError, IndexError):
                continue
            if pid > 1:
                return pid
            continue
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            continue
        try:
            pid = int(payload.get("scheduler_owner_pid") or 0)
        except (TypeError, ValueError):
            continue
        if pid > 1:
            return pid
    return None


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
    if autonomy:
        pid = autonomy_pid()
        if pid and pid not in seen:
            if stop_pid(pid):
                stopped.append(pid)
            seen.add(pid)
    for port in ports:
        for pid in pids_on_port(port):
            if pid in seen:
                continue
            if stop_pid(pid):
                stopped.append(pid)
            seen.add(pid)
    return stopped


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("stop", "pids"))
    parser.add_argument("--ports", default="5173,8765,8766")
    parser.add_argument("--no-autonomy", action="store_true")
    args = parser.parse_args(argv)
    ports = tuple(int(part) for part in args.ports.split(",") if part.strip())
    if args.action == "pids":
        payload = {str(port): pids_on_port(port) for port in ports}
        if not args.no_autonomy:
            payload["autonomy"] = [autonomy_pid()] if autonomy_pid() else []
        print(json.dumps(payload))
        return 0
    stopped = stop_local_stack(ports=ports, autonomy=not args.no_autonomy)
    print(json.dumps({"stopped": stopped}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
