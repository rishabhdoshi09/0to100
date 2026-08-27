"""Stop the local QuantTerm listeners by PID, or queue a market scan.

``bash scripts/run_quantterm_complete.sh`` is the one-terminal start. After git
pull, operators pass ``--restart`` which calls this module. ``scan`` POSTs
``RUN_SCAN_NOW`` to the local API. It never uses ``pkill -f``; it only signals
the PIDs it resolved from TCP listen tables, ``lsof`` on macOS, and autonomy /
market-ops runtime files.
"""
from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
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


def queue_scan_now(*, origin: str = "http://127.0.0.1:8765", timeout_s: float = 8.0) -> dict:
    """POST RUN_SCAN_NOW to the local market API. Used by the one-terminal stack."""
    url = origin.rstrip("/") + "/api/controls/RUN_SCAN_NOW"
    req = urllib.request.Request(url, data=b"", method="POST")
    try:
        with urllib.request.urlopen(req, timeout=max(1.0, float(timeout_s))) as response:
            raw = response.read().decode("utf-8", "replace")
            status = int(response.status)
    except urllib.error.HTTPError as exc:
        raise RuntimeError(f"scan control HTTP {exc.code}") from exc
    except Exception as exc:
        raise RuntimeError(str(exc)) from exc
    if status >= 400:
        raise RuntimeError(f"scan control HTTP {status}")
    payload = json.loads(raw) if raw.strip() else {}
    if not isinstance(payload, dict):
        payload = {"raw": raw}
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("stop", "pids", "scan"))
    parser.add_argument("--ports", default="5173,8765,8766")
    parser.add_argument("--no-autonomy", action="store_true")
    parser.add_argument("--origin", default="http://127.0.0.1:8765")
    args = parser.parse_args(argv)
    if args.action == "scan":
        try:
            payload = queue_scan_now(origin=args.origin)
        except Exception as exc:
            print(f"[STACK] Market scan not queued yet: {exc}", file=sys.stderr)
            return 1
        print(json.dumps(payload))
        return 0
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
