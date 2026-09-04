"""Cheap process resource probes. No network. No scans."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

RESOURCE_OK = "OK"
RESOURCE_PRESSURE = "RESOURCE_PRESSURE"
RESOURCE_EXHAUSTED = "RESOURCE_EXHAUSTED"
RESOURCE_UNKNOWN = "RESOURCE_UNKNOWN"

PRESSURE_RATIO = 0.70
EXHAUSTED_RATIO = 0.90
_SELF_FD_SCAN_CAP = 4096


def fd_soft_limit() -> int | None:
    try:
        import resource as posix_resource

        soft, _hard = posix_resource.getrlimit(posix_resource.RLIMIT_NOFILE)
    except Exception:
        return None
    if soft is None or soft <= 0:
        return None
    return int(soft)


def _count_self_fds_fstat() -> int | None:
    """macOS/Linux self-count without /proc or lsof. Bounded by the soft limit."""
    soft = fd_soft_limit() or 256
    cap = min(int(soft), _SELF_FD_SCAN_CAP)
    n = 0
    for fd in range(cap):
        try:
            os.fstat(fd)
        except OSError:
            continue
        n += 1
    return n


def count_open_fds(pid: int | None = None) -> int | None:
    """Open file-descriptor count for ``pid``. None when the OS cannot tell.

    Linux uses ``/proc/<pid>/fd``. macOS has no /proc; the current process
    counts its own descriptors with ``fstat``. Other processes must persist
    their own count — this function does not shell out to lsof.
    """
    try:
        value = int(pid or os.getpid())
    except (TypeError, ValueError):
        return None
    if value <= 1:
        return None
    proc_fd = Path(f"/proc/{value}/fd")
    if proc_fd.is_dir():
        try:
            return len(os.listdir(proc_fd))
        except OSError:
            return None
    if value == os.getpid():
        return _count_self_fds_fstat()
    return None


def classify_fd_pressure(fd_count: int | None, soft_limit: int | None) -> str:
    if fd_count is None or soft_limit is None:
        return RESOURCE_UNKNOWN
    if soft_limit <= 0:
        return RESOURCE_UNKNOWN
    ratio = float(fd_count) / float(soft_limit)
    if ratio >= EXHAUSTED_RATIO:
        return RESOURCE_EXHAUSTED
    if ratio >= PRESSURE_RATIO:
        return RESOURCE_PRESSURE
    return RESOURCE_OK


def process_fd_snapshot(pid: int | None = None, *, persisted_fd_count: int | None = None) -> dict[str, Any]:
    owner = int(pid or os.getpid())
    fd_count = count_open_fds(owner)
    if fd_count is None and persisted_fd_count is not None:
        try:
            fd_count = int(persisted_fd_count)
        except (TypeError, ValueError):
            fd_count = None
    soft = fd_soft_limit()
    used_pct = None
    if fd_count is not None and soft:
        used_pct = round(100.0 * float(fd_count) / float(soft), 1)
    state = classify_fd_pressure(fd_count, soft)
    return {
        "pid": owner,
        "fd_count": fd_count,
        "fd_soft_limit": soft,
        "fd_used_pct": used_pct,
        "state": state,
    }


def resource_diagnostics(
    *,
    api_pid: int | None = None,
    market_ops_pid: int | None = None,
    market_ops_fd_count: int | None = None,
    oldest_running: dict[str, Any] | None = None,
    active_operation_age_s: float | None = None,
) -> dict[str, Any]:
    api = process_fd_snapshot(api_pid or os.getpid())
    if market_ops_pid:
        ops = process_fd_snapshot(market_ops_pid, persisted_fd_count=market_ops_fd_count)
    else:
        ops = {
            "pid": None,
            "fd_count": None,
            "fd_soft_limit": api.get("fd_soft_limit"),
            "fd_used_pct": None,
            "state": RESOURCE_UNKNOWN,
        }
    states = {api.get("state"), ops.get("state")}
    if RESOURCE_EXHAUSTED in states:
        state = RESOURCE_EXHAUSTED
        reason = "Process file-descriptor usage is exhausted. Status pages must not claim data is still preparing."
    elif RESOURCE_PRESSURE in states:
        state = RESOURCE_PRESSURE
        reason = "Process file-descriptor usage is high. New sockets may start failing."
    elif RESOURCE_UNKNOWN in states:
        state = RESOURCE_UNKNOWN
        reason = "File-descriptor usage could not be measured. This is not a safe-band OK."
    else:
        state = RESOURCE_OK
        reason = "File-descriptor usage is within the safe band."
    return {
        "state": state,
        "reason": reason,
        "api": api,
        "market_ops": ops,
        "active_operation_age_s": active_operation_age_s,
        "oldest_running_operation": oldest_running,
    }
