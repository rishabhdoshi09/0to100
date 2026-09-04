"""Cheap process resource probes. No network. No scans."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

RESOURCE_OK = "OK"
RESOURCE_PRESSURE = "RESOURCE_PRESSURE"
RESOURCE_EXHAUSTED = "RESOURCE_EXHAUSTED"

PRESSURE_RATIO = 0.70
EXHAUSTED_RATIO = 0.90


def count_open_fds(pid: int | None = None) -> int | None:
    """Open file-descriptor count for ``pid``. None when the OS cannot tell."""
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
    try:
        import resource as posix_resource

        usage = posix_resource.getrusage(posix_resource.RUSAGE_SELF)
        soft = fd_soft_limit()
        nfile = getattr(usage, "ru_nfile", None)
        if nfile is not None:
            return int(nfile)
        if soft is not None and pid in {None, os.getpid()}:
            return None
    except Exception:
        return None
    return None


def fd_soft_limit() -> int | None:
    try:
        import resource as posix_resource

        soft, _hard = posix_resource.getrlimit(posix_resource.RLIMIT_NOFILE)
    except Exception:
        return None
    if soft is None or soft <= 0:
        return None
    return int(soft)


def classify_fd_pressure(fd_count: int | None, soft_limit: int | None) -> str:
    if not fd_count or not soft_limit:
        return RESOURCE_OK
    ratio = float(fd_count) / float(soft_limit)
    if ratio >= EXHAUSTED_RATIO:
        return RESOURCE_EXHAUSTED
    if ratio >= PRESSURE_RATIO:
        return RESOURCE_PRESSURE
    return RESOURCE_OK


def process_fd_snapshot(pid: int | None = None) -> dict[str, Any]:
    owner = int(pid or os.getpid())
    fd_count = count_open_fds(owner)
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
    oldest_running: dict[str, Any] | None = None,
    active_operation_age_s: float | None = None,
) -> dict[str, Any]:
    api = process_fd_snapshot(api_pid or os.getpid())
    ops = process_fd_snapshot(market_ops_pid) if market_ops_pid else {
        "pid": None,
        "fd_count": None,
        "fd_soft_limit": api.get("fd_soft_limit"),
        "fd_used_pct": None,
        "state": RESOURCE_OK,
    }
    states = {api.get("state"), ops.get("state")}
    if RESOURCE_EXHAUSTED in states:
        state = RESOURCE_EXHAUSTED
        reason = "Process file-descriptor usage is exhausted. Status pages must not claim data is still preparing."
    elif RESOURCE_PRESSURE in states:
        state = RESOURCE_PRESSURE
        reason = "Process file-descriptor usage is high. New sockets may start failing."
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
