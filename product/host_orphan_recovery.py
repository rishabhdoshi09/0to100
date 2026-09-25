"""Conservative recovery of children orphaned by a dead QuantTerm supervisor.

Recovery is deliberately evidence-heavy.  We never signal a PID merely because
its command looks familiar or a lock file contains a number.  The durable
supervisor snapshot and machine-owner record must identify the same dead parent,
the same repository/runtime generation, and every still-live child must match
the process-group and start-time facts created by HostSupervisor.Child.start().
All candidates are validated before any signal is sent.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import signal
import subprocess
import time
from datetime import datetime, timezone
from typing import Any

from core.runtime_paths import REPO_ROOT, runtime_path, runtime_root
from product.host_supervisor import machine_owner_path

STATUS_REL = Path("state") / "host_supervisor.json"
RECOVERY_REL = Path("state") / "host_orphan_recovery.json"
TERM_GRACE_S = 4.0
KILL_GRACE_S = 4.0
START_SLOP_BEFORE_S = 10.0
START_SLOP_AFTER_S = 30.0


class OrphanRecoveryError(RuntimeError):
    pass


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read_json(path: Path) -> dict[str, Any]:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        return raw if isinstance(raw, dict) else {}
    except Exception:
        return {}


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    # Pin the already-existing state directory.  Startup recovery runs before
    # the normal storage watchdog, so it must not mkdir an internal /Volumes
    # lookalike if removable storage disappears between preflight and write.
    parent_fd = -1
    file_fd = -1
    tmp = f".{path.name}.{os.getpid()}.{time.time_ns()}.tmp"
    flags = os.O_RDONLY | int(getattr(os, "O_DIRECTORY", 0))
    try:
        parent_fd = os.open(str(path.parent), flags)
        data = json.dumps(payload, indent=2, default=str).encode("utf-8")
        file_fd = os.open(
            tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600, dir_fd=parent_fd,
        )
        view = memoryview(data)
        while view:
            written = os.write(file_fd, view)
            view = view[written:]
        os.fsync(file_fd)
        os.close(file_fd)
        file_fd = -1
        os.replace(tmp, path.name, src_dir_fd=parent_fd, dst_dir_fd=parent_fd)
    except Exception as exc:
        raise OrphanRecoveryError(
            f"cannot persist orphan-recovery truth safely: {type(exc).__name__}: {exc}"
        ) from exc
    finally:
        if file_fd >= 0:
            os.close(file_fd)
        if parent_fd >= 0:
            try:
                os.unlink(tmp, dir_fd=parent_fd)
            except OSError:
                pass
            os.close(parent_fd)


def _iso_epoch(value: object) -> float | None:
    try:
        parsed = datetime.fromisoformat(str(value or "").replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed.timestamp()
    except Exception:
        return None


def _pid_alive(pid: int) -> bool:
    if pid <= 1:
        return False
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False


def _process_row(pid: int) -> dict[str, Any]:
    proc = subprocess.run(
        ["/bin/ps", "-ww", "-o", "pid=", "-o", "ppid=", "-o", "pgid=",
         "-o", "lstart=", "-o", "command=", "-p", str(pid)],
        capture_output=True, text=True, check=False, timeout=5,
    )
    line = (proc.stdout or "").strip()
    if proc.returncode != 0 or not line:
        raise OrphanRecoveryError(f"cannot inspect candidate child pid={pid}")
    parts = line.split(None, 8)
    if len(parts) < 9:
        raise OrphanRecoveryError(f"unexpected ps row for candidate child pid={pid}: {line[:200]}")
    try:
        started = datetime.strptime(" ".join(parts[3:8]), "%a %b %d %H:%M:%S %Y")
        started_epoch = started.timestamp()  # ps lstart is host-local time
        return {
            "pid": int(parts[0]),
            "ppid": int(parts[1]),
            "pgid": int(parts[2]),
            "started": " ".join(parts[3:8]),
            "started_epoch": started_epoch,
            "command": parts[8],
        }
    except Exception as exc:
        raise OrphanRecoveryError(
            f"cannot parse ps row for candidate child pid={pid}: {type(exc).__name__}"
        ) from exc


def _command_matches(name: str, command: str) -> bool:
    text = str(command or "")
    repo = str(REPO_ROOT)
    if name == "frontend":
        # npm may rewrite its visible argv from the original
        #   npm --prefix <repo>/frontend run dev -- --host ... --port 5173
        # to a shorter
        #   npm run dev -- --host ... --port 5173
        # while keeping the same recorded PID/process group. Accept either that
        # canonical npm dev-server shape or the Vite node executable it launches.
        exact_launch = all(
            piece in text
            for piece in ("npm", "--prefix", f"{repo}/frontend", "run", "dev")
        )
        npm_rewritten = all(
            piece in text
            for piece in ("npm", "run", "dev", "--host", "127.0.0.1", "--port", "5173")
        )
        vite_child_shape = all(
            piece in text
            for piece in (f"{repo}/frontend/node_modules/.bin/vite", "--host", "127.0.0.1", "--port", "5173")
        )
        return exact_launch or npm_rewritten or vite_child_shape

    signatures = {
        "autonomy": ("main.py", "autonomy"),
        "market_ops": ("operations.market_ops",),
        "market_api": ("uvicorn", "terminal_product_api_parallel:app"),
        "report_api": ("uvicorn", "report_api:app"),
    }
    required = signatures.get(name)
    return bool(required) and all(piece in text for piece in required)


def _group_commands(pgid: int) -> list[str]:
    if pgid <= 1:
        return []
    proc = subprocess.run(
        ["/bin/ps", "-ww", "-axo", "pid=", "-o", "ppid=", "-o", "pgid=", "-o", "command="],
        capture_output=True, text=True, check=False, timeout=5,
    )
    commands: list[str] = []
    for raw in (proc.stdout or "").splitlines():
        row = raw.strip()
        if not row:
            continue
        parts = row.split(None, 3)
        if len(parts) < 4:
            continue
        try:
            row_pgid = int(parts[2])
        except ValueError:
            continue
        if row_pgid == pgid:
            commands.append(parts[3])
    return commands


def _frontend_group_matches(pgid: int) -> bool:
    repo = str(REPO_ROOT)
    for command in _group_commands(pgid):
        text = str(command or "")
        if all(
            piece in text
            for piece in (
                f"{repo}/frontend/node_modules/.bin/vite",
                "--host",
                "127.0.0.1",
                "--port",
                "5173",
            )
        ):
            return True
    return False


def _group_alive(pgid: int) -> bool:
    if pgid <= 1:
        return False
    try:
        os.killpg(pgid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False


def _wait_group_gone(pgid: int, timeout_s: float) -> bool:
    deadline = time.monotonic() + max(0.0, timeout_s)
    while _group_alive(pgid):
        if time.monotonic() >= deadline:
            return False
        time.sleep(0.05)
    return True

def _supervisor_command_matches(command: str) -> bool:
    text = str(command or "")
    return any(
        marker in text
        for marker in (
            "product.host_launchd_entrypoint",
            "product.host_entrypoint",
            "product.host_supervisor",
        )
    )


def _wait_pid_gone(pid: int, timeout_s: float) -> bool:
    deadline = time.monotonic() + max(0.0, timeout_s)
    while _pid_alive(pid):
        if time.monotonic() >= deadline:
            return False
        time.sleep(0.05)
    return True


def quiesce_live_previous_supervisor() -> dict[str, Any]:
    """Stop only a fully-proven old canonical supervisor generation.

    This is intended for the macOS updater after verified launchd bootout.
    launchd can be unloaded while the Python supervisor is still reparented
    to PID 1. That survivor keeps the machine-wide flock and blocks the new
    exact-SHA generation.

    Safety is fail-closed: durable owner/status identity, repo/runtime identity,
    generation timestamp and the live process command must all agree before a
    signal is sent. After the parent is gone, existing orphan recovery validates
    and reaps any proven child process groups.
    """
    status_path = runtime_path(STATUS_REL)
    owner_path = machine_owner_path()
    status = _read_json(status_path)
    owner = _read_json(owner_path)

    if not owner or not status:
        return {
            "schema_version": 1,
            "state": "NO_PROVEN_LIVE_SUPERVISOR",
            "owner_present": bool(owner),
            "status_present": bool(status),
            "terminated": False,
        }

    status_pid = int(status.get("pid") or 0)
    owner_pid = int(owner.get("pid") or 0)
    repo = str(REPO_ROOT)
    runtime = str(runtime_root())

    if status_pid <= 1 or owner_pid != status_pid:
        raise OrphanRecoveryError("previous supervisor owner/status PID identity is not trustworthy")
    if str(owner.get("root") or "") != repo:
        raise OrphanRecoveryError("previous supervisor owner repository does not match current checkout")
    if str(owner.get("runtime_root") or "") != runtime:
        raise OrphanRecoveryError("previous supervisor owner runtime does not match configured runtime")
    if str(status.get("runtime_root") or "") != runtime:
        raise OrphanRecoveryError("previous supervisor status runtime does not match configured runtime")
    if str(owner.get("started_at") or "") != str(status.get("started_at") or ""):
        raise OrphanRecoveryError("previous supervisor owner/status generation timestamps do not match")

    if not _pid_alive(status_pid):
        recovered = reconcile_previous_children()
        return {
            "schema_version": 1,
            "state": "SUPERVISOR_ALREADY_GONE",
            "previous_supervisor_pid": status_pid,
            "terminated": False,
            "recovery": recovered,
        }

    row = _process_row(status_pid)
    if int(row.get("pid") or 0) != status_pid:
        raise OrphanRecoveryError("previous supervisor PID changed while inspecting it")
    if int(row.get("ppid") or 0) != 1:
        raise OrphanRecoveryError(
            f"previous supervisor pid={status_pid} is not launchd-orphaned (ppid={row.get('ppid')})"
        )
    if not _supervisor_command_matches(str(row.get("command") or "")):
        raise OrphanRecoveryError(
            f"previous supervisor pid={status_pid} command is not canonical QuantTerm host"
        )

    try:
        os.kill(status_pid, signal.SIGTERM)
    except ProcessLookupError:
        pass

    signal_used = "SIGTERM"
    if not _wait_pid_gone(status_pid, 20.0):
        try:
            os.kill(status_pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        signal_used = "SIGKILL"
        if not _wait_pid_gone(status_pid, KILL_GRACE_S):
            raise OrphanRecoveryError(
                f"verified previous supervisor pid={status_pid} remained alive after SIGKILL"
            )

    recovered = reconcile_previous_children()
    return {
        "schema_version": 1,
        "state": "QUIESCED",
        "previous_supervisor_pid": status_pid,
        "signal": signal_used,
        "terminated": True,
        "recovery": recovered,
    }


def _terminate_groups(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    terminated: list[dict[str, Any]] = []
    for candidate in candidates:
        pgid = int(candidate["pgid"])
        if not _group_alive(pgid):
            terminated.append({**candidate, "signal": "already_gone"})
            continue
        try:
            os.killpg(pgid, signal.SIGTERM)
        except ProcessLookupError:
            terminated.append({**candidate, "signal": "already_gone"})
            continue
        if _wait_group_gone(pgid, TERM_GRACE_S):
            terminated.append({**candidate, "signal": "SIGTERM"})
            continue
        try:
            os.killpg(pgid, signal.SIGKILL)
        except ProcessLookupError:
            terminated.append({**candidate, "signal": "SIGTERM"})
            continue
        if not _wait_group_gone(pgid, KILL_GRACE_S):
            raise OrphanRecoveryError(
                f"verified orphan process group {pgid} remained alive after SIGKILL"
            )
        terminated.append({**candidate, "signal": "SIGKILL"})
    return terminated


def _unowned_recorded_conflicts(status: dict[str, Any]) -> list[dict[str, Any]]:
    """Find recorded children that still look like the old generation.

    With no machine-owner record we never signal anything.  This function only
    distinguishes a credible lingering QuantTerm child from an unrelated PID
    that happened to be reused after a clean stop.
    """
    started_epoch = _iso_epoch(status.get("started_at"))
    heartbeat_epoch = _iso_epoch(status.get("heartbeat_at"))
    conflicts: list[dict[str, Any]] = []
    for name, child in (status.get("children") or {}).items():
        if not isinstance(child, dict):
            continue
        pid = int(child.get("pid") or 0)
        if pid <= 1 or not _pid_alive(pid):
            continue
        try:
            row = _process_row(pid)
        except OrphanRecoveryError:
            # An uninspectable live recorded PID cannot be safely dismissed.
            conflicts.append({"name": str(name), "pid": pid, "reason": "uninspectable"})
            continue
        same_role = _command_matches(str(name), str(row.get("command") or ""))
        same_group_shape = int(row.get("ppid") or 0) == 1 and int(row.get("pgid") or 0) == pid
        same_generation = False
        if started_epoch is not None and heartbeat_epoch is not None:
            child_started = float(row.get("started_epoch") or 0)
            same_generation = (
                child_started >= started_epoch - START_SLOP_BEFORE_S
                and child_started <= heartbeat_epoch + START_SLOP_AFTER_S
            )
        if same_role and same_group_shape and same_generation:
            conflicts.append({"name": str(name), "pid": pid, "reason": "credible_orphan"})
    return conflicts


def _validate_previous_generation(
    status: dict[str, Any],
    owner: dict[str, Any],
) -> list[dict[str, Any]]:
    runtime = str(runtime_root())
    repo = str(REPO_ROOT)
    status_pid = int(status.get("pid") or 0)
    owner_pid = int(owner.get("pid") or 0)

    if status_pid <= 1 or owner_pid != status_pid:
        raise OrphanRecoveryError("previous supervisor owner/status PID identity is not trustworthy")
    if _pid_alive(status_pid):
        raise OrphanRecoveryError(
            f"previous supervisor pid={status_pid} is still alive; refusing orphan cleanup"
        )
    if str(owner.get("root") or "") != repo:
        raise OrphanRecoveryError("previous supervisor owner repository does not match current checkout")
    if str(owner.get("runtime_root") or "") != runtime:
        raise OrphanRecoveryError("previous supervisor owner runtime does not match configured runtime")
    if str(status.get("runtime_root") or "") != runtime:
        raise OrphanRecoveryError("previous supervisor status runtime does not match configured runtime")
    if str(owner.get("started_at") or "") != str(status.get("started_at") or ""):
        raise OrphanRecoveryError("previous supervisor owner/status generation timestamps do not match")

    started_epoch = _iso_epoch(status.get("started_at"))
    heartbeat_epoch = _iso_epoch(status.get("heartbeat_at"))
    if started_epoch is None or heartbeat_epoch is None or heartbeat_epoch < started_epoch:
        raise OrphanRecoveryError("previous supervisor timestamps are not trustworthy")

    children = status.get("children") or {}
    candidates: list[dict[str, Any]] = []
    for name, child in children.items():
        if not isinstance(child, dict):
            continue
        pid = int(child.get("pid") or 0)
        if pid <= 1 or not _pid_alive(pid):
            continue
        row = _process_row(pid)
        if row["pid"] != pid:
            raise OrphanRecoveryError(f"candidate PID changed while inspecting {name}")
        if int(row["ppid"]) != 1:
            raise OrphanRecoveryError(
                f"candidate {name} pid={pid} is not orphaned (ppid={row['ppid']})"
            )
        if int(row["pgid"]) != pid:
            raise OrphanRecoveryError(
                f"candidate {name} pid={pid} is not its expected process-group leader "
                f"(pgid={row['pgid']})"
            )
        command_matches = _command_matches(str(name), str(row["command"]))
        if not command_matches and str(name) == "frontend":
            command_matches = _frontend_group_matches(int(row["pgid"]))
        if not command_matches:
            raise OrphanRecoveryError(
                f"candidate {name} pid={pid} command/process-group does not match the recorded child role: "
                f"{str(row.get('command') or '')[:220]}"
            )
        child_started = float(row["started_epoch"])
        if child_started < started_epoch - START_SLOP_BEFORE_S:
            raise OrphanRecoveryError(
                f"candidate {name} pid={pid} predates its recorded supervisor generation"
            )
        if child_started > heartbeat_epoch + START_SLOP_AFTER_S:
            raise OrphanRecoveryError(
                f"candidate {name} pid={pid} started after the recorded generation ended"
            )
        candidates.append({"name": str(name), **row})

    return candidates


def reconcile_previous_children() -> dict[str, Any]:
    """Reap a fully-proven dead generation, otherwise fail closed before startup."""
    status_path = runtime_path(STATUS_REL)
    owner_path = machine_owner_path()
    recovery_path = runtime_path(RECOVERY_REL)
    status = _read_json(status_path)
    owner = _read_json(owner_path)

    if not owner:
        # A persisted terminal snapshot without a machine-owner record is normal
        # after a clean stop, but only if none of its recorded child PIDs are
        # still alive.  Live recorded children without owner evidence are an
        # ownership conflict, not permission to guess.
        live_recorded = _unowned_recorded_conflicts(status) if status else []
        if live_recorded:
            raise OrphanRecoveryError(
                f"recorded QuantTerm children are still alive but owner evidence is missing: "
                f"{live_recorded}"
            )
        result = {
            "schema_version": 1,
            "state": "NO_LIVE_PREVIOUS_CHILDREN",
            "checked_at": _now(),
            "status_present": bool(status),
            "owner_present": False,
            "terminated": [],
        }
        _atomic_json(recovery_path, result)
        return result

    if not status:
        owner_pid = int(owner.get("pid") or 0)
        raise OrphanRecoveryError(
            f"previous machine-owner evidence exists for pid={owner_pid} but supervisor "
            "status is missing; refusing to guess child ownership"
        )

    candidates = _validate_previous_generation(status, owner)
    terminated = _terminate_groups(candidates) if candidates else []
    result = {
        "schema_version": 1,
        "state": "RECOVERED" if terminated else "CLEAR",
        "checked_at": _now(),
        "previous_supervisor_pid": int(status.get("pid") or 0),
        "previous_started_at": status.get("started_at"),
        "previous_sha": status.get("production_sha"),
        "terminated": terminated,
    }
    _atomic_json(recovery_path, result)
    try:
        owner_path.unlink(missing_ok=True)
    except OSError:
        pass
    return result
