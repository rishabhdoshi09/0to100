"""Always-on host supervisor for the QuantTerm paper/shadow desk.

The OS service manager owns exactly this process; this process owns the desk
children. Mutable state stays under QT_RUNTIME_ROOT, live execution is verified
locked before any child starts, and crash/recovery truth is persisted rather
than overwritten by a generic STOPPED marker.
"""
from __future__ import annotations

import fcntl
import json
import os
import shutil
import signal
import subprocess
import sys
import threading
import time
import urllib.request
from collections import defaultdict, deque
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import IO, Any, Callable

from core.runtime_paths import REPO_ROOT, ensure_logs_path, logs_path, runtime_path, runtime_root

STATUS_PATH = Path("state") / "host_supervisor.json"
LOCK_NAME = "quantterm.supervisor.lock"
OWNER_NAME = "quantterm.supervisor.owner.json"
HEARTBEAT_SECONDS = 2.0
MARKET_REPROBE_SECONDS = 1800.0
HEALTH_FAILURE_LIMIT = 3
CRASH_LOOP_WINDOW_S = 300.0
CRASH_LOOP_LIMIT = 5
STOP_GRACE_S = 12.0


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _actual_git_sha() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT,
            capture_output=True, text=True, timeout=8, check=True,
        ).stdout.strip()
    except Exception:
        return ""


def _git_sha() -> str:
    return os.environ.get("QT_BUILD_SHA", "").strip() or _actual_git_sha()


def machine_lock_path() -> Path:
    runtime = os.environ.get("XDG_RUNTIME_DIR") or os.environ.get("TMPDIR") or "/tmp"
    return Path(runtime) / LOCK_NAME


def machine_owner_path() -> Path:
    return machine_lock_path().with_name(OWNER_NAME)


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    os.replace(tmp, path)


def _port_url_ok(url: str, timeout: float = 1.5) -> bool:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            return int(response.status) == 200
    except Exception:
        return False


def _market_ops_health() -> bool:
    path = logs_path("market_ops", "runtime.json")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        pid = int(payload.get("worker_pid") or 0)
        heartbeat = float(payload.get("heartbeat_epoch") or 0)
        if not payload.get("process_running") or pid <= 1 or time.time() - heartbeat > 15:
            return False
        os.kill(pid, 0)
        return True
    except Exception:
        return False


def _autonomy_health() -> bool:
    try:
        from product.autonomy_status import read_autonomy_status
        return bool(read_autonomy_status().get("running"))
    except Exception:
        return False


def _rotate_service_log(path: Path, *, max_bytes: int = 20 * 1024 * 1024, backups: int = 5) -> None:
    try:
        if not path.exists() or path.stat().st_size < max_bytes:
            return
        for index in range(backups, 0, -1):
            src = path.with_name(path.name + f".{index}")
            if index == backups:
                src.unlink(missing_ok=True)
            else:
                dst = path.with_name(path.name + f".{index + 1}")
                if src.exists():
                    os.replace(src, dst)
        os.replace(path, path.with_name(path.name + ".1"))
    except OSError:
        pass


@dataclass(frozen=True)
class ChildSpec:
    name: str
    argv: tuple[str, ...]
    health: Callable[[], bool]
    start_after: tuple[str, ...] = ()


def child_specs() -> tuple[ChildSpec, ...]:
    python = sys.executable
    npm = shutil.which("npm") or "npm"
    return (
        ChildSpec("autonomy", (python, "-u", "main.py", "autonomy"), _autonomy_health),
        ChildSpec("market_ops", (python, "-u", "-m", "operations.market_ops"), _market_ops_health),
        ChildSpec(
            "market_api",
            (python, "-u", "-m", "uvicorn", "terminal_product_api_parallel:app",
             "--host", "127.0.0.1", "--port", "8765"),
            lambda: _port_url_ok("http://127.0.0.1:8765/api/health"),
        ),
        ChildSpec(
            "report_api",
            (python, "-u", "-m", "uvicorn", "report_api:app",
             "--host", "127.0.0.1", "--port", "8766"),
            lambda: _port_url_ok("http://127.0.0.1:8766/health"),
        ),
        ChildSpec(
            "frontend",
            (npm, "--prefix", str(REPO_ROOT / "frontend"), "run", "dev", "--",
             "--host", "127.0.0.1", "--port", "5173"),
            lambda: _port_url_ok("http://127.0.0.1:5173/"),
            start_after=("market_api",),
        ),
    )


class Child:
    def __init__(self, spec: ChildSpec):
        self.spec = spec
        self.proc: subprocess.Popen | None = None
        self.log: IO[bytes] | None = None
        self.restarts = 0
        self.health_failures = 0
        self.last_start = 0.0
        self.last_health_ok: bool | None = None

    @property
    def alive(self) -> bool:
        return bool(self.proc and self.proc.poll() is None)

    def start(self) -> None:
        if self.alive:
            return
        log_path = ensure_logs_path("service", f"{self.spec.name}.log")
        _rotate_service_log(log_path)
        self.log = open(log_path, "ab", buffering=0)
        env = os.environ.copy()
        env["PYTHONPATH"] = str(REPO_ROOT)
        env["QT_NONINTERACTIVE"] = "1"
        env["QT_NO_BROWSER"] = "1"
        self.proc = subprocess.Popen(
            list(self.spec.argv), cwd=REPO_ROOT, env=env,
            stdout=self.log, stderr=subprocess.STDOUT, start_new_session=True,
        )
        self.last_start = time.time()
        self.health_failures = 0
        self.last_health_ok = None

    def terminate(self, grace_s: float = STOP_GRACE_S) -> None:
        proc = self.proc
        if not proc or proc.poll() is not None:
            self.close_log()
            return
        try:
            os.killpg(proc.pid, signal.SIGTERM)
        except OSError:
            try:
                proc.terminate()
            except OSError:
                pass
        deadline = time.monotonic() + grace_s
        while proc.poll() is None and time.monotonic() < deadline:
            time.sleep(0.1)
        if proc.poll() is None:
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except OSError:
                try:
                    proc.kill()
                except OSError:
                    pass
        try:
            proc.wait(timeout=2)
        except Exception:
            pass
        self.close_log()

    def close_log(self) -> None:
        if self.log:
            try:
                self.log.close()
            except Exception:
                pass
            self.log = None


class HostSupervisor:
    def __init__(self):
        self.children = {spec.name: Child(spec) for spec in child_specs()}
        self.stop_requested = False
        self.started_at = _now()
        self.restart_times: dict[str, deque[float]] = defaultdict(deque)
        self.scan_kicked = False
        self.lock_fh: IO[str] | None = None
        self.bootstrap: dict[str, Any] = {}
        self.market_access: dict[str, Any] = {}
        self._last_market_probe = 0.0
        self._market_probe_thread: threading.Thread | None = None

    def verify_safety(self) -> None:
        if not os.environ.get("QT_RUNTIME_ROOT", "").strip():
            raise RuntimeError("QT_RUNTIME_ROOT is required for installed host operation")
        if runtime_root().resolve() == REPO_ROOT.resolve():
            raise RuntimeError("installed host runtime root must be outside the source checkout")
        expected_sha = os.environ.get("QT_BUILD_SHA", "").strip()
        actual_sha = _actual_git_sha()
        if expected_sha and actual_sha and expected_sha != actual_sha:
            raise RuntimeError(
                f"installed service is pinned to {expected_sha} but checkout is {actual_sha}; "
                "reinstall/validate the exact SHA before starting"
            )
        from product.live_execution_interlock import get_live_execution_state
        state = get_live_execution_state()
        if not (state.locked and state.verified) or state.authorized:
            raise RuntimeError(f"live execution is not verified-locked: {state.status}")

    def bootstrap_host(self) -> None:
        from product.host_bootstrap import bootstrap_host_state
        self.bootstrap = bootstrap_host_state(should_stop=lambda: self.stop_requested)
        state = str(self.bootstrap.get("state") or "FAILED")
        if state in {"FAILED", "CANCELLED"}:
            raise RuntimeError(
                f"host bootstrap {state.lower()}: {self.bootstrap.get('error') or 'no detail'}"
            )

    def _probe_market_access_worker(self) -> None:
        try:
            from product.host_preflight import FAIL, UNKNOWN, probe_market_access
            checks, environment = probe_market_access()
            self.market_access = {
                "checked_at": _now(),
                "environment": environment,
                "checks": [c.to_dict() for c in checks],
            }
            required_blocked = any(c.required and c.status in {FAIL, UNKNOWN} for c in checks)
            # If the first official-history bootstrap was degraded only because
            # market egress was unavailable, a later successful coarse reprobe
            # owns one retry. No fast provider retry loop is introduced.
            if str(self.bootstrap.get("state") or "") == "DEGRADED" and not required_blocked:
                from product.host_bootstrap import bootstrap_host_state
                self.bootstrap = bootstrap_host_state(should_stop=lambda: self.stop_requested)
        except Exception as exc:
            self.market_access = {
                "checked_at": _now(), "environment": {}, "checks": [],
                "error": f"{type(exc).__name__}: {exc}"[:300],
            }

    def maybe_reprobe_market_access(self, *, force: bool = False) -> None:
        now = time.monotonic()
        if not force and now - self._last_market_probe < MARKET_REPROBE_SECONDS:
            return
        if self._market_probe_thread is not None and self._market_probe_thread.is_alive():
            return
        self._last_market_probe = now
        self._market_probe_thread = threading.Thread(
            target=self._probe_market_access_worker,
            name="quantterm-market-egress-probe", daemon=True,
        )
        self._market_probe_thread.start()

    def acquire_machine_lock(self) -> None:
        path = machine_lock_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        self.lock_fh = open(path, "a+", encoding="utf-8")
        try:
            fcntl.flock(self.lock_fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            self.lock_fh.close()
            self.lock_fh = None
            raise RuntimeError("another QuantTerm supervisor owns the machine lock") from exc
        _atomic_json(machine_owner_path(), {
            "pid": os.getpid(), "root": str(REPO_ROOT.resolve()),
            "runtime_root": str(runtime_root().resolve()), "sha": _git_sha(),
            "started_at": self.started_at,
        })

    def _record_restart(self, name: str) -> None:
        now = time.monotonic()
        q = self.restart_times[name]
        q.append(now)
        while q and now - q[0] > CRASH_LOOP_WINDOW_S:
            q.popleft()
        if len(q) > CRASH_LOOP_LIMIT:
            raise RuntimeError(f"{name} entered a crash loop ({len(q)} starts in {CRASH_LOOP_WINDOW_S:.0f}s)")

    def start_child(self, name: str) -> None:
        child = self.children[name]
        for dependency in child.spec.start_after:
            dep = self.children[dependency]
            if not dep.alive or not dep.spec.health():
                return
        self._record_restart(name)
        child.start()
        child.restarts += 1

    def start_all(self) -> None:
        for name in ("autonomy", "market_ops", "market_api", "report_api"):
            self.start_child(name)
        deadline = time.monotonic() + 45
        while time.monotonic() < deadline and not self.children["market_api"].spec.health():
            if not self.children["market_api"].alive:
                break
            time.sleep(0.5)
        self.start_child("frontend")

    def _kick_scan_once(self) -> None:
        if self.scan_kicked:
            return
        if not self.children["market_api"].spec.health() or not _market_ops_health():
            return
        try:
            proc = subprocess.run(
                [sys.executable, "scripts/local_stack.py", "scan"], cwd=REPO_ROOT,
                env={**os.environ, "PYTHONPATH": str(REPO_ROOT)},
                capture_output=True, text=True, timeout=20,
            )
            if proc.returncode == 0:
                self.scan_kicked = True
        except Exception:
            pass

    def _supervise_child(self, child: Child) -> None:
        if not child.alive:
            child.last_health_ok = False
            child.close_log()
            self.start_child(child.spec.name)
            return
        try:
            healthy = bool(child.spec.health())
        except Exception:
            healthy = False
        child.last_health_ok = healthy
        if healthy:
            child.health_failures = 0
            return
        if time.time() - child.last_start < 20:
            return
        child.health_failures += 1
        if child.health_failures < HEALTH_FAILURE_LIMIT:
            return
        child.terminate()
        self.start_child(child.spec.name)

    def status_payload(self, *, state: str = "RUNNING", error: str = "") -> dict[str, Any]:
        return {
            "schema_version": 1,
            "state": state,
            "pid": os.getpid(),
            "started_at": self.started_at,
            "heartbeat_at": _now(),
            "production_sha": _git_sha(),
            "runtime_root": str(runtime_root()),
            "live_locked_required": True,
            "error": error,
            "bootstrap": self.bootstrap,
            "market_access": self.market_access,
            "children": {
                name: {
                    "pid": child.proc.pid if child.proc else None,
                    "alive": child.alive,
                    "healthy": child.last_health_ok,
                    "health_failures": child.health_failures,
                    "starts": child.restarts,
                }
                for name, child in self.children.items()
            },
        }

    def write_status(self, **kwargs: Any) -> None:
        try:
            _atomic_json(runtime_path(STATUS_PATH), self.status_payload(**kwargs))
        except Exception:
            pass

    def stop_all(self, *, terminal_state: str = "STOPPED", error: str = "") -> None:
        for name in ("frontend", "market_api", "report_api", "market_ops", "autonomy"):
            self.children[name].terminate()
        self.write_status(state=terminal_state, error=error)

    def run(self) -> int:
        terminal_state = "STOPPED"
        terminal_error = ""
        try:
            self.verify_safety()
            self.acquire_machine_lock()
            self.bootstrap_host()
            self.maybe_reprobe_market_access(force=True)
            self.start_all()
            self.write_status()
            while not self.stop_requested:
                for child in self.children.values():
                    self._supervise_child(child)
                self._kick_scan_once()
                self.maybe_reprobe_market_access()
                self.write_status()
                time.sleep(HEARTBEAT_SECONDS)
            return 0
        except Exception as exc:
            terminal_state = "FAILED"
            terminal_error = f"{type(exc).__name__}: {exc}"[:300]
            self.write_status(state=terminal_state, error=terminal_error)
            print(f"HOST SUPERVISOR FAILED: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
            return 1
        finally:
            self.stop_all(terminal_state=terminal_state, error=terminal_error)


def main() -> int:
    supervisor = HostSupervisor()

    def _stop(_signum, _frame):
        supervisor.stop_requested = True

    signal.signal(signal.SIGTERM, _stop)
    signal.signal(signal.SIGINT, _stop)
    return supervisor.run()


if __name__ == "__main__":
    raise SystemExit(main())
