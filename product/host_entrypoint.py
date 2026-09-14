"""Installed-host entrypoint for the QuantTerm paper/shadow desk.

The service manager starts this module, not the interactive launcher. It loads an
optional operator-owned environment file, guards the already-adopted durable
runtime, starts one low-frequency post-session report scheduler, then hands child
ownership to the canonical host supervisor.

Runtime loss is fail-closed: the storage guard signals the supervisor to stop all
children, report generation pauses, and the entrypoint waits without creating a
replacement path. If the same runtime returns, the canonical supervisor is
started again and re-verifies the exact SHA and live-money interlock.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import threading
import time
from typing import Any

REPORT_POLL_SECONDS = 15 * 60
REPORT_SCHEDULER_REL = Path("state") / "daily_report_scheduler.json"


def load_env_file(path: str | os.PathLike[str] | None) -> list[str]:
    if not path:
        return []
    target = Path(path).expanduser()
    if not target.exists():
        raise RuntimeError(f"configured QT_HOST_ENV_FILE does not exist: {target}")
    loaded: list[str] = []
    for raw in target.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:].lstrip()
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if not key or not (key[0].isalpha() or key[0] == "_"):
            continue
        if not all(ch.isalnum() or ch == "_" for ch in key):
            continue
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        if key not in os.environ:
            os.environ[key] = value
            loaded.append(key)
    return loaded


def _truthy(value: object) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "on"}


def prepare_runtime_storage_for_startup() -> str:
    """Attach/verify the configured macOS APFS runtime before strict path use.

    The shell preflight is deliberately the one existing storage authority: it
    may attach the already-configured sparsebundle, but it never creates or
    repairs a missing canonical runtime path.  Non-macOS hosts and macOS hosts
    that have not explicitly opted into the external-storage contract are left
    unchanged.
    """
    if sys.platform != "darwin" or not _truthy(os.environ.get("QT_STORAGE_PREFLIGHT_REQUIRED")):
        return "NOT_REQUIRED"

    from core.runtime_paths import REPO_ROOT

    script = REPO_ROOT / "scripts" / "quantterm_storage_preflight.sh"
    if not script.is_file():
        raise RuntimeError(f"required macOS storage preflight is missing: {script}")

    env = os.environ.copy()
    env["QT_STORAGE_PREFLIGHT_REQUIRED"] = "1"
    env["QT_STORAGE_PREFLIGHT_ATTACH"] = "1"
    try:
        completed = subprocess.run(
            ["/bin/bash", str(script)],
            cwd=REPO_ROOT,
            env=env,
            capture_output=True,
            text=True,
            timeout=90,
            check=False,
        )
    except Exception as exc:
        raise RuntimeError(
            f"macOS storage preflight could not run: {type(exc).__name__}: {exc}"
        ) from exc
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout or "storage preflight failed").strip()
        if "\n" in detail:
            detail = detail.splitlines()[-1]
        raise RuntimeError(
            f"macOS storage preflight failed with rc={completed.returncode}: {detail[:500]}"
        )
    return (completed.stdout or "[STORAGE PREFLIGHT] PASS").strip().splitlines()[-1]


def prepare_frontend_toolchain() -> str:
    """Resolve npm before host-supervisor child specs are constructed.

    launchd does not source the user's interactive shell configuration, so npm
    installed by Homebrew/nvm/Volta/asdf can disappear even though it works in
    Terminal.  Prefer an explicitly pinned QT_NPM_BIN, then the current PATH,
    then common per-user/macOS locations.  The npm directory is prepended to
    PATH so npm's ``/usr/bin/env node`` shebang resolves the matching node too.
    """
    candidates: list[Path] = []
    configured = str(os.environ.get("QT_NPM_BIN") or "").strip()
    if configured:
        candidates.append(Path(configured).expanduser())
    discovered = shutil.which("npm")
    if discovered:
        candidates.append(Path(discovered))

    home = Path.home()
    nvm_root = home / ".nvm" / "versions" / "node"
    if nvm_root.is_dir():
        candidates.extend(sorted(nvm_root.glob("*/bin/npm"), reverse=True))
    candidates.extend((
        home / ".volta" / "bin" / "npm",
        home / ".asdf" / "shims" / "npm",
        Path("/opt/homebrew/bin/npm"),
        Path("/usr/local/bin/npm"),
    ))

    seen: set[str] = set()
    for candidate in candidates:
        text = str(candidate)
        if not text or text in seen:
            continue
        seen.add(text)
        if not candidate.is_file() or not os.access(candidate, os.X_OK):
            continue
        absolute = candidate.absolute()
        bin_dir = str(absolute.parent)
        current = [p for p in os.environ.get("PATH", "").split(os.pathsep) if p]
        os.environ["PATH"] = os.pathsep.join([bin_dir, *[p for p in current if p != bin_dir]])
        os.environ["QT_NPM_BIN"] = str(absolute)
        return str(absolute)

    raise RuntimeError(
        "installed QuantTerm requires npm for the frontend, but launchd could not resolve it; "
        "set QT_NPM_BIN to the absolute npm executable and reinstall the host"
    )


def _read_scheduler_status() -> dict[str, Any]:
    from core.runtime_paths import runtime_path
    path = runtime_path(REPORT_SCHEDULER_REL)
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        return raw if isinstance(raw, dict) else {}
    except Exception:
        return {}


def _write_scheduler_status(payload: dict[str, Any]) -> None:
    from core.runtime_paths import REQUIRE_EXISTING_ENV, runtime_path

    path = runtime_path(REPORT_SCHEDULER_REL)
    # Installed-host strict mode means the adopted runtime already owns state/.
    # Never recreate that directory: if external storage vanished, mkdir could
    # silently create split state on the system disk before the watchdog fires.
    strict = str(os.environ.get(REQUIRE_EXISTING_ENV, "")).strip().lower() in {
        "1", "true", "yes", "on",
    }
    if not path.parent.is_dir():
        if strict:
            raise RuntimeError(f"runtime state directory is unavailable: {path.parent}")
        # Unit/dev callers are allowed to initialise an ephemeral runtime tree.
        # product.host_entrypoint.main() always enables strict mode before this
        # scheduler starts, so this branch is unreachable on the installed host.
        path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    os.replace(tmp, path)


def _report_iteration() -> dict[str, Any]:
    from core.market_clock import now_ist
    now = now_ist()
    previous = _read_scheduler_status()
    payload: dict[str, Any] = {
        "schema_version": 1,
        "last_attempt_ist": now.isoformat(),
        "ist_date": now.date().isoformat(),
        "state": "OK",
        "last_error": "",
        "alerted_date": str(previous.get("alerted_date") or ""),
    }
    try:
        from product.host_report_job import run_once
        result = run_once(now=now)
        payload["report_result"] = result
        payload["state"] = "RAN" if result.get("ran") else "NOT_DUE"
    except Exception as exc:
        payload["state"] = "FAILED"
        payload["last_error"] = f"{type(exc).__name__}: report scheduler iteration failed"
        if payload["alerted_date"] != now.date().isoformat():
            try:
                from product.host_alerts import send_operational_alert
                alert = send_operational_alert(
                    "QuantTerm daily operating report scheduler failed\n"
                    f"IST date: {now.date().isoformat()}\n"
                    f"Failure class: {type(exc).__name__}\n"
                    "Read state/daily_report_scheduler.json and service logs."
                ).to_dict()
                payload["alert"] = alert
                if alert.get("attempted"):
                    payload["alerted_date"] = now.date().isoformat()
            except Exception as alert_exc:
                payload["alert"] = {
                    "attempted": True, "delivered": False,
                    "detail": f"{type(alert_exc).__name__}: alert delivery failed",
                }
    _write_scheduler_status(payload)
    return payload


def _report_loop(storage_guard) -> None:
    while not storage_guard.shutdown.is_set():
        if storage_guard.check():
            try:
                _report_iteration()
            except Exception:
                # Scheduler failures are persisted by _report_iteration when the
                # runtime is available. Storage loss is owned by the guard and
                # must not create a fallback state tree here.
                pass
        storage_guard.shutdown.wait(REPORT_POLL_SECONDS)


def main() -> int:
    load_env_file(os.environ.get("QT_HOST_ENV_FILE"))
    storage = prepare_runtime_storage_for_startup()
    if storage != "NOT_REQUIRED":
        print(f"[HOST STORAGE] {storage}", flush=True)
    npm = prepare_frontend_toolchain()
    print(f"[HOST TOOLCHAIN] npm={npm}", flush=True)
    if not os.environ.get("QT_RUNTIME_ROOT", "").strip():
        raise RuntimeError("installed QuantTerm requires QT_RUNTIME_ROOT")
    if not os.environ.get("QT_BUILD_SHA", "").strip():
        raise RuntimeError("installed QuantTerm requires QT_BUILD_SHA")

    # This invariant is inherited by every child. Once the installed service is
    # running, a vanished configured runtime is an error at path resolution
    # time; helpers such as ensure_logs_path are never allowed to mkdir a fresh
    # replacement tree on another filesystem while the watchdog is reacting.
    os.environ["QT_RUNTIME_ROOT_REQUIRE_EXISTING"] = "1"

    from product.runtime_storage_guard import RuntimeStorageGuard

    guard_interval = float(os.environ.get("QT_RUNTIME_STORAGE_WATCH_S", "15") or 15)
    storage_guard = RuntimeStorageGuard(interval_s=guard_interval)
    outer_shutdown = threading.Event()
    loss_reason = {"detail": ""}

    def storage_lost(reason: str) -> None:
        loss_reason["detail"] = str(reason or "runtime storage unavailable")[:500]
        print(
            f"[HOST STORAGE] LOST: {loss_reason['detail']} · stopping all QuantTerm children",
            file=sys.stderr,
            flush=True,
        )
        # host_supervisor.main owns SIGTERM while it is running. Its handler only
        # requests shutdown, so the supervisor's finally block still reaps every
        # child and releases the machine lock cleanly.
        os.kill(os.getpid(), signal.SIGTERM)

    def outer_stop(_signum, _frame) -> None:
        outer_shutdown.set()
        storage_guard.close()

    storage_guard.start(storage_lost)
    threading.Thread(
        target=_report_loop,
        args=(storage_guard,),
        name="quantterm-post-session-report-scheduler",
        daemon=True,
    ).start()

    try:
        from product.host_supervisor import main as supervisor_main

        while not outer_shutdown.is_set():
            rc = int(supervisor_main())
            if not storage_guard.lost.is_set():
                return rc

            # supervisor_main installed its own signal handlers. While children
            # are down and storage is absent, restore an outer handler so a real
            # service stop does not get mistaken for storage recovery.
            signal.signal(signal.SIGTERM, outer_stop)
            signal.signal(signal.SIGINT, outer_stop)
            print(
                "[HOST STORAGE] QuantTerm is paused. Waiting for the same durable runtime to return; no fallback directory will be created.",
                file=sys.stderr,
                flush=True,
            )
            if not storage_guard.wait_until_recovered(
                should_stop=outer_shutdown.is_set,
                prepare=prepare_runtime_storage_for_startup,
            ):
                return 0
            print(
                "[HOST STORAGE] Runtime recovered. Re-starting the canonical supervisor; safety and exact-SHA checks will run again.",
                file=sys.stderr,
                flush=True,
            )
            loss_reason["detail"] = ""
        return 0
    finally:
        storage_guard.close()


if __name__ == "__main__":
    raise SystemExit(main())
