"""macOS launchd-owned wrapper for the installed QuantTerm host.

launchd must own the Python host process directly.  ``caffeinate`` is only a
companion tied to this PID; it is not the service-manager parent.  Before the
host starts, a dead prior supervisor generation may be reconciled only when its
durable ownership and process facts prove every candidate child is ours.
"""
from __future__ import annotations

import os
import subprocess
import sys

from product import host_entrypoint


def _start_caffeinate_companion() -> subprocess.Popen[bytes] | None:
    path = "/usr/bin/caffeinate"
    if sys.platform != "darwin" or not os.path.isfile(path):
        return None
    try:
        return subprocess.Popen(
            [path, "-i", "-w", str(os.getpid())],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=False,
        )
    except Exception as exc:
        raise RuntimeError(
            f"cannot start macOS idle-sleep guard: {type(exc).__name__}: {exc}"
        ) from exc


def _stop_companion(proc: subprocess.Popen[bytes] | None) -> None:
    if proc is None or proc.poll() is not None:
        return
    try:
        proc.terminate()
        proc.wait(timeout=2)
    except Exception:
        try:
            proc.kill()
            proc.wait(timeout=1)
        except Exception:
            pass


def main() -> int:
    host_entrypoint.load_env_file(os.environ.get("QT_HOST_ENV_FILE"))
    storage = host_entrypoint.prepare_runtime_storage_for_startup()
    if storage != "NOT_REQUIRED":
        print(f"[HOST STORAGE] {storage}", flush=True)

    if not os.environ.get("QT_RUNTIME_ROOT", "").strip():
        raise RuntimeError("installed QuantTerm requires QT_RUNTIME_ROOT")
    if not os.environ.get("QT_BUILD_SHA", "").strip():
        raise RuntimeError("installed QuantTerm requires QT_BUILD_SHA")

    # Once storage identity is proven, all recovery reads/writes must fail closed
    # if the configured runtime disappears.
    os.environ["QT_RUNTIME_ROOT_REQUIRE_EXISTING"] = "1"

    from product.host_orphan_recovery import OrphanRecoveryError, reconcile_previous_children

    try:
        recovery = reconcile_previous_children()
    except OrphanRecoveryError as exc:
        # Successful exit is intentional: launchd's KeepAlive SuccessfulExit=false
        # must not turn an ownership conflict into another restart storm.
        print(f"[HOST OWNERSHIP] BLOCKED: {exc}", file=sys.stderr, flush=True)
        return 0
    print(f"[HOST OWNERSHIP] {recovery.get('state')}", flush=True)

    companion = _start_caffeinate_companion()
    try:
        # Existing entrypoint re-runs the storage preflight intentionally; it is
        # the recovery authority after a later removable-storage interruption.
        return int(host_entrypoint.main())
    finally:
        _stop_companion(companion)


if __name__ == "__main__":
    raise SystemExit(main())
