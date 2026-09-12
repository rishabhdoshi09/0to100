"""Installed-host entrypoint for the QuantTerm paper/shadow desk.

The service manager starts this module, not the interactive launcher. It loads an
optional operator-owned environment file, starts the low-frequency post-session
report scheduler, then hands process ownership to the canonical host supervisor.
No credential value is logged or written back to disk here.
"""
from __future__ import annotations

import os
from pathlib import Path
import threading
import time

REPORT_POLL_SECONDS = 15 * 60


def load_env_file(path: str | os.PathLike[str] | None) -> list[str]:
    """Load simple KEY=VALUE lines without echoing values.

    Existing process variables win, so a service-manager override is never silently
    replaced by a stale file. The return value contains key names only.
    """
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


def _report_loop() -> None:
    """Poll the idempotent report job without creating a second scheduler owner.

    ``host_report_job.run_once`` decides whether a report is actually due and
    de-duplicates by IST date. This loop only supplies a coarse wake-up cadence.
    A report failure is intentionally non-fatal to the desk: the report job
    persists its own evidence and operational alerts cover critical results.
    """
    while True:
        try:
            from product.host_report_job import run_once

            run_once()
        except Exception:
            pass
        time.sleep(REPORT_POLL_SECONDS)


def main() -> int:
    load_env_file(os.environ.get("QT_HOST_ENV_FILE"))
    if not os.environ.get("QT_RUNTIME_ROOT", "").strip():
        raise RuntimeError("installed QuantTerm requires QT_RUNTIME_ROOT")
    if not os.environ.get("QT_BUILD_SHA", "").strip():
        raise RuntimeError("installed QuantTerm requires QT_BUILD_SHA")

    threading.Thread(
        target=_report_loop,
        name="quantterm-post-session-report-scheduler",
        daemon=True,
    ).start()

    from product.host_supervisor import main as supervisor_main

    return int(supervisor_main())


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
