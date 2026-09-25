"""Verified launchd lifecycle control for the canonical QuantTerm host.

The generic installer historically treated ``launchctl bootout`` as best-effort.
That is unsafe for an exact-SHA installed host: dependency mutation or a restart
must not proceed while the previous launchd job is still loaded.  This module
makes service absence/presence an observed post-condition instead of inferring it
from one launchctl return code.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import time
from typing import Sequence

DEFAULT_LABEL = "com.quantterm.desk"
DEFAULT_TIMEOUT_S = 20.0
_ABSENT_MARKERS = (
    "could not find service",
    "service not found",
    "no such process",
    "not found in domain",
)


class LaunchdControlError(RuntimeError):
    pass


def _run(args: list[str], *, timeout: float = 20.0) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(
            args, capture_output=True, text=True, check=False, timeout=timeout,
        )
    except Exception as exc:
        raise LaunchdControlError(
            f"launchctl invocation failed: {type(exc).__name__}: {exc}"
        ) from exc


def domain() -> str:
    return f"gui/{os.getuid()}"


def target(label: str = DEFAULT_LABEL) -> str:
    return f"{domain()}/{label}"


def default_plist(label: str = DEFAULT_LABEL) -> Path:
    return Path.home() / "Library" / "LaunchAgents" / f"{label}.plist"


def _detail(proc: subprocess.CompletedProcess[str]) -> str:
    return (proc.stderr or proc.stdout or "").strip()


def _is_explicit_absence(proc: subprocess.CompletedProcess[str]) -> bool:
    if proc.returncode == 0:
        return False
    text = f"{proc.stdout or ''}\n{proc.stderr or ''}".lower()
    return any(marker in text for marker in _ABSENT_MARKERS)


def query_loaded(label: str = DEFAULT_LABEL) -> tuple[bool, subprocess.CompletedProcess[str]]:
    proc = _run(["launchctl", "print", target(label)], timeout=10.0)
    if proc.returncode == 0:
        return True, proc
    if _is_explicit_absence(proc):
        return False, proc
    raise LaunchdControlError(
        f"cannot establish launchd state for {target(label)}: "
        f"rc={proc.returncode} {_detail(proc)[:500]}"
    )


def stop_verified(
    *,
    label: str = DEFAULT_LABEL,
    timeout_s: float = DEFAULT_TIMEOUT_S,
) -> subprocess.CompletedProcess[str]:
    loaded, before = query_loaded(label)
    if not loaded:
        return before

    bootout = _run(["launchctl", "bootout", target(label)], timeout=20.0)
    deadline = time.monotonic() + max(0.0, float(timeout_s))
    last = before
    while True:
        try:
            loaded, last = query_loaded(label)
        except LaunchdControlError:
            # An opaque launchctl error is not proof of absence.
            if time.monotonic() >= deadline:
                raise
            time.sleep(0.1)
            continue
        if not loaded:
            return bootout
        if time.monotonic() >= deadline:
            break
        time.sleep(0.1)

    raise LaunchdControlError(
        f"launchd stop did not unload {target(label)} within {timeout_s:.1f}s; "
        f"bootout_rc={bootout.returncode} bootout={_detail(bootout)[:300]} "
        f"print={_detail(last)[:300]}"
    )


def _bootstrap_verified(
    *,
    label: str = DEFAULT_LABEL,
    plist: Path | None = None,
    timeout_s: float = DEFAULT_TIMEOUT_S,
) -> subprocess.CompletedProcess[str]:
    plist = Path(plist or default_plist(label)).expanduser()
    if not plist.is_file():
        raise LaunchdControlError(f"launchd plist is missing: {plist}")

    primary = _run(["launchctl", "bootstrap", domain(), str(plist)], timeout=20.0)
    if primary.returncode != 0:
        # Compatibility fallback for older macOS launchctl behavior.  It is only
        # accepted when the same target is subsequently proven loaded.
        fallback = _run(["launchctl", "load", "-w", str(plist)], timeout=20.0)
        if fallback.returncode != 0:
            try:
                loaded, _ = query_loaded(label)
            except LaunchdControlError:
                loaded = False
            if not loaded:
                raise LaunchdControlError(
                    f"cannot load {target(label)}: bootstrap_rc={primary.returncode} "
                    f"bootstrap={_detail(primary)[:250]} load_rc={fallback.returncode} "
                    f"load={_detail(fallback)[:250]}"
                )

    _run(["launchctl", "enable", target(label)], timeout=10.0)
    # bootstrap + RunAtLoad may already have started the host. Do not use -k
    # here: killing a just-started exact-SHA supervisor creates a second service
    # generation and can trip the bounded restart guard during upgrades.
    kick = _run(["launchctl", "kickstart", target(label)], timeout=20.0)

    deadline = time.monotonic() + max(0.0, float(timeout_s))
    while True:
        loaded, current = query_loaded(label)
        if loaded:
            return current if current.returncode == 0 else kick
        if time.monotonic() >= deadline:
            raise LaunchdControlError(
                f"launchd start did not load {target(label)} within {timeout_s:.1f}s; "
                f"kick_rc={kick.returncode} {_detail(kick)[:300]}"
            )
        time.sleep(0.1)


def start_verified(
    *,
    label: str = DEFAULT_LABEL,
    plist: Path | None = None,
    timeout_s: float = DEFAULT_TIMEOUT_S,
) -> subprocess.CompletedProcess[str]:
    loaded, current = query_loaded(label)
    if not loaded:
        return _bootstrap_verified(label=label, plist=plist, timeout_s=timeout_s)
    # "start" is idempotent. If the job is already loaded, ask launchd to
    # start it only if needed; never use -k here because that would turn a start
    # command into an implicit restart and SIGKILL a healthy exact-SHA host.
    kick = _run(["launchctl", "kickstart", target(label)], timeout=20.0)
    loaded, after = query_loaded(label)
    if not loaded:
        raise LaunchdControlError(
            f"launchd target disappeared after kickstart: {target(label)} "
            f"rc={kick.returncode} {_detail(kick)[:300]}"
        )
    return after if after.returncode == 0 else current


def restart_verified(
    *,
    label: str = DEFAULT_LABEL,
    plist: Path | None = None,
    timeout_s: float = DEFAULT_TIMEOUT_S,
) -> subprocess.CompletedProcess[str]:
    stop_verified(label=label, timeout_s=timeout_s)
    return _bootstrap_verified(label=label, plist=plist, timeout_s=timeout_s)


def perform(
    action: str,
    *,
    label: str = DEFAULT_LABEL,
    plist: Path | None = None,
    timeout_s: float = DEFAULT_TIMEOUT_S,
) -> subprocess.CompletedProcess[str]:
    if action == "stop":
        return stop_verified(label=label, timeout_s=timeout_s)
    if action in {"install", "restart"}:
        return restart_verified(label=label, plist=plist, timeout_s=timeout_s)
    if action == "start":
        return start_verified(label=label, plist=plist, timeout_s=timeout_s)
    if action == "status":
        _loaded, proc = query_loaded(label)
        return proc
    raise LaunchdControlError(f"unsupported launchd action: {action}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Verified QuantTerm launchd control")
    parser.add_argument("action", choices=("start", "stop", "restart", "status"))
    parser.add_argument("--label", default=DEFAULT_LABEL)
    parser.add_argument("--plist", type=Path)
    parser.add_argument("--manager", choices=("launchd",), default="launchd")
    parser.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT_S)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        proc = perform(
            args.action,
            label=args.label,
            plist=args.plist,
            timeout_s=args.timeout,
        )
    except LaunchdControlError as exc:
        print(f"[launchd] ERROR: {exc}", flush=True)
        return 1
    detail = _detail(proc)
    if detail:
        print(detail)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
