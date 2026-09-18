"""Strict macOS installer adapter with verified launchd lifecycle semantics.

This layers two release-safety changes on top of the already-audited
``host_install_existing`` runtime pinning:
1. launchd directly owns Python's host wrapper instead of ``caffeinate``;
2. every launchd stop/restart/install proves the requested post-condition.
"""
from __future__ import annotations

from pathlib import Path
import subprocess
import sys
from typing import Sequence

from product import host_install as HI
from product import host_install_existing as legacy
from product import launchd_control

_ORIGINAL_RENDER = HI.render_launchd_plist
_ORIGINAL_LAUNCHD_ACTION = HI._launchd_action


def _render_launchd_plist_direct_host(**kwargs) -> str:
    text = _ORIGINAL_RENDER(**kwargs)
    caffeinate = "    <string>/usr/bin/caffeinate</string><string>-i</string>\n"
    old_module = "<string>product.host_entrypoint</string>"
    new_module = "<string>product.host_launchd_entrypoint</string>"
    if caffeinate not in text or old_module not in text:
        raise HI.HostInstallError(
            "launchd plist template changed; refusing an unverified lifecycle rewrite"
        )
    text = text.replace(caffeinate, "", 1)
    text = text.replace(old_module, new_module, 1)
    return text


def _verified_launchd_action(action: str, plist: Path) -> subprocess.CompletedProcess[str]:
    try:
        return launchd_control.perform(
            action,
            label=HI.LAUNCHD_LABEL,
            plist=plist,
        )
    except launchd_control.LaunchdControlError as exc:
        raise HI.HostInstallError(str(exc)) from exc


def main(argv: Sequence[str] | None = None) -> int:
    # legacy.main patches additional runtime-write functions and restores them
    # itself.  We patch only launchd rendering/control around that strict scope.
    HI.render_launchd_plist = _render_launchd_plist_direct_host
    HI._launchd_action = _verified_launchd_action
    try:
        return int(legacy.main(list(sys.argv[1:] if argv is None else argv)))
    finally:
        HI.render_launchd_plist = _ORIGINAL_RENDER
        HI._launchd_action = _ORIGINAL_LAUNCHD_ACTION


if __name__ == "__main__":
    raise SystemExit(main())
