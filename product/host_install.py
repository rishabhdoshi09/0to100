"""Turn a validated QuantTerm checkout into an always-on paper/shadow host.

This module deliberately keeps *code* and *runtime evidence* separate.  The git
checkout may move between validated SHAs; accumulated scans, databases, paper
positions, provenance and reports live under one persistent ``QT_RUNTIME_ROOT``.

The installer is conservative:

* repo-local state is copied, never deleted;
* two divergent durable roots are a split-brain error, never auto-merged;
* the deployed service is pinned to the exact git SHA it was installed from;
* the live-execution interlock and real-market preflight must pass before start;
* systemd/launchd own one top-level host supervisor, and that supervisor owns all
  QuantTerm children.

No credential value is rendered into status output.  An optional operator-owned
``QT_HOST_ENV_FILE`` is referenced by path only.
"""
from __future__ import annotations

import argparse
import hashlib
import html
import json
import os
import platform
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

from core.runtime_paths import REPO_ROOT

SERVICE_NAME = "quantterm"
LAUNCHD_LABEL = "com.quantterm.desk"
SCHEMA_VERSION = 1
EPHEMERAL_PREFIXES = ("/tmp/", "/var/tmp/", "/dev/shm/")
DURABLE_DIRS = (
    "logs",
    "db",
    "state",
    "cache",
    "evidence",
    "paper",
    "scans",
    "history",
    "provenance",
    "reports",
)
MANIFEST_REL = Path("state") / "host_runtime_manifest.json"
DEPLOYMENT_REL = Path("state") / "host_deployment.json"
SUPERVISOR_REL = Path("state") / "host_supervisor.json"


class HostInstallError(RuntimeError):
    pass


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _run(args: list[str], *, check: bool = True, timeout: float = 30.0) -> subprocess.CompletedProcess:
    return subprocess.run(args, capture_output=True, text=True, check=check, timeout=timeout)


def git_sha(repo_root: Path = REPO_ROOT) -> str:
    try:
        return _run(["git", "-C", str(repo_root), "rev-parse", "HEAD"], timeout=8).stdout.strip()
    except Exception as exc:
        raise HostInstallError(f"cannot establish git SHA: {type(exc).__name__}") from exc


def ensure_clean_checkout(repo_root: Path = REPO_ROOT) -> None:
    try:
        dirty = _run(["git", "-C", str(repo_root), "status", "--porcelain"], timeout=8).stdout.strip()
    except Exception as exc:
        raise HostInstallError(f"cannot inspect checkout: {type(exc).__name__}") from exc
    if dirty:
        raise HostInstallError("installed host requires a clean exact-SHA checkout")


def default_runtime_root(*, home: Path | None = None, system: str | None = None) -> Path:
    home = Path(home or Path.home()).expanduser()
    system = (system or platform.system()).lower()
    if system == "darwin":
        return home / "Library" / "Application Support" / "QuantTerm" / "runtime"
    state_home = os.environ.get("XDG_STATE_HOME", "").strip()
    base = Path(state_home).expanduser() if state_home else home / ".local" / "state"
    return base / "quantterm" / "runtime"


def _resolved(path: Path) -> Path:
    return Path(path).expanduser().resolve()


def _under(path: Path, parent: Path) -> bool:
    try:
        _resolved(path).relative_to(_resolved(parent))
        return True
    except ValueError:
        return False


def ensure_persistent_runtime_root(path: Path, *, repo_root: Path = REPO_ROOT) -> Path:
    root = _resolved(path)
    if root == _resolved(repo_root) or _under(root, repo_root):
        raise HostInstallError("QT_RUNTIME_ROOT must be outside the source checkout")
    text = str(root) + "/"
    if text.startswith(EPHEMERAL_PREFIXES):
        raise HostInstallError(f"QT_RUNTIME_ROOT is on an ephemeral filesystem: {root}")
    root.mkdir(parents=True, exist_ok=True)
    probe = root / ".quantterm_write_probe"
    try:
        probe.write_text("ok", encoding="utf-8")
        probe.unlink()
    except Exception as exc:
        raise HostInstallError(f"QT_RUNTIME_ROOT is not writable: {root}: {exc}") from exc
    return root


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def inventory(root: Path, dirs: Iterable[str] = DURABLE_DIRS) -> dict[str, dict[str, Any]]:
    root = Path(root)
    result: dict[str, dict[str, Any]] = {}
    for dirname in dirs:
        base = root / dirname
        if not base.exists():
            continue
        for path in sorted(p for p in base.rglob("*") if p.is_file() and not p.is_symlink()):
            rel = path.relative_to(root).as_posix()
            result[rel] = {"size": path.stat().st_size, "sha256": _sha256(path)}
    return result


def inventory_digest(rows: Mapping[str, Mapping[str, Any]]) -> str:
    blob = json.dumps(rows, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(dict(payload), indent=2, default=str), encoding="utf-8")
    os.replace(tmp, path)


def migrate_repo_runtime(
    target_root: Path,
    *,
    repo_root: Path = REPO_ROOT,
    build_sha: str = "",
) -> dict[str, Any]:
    """Adopt a persistent root without guessing between divergent histories.

    The first successful migration writes a durable manifest.  Once that marker
    exists, the persistent root is authoritative and later repo-local files are
    deliberately ignored; otherwise every software upgrade would look like a new
    split-brain after the real host had accumulated newer evidence.
    """
    target = ensure_persistent_runtime_root(target_root, repo_root=repo_root)
    marker = target / MANIFEST_REL
    if marker.exists():
        try:
            payload = json.loads(marker.read_text(encoding="utf-8"))
        except Exception as exc:
            raise HostInstallError(f"runtime manifest is unreadable: {marker}: {exc}") from exc
        if not isinstance(payload, dict) or not payload.get("initialized_at"):
            raise HostInstallError(f"runtime manifest is invalid: {marker}")
        return {"state": "ADOPTED", "runtime_root": str(target), "manifest": payload}

    source_rows = inventory(repo_root)
    target_rows = inventory(target)
    if source_rows and target_rows and source_rows != target_rows:
        src_only = sorted(set(source_rows) - set(target_rows))[:10]
        dst_only = sorted(set(target_rows) - set(source_rows))[:10]
        conflicts = sorted(
            key for key in set(source_rows) & set(target_rows)
            if source_rows[key] != target_rows[key]
        )[:10]
        raise HostInstallError(
            "split-brain runtime state: repo-local and persistent roots both contain "
            f"different durable data (source_only={src_only}, target_only={dst_only}, "
            f"conflicts={conflicts}); reconcile explicitly before install"
        )

    copied = 0
    if source_rows and not target_rows:
        for rel in sorted(source_rows):
            src = repo_root / rel
            dst = target / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
            copied += 1
        verified = inventory(target)
        if verified != source_rows:
            raise HostInstallError("runtime migration verification failed; source was left untouched")
    elif source_rows == target_rows and source_rows:
        copied = 0

    payload = {
        "schema_version": SCHEMA_VERSION,
        "initialized_at": _now(),
        "source_root": str(_resolved(repo_root)),
        "runtime_root": str(target),
        "build_sha": build_sha or git_sha(repo_root),
        "source_files": len(source_rows),
        "copied_files": copied,
        "source_inventory_sha256": inventory_digest(source_rows),
        "source_preserved": True,
    }
    _atomic_json(marker, payload)
    return {"state": "MIGRATED" if copied else "INITIALIZED", "runtime_root": str(target), "manifest": payload}


def _safe_env_file(path: Path | None) -> str:
    if path is None:
        return ""
    target = _resolved(path)
    if not target.exists() or not target.is_file():
        raise HostInstallError(f"environment file does not exist: {target}")
    try:
        mode = target.stat().st_mode & 0o777
        if mode & 0o077:
            raise HostInstallError(
                f"environment file may contain credentials and must not be group/world readable: "
                f"{target} mode={oct(mode)}; chmod 600 it first"
            )
    except OSError as exc:
        raise HostInstallError(f"cannot inspect environment file permissions: {target}: {exc}") from exc
    return str(target)


def _systemd_quote(value: str) -> str:
    return value.replace("\\", "\\\\").replace('"', '\\"')


def render_systemd_unit(
    *, repo_root: Path, runtime_root: Path, python: str, build_sha: str, env_file: str = ""
) -> str:
    lines = [
        "[Unit]",
        "Description=QuantTerm paper/shadow trading desk",
        "After=network-online.target",
        "Wants=network-online.target",
        "StartLimitIntervalSec=300",
        "StartLimitBurst=5",
        "",
        "[Service]",
        "Type=simple",
        f'WorkingDirectory="{_systemd_quote(str(_resolved(repo_root)))}"',
        f'Environment="PYTHONPATH={_systemd_quote(str(_resolved(repo_root)))}"',
        f'Environment="QT_RUNTIME_ROOT={_systemd_quote(str(_resolved(runtime_root)))}"',
        f'Environment="QT_BUILD_SHA={_systemd_quote(build_sha)}"',
    ]
    if env_file:
        lines.append(f'Environment="QT_HOST_ENV_FILE={_systemd_quote(env_file)}"')
    lines += [
        f'ExecStart="{_systemd_quote(python)}" -u -m product.host_entrypoint',
        "Restart=on-failure",
        "RestartSec=5",
        "TimeoutStopSec=30",
        "KillMode=control-group",
        "",
        "[Install]",
        "WantedBy=default.target",
        "",
    ]
    return "\n".join(lines)


def render_launchd_plist(
    *, repo_root: Path, runtime_root: Path, python: str, build_sha: str, env_file: str = ""
) -> str:
    def esc(value: str) -> str:
        return html.escape(value, quote=True)

    service_logs = _resolved(runtime_root) / "logs" / "service"
    env_entries = {
        "PYTHONPATH": str(_resolved(repo_root)),
        "QT_RUNTIME_ROOT": str(_resolved(runtime_root)),
        "QT_BUILD_SHA": build_sha,
    }
    if env_file:
        env_entries["QT_HOST_ENV_FILE"] = env_file
    env_xml = "\n".join(
        f"      <key>{esc(key)}</key><string>{esc(value)}</string>" for key, value in env_entries.items()
    )
    return f'''<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key><string>{LAUNCHD_LABEL}</string>
  <key>ProgramArguments</key>
  <array>
    <string>{esc(python)}</string><string>-u</string><string>-m</string><string>product.host_entrypoint</string>
  </array>
  <key>WorkingDirectory</key><string>{esc(str(_resolved(repo_root)))}</string>
  <key>EnvironmentVariables</key>
  <dict>
{env_xml}
  </dict>
  <key>RunAtLoad</key><true/>
  <key>KeepAlive</key><true/>
  <key>ThrottleInterval</key><integer>5</integer>
  <key>ProcessType</key><string>Background</string>
  <key>StandardOutPath</key><string>{esc(str(service_logs / "launchd.out.log"))}</string>
  <key>StandardErrorPath</key><string>{esc(str(service_logs / "launchd.err.log"))}</string>
</dict>
</plist>
'''


def service_manager(*, system: str | None = None) -> str:
    system = (system or platform.system()).lower()
    if system == "darwin" and shutil.which("launchctl"):
        return "launchd"
    if shutil.which("systemctl"):
        return "systemd"
    raise HostInstallError("no supported service manager found (systemd or launchd)")


def _service_paths(manager: str, *, home: Path | None = None) -> tuple[Path, str]:
    home = Path(home or Path.home()).expanduser()
    if manager == "systemd":
        return home / ".config" / "systemd" / "user" / f"{SERVICE_NAME}.service", SERVICE_NAME
    if manager == "launchd":
        return home / "Library" / "LaunchAgents" / f"{LAUNCHD_LABEL}.plist", LAUNCHD_LABEL
    raise HostInstallError(f"unsupported service manager: {manager}")


def install_service_definition(
    *, runtime_root: Path, build_sha: str, env_file: str = "", repo_root: Path = REPO_ROOT,
    python: str | None = None, manager: str | None = None, home: Path | None = None,
) -> dict[str, Any]:
    manager = manager or service_manager()
    python = python or sys.executable
    path, label = _service_paths(manager, home=home)
    path.parent.mkdir(parents=True, exist_ok=True)
    content = (
        render_systemd_unit(repo_root=repo_root, runtime_root=runtime_root, python=python,
                            build_sha=build_sha, env_file=env_file)
        if manager == "systemd"
        else render_launchd_plist(repo_root=repo_root, runtime_root=runtime_root, python=python,
                                  build_sha=build_sha, env_file=env_file)
    )
    previous = path.read_text(encoding="utf-8") if path.exists() else None
    backup = path.with_suffix(path.suffix + ".previous")
    if previous is not None:
        backup.write_text(previous, encoding="utf-8")
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(content, encoding="utf-8")
    os.replace(tmp, path)
    return {"manager": manager, "path": str(path), "label": label,
            "backup": str(backup) if previous is not None else ""}


def _systemctl_action(action: str) -> subprocess.CompletedProcess:
    command = ["systemctl", "--user"]
    if action == "install":
        _run(command + ["daemon-reload"], timeout=20)
        _run(command + ["enable", "--now", f"{SERVICE_NAME}.service"], timeout=30)
        return _run(command + ["status", "--no-pager", f"{SERVICE_NAME}.service"], check=False, timeout=20)
    verb = {"start": "start", "stop": "stop", "restart": "restart", "status": "status"}[action]
    return _run(command + [verb, "--no-pager", f"{SERVICE_NAME}.service"] if verb == "status"
                else command + [verb, f"{SERVICE_NAME}.service"], check=action != "status", timeout=30)


def _launchd_action(action: str, plist: Path) -> subprocess.CompletedProcess:
    uid = os.getuid()
    domain = f"gui/{uid}"
    target = f"{domain}/{LAUNCHD_LABEL}"
    if action == "install":
        _run(["launchctl", "bootout", target], check=False, timeout=15)
        try:
            _run(["launchctl", "bootstrap", domain, str(plist)], timeout=20)
        except Exception:
            _run(["launchctl", "load", "-w", str(plist)], timeout=20)
        _run(["launchctl", "enable", target], check=False, timeout=10)
        return _run(["launchctl", "print", target], check=False, timeout=20)
    if action == "start":
        return _run(["launchctl", "kickstart", "-k", target], timeout=20)
    if action == "restart":
        return _run(["launchctl", "kickstart", "-k", target], timeout=20)
    if action == "stop":
        return _run(["launchctl", "kill", "SIGTERM", target], check=False, timeout=20)
    return _run(["launchctl", "print", target], check=False, timeout=20)


def service_action(action: str, *, manager: str | None = None, home: Path | None = None) -> subprocess.CompletedProcess:
    manager = manager or service_manager()
    path, _ = _service_paths(manager, home=home)
    if manager == "systemd":
        return _systemctl_action(action)
    return _launchd_action(action, path)


def deployment_manifest(runtime_root: Path) -> dict[str, Any]:
    path = _resolved(runtime_root) / DEPLOYMENT_REL
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return payload if isinstance(payload, dict) else {}
    except Exception:
        return {}


def write_deployment_manifest(
    runtime_root: Path, *, build_sha: str, manager: str, service_path: str, env_file: str = ""
) -> Path:
    path = _resolved(runtime_root) / DEPLOYMENT_REL
    previous = deployment_manifest(runtime_root)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "deployed_at": _now(),
        "build_sha": build_sha,
        "previous_build_sha": str(previous.get("build_sha") or ""),
        "runtime_root": str(_resolved(runtime_root)),
        "repo_root": str(_resolved(REPO_ROOT)),
        "service_manager": manager,
        "service_path": service_path,
        "env_file": env_file,
        "live_mode": "PAPER_SHADOW_ONLY",
    }
    _atomic_json(path, payload)
    return path


def _prepare_process_environment(runtime_root: Path, build_sha: str, env_file: str = "") -> None:
    os.environ["QT_RUNTIME_ROOT"] = str(_resolved(runtime_root))
    os.environ["QT_BUILD_SHA"] = build_sha
    if env_file:
        os.environ["QT_HOST_ENV_FILE"] = env_file
        from product.host_entrypoint import load_env_file
        load_env_file(env_file)


def run_required_preflight(runtime_root: Path, build_sha: str, env_file: str = "") -> dict[str, Any]:
    _prepare_process_environment(runtime_root, build_sha, env_file)
    from product.host_preflight import READY, run_host_preflight

    report = run_host_preflight(skip_network=False)
    if report.get("verdict") != READY:
        blockers = "; ".join(f"{b.get('check')}: {b.get('detail')}" for b in report.get("blockers") or [])
        raise HostInstallError(f"host preflight blocked installation: {blockers or 'unknown blocker'}")
    return report


def wait_for_supervisor(runtime_root: Path, *, timeout_s: float = 45.0) -> dict[str, Any]:
    path = _resolved(runtime_root) / SUPERVISOR_REL
    deadline = time.monotonic() + timeout_s
    last: dict[str, Any] = {}
    while time.monotonic() < deadline:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(payload, dict):
                last = payload
                state = str(payload.get("state") or "")
                if state == "RUNNING":
                    return payload
                if state in {"FAILED", "STOPPED"}:
                    break
        except Exception:
            pass
        time.sleep(0.5)
    raise HostInstallError(
        f"installed supervisor did not reach RUNNING within {timeout_s:.0f}s; "
        f"last_state={last.get('state') or 'missing'} error={last.get('error') or ''}"
    )


def install_host(
    *, runtime_root: Path | None = None, env_file: Path | None = None,
    manager: str | None = None, start: bool = True,
) -> dict[str, Any]:
    ensure_clean_checkout(REPO_ROOT)
    sha = git_sha(REPO_ROOT)
    root = ensure_persistent_runtime_root(runtime_root or default_runtime_root())
    migration = migrate_repo_runtime(root, repo_root=REPO_ROOT, build_sha=sha)
    env_path = _safe_env_file(env_file) if env_file else ""
    preflight = run_required_preflight(root, sha, env_path)
    selected = manager or service_manager()
    definition = install_service_definition(
        runtime_root=root, build_sha=sha, env_file=env_path, manager=selected,
    )
    write_deployment_manifest(
        root, build_sha=sha, manager=selected, service_path=definition["path"], env_file=env_path,
    )
    result: dict[str, Any] = {
        "ok": True, "build_sha": sha, "runtime_root": str(root),
        "migration": migration, "preflight": preflight, "service": definition,
        "started": False,
    }
    if not start:
        return result
    try:
        service_action("install", manager=selected)
        running = wait_for_supervisor(root)
    except Exception:
        # Restore the previous service definition if this was an upgrade.  Durable
        # runtime data is never rolled back or deleted here.
        backup = Path(definition.get("backup") or "") if definition.get("backup") else None
        service_path = Path(definition["path"])
        if backup is not None and backup.exists():
            shutil.copy2(backup, service_path)
            try:
                service_action("install", manager=selected)
            except Exception:
                pass
        raise
    result["started"] = True
    result["supervisor"] = running
    return result


def host_status(runtime_root: Path | None = None, *, manager: str | None = None) -> dict[str, Any]:
    root = _resolved(runtime_root or Path(os.environ.get("QT_RUNTIME_ROOT") or default_runtime_root()))
    deployment = deployment_manifest(root)
    supervisor: dict[str, Any] = {}
    try:
        raw = json.loads((root / SUPERVISOR_REL).read_text(encoding="utf-8"))
        if isinstance(raw, dict):
            supervisor = raw
    except Exception:
        supervisor = {}
    selected = manager or str(deployment.get("service_manager") or "")
    service = {"manager": selected, "known": bool(selected), "returncode": None, "detail": ""}
    if selected:
        try:
            proc = service_action("status", manager=selected)
            service.update(returncode=proc.returncode, detail=(proc.stdout or proc.stderr)[-2000:])
        except Exception as exc:
            service.update(returncode=1, detail=f"{type(exc).__name__}: {exc}")
    return {
        "runtime_root": str(root),
        "deployment": deployment,
        "supervisor": supervisor,
        "service": service,
        "paper_shadow_only": True,
    }


def render_status(payload: Mapping[str, Any]) -> str:
    dep = payload.get("deployment") or {}
    sup = payload.get("supervisor") or {}
    service = payload.get("service") or {}
    children = sup.get("children") or {}
    alive = [name for name, row in children.items() if isinstance(row, dict) and row.get("alive")]
    return "\n".join([
        "QUANTTERM HOST STATUS",
        f"runtime_root   {payload.get('runtime_root')}",
        f"deployed_sha   {dep.get('build_sha') or 'UNKNOWN'}",
        f"service        {service.get('manager') or 'UNKNOWN'} rc={service.get('returncode')}",
        f"supervisor     {sup.get('state') or 'MISSING'} pid={sup.get('pid') or '-'}",
        f"heartbeat      {sup.get('heartbeat_at') or '-'}",
        f"children       {', '.join(alive) if alive else 'none evidenced'}",
        f"live_mode      PAPER/SHADOW ONLY",
    ])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Install/control the QuantTerm always-on paper host")
    sub = parser.add_subparsers(dest="command", required=True)

    install = sub.add_parser("install")
    install.add_argument("--runtime-root", type=Path)
    install.add_argument("--env-file", type=Path)
    install.add_argument("--manager", choices=("systemd", "launchd"))
    install.add_argument("--no-start", action="store_true")
    install.add_argument("--json", action="store_true")

    status = sub.add_parser("status")
    status.add_argument("--runtime-root", type=Path)
    status.add_argument("--manager", choices=("systemd", "launchd"))
    status.add_argument("--json", action="store_true")

    for name in ("start", "stop", "restart"):
        action = sub.add_parser(name)
        action.add_argument("--manager", choices=("systemd", "launchd"))

    migrate = sub.add_parser("migrate")
    migrate.add_argument("--runtime-root", type=Path, required=True)
    migrate.add_argument("--json", action="store_true")

    args = parser.parse_args(argv)
    try:
        if args.command == "install":
            payload = install_host(
                runtime_root=args.runtime_root, env_file=args.env_file,
                manager=args.manager, start=not args.no_start,
            )
            print(json.dumps(payload, indent=2, default=str) if args.json else (
                f"HOST INSTALLED\nsha={payload['build_sha']}\nruntime={payload['runtime_root']}\n"
                f"service={payload['service']['manager']}\nstarted={payload['started']}"
            ))
            return 0
        if args.command == "status":
            payload = host_status(args.runtime_root, manager=args.manager)
            print(json.dumps(payload, indent=2, default=str) if args.json else render_status(payload))
            return 0 if (payload.get("supervisor") or {}).get("state") == "RUNNING" else 1
        if args.command == "migrate":
            payload = migrate_repo_runtime(args.runtime_root, build_sha=git_sha())
            print(json.dumps(payload, indent=2, default=str) if args.json else (
                f"RUNTIME {payload['state']} -> {payload['runtime_root']}"
            ))
            return 0
        proc = service_action(args.command, manager=args.manager)
        if proc.stdout:
            print(proc.stdout.rstrip())
        if proc.stderr:
            print(proc.stderr.rstrip(), file=sys.stderr)
        return int(proc.returncode)
    except HostInstallError as exc:
        print(f"HOST INSTALL BLOCKED: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
