from __future__ import annotations

import json
import shutil
import subprocess
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest

import core.runtime_paths as RP
import product.host_install as HI


@pytest.fixture(autouse=True)
def _allow_tmp_runtime(monkeypatch):
    monkeypatch.setattr(HI, "EPHEMERAL_PREFIXES", ("/not-a-real-prefix/",))


def _children(*, healthy=True):
    return {
        name: {"alive": True, "healthy": healthy, "health_failures": 0}
        for name in HI.EXPECTED_CHILDREN
    }


def test_systemd_working_directory_is_valid_path_directive(tmp_path):
    analyze = shutil.which("systemd-analyze")
    if not analyze:
        pytest.skip("systemd-analyze unavailable")
    repo = tmp_path / "repo with spaces"
    runtime = tmp_path / "runtime with spaces"
    repo.mkdir()
    runtime.mkdir()
    unit = tmp_path / "quantterm.service"
    unit.write_text(HI.render_systemd_unit(
        repo_root=repo, runtime_root=runtime,
        python=sys.executable, build_sha="deadbeef",
    ), encoding="utf-8")
    proc = subprocess.run([analyze, "verify", str(unit)], capture_output=True, text=True)
    diagnostic = (proc.stdout + proc.stderr).lower()
    assert "workingdirectory= path is not absolute" not in diagnostic
    assert "fatal error" not in diagnostic
    assert proc.returncode == 0, diagnostic


def test_systemd_install_enables_linger_then_restarts_existing_service(monkeypatch):
    calls: list[list[str]] = []

    def fake_run(args, *, check=True, timeout=30.0):
        calls.append(list(args))
        stdout = "yes\n" if "show-user" in args else ""
        return subprocess.CompletedProcess(args, 0, stdout, "")

    monkeypatch.setattr(HI, "_run", fake_run)
    monkeypatch.setattr(HI.shutil, "which", lambda name: f"/usr/bin/{name}" if name == "loginctl" else None)
    HI._systemctl_action("install")

    assert any("enable-linger" in call for call in calls)
    assert ["systemctl", "--user", "enable", "quantterm.service"] in calls
    assert ["systemctl", "--user", "restart", "quantterm.service"] in calls
    assert not any("--now" in call for call in calls)


def test_wait_for_supervisor_rejects_preexisting_running_marker(tmp_path):
    root = tmp_path / "runtime"
    path = root / HI.SUPERVISOR_REL
    path.parent.mkdir(parents=True)
    old = datetime.now(timezone.utc) - timedelta(minutes=10)
    path.write_text(json.dumps({
        "state": "RUNNING",
        "production_sha": "newsha",
        "started_at": old.isoformat(),
        "heartbeat_at": datetime.now(timezone.utc).isoformat(),
        "children": _children(),
    }), encoding="utf-8")
    with pytest.raises(HI.HostInstallError, match="did not prove RUNNING"):
        HI.wait_for_supervisor(root, expected_sha="newsha", started_after=time.time(), timeout_s=0.02)


def test_wait_for_supervisor_requires_exact_sha_and_healthy_children(tmp_path):
    root = tmp_path / "runtime"
    path = root / HI.SUPERVISOR_REL
    path.parent.mkdir(parents=True)
    started = time.time() - 0.1
    path.write_text(json.dumps({
        "state": "RUNNING",
        "production_sha": "newsha",
        "started_at": datetime.now(timezone.utc).isoformat(),
        "heartbeat_at": datetime.now(timezone.utc).isoformat(),
        "children": _children(),
    }), encoding="utf-8")
    result = HI.wait_for_supervisor(root, expected_sha="newsha", started_after=started, timeout_s=0.5)
    assert result["production_sha"] == "newsha"


def test_launchd_definition_materializes_log_parent(tmp_path):
    runtime = tmp_path / "persistent"
    home = tmp_path / "home"
    repo = tmp_path / "repo"
    repo.mkdir()
    HI.install_service_definition(
        runtime_root=runtime, build_sha="abc", manager="launchd",
        repo_root=repo, python=sys.executable, home=home,
    )
    assert runtime.joinpath("logs", "service").is_dir()


def test_launchd_stop_boots_out_keepalive_job(monkeypatch, tmp_path):
    calls: list[list[str]] = []

    def fake_run(args, *, check=True, timeout=30.0):
        calls.append(list(args))
        return subprocess.CompletedProcess(args, 0, "", "")

    monkeypatch.setattr(HI, "_run", fake_run)
    monkeypatch.setattr(HI.os, "getuid", lambda: 501)
    HI._launchd_action("stop", tmp_path / "agent.plist")
    assert any(call[:2] == ["launchctl", "bootout"] for call in calls)
    assert not any("kill" in call for call in calls)


def test_migration_pointer_redirects_cli_after_environment_is_unset(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    runtime = tmp_path / "runtime"
    repo.mkdir()
    source = repo / "logs" / "product" / "state.json"
    source.parent.mkdir(parents=True)
    source.write_text("evidence", encoding="utf-8")

    HI.migrate_repo_runtime(runtime, repo_root=repo, build_sha="abc")
    assert (repo / RP.POINTER_NAME).read_text().strip() == str(runtime.resolve())

    monkeypatch.delenv(RP.ENV_VAR, raising=False)
    monkeypatch.setattr(RP, "REPO_ROOT", repo)
    assert RP.runtime_root() == runtime.resolve()
    assert RP.logs_path("x.json") == runtime.resolve() / "logs" / "x.json"

    override = tmp_path / "override"
    monkeypatch.setenv(RP.ENV_VAR, str(override))
    assert RP.runtime_root() == override


def test_status_rejects_stale_running_marker(tmp_path, monkeypatch):
    root = tmp_path / "runtime"
    path = root / HI.SUPERVISOR_REL
    path.parent.mkdir(parents=True)
    old = datetime.now(timezone.utc) - timedelta(minutes=10)
    path.write_text(json.dumps({
        "state": "RUNNING", "pid": 999,
        "heartbeat_at": old.isoformat(), "children": _children(),
    }), encoding="utf-8")
    (root / HI.DEPLOYMENT_REL).write_text(json.dumps({"service_manager": "systemd", "build_sha": "abc"}))
    monkeypatch.setattr(HI, "service_action", lambda *a, **k: subprocess.CompletedProcess([], 0, "active", ""))
    monkeypatch.setattr(HI, "_read_live_safety", lambda: {
        "readable": True, "locked": True, "verified": True, "authorized": False,
        "status": "LOCKED", "safe": True,
    })
    status = HI.host_status(root)
    assert status["effective_state"] == "STALE"
    assert status["healthy"] is False


def test_status_does_not_hardcode_paper_shadow_when_interlock_authorized(tmp_path, monkeypatch):
    root = tmp_path / "runtime"
    path = root / HI.SUPERVISOR_REL
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({
        "state": "RUNNING", "pid": 10,
        "heartbeat_at": datetime.now(timezone.utc).isoformat(),
        "children": _children(),
    }), encoding="utf-8")
    (root / HI.DEPLOYMENT_REL).write_text(json.dumps({"service_manager": "systemd", "build_sha": "abc"}))
    monkeypatch.setattr(HI, "service_action", lambda *a, **k: subprocess.CompletedProcess([], 0, "active", ""))
    monkeypatch.setattr(HI, "_read_live_safety", lambda: {
        "readable": True, "locked": False, "verified": True, "authorized": True,
        "status": "AUTHORIZED", "safe": False,
    })
    status = HI.host_status(root)
    text = HI.render_status(status)
    assert "authorized=True" in text
    assert "PAPER/SHADOW ONLY" not in text
    assert "UNHEALTHY" in text


def test_preflight_queries_systemd_user_scope_and_linger(monkeypatch):
    import product.host_preflight as HP
    calls: list[list[str]] = []
    monkeypatch.setattr(HP.shutil, "which", lambda name: f"/usr/bin/{name}" if name in {"systemctl", "loginctl"} else None)

    def fake_run(args, **kwargs):
        calls.append(list(args))
        if "is-enabled" in args:
            return subprocess.CompletedProcess(args, 0, "enabled\n", "")
        if "is-active" in args:
            return subprocess.CompletedProcess(args, 0, "active\n", "")
        if "show-user" in args:
            return subprocess.CompletedProcess(args, 0, "yes\n", "")
        return subprocess.CompletedProcess(args, 0, "", "")

    monkeypatch.setattr(HP.subprocess, "run", fake_run)
    check = HP.check_service_installation()
    systemd_calls = [c for c in calls if c and c[0].endswith("systemctl")]
    assert systemd_calls and all("--user" in c for c in systemd_calls)
    assert check.status == HP.PASS
    assert check.evidence["linger"] is True


def test_supervisor_preserves_failed_terminal_reason(monkeypatch, tmp_path):
    import product.host_supervisor as HS
    monkeypatch.setenv("QT_RUNTIME_ROOT", str(tmp_path))
    supervisor = HS.HostSupervisor()
    calls = []
    monkeypatch.setattr(supervisor, "verify_safety", lambda: (_ for _ in ()).throw(RuntimeError("boom")))
    monkeypatch.setattr(supervisor, "write_status", lambda **kwargs: calls.append(kwargs))
    assert supervisor.run() == 1
    assert calls[-1]["state"] == "FAILED"
    assert "boom" in calls[-1]["error"]


def test_recovered_market_access_retries_degraded_bootstrap(monkeypatch):
    import product.host_bootstrap as HB
    import product.host_preflight as HP
    import product.host_supervisor as HS
    supervisor = HS.HostSupervisor()
    supervisor.bootstrap = {"state": "DEGRADED"}
    checks = [HP.Check("nse_official", HP.PASS, "ok", True, {})]
    monkeypatch.setattr(HP, "probe_market_access", lambda: (checks, {"market_blocked": False}))
    retried = []
    monkeypatch.setattr(HB, "bootstrap_host_state", lambda **kwargs: retried.append(True) or {"state": "READY"})
    supervisor._probe_market_access_worker()
    assert retried == [True]
    assert supervisor.bootstrap["state"] == "READY"


def test_report_scheduler_failure_is_persisted_and_alerted_once(tmp_path, monkeypatch):
    import product.host_alerts as HA
    import product.host_entrypoint as HE
    import product.host_report_job as HR
    monkeypatch.setenv(RP.ENV_VAR, str(tmp_path))
    monkeypatch.setattr(HR, "run_once", lambda **kwargs: (_ for _ in ()).throw(RuntimeError("report-broke")))
    alerts = []
    monkeypatch.setattr(HA, "send_operational_alert", lambda msg: alerts.append(msg) or SimpleNamespace(
        to_dict=lambda: {"attempted": True, "delivered": True, "channel": "test", "detail": "delivered"}
    ))
    first = HE._report_iteration()
    second = HE._report_iteration()
    marker = json.loads((tmp_path / HE.REPORT_SCHEDULER_REL).read_text())
    assert first["state"] == second["state"] == "FAILED"
    assert marker["last_error"].startswith("RuntimeError")
    assert len(alerts) == 1


def test_alert_failure_never_persists_secret_bearing_exception(monkeypatch):
    import product.host_alerts as HA
    secret = "12345:super-secret-token"
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", secret)
    monkeypatch.setenv("TELEGRAM_CHAT_ID", "42")
    monkeypatch.delenv("QT_ALERT_WEBHOOK_URL", raising=False)

    def explode(request, timeout=0):
        raise ValueError(f"bad URL {request.full_url}")

    monkeypatch.setattr(HA.urllib.request, "urlopen", explode)
    result = HA.send_operational_alert("test")
    assert result.delivered is False
    assert secret not in result.detail
    assert "alert delivery failed" in result.detail
