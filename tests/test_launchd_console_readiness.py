"""Execute the launcher's real probe through Bash, including its exit status."""
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

import pytest


LAUNCHER = Path(__file__).resolve().parents[1] / "scripts" / "run_quantterm_complete.sh"


@pytest.mark.parametrize("ok,operational,started,expected", [
    (True, True, 101.0, 1),
    (False, True, 101.0, 0),
    (True, False, 101.0, 0),
    (False, False, 101.0, 0),
    (True, True, 99.0, 0),
    (True, True, None, 0),
])
def test_readiness_requires_successful_health_and_current_generation(
    tmp_path, ok, operational, started, expected,
):
    state = tmp_path / "state"
    state.mkdir()
    if started is not None:
        (state / "host_supervisor.json").write_text(json.dumps({
            "started_at": datetime.fromtimestamp(started, timezone.utc).isoformat(),
        }))
    # Inject a health response in the probe interpreter; no sockets or service needed.
    (tmp_path / "sitecustomize.py").write_text(
        "import io, os, urllib.request\n"
        "urllib.request.urlopen = lambda *a, **k: io.StringIO(os.environ['TEST_HEALTH'])\n"
    )
    source = LAUNCHER.read_text()
    conditional = source.split('    if summary="$(', 1)[1].split(
        '    if (( i % 5 == 0 )); then', 1,
    )[0]
    conditional = '    if summary="$(' + conditional
    script = (
        f"py={shlex.quote(sys.executable)}\ninstalled_runtime={shlex.quote(str(tmp_path))}\n"
        "restart_after=100\nready=0\n"
        f"for i in 1; do\n{conditional}\ndone\necho result=$ready\n"
    )
    proc = subprocess.run(["bash", "-c", script], cwd=tmp_path, env={
        **os.environ,
        "PYTHONPATH": str(tmp_path),
        "QT_RUNTIME_ROOT": str(tmp_path / "wrong-shell-runtime"),
        "TEST_HEALTH": json.dumps({"ok": ok, "operational_ready": operational}),
    }, capture_output=True, text=True, timeout=10)
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip() == f"result={expected}"


@pytest.mark.parametrize("started,expected", [
    (99.0, "Waiting for current host generation"),
    (101.0, "RUNNING · children_alive=1/1 healthy=1/1"),
])
def test_progress_does_not_present_previous_generation_as_current(
    tmp_path, monkeypatch, capsys, started, expected,
):
    state = tmp_path / "state"
    state.mkdir()
    (state / "host_supervisor.json").write_text(json.dumps({
        "state": "RUNNING",
        "started_at": datetime.fromtimestamp(started, timezone.utc).isoformat(),
        "children": {"market_api": {"alive": True, "healthy": True}},
    }))
    monkeypatch.setenv("QT_RUNTIME_ROOT", str(tmp_path / "wrong-shell-runtime"))
    monkeypatch.setattr(sys, "argv", ["-", "100", str(tmp_path)])
    source = LAUNCHER.read_text().split('    if (( i % 5 == 0 )); then', 1)[1]
    probe = source.split("<<'PY' 2>/dev/null || true\n", 1)[1].split("\nPY\n", 1)[0]
    try:
        exec(compile(probe, "launcher-progress", "exec"), {})
    except SystemExit as exc:
        assert exc.code == 0
    assert expected in capsys.readouterr().out
