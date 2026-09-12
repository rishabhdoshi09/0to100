"""
The runtime root is the single place every durable artifact path comes from.

Two things this suite has already been burned by:

  * a module that resolves `<repo>/logs/...` itself keeps writing into the
    checkout no matter what the test harness redirects, and the running desk
    then reads test fixtures as genuine market state;
  * a module that resolves a RELATIVE `logs/...` writes wherever the process
    happened to be started from, so a daemon launched by systemd and the same
    daemon launched by hand disagree about where their state lives.

`core.runtime_paths` fixes both: production resolves to `<repo>/logs`, and a
process (the test harness, a sandbox, a second checkout) can redirect the whole
tree with `QT_RUNTIME_ROOT`. These tests assert that no production module has
quietly gone back to resolving the path itself.

One narrow exception exists for the host installer: it renders a launchd file
for an explicit *target* ``runtime_root`` supplied to the renderer. That path is
not process runtime state and is independently pinned by host-install tests. The
exception is deliberately exact so another self-resolved logs path still fails.
"""
from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

from core.runtime_paths import ENV_VAR, REPO_ROOT, logs_dir, logs_path, runtime_root

# Directories that are NOT the running product: the archived Streamlit pages,
# the legacy entrypoint, developer scripts run by hand from the checkout, the
# test suite itself, and the virtualenv.
_EXCLUDED_PREFIXES = ("tests/", "ui/", "scripts/", "venv/", ".venv/", "frontend/")
_EXCLUDED_FILES = ("legacy_app.py", "core/runtime_paths.py")

# `something / "logs"` and `Path("logs...")` — the two shapes that resolve a
# durable path without going through the runtime root. Comments and prose that
# merely name a file under logs/ do not match either.
_SELF_RESOLVED = re.compile(r'/\s*["\']logs["\']|Path\(\s*["\']logs[/"\']')

# The installer is not resolving *this process's* runtime tree here. It is
# serialising a launchd path for the explicit destination root supplied by the
# caller. Keep this exact: do not turn it into a file-wide or regex exemption.
_EXPLICIT_TARGET_PATHS = {
    (
        "product/host_install.py",
        'service_logs = _resolved(runtime_root) / "logs" / "service"',
    ),
}


def _production_sources() -> list[Path]:
    out = subprocess.run(
        ["git", "ls-files", "*.py"],
        cwd=REPO_ROOT, capture_output=True, text=True, check=True,
    ).stdout.split()
    return [
        REPO_ROOT / rel for rel in out
        if not rel.startswith(_EXCLUDED_PREFIXES) and rel not in _EXCLUDED_FILES
    ]


def test_no_production_module_resolves_a_logs_path_itself():
    offenders: list[str] = []
    for path in _production_sources():
        rel = path.relative_to(REPO_ROOT).as_posix()
        for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            code = line.split("#", 1)[0]
            stripped = code.strip()
            if _SELF_RESOLVED.search(code) and (rel, stripped) not in _EXPLICIT_TARGET_PATHS:
                offenders.append(f"{rel}:{lineno}: {line.strip()}")
    assert not offenders, (
        "these modules resolve a logs path without core.runtime_paths, so "
        "QT_RUNTIME_ROOT cannot redirect them:\n  " + "\n  ".join(offenders)
    )


def test_explicit_target_exception_is_still_exact():
    """If the installer line changes, this exception must be reviewed rather than widening silently."""
    rel, expected = next(iter(_EXPLICIT_TARGET_PATHS))
    text = (REPO_ROOT / rel).read_text(encoding="utf-8")
    matching = [line.strip() for line in text.splitlines() if line.strip() == expected]
    assert matching == [expected]


def test_production_default_is_the_checkout_logs_tree():
    """With no override set, every path must land in <repo>/logs — unchanged."""
    env = {k: v for k, v in os.environ.items() if k != ENV_VAR}
    probe = (
        "from core.runtime_paths import logs_dir, logs_path, runtime_root;"
        "print(runtime_root());print(logs_dir());print(logs_path('a', 'b.json'))"
    )
    proc = subprocess.run(
        [sys.executable, "-c", probe],
        cwd=REPO_ROOT, env={**env, "PYTHONPATH": str(REPO_ROOT)},
        capture_output=True, text=True, check=True,
    )
    root, logs, nested = proc.stdout.split("\n")[:3]
    assert Path(root) == REPO_ROOT
    assert Path(logs) == REPO_ROOT / "logs"
    assert Path(nested) == REPO_ROOT / "logs" / "a" / "b.json"


def test_override_redirects_the_whole_tree(tmp_path, monkeypatch):
    monkeypatch.setenv(ENV_VAR, str(tmp_path))
    assert runtime_root() == tmp_path
    assert logs_dir() == tmp_path / "logs"
    assert logs_path("product", "x.json") == tmp_path / "logs" / "product" / "x.json"


def test_override_is_read_per_call_not_frozen_at_import(tmp_path, monkeypatch):
    """A cached root would silently ignore a harness that redirects later."""
    first = logs_dir()
    monkeypatch.setenv(ENV_VAR, str(tmp_path))
    assert logs_dir() == tmp_path / "logs" != first


def test_blank_override_falls_back_to_the_checkout(monkeypatch):
    monkeypatch.setenv(ENV_VAR, "   ")
    assert runtime_root() == REPO_ROOT


def test_suite_itself_runs_under_an_override():
    """Guards the conftest block: if it stops applying, every test writes live."""
    assert os.environ.get(ENV_VAR), "conftest must redirect the runtime root"
    assert runtime_root() != REPO_ROOT
    assert REPO_ROOT not in logs_dir().parents


@pytest.mark.parametrize("module,attr", [
    ("data.institutional_flows", "_CACHE"),
    ("execution.us_autopilot", "_STATE_FILE"),
    ("product.reco_ledger", "LEDGER_PATH"),
    ("product.recommendations_store", "DEFAULT_RECO_PATH"),
])
def test_module_level_path_constants_follow_the_override(module, attr):
    """These bind at import time, so they must bind to the redirected root."""
    import importlib

    resolved = Path(getattr(importlib.import_module(module), attr))
    assert runtime_root() in resolved.parents