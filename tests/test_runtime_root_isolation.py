"""Guard the single mutable runtime root across production modules.

Production code must not self-resolve ``logs/...``. Tests and service managers
can redirect the whole mutable tree with QT_RUNTIME_ROOT, while an installed
checkout may use the ignored ``.quantterm_runtime_root`` breadcrumb when no
environment override is present.
"""
from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

from core.runtime_paths import (
    ENV_VAR,
    POINTER_NAME,
    REPO_ROOT,
    logs_dir,
    logs_path,
    read_runtime_pointer,
    runtime_root,
)

_EXCLUDED_PREFIXES = ("tests/", "ui/", "scripts/", "venv/", ".venv/", "frontend/")
_EXCLUDED_FILES = ("legacy_app.py", "core/runtime_paths.py")
_SELF_RESOLVED = re.compile(
    r'/\s*["\']logs["\']|Path\(\s*["\']logs[/"\']|\.joinpath\(\s*["\']logs["\']'
)

# These two installer lines build paths beneath the explicit destination root
# passed by the caller. They are not process-state path resolution. Keep the
# exemption exact so any new use of joinpath("logs", ...) still fails the guard.
_EXPLICIT_TARGET_PATHS = {
    (
        "product/host_install.py",
        'service_logs = _resolved(runtime_root).joinpath("logs", "service")',
    ),
    (
        "product/host_install.py",
        '_resolved(runtime_root).joinpath("logs", "service").mkdir(parents=True, exist_ok=True)',
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


def test_joinpath_logs_spelling_is_guarded():
    assert _SELF_RESOLVED.search('root.joinpath("logs", "x.json")')


def test_explicit_installer_target_exceptions_remain_exact():
    by_file: dict[str, list[str]] = {}
    for rel, expected in _EXPLICIT_TARGET_PATHS:
        by_file.setdefault(rel, []).append(expected)
    for rel, expected_lines in by_file.items():
        text = (REPO_ROOT / rel).read_text(encoding="utf-8")
        for expected in expected_lines:
            matching = [line.strip() for line in text.splitlines() if line.strip() == expected]
            assert matching == [expected]


def test_default_contract_respects_installed_pointer_when_present():
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
    expected_root = read_runtime_pointer() or REPO_ROOT
    assert Path(root) == expected_root
    assert Path(logs) == expected_root / "logs"
    assert Path(nested) == expected_root / "logs" / "a" / "b.json"


def test_override_redirects_the_whole_tree(tmp_path, monkeypatch):
    monkeypatch.setenv(ENV_VAR, str(tmp_path))
    assert runtime_root() == tmp_path
    assert logs_dir() == tmp_path / "logs"
    assert logs_path("product", "x.json") == tmp_path / "logs" / "product" / "x.json"


def test_override_is_read_per_call_not_frozen_at_import(tmp_path, monkeypatch):
    first = logs_dir()
    monkeypatch.setenv(ENV_VAR, str(tmp_path))
    assert logs_dir() == tmp_path / "logs" != first


def test_blank_override_uses_pointer_or_checkout(monkeypatch):
    monkeypatch.setenv(ENV_VAR, "   ")
    assert runtime_root() == (read_runtime_pointer() or REPO_ROOT)


def test_suite_itself_runs_under_an_override():
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
    import importlib
    resolved = Path(getattr(importlib.import_module(module), attr))
    assert runtime_root() in resolved.parents
