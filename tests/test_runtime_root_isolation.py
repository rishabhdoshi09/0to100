"""
The runtime root is the single place every durable artifact path comes from.

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

from core.runtime_paths import ENV_VAR, REPO_ROOT, logs_dir, logs_path, runtime_root

_EXCLUDED_PREFIXES = ("tests/", "ui/", "scripts/", "venv/", ".venv/", "frontend/")
_EXCLUDED_FILES = ("legacy_app.py", "core/runtime_paths.py")
_SELF_RESOLVED = re.compile(r'/\s*["\']logs["\']|Path\(\s*["\']logs[/"\']')


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
            if _SELF_RESOLVED.search(code):
                offenders.append(f"{rel}:{lineno}: {line.strip()}")
    assert not offenders, (
        "these modules resolve a logs path without core.runtime_paths, so "
        "QT_RUNTIME_ROOT cannot redirect them:\n  " + "\n  ".join(offenders)
    )


def test_production_default_is_the_checkout_logs_tree():
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
    first = logs_dir()
    monkeypatch.setenv(ENV_VAR, str(tmp_path))
    assert logs_dir() == tmp_path / "logs" != first


def test_blank_override_falls_back_to_the_checkout(monkeypatch):
    monkeypatch.setenv(ENV_VAR, "   ")
    assert runtime_root() == REPO_ROOT


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
