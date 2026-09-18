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

# There are no longer any exemptions. The two installer lines that used to be
# listed here built launchd's stdout/stderr targets beneath QT_RUNTIME_ROOT;
# they were removed because launchd opens those paths before the entrypoint can
# attach the external runtime. Keep this set empty: every remaining
# joinpath("logs", ...) in production code must fail the guard above.
_EXPLICIT_TARGET_PATHS: set[tuple[str, str]] = set()


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


_RESEARCH_EVIDENCE_CONSTANTS = (
    ("product.due_diligence.acquire", "EVIDENCE_ROOT"),
    ("product.due_diligence.isolation", "RUN_ROOT"),
    ("product.due_diligence.providers.base", "ARCHIVE"),
    ("reporting.evidence_intake", "EVIDENCE_ROOT"),
    ("product.pit_ingest", "EVIDENCE_ROOT"),
    ("product.pit_backfill", "EVIDENCE_ROOT"),
)


@pytest.mark.parametrize("module,attr", [
    ("data.institutional_flows", "_CACHE"),
    ("execution.us_autopilot", "_STATE_FILE"),
    ("product.reco_ledger", "LEDGER_PATH"),
    ("product.recommendations_store", "DEFAULT_RECO_PATH"),
    *_RESEARCH_EVIDENCE_CONSTANTS,
])
def test_module_level_path_constants_follow_the_override(module, attr):
    import importlib
    resolved = Path(getattr(importlib.import_module(module), attr))
    assert runtime_root() in resolved.parents


def test_research_evidence_root_is_not_frozen_at_import(tmp_path, monkeypatch):
    """Acquire/isolation constants must retarget after import.

    Isolated due-diligence children and tests that change QT_RUNTIME_ROOT
    after the parent has already imported these modules used to keep writing
    into the checkout logs/research_evidence tree because Path(logs_dir()) was
    captured once at import.
    """
    import importlib

    imported = [
        (importlib.import_module(module), attr)
        for module, attr in _RESEARCH_EVIDENCE_CONSTANTS
    ]
    redirected = tmp_path / "post-import-runtime"
    monkeypatch.setenv(ENV_VAR, str(redirected))
    checkout_logs = REPO_ROOT / "logs"
    checkout_before = {
        str(path.relative_to(checkout_logs))
        for path in checkout_logs.rglob("*")
        if path.is_file()
    } if checkout_logs.exists() else set()

    for module, attr in imported:
        root = Path(getattr(module, attr))
        expected = redirected / "logs" / "research_evidence"
        assert expected == root or expected in root.parents
        marker = root / "_qt_runtime_probe.txt"
        marker.parent.mkdir(parents=True, exist_ok=True)
        marker.write_text("probe", encoding="utf-8")
        resolved = marker.resolve()
        assert redirected.resolve() in resolved.parents
        assert checkout_logs.resolve() not in resolved.parents
        assert REPO_ROOT.resolve() not in resolved.parents or redirected.resolve() in resolved.parents

    checkout_after = {
        str(path.relative_to(checkout_logs))
        for path in checkout_logs.rglob("*")
        if path.is_file()
    } if checkout_logs.exists() else set()
    assert checkout_after == checkout_before


def test_isolated_acquire_child_inherits_runtime_root_and_does_not_touch_checkout(tmp_path):
    redirected = tmp_path / "isolated-child-runtime"
    redirected.mkdir()
    probe = (
        "from pathlib import Path;"
        "from product.due_diligence.acquire import EVIDENCE_ROOT, _symbol_dir;"
        "from core.runtime_paths import runtime_root;"
        "marker = _symbol_dir('QTISOL') / 'child_probe.json';"
        "marker.write_text('{}', encoding='utf-8');"
        "print(runtime_root());"
        "print(marker.resolve());"
        "print(Path(EVIDENCE_ROOT).resolve())"
    )
    env = {**os.environ, ENV_VAR: str(redirected), "PYTHONPATH": str(REPO_ROOT)}
    proc = subprocess.run(
        [sys.executable, "-c", probe],
        cwd=str(REPO_ROOT),
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )
    root, marker, evidence = proc.stdout.split("\n")[:3]
    assert Path(root) == redirected
    marker_path = Path(marker)
    assert (redirected / "logs" / "research_evidence" / "QTISOL" / "autonomy" / "child_probe.json") == marker_path
    assert redirected.resolve() in marker_path.parents
    assert (REPO_ROOT / "logs").resolve() not in marker_path.parents
    assert Path(evidence) == redirected / "logs" / "research_evidence"
    assert not (REPO_ROOT / "logs" / "research_evidence" / "QTISOL").exists()


def test_isolated_child_env_pins_current_runtime_root():
    import inspect

    from product.due_diligence import isolation

    source = inspect.getsource(isolation._run_child_request)
    assert 'env["QT_RUNTIME_ROOT"] = str(runtime_root())' in source
