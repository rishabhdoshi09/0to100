"""Synthetic replay data must never contaminate QuantTerm's real runtime."""
from __future__ import annotations

from pathlib import Path

import pytest

import core.runtime_paths as RP
import scripts.seed_replay_store as S


def _configure(monkeypatch, root: Path, *, opt_in: bool = True) -> None:
    monkeypatch.setenv(RP.ENV_VAR, str(root))
    if opt_in:
        monkeypatch.setenv(S.FIXTURE_OPT_IN_ENV, "1")
    else:
        monkeypatch.delenv(S.FIXTURE_OPT_IN_ENV, raising=False)


def test_fixture_guard_requires_explicit_opt_in(monkeypatch, tmp_path):
    root = tmp_path / "fixture"
    _configure(monkeypatch, root, opt_in=False)
    monkeypatch.setattr(RP, "read_runtime_pointer", lambda *args, **kwargs: None)

    with pytest.raises(RuntimeError, match=S.FIXTURE_OPT_IN_ENV):
        S._fixture_runtime_root()

    assert not root.exists()


def test_fixture_guard_refuses_adopted_production_runtime(monkeypatch, tmp_path):
    root = tmp_path / "production-runtime"
    _configure(monkeypatch, root)
    monkeypatch.setattr(RP, "read_runtime_pointer", lambda *args, **kwargs: root)

    with pytest.raises(RuntimeError, match="production runtime"):
        S._fixture_runtime_root()

    assert not root.exists()


def test_fixture_guard_refuses_nonempty_unmarked_runtime(monkeypatch, tmp_path):
    root = tmp_path / "existing-runtime"
    (root / "logs").mkdir(parents=True)
    (root / "logs" / "real-state.db").write_text("real", encoding="utf-8")
    _configure(monkeypatch, root)
    monkeypatch.setattr(RP, "read_runtime_pointer", lambda *args, **kwargs: None)

    with pytest.raises(RuntimeError, match="not empty"):
        S._fixture_runtime_root()

    assert not (root / S.FIXTURE_MARKER).exists()
    assert (root / "logs" / "real-state.db").read_text(encoding="utf-8") == "real"


def test_fixture_guard_marks_empty_runtime_and_allows_reuse(monkeypatch, tmp_path):
    root = tmp_path / "fixture"
    _configure(monkeypatch, root)
    monkeypatch.setattr(RP, "read_runtime_pointer", lambda *args, **kwargs: None)

    assert S._fixture_runtime_root() == root.resolve()
    marker = root / S.FIXTURE_MARKER
    assert marker.exists()

    (root / "fixture-state.txt").write_text("fixture", encoding="utf-8")
    assert S._fixture_runtime_root() == root.resolve()


def test_main_refuses_before_generating_or_writing_market_data(monkeypatch, tmp_path):
    root = tmp_path / "would-be-runtime"
    _configure(monkeypatch, root, opt_in=False)
    monkeypatch.setattr(RP, "read_runtime_pointer", lambda *args, **kwargs: None)

    def must_not_run(*args, **kwargs):
        raise AssertionError("synthetic store generation ran before runtime isolation")

    monkeypatch.setattr(S, "build_store", must_not_run)
    assert S.main(["--symbols", "1", "--sessions", "20"]) == 2
    assert not root.exists()
