from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_api_runtime_never_silently_accepts_without_market_ops_worker():
    façade = (ROOT / "api" / "app.py").read_text(encoding="utf-8")
    core = (ROOT / "api" / "runtime.py").read_text(encoding="utf-8")
    assert "from . import runtime as _core" in façade
    assert "app = _core.app" in façade
    assert "_ensure_ops_worker_strict" in core
    assert "_base_ensure_ops_worker(wait=True)" in core
    assert "Market operations worker did not become ready" in core
    assert "core._ensure_ops_worker = _ensure_ops_worker_strict" in core
    assert "_live_owner_pid" in core


def test_api_runtime_kills_only_verified_stale_worker_before_replacement():
    src = (ROOT / "api" / "runtime.py").read_text(encoding="utf-8")
    assert "_stop_stale_owner" in src
    assert "operations.market_ops" in src
    assert "signal.SIGTERM" in src
    assert "signal.SIGKILL" in src
    # Product import order must not silently create a second long-term scanner path.
    assert 'core._OPERATION_CONTROLS["RUN_LONG_TERM_SCAN_NOW"] = "LONG_TERM_SCAN"' not in src
