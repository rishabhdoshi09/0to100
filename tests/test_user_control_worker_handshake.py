from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_parallel_api_never_silently_accepts_without_market_ops_worker():
    src = (ROOT / "terminal_product_api_parallel.py").read_text(encoding="utf-8")
    assert "_ensure_ops_worker_strict" in src
    assert "_base_ensure_ops_worker(wait=True)" in src
    assert "Market operations worker did not become ready" in src
    assert "core._ensure_ops_worker = _ensure_ops_worker_strict" in src


def test_parallel_api_kills_stale_worker_ownership_before_replacement():
    src = (ROOT / "terminal_product_api_parallel.py").read_text(encoding="utf-8")
    assert "_stop_stale_owner" in src
    assert "signal.SIGTERM" in src
    assert "signal.SIGKILL" in src
    assert 'core._OPERATION_CONTROLS["RUN_LONG_TERM_SCAN_NOW"] = "LONG_TERM_SCAN"' in src
