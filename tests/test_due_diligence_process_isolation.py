from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys
import time

import pytest

from product.due_diligence import acquire as acquire_mod
from product.due_diligence import isolation


def test_production_acquire_facade_is_installed() -> None:
    """Durable long-budget calls must not use the raw cooperative function."""
    assert acquire_mod.acquire_symbol.__name__ == "acquire_symbol"
    assert acquire_mod.acquire_shortlist.__name__ == "acquire_shortlist"
    assert any(
        getattr(cell, "cell_contents", None) is isolation.acquire_symbol_isolated
        for cell in (acquire_mod.acquire_symbol.__closure__ or ())
    )
    assert any(
        getattr(cell, "cell_contents", None) is isolation.acquire_shortlist_isolated
        for cell in (acquire_mod.acquire_shortlist.__closure__ or ())
    )


def test_worker_deadline_has_large_observer_margin() -> None:
    from operations.market_ops import DUE_DILIGENCE_SYMBOL_S
    from scripts._product_acceptance_core import build_parser

    args = build_parser().parse_args([])
    assert DUE_DILIGENCE_SYMBOL_S <= 90.0
    assert float(args.acquire_timeout) - float(DUE_DILIGENCE_SYMBOL_S) >= 300.0


def test_atomic_bytes_never_exposes_partial_canonical_file(tmp_path: Path, monkeypatch) -> None:
    target = tmp_path / "evidence.pdf"
    target.write_bytes(b"last-good")
    real_replace = isolation.os.replace

    def fail_replace(src, dst):
        assert Path(dst) == target
        raise OSError("simulated publication failure")

    monkeypatch.setattr(isolation.os, "replace", fail_replace)
    with pytest.raises(OSError):
        isolation._atomic_bytes(target, b"new-partial-content")
    assert target.read_bytes() == b"last-good"
    assert not list(tmp_path.glob("*.tmp"))
    monkeypatch.setattr(isolation.os, "replace", real_replace)


@pytest.mark.skipif(os.name != "posix", reason="process-group contract is POSIX/macOS specific")
def test_timeout_cleanup_kills_and_reaps_entire_process_group(tmp_path: Path) -> None:
    grandchild_pid = tmp_path / "grandchild.pid"
    code = (
        "import pathlib,signal,subprocess,sys,time;"
        "p=subprocess.Popen([sys.executable,'-c','import signal,time; signal.signal(signal.SIGTERM, signal.SIG_IGN); time.sleep(30)']);"
        f"pathlib.Path({str(grandchild_pid)!r}).write_text(str(p.pid));"
        "signal.signal(signal.SIGTERM, signal.SIG_IGN);"
        "time.sleep(30)"
    )
    proc = subprocess.Popen(
        [sys.executable, "-c", code],
        start_new_session=True,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    deadline = time.monotonic() + 3.0
    while not grandchild_pid.exists() and time.monotonic() < deadline:
        time.sleep(0.02)
    assert grandchild_pid.exists()

    isolation._terminate_process_group(proc, grace_s=0.05)
    assert proc.poll() is not None

    # A process group exists while any member is still alive.  Give init a
    # brief chance to reap the killed grandchild before asserting no orphan.
    gone = False
    deadline = time.monotonic() + 2.0
    while time.monotonic() < deadline:
        try:
            os.killpg(proc.pid, 0)
        except ProcessLookupError:
            gone = True
            break
        time.sleep(0.05)
    assert gone, "isolated acquisition left a descendant in its process group"


def test_shortlist_timeout_does_not_block_next_symbol(monkeypatch) -> None:
    monkeypatch.setattr(acquire_mod, "shortlist_symbols", lambda limit=6, scan_payload=None: ["AAA", "BBB"])
    calls: list[str] = []

    def fake_acquire(symbol, **kwargs):
        calls.append(symbol)
        if symbol == "AAA":
            raise isolation.HardAcquireTimeout(
                "deadline exceeded while screener",
                result={"symbol": symbol, "provider": "screener", "elapsed_s": 0.1},
            )
        return {"symbol": symbol, "ok": True}

    monkeypatch.setattr(isolation, "acquire_symbol_isolated", fake_acquire)
    monkeypatch.setattr(isolation.time, "sleep", lambda _seconds: None)

    result = isolation.acquire_shortlist_isolated(
        deadline_monotonic=time.monotonic() + 60.0,
        per_symbol_s=30.0,
    )
    assert calls == ["AAA", "BBB"]
    assert result["timed_out"] is True
    assert result["status"] == "TIMEOUT"
    assert result["n_ok"] == 1
    assert result["n_failed"] == 1
    assert result["acquired"] == ["BBB"]
    assert result["errors"][0]["result"]["provider"] == "screener"


def test_shortlist_fast_success_shape_is_unchanged(monkeypatch) -> None:
    monkeypatch.setattr(acquire_mod, "shortlist_symbols", lambda limit=6, scan_payload=None: ["AAA", "BBB"])
    monkeypatch.setattr(
        isolation,
        "acquire_symbol_isolated",
        lambda symbol, **kwargs: {"symbol": symbol, "ok": True, "files_on_disk": [f"{symbol}.json"]},
    )
    monkeypatch.setattr(isolation.time, "sleep", lambda _seconds: None)

    result = isolation.acquire_shortlist_isolated(
        deadline_monotonic=time.monotonic() + 60.0,
        per_symbol_s=30.0,
    )
    assert result == {
        "accepted": True,
        "status": "SUCCEEDED",
        "timed_out": False,
        "symbols": ["AAA", "BBB"],
        "acquired": ["AAA", "BBB"],
        "skipped": [],
        "errors": [],
        "n_ok": 2,
        "n_failed": 0,
        "places_orders": False,
        "gates_scanner": False,
    }


def test_expired_overall_deadline_skips_without_starting_provider(monkeypatch) -> None:
    monkeypatch.setattr(acquire_mod, "shortlist_symbols", lambda limit=6, scan_payload=None: ["AAA", "BBB"])

    def must_not_run(*args, **kwargs):
        raise AssertionError("provider must not start after overall deadline")

    monkeypatch.setattr(isolation, "acquire_symbol_isolated", must_not_run)
    result = isolation.acquire_shortlist_isolated(
        deadline_monotonic=time.monotonic() - 0.01,
        per_symbol_s=30.0,
    )
    assert result["timed_out"] is True
    assert result["n_ok"] == 0
    assert result["skipped"] == ["AAA", "BBB"]
    assert [row["status"] for row in result["errors"]] == ["TIMEOUT", "TIMEOUT"]
