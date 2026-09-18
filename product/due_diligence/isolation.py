"""Hard wall-clock isolation for Due Diligence provider acquisition.

Network providers are intentionally executed in a separate process group.  The
market-operations worker owns the deadline and can therefore terminate a stuck
provider without leaving a Python thread running in the background or keeping
the due_diligence lane leased forever.

The child writes only through the existing canonical acquisition functions. Raw
attachment writes are replaced in the child with same-directory temp-file +
``os.replace`` publication, so a forced kill can leave at worst an unreferenced
.temp file, never a truncated canonical evidence file.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback
import uuid
from typing import Any, Mapping

from core.runtime_paths import logs_dir, RuntimeLogsPath

ROOT = Path(__file__).resolve().parents[2]
RUN_ROOT = RuntimeLogsPath("research_evidence", "_acquire_runs")
DEFAULT_SYMBOL_TIMEOUT_S = 90.0
TERMINATE_GRACE_S = 2.0


class HardAcquireTimeout(TimeoutError):
    """Provider process exceeded its hard wall-clock budget."""

    def __init__(self, message: str, *, result: Mapping[str, Any] | None = None) -> None:
        super().__init__(message)
        self.result = dict(result or {})


class IsolatedAcquireError(RuntimeError):
    """The isolated acquisition child terminated unsuccessfully."""

    def __init__(self, message: str, *, result: Mapping[str, Any] | None = None) -> None:
        super().__init__(message)
        self.result = dict(result or {})


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.{time.time_ns()}.tmp")
    try:
        with tmp.open("w", encoding="utf-8") as handle:
            json.dump(dict(payload), handle, indent=2, default=str)
            handle.flush()
            try:
                os.fsync(handle.fileno())
            except OSError:
                pass
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def _atomic_bytes(path: Path, content: bytes) -> None:
    """Publish one canonical evidence file atomically."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.{time.time_ns()}.tmp")
    try:
        with tmp.open("wb") as handle:
            handle.write(content)
            handle.flush()
            try:
                os.fsync(handle.fileno())
            except OSError:
                pass
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return dict(payload) if isinstance(payload, Mapping) else {}
    except Exception:
        return {}


def _evidence_snapshot(symbol: str) -> dict[str, Any]:
    """Cache-only provenance retained when a later provider is killed."""
    try:
        from product.due_diligence.acquire import facts_path, load_autonomy_facts

        path = facts_path(symbol)
        facts = load_autonomy_facts(symbol)
        return {
            "facts_exists": path.exists(),
            "facts_path": str(path),
            "acquired_at": str(facts.get("acquired_at") or ""),
            "files_on_disk": [str(item) for item in list(facts.get("files_on_disk") or []) if item],
        }
    except Exception as exc:
        return {"facts_exists": False, "snapshot_error": f"{type(exc).__name__}: {exc}"[:240]}


def _terminate_process_group(proc: subprocess.Popen[Any], *, grace_s: float = TERMINATE_GRACE_S) -> None:
    """Terminate the isolated child *and* descendants, then reap the child."""
    if proc.poll() is not None:
        return
    if os.name == "posix":
        try:
            os.killpg(proc.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        except OSError:
            try:
                proc.terminate()
            except OSError:
                pass
    else:  # pragma: no cover - production host is macOS/Linux
        try:
            proc.terminate()
        except OSError:
            pass
    try:
        proc.wait(timeout=max(0.1, float(grace_s)))
        return
    except subprocess.TimeoutExpired:
        pass
    if os.name == "posix":
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        except OSError:
            try:
                proc.kill()
            except OSError:
                pass
    else:  # pragma: no cover
        try:
            proc.kill()
        except OSError:
            pass
    try:
        proc.wait(timeout=max(1.0, float(grace_s)))
    except subprocess.TimeoutExpired as exc:  # should be impossible after SIGKILL
        raise RuntimeError(f"isolated acquisition child {proc.pid} could not be reaped") from exc


def _bounded_timeout(*, requested_s: float, deadline_monotonic: float | None) -> float:
    timeout = max(0.1, float(requested_s))
    if deadline_monotonic is not None:
        remaining = float(deadline_monotonic) - time.monotonic()
        if remaining <= 0:
            return 0.0
        timeout = min(timeout, remaining)
    return timeout


def _run_child_request(
    request: Mapping[str, Any],
    *,
    timeout_s: float,
    deadline_monotonic: float | None = None,
) -> dict[str, Any]:
    """Run one symbol acquisition under a hard process-group deadline."""
    symbol = str(request.get("symbol") or "").strip().upper()
    timeout = _bounded_timeout(requested_s=timeout_s, deadline_monotonic=deadline_monotonic)
    if timeout <= 0:
        result = {
            "symbol": symbol,
            "provider": "not_started",
            "stage": "DEADLINE_EXCEEDED",
            "elapsed_s": 0.0,
            "hard_timeout_s": 0.0,
            "partial_evidence": _evidence_snapshot(symbol),
        }
        raise HardAcquireTimeout(f"due-diligence deadline exceeded before {symbol or 'symbol'} started", result=result)

    run_id = uuid.uuid4().hex
    run_dir = RUN_ROOT / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    request_path = run_dir / "request.json"
    result_path = run_dir / "result.json"
    status_path = run_dir / "status.json"
    log_path = run_dir / "child.log"
    _atomic_json(request_path, request)
    _atomic_json(status_path, {
        "run_id": run_id,
        "symbol": symbol,
        "stage": "STARTING",
        "provider": "isolated_child",
        "updated_at": time.time(),
    })

    command = [
        sys.executable,
        "-m",
        "product.due_diligence.isolation",
        "--child",
        str(request_path),
        str(result_path),
        str(status_path),
    ]
    env = dict(os.environ)
    env["QT_DD_ISOLATED_CHILD"] = "1"
    from core.runtime_paths import runtime_root
    env["QT_RUNTIME_ROOT"] = str(runtime_root())
    started = time.monotonic()
    with log_path.open("ab", buffering=0) as log_handle:
        proc = subprocess.Popen(
            command,
            cwd=str(ROOT),
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            start_new_session=(os.name == "posix"),
        )
        try:
            return_code = proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            _terminate_process_group(proc)
            elapsed = time.monotonic() - started
            status = _read_json(status_path)
            result = {
                "run_id": run_id,
                "symbol": symbol,
                "provider": str(status.get("provider") or "unknown"),
                "stage": str(status.get("stage") or "provider_call"),
                "elapsed_s": round(elapsed, 3),
                "hard_timeout_s": float(timeout),
                "child_pid": proc.pid,
                "child_reaped": proc.poll() is not None,
                "partial_evidence": _evidence_snapshot(symbol),
                "log_path": str(log_path),
            }
            raise HardAcquireTimeout(
                f"due-diligence deadline exceeded for {symbol or 'symbol'} after {elapsed:.1f}s "
                f"while {result['provider']}",
                result=result,
            )

    elapsed = time.monotonic() - started
    payload = _read_json(result_path)
    if return_code != 0 or payload.get("ok") is not True:
        detail = str(payload.get("error") or f"child exited {return_code}")
        result = {
            "run_id": run_id,
            "symbol": symbol,
            "provider": str(payload.get("provider") or _read_json(status_path).get("provider") or "unknown"),
            "stage": str(payload.get("stage") or "FAILED"),
            "elapsed_s": round(elapsed, 3),
            "return_code": int(return_code),
            "partial_evidence": _evidence_snapshot(symbol),
            "log_path": str(log_path),
        }
        raise IsolatedAcquireError(f"isolated due-diligence acquisition failed for {symbol}: {detail}"[:400], result=result)
    result = payload.get("result")
    if not isinstance(result, Mapping):
        raise IsolatedAcquireError(
            f"isolated due-diligence acquisition returned no result for {symbol}",
            result={"symbol": symbol, "elapsed_s": round(elapsed, 3), "run_id": run_id},
        )
    return dict(result)


def acquire_symbol_isolated(
    symbol: str,
    *,
    force: bool = False,
    datasets: list[str] | None = None,
    deadline_monotonic: float | None = None,
    timeout_s: float = DEFAULT_SYMBOL_TIMEOUT_S,
) -> dict[str, Any]:
    """Acquire one symbol with a hard wall-clock budget."""
    return _run_child_request(
        {
            "symbol": str(symbol).strip().upper(),
            "force": bool(force),
            "datasets": list(datasets) if datasets else None,
            "timeout_s": float(timeout_s),
        },
        timeout_s=timeout_s,
        deadline_monotonic=deadline_monotonic,
    )


def acquire_shortlist_isolated(
    *,
    limit: int = 6,
    force: bool = False,
    scan_payload: Mapping[str, Any] | None = None,
    deadline_monotonic: float | None = None,
    per_symbol_s: float = DEFAULT_SYMBOL_TIMEOUT_S,
    progress_cb=None,
) -> dict[str, Any]:
    """Acquire shortlist symbols one isolated child at a time.

    One stalled provider cannot consume another symbol's budget. The overall
    market-ops deadline still bounds the complete shortlist.
    """
    from product.due_diligence.acquire import shortlist_symbols

    symbols = shortlist_symbols(limit=limit, scan_payload=scan_payload)
    results: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    skipped: list[str] = []
    timed_out = False
    for index, symbol in enumerate(symbols):
        if deadline_monotonic is not None and time.monotonic() >= float(deadline_monotonic):
            timed_out = True
            skipped.extend(symbols[index:])
            errors.extend({"symbol": name, "error": "operation deadline exceeded", "status": "TIMEOUT"} for name in symbols[index:])
            break
        if progress_cb:
            progress_cb(symbol)
        try:
            row = acquire_symbol_isolated(
                symbol,
                force=force,
                deadline_monotonic=deadline_monotonic,
                timeout_s=per_symbol_s,
            )
            results.append(row)
        except HardAcquireTimeout as exc:
            timed_out = True
            errors.append({
                "symbol": symbol,
                "error": str(exc)[:240],
                "status": "TIMEOUT",
                "result": dict(exc.result),
            })
        except Exception as exc:
            errors.append({"symbol": symbol, "error": str(exc)[:240], "status": "FAILED"})
        time.sleep(0.2)
    status = "TIMEOUT" if timed_out else ("PARTIAL" if errors and results else "SUCCEEDED")
    return {
        "accepted": True,
        "status": status,
        "timed_out": timed_out,
        "symbols": symbols,
        "acquired": [str(row.get("symbol") or "") for row in results if row.get("symbol")],
        "skipped": skipped,
        "errors": errors,
        "n_ok": len(results),
        "n_failed": len(errors),
        "places_orders": False,
        "gates_scanner": False,
    }


def _patch_child_acquire(status_path: Path, symbol: str):
    """Install child-only progress probes and atomic attachment publication."""
    from product.due_diligence import acquire as acquire_mod

    def publish(stage: str, provider: str) -> None:
        _atomic_json(status_path, {
            "symbol": symbol,
            "stage": stage,
            "provider": provider,
            "updated_at": time.time(),
            "pid": os.getpid(),
        })

    original_save = acquire_mod._save_bytes

    def atomic_save_bytes(symbol_arg: str, name: str, content: bytes) -> Path:
        path = acquire_mod._symbol_dir(symbol_arg) / name
        _atomic_bytes(path, content)
        return path

    acquire_mod._save_bytes = atomic_save_bytes

    for name, provider in (
        ("_fetch_screener", "screener"),
        ("_fetch_nse", "nse_filings"),
        ("_fetch_annual_reports", "nse_annual_reports"),
        ("_fetch_option_chain", "nse_option_chain"),
    ):
        original = getattr(acquire_mod, name, None)
        if not callable(original):
            continue

        def wrapped(*args, __original=original, __provider=provider, **kwargs):
            publish("PROVIDER_CALL", __provider)
            value = __original(*args, **kwargs)
            publish("PROVIDER_COMPLETE", __provider)
            return value

        setattr(acquire_mod, name, wrapped)

    return acquire_mod, original_save


def _child_main(request_path: Path, result_path: Path, status_path: Path) -> int:
    request = _read_json(request_path)
    symbol = str(request.get("symbol") or "").strip().upper()
    timeout_s = max(1.0, float(request.get("timeout_s") or DEFAULT_SYMBOL_TIMEOUT_S))
    acquire_mod, _original_save = _patch_child_acquire(status_path, symbol)
    started = time.monotonic()
    try:
        _atomic_json(status_path, {
            "symbol": symbol,
            "stage": "ACQUIRING",
            "provider": "planner",
            "updated_at": time.time(),
            "pid": os.getpid(),
        })
        result = acquire_mod.acquire_symbol(
            symbol,
            force=bool(request.get("force")),
            datasets=list(request.get("datasets") or []) or None,
            deadline_monotonic=time.monotonic() + timeout_s,
        )
        _atomic_json(result_path, {
            "ok": True,
            "symbol": symbol,
            "elapsed_s": round(time.monotonic() - started, 3),
            "result": result,
        })
        _atomic_json(status_path, {
            "symbol": symbol,
            "stage": "COMPLETE",
            "provider": "done",
            "updated_at": time.time(),
            "pid": os.getpid(),
        })
        return 0
    except Exception as exc:
        status = _read_json(status_path)
        _atomic_json(result_path, {
            "ok": False,
            "symbol": symbol,
            "stage": str(status.get("stage") or "FAILED"),
            "provider": str(status.get("provider") or "unknown"),
            "elapsed_s": round(time.monotonic() - started, 3),
            "error": f"{type(exc).__name__}: {exc}"[:400],
            "traceback": traceback.format_exc(limit=8),
        })
        return 2


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--child", action="store_true")
    parser.add_argument("request", nargs="?")
    parser.add_argument("result", nargs="?")
    parser.add_argument("status", nargs="?")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if not args.child or not args.request or not args.result or not args.status:
        raise SystemExit("isolation.py is an internal due-diligence child entry point")
    return _child_main(Path(args.request), Path(args.result), Path(args.status))


if __name__ == "__main__":
    raise SystemExit(main())
