"""
QuantTerm autonomy — the durable closed loop that lets the organisation acquire, verify, observe,
hypothesise, challenge, test, paper-deploy, measure, learn and adapt without routine human input.

Thin orchestration over the existing canonical components. Never a second source of truth; never a
broker order; never live capital in this milestone.
"""
from __future__ import annotations

DEFAULT_ROOT = "logs/autonomy"


def default_root():
    from pathlib import Path
    return Path(__file__).resolve().parents[2] / "logs" / "autonomy"


def _existing_owner(root) -> tuple[str, str]:
    """Best-effort PID/state detail when another process owns the supervisor lock."""
    from pathlib import Path
    import json

    base = Path(root)
    pid = "unknown"
    state = "unknown"
    try:
        pid = (base / "supervisor.lock").read_text(encoding="utf-8").strip() or "unknown"
    except Exception:
        pass
    try:
        payload = json.loads((base / "status.json").read_text(encoding="utf-8"))
        state = str(payload.get("state", "unknown"))
        heartbeat = str(payload.get("heartbeat_ist", ""))
        if heartbeat:
            state = f"{state}, heartbeat {heartbeat}"
    except Exception:
        pass
    return pid, state


def run_supervisor(*, root=None, interval_s: float = 15.0, max_iterations=None) -> int:
    """Entrypoint for ``python main.py autonomy``.

    Acquires the single-instance lock and runs the visible, fault-contained console loop. Streamlit
    and the dedicated React terminal are optional observers; neither owns the scheduler.
    """
    import os
    import traceback

    from research.autonomy.operational_guards import install_operational_guards
    install_operational_guards()
    try:
        import scan.market_scan_service as _mss
        from research.feature002.observe import try_observe_production_scan
        if getattr(_mss, "_feature002_hook", None) is None:
            _mss._feature002_hook = try_observe_production_scan
    except Exception:
        pass

    # Heavy read/data work has a separate execution plane. Install this before
    # constructing Supervisor so scheduled scans can start in parallel with a
    # long DATA_REFRESH and corporate-action backfill never monopolises the
    # single mutation-owner loop.
    from research.autonomy.parallel_runtime import install_parallel_runtime
    install_parallel_runtime()

    from research.autonomy.supervisor import Supervisor
    from research.autonomy.console_runtime import run_visible_loop

    resolved_root = root or default_root()
    try:
        sup = Supervisor(resolved_root)
    except Exception as exc:
        print(f"autonomy: failed to initialise: {type(exc).__name__}: {exc}", flush=True)
        traceback.print_exc()
        return 2

    if not sup.start():
        pid, state = _existing_owner(resolved_root)
        print(
            "autonomy: another supervisor already owns the scheduler lock.\n"
            f"  owner pid : {pid}\n"
            f"  last state: {state}\n"
            "This command is returning because only one mutation owner is allowed.",
            flush=True,
        )
        return 1

    telegram_ok = False
    try:
        telegram_ok = bool(sup.deps.telegram.configured())
    except Exception:
        telegram_ok = False
    print(
        "\n=== QuantTerm Autonomy Supervisor ===\n"
        f"PID       : {os.getpid()}\n"
        f"Root      : {sup.root}\n"
        f"Interval  : {float(interval_s):.1f}s\n"
        "Mode      : PAPER only · LIVE broker orders locked\n"
        f"Telegram  : {'ON · sniper breakouts + scan alerts' if telegram_ok else 'OFF · set TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID in .env'}\n"
        "Console   : heartbeat every 30s + every completed job\n"
        "Stop      : Ctrl-C\n",
        flush=True,
    )
    try:
        run_visible_loop(
            sup,
            interval_s=interval_s,
            max_iterations=max_iterations,
        )
    except KeyboardInterrupt:
        print("\nautonomy: stopping on owner interrupt…", flush=True)
    except Exception as exc:
        # The visible loop already contains per-tick faults. This guard covers only driver-level faults.
        print(f"autonomy: fatal driver error: {type(exc).__name__}: {exc}", flush=True)
        traceback.print_exc()
        return 2
    finally:
        try:
            sup.shutdown()
        finally:
            print("autonomy: shutdown complete; scheduler lock released.", flush=True)
    return 0
