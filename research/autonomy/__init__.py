"""
QuantTerm autonomy — the durable closed loop that lets the organisation acquire, verify, observe,
hypothesise, challenge, test, paper-deploy, measure, learn and adapt without routine human input.

Thin orchestration over the existing canonical components. Never a second source of truth; never a
broker order; never live capital in this milestone.
"""
from __future__ import annotations
from core.runtime_paths import logs_dir, logs_path

DEFAULT_ROOT = "logs/autonomy"


def default_root():
    from pathlib import Path
    return logs_path("autonomy")


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


class PipeWake:
    """A LOCK-FREE wake for the loop's sleep, safe to set from a signal handler.

    A signal handler runs in the main thread between bytecodes, possibly while that thread
    is inside threading.Event's own critical section (entering or leaving every wait):
    Event.set() then blocks forever on the Event's non-reentrant lock (reproduced: a
    SIGTERM during ``with wake._cond`` hung until killed; GPT review of 24 Nov). ``set``
    writes one byte to a non-blocking pipe (no Python lock), ``wait`` selects on it.
    """

    def __init__(self) -> None:
        import os

        self._r, self._w = os.pipe()
        os.set_blocking(self._r, False)
        os.set_blocking(self._w, False)

    def set(self) -> None:
        import os

        try:
            os.write(self._w, b"x")
        except BlockingIOError:
            pass                                  # a wake is already pending

    def wait(self, timeout: float | None = None) -> bool:
        import os
        import select

        ready, _, _ = select.select([self._r], [], [], None if timeout is None else max(0.0, float(timeout)))
        if not ready:
            return False
        try:
            os.read(self._r, 4096)
        except BlockingIOError:
            pass
        return True


def install_sigterm_as_stop(supervisor, wake) -> bool:
    """Make SIGTERM a COOPERATIVE stop: ask the loop to finish, never interrupt a job.

    The host supervisor stops its children with SIGTERM, whose default action kills the
    process outright: no ``finally``, no ``atexit``, so the decision-freeze keepers
    (product.sqlite_runtime.keep_open) never checkpoint and leave a durable but hot WAL that
    the F1 repair tool and byte copies must refuse (cycle-8 review, part 7). Raising an
    exception at an arbitrary point instead was rejected (GPT C8.7-01/02): it can split a
    job-store write from its commit on a connection the heartbeat shares, and broad handlers
    swallow it. This handler only calls ``supervisor.stop()`` (a plain attribute store) and
    sets ``wake``, a PipeWake the loop sleeps on (lock-free: an Event deadlocks when the
    signal lands inside its own critical section), so the loop exits between ticks, ``sup.shutdown()`` runs and the
    interpreter exits normally. A job already running finishes, or the host's SIGKILL
    arrives after its grace period, exactly as before. Only from the main thread, never
    over an existing handler. Returns whether it was installed.
    """
    import signal
    import threading

    if threading.current_thread() is not threading.main_thread():
        return False
    if signal.getsignal(signal.SIGTERM) is not signal.SIG_DFL:
        return False

    def _stop(_signum, _frame):
        supervisor.stop()
        wake.set()

    signal.signal(signal.SIGTERM, _stop)
    return True


def install_runtime_handlers() -> None:
    """Install every production handler wrapper, in production order.

    Several modules REPLACE entries in research.autonomy.jobs.HANDLERS
    (outcome_liveness, outcome_integrity, data_refresh_parallel, ...). The
    nightly BUY-outcome settlement was lost for months because a fix went
    into a replaced handler. Keeping the sequence in one function lets tests
    exercise exactly the chain run_supervisor() installs.
    """
    from research.autonomy.operational_guards import install_operational_guards
    install_operational_guards()
    try:
        import scan.market_scan_service as _mss
        from research.feature002.observe import try_observe_production_scan
        if getattr(_mss, "_feature002_hook", None) is None:
            _mss._feature002_hook = try_observe_production_scan
    except Exception:
        pass

    # Exchange feeds can describe distributions of preference shares as "bonus".
    # They are genuine corporate actions but do not multiply ordinary-equity share
    # count, so install the security-type guard before any CA provider is used.
    from data.corporate_actions_equity_filter import install as install_ca_equity_filter
    install_ca_equity_filter()

    # BSE's official wire format differs from NSE (e.g. "25 Oct 2023").
    # Install the provider adapter before the background CA worker imports the
    # canonical resilient refresh function.
    from data.corporate_actions_bse_adapter import install as install_bse_ca_adapter
    install_bse_ca_adapter()

    # Heavy read/data work has a separate execution plane. Install this before
    # constructing Supervisor so scheduled scans can start in parallel with a
    # long DATA_REFRESH and corporate-action backfill never monopolises the
    # single mutation-owner loop.
    from research.autonomy.parallel_runtime import install_parallel_runtime
    install_parallel_runtime()

    # DATA_REFRESH used to remain inside Supervisor.tick() for the full Kite
    # history/snapshot catch-up. Keep the canonical handler and all of its data
    # gates, but execute that heavy I/O in a background data lane so controls,
    # health and other due jobs remain responsive while the refresh is running.
    from research.autonomy.data_refresh_parallel import install_parallel_data_refresh
    install_parallel_data_refresh()

    # OUTCOME_RESOLUTION must be a bounded EOD settlement lane. The full intelligence
    # cycle evaluates strategies and may run historical evidence; doing that here used
    # to keep a critical job alive for many minutes and starve the rest of the desk.
    from research.autonomy.outcome_liveness import install_outcome_liveness
    install_outcome_liveness()

    # A partial official counterfactual settlement must not report SUCCESS and unblock
    # learning. Successful rows are idempotent, so the same durable job can retry only
    # the unresolved rows after a provider/decoder failure.
    from research.autonomy.outcome_integrity import install_outcome_integrity
    install_outcome_integrity()

    # PAPER_CYCLE has two phases: management/evidence and the recommendation
    # selection authority. The first intentionally disables new entries, so its
    # mechanical BLOCKED_SAFETY must never overwrite the later canonical PaperBook
    # result. Install the composer before Supervisor imports/constructs Deps.
    from research.autonomy.paper_cycle_truth import install_paper_cycle_truth
    install_paper_cycle_truth()

    # The legacy lock unlinked its flock path on release. A waiting process can acquire
    # the old inode between unlock and unlink while a third process creates a new inode,
    # producing two apparent owners. Install the non-unlinking flock before Supervisor
    # construction; the lock file is stable and its PID is rewritten only after ownership.
    from research.autonomy.safe_singleton_lock import install_supervisor_singleton_lock
    install_supervisor_singleton_lock()


def run_supervisor(*, root=None, interval_s: float = 15.0, max_iterations=None) -> int:
    """Entrypoint for ``python main.py autonomy``.

    Acquires the single-instance lock and runs the visible, fault-contained console loop. Streamlit
    and the dedicated React terminal are optional observers; neither owns the scheduler.
    """
    import os
    import traceback

    install_runtime_handlers()

    from research.autonomy.supervisor import Supervisor
    from research.autonomy.console_runtime import run_visible_loop

    resolved_root = root or default_root()
    try:
        sup = Supervisor(resolved_root)
    except Exception as exc:
        print(f"autonomy: failed to initialise: {type(exc).__name__}: {exc}", flush=True)
        traceback.print_exc()
        return 2
    wake = PipeWake()                               # lock-free: set from the signal handler
    cooperative = install_sigterm_as_stop(sup, wake)  # before start(): a stop then still ends cleanly

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
            sleep_fn=wake.wait,                     # a SIGTERM wakes the sleep at once
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
            if cooperative:
                # The stop is done; a LATER SIGTERM (repeated, or arriving during interpreter
                # finalization after Python restored the default handlers) must not turn this
                # clean exit into a kill. atexit (the keepers' checkpoints) still runs; the
                # host's SIGKILL after its grace period still works.
                import signal

                signal.signal(signal.SIGTERM, signal.SIG_IGN)
            print("autonomy: shutdown complete; scheduler lock released.", flush=True)
    return 0
