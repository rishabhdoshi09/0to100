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


def run_supervisor(*, root=None, interval_s: float = 15.0, max_iterations=None) -> int:
    """Entrypoint for `python main.py autonomy`. Acquires the single-instance lock and runs the loop.
    Returns a process exit code. Streamlit is NOT required."""
    from research.autonomy.supervisor import Supervisor
    sup = Supervisor(root or default_root())
    if not sup.start():
        print("autonomy: another supervisor instance is already running — exiting.")
        return 1
    print(f"autonomy: supervisor started (root={sup.root}). Ctrl-C to stop.")
    try:
        sup.run(interval_s=interval_s, max_iterations=max_iterations)
    except KeyboardInterrupt:
        print("\nautonomy: stopping…")
    finally:
        sup.shutdown()
    return 0
