"""Guards for QuantTerm local stack launcher scripts."""
from __future__ import annotations

import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"


def test_uvicorn_starters_do_not_use_command_substitution() -> None:
    """Backgrounding inside $(...) kills the child when the subshell exits."""
    pattern = re.compile(r"\$\(\s*stack_start_or_reuse_uvicorn\b")
    offenders: list[str] = []
    for name in ("run_quantterm.sh", "run_quantterm_complete.sh"):
        text = (SCRIPTS / name).read_text(encoding="utf-8")
        if pattern.search(text):
            offenders.append(name)
    assert not offenders, (
        "stack_start_or_reuse_uvicorn must be called directly (not via $(...)); "
        f"offenders: {offenders}"
    )


def test_vite_is_not_started_in_subshell() -> None:
    """`( cd frontend; npm run dev ) &` leaves orphan Vite after the subshell PID dies."""
    text = (SCRIPTS / "run_quantterm.sh").read_text(encoding="utf-8")
    assert "npm --prefix" in text
    assert not re.search(r"\(\s*cd\s+frontend.*npm run dev", text, re.S), text


def test_stack_cleanup_reaps_vite_by_port() -> None:
    text = (SCRIPTS / "run_quantterm.sh").read_text(encoding="utf-8")
    complete = (SCRIPTS / "run_quantterm_complete.sh").read_text(encoding="utf-8")
    assert "stack_free_port 5173" in text
    assert "stack_free_port 5173" in complete


def test_stack_monitor_is_resilient() -> None:
    """UI stack must not die on autonomy exit or a busy-worker HTTP health flap."""
    text = (SCRIPTS / "run_quantterm.sh").read_text(encoding="utf-8")
    complete = (SCRIPTS / "run_quantterm_complete.sh").read_text(encoding="utf-8")
    assert "Autonomy supervisor exited; leaving API + Vite running" in text
    assert "API_DEAD_LIMIT" in text
    assert "process still up) — not exiting" in text
    assert "stack_port_listening 5173" in text
    assert "continuing with API + Vite only" in text
    # Low-CPU watch: long sleep + infrequent HTTP probes (not every 3s curl).
    assert 'WATCH_SLEEP_S="${QT_STACK_WATCH_SLEEP_S:-15}"' in text
    assert "HTTP_PROBE_EVERY" in text
    assert 'WATCH_SLEEP_S="${QT_STACK_WATCH_SLEEP_S:-15}"' in complete
    assert "stack_port_listening 8766" in complete
    assert "QT_AUTONOMY_INTERVAL_S" in text
    # Ctrl+C / SIGTERM must exit after cleanup — not resume the watch loop with
    # false "FRONTEND/Vite exited unexpectedly" alarms (nested + complete).
    assert "on_signal" in text
    assert "SHUTTING_DOWN" in text
    assert "on_signal" in complete
    assert "SHUTTING_DOWN" in complete
    assert 'trap cleanup EXIT' in text
    assert "trap on_signal INT TERM" in text
    # Vite should restart a few times before abandoning the stack.
    assert "VITE_RESTART_LIMIT" in text
    assert "start_vite" in text
    assert "vite.dev.log" in text
    # Clean nested exit (0) must not look like a crash to the complete wrapper.
    assert 'stack_ec" -eq 0' in complete or "stack_ec -eq 0" in complete
    # Report API flaps must not tear down the terminal backend.
    assert "not stopping backend" in complete
    assert "Leaving terminal API + Vite running" in complete
    assert "_restart_report_api" in complete
    assert "Report API health check failed" not in complete
    assert "REPORT_MONITOR" in complete
    assert (SCRIPTS / "run_quantterm_low_power.sh").exists()
    assert (SCRIPTS / "apply_low_power_env.sh").exists()
    low = (SCRIPTS / "apply_low_power_env.sh").read_text(encoding="utf-8")
    assert "QT_LOW_POWER=1" in low
    assert "QT_DISABLE_IDLE_BACKTEST=1" in low


def test_nested_stack_signal_exits_cleanly() -> None:
    """SIGTERM must run cleanup and exit 0 — not fall into the Vite-dead watch path."""
    script = r"""
set -euo pipefail
SHUTTING_DOWN=0
cleanup() {
  if [[ "$SHUTTING_DOWN" == "1" ]]; then
    return 0
  fi
  SHUTTING_DOWN=1
  echo CLEANUP
}
on_signal() {
  cleanup
  exit 0
}
trap cleanup EXIT
trap on_signal INT TERM

# Simulate the watch loop briefly, then deliver TERM like the complete wrapper.
(
  sleep 0.3
  kill -TERM $$
) &
while true; do
  if [[ "$SHUTTING_DOWN" == "1" ]]; then
    exit 0
  fi
  sleep 0.05
done
"""
    result = subprocess.run(
        ["bash", "-c", script],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )
    assert result.returncode == 0, result.stderr + result.stdout
    assert "CLEANUP" in result.stdout


def test_idle_backtest_watcher_is_quiet_while_waiting() -> None:
    text = (SCRIPTS / "idle_full_universe_backtest.py").read_text(encoding="utf-8")
    assert "waiting_idle" in text
    assert "QT_IDLE_BACKTEST_HEARTBEAT_S" in text


def test_terminal_health_endpoint_is_pure_liveness() -> None:
    import inspect

    import terminal_api

    source = inspect.getsource(terminal_api.health)
    assert "_autonomy_payload" not in source
    assert "_operations_payload" not in source
    detail = inspect.getsource(terminal_api.health_detail)
    assert "_autonomy_payload" in detail
    assert "_operations_payload" in detail


def test_stack_start_keeps_background_child_alive() -> None:
    """Direct call pattern must leave the backgrounded child running."""
    script = r"""
set -euo pipefail
source scripts/stack_lib.sh

# Simulate the uvicorn start path without needing Python deps.
stack_start_or_reuse_uvicorn() {
  local port="$1"
  STACK_UVICORN_PID=""
  STACK_UVICORN_RC=0
  sleep 30 &
  STACK_UVICORN_PID=$!
  STACK_UVICORN_RC=0
  return 0
}

stack_start_or_reuse_uvicorn 8766 "report_api:app" "http://127.0.0.1:8766/health" "Research-report API"
pid=$STACK_UVICORN_PID
sleep 0.2
if ! kill -0 "$pid" >/dev/null 2>&1; then
  echo "child died after starter returned" >&2
  exit 1
fi
kill "$pid" >/dev/null 2>&1 || true
echo OK
"""
    result = subprocess.run(
        ["bash", "-c", script],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "OK" in result.stdout
