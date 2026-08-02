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
    """UI stack must not die on autonomy exit or a single busy-worker health flap."""
    text = (SCRIPTS / "run_quantterm.sh").read_text(encoding="utf-8")
    assert "Autonomy supervisor exited; leaving API + Vite running" in text
    assert "API_FAIL_LIMIT" in text
    assert "stack_port_listening 5173" in text
    assert "continuing with API + Vite only" in text


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
