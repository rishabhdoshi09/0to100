from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
STOP = ROOT / "scripts" / "stop_quantterm.sh"


def test_stop_command_is_owner_directed_and_fail_closed():
    text = STOP.read_text(encoding="utf-8")

    # Shutdown must address the durable complete-stack owner, not enumerate and
    # kill arbitrary listeners that happen to occupy product ports.
    assert "local_stack.py owner-status" in text
    assert 'kill -TERM "$OWNER_PID"' in text
    assert "run_quantterm_complete.sh" in text
    assert "pids_on_port" not in text
    assert "local_stack.py stop" not in text
    assert "pkill" not in text
    assert "kill -KILL" not in text


def test_operations_runbook_exposes_one_command_start_and_stop():
    text = (ROOT / "OPERATIONS.md").read_text(encoding="utf-8")
    assert "bash scripts/run_quantterm_complete.sh" in text
    assert "bash scripts/stop_quantterm.sh" in text
    assert "recorded complete-stack supervisor" in text
