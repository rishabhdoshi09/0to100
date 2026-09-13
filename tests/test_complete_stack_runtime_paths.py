from __future__ import annotations

from pathlib import Path


def test_complete_stack_uses_canonical_runtime_logs() -> None:
    root = Path(__file__).resolve().parents[1]
    script = (root / "scripts" / "run_quantterm_complete.sh").read_text(encoding="utf-8")

    assert "from core.runtime_paths import logs_dir" in script
    assert 'STACK_LOG_DIR="$RUNTIME_LOGS/stack"' in script
    assert '$ROOT/logs/stack' not in script
    assert '>>"$STACK_LOG_DIR/report_api.log"' in script
    assert 'INNER_HEARTBEAT_FILE="$STACK_LOG_DIR/inner_supervisor.heartbeat"' in script
    assert '>>"$STACK_LOG_DIR/vite.log"' in script
