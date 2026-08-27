"""The complete stack is one terminal and one command, including the market scan."""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_complete_script_starts_every_local_service_in_one_process_tree():
    complete = (ROOT / "scripts" / "run_quantterm_complete.sh").read_text(encoding="utf-8")
    inner = (ROOT / "scripts" / "run_quantterm.sh").read_text(encoding="utf-8")
    desk = (ROOT / "scripts" / "run_desk.sh").read_text(encoding="utf-8")

    assert complete.startswith("#!/usr/bin/env bash")
    assert "run_quantterm.sh" in complete
    assert "python main.py login" in complete
    assert "report_api:app" in complete
    assert "terminal_product_api:app" in inner
    assert "npm run dev" in inner
    assert "python -u main.py autonomy" in inner
    assert "scripts/local_stack.py scan" in inner
    assert "curl" not in complete
    assert "curl" not in inner
    assert "Do not start a second terminal" in complete
    assert 'exec bash "$ROOT/scripts/run_quantterm_complete.sh" "$@"' in desk


def test_complete_script_always_stops_old_stack_then_starts_everything():
    complete = (ROOT / "scripts" / "run_quantterm_complete.sh").read_text(encoding="utf-8")
    inner = (ROOT / "scripts" / "run_quantterm.sh").read_text(encoding="utf-8")
    assert "python scripts/local_stack.py stop --ports 5173,8765,8766" in complete
    assert "python scripts/local_stack.py stop --ports 5173,8765" in inner
    assert "One command, one terminal" in complete
    assert "scripts/local_stack.py scan" in inner
    assert "run_quantterm_complete.sh --restart" not in inner
    assert 'url_ok "http://127.0.0.1:8766/health"' in complete
    assert complete.count('url_ok "http://127.0.0.1:8766/health"') == 1
    assert 'alive "$REPORT_PID"' in complete
    assert "Use --restart" not in inner
    assert "Use --restart" not in complete
