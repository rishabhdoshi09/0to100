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


def test_restart_flag_stops_local_pids_before_start():
    complete = (ROOT / "scripts" / "run_quantterm_complete.sh").read_text(encoding="utf-8")
    inner = (ROOT / "scripts" / "run_quantterm.sh").read_text(encoding="utf-8")
    assert 'if [[ "${1:-}" == "--restart" ]]; then' in complete
    assert "python scripts/local_stack.py stop --ports 5173,8765,8766" in complete
    assert "QT_RESTART=1 bash scripts/run_quantterm.sh --restart" in complete
    assert "RESTART=0" in complete
    assert "python scripts/local_stack.py stop --ports 5173,8765" in inner
