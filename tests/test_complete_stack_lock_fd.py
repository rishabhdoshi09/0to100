from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "run_quantterm_complete.sh"


def test_long_lived_children_close_machine_lock_fd() -> None:
    """Only the outer complete-stack supervisor may retain lock FD 200."""
    text = SCRIPT.read_text(encoding="utf-8")

    assert 'bash scripts/run_quantterm.sh 200>&- &' in text
    assert '>>"$ROOT/logs/stack/report_api.log" 2>&1 200>&- &' in text
    assert '>>"$ROOT/logs/stack/vite.log" 2>&1 200>&- &' in text


def test_launcher_documents_single_lock_owner_contract() -> None:
    text = SCRIPT.read_text(encoding="utf-8")

    assert "Only this outer supervisor may retain FD 200" in text
    assert "Every long-lived child explicitly closes FD 200" in text
