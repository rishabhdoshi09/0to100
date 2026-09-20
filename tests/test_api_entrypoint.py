from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_canonical_api_entrypoint_is_api_app():
    from api.app import app

    assert app is not None


def test_launcher_uses_only_canonical_api_entrypoint():
    launcher = (ROOT / "scripts" / "run_quantterm.sh").read_text(encoding="utf-8")
    assert "uvicorn api.app:app" in launcher
    assert "uvicorn terminal_product_api_parallel:app" not in launcher
    assert "uvicorn terminal_product_api:app" not in launcher


def test_handoff_docs_name_canonical_api_entrypoint():
    architecture = (ROOT / "ARCHITECTURE.md").read_text(encoding="utf-8")
    assert "api/app.py" in architecture
    assert "api.app:app" in architecture
