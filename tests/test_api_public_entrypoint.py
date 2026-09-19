"""Architecture contracts for QuantTerm's stable public API seam."""
from __future__ import annotations

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ENTRYPOINT = ROOT / "api" / "app.py"


def test_public_api_entrypoint_is_thin_alias_not_second_application() -> None:
    """The public seam must not become a duplicate FastAPI authority."""
    source = ENTRYPOINT.read_text(encoding="utf-8")
    tree = ast.parse(source)

    # A second FastAPI() constructor here would create parallel route/runtime
    # ownership.  The seam may only re-export the established hardened app.
    calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)]
    assert not any(
        isinstance(call.func, ast.Name) and call.func.id == "FastAPI"
        for call in calls
    )
    assert "from terminal_product_api_parallel import app as app" in source
    assert '__all__ = ["app"]' in source


def test_public_api_package_exists_at_stable_path() -> None:
    assert ENTRYPOINT.is_file()
    assert (ROOT / "api" / "__init__.py").is_file()
