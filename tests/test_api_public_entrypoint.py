"""Architecture contracts for QuantTerm's stable public API seam."""
from __future__ import annotations

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ENTRYPOINT = ROOT / "api" / "app.py"


def test_public_api_entrypoint_delegates_to_canonical_runtime_without_second_application() -> None:
    """The public seam may own façade routes but must not construct a second FastAPI app."""
    source = ENTRYPOINT.read_text(encoding="utf-8")
    tree = ast.parse(source)

    # A second FastAPI() constructor here would create parallel route/runtime
    # ownership. The public seam delegates application ownership to api.runtime.
    calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)]
    assert not any(
        isinstance(call.func, ast.Name) and call.func.id == "FastAPI"
        for call in calls
    )
    assert "from . import runtime as _core" in source
    assert "app = _core.app" in source
    runtime = (ROOT / "api" / "runtime.py").read_text(encoding="utf-8")
    assert "import terminal_product_api as product" in runtime
    assert "app = product.app" in runtime

    legacy_facade = (ROOT / "terminal_product_api_parallel.py").read_text(encoding="utf-8")
    legacy_core = (ROOT / "_terminal_product_api_parallel_core.py").read_text(encoding="utf-8")
    assert "import api.app as _canonical" in legacy_facade
    assert "import api.runtime as _canonical" in legacy_core


def test_public_api_package_exists_at_stable_path() -> None:
    assert ENTRYPOINT.is_file()
    assert (ROOT / "api" / "__init__.py").is_file()
