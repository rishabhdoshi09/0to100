"""The product desk is Vite/React. Runtime libraries must not import Streamlit."""
from __future__ import annotations

import ast
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

RUNTIME_DIRS = (
    "ai",
    "analytics",
    "charting",
    "core",
    "data",
    "engine",
    "execution",
    "options",
    "product",
    "risk",
    "scan",
    "reports",
)
RUNTIME_FILES = (
    "app.py",
    "terminal_product_api.py",
    "pages/2_F&O_Momentum.py",
)
DEPLOY_FILES = (
    "deploy/setup_mac.sh",
    "deploy/setup_server.sh",
    "deploy/quantterm-ui.service",
    "scripts/run_desk.sh",
    "scripts/run_quantterm.sh",
    "scripts/run_quantterm_complete.sh",
)


def _py_files():
    for folder in RUNTIME_DIRS:
        yield from (ROOT / folder).glob("*.py")
    for rel in RUNTIME_FILES:
        yield ROOT / rel


def _imports_streamlit(path: Path) -> bool:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            if any(alias.name.split(".", 1)[0] == "streamlit" for alias in node.names):
                return True
        elif isinstance(node, ast.ImportFrom):
            if node.module and node.module.split(".", 1)[0] == "streamlit":
                return True
    return False


def test_runtime_libraries_do_not_import_streamlit():
    offenders = [str(path.relative_to(ROOT)) for path in _py_files() if _imports_streamlit(path)]
    assert offenders == []


def test_app_py_is_a_desk_stub():
    src = (ROOT / "app.py").read_text(encoding="utf-8")
    assert "import streamlit" not in src
    assert "st.Page" not in src and "st.navigation" not in src
    assert "run_quantterm_complete.sh" in src
    assert "run_desk.sh" in src
    assert "127.0.0.1:5173" in src


def test_deploy_and_run_scripts_do_not_start_streamlit():
    for rel in DEPLOY_FILES:
        src = (ROOT / rel).read_text(encoding="utf-8")
        assert "streamlit" not in src, rel
        assert "8501" not in src, rel


def test_product_api_import_does_not_load_streamlit(monkeypatch):
    monkeypatch.delitem(sys.modules, "streamlit", raising=False)
    import importlib

    importlib.invalidate_caches()
    import terminal_product_api  # noqa: F401
    import scan.market_scan_service  # noqa: F401
    import core.regime_engine  # noqa: F401
    import core.sector_rotation  # noqa: F401
    import core.regime_drift  # noqa: F401
    import core.trade_score  # noqa: F401
    import data.fii_dii  # noqa: F401
    import options.analytics  # noqa: F401
    import engine.trade_engine  # noqa: F401
    import risk.portfolio_risk  # noqa: F401
    import product.trade_plan  # noqa: F401

    assert "streamlit" not in sys.modules
