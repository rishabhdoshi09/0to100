"""Production product paths must not import demo market fixtures."""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

PRODUCTION_PATHS = (
    "core/regime_engine.py",
    "scan/quality_engine.py",
    "scan/market_scan_service.py",
    "scan/unified_scanner.py",
    "product/market_view.py",
    "product/recommendations_workspace.py",
    "product/recommendations_liveness.py",
    "operations/market_ops.py",
    "terminal_api.py",
    "analytics/regime_engine.py",
)


def test_production_modules_do_not_import_demo_success_fixtures():
    for rel in PRODUCTION_PATHS:
        src = (ROOT / rel).read_text(encoding="utf-8")
        assert "from core.demo_data" not in src, rel
        assert "DEMO_REGIME" not in src, rel
        assert "make_demo_ohlcv" not in src, rel


def test_legacy_streamlit_pages_do_not_silently_return_demo_holdings_or_ipos():
    holdings = (ROOT / "ui/real_holdings.py").read_text(encoding="utf-8")
    assert "holdings = _DEMO_HOLDINGS" not in holdings
    ipos = (ROOT / "ui/ipo_calendar.py").read_text(encoding="utf-8")
    assert "return _DEMO_IPOS" not in ipos
    replay = (ROOT / "ui/trade_replay.py").read_text(encoding="utf-8")
    assert "return _demo_trades(), True" not in replay
    assert "df_full = _synthetic_ohlcv" not in replay
