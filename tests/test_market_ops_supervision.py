from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _inner() -> str:
    return (ROOT / "scripts" / "run_quantterm.sh").read_text(encoding="utf-8")


def test_launcher_starts_market_ops_before_user_scan_kick():
    src = _inner()
    assert "python -u -m operations.market_ops" in src
    assert "market_ops_healthy" in src
    assert src.index("start_market_ops || true") < src.index("kick_scan || true")
    assert "Scan kick waiting for market-operations worker" in src


def test_launcher_supervises_and_restarts_stale_market_ops():
    src = _inner()
    assert "Market operations is down/stale; restarting." in src
    assert "Reused market-operations worker became stale; taking ownership." in src
    assert "heartbeat_epoch" in src
    assert "time.time() - hb > 8" in src
    assert "SCAN_KICKED=0" in src


def test_launcher_cleanup_owns_market_ops_process():
    src = _inner()
    assert 'MARKET_OPS_PID=""' in src
    assert 'for pid in "$FRONTEND_PID" "$API_PID" "$MARKET_OPS_PID" "$AUTONOMY_PID"' in src
    assert "market operations, market scan" in src
