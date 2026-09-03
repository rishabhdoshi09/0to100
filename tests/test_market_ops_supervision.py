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
    assert "worker.lock" in src
    assert "lock_pid" in src
    assert "SCAN_KICKED=0" in src


def test_launcher_waits_for_api_before_starting_frontend():
    src = _inner()
    initial = src.split("start_api || true", 1)[1].split('while [[ "$STOP" != "1" ]]', 1)[0]
    assert "wait_for_api" in initial
    assert initial.index("wait_for_api") < initial.index("start_frontend")
    assert initial.index("start_frontend") < initial.index("kick_scan")
    assert "frontend waits" in initial.lower() or "Frontend waits" in initial
    loop = src.split('while [[ "$STOP" != "1" ]]', 1)[1]
    assert loop.index('url_ok "http://127.0.0.1:8765/api/health"') < loop.index("start_frontend")
    assert loop.index("FRONTEND_EXTERNAL") < loop.index("Market operations is down/stale")
    assert "port_open 8765" in src.split("wait_for_api()", 1)[1].split("alive()", 1)[0]
    assert "desk waits until the market API is listening" in loop


def test_launcher_cleanup_owns_market_ops_process():
    src = _inner()
    assert 'MARKET_OPS_PID=""' in src
    assert 'for pid in "$FRONTEND_PID" "$API_PID" "$MARKET_OPS_PID" "$AUTONOMY_PID"' in src
    assert "market operations, market scan" in src


def test_launcher_restarts_dead_external_autonomy():
    src = _inner()
    loop = src.split('while [[ "$STOP" != "1" ]]', 1)[1]
    assert "Autonomy is down; restarting." in loop
    # Reused autonomy is re-probed every cycle, not trusted forever.
    assert loop.index("read_autonomy_status") < loop.index("Autonomy is down; restarting.")
    assert "AUTONOMY_EXTERNAL=0" in loop


def test_launcher_never_starts_api_while_port_is_listening():
    src = _inner()
    assert "adopt_api" in src
    assert "api_listening" in src
    start = src.split("start_api()", 1)[1].split("start_frontend()", 1)[0]
    assert "adopt_api" in start
    assert start.index("adopt_api") < start.index("Starting local API")
    loop = src.split('while [[ "$STOP" != "1" ]]', 1)[1]
    assert "Not killing it" in loop
    assert "adopt_api" in loop
    assert "scan_is_fresh" in src
    assert "not queueing another" in src


def test_complete_script_reuses_healthy_stack_without_second_inner():
    complete = (ROOT / "scripts" / "run_quantterm_complete.sh").read_text(encoding="utf-8")
    inner = _inner()
    assert "Another QuantTerm supervisor already owns this machine" in complete
    assert "adopt_report" in complete
    assert 'STACK_EXTERNAL' in complete
    assert '[[ "$STACK_EXTERNAL" != "1" ]]' in complete
    assert "will not stop :5173/:8765/:8766" in complete
    assert "Follower: another process owns the machine lock" in inner
    assert "Not starting or restarting" in inner
