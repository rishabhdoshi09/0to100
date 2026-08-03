#!/usr/bin/env bash
# Stop local QuantTerm API, report API, and Vite dev server (macOS + Linux).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck source=stack_lib.sh
source "$ROOT/scripts/stack_lib.sh"

echo "[STOP] Stopping QuantTerm local services on ports 8765, 8766, 5173…"
# Best-effort: idle research watcher + breakout sniper (no dedicated port).
pkill -f "idle_full_universe_backtest.py" >/dev/null 2>&1 || true
pkill -f "scan.sniper_runtime" >/dev/null 2>&1 || true
pkill -f "python -u -m scan.sniper_runtime" >/dev/null 2>&1 || true
# Market-ops is a child of the API and has no port. If left alive it keeps the
# worker.lock and the next stack cannot lease MARKET_SCAN (stays PENDING forever).
pkill -f "operations.market_ops" >/dev/null 2>&1 || true
pkill -f "python -u -m operations.market_ops" >/dev/null 2>&1 || true
stack_stop_ports 8765 8766 5173
# Clear stale runtime so UI does not think a dead worker is still coming online.
rm -f "$ROOT/logs/market_ops/runtime.json" "$ROOT/logs/market_ops/worker.lock" 2>/dev/null || true
rm -f "$ROOT/logs/sniper/runtime.json" 2>/dev/null || true
echo "[STOP] Done. Restart with: bash scripts/run_quantterm_complete.sh"
