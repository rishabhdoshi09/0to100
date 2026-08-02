#!/usr/bin/env bash
# Stop local QuantTerm API, report API, and Vite dev server (macOS + Linux).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck source=stack_lib.sh
source "$ROOT/scripts/stack_lib.sh"

echo "[STOP] Stopping QuantTerm local services on ports 8765, 8766, 5173…"
# Best-effort: idle research watcher (no dedicated port).
pkill -f "idle_full_universe_backtest.py" >/dev/null 2>&1 || true
stack_stop_ports 8765 8766 5173
echo "[STOP] Done. Restart with: bash scripts/run_quantterm_complete.sh"
