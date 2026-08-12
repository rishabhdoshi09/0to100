#!/usr/bin/env bash
# Lean start for ~3GB RAM machines (Mac/Linux).
# Trading stack only — skips research-report API :8766 to save RAM.
# Market scan + autopilot feed still auto-start.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export QT_LOW_POWER=1
export QT_LEAN=1
export QT_DISABLE_IDLE_BACKTEST="${QT_DISABLE_IDLE_BACKTEST:-1}"
export QT_DISABLE_US_BOOTSTRAP="${QT_DISABLE_US_BOOTSTRAP:-1}"
unset QT_DISABLE_AUTO_MARKET_SCAN 2>/dev/null || true
unset QT_DISABLE_AUTO_LONG_TERM 2>/dev/null || true
echo "[LEAN] Trading stack only — no report API :8766 (saves RAM)."
echo "[LEAN] Market scan + autopilot still auto-start."
if [[ -x "$ROOT/venv/bin/python" ]]; then
  exec "$ROOT/venv/bin/python" "$ROOT/scripts/quantterm_stack.py" run --lean "$@"
fi
exec python3 "$ROOT/scripts/quantterm_stack.py" run --lean "$@"
