#!/usr/bin/env bash
# Low-power Mac/Linux start — mirrors Windows low-power mode.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export QT_LOW_POWER=1
export QT_DISABLE_IDLE_BACKTEST="${QT_DISABLE_IDLE_BACKTEST:-1}"
if [[ -x "$ROOT/venv/bin/python" ]]; then
  exec "$ROOT/venv/bin/python" "$ROOT/scripts/quantterm_stack.py" run --low-power "$@"
fi
exec python3 "$ROOT/scripts/quantterm_stack.py" run --low-power "$@"
