#!/usr/bin/env bash
# Stop local QuantTerm API, report API, and Vite dev server (macOS + Linux).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck source=stack_lib.sh
source "$ROOT/scripts/stack_lib.sh"

echo "[STOP] Stopping QuantTerm local services on ports 8765, 8766, 5173…"
for port in 8765 8766 5173; do
  stack_free_port "$port" "port-${port}"
done
echo "[STOP] Done. Restart with: bash scripts/run_quantterm_complete.sh"
