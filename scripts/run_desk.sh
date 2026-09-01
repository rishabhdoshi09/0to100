#!/usr/bin/env bash
# Compatibility wrapper: execs scripts/run_quantterm_complete.sh (the canonical launcher).
# Bootstrap (venv, deps, frontend, .env) lives in the complete launcher so a
# fresh clone can run one command.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT"

echo "[DESK] Handing off to the complete one-terminal stack (API, desk, reports, autonomy, market scan)."
exec bash "$ROOT/scripts/run_quantterm_complete.sh" "$@"
