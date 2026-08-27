#!/usr/bin/env bash
# Compatibility wrapper: execs scripts/run_quantterm_complete.sh (the canonical launcher).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT"

echo "[DESK] QuantTerm desk — one process. Ctrl-C stops API, UI and autonomy."

if ! command -v python3 >/dev/null 2>&1; then
  echo "[DESK] python3 is required." >&2
  exit 1
fi

if [[ ! -d venv ]]; then
  echo "[DESK] Creating venv…"
  python3 -m venv venv
fi
# shellcheck disable=SC1091
source venv/bin/activate

if ! python -c 'import fastapi, uvicorn, pypdf' >/dev/null 2>&1; then
  echo "[DESK] Installing Python packages (first run takes a few minutes)…"
  python -m pip install -r requirements.txt
fi

if [[ ! -d frontend/node_modules ]]; then
  if ! command -v npm >/dev/null 2>&1; then
    echo "[DESK] npm is required for the desk UI. Install Node.js, then re-run." >&2
    exit 1
  fi
  echo "[DESK] Installing frontend packages…"
  (cd frontend && npm install)
fi

if [[ ! -f .env ]]; then
  if [[ -f .env.example ]]; then
    cp .env.example .env
    chmod 600 .env 2>/dev/null || true
    echo "[DESK] Wrote .env from .env.example. Put KITE_API_KEY and KITE_API_SECRET in it, then re-run."
    exit 2
  fi
  echo "[DESK] Missing .env. Create it with KITE_API_KEY and KITE_API_SECRET." >&2
  exit 2
fi

echo "[DESK] Handing off to the complete one-terminal stack (API, desk, reports, autonomy, market scan)."
exec bash "$ROOT/scripts/run_quantterm_complete.sh" "$@"
