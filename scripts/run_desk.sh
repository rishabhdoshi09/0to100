#!/usr/bin/env bash
# One terminal: venv, deps, daily Zerodha login if needed, then API + desk + autonomy.
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

auth_rc=0
python - <<'PY' >/dev/null || auth_rc=$?
from data.kite_client import _fresh_env

if not _fresh_env("KITE_API_KEY") or not _fresh_env("KITE_API_SECRET"):
    raise SystemExit(2)
if not _fresh_env("KITE_ACCESS_TOKEN"):
    raise SystemExit(1)
try:
    from research.autonomy.auth import TOKEN_MISSING, SESSION_EXPIRED, probe_auth
    health = probe_auth()
    if health.valid:
        raise SystemExit(0)
    if health.status in {TOKEN_MISSING, SESSION_EXPIRED}:
        raise SystemExit(1)
except Exception:
    pass
raise SystemExit(0)
PY

if [[ "$auth_rc" -eq 2 ]]; then
  echo "[DESK] Fill KITE_API_KEY and KITE_API_SECRET in .env, then run this command again." >&2
  exit 2
fi

if [[ "$auth_rc" -eq 1 ]]; then
  echo "[DESK] Zerodha login is needed (once per trading day). Browser will open; paste the redirect URL here."
  python main.py login
fi

echo "[DESK] Starting API :8765, UI :5173, and autonomy in this terminal."
echo "[DESK] Open http://127.0.0.1:5173"
exec bash "$ROOT/scripts/run_quantterm.sh"
