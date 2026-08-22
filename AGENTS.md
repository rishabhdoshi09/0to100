# AGENTS.md

QuantTerm — NSE India trading terminal (Streamlit retail UI, React terminal, autonomy supervisor). Branch of record: `overhaul/evidence-lab`. See `CLAUDE.md` for architecture.

## Cursor Cloud specific instructions

### First-time VM setup (not in update script)

Ubuntu images may lack `python3-venv`. If `python3 -m venv venv` fails:

```bash
sudo apt-get update && sudo apt-get install -y python3.12-venv
```

Then create the venv and install deps (see README quickstart):

```bash
cd /workspace
python3 -m venv venv
./venv/bin/pip install --upgrade pip
./venv/bin/pip install -r requirements.txt
cd frontend && npm install
```

Copy env template if missing: `cp .env.example .env`. Kite/Telegram/LLM keys are optional for UI boot and network-free tests; pages show honest offline states without them.

### Running services (use tmux — do not rely on one-shot background shells)

The production dev loop is **autonomy supervisor + UI**. Autonomy owns scans, paper trading, and data refresh; UIs are read-only observers.

| Service | Command | Port |
|---------|---------|------|
| Autonomy supervisor | `./venv/bin/python -u main.py autonomy` | (writes `logs/autonomy/status.json`) |
| Retail Streamlit UI | `./venv/bin/streamlit run app.py --server.headless true --server.port 8501 --server.address 127.0.0.1` | 8501 |
| Terminal FastAPI | `./venv/bin/python -u -m uvicorn terminal_product_api:app --host 127.0.0.1 --port 8765` | 8765 |
| React terminal (Vite) | `cd frontend && npm run dev -- --host 127.0.0.1 --port 5173` | 5173 |

**All-in-one:** `bash scripts/run_quantterm.sh` starts/reuses autonomy, API (`8765`), and Vite (`5173`) — requires `venv` and `frontend/node_modules` to exist.

**Legacy institutional UI:** `streamlit run legacy_app.py` (separate from retail `app.py`).

### Lint / test / build

| Task | Command |
|------|---------|
| Tests (CI parity, network-free) | `source venv/bin/activate && KITE_ACCESS_TOKEN="" TELEGRAM_BOT_TOKEN="" DEEPSEEK_API_KEY="" python -m pytest` |
| Import safety (CI) | `python -m compileall -q data scan risk execution ui reports ai core alerts news research product reporting operations fundamentals` plus `python -m py_compile terminal_api.py terminal_product_api.py report_api.py` |
| Frontend typecheck + build | `cd frontend && npm run build` |
| Ruff (installed in venv, not CI-gated) | `ruff check <paths>` |

Integration tests (`tests/integration`) require `QT_INTEGRATION=1` and may hit the network; CI runs them with `continue-on-error`.

### Health checks

- Streamlit: `curl -sf http://127.0.0.1:8501/_stcore/health`
- API readiness: `curl -sf http://127.0.0.1:8765/api/product-readiness`
- React dev server: `curl -sf -o /dev/null -w '%{http_code}' http://127.0.0.1:5173/`

### Gotchas

- **No Redis/Postgres** for the main QuantTerm stack — state lives under `logs/` (SQLite, JSON, pickle).
- **Fintel** (`fintel/`) is a separate Docker Compose product (`make -C fintel up`); not required for retail/React E2E.
- **Zerodha Kite** (`python main.py login`) is daily and only needed for live quotes/execution; offline dev is supported.
- **Telegram listener** is not started by retail `app.py`; legacy daemons need `QT_ENABLE_LEGACY_DAEMONS=1` or use the autonomy supervisor path instead.
