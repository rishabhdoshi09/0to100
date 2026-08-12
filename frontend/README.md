# QuantTerm Terminal

This is the dedicated product frontend for QuantTerm. It is intentionally separate from Streamlit.

## Architecture

- `terminal_product_api.py` (via `scripts/run_quantterm.sh`) is the full product API surface.
- Legacy `terminal_api.py` alone omits scanner-workspace, stock-intelligence, and data platform routes.
- `frontend/` is a React + TypeScript + Vite application.
- Scanner, paper book, market regime, long-term shortlist and autonomy stores remain authoritative.
- The frontend never submits broker orders and does not run scanners or trading loops.

## Run

Keep the autonomy supervisor running in one terminal:

```bash
python main.py autonomy
```

Start the dedicated UI in another terminal:

```bash
bash scripts/run_quantterm.sh
# or full stack (reports on :8766): bash scripts/run_quantterm_complete.sh
# run_terminal.sh is an alias that starts the same product API (terminal_product_api)
```

Then open:

```text
http://127.0.0.1:5173
```

## Manual development

```bash
source venv/bin/activate
python -m uvicorn terminal_product_api:app --host 127.0.0.1 --port 8765
```

In a second terminal:

```bash
cd frontend
npm install
npm run dev
```

## Truthfulness rules

- Missing data renders as unavailable, never as a fabricated metric.
- Current fundamentals are not treated as point-in-time research evidence.
- Controls are queued to the autonomy supervisor; the API does not execute jobs directly.
- LIVE broker order submission is outside this frontend.

## React terminal vs Streamlit

| React terminal (`frontend/`) | Streamlit only (`app.py`, `legacy_app.py`, `ui/*`) |
|------------------------------|-----------------------------------------------------|
| Market radar, scanner, stock workspace, paper portfolio | JARVIS chat, AlgoLab, journal, alerts, order pad |
| Research Data + evidence uploads | Full research lab, strategy studio |
| System Health + institutional stack (read-only) | Iron Lock, live OMS ticket UI |
| F&O derivative **coverage** desk | Options flow scanners, VCP labs |

Inspectors should not flag Streamlit-only surfaces as React bugs unless they are duplicated with conflicting data paths.
