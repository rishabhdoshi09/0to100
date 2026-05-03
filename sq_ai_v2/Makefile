SHELL := /bin/bash
PYTHON := python3
VENV   := venv
PIP    := $(VENV)/bin/pip
PY     := $(VENV)/bin/python

.PHONY: install ingest train backtest api dashboard docker-up docker-down clean lint test

# ── Setup ────────────────────────────────────────────────────────────────────
install:
	$(PYTHON) -m venv $(VENV)
	$(PIP) install --upgrade pip wheel
	$(PIP) install -r requirements.txt
	@echo "✓ Virtual environment ready. Activate with: source $(VENV)/bin/activate"
	@cp -n .env.example .env 2>/dev/null && echo "✓ Created .env from .env.example" || echo "  .env already exists"

# ── Data ─────────────────────────────────────────────────────────────────────
ingest:
	$(PY) scripts/ingest_data.py

# ── Training ─────────────────────────────────────────────────────────────────
train:
	$(PY) scripts/train_models.py

# ── Backtest ─────────────────────────────────────────────────────────────────
backtest:
	$(PY) scripts/run_backtest.py

# ── API ──────────────────────────────────────────────────────────────────────
api:
	$(PY) -m uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload

# ── Dashboard ────────────────────────────────────────────────────────────────
dashboard:
	$(VENV)/bin/streamlit run dashboard/streamlit_app.py --server.port 8501

# ── Docker ───────────────────────────────────────────────────────────────────
docker-up:
	docker compose up -d
	@echo "Services starting..."
	@echo "  QuestDB  → http://localhost:9000"
	@echo "  Grafana  → http://localhost:3000  (admin/admin)"
	@echo "  API      → run 'make api' separately"

docker-down:
	docker compose down

docker-logs:
	docker compose logs -f

# ── Linting / Tests ──────────────────────────────────────────────────────────
lint:
	$(VENV)/bin/ruff check . --fix

test:
	$(VENV)/bin/pytest tests/ -v

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null; true
	find . -name "*.pyc" -delete 2>/dev/null; true
	rm -rf .pytest_cache dist build *.egg-info
