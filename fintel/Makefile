.PHONY: up down build migrate revision logs shell test lint

up:
	docker compose -f infra/docker-compose.yml up -d

down:
	docker compose -f infra/docker-compose.yml down

build:
	docker compose -f infra/docker-compose.yml build

migrate:
	docker compose -f infra/docker-compose.yml exec api alembic upgrade head

revision:
	docker compose -f infra/docker-compose.yml exec api alembic revision --autogenerate -m "$(MSG)"

logs:
	docker compose -f infra/docker-compose.yml logs -f $(SERVICE)

shell:
	docker compose -f infra/docker-compose.yml exec api bash

test:
	cd backend && pytest -v

lint:
	cd backend && ruff check . && ruff format --check .
