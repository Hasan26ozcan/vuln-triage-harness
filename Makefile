.PHONY: install test lint security up down

install:
	pip install -e ".[dev]"

test:
	pytest tests/unit -v --cov=app --cov-report=term-missing

lint:
	ruff check .

security:
	bandit -r app -q

up:
	docker compose up -d

down:
	docker compose down
