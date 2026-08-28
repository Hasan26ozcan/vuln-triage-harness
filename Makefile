.PHONY: install test lint security scan typecheck up down

install:
	pip install -e ".[dev]"

test:
	pytest tests/unit -v --cov=app --cov-report=term-missing

lint:
	ruff check .

security:
	bandit -r app -q

scan:
	trivy fs --skip-dirs .venv,output --severity CRITICAL,HIGH .

typecheck:
	mypy app --config-file pyproject.toml

up:
	docker compose up -d

down:
	docker compose down
