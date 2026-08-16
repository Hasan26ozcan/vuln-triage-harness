.PHONY: install test lint security scan up down

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

up:
	docker compose up -d

down:
	docker compose down
