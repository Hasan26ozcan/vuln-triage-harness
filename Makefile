.PHONY: install test lint security scan typecheck up infra down

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

infra:
	docker compose -f docker-compose.infra.yml up -d

down:
	docker compose -f docker-compose.infra.yml down
	docker compose -f docker-compose.yml --profile gpu down

up: infra
	docker compose -f docker-compose.yml --profile gpu up serving-gpu -d
