.PHONY: install test lint security scan typecheck up infra down worker start-worker stop-worker celery-health

install:
	pip install -e ".[dev,data,ml]"

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

# Celery worker management
worker:
	celery -A app.celery_app worker --loglevel=info --concurrency=2 --pool=prefork

start-worker:
	docker compose -f docker-compose.infra.yml --profile worker up -d celery-worker

stop-worker:
	docker compose -f docker-compose.infra.yml --profile worker stop celery-worker

celery-health:
	celery -A app.celery_app inspect ping

# Clean up all containers and volumes
clean: down
	docker compose -f docker-compose.infra.yml --profile worker down --volumes
