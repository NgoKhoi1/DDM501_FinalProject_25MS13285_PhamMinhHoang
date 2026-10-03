.PHONY: up down build test lint train simulate-drift logs clean

up:
	docker compose up -d

down:
	docker compose down

build:
	docker compose build

test:
	pytest tests/ -v --cov=src --cov=api --cov-report=term-missing

lint:
	flake8 src/ tests/
	ruff check src/ tests/

train:
	python scripts/training.py

simulate-drift:
	python scripts/simulate_drift_and_retrain.py

logs:
	docker compose logs -f

clean:
	docker compose down -v --remove-orphans
	docker system prune -f
