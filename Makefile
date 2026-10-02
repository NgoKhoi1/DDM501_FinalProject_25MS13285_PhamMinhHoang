.PHONY: up down build test lint train simulate-drift logs airflow-up airflow-down clean

up:
	docker compose up -d

down:
	docker compose down

build:
	docker compose build

test:
	pytest tests/ -v --cov=src --cov-report=term-missing

lint:
	flake8 src/ tests/ --max-line-length=120
	ruff check src/ tests/

train:
	python scripts/training.py

simulate-drift:
	python scripts/simulate_drift_and_retrain.py

logs:
	docker compose logs -f

airflow-up:
	docker compose -f airflow/docker-compose.airflow.yml up -d

airflow-down:
	docker compose -f airflow/docker-compose.airflow.yml down

clean:
	docker compose down -v --remove-orphans
	docker system prune -f
