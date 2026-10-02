#!/bin/bash
set -e

# Create additional databases for Airflow
psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" --dbname "$POSTGRES_DB" <<-EOSQL
    CREATE DATABASE airflow;
    GRANT ALL PRIVILEGES ON DATABASE airflow TO mlflow;
EOSQL

echo "Additional database 'airflow' created successfully!"
