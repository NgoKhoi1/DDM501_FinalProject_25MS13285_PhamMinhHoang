# Contributing

## Team and responsibilities

| Member | Student ID | Responsibilities |
|---|---|---|
| Pham Minh Hoang | 25MS13285 | TODO |
| Nguyen Cong Trong | 25MS13297 | TODO |
| Ngo Minh Khoi | 25MSA13236 | TODO |

<!-- Fill in one line per member. Areas of the project to divide:
     data pipeline and training (src/, airflow/), serving API (api/), drift detection (evidently/),
     monitoring and alerting (config/), testing and CI/CD (tests/, .github/), responsible AI, documentation. -->

## Workflow

1. Branch from `main`: `feature/<short-description>` or `fix/<short-description>`.
2. Make the change, with tests for new logic.
3. Run the checks locally (below).
4. Open a pull request to `main`. CI must pass and one other member reviews before merging.

`main` is deployed by the CI/CD pipeline, so it must always start with `docker compose up`.

## Local checks

```bash
pip install -r requirements.txt
make lint     # flake8 + ruff on src/ and tests/
make test     # pytest, coverage of src/ and api/ (CI fails under 80%)
```

For changes to the pipeline, the API or the drift service, also run the system end to end:

```bash
docker compose up -d --build
python scripts/training.py
python scripts/simulate_drift_and_retrain.py
```

## Conventions

- **Commits**: one logical change per commit, imperative subject with a type prefix:
  `feat: ...`, `fix: ...`, `test: ...`, `docs: ...`, `ci: ...`, `refactor: ...`.
- **Code**: line length 120, type hints and a docstring on public functions. ML logic goes in `src/` as plain
  functions; the Airflow DAG only orchestrates them.
- **Tests**: unit tests next to the module they cover (`tests/test_<module>.py`); anything that talks to
  MLflow, Airflow or Evidently is mocked in tests.
- **Versions**: `numpy` and `scikit-learn` are pinned to the same versions in `api/requirements.txt` and
  `airflow/Dockerfile`. Change them together, otherwise models trained in Airflow may not load in the API.
- **Data**: files in `data/` are the versioned training data. Do not edit them in place; add new files and
  extend `COLORS` in `src/data_pipeline.py`.
