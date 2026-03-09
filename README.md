# Brain-region–aware, BBB-conditioned virtual screening for Alzheimer’s disease

## Setup
- Create the env: `conda env create -f environment.yml`
- Activate it: `conda activate ad_screening`
- Install the package in editable mode: `python -m pip install -e .`
- Install git hooks: `pre-commit install`

After modifying deps in `environment.yml`,  sync env with: `conda env update -n ad_screening -f environment.yml --prune`

## Local Postgres (Docker)
- Files live under [`docker/`](/mnt/perma/ad_screening/docker).
- Copy the env template: `cp docker/.env.postgres.example docker/.env.postgres`
- Start Postgres: `sudo docker compose -f docker/docker-compose.yml up -d`
- Stop it: `docker compose -f docker/docker-compose.yml down`

This setup uses one shared Postgres database, `tracking`, for both MLflow and Prefect.
They keep separate tables in the same database.

Connection examples:
- MLflow backend URI: `postgresql://ad_screening:pg_password_123@127.0.0.1:5432/tracking`
- Prefect API database URI: `postgresql+asyncpg://ad_screening:pg_password_123@127.0.0.1:5432/tracking`

Start Prefect against that Postgres instance instead of SQLite:
`bash src/scripts/start_prefect_server.sh`

## Data
Download and extract all datasets: `python src/scripts/download_data.py`

## Prefect Experiment Workflow
Default config: `experiments/prefect_experiments.yaml`

Run matrix and auto-bootstrap missing downloaded inputs (run first time):
`python src/scripts/run_prefect_experiments.py --bootstrap-data`

Run matrix:
`python src/scripts/run_prefect_experiments.py`

The runner auto-loads `docker/.env.postgres` and sets `PREFECT_API_DATABASE_CONNECTION_URL` plus `PREFECT_API_URL=http://127.0.0.1:4200/api` before importing Prefect, so local runs use the Docker Postgres-backed Prefect server by default when that file exists.

Smoke test only first N runs:
Configure experiment matrix size in `experiments/prefect_experiments.yaml`.

Each run root now includes reproducibility manifests:
- `dependency_manifest.json` (Python/Conda environment snapshot, pip freeze, lockfile hashes)

Workflow graph structure in Prefect:
- Parent flow: `ad-screening-experiments`
- Parent flow tasks: ensure/download data, dependency manifest, final aggregation
- Child flow per matrix entry: `ad-screening-matrix-run`
- Per-run tasks: materialize labels, classifier probe, PPI signal, cosine hops, run manifest, summary row (classifier + PPI + hops metrics)

Standalone aggregate step (optional, already run by the Prefect parent flow):
`python src/scripts/aggregate_experiment_results.py --run-root results/prefect_experiments_YYYYMMDD_HHMMSS`
