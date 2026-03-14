# Brain-region–aware, BBB-conditioned virtual screening for Alzheimer’s disease

## Setup
- Create the env: `conda env create -f environment.yml`
- Activate it: `conda activate ad_screening`
- Install the package in editable mode: `python -m pip install -e .`
- Install git hooks: `pre-commit install`

After modifying deps in `environment.yml`,  sync env with: `conda env update -n ad_screening -f environment.yml --prune`

## Local Prefect + Postgres (Docker)
- Files live under [`docker/`](/mnt/perma/ad_screening/docker).
- Copy the env template: `cp docker/.env.postgres.example docker/.env.postgres`
- Start Postgres and the Prefect API server: `docker compose -f docker/docker-compose.yml up -d`
- Start only the Prefect server in the foreground: `docker compose -f docker/docker-compose.yml up prefect-server`
- Check service status: `docker compose -f docker/docker-compose.yml ps`
- Tail logs: `docker compose -f docker/docker-compose.yml logs -f prefect-server`
- Stop it: `docker compose -f docker/docker-compose.yml down`

This setup uses one shared Postgres database, `tracking`, for both MLflow and Prefect.
They keep separate tables in the same database.

Connection examples:
- MLflow backend URI: `postgresql://ad_screening:pg_password_123@127.0.0.1:5432/tracking`
- Prefect API database URI: `postgresql+asyncpg://ad_screening:pg_password_123@127.0.0.1:5432/tracking`
- Prefect API URL: `http://127.0.0.1:4200/api`

## Data
Download and extract all datasets: `python src/scripts/download_data.py`

## Run Experiments
The main launcher is [`src/scripts/run_experiment.py`](/mnt/perma/ad_screening/src/scripts/run_experiment.py). It takes an experiment name and resolves the matching config at `experiments/{name}.yaml`, the matching script at `src/scripts/{name}.py`, and writes outputs under `results/{name}/...`. The launcher owns Prefect integration and experiment-YAML expansion, then executes fully specified `ad_predictor.py` subprocesses so the experiment script itself only accepts raw CLI arguments.

Basic usage:
`python src/scripts/run_experiment.py --experiment-name ad_predictor`

Useful examples:
- First local run with automatic data bootstrap: `python src/scripts/run_experiment.py --experiment-name ad_predictor --bootstrap-data --local-server`
- Run with 4 Prefect task threads: `python src/scripts/run_experiment.py --experiment-name ad_predictor --num-threads 4 --local-server`
- Write outputs to a custom directory: `python src/scripts/run_experiment.py --experiment-name ad_predictor --results-dir scratch_results --local-server`
- Run the experiment script directly for one concrete configuration: `python src/scripts/ad_predictor.py --output-dir results/ad_predictor/manual_run --seed 42 --no-label-shuffle --no-random-embeddings --hidden-dim 64 --loss-selection bce`

CLI flags:
- `--experiment-name`: choose the experiment, matched across `experiments/`, `src/scripts/`, and `results/`
- `--bootstrap-data`: download missing required inputs before launching the run
- `--local-server`: force the launcher's Prefect flow run to register against the local Prefect API at `http://127.0.0.1:4200/api` and fail fast if that server is unavailable
- `--num-threads`: set `PREFECT_TASK_RUNNER_THREAD_POOL_MAX_WORKERS` plus common BLAS/OpenMP thread env vars for the launched experiment process
- `--results-dir`: write run outputs somewhere other than `results/`
