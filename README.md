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
The main launcher is [`src/scripts/run_experiment.py`](/mnt/perma/ad_screening/src/scripts/run_experiment.py). It takes an experiment name and resolves the matching config at `experiments/{name}.yaml`, the matching script at `src/scripts/{name}.py`, and writes outputs under `results/{name}/...`. The launcher is thin; each experiment script is responsible for running its own full sweep.

Basic usage:
`python src/scripts/run_experiment.py --experiment-name ad_predictor`

Useful examples:
- First local run with automatic data bootstrap: `python src/scripts/run_experiment.py --experiment-name ad_predictor --bootstrap-data --local-server`
- Run with 4 Prefect task threads: `python src/scripts/run_experiment.py --experiment-name ad_predictor --num-workers 4 --local-server`
- Write outputs to a custom directory: `python src/scripts/run_experiment.py --experiment-name ad_predictor --results-dir scratch_results --local-server`
- Override the config path explicitly: `python src/scripts/run_experiment.py --experiment-name ad_predictor --config experiments/ad_predictor.yaml`
- Run the experiment script directly: `python src/scripts/ad_predictor.py --config experiments/ad_predictor.yaml --results-dir results/ad_predictor`

CLI flags:
- `--experiment-name`: choose the experiment, matched across `experiments/`, `src/scripts/`, and `results/`
- `--config`: override the default experiment YAML path when needed
- `--bootstrap-data`: download missing required inputs before launching the run
- `--local-server`: explicitly target the local Prefect API at `http://127.0.0.1:4200/api`
- `--results-dir`: write run outputs somewhere other than `results/`
