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
The main runner is [`src/scripts/run_experiments.py`](/mnt/perma/ad_screening/src/scripts/run_experiments.py). If `docker/.env.postgres` exists, it auto-loads the local Postgres connection info and defaults `PREFECT_API_URL` to `http://127.0.0.1:4200/api`, so local runs target the Docker Compose Prefect server without extra shell setup.

Available configs:
- `experiments/exp_colab_test.yaml`
- `experiments/exp_colab_all_ablations.yaml`
- `experiments/exp_gcp_all_ablations_downscaled.yaml`
- `experiments/exp_gcp_all_ablations.yaml`
- `experiments/prefect_experiments_smoke.yaml`
- `experiments/prefect_experiments_multiseed.yaml`

Basic usage:
`python src/scripts/run_experiments.py --config experiments/prefect_experiments_smoke.yaml`

Useful examples:
- First local run with automatic data bootstrap: `python src/scripts/run_experiments.py --config experiments/prefect_experiments_smoke.yaml --bootstrap-data --local-server`
- Run with 4 Prefect task threads: `python src/scripts/run_experiments.py --config experiments/prefect_experiments_smoke.yaml --num-workers 4 --local-server`
- Write outputs to a custom directory: `python src/scripts/run_experiments.py --config experiments/prefect_experiments_smoke.yaml --results-dir scratch_results --local-server`
- Run the larger multiseed config: `python src/scripts/run_experiments.py --config experiments/prefect_experiments_multiseed.yaml --num-workers 4 --local-server`
- Run the Colab-sized smoke test config: `python src/scripts/run_experiments.py --config experiments/exp_colab_test.yaml --bootstrap-data`

CLI flags:
- `--config`: choose the experiment YAML file
- `--bootstrap-data`: download missing required inputs before launching the run
- `--num-workers`: set `PREFECT_TASK_RUNNER_THREAD_POOL_MAX_WORKERS`
- `--local-server`: explicitly target the local Prefect API at `http://127.0.0.1:4200/api`
- `--results-dir`: write run outputs somewhere other than `results/`

Each run root now includes reproducibility manifests:
- `dependency_manifest.json` (Python/Conda environment snapshot, pip freeze, lockfile hashes)

Workflow graph structure in Prefect:
- Parent flow: `ad-screening-experiment-batch`
- `ad-screening-experiment-batch` runs one full experiment config end to end. It prepares shared cohort artifacts, writes the dependency manifest, builds the experiment matrix, submits one child flow per matrix entry, and writes the top-level summary outputs.
- Parent flow tasks: ensure/download data, dependency manifest, summary artifacts
- Child flow per matrix entry: `ad-screening-experiment-run`
- `ad-screening-experiment-run` executes one matrix row. It materializes labels for that specific cohort/baseline/seed combination, runs the classifier probe, PPI signal analysis, and cosine-PPI hops analysis, then writes the per-run manifest and summary row.
- Per-run tasks: materialize labels, classifier probe, PPI signal, cosine hops, run manifest, summary row (classifier + PPI + hops metrics)
