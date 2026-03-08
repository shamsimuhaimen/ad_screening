# Brain-region–aware, BBB-conditioned virtual screening for Alzheimer’s disease

## Setup
- Create the env: `conda env create -f environment.yml`
- Activate it: `conda activate ad_screening`
- Install the package in editable mode: `python -m pip install -e .`
- Install git hooks: `pre-commit install`

After modifying deps in `environment.yml`,  sync env with: `conda env update -n ad_screening -f environment.yml --prune`

## Data
Download and extract all datasets: `python src/scripts/download_data.py`

## Prefect Experiment Workflow
Default config: `experiments/prefect_experiments.yaml`

Run matrix (one command):
`python src/scripts/run_prefect_experiments.py`

Smoke test only first N runs:
`python src/scripts/run_prefect_experiments.py --max-runs 3`

Aggregate observed-vs-null tests (empirical p + BH-FDR):
`python src/scripts/aggregate_experiment_results.py --run-root results/prefect_experiments_YYYYMMDD_HHMMSS`
