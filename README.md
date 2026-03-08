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

Run matrix and auto-bootstrap missing downloaded inputs:
`python src/scripts/run_prefect_experiments.py --bootstrap-data`

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
