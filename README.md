# Brain-region–aware, BBB-conditioned virtual screening for Alzheimer’s disease

## Setup
- Create the env: `conda env create -f environment.yml`
- Activate it: `conda activate ad_screening`
- Install the package in editable mode: `python -m pip install -e .`
- Install git hooks: `pre-commit install`

After modifying deps in `environment.yml`,  sync env with: `conda env update -n ad_screening -f environment.yml --prune`

## Data
Download and extract all datasets: `python src/scripts/download_data.py`

## Run Experiments
The main entrypoint is [`src/scripts/run_experiment.py`](/mnt/perma/ad_screening/src/scripts/run_experiment.py).

Run the AD predictor experiment with:
`python src/scripts/run_experiment.py --experiment ad_predictor`

Relationship between the pieces:
- An experiment file such as [`experiments/ad_predictor.yaml`](/mnt/perma/ad_screening/experiments/ad_predictor.yaml) defines the sweep: defaults, seed count, and ablation-specific overrides.
- [`src/scripts/run_experiment.py`](/mnt/perma/ad_screening/src/scripts/run_experiment.py) is the orchestrator. It reads `experiments/{name}.yaml`, expands that config into concrete runs, and calls the matching package implementation.
- An implementation such as [`src/package/ad_predictor.py`](/mnt/perma/ad_screening/src/package/ad_predictor.py) executes one concrete run through `run_ad_predictor(...)`, using the experiment YAML plus the selected ablation, seed, and output directory.
- Outputs for a launcher run are written under `results/{experiment_name}/experiment_runs_<timestamp>/...`, with one subdirectory per ablation and seed.

Basic usage:
`python src/scripts/run_experiment.py --experiment ad_predictor`

Useful examples:
- First local run with automatic data bootstrap: `python src/scripts/run_experiment.py --experiment ad_predictor --bootstrap-data`
- Run with up to 4 concurrent worker threads: `python src/scripts/run_experiment.py --experiment ad_predictor --num-threads 4`
- Write outputs to a custom directory: `python src/scripts/run_experiment.py --experiment ad_predictor --results-dir scratch_results`

CLI flags:
- `--experiment-name`: choose the experiment, matched to `experiments/{name}.yaml` and `results/{name}/...`
- `--bootstrap-data`: download missing required inputs before launching the run
- `--num-threads`: set the maximum number of experiment runs the launcher executes concurrently
- `--results-dir`: write run outputs somewhere other than `results/`
