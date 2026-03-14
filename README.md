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
- An experiment file such as [`experiments/ad_predictor.yaml`](/mnt/perma/ad_screening/experiments/ad_predictor.yaml) defines a sweep. It declares the implementation name, shared defaults, the number of seeds, and the ablation-specific overrides.
- [`src/scripts/run_experiment.py`](/mnt/perma/ad_screening/src/scripts/run_experiment.py) is the orchestrator. It loads `experiments/{name}.yaml`, expands the ablations and seeds into concrete runs, and dispatches each run to the requested implementation.
- The implementation for `ad_predictor` lives in [`src/package/ad_predictor.py`](/mnt/perma/ad_screening/src/package/ad_predictor.py). It exposes `run_ad_predictor(...)`, which receives the experiment YAML, the selected ablation name, the seed, and the output directory for a single run.
- Outputs are written under `results/{experiment_name}/experiment_runs_<timestamp>/...`, with one subdirectory per ablation and seed plus aggregate summaries at the run root.

Basic usage:
`python src/scripts/run_experiment.py --experiment ad_predictor`

Useful examples:
- First local run with automatic data bootstrap: `python src/scripts/run_experiment.py --experiment ad_predictor --bootstrap-data`
- Run with up to 4 concurrent worker threads: `python src/scripts/run_experiment.py --experiment ad_predictor --num-threads 4`
- Write outputs to a custom directory: `python src/scripts/run_experiment.py --experiment ad_predictor --results-dir scratch_results`

CLI flags:
- `--experiment`: choose the experiment, matched to `experiments/{name}.yaml` and `results/{name}/...`
- `--bootstrap-data`: download missing required inputs before launching the run
- `--num-threads`: set the maximum number of experiment runs the launcher executes concurrently
- `--results-dir`: write run outputs somewhere other than `results/`
