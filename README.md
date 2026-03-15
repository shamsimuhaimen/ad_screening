# Brain-region–aware, BBB-conditioned virtual screening for Alzheimer’s disease

## Sample Results Dashboard
The experiment runner writes per-run ROC, precision-recall, and loss curves for each ablation. Below is a sampled dashboard from `results/ad_predictor/experiment_runs_20260314_235446` generated on March 14, 2026 (UTC), using `seed_42` for each ablation.

Experiment scale from [`results/ad_predictor/experiment_runs_20260314_235446/summary.csv`](results/ad_predictor/experiment_runs_20260314_235446/summary.csv): `762` total labeled protein samples, with `254` positives and `508` negatives. The reported metrics come from `5`-fold cross-validation, with an average split size of `609.6` train and `152.4` test samples per fold.

Summary from [`results/ad_predictor/experiment_runs_20260314_235446/summary_by_ablation.csv`](results/ad_predictor/experiment_runs_20260314_235446/summary_by_ablation.csv):

| Ablation | Mean test AUROC | Mean test AUPRC | Samples |
| --- | ---: | ---: | ---: |
| `weighted_bce` | 0.545 | 0.391 | 762 |
| `random_embedding` | 0.518 | 0.354 | 762 |
| `ad_embedding` | 0.515 | 0.350 | 762 |
| `label_shuffle` | 0.483 | 0.338 | 762 |

| `weighted_bce` | `random_embedding` | `ad_embedding` | `label_shuffle` |
| --- | --- | --- | --- |
| ![weighted_bce ROC](docs/readme_assets/ad_predictor_20260314/weighted_bce_roc_seed42.png) | ![random_embedding ROC](docs/readme_assets/ad_predictor_20260314/random_embedding_roc_seed42.png) | ![ad_embedding ROC](docs/readme_assets/ad_predictor_20260314/ad_embedding_roc_seed42.png) | ![label_shuffle ROC](docs/readme_assets/ad_predictor_20260314/label_shuffle_roc_seed42.png) |
| ![weighted_bce PR](docs/readme_assets/ad_predictor_20260314/weighted_bce_pr_seed42.png) | ![random_embedding PR](docs/readme_assets/ad_predictor_20260314/random_embedding_pr_seed42.png) | ![ad_embedding PR](docs/readme_assets/ad_predictor_20260314/ad_embedding_pr_seed42.png) | ![label_shuffle PR](docs/readme_assets/ad_predictor_20260314/label_shuffle_pr_seed42.png) |
| ![weighted_bce loss](docs/readme_assets/ad_predictor_20260314/weighted_bce_loss_seed42.png) | ![random_embedding loss](docs/readme_assets/ad_predictor_20260314/random_embedding_loss_seed42.png) | ![ad_embedding loss](docs/readme_assets/ad_predictor_20260314/ad_embedding_loss_seed42.png) | ![label_shuffle loss](docs/readme_assets/ad_predictor_20260314/label_shuffle_loss_seed42.png) |

This gives a quick visual comparison between the weighted-loss variant, the learned AD embedding, and the two controls. To regenerate a similar dashboard, rerun the experiment and point the image links at the newest timestamped directory under `results/ad_predictor/`.

## Setup
- Create the env: `conda env create -f environment.yml`
- Activate it: `conda activate ad_screening`
- Install the package in editable mode: `python -m pip install -e .`
- Install git hooks: `pre-commit install`

After modifying deps in `environment.yml`,  sync env with: `conda env update -n ad_screening -f environment.yml --prune`

## Data
Download and extract all datasets: `python src/scripts/download_data.py`

## Run Experiments
The main entrypoint is [`src/scripts/run_experiment.py`](src/scripts/run_experiment.py).

Run the AD predictor experiment with:
`python src/scripts/run_experiment.py --experiment ad_predictor`

Relationship between the pieces:
- An experiment file such as [`experiments/ad_predictor.yaml`](experiments/ad_predictor.yaml) defines a sweep. It declares the implementation name, shared defaults, the number of seeds, and the ablation-specific overrides.
- [`src/scripts/run_experiment.py`](src/scripts/run_experiment.py) is the orchestrator. It loads `experiments/{name}.yaml`, expands the ablations and seeds into concrete runs, and dispatches each run to the requested implementation.
- The implementation for `ad_predictor` lives in [`src/package/ad_predictor.py`](src/package/ad_predictor.py). It exposes `run_ad_predictor(...)`, which receives the experiment YAML, the selected ablation name, the seed, and the output directory for a single run.
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
