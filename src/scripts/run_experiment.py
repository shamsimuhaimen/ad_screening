#!/usr/bin/env python3
"""Launch a named experiment script.

One-click usage:
    python src/scripts/run_experiment.py --experiment-name ad_predictor
"""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import yaml


DEFAULT_EXPERIMENT_NAME = "ad_predictor"


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--experiment-name", "--experiment", dest="experiment_name", type=str, default=DEFAULT_EXPERIMENT_NAME
    )
    p.add_argument("--results-dir", type=Path, default=Path("results"))
    p.add_argument("--bootstrap-data", action="store_true")
    p.add_argument("--num-threads", type=_positive_int, default=6)
    p.add_argument(
        "script_args",
        nargs=argparse.REMAINDER,
        help="Additional arguments passed through to the experiment script after `--`.",
    )
    return p.parse_args()


def resolve_experiment_paths(experiment_name: str) -> tuple[Path, Path, Path]:
    config_path = Path("experiments") / f"{experiment_name}.yaml"
    script_path = Path("src/scripts") / f"{experiment_name}.py"
    results_path = Path("results") / experiment_name
    return config_path, script_path, results_path


def _required_download_inputs() -> list[Path]:
    return [
        Path("data/raw/bulk_rna_seq_human_brain/Genes.csv"),
        Path("data/raw/bulk_rna_seq_human_brain/SampleAnnot.csv"),
        Path("data/raw/bulk_rna_seq_human_brain/RNAseqTPM.csv"),
        Path("data/download/dtwg_af_embeddings.npy"),
        Path("data/download/dtwg_af_names_.npy"),
        Path("data/download/hgnc_complete_set.txt"),
    ]


def ensure_bootstrap_data(bootstrap_data: bool) -> None:
    missing = [p for p in _required_download_inputs() if not p.exists()]
    if not missing:
        return
    if not bootstrap_data:
        names = ", ".join(str(p) for p in missing[:4])
        if len(missing) > 4:
            names += ", ..."
        raise FileNotFoundError(
            "Missing downloaded inputs required by the experiment workflow: " f"{names}. Re-run with --bootstrap-data."
        )
    proc = subprocess.run(
        [sys.executable, "src/scripts/download_data.py"],
        check=False,
        text=True,
    )
    if proc.returncode != 0:
        raise RuntimeError("Bootstrap download failed.")


def _passthrough_args(script_args: list[str]) -> list[str]:
    passthrough = script_args
    if passthrough and passthrough[0] == "--":
        passthrough = passthrough[1:]
    return passthrough


def _cli_flag(name: str) -> str:
    return f"--{name.replace('_', '-')}"


def _stringify_arg(value: object) -> str:
    if isinstance(value, Path):
        return str(value)
    return str(value)


def _resolve_seeds(num_seeds: int, default_seed: int) -> list[int]:
    return list(range(default_seed, default_seed + num_seeds))


def _build_single_run_command(
    script_path: Path,
    output_dir: Path,
    base_args: dict[str, object],
    run_overrides: dict[str, object],
    passthrough: list[str],
) -> list[str]:
    cmd = [sys.executable, str(script_path), "--output-dir", str(output_dir)]
    merged_args = dict(base_args)
    merged_args.update(run_overrides)
    for key, value in merged_args.items():
        if value is None:
            continue
        if isinstance(value, bool):
            cmd.append(_cli_flag(key) if value else f"--no-{key.replace('_', '-')}")
            continue
        cmd.extend([_cli_flag(key), _stringify_arg(value)])
    cmd.extend(passthrough)
    return cmd


def _load_experiment_spec(
    experiment_name: str,
    results_dir: Path,
    script_args: list[str],
) -> tuple[Path, Path, list[dict[str, object]]]:
    config_path, script_path, _ = resolve_experiment_paths(experiment_name)
    root_results_dir = results_dir / experiment_name if results_dir == Path("results") else results_dir

    if not config_path.exists():
        raise FileNotFoundError(f"Experiment config not found: {config_path}")
    if not script_path.exists():
        raise FileNotFoundError(f"Experiment script not found: {script_path}")

    cfg = yaml.safe_load(config_path.read_text())
    expected_script = Path(str(cfg.get("script", "")))
    if expected_script != script_path:
        raise ValueError(f"Config script mismatch: expected `{script_path}`, found `{expected_script}`.")

    defaults = dict(cfg.get("defaults", {}))
    if not defaults:
        raise ValueError("Experiment config must define `defaults`.")

    seeds = _resolve_seeds(int(cfg["seeds"]), int(defaults.get("seed", 42)))
    ablations = cfg.get("ablations", [])
    if not ablations:
        raise ValueError("Experiment config must define non-empty `ablations`.")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    root_dir = root_results_dir / f"experiment_runs_{timestamp}"
    runs: list[dict[str, object]] = []
    passthrough = _passthrough_args(script_args)
    for ablation_cfg in ablations:
        ablation_name = str(ablation_cfg["name"])
        ablation_overrides = {key: value for key, value in ablation_cfg.items() if key not in {"name", "description"}}
        for seed in seeds:
            run_dir = root_dir / "runs" / ablation_name / f"seed_{int(seed)}"
            run_overrides = {**ablation_overrides, "seed": int(seed)}
            cmd = _build_single_run_command(script_path, run_dir, defaults, run_overrides, passthrough)
            runs.append(
                {
                    "experiment_name": experiment_name,
                    "ablation_name": ablation_name,
                    "seed": int(seed),
                    "cmd": cmd,
                    "output_dir": run_dir,
                    "run_overrides": run_overrides,
                }
            )

    return root_dir, config_path, runs


def _write_summary_csv(rows: list[dict[str, object]], path: Path) -> None:
    if not rows:
        raise ValueError("No run rows available for summary output.")
    fieldnames: list[str] = list(rows[0].keys())
    for row in rows[1:]:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _mean(values: list[float]) -> float:
    return float(sum(values) / len(values))


def _write_summary_by_ablation(rows: list[dict[str, object]], path: Path) -> None:
    grouped: dict[str, list[dict[str, object]]] = {}
    for row in rows:
        grouped.setdefault(str(row["ablation_name"]), []).append(row)

    summary_rows: list[dict[str, object]] = []
    for ablation_name, group_rows in grouped.items():
        summary_rows.append(
            {
                "ablation_name": ablation_name,
                "mean_test_accuracy": _mean([float(row["test_accuracy"]) for row in group_rows]),
                "mean_test_auroc": _mean([float(row["test_auroc"]) for row in group_rows]),
                "mean_test_auprc": _mean([float(row["test_auprc"]) for row in group_rows]),
                "mean_train_accuracy": _mean([float(row["train_accuracy_mean"]) for row in group_rows]),
            }
        )
    summary_rows.sort(key=lambda row: float(row["mean_test_auprc"]), reverse=True)
    _write_summary_csv(summary_rows, path)


def _run_subprocess(run_spec: dict[str, object]) -> dict[str, object]:
    command = list(run_spec["cmd"])
    output_dir = Path(str(run_spec["output_dir"]))
    output_dir.mkdir(parents=True, exist_ok=True)
    proc = subprocess.run(command, check=False)
    if proc.returncode != 0:
        raise RuntimeError(f"Experiment subprocess failed with exit code {proc.returncode}.")
    metrics = json.loads((output_dir / "metrics.json").read_text())
    return {
        "ablation_name": str(run_spec["ablation_name"]),
        "seed": int(run_spec["seed"]),
        **run_spec["run_overrides"],
        **metrics,
    }


def run_experiment(
    root_dir: Path,
    config_path: Path,
    runs: list[dict[str, object]],
    num_threads: int | None,
) -> None:
    root_dir.mkdir(parents=True, exist_ok=True)
    (root_dir / "config.snapshot.yaml").write_text(config_path.read_text())
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
        rows = list(executor.map(_run_subprocess, runs))
    _write_summary_csv(rows, root_dir / "summary.csv")
    _write_summary_by_ablation(rows, root_dir / "summary_by_ablation.csv")
    print(f"Wrote: {root_dir / 'summary.csv'}")
    print(f"Wrote: {root_dir / 'summary_by_ablation.csv'}")


def main() -> None:
    args = parse_args()
    ensure_bootstrap_data(bootstrap_data=args.bootstrap_data)
    root_dir, config_path, runs = _load_experiment_spec(args.experiment_name, args.results_dir, args.script_args)
    run_experiment(root_dir, config_path, runs, args.num_threads)


if __name__ == "__main__":
    main()
