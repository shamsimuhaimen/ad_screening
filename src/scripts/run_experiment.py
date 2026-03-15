#!/usr/bin/env python3
"""Launch a named experiment script.

One-click usage:
    python src/scripts/run_experiment.py --experiment ad_predictor
"""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from matplotlib.lines import Line2D

SRC_ROOT = Path(__file__).resolve().parents[1]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from package.ad_predictor import run_ad_predictor


DEFAULT_EXPERIMENT_NAME = "ad_predictor"


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--experiment", dest="experiment_name", type=str, default=DEFAULT_EXPERIMENT_NAME)
    p.add_argument("--results-dir", type=Path, default=Path("results"))
    p.add_argument("--bootstrap-data", action="store_true")
    p.add_argument("--num-threads", type=_positive_int, default=6)
    p.add_argument("--debug-plots", action="store_true")
    return p.parse_args()


def resolve_experiment_paths(experiment_name: str) -> tuple[Path, Path]:
    config_path = Path("experiments") / f"{experiment_name}.yaml"
    results_path = Path("results") / experiment_name
    return config_path, results_path


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


def _resolve_seeds(num_seeds: int, default_seed: int) -> list[int]:
    return list(range(default_seed, default_seed + num_seeds))


def _load_experiment_spec(
    experiment_name: str,
    results_dir: Path,
    debug_plots: bool,
) -> tuple[Path, Path, dict[str, object], list[dict[str, object]]]:
    """Expand an experiment YAML into concrete run specs and a timestamped output root."""
    config_path, _ = resolve_experiment_paths(experiment_name)
    root_results_dir = results_dir / experiment_name if results_dir == Path("results") else results_dir

    if not config_path.exists():
        raise FileNotFoundError(f"Experiment config not found: {config_path}")

    cfg = yaml.safe_load(config_path.read_text())
    implementation = str(cfg.get("implementation", ""))
    if implementation != "ad_predictor":
        raise ValueError(f"Unsupported implementation `{implementation}` in {config_path}.")

    defaults = dict(cfg.get("defaults", {}))
    if not defaults:
        raise ValueError("Experiment config must define `defaults`.")
    defaults["debug_plots"] = bool(debug_plots)
    cfg["defaults"] = defaults

    seeds = _resolve_seeds(int(cfg["seeds"]), int(defaults.get("seed", 42)))
    ablations = cfg.get("ablations", [])
    if not ablations:
        raise ValueError("Experiment config must define non-empty `ablations`.")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    root_dir = root_results_dir / f"experiment_runs_{timestamp}"
    runs: list[dict[str, object]] = []
    for ablation_cfg in ablations:
        ablation_name = str(ablation_cfg["name"])
        ablation_overrides = {key: value for key, value in ablation_cfg.items() if key not in {"name", "description"}}
        for seed in seeds:
            run_dir = root_dir / "runs" / ablation_name / f"seed_{int(seed)}"
            run_overrides = {**ablation_overrides, "seed": int(seed)}
            runs.append(
                {
                    "experiment_name": experiment_name,
                    "ablation_name": ablation_name,
                    "seed": int(seed),
                    "output_dir": run_dir,
                    "run_overrides": run_overrides,
                }
            )

    return root_dir, config_path, cfg, runs


def _run_dir(root_dir: Path, ablation_name: str, seed: int) -> Path:
    return root_dir / "runs" / ablation_name / f"seed_{seed}"


def _is_hidden_ablation(ablation_name: str) -> bool:
    return ablation_name.startswith("embedding_hidden_")


def _interp_curve(x: np.ndarray, y: np.ndarray, grid: np.ndarray) -> np.ndarray:
    order = np.argsort(x)
    x_sorted = np.asarray(x[order], dtype=np.float64)
    y_sorted = np.asarray(y[order], dtype=np.float64)
    unique_x, inverse = np.unique(x_sorted, return_inverse=True)
    y_accum = np.zeros_like(unique_x, dtype=np.float64)
    counts = np.zeros_like(unique_x, dtype=np.int64)
    for idx, value in enumerate(inverse):
        y_accum[value] += y_sorted[idx]
        counts[value] += 1
    unique_y = y_accum / counts
    return np.interp(grid, unique_x, unique_y)


def _write_mean_curve_csv(rows: list[dict[str, object]], path: Path) -> None:
    if not rows:
        raise ValueError(f"No rows available for {path.name}")
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _plot_mean_roc_by_ablation(
    summary_rows: list[dict[str, object]],
    summary_by_ablation_rows: list[dict[str, object]],
    root_dir: Path,
    output_name: str = "mean_roc_by_ablation",
    title: str = "Mean ROC by Ablation",
) -> None:
    grid = np.linspace(0.0, 1.0, 201)
    curve_rows: list[dict[str, object]] = []
    plt.figure(figsize=(7, 6))
    for row in summary_by_ablation_rows:
        ablation_name = str(row["ablation_name"])
        curves = []
        for seed_row in summary_rows:
            if str(seed_row["ablation_name"]) != ablation_name:
                continue
            curve_path = _run_dir(root_dir, ablation_name, int(seed_row["seed"])) / "roc_curve.csv"
            curve_df = pd.read_csv(curve_path)
            curves.append(_interp_curve(curve_df["fpr"].to_numpy(), curve_df["tpr"].to_numpy(), grid))
        mean_curve = np.mean(np.vstack(curves), axis=0)
        for x_value, y_value in zip(grid, mean_curve):
            curve_rows.append({"ablation_name": ablation_name, "fpr": float(x_value), "mean_tpr": float(y_value)})
        plt.plot(grid, mean_curve, linewidth=2, label=f"{ablation_name} (AUROC {float(row['mean_test_auroc']):.3f})")
    plt.plot([0.0, 1.0], [0.0, 1.0], linestyle="--", color="#9A9A9A", linewidth=1)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(title)
    plt.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=2, frameon=True)
    plt.tight_layout(rect=(0, 0.08, 1, 1))
    plt.savefig(root_dir / f"{output_name}.png", dpi=200)
    plt.close()
    _write_mean_curve_csv(curve_rows, root_dir / f"{output_name}.csv")


def _plot_mean_pr_by_ablation(
    summary_rows: list[dict[str, object]],
    summary_by_ablation_rows: list[dict[str, object]],
    root_dir: Path,
    output_name: str = "mean_pr_by_ablation",
    title: str = "Mean Precision-Recall by Ablation",
) -> None:
    grid = np.linspace(0.0, 1.0, 201)
    curve_rows: list[dict[str, object]] = []
    plt.figure(figsize=(7, 6))
    for row in summary_by_ablation_rows:
        ablation_name = str(row["ablation_name"])
        curves = []
        for seed_row in summary_rows:
            if str(seed_row["ablation_name"]) != ablation_name:
                continue
            curve_path = _run_dir(root_dir, ablation_name, int(seed_row["seed"])) / "pr_curve.csv"
            curve_df = pd.read_csv(curve_path)
            curves.append(_interp_curve(curve_df["recall"].to_numpy(), curve_df["precision"].to_numpy(), grid))
        mean_curve = np.mean(np.vstack(curves), axis=0)
        for x_value, y_value in zip(grid, mean_curve):
            curve_rows.append(
                {"ablation_name": ablation_name, "recall": float(x_value), "mean_precision": float(y_value)}
            )
        plt.plot(grid, mean_curve, linewidth=2, label=f"{ablation_name} (AUPRC {float(row['mean_test_auprc']):.3f})")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(title)
    plt.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=2, frameon=True)
    plt.tight_layout(rect=(0, 0.08, 1, 1))
    plt.savefig(root_dir / f"{output_name}.png", dpi=200)
    plt.close()
    _write_mean_curve_csv(curve_rows, root_dir / f"{output_name}.csv")


def _plot_mean_loss_by_ablation(
    summary_rows: list[dict[str, object]],
    summary_by_ablation_rows: list[dict[str, object]],
    root_dir: Path,
    output_prefix: str = "mean",
    title_suffix: str = "by Ablation",
) -> None:
    curve_rows: list[dict[str, object]] = []
    combined_fig, combined_ax = plt.subplots(figsize=(10, 7))
    train_fig, train_ax = plt.subplots(figsize=(10, 7))
    test_fig, test_ax = plt.subplots(figsize=(10, 7))
    ablation_handles: list[Line2D] = []
    for row in summary_by_ablation_rows:
        ablation_name = str(row["ablation_name"])
        train_seed_means = []
        test_seed_means = []
        for seed_row in summary_rows:
            if str(seed_row["ablation_name"]) != ablation_name:
                continue
            curve_path = _run_dir(root_dir, ablation_name, int(seed_row["seed"])) / "loss_history.csv"
            loss_df = pd.read_csv(curve_path)
            seed_mean = loss_df.groupby("epoch", as_index=False)[["train_loss", "test_loss"]].mean()
            train_seed_means.append(seed_mean.rename(columns={"train_loss": "value"})[["epoch", "value"]])
            test_seed_means.append(seed_mean.rename(columns={"test_loss": "value"})[["epoch", "value"]])

        train_df = pd.concat(train_seed_means, ignore_index=True).groupby("epoch", as_index=False)["value"].mean()
        test_df = pd.concat(test_seed_means, ignore_index=True).groupby("epoch", as_index=False)["value"].mean()

        train_line = combined_ax.plot(train_df["epoch"], train_df["value"], linewidth=2)[0]
        combined_ax.plot(test_df["epoch"], test_df["value"], linewidth=2, linestyle="--", color=train_line.get_color())
        train_ax.plot(
            train_df["epoch"], train_df["value"], linewidth=2, label=ablation_name, color=train_line.get_color()
        )
        test_ax.plot(test_df["epoch"], test_df["value"], linewidth=2, label=ablation_name, color=train_line.get_color())
        ablation_handles.append(Line2D([0], [0], color=train_line.get_color(), lw=2, label=ablation_name))

        for epoch, value in zip(train_df["epoch"], train_df["value"]):
            curve_rows.append(
                {"ablation_name": ablation_name, "split": "train", "epoch": float(epoch), "mean_loss": float(value)}
            )
        for epoch, value in zip(test_df["epoch"], test_df["value"]):
            curve_rows.append(
                {"ablation_name": ablation_name, "split": "test", "epoch": float(epoch), "mean_loss": float(value)}
            )

    combined_ax.set_xlabel("Epoch")
    combined_ax.set_ylabel("Loss")
    combined_ax.set_title(f"Mean Loss {title_suffix}")
    ablation_legend = combined_ax.legend(
        handles=ablation_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.15),
        ncol=3,
        frameon=True,
        title="Ablation",
    )
    split_handles = [
        Line2D([0], [0], color="#444444", lw=2, linestyle="-", label="train"),
        Line2D([0], [0], color="#444444", lw=2, linestyle="--", label="test"),
    ]
    split_legend = combined_ax.legend(
        handles=split_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.39),
        ncol=2,
        frameon=True,
        title="Split",
    )
    combined_ax.add_artist(ablation_legend)
    combined_ax.add_artist(split_legend)
    combined_fig.tight_layout(rect=(0, 0.27, 1, 1))
    combined_fig.savefig(root_dir / f"{output_prefix}_loss_by_ablation.png", dpi=200)
    plt.close(combined_fig)

    train_ax.set_xlabel("Epoch")
    train_ax.set_ylabel("Train Loss")
    train_ax.set_title(f"Mean Train Loss {title_suffix}")
    train_ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=3, frameon=True)
    train_fig.tight_layout(rect=(0, 0.08, 1, 1))
    train_fig.savefig(root_dir / f"{output_prefix}_train_loss_by_ablation.png", dpi=200)
    plt.close(train_fig)

    test_ax.set_xlabel("Epoch")
    test_ax.set_ylabel("Test Loss")
    test_ax.set_title(f"Mean Test Loss {title_suffix}")
    test_ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=3, frameon=True)
    test_fig.tight_layout(rect=(0, 0.08, 1, 1))
    test_fig.savefig(root_dir / f"{output_prefix}_test_loss_by_ablation.png", dpi=200)
    plt.close(test_fig)

    _write_mean_curve_csv(curve_rows, root_dir / f"{output_prefix}_loss_by_ablation.csv")


def _plot_grouped_aggregates(
    root_dir: Path,
    summary_rows: list[dict[str, object]],
    summary_by_ablation_rows: list[dict[str, object]],
    output_prefix: str,
    title_suffix: str,
) -> None:
    _plot_mean_roc_by_ablation(
        summary_rows,
        summary_by_ablation_rows,
        root_dir,
        output_name=f"{output_prefix}_roc_by_ablation",
        title=f"Mean ROC {title_suffix}",
    )
    _plot_mean_pr_by_ablation(
        summary_rows,
        summary_by_ablation_rows,
        root_dir,
        output_name=f"{output_prefix}_pr_by_ablation",
        title=f"Mean Precision-Recall {title_suffix}",
    )
    _plot_mean_loss_by_ablation(
        summary_rows,
        summary_by_ablation_rows,
        root_dir,
        output_prefix=output_prefix,
        title_suffix=title_suffix,
    )


def _write_debug_plots(rows: list[dict[str, object]], root_dir: Path) -> None:
    for row in rows:
        ablation_name = str(row["ablation_name"])
        seed = int(row["seed"])
        run_dir = _run_dir(root_dir, ablation_name, seed)

        roc_df = pd.read_csv(run_dir / "roc_curve.csv")
        plt.figure(figsize=(6, 6))
        plt.plot(roc_df["fpr"], roc_df["tpr"], color="#F58518", label=f"AUROC = {float(row['test_auroc']):.3f}")
        plt.plot([0.0, 1.0], [0.0, 1.0], linestyle="--", color="#9A9A9A", linewidth=1)
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curve")
        plt.legend(loc="lower right")
        plt.tight_layout()
        plt.savefig(run_dir / "roc_curve.png", dpi=200)
        plt.close()

        pr_df = pd.read_csv(run_dir / "pr_curve.csv")
        plt.figure(figsize=(6, 6))
        plt.plot(pr_df["recall"], pr_df["precision"], color="#54A24B", label=f"AUPRC = {float(row['test_auprc']):.3f}")
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title("Precision-Recall Curve")
        plt.legend(loc="lower left")
        plt.tight_layout()
        plt.savefig(run_dir / "pr_curve.png", dpi=200)
        plt.close()

        loss_df = pd.read_csv(run_dir / "loss_history.csv")
        plt.figure(figsize=(8, 5))
        plt.plot(loss_df["epoch"], loss_df["train_loss"], label="train_loss")
        plt.plot(loss_df["epoch"], loss_df["test_loss"], label="test_loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Training Curves")
        plt.legend()
        plt.tight_layout()
        plt.savefig(run_dir / "loss_curve.png", dpi=200)
        plt.close()


def _write_aggregate_plots(root_dir: Path, rows: list[dict[str, object]], debug_plots: bool) -> None:
    summary_by_ablation_rows = _read_summary_rows(root_dir / "summary_by_ablation.csv")
    _plot_grouped_aggregates(root_dir, rows, summary_by_ablation_rows, "mean", "by Ablation")

    hidden_summary = [row for row in summary_by_ablation_rows if _is_hidden_ablation(str(row["ablation_name"]))]
    hidden_rows = [row for row in rows if _is_hidden_ablation(str(row["ablation_name"]))]
    if hidden_summary:
        _plot_grouped_aggregates(
            root_dir, hidden_rows, hidden_summary, "hidden_only_mean", "for Hidden-Width Ablations"
        )
    if debug_plots:
        _write_debug_plots(rows, root_dir)


def _read_summary_rows(path: Path) -> list[dict[str, object]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


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


def _run_implementation(run_spec: dict[str, object], experiment_config: dict[str, object]) -> dict[str, object]:
    """Execute one expanded run spec against the package implementation and return summary fields."""
    output_dir = Path(str(run_spec["output_dir"]))
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics = run_ad_predictor(
        experiment_config=experiment_config,
        ablation_name=str(run_spec["ablation_name"]),
        seed=int(run_spec["seed"]),
        output_dir=output_dir,
    )
    return {
        "ablation_name": str(run_spec["ablation_name"]),
        "seed": int(run_spec["seed"]),
        **run_spec["run_overrides"],
        **metrics,
    }


def run_experiment(
    root_dir: Path,
    config_path: Path,
    experiment_config: dict[str, object],
    runs: list[dict[str, object]],
    num_threads: int | None,
    debug_plots: bool,
) -> None:
    """Run all expanded jobs for an experiment and write aggregated summary artifacts."""
    root_dir.mkdir(parents=True, exist_ok=True)
    (root_dir / "config.snapshot.yaml").write_text(config_path.read_text())
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = [executor.submit(_run_implementation, run_spec, experiment_config) for run_spec in runs]
        rows = [future.result() for future in futures]
    _write_summary_csv(rows, root_dir / "summary.csv")
    _write_summary_by_ablation(rows, root_dir / "summary_by_ablation.csv")
    _write_aggregate_plots(root_dir, rows, debug_plots)
    print(f"Wrote: {root_dir / 'summary.csv'}")
    print(f"Wrote: {root_dir / 'summary_by_ablation.csv'}")
    print(f"Wrote: {root_dir / 'mean_roc_by_ablation.png'}")
    print(f"Wrote: {root_dir / 'mean_pr_by_ablation.png'}")
    print(f"Wrote: {root_dir / 'mean_loss_by_ablation.png'}")
    print(f"Wrote: {root_dir / 'mean_train_loss_by_ablation.png'}")
    print(f"Wrote: {root_dir / 'mean_test_loss_by_ablation.png'}")
    print(f"Wrote: {root_dir / 'hidden_only_mean_roc_by_ablation.png'}")
    print(f"Wrote: {root_dir / 'hidden_only_mean_pr_by_ablation.png'}")
    print(f"Wrote: {root_dir / 'hidden_only_mean_loss_by_ablation.png'}")
    print(f"Wrote: {root_dir / 'hidden_only_mean_train_loss_by_ablation.png'}")
    print(f"Wrote: {root_dir / 'hidden_only_mean_test_loss_by_ablation.png'}")


def main() -> None:
    args = parse_args()
    ensure_bootstrap_data(bootstrap_data=args.bootstrap_data)
    root_dir, config_path, experiment_config, runs = _load_experiment_spec(
        args.experiment_name,
        args.results_dir,
        args.debug_plots,
    )
    run_experiment(root_dir, config_path, experiment_config, runs, args.num_threads, args.debug_plots)


if __name__ == "__main__":
    main()
