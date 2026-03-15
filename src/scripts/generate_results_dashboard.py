#!/usr/bin/env python3
"""Render aggregate AD predictor plots used by the README and milestone report."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "matplotlib"))

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D


DEFAULT_RUN_DIR = Path("results/ad_predictor_full_report")
DEFAULT_REAL_EMBEDDING_LOSS_NAME = "real_embedding_loss_by_ablation.png"
PREFERRED_ABLATION_ORDER = [
    "ad_embedding",
    "embedding_hidden_4",
    "embedding_hidden_8",
    "embedding_hidden_16",
    "embedding_hidden_32",
    "embedding_hidden_64",
]


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for aggregate plot generation."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--real-embedding-loss-output-path", type=Path, default=None)
    return parser.parse_args()


def _ordered_ablations(ablation_names: list[str]) -> list[str]:
    """Sort ablations with the real-embedding baseline first, then hidden-width variants."""
    preferred = [name for name in PREFERRED_ABLATION_ORDER if name in ablation_names]
    remaining = sorted(name for name in ablation_names if name not in preferred)
    return preferred + remaining


def _mean(values: list[float]) -> float:
    return float(sum(values) / len(values))


def _mean_metric_by_ablation(run_dir: Path, ablation_names: list[str], metric_key: str) -> dict[str, float]:
    """Average one metric across all available seed metrics files for each ablation."""
    metrics_by_ablation: dict[str, list[float]] = {}
    runs_dir = run_dir / "runs"
    for ablation_name in ablation_names:
        ablation_dir = runs_dir / ablation_name
        if not ablation_dir.exists():
            continue
        metric_values: list[float] = []
        for seed_dir in sorted(path for path in ablation_dir.glob("seed_*") if path.is_dir()):
            metrics_path = seed_dir / "metrics.json"
            if not metrics_path.exists():
                continue
            metric_values.append(float(json.loads(metrics_path.read_text())[metric_key]))
        if metric_values:
            metrics_by_ablation[ablation_name] = _mean(metric_values)
    return metrics_by_ablation


def _copy_png_and_csv_aliases(run_dir: Path, output_name: str, alias_name: str) -> None:
    """Copy a rendered PNG and its CSV data to an alias stem."""
    output_png = run_dir / output_name
    output_csv = run_dir / output_name.replace(".png", ".csv")
    alias_png = run_dir / alias_name
    alias_csv = run_dir / alias_name.replace(".png", ".csv")
    alias_png.write_bytes(output_png.read_bytes())
    alias_csv.write_text(output_csv.read_text())


def _render_hidden_curve_plot(
    run_dir: Path,
    hidden_curve_csv_name: str,
    base_curve_csv_name: str,
    output_name: str,
    alias_name: str,
    x_key: str,
    y_key: str,
    metric_key: str,
    x_label: str,
    y_label: str,
    title: str,
    baseline_line: tuple[list[float], list[float]] | None = None,
) -> None:
    """Render a hidden-width aggregate curve plot augmented with the no-hidden baseline."""
    hidden_curve_df = pd.read_csv(run_dir / hidden_curve_csv_name)
    base_curve_df = pd.read_csv(run_dir / base_curve_csv_name)
    baseline_df = base_curve_df[base_curve_df["ablation_name"] == "ad_embedding"]
    combined_df = pd.concat([baseline_df, hidden_curve_df], ignore_index=True)
    combined_df = combined_df[combined_df["ablation_name"].isin(PREFERRED_ABLATION_ORDER)].copy()

    metric_by_ablation = _mean_metric_by_ablation(run_dir, _ordered_ablations(PREFERRED_ABLATION_ORDER), metric_key)

    plt.figure(figsize=(7, 6))
    for ablation_name in _ordered_ablations(sorted(combined_df["ablation_name"].unique())):
        ablation_df = combined_df[combined_df["ablation_name"] == ablation_name].sort_values(x_key)
        metric_value = metric_by_ablation.get(ablation_name)
        label = (
            ablation_name
            if metric_value is None
            else f"{ablation_name} ({metric_key.replace('test_', '').upper()} {metric_value:.3f})"
        )
        plt.plot(ablation_df[x_key], ablation_df[y_key], linewidth=2, label=label)
    if baseline_line is not None:
        plt.plot(baseline_line[0], baseline_line[1], linestyle="--", color="#9A9A9A", linewidth=1)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=2, frameon=True)
    plt.tight_layout(rect=(0, 0.08, 1, 1))
    plt.savefig(run_dir / output_name, dpi=200)
    plt.close()

    combined_df.to_csv(run_dir / output_name.replace(".png", ".csv"), index=False)
    _copy_png_and_csv_aliases(run_dir, output_name, alias_name)
    print(f"Wrote: {run_dir / output_name}")
    print(f"Wrote: {run_dir / alias_name}")


def render_hidden_layer_baseline_plots(run_dir: Path) -> None:
    """Write hidden-width mean ROC and PR plots that include `ad_embedding`."""
    _render_hidden_curve_plot(
        run_dir=run_dir,
        hidden_curve_csv_name="hidden_only_mean_roc_by_ablation.csv",
        base_curve_csv_name="mean_roc_by_ablation.csv",
        output_name="hidden_only_mean_roc_by_ablation.png",
        alias_name="hidden_only_mean_auroc_by_ablation.png",
        x_key="fpr",
        y_key="mean_tpr",
        metric_key="test_auroc",
        x_label="False Positive Rate",
        y_label="True Positive Rate",
        title="Mean ROC for Hidden-Width Ablations",
        baseline_line=([0.0, 1.0], [0.0, 1.0]),
    )
    _render_hidden_curve_plot(
        run_dir=run_dir,
        hidden_curve_csv_name="hidden_only_mean_pr_by_ablation.csv",
        base_curve_csv_name="mean_pr_by_ablation.csv",
        output_name="hidden_only_mean_pr_by_ablation.png",
        alias_name="hidden_only_mean_auprc_by_ablation.png",
        x_key="recall",
        y_key="mean_precision",
        metric_key="test_auprc",
        x_label="Recall",
        y_label="Precision",
        title="Mean Precision-Recall for Hidden-Width Ablations",
    )


def render_real_embedding_loss_plot(run_dir: Path, output_path: Path) -> None:
    """Render a train/test mean-loss plot for `ad_embedding` plus hidden-width runs."""
    base_loss_path = run_dir / "mean_loss_by_ablation.csv"
    hidden_loss_path = run_dir / "hidden_only_mean_loss_by_ablation.csv"
    if not base_loss_path.exists():
        raise FileNotFoundError(f"Required loss CSV not found: {base_loss_path}")
    if not hidden_loss_path.exists():
        raise FileNotFoundError(f"Required hidden-only loss CSV not found: {hidden_loss_path}")

    combined_df = pd.concat([pd.read_csv(base_loss_path), pd.read_csv(hidden_loss_path)], ignore_index=True)
    combined_df = (
        combined_df.groupby(["ablation_name", "split", "epoch"], as_index=False)["mean_loss"]
        .mean()
        .sort_values(["ablation_name", "split", "epoch"])
    )
    combined_df = combined_df[combined_df["ablation_name"].isin(PREFERRED_ABLATION_ORDER)].copy()
    if combined_df.empty:
        raise ValueError("No eligible ablations found for the real-embedding loss plot.")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    combined_df.to_csv(output_path.with_suffix(".csv"), index=False)

    fig, ax = plt.subplots(figsize=(14, 9))
    ablation_handles: list[Line2D] = []
    for ablation_name in _ordered_ablations(sorted(combined_df["ablation_name"].unique())):
        ablation_df = combined_df[combined_df["ablation_name"] == ablation_name]
        train_df = ablation_df[ablation_df["split"] == "train"]
        test_df = ablation_df[ablation_df["split"] == "test"]

        train_line = ax.plot(train_df["epoch"], train_df["mean_loss"], linewidth=2)[0]
        ax.plot(test_df["epoch"], test_df["mean_loss"], linewidth=2, linestyle="--", color=train_line.get_color())
        ablation_handles.append(Line2D([0], [0], color=train_line.get_color(), lw=2, label=ablation_name))

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Real-Embedding Loss by Ablation")
    ablation_legend = ax.legend(
        handles=ablation_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.14),
        ncol=4,
        frameon=True,
        title="Ablation",
    )
    split_handles = [
        Line2D([0], [0], color="#444444", lw=2, linestyle="-", label="train"),
        Line2D([0], [0], color="#444444", lw=2, linestyle="--", label="test"),
    ]
    split_legend = ax.legend(
        handles=split_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.35),
        ncol=2,
        frameon=True,
        title="Split",
    )
    ax.add_artist(ablation_legend)
    ax.add_artist(split_legend)
    fig.tight_layout(rect=(0, 0.24, 1, 1))
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    print(f"Wrote: {output_path}")


def main() -> None:
    """Generate the aggregate plots still consumed by the README and milestone report."""
    args = parse_args()
    render_hidden_layer_baseline_plots(args.run_dir)
    real_embedding_loss_output_path = args.real_embedding_loss_output_path
    if real_embedding_loss_output_path is None:
        real_embedding_loss_output_path = args.run_dir / DEFAULT_REAL_EMBEDDING_LOSS_NAME
    render_real_embedding_loss_plot(args.run_dir, real_embedding_loss_output_path)


if __name__ == "__main__":
    main()
