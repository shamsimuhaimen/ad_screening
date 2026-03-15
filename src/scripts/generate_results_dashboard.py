#!/usr/bin/env python3
"""Render a results dashboard image for an experiment run."""

from __future__ import annotations

import argparse
import csv
import os
import re
import tempfile
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "matplotlib"))

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from PIL import Image, ImageStat


DEFAULT_RUN_DIR = Path("results/ad_predictor/experiment_runs_20260314_235446")
DEFAULT_OUTPUT_PATH = Path("results/ad_predictor_dashboard.png")
DEFAULT_SEED = 42
CURVE_FILES = (
    ("roc_curve.png", "ROC"),
    ("pr_curve.png", "PR"),
    ("loss_curve.png", "Loss"),
)
NONBLANK_STDDEV_THRESHOLD = 1.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--output-path", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--title", type=str, default="AD Predictor Results Dashboard")
    return parser.parse_args()


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"Required CSV not found: {path}")
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def _sample_row(rows: list[dict[str, str]], seed: int) -> dict[str, str]:
    if not rows:
        raise ValueError("summary.csv is empty")
    for row in rows:
        if int(row["seed"]) == seed:
            return row
    available_seeds = sorted({int(row["seed"]) for row in rows})
    raise ValueError(f"Seed {seed} not found in summary.csv. Available seeds: {available_seeds}")


def _format_run_date(run_dir: Path) -> str:
    match = re.search(r"experiment_runs_(\d{8})_(\d{6})", run_dir.name)
    if not match:
        return run_dir.name
    dt = datetime.strptime("".join(match.groups()), "%Y%m%d%H%M%S").replace(tzinfo=timezone.utc)
    return dt.strftime("%Y-%m-%d %H:%M UTC")


def _metric_value(row: dict[str, str], key: str) -> float:
    return float(row[key])


def _ablation_run_row(rows: list[dict[str, str]], ablation_name: str, seed: int) -> dict[str, str]:
    for row in rows:
        if row["ablation_name"] == ablation_name and int(row["seed"]) == seed:
            return row
    raise ValueError(f"No row found for ablation={ablation_name!r}, seed={seed}")


def _build_scale_text(sample_row: dict[str, str]) -> str:
    n_samples = int(round(float(sample_row["n_samples"])))
    n_pos = int(round(float(sample_row["n_pos"])))
    n_neg = int(round(float(sample_row["n_neg"])))
    num_folds = int(round(float(sample_row["num_folds"])))
    mean_train_size = float(sample_row["mean_train_size"])
    mean_test_size = float(sample_row["mean_test_size"])
    return (
        f"Scale: {n_samples} samples ({n_pos} positive, {n_neg} negative)   "
        f"CV: {num_folds} folds   Avg split: {mean_train_size:.1f} train / {mean_test_size:.1f} test"
    )


def _render_header(ax: plt.Axes, title: str, subtitle: str, scale_text: str) -> None:
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.text(0.01, 0.78, title, fontsize=24, fontweight="bold", ha="left", va="center")
    ax.text(0.01, 0.42, subtitle, fontsize=12, color="#444444", ha="left", va="center")
    ax.text(0.01, 0.10, scale_text, fontsize=12, color="#222222", ha="left", va="center")


def _render_metrics_row(
    ax: plt.Axes,
    summary_by_ablation_rows: list[dict[str, str]],
    sample_row: dict[str, str],
) -> None:
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    cards = [
        ("Ablations", str(len(summary_by_ablation_rows))),
        ("Samples", str(int(round(float(sample_row["n_samples"]))))),
        ("Best AUROC", f"{max(_metric_value(row, 'mean_test_auroc') for row in summary_by_ablation_rows):.3f}"),
        ("Best AUPRC", f"{max(_metric_value(row, 'mean_test_auprc') for row in summary_by_ablation_rows):.3f}"),
    ]

    card_width = 0.22
    card_gap = 0.03
    x_positions = [0.01 + idx * (card_width + card_gap) for idx in range(len(cards))]
    for x_pos, (label, value) in zip(x_positions, cards):
        ax.add_patch(
            plt.Rectangle(
                (x_pos, 0.10),
                card_width,
                0.80,
                facecolor="#F5F7FA",
                edgecolor="#D9E0E8",
                linewidth=1.2,
            )
        )
        ax.text(x_pos + 0.03, 0.62, value, fontsize=18, fontweight="bold", ha="left", va="center")
        ax.text(x_pos + 0.03, 0.32, label, fontsize=10, color="#5B6573", ha="left", va="center")


def _render_column_header(
    ax: plt.Axes,
    ablation_name: str,
    summary_row: dict[str, str],
    run_row: dict[str, str],
) -> None:
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.add_patch(
        plt.Rectangle(
            (0.0, 0.02),
            1.0,
            0.96,
            facecolor="#EEF3F7",
            edgecolor="#D9E0E8",
            linewidth=1.0,
        )
    )
    ax.text(0.04, 0.76, ablation_name, fontsize=13, fontweight="bold", ha="left", va="center")
    ax.text(
        0.04,
        0.45,
        f"Mean AUROC {float(summary_row['mean_test_auroc']):.3f}   Mean AUPRC {float(summary_row['mean_test_auprc']):.3f}",
        fontsize=9,
        color="#364152",
        ha="left",
        va="center",
    )
    ax.text(
        0.04,
        0.17,
        f"Seed {int(run_row['seed'])}   Test acc {float(run_row['test_accuracy']):.3f}",
        fontsize=9,
        color="#364152",
        ha="left",
        va="center",
    )


def _render_curve(ax: plt.Axes, image_path: Path, row_label: str) -> None:
    image = plt.imread(image_path)
    ax.imshow(image)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_title(row_label, loc="left", fontsize=10, pad=6)


def _is_nonblank_image(image_path: Path) -> bool:
    image = Image.open(image_path).convert("L")
    stddev = ImageStat.Stat(image).stddev[0]
    return stddev > NONBLANK_STDDEV_THRESHOLD


def _candidate_run_dirs(run_dir: Path) -> list[Path]:
    siblings = sorted(path for path in run_dir.parent.glob("experiment_runs_*") if path.is_dir())
    if run_dir not in siblings:
        return [run_dir]
    run_idx = siblings.index(run_dir)
    candidates = [run_dir]
    for offset in range(1, len(siblings)):
        prev_idx = run_idx - offset
        next_idx = run_idx + offset
        if prev_idx >= 0:
            candidates.append(siblings[prev_idx])
        if next_idx < len(siblings):
            candidates.append(siblings[next_idx])
    return candidates


def _candidate_seed_dirs(run_dir: Path, ablation_name: str, preferred_seed: int) -> list[Path]:
    ablation_dir = run_dir / "runs" / ablation_name
    if not ablation_dir.exists():
        return []
    seed_dirs = sorted(path for path in ablation_dir.glob("seed_*") if path.is_dir())
    preferred_name = f"seed_{preferred_seed}"
    preferred = [path for path in seed_dirs if path.name == preferred_name]
    others = [path for path in seed_dirs if path.name != preferred_name]
    return preferred + others


def _resolve_curve_path(run_dir: Path, ablation_name: str, preferred_seed: int, curve_filename: str) -> Path:
    for candidate_run_dir in _candidate_run_dirs(run_dir):
        for seed_dir in _candidate_seed_dirs(candidate_run_dir, ablation_name, preferred_seed):
            curve_path = seed_dir / curve_filename
            if not curve_path.exists():
                continue
            if _is_nonblank_image(curve_path):
                return curve_path
    raise FileNotFoundError(
        f"Could not find a nonblank `{curve_filename}` for ablation `{ablation_name}` near run {run_dir}"
    )


def render_dashboard(run_dir: Path, seed: int, output_path: Path, title: str) -> None:
    summary_rows = _read_csv_rows(run_dir / "summary.csv")
    summary_by_ablation_rows = _read_csv_rows(run_dir / "summary_by_ablation.csv")
    sample_row = _sample_row(summary_rows, seed)
    ordered_summary_rows = sorted(summary_by_ablation_rows, key=lambda row: float(row["mean_test_auprc"]), reverse=True)
    ablation_names = [row["ablation_name"] for row in ordered_summary_rows]

    num_cols = len(ablation_names)
    if num_cols == 0:
        raise ValueError("summary_by_ablation.csv is empty")

    fig_width = max(16.0, num_cols * 4.2)
    fig_height = 16.0
    fig = plt.figure(figsize=(fig_width, fig_height), constrained_layout=True, facecolor="white")
    grid = GridSpec(
        nrows=6,
        ncols=num_cols,
        figure=fig,
        height_ratios=[0.9, 0.9, 0.9, 3.0, 3.0, 3.0],
    )

    subtitle = f"Run: {run_dir.as_posix()}   Generated: {_format_run_date(run_dir)}   Seed: {seed}"
    _render_header(fig.add_subplot(grid[0, :]), title, subtitle, _build_scale_text(sample_row))
    _render_metrics_row(fig.add_subplot(grid[1, :]), ordered_summary_rows, sample_row)

    for col_idx, summary_row in enumerate(ordered_summary_rows):
        ablation_name = summary_row["ablation_name"]
        run_row = _ablation_run_row(summary_rows, ablation_name, seed)
        _render_column_header(fig.add_subplot(grid[2, col_idx]), ablation_name, summary_row, run_row)

        for row_offset, (curve_filename, row_label) in enumerate(CURVE_FILES, start=3):
            curve_path = _resolve_curve_path(run_dir, ablation_name, seed, curve_filename)
            _render_curve(fig.add_subplot(grid[row_offset, col_idx]), curve_path, row_label)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote: {output_path}")


def main() -> None:
    args = parse_args()
    render_dashboard(
        run_dir=args.run_dir,
        seed=args.seed,
        output_path=args.output_path,
        title=args.title,
    )


if __name__ == "__main__":
    main()
