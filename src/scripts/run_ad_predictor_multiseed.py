#!/usr/bin/env python3
"""Run multi-seed ablations for the AD predictor and summarize metrics.

This script runs `train_ad_predictor.py` across seeds for:
- embedding
- random_embedding
- label_shuffle

For each seed, it uses one shared split file across ablations, so comparisons
are apples-to-apples. It then writes per-run and aggregated summary outputs.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from datetime import datetime
from pathlib import Path

import pandas as pd


ABLATIONS = ["embedding", "random_embedding", "label_shuffle"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seeds", type=str, default="42,43,44,45,46")
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--hidden-dim", type=int, default=64)
    p.add_argument("--lr", type=float, default=0.001)
    p.add_argument("--l2", type=float, default=1e-4)
    p.add_argument("--test-size", type=float, default=0.2)
    p.add_argument("--output-dir", type=Path, default=Path("results"))
    return p.parse_args()


def newest_run_dir_for_ablation(ablation: str) -> Path:
    matches = sorted(
        Path("results").glob(f"ad_predictor_*_{ablation}"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not matches:
        raise FileNotFoundError(f"No run directory found for ablation: {ablation}")
    return matches[0]


def run_one(seed: int, ablation: str, args: argparse.Namespace) -> dict:
    split_file = Path(f"data/processed/ad_predictor_split_seed_{seed}.json")

    cmd = [
        "python3",
        "src/scripts/train_ad_predictor.py",
        "--ablation",
        ablation,
        "--seed",
        str(seed),
        "--split-file",
        str(split_file),
        "--epochs",
        str(args.epochs),
        "--hidden-dim",
        str(args.hidden_dim),
        "--lr",
        str(args.lr),
        "--l2",
        str(args.l2),
        "--test-size",
        str(args.test_size),
    ]
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)

    run_dir = newest_run_dir_for_ablation(ablation)
    with open(run_dir / "metrics.json") as f:
        metrics = json.load(f)

    row = {
        "seed": seed,
        "ablation": ablation,
        "run_dir": str(run_dir),
        "n_samples": metrics.get("n_samples"),
        "n_train": metrics.get("n_train"),
        "n_test": metrics.get("n_test"),
        "n_pos": metrics.get("n_pos"),
        "n_neg": metrics.get("n_neg"),
        "train_accuracy": metrics.get("train_accuracy"),
        "test_accuracy": metrics.get("test_accuracy"),
        "test_auroc": metrics.get("test_auroc"),
        "test_auprc": metrics.get("test_auprc"),
    }
    return row


def main() -> None:
    args = parse_args()
    seeds = [int(x.strip()) for x in args.seeds.split(",") if x.strip()]

    rows: list[dict] = []
    for seed in seeds:
        for ablation in ABLATIONS:
            rows.append(run_one(seed=seed, ablation=ablation, args=args))

    per_run = pd.DataFrame(rows)
    summary = (
        per_run.groupby("ablation", as_index=False)
        .agg(
            runs=("seed", "count"),
            test_accuracy_mean=("test_accuracy", "mean"),
            test_accuracy_std=("test_accuracy", "std"),
            test_auroc_mean=("test_auroc", "mean"),
            test_auroc_std=("test_auroc", "std"),
            test_auprc_mean=("test_auprc", "mean"),
            test_auprc_std=("test_auprc", "std"),
        )
        .sort_values("test_auroc_mean", ascending=False)
    )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = args.output_dir / f"ad_predictor_multiseed_{timestamp}"
    out_dir.mkdir(parents=True, exist_ok=True)
    per_run.to_csv(out_dir / "per_run_metrics.csv", index=False)
    summary.to_csv(out_dir / "summary_by_ablation.csv", index=False)

    print("\nPer-run metrics:")
    print(per_run.to_string(index=False))
    print("\nSummary:")
    print(summary.to_string(index=False))
    print(f"\nWrote: {out_dir / 'per_run_metrics.csv'}")
    print(f"Wrote: {out_dir / 'summary_by_ablation.csv'}")


if __name__ == "__main__":
    main()
