#!/usr/bin/env python3
"""Aggregate Prefect experiment outputs into empirical p-values and BH-FDR q-values.

Usage:
    python src/scripts/aggregate_experiment_results.py --run-root results/prefect_experiments_YYYYMMDD_HHMMSS
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-root", type=Path, required=True)
    return p.parse_args()


def bh_fdr(pvals: np.ndarray) -> np.ndarray:
    p = np.asarray(pvals, dtype=float)
    n = len(p)
    order = np.argsort(p)
    ranked = p[order]
    q = np.empty(n, dtype=float)
    prev = 1.0
    for i in range(n - 1, -1, -1):
        rank = i + 1
        val = ranked[i] * n / rank
        prev = min(prev, val)
        q[i] = min(prev, 1.0)
    out = np.empty(n, dtype=float)
    out[order] = q
    return out


def collect_runs(run_root: Path) -> pd.DataFrame:
    rows = []
    for manifest_path in sorted(run_root.glob("matrix_runs/*/run_manifest.json")):
        man = json.loads(manifest_path.read_text())
        run_dir = manifest_path.parent

        cls = json.loads((run_dir / "classifier_metrics.json").read_text())
        ppi = json.loads((run_dir / "ppi_signal" / "summary.json").read_text())
        hops = pd.read_csv(run_dir / "cosine_hops" / "summary_significance.csv")

        row = {
            "run_id": man["run_id"],
            "cohort": man["cohort"],
            "ablation": man["ablation"],
            "baseline": man["baseline"],
            "replicate": int(man["replicate"]),
            "seed": int(man["seed"]),
            "probe_test_accuracy": float(cls.get("test_accuracy", np.nan)),
            "probe_test_auroc": float(cls.get("test_auroc", np.nan)),
            "probe_test_auprc": float(cls.get("test_auprc", np.nan)),
            "ppi_ad_ad_edge_enrichment": float(ppi.get("ad_ad_edge_enrichment", np.nan)),
            "ppi_mean_ad_neighbor_fraction": float(ppi.get("mean_ad_neighbor_fraction", np.nan)),
            "ppi_ad_lcc_size": float(ppi.get("ad_lcc_size", np.nan)),
            "ppi_mean_ad_shortest_path": float(ppi.get("mean_ad_shortest_path", np.nan)),
        }

        # Flatten cosine-hop p-values by metric and k.
        for _, r in hops.iterrows():
            k = int(r["k"])
            m = str(r["metric"])
            row[f"hops_{m}_k{k}_delta"] = float(r["delta_ad_minus_control"])
            row[f"hops_{m}_k{k}_pvalue_right"] = float(r["pvalue_right"])

        rows.append(row)

    if not rows:
        raise FileNotFoundError(f"No manifests found under: {run_root}")
    return pd.DataFrame(rows)


def empirical_pvalue(null_values: np.ndarray, observed_value: float, direction: str = "right") -> float:
    if len(null_values) == 0 or np.isnan(observed_value):
        return np.nan
    if direction == "right":
        return float((1 + np.sum(null_values >= observed_value)) / (1 + len(null_values)))
    return float((1 + np.sum(null_values <= observed_value)) / (1 + len(null_values)))


def main() -> None:
    args = parse_args()
    run_root = args.run_root

    df = collect_runs(run_root)
    df.to_csv(run_root / "aggregated_per_run.csv", index=False)

    metric_cols = [
        "probe_test_auroc",
        "probe_test_auprc",
        "probe_test_accuracy",
        "ppi_ad_ad_edge_enrichment",
        "ppi_mean_ad_neighbor_fraction",
        "ppi_ad_lcc_size",
        "ppi_mean_ad_shortest_path",
    ]
    metric_cols.extend([c for c in df.columns if c.startswith("hops_") and c.endswith("_delta")])

    results = []
    group_cols = ["cohort", "ablation", "seed"]
    for keys, sub in df.groupby(group_cols):
        obs = sub[sub["baseline"] == "observed_ad"]
        null = sub[sub["baseline"] != "observed_ad"]
        if obs.empty or null.empty:
            continue
        for m in metric_cols:
            if m not in sub.columns:
                continue
            obs_val = float(obs[m].mean())
            null_vals = null[m].dropna().to_numpy(dtype=float)
            direction = "left" if m == "ppi_mean_ad_shortest_path" else "right"
            p_emp = empirical_pvalue(null_vals, obs_val, direction=direction)
            null_mean = float(np.mean(null_vals)) if len(null_vals) else np.nan
            null_std = float(np.std(null_vals, ddof=1)) if len(null_vals) > 1 else np.nan
            z = (obs_val - null_mean) / null_std if np.isfinite(null_std) and null_std > 0 else np.nan
            results.append(
                {
                    "cohort": keys[0],
                    "ablation": keys[1],
                    "seed": keys[2],
                    "metric": m,
                    "observed": obs_val,
                    "null_mean": null_mean,
                    "null_std": null_std,
                    "effect_z": z,
                    "empirical_pvalue": p_emp,
                    "n_null": int(len(null_vals)),
                    "direction": direction,
                }
            )

    summary = pd.DataFrame(results)
    if summary.empty:
        raise RuntimeError("No comparable observed-vs-null groups found.")

    summary["q_value_bh_fdr"] = bh_fdr(summary["empirical_pvalue"].to_numpy(dtype=float))
    summary = summary.sort_values(["q_value_bh_fdr", "empirical_pvalue", "metric"]).reset_index(drop=True)

    summary.to_csv(run_root / "hypothesis_tests.csv", index=False)

    top = summary.head(30)
    print(top.to_string(index=False))
    print(f"\nWrote: {run_root / 'aggregated_per_run.csv'}")
    print(f"Wrote: {run_root / 'hypothesis_tests.csv'}")


if __name__ == "__main__":
    main()
