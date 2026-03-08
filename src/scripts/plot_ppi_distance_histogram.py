#!/usr/bin/env python3
"""Plot PPI shortest-path distance histograms for AD vs control gene pairs."""

from __future__ import annotations

import argparse
import json
import time
from collections import deque
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--labels-csv", type=Path, required=True)
    p.add_argument("--ppi-path", type=Path, required=True)
    p.add_argument("--source-col", type=str, default="gene1")
    p.add_argument("--target-col", type=str, default="gene2")
    p.add_argument("--score-col", type=str, default="combined_score")
    p.add_argument("--min-score", type=float, default=700.0)
    p.add_argument("--max-sources", type=int, default=0, help="0 means use all labeled genes as BFS sources.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output-dir", type=Path, default=None)
    return p.parse_args()


def load_labels(labels_csv: Path) -> pd.DataFrame:
    labels = pd.read_csv(labels_csv)
    labels = labels[["gene_symbol", "y"]].drop_duplicates(subset=["gene_symbol"], keep="first").copy()
    labels["gene_symbol"] = labels["gene_symbol"].astype(str).str.strip().str.upper()
    labels["y"] = labels["y"].astype(int)
    return labels


def load_ppi(ppi_path: Path, source_col: str, target_col: str, score_col: str, min_score: float) -> dict[str, set[str]]:
    path = str(ppi_path).lower()
    if path.endswith(".tsv") or path.endswith(".tsv.gz") or path.endswith(".txt") or path.endswith(".txt.gz"):
        ppi = pd.read_csv(ppi_path, sep="\t", low_memory=False)
    else:
        ppi = pd.read_csv(ppi_path, sep=None, engine="python")
    cols = [source_col, target_col] + ([score_col] if score_col in ppi.columns else [])
    ppi = ppi[cols].copy()
    if score_col in ppi.columns:
        ppi = ppi[pd.to_numeric(ppi[score_col], errors="coerce") >= float(min_score)]

    ppi[source_col] = ppi[source_col].astype(str).str.strip().str.upper()
    ppi[target_col] = ppi[target_col].astype(str).str.strip().str.upper()
    ppi = ppi[(ppi[source_col] != "") & (ppi[target_col] != "") & (ppi[source_col] != ppi[target_col])]

    adj: dict[str, set[str]] = {}
    for s, t in ppi[[source_col, target_col]].itertuples(index=False):
        adj.setdefault(s, set()).add(t)
        adj.setdefault(t, set()).add(s)
    return adj


def bfs_dist_to_targets(source: str, adj: dict[str, set[str]], target_set: set[str]) -> dict[str, int]:
    dist = {source: 0}
    q = deque([source])
    hit = {}
    remaining = set(target_set)
    if source in remaining:
        remaining.remove(source)
    while q and remaining:
        u = q.popleft()
        du = dist[u]
        for v in adj.get(u, set()):
            if v in dist:
                continue
            dist[v] = du + 1
            if v in remaining:
                hit[v] = dist[v]
                remaining.remove(v)
            q.append(v)
    return hit


def main() -> None:
    t0 = time.perf_counter()
    args = parse_args()
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = args.output_dir or Path(f"results/ppi_distance_hist_{ts}")
    out_dir.mkdir(parents=True, exist_ok=True)

    print("[1/5] Loading labels...")
    labels = load_labels(args.labels_csv)
    print(f"      labels: {len(labels)}")

    print("[2/5] Loading PPI...")
    adj = load_ppi(args.ppi_path, args.source_col, args.target_col, args.score_col, args.min_score)
    print(f"      ppi nodes: {len(adj)}")

    print("[3/5] Intersecting labels with PPI...")
    labels = labels[labels["gene_symbol"].isin(adj)].copy().reset_index(drop=True)
    if len(labels) < 20:
        raise ValueError("Too few labeled genes overlap with PPI.")
    print(f"      labeled genes in PPI: {len(labels)}")

    genes = labels["gene_symbol"].tolist()
    y = labels["y"].to_numpy(dtype=int)
    index = {g: i for i, g in enumerate(genes)}
    target_set = set(genes)

    source_idx = np.arange(len(genes))
    if args.max_sources > 0 and args.max_sources < len(genes):
        rng = np.random.default_rng(args.seed)
        source_idx = np.sort(rng.choice(source_idx, size=args.max_sources, replace=False))
    print(f"      BFS sources: {len(source_idx)}")

    print("[4/5] Computing pairwise shortest-path distances...")
    rows = []
    progress_every = max(1, len(source_idx) // 10)
    for step, i in enumerate(source_idx, start=1):
        g = genes[i]
        dist = bfs_dist_to_targets(g, adj, target_set)
        for h, d in dist.items():
            j = index[h]
            if j <= i:
                continue
            a = y[i]
            b = y[j]
            if a == 1 and b == 1:
                pair_type = "AD-AD"
            elif a == 0 and b == 0:
                pair_type = "control-control"
            else:
                pair_type = "AD-control"
            rows.append({"gene_i": g, "gene_j": h, "pair_type": pair_type, "distance": int(d)})
        if step % progress_every == 0 or step == len(source_idx):
            print(f"      progress: {step}/{len(source_idx)}")

    if not rows:
        raise ValueError("No reachable labeled pairs found.")

    df = pd.DataFrame(rows)
    summary = (
        df.groupby("pair_type", as_index=False)
        .agg(
            n_pairs=("distance", "size"),
            mean_distance=("distance", "mean"),
            median_distance=("distance", "median"),
            p90_distance=("distance", lambda s: float(np.percentile(s.to_numpy(), 90))),
        )
        .sort_values("pair_type")
    )

    print("[5/5] Writing histogram and tables...")
    plt.figure(figsize=(8, 5))
    colors = {"AD-AD": "#d95f02", "AD-control": "#7570b3", "control-control": "#1b9e77"}
    max_d = int(df["distance"].max())
    bins = np.arange(0.5, max_d + 1.5, 1.0)
    for pair_type in ["AD-AD", "AD-control", "control-control"]:
        vals = df.loc[df["pair_type"] == pair_type, "distance"].to_numpy(dtype=float)
        if len(vals):
            plt.hist(vals, bins=bins, alpha=0.45, density=True, label=pair_type, color=colors[pair_type])
    plt.xlabel("Shortest Path Distance in PPI")
    plt.ylabel("Density")
    plt.title("PPI Distance Histogram by Pair Type")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "ppi_distance_histogram.png", dpi=220)
    plt.close()

    df.to_csv(out_dir / "pair_distances.csv", index=False)
    summary.to_csv(out_dir / "distance_summary.csv", index=False)
    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "n_labeled_genes_in_ppi": int(len(genes)),
                "n_sources_used": int(len(source_idx)),
                "n_pairs_with_path": int(len(df)),
                "max_distance": int(max_d),
                "runtime_seconds": float(time.perf_counter() - t0),
            },
            f,
            indent=2,
        )

    print(summary.to_string(index=False))
    print(f"Wrote: {out_dir / 'ppi_distance_histogram.png'}")
    print(f"Wrote: {out_dir / 'distance_summary.csv'}")


if __name__ == "__main__":
    main()
