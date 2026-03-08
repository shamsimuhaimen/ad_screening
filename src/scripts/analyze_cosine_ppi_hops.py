#!/usr/bin/env python3
"""Compare cosine-neighbor hops against PPI connectivity for AD vs control genes.

For each seed gene embedding, we take top-k cosine neighbors (k in {1,2,5} by
default) and measure:
- direct PPI connectivity from seed -> neighbors
- fraction of neighbors labeled AD
- fraction of neighbors connected to any AD gene in PPI

Then we compare these metrics between AD seeds and control seeds.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--labels-csv", type=Path, required=True)
    p.add_argument("--mapping-csv", type=Path, required=True)
    p.add_argument("--ppi-path", type=Path, default=Path("data/processed/string_gene_edges.tsv"))
    p.add_argument("--source-col", type=str, default="gene1")
    p.add_argument("--target-col", type=str, default="gene2")
    p.add_argument("--score-col", type=str, default="combined_score")
    p.add_argument("--min-score", type=float, default=700.0)
    p.add_argument("--embeddings-npy", type=Path, default=Path("data/download/dtwg_af_embeddings.npy"))
    p.add_argument("--names-npy", type=Path, default=Path("data/download/dtwg_af_names_.npy"))
    p.add_argument("--hops", type=str, default="1,2,5")
    p.add_argument("--n-permutations", type=int, default=1000)
    p.add_argument(
        "--permute-mode",
        choices=["label_shuffle", "degree_matched"],
        default="degree_matched",
        help="Permutation null for AD/control seed labels.",
    )
    p.add_argument("--degree-bins", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output-dir", type=Path, default=None)
    return p.parse_args()


def load_gene_embeddings(
    labels_csv: Path,
    mapping_csv: Path,
    embeddings_npy: Path,
    names_npy: Path,
) -> tuple[pd.DataFrame, np.ndarray]:
    labels = pd.read_csv(labels_csv)
    labels = labels[["gene_symbol", "label"]].drop_duplicates(subset=["gene_symbol"], keep="first").copy()
    labels["gene_symbol"] = labels["gene_symbol"].astype(str).str.strip().str.upper()
    labels["y"] = (labels["label"].astype(str).str.upper() == "AD").astype(int)

    mapping = pd.read_csv(mapping_csv)
    mapping = mapping[["gene_symbol", "uniprot_accession"]].dropna().copy()
    mapping["gene_symbol"] = mapping["gene_symbol"].astype(str).str.strip().str.upper()
    mapping["uniprot_accession"] = mapping["uniprot_accession"].astype(str).str.strip().str.upper()

    names = np.load(names_npy, allow_pickle=True)
    embeddings = np.load(embeddings_npy)
    accession = pd.Series(names).astype(str).str.extract(r"AF-([A-Z0-9]+)-F1", expand=False).str.upper()
    names_df = pd.DataFrame({"uniprot_accession": accession, "row_idx": np.arange(len(accession), dtype=int)}).dropna()

    merged = labels.merge(mapping, on="gene_symbol", how="left").merge(names_df, on="uniprot_accession", how="inner")
    if merged.empty:
        raise ValueError("No overlap between labels/mapping and embedding names.")

    idx_by_gene = merged.groupby("gene_symbol")["row_idx"].apply(list).to_dict()
    gene_df = merged.groupby("gene_symbol", as_index=False).agg({"label": "first", "y": "first"})
    x = np.vstack([embeddings[np.asarray(idx_by_gene[g], dtype=int)].mean(axis=0) for g in gene_df["gene_symbol"]])
    return gene_df, x


def load_ppi_adjacency(
    ppi_path: Path,
    source_col: str,
    target_col: str,
    score_col: str | None,
    min_score: float,
) -> dict[str, set[str]]:
    path_str = str(ppi_path).lower()
    if path_str.endswith(".tsv") or path_str.endswith(".tsv.gz") or path_str.endswith(".txt") or path_str.endswith(".txt.gz"):
        ppi = pd.read_csv(ppi_path, sep="\t", low_memory=False)
    else:
        ppi = pd.read_csv(ppi_path, sep=None, engine="python")
    cols = [source_col, target_col] + ([score_col] if score_col and score_col in ppi.columns else [])
    ppi = ppi[cols].copy()
    if score_col and score_col in ppi.columns:
        ppi = ppi[pd.to_numeric(ppi[score_col], errors="coerce") >= float(min_score)]

    ppi[source_col] = ppi[source_col].astype(str).str.strip().str.upper()
    ppi[target_col] = ppi[target_col].astype(str).str.strip().str.upper()
    ppi = ppi[(ppi[source_col] != "") & (ppi[target_col] != "") & (ppi[source_col] != ppi[target_col])]

    adj: dict[str, set[str]] = {}
    for s, t in ppi[[source_col, target_col]].itertuples(index=False):
        adj.setdefault(s, set()).add(t)
        adj.setdefault(t, set()).add(s)
    return adj


def compute_neighbor_rows(
    gene_df: pd.DataFrame,
    x: np.ndarray,
    adj: dict[str, set[str]],
    hops: list[int],
) -> pd.DataFrame:
    genes = gene_df["gene_symbol"].tolist()
    y = gene_df["y"].to_numpy(dtype=int)
    gene_set = set(genes)
    ad_genes = {g for g, yy in zip(genes, y) if yy == 1}

    x_norm = x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)
    sim = x_norm @ x_norm.T
    np.fill_diagonal(sim, -np.inf)

    max_k = max(hops)
    order = np.argsort(-sim, axis=1)[:, :max_k]
    idx_to_gene = np.asarray(genes, dtype=object)

    rows = []
    for i, seed in enumerate(genes):
        seed_neighbors_ppi = adj.get(seed, set())
        for k in hops:
            n_idx = order[i, :k]
            neighbors = idx_to_gene[n_idx].tolist()
            neigh_y = y[n_idx]

            seed_to_neighbor_ppi = [1 if n in seed_neighbors_ppi else 0 for n in neighbors]
            neighbor_to_any_ad_ppi = []
            for n in neighbors:
                n_adj = adj.get(n, set()).intersection(gene_set)
                neighbor_to_any_ad_ppi.append(1 if len(n_adj.intersection(ad_genes)) > 0 else 0)

            rows.append(
                {
                    "seed_gene": seed,
                    "seed_y": int(y[i]),
                    "k": int(k),
                    "seed_neighbor_ppi_rate": float(np.mean(seed_to_neighbor_ppi)),
                    "neighbor_ad_fraction": float(np.mean(neigh_y)),
                    "neighbor_to_any_ad_ppi_rate": float(np.mean(neighbor_to_any_ad_ppi)),
                }
            )
    return pd.DataFrame(rows)


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    g = (
        df.groupby(["k", "seed_y"], as_index=False)
        .agg(
            n_seeds=("seed_gene", "nunique"),
            mean_seed_neighbor_ppi_rate=("seed_neighbor_ppi_rate", "mean"),
            mean_neighbor_ad_fraction=("neighbor_ad_fraction", "mean"),
            mean_neighbor_to_any_ad_ppi_rate=("neighbor_to_any_ad_ppi_rate", "mean"),
        )
        .copy()
    )
    g["seed_label"] = np.where(g["seed_y"] == 1, "AD", "control")

    deltas = []
    for k, sub in g.groupby("k"):
        ad = sub[sub["seed_y"] == 1]
        ct = sub[sub["seed_y"] == 0]
        if ad.empty or ct.empty:
            continue
        deltas.append(
            {
                "k": int(k),
                "delta_seed_neighbor_ppi_rate_ad_minus_control": float(
                    ad["mean_seed_neighbor_ppi_rate"].iloc[0] - ct["mean_seed_neighbor_ppi_rate"].iloc[0]
                ),
                "delta_neighbor_ad_fraction_ad_minus_control": float(
                    ad["mean_neighbor_ad_fraction"].iloc[0] - ct["mean_neighbor_ad_fraction"].iloc[0]
                ),
                "delta_neighbor_to_any_ad_ppi_rate_ad_minus_control": float(
                    ad["mean_neighbor_to_any_ad_ppi_rate"].iloc[0] - ct["mean_neighbor_to_any_ad_ppi_rate"].iloc[0]
                ),
            }
        )
    d = pd.DataFrame(deltas)
    return g, d


def compute_degree_bins(gene_df: pd.DataFrame, adj: dict[str, set[str]], n_bins: int) -> dict[str, int]:
    deg = pd.Series({g: len(adj.get(g, set())) for g in gene_df["gene_symbol"].tolist()})
    q = min(max(2, n_bins), int(deg.nunique()))
    if q <= 1:
        return {g: 0 for g in gene_df["gene_symbol"].tolist()}
    try:
        bins = pd.qcut(deg, q=q, labels=False, duplicates="drop")
    except ValueError:
        return {g: 0 for g in gene_df["gene_symbol"].tolist()}
    return {g: int(bins.loc[g]) for g in gene_df["gene_symbol"].tolist()}


def delta_from_rows(rows: pd.DataFrame, metric: str, k: int, y_col: str = "seed_y") -> float:
    sub = rows[rows["k"] == k]
    ad = sub[sub[y_col] == 1][metric].to_numpy(dtype=float)
    ct = sub[sub[y_col] == 0][metric].to_numpy(dtype=float)
    if len(ad) == 0 or len(ct) == 0:
        return float("nan")
    return float(np.mean(ad) - np.mean(ct))


def permutation_significance(
    rows: pd.DataFrame,
    gene_df: pd.DataFrame,
    degree_bin_by_gene: dict[str, int],
    metrics: list[str],
    hops: list[int],
    n_perm: int,
    permute_mode: str,
    seed: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    gene_to_y = dict(zip(gene_df["gene_symbol"], gene_df["y"].astype(int)))
    genes = gene_df["gene_symbol"].tolist()
    y0 = np.asarray([gene_to_y[g] for g in genes], dtype=int)
    bin_arr = np.asarray([degree_bin_by_gene[g] for g in genes], dtype=int)
    uniq_bins = np.unique(bin_arr)

    obs = {}
    for k in hops:
        for m in metrics:
            obs[(k, m)] = delta_from_rows(rows, metric=m, k=k)

    ge_count = {(k, m): 0 for k in hops for m in metrics}
    progress_every = max(1, n_perm // 10)
    for i in range(n_perm):
        yp = y0.copy()
        if permute_mode == "label_shuffle":
            yp = rng.permutation(yp)
        else:
            for b in uniq_bins:
                idx = np.where(bin_arr == b)[0]
                if len(idx) > 1:
                    yp[idx] = rng.permutation(yp[idx])
        y_map = dict(zip(genes, yp.tolist()))
        perm_rows = rows.copy()
        perm_rows["seed_y_perm"] = perm_rows["seed_gene"].map(y_map).astype(int)
        for k in hops:
            sub = perm_rows[perm_rows["k"] == k]
            for m in metrics:
                ad = sub[sub["seed_y_perm"] == 1][m].to_numpy(dtype=float)
                ct = sub[sub["seed_y_perm"] == 0][m].to_numpy(dtype=float)
                if len(ad) == 0 or len(ct) == 0:
                    continue
                d = float(np.mean(ad) - np.mean(ct))
                if d >= obs[(k, m)]:
                    ge_count[(k, m)] += 1
        if (i + 1) % progress_every == 0 or (i + 1) == n_perm:
            print(f"      permutation progress: {i + 1}/{n_perm}")

    out = []
    for k in hops:
        for m in metrics:
            out.append(
                {
                    "k": int(k),
                    "metric": m,
                    "delta_ad_minus_control": float(obs[(k, m)]),
                    "pvalue_right": float((ge_count[(k, m)] + 1) / (n_perm + 1)),
                    "permute_mode": permute_mode,
                    "n_permutations": int(n_perm),
                }
            )
    return pd.DataFrame(out)


def main() -> None:
    args = parse_args()
    hops = sorted({int(x.strip()) for x in args.hops.split(",") if x.strip()})
    if not hops:
        raise ValueError("No valid hops provided.")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = args.output_dir or Path(f"results/cosine_ppi_hops_{timestamp}")
    out_dir.mkdir(parents=True, exist_ok=True)

    print("[1/4] Loading gene-level embeddings...")
    gene_df, x = load_gene_embeddings(args.labels_csv, args.mapping_csv, args.embeddings_npy, args.names_npy)
    print(f"      genes with embeddings: {len(gene_df)}")

    print("[2/4] Loading PPI adjacency...")
    adj = load_ppi_adjacency(args.ppi_path, args.source_col, args.target_col, args.score_col, args.min_score)
    keep = gene_df["gene_symbol"].isin(adj)
    gene_df = gene_df.loc[keep].reset_index(drop=True)
    x = x[keep.to_numpy()]
    print(f"      genes with embeddings+PPI: {len(gene_df)}")

    print("[3/4] Computing cosine-neighbor hop metrics...")
    rows = compute_neighbor_rows(gene_df, x, adj, hops=hops)

    print("[3.5/4] Running permutation significance...")
    degree_bin_by_gene = compute_degree_bins(gene_df, adj, args.degree_bins)
    metrics = ["seed_neighbor_ppi_rate", "neighbor_ad_fraction", "neighbor_to_any_ad_ppi_rate"]
    signif = permutation_significance(
        rows=rows,
        gene_df=gene_df,
        degree_bin_by_gene=degree_bin_by_gene,
        metrics=metrics,
        hops=hops,
        n_perm=args.n_permutations,
        permute_mode=args.permute_mode,
        seed=args.seed,
    )

    print("[4/4] Writing outputs...")
    summary_by_label, summary_delta = summarize(rows)
    rows.to_csv(out_dir / "per_seed_hop_metrics.csv", index=False)
    summary_by_label.to_csv(out_dir / "summary_by_seed_label.csv", index=False)
    summary_delta.to_csv(out_dir / "summary_delta_ad_minus_control.csv", index=False)
    signif.to_csv(out_dir / "summary_significance.csv", index=False)

    summary = {
        "n_genes_used": int(len(gene_df)),
        "n_ad_genes_used": int(gene_df["y"].sum()),
        "n_control_genes_used": int((gene_df["y"] == 0).sum()),
        "hops": hops,
        "permute_mode": args.permute_mode,
        "n_permutations": int(args.n_permutations),
        "degree_bins": int(args.degree_bins),
    }
    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))
    print(f"Wrote: {out_dir / 'summary_by_seed_label.csv'}")
    print(f"Wrote: {out_dir / 'summary_delta_ad_minus_control.csv'}")
    print(f"Wrote: {out_dir / 'summary_significance.csv'}")


if __name__ == "__main__":
    main()
