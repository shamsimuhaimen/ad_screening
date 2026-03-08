#!/usr/bin/env python3
"""Characterize DrugCLIP embedding space for AD-vs-control signal.

This script computes compact diagnostics for the LOI:
- kNN label purity and neighbor-label entropy
- pairwise cosine-distance distributions by class pair
- PCA structure (explained variance, 2D projection)
- lightweight linear probe summary across seeds
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--labels-csv", type=Path, required=True)
    p.add_argument("--mapping-csv", type=Path, required=True)
    p.add_argument("--embeddings-npy", type=Path, default=Path("data/download/dtwg_af_embeddings.npy"))
    p.add_argument("--names-npy", type=Path, default=Path("data/download/dtwg_af_names_.npy"))
    p.add_argument("--k", type=int, default=10)
    p.add_argument("--probe-seeds", type=str, default="42,43,44,45,46")
    p.add_argument("--probe-test-size", type=float, default=0.2)
    p.add_argument("--probe-epochs", type=int, default=200)
    p.add_argument("--probe-lr", type=float, default=0.05)
    p.add_argument("--probe-l2", type=float, default=1e-3)
    p.add_argument("--output-dir", type=Path, default=None)
    return p.parse_args()


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -40, 40)))


def roc_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    pos = y_true == 1
    neg = y_true == 0
    n_pos = int(pos.sum())
    n_neg = int(neg.sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = pd.Series(y_score).rank(method="average").to_numpy()
    rank_sum_pos = ranks[pos].sum()
    return float((rank_sum_pos - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def pr_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    order = np.argsort(-y_score)
    y = y_true[order]
    tp = np.cumsum(y == 1)
    fp = np.cumsum(y == 0)
    total_pos = int((y_true == 1).sum())
    if total_pos == 0:
        return float("nan")
    recall = tp / total_pos
    precision = tp / np.maximum(tp + fp, 1)
    recall = np.concatenate(([0.0], recall))
    precision = np.concatenate(([1.0], precision))
    if hasattr(np, "trapezoid"):
        return float(np.trapezoid(precision, recall))
    return float(np.trapz(precision, recall))


def train_logistic(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    epochs: int,
    lr: float,
    l2: float,
) -> dict[str, float]:
    n, d = x_train.shape
    w = np.zeros(d, dtype=np.float64)
    b = 0.0
    for _ in range(epochs):
        probs = sigmoid(x_train @ w + b)
        err = probs - y_train
        grad_w = (x_train.T @ err) / n + l2 * w
        grad_b = float(err.mean())
        w -= lr * grad_w
        b -= lr * grad_b

    train_probs = sigmoid(x_train @ w + b)
    test_probs = sigmoid(x_test @ w + b)
    train_pred = (train_probs >= 0.5).astype(int)
    test_pred = (test_probs >= 0.5).astype(int)
    return {
        "train_accuracy": float((train_pred == y_train).mean()),
        "test_accuracy": float((test_pred == y_test).mean()),
        "test_auroc": roc_auc(y_test, test_probs),
        "test_auprc": pr_auc(y_test, test_probs),
    }


def build_gene_embeddings(
    labels_csv: Path,
    mapping_csv: Path,
    embeddings_npy: Path,
    names_npy: Path,
) -> tuple[pd.DataFrame, np.ndarray]:
    labels = pd.read_csv(labels_csv)
    required_labels = {"gene_symbol", "label"}
    if not required_labels.issubset(labels.columns):
        raise ValueError(f"{labels_csv} must contain columns: {sorted(required_labels)}")
    labels = labels[["gene_symbol", "label"]].drop_duplicates(subset=["gene_symbol"], keep="first").copy()
    labels["gene_symbol"] = labels["gene_symbol"].astype(str).str.upper().str.strip()
    labels["y"] = (labels["label"].astype(str).str.upper() == "AD").astype(int)

    mapping = pd.read_csv(mapping_csv)
    required_mapping = {"gene_symbol", "uniprot_accession"}
    if not required_mapping.issubset(mapping.columns):
        raise ValueError(f"{mapping_csv} must contain columns: {sorted(required_mapping)}")
    mapping = mapping[list(required_mapping)].copy()
    mapping["gene_symbol"] = mapping["gene_symbol"].astype(str).str.upper().str.strip()
    mapping["uniprot_accession"] = mapping["uniprot_accession"].astype(str).str.upper().str.strip()

    names = np.load(names_npy, allow_pickle=True)
    embeddings = np.load(embeddings_npy)
    acc = pd.Series(names).astype(str).str.extract(r"AF-([A-Z0-9]+)-F1", expand=False).str.upper()
    names_df = pd.DataFrame({"uniprot_accession": acc, "row_idx": np.arange(len(acc))}).dropna()

    merged = labels.merge(mapping, on="gene_symbol", how="left").merge(names_df, on="uniprot_accession", how="inner")
    if merged.empty:
        raise ValueError("No overlap after labels/mapping/embedding-name join.")

    idx_by_gene = merged.groupby("gene_symbol")["row_idx"].apply(list)
    gene_df = (
        merged.groupby("gene_symbol", as_index=False)
        .agg({"label": "first", "y": "first", "uniprot_accession": "first"})
        .copy()
    )
    x = np.vstack([embeddings[np.asarray(idx_by_gene[g], dtype=int)].mean(axis=0) for g in gene_df["gene_symbol"]])
    return gene_df, x


def knn_metrics(y: np.ndarray, x: np.ndarray, k: int) -> pd.DataFrame:
    x_norm = x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)
    sim = x_norm @ x_norm.T
    np.fill_diagonal(sim, -np.inf)
    neigh_idx = np.argpartition(-sim, kth=min(k, sim.shape[1] - 1), axis=1)[:, :k]

    rows = []
    for i in range(len(y)):
        ny = y[neigh_idx[i]]
        p1 = float(ny.mean())
        p0 = 1.0 - p1
        entropy = 0.0
        if p0 > 0:
            entropy -= p0 * np.log2(p0)
        if p1 > 0:
            entropy -= p1 * np.log2(p1)
        purity = float((ny == y[i]).mean())
        rows.append({"idx": i, "y": int(y[i]), "knn_purity": purity, "knn_entropy": float(entropy)})
    return pd.DataFrame(rows)


def distance_summary(y: np.ndarray, x: np.ndarray) -> pd.DataFrame:
    x_norm = x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)
    dist = 1.0 - (x_norm @ x_norm.T)
    iu = np.triu_indices(len(y), k=1)
    a = y[iu[0]]
    b = y[iu[1]]
    d = dist[iu]

    pairs = []
    masks = {
        "AD-AD": (a == 1) & (b == 1),
        "AD-control": (a != b),
        "control-control": (a == 0) & (b == 0),
    }
    for name, m in masks.items():
        vals = d[m]
        pairs.append(
            {
                "pair_type": name,
                "n_pairs": int(len(vals)),
                "mean_cosine_distance": float(np.mean(vals)) if len(vals) else float("nan"),
                "median_cosine_distance": float(np.median(vals)) if len(vals) else float("nan"),
            }
        )
    return pd.DataFrame(pairs)


def plot_dist_hist(y: np.ndarray, x: np.ndarray, out_path: Path) -> None:
    x_norm = x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)
    dist = 1.0 - (x_norm @ x_norm.T)
    iu = np.triu_indices(len(y), k=1)
    a = y[iu[0]]
    b = y[iu[1]]
    d = dist[iu]

    plt.figure(figsize=(8, 5))
    for label, mask, color in [
        ("AD-AD", (a == 1) & (b == 1), "#d95f02"),
        ("AD-control", (a != b), "#7570b3"),
        ("control-control", (a == 0) & (b == 0), "#1b9e77"),
    ]:
        vals = d[mask]
        if len(vals):
            plt.hist(vals, bins=40, alpha=0.45, density=True, label=label, color=color)
    plt.xlabel("Cosine Distance")
    plt.ylabel("Density")
    plt.title("Pairwise Distance Distributions")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def pca_plots(x: np.ndarray, y: np.ndarray, out_var: Path, out_scatter: Path) -> dict[str, float]:
    x_centered = x - x.mean(axis=0, keepdims=True)
    u, s, vt = np.linalg.svd(x_centered, full_matrices=False)
    var = (s**2) / max(len(x_centered) - 1, 1)
    evr = var / np.maximum(var.sum(), 1e-12)

    n_pc = min(50, len(evr))
    plt.figure(figsize=(8, 5))
    plt.plot(np.arange(1, n_pc + 1), np.cumsum(evr[:n_pc]), marker="o", linewidth=1)
    plt.xlabel("Number of PCs")
    plt.ylabel("Cumulative Explained Variance")
    plt.title("PCA Variance Curve")
    plt.tight_layout()
    plt.savefig(out_var, dpi=200)
    plt.close()

    pc = x_centered @ vt.T
    plt.figure(figsize=(7, 6))
    mask_ad = y == 1
    plt.scatter(pc[~mask_ad, 0], pc[~mask_ad, 1], s=12, alpha=0.7, label="control", c="#1b9e77")
    plt.scatter(pc[mask_ad, 0], pc[mask_ad, 1], s=12, alpha=0.7, label="AD", c="#d95f02")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title("PCA Projection (PC1 vs PC2)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_scatter, dpi=200)
    plt.close()

    effective_dim = float((var.sum() ** 2) / np.maximum((var**2).sum(), 1e-12))
    return {
        "pc1_explained_variance_ratio": float(evr[0]) if len(evr) else float("nan"),
        "pc2_explained_variance_ratio": float(evr[1]) if len(evr) > 1 else float("nan"),
        "effective_rank_participation_ratio": effective_dim,
    }


def run_probe_multi_seed(
    x: np.ndarray,
    y: np.ndarray,
    seeds: list[int],
    test_size: float,
    epochs: int,
    lr: float,
    l2: float,
) -> tuple[pd.DataFrame, dict[str, float]]:
    rows = []
    gene_ids = np.arange(len(y))
    for seed in seeds:
        rng = np.random.default_rng(seed)
        pos = gene_ids[y == 1].copy()
        neg = gene_ids[y == 0].copy()
        rng.shuffle(pos)
        rng.shuffle(neg)
        n_pos_test = max(1, int(round(len(pos) * test_size)))
        n_neg_test = max(1, int(round(len(neg) * test_size)))
        test_idx = np.concatenate([pos[:n_pos_test], neg[:n_neg_test]])
        train_idx = np.setdiff1d(gene_ids, test_idx)

        x_train = x[train_idx]
        y_train = y[train_idx]
        x_test = x[test_idx]
        y_test = y[test_idx]

        mu = x_train.mean(axis=0)
        sd = x_train.std(axis=0)
        sd[sd == 0] = 1.0
        x_train = (x_train - mu) / sd
        x_test = (x_test - mu) / sd

        m = train_logistic(x_train, y_train, x_test, y_test, epochs=epochs, lr=lr, l2=l2)
        m["seed"] = seed
        rows.append(m)

    df = pd.DataFrame(rows)
    summary = {
        "probe_runs": int(len(df)),
        "probe_test_accuracy_mean": float(df["test_accuracy"].mean()),
        "probe_test_accuracy_std": float(df["test_accuracy"].std(ddof=1)),
        "probe_test_auroc_mean": float(df["test_auroc"].mean()),
        "probe_test_auroc_std": float(df["test_auroc"].std(ddof=1)),
        "probe_test_auprc_mean": float(df["test_auprc"].mean()),
        "probe_test_auprc_std": float(df["test_auprc"].std(ddof=1)),
    }
    return df, summary


def main() -> None:
    args = parse_args()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = args.output_dir or Path(f"results/embedding_space_characterization_{timestamp}")
    out_dir.mkdir(parents=True, exist_ok=True)

    print("[1/4] Building gene-level embedding table...")
    gene_df, x = build_gene_embeddings(
        labels_csv=args.labels_csv,
        mapping_csv=args.mapping_csv,
        embeddings_npy=args.embeddings_npy,
        names_npy=args.names_npy,
    )
    y = gene_df["y"].to_numpy(dtype=int)
    gene_df.to_csv(out_dir / "gene_embedding_index.csv", index=False)
    print(f"      genes with embeddings: {len(gene_df)}")

    print("[2/4] Computing kNN and distance diagnostics...")
    k = min(args.k, max(1, len(gene_df) - 1))
    knn_df = knn_metrics(y=y, x=x, k=k)
    knn_df.to_csv(out_dir / "knn_gene_metrics.csv", index=False)
    knn_summary = (
        knn_df.groupby("y")[["knn_purity", "knn_entropy"]]
        .mean()
        .rename(index={0: "control", 1: "AD"})
        .reset_index()
    )
    knn_summary.to_csv(out_dir / "knn_summary_by_label.csv", index=False)

    dist_df = distance_summary(y=y, x=x)
    dist_df.to_csv(out_dir / "distance_summary.csv", index=False)
    plot_dist_hist(y=y, x=x, out_path=out_dir / "distance_hist.png")

    print("[3/4] Computing PCA diagnostics...")
    pca_summary = pca_plots(
        x=x,
        y=y,
        out_var=out_dir / "pca_variance_curve.png",
        out_scatter=out_dir / "pca_scatter.png",
    )

    print("[4/4] Running linear probe across seeds...")
    seeds = [int(s.strip()) for s in args.probe_seeds.split(",") if s.strip()]
    probe_df, probe_summary = run_probe_multi_seed(
        x=x,
        y=y,
        seeds=seeds,
        test_size=args.probe_test_size,
        epochs=args.probe_epochs,
        lr=args.probe_lr,
        l2=args.probe_l2,
    )
    probe_df.to_csv(out_dir / "probe_per_seed.csv", index=False)

    summary = {
        "n_genes": int(len(gene_df)),
        "n_ad": int((y == 1).sum()),
        "n_control": int((y == 0).sum()),
        "k_neighbors": int(k),
        "knn_mean_purity": float(knn_df["knn_purity"].mean()),
        "knn_mean_entropy": float(knn_df["knn_entropy"].mean()),
    }
    summary.update(pca_summary)
    summary.update(probe_summary)

    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))
    print(f"Wrote: {out_dir / 'summary.json'}")
    print(f"Wrote: {out_dir / 'knn_gene_metrics.csv'}")
    print(f"Wrote: {out_dir / 'distance_summary.csv'}")
    print(f"Wrote: {out_dir / 'probe_per_seed.csv'}")


if __name__ == "__main__":
    main()
