#!/usr/bin/env python3
"""Train AD predictor with a LoRA-style adapter over fixed DrugCLIP embeddings.

This script keeps base DrugCLIP embeddings fixed, learns a low-rank residual
adapter (LoRA-style), and trains a small prediction head.
Outputs are written to results/ with metrics and figures.
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from train_ad_predictor import (
    build_gene_to_uniprot_map,
    build_label_table,
    pr_auc,
    roc_auc,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-dir", type=Path, default=Path("data/raw/bulk_rna_seq_human_brain"))
    p.add_argument("--ad-genes-path", type=Path, default=Path("data/processed/ad_genes.csv"))
    p.add_argument("--hgnc-mapping-path", type=Path, default=Path("data/download/hgnc_complete_set.txt"))
    p.add_argument("--embeddings-npy", type=Path, default=Path("data/download/dtwg_af_embeddings.npy"))
    p.add_argument("--names-npy", type=Path, default=Path("data/download/dtwg_af_names_.npy"))
    p.add_argument("--controls-per-ad", type=int, default=2)
    p.add_argument("--min-global-expression", type=float, default=1e-6)
    p.add_argument("--test-size", type=float, default=0.2)
    p.add_argument("--split-file", type=Path, default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--lr", type=float, default=0.001)
    p.add_argument("--l2", type=float, default=1e-4)
    p.add_argument("--hidden-dim", type=int, default=64)
    p.add_argument("--lora-rank", type=int, default=8)
    p.add_argument("--lora-alpha", type=float, default=16.0)
    return p.parse_args()


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -40, 40)))


def relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(x, 0.0)


def bce(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    eps = 1e-8
    return float(-(y_true * np.log(y_prob + eps) + (1 - y_true) * np.log(1 - y_prob + eps)).mean())


def forward(
    x: np.ndarray,
    a: np.ndarray,
    b: np.ndarray,
    scale: float,
    w1: np.ndarray,
    b1: np.ndarray,
    w2: np.ndarray,
    b2: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    m = x @ a
    n = m @ b
    z = x + scale * n
    h_pre = z @ w1 + b1
    h = relu(h_pre)
    logits = h @ w2 + b2
    probs = sigmoid(logits)
    return z, h_pre, h, probs


def main() -> None:
    t0 = time.perf_counter()
    args = parse_args()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(f"results/ad_predictor_lora_{timestamp}")
    out_dir.mkdir(parents=True, exist_ok=True)

    hparams = vars(args).copy()
    hparams["output_dir"] = str(out_dir)
    hparams["timestamp"] = timestamp

    print("[1/7] Building AD/control labels...")
    labels = build_label_table(args)
    labels.to_csv(out_dir / "labels_used.csv", index=False)
    print(f"      labels built: {len(labels)} genes")

    print("[2/7] Loading embeddings and names...")
    names = np.load(args.names_npy, allow_pickle=True)
    embeddings = np.load(args.embeddings_npy)
    if len(names) != embeddings.shape[0]:
        raise ValueError("names and embeddings row counts do not match.")
    print(f"      embeddings shape: {embeddings.shape}")

    print("[3/7] Mapping genes to embedding rows...")
    acc = pd.Series(names).astype(str).str.extract(r"AF-([A-Z0-9]+)-F1", expand=False).str.upper()
    names_df = pd.DataFrame({"uniprot_accession": acc, "row_idx": np.arange(len(names))}).dropna()
    mapping_df = build_gene_to_uniprot_map(labels["gene_symbol"].tolist(), args.hgnc_mapping_path)
    mapping_df.to_csv(out_dir / "gene_to_uniprot_mapping.csv", index=False)
    merged = labels.merge(mapping_df, on="gene_symbol", how="left").merge(names_df, on="uniprot_accession", how="inner")
    if merged.empty:
        raise ValueError("No overlapping genes between labels and embedding names.")

    idx_by_gene = merged.groupby("gene_symbol")["row_idx"].apply(list)
    gene_df = (
        merged.groupby("gene_symbol", as_index=False)
        .agg({"label": "first", "y": "first", "integrated_score": "mean", "uniprot_accession": "first"})
        .copy()
    )
    x = np.vstack([embeddings[np.asarray(idx_by_gene[g], dtype=int)].mean(axis=0) for g in gene_df["gene_symbol"]])
    y = gene_df["y"].to_numpy(dtype=np.float64)
    print(f"      aggregated genes: {len(gene_df)}")

    print("[4/7] Preparing split...")
    split_file = args.split_file or Path(f"data/processed/ad_predictor_split_seed_{args.seed}.json")
    split_file.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    available_genes = set(gene_df["gene_symbol"].tolist())

    if split_file.exists():
        split_data = json.loads(split_file.read_text())
        test_genes = set(split_data.get("test_genes", [])).intersection(available_genes)
    else:
        pos = gene_df.loc[gene_df["y"] == 1, "gene_symbol"].to_numpy()
        neg = gene_df.loc[gene_df["y"] == 0, "gene_symbol"].to_numpy()
        rng.shuffle(pos)
        rng.shuffle(neg)
        n_pos_test = max(1, int(round(len(pos) * args.test_size)))
        n_neg_test = max(1, int(round(len(neg) * args.test_size)))
        test_genes = set(pos[:n_pos_test]).union(set(neg[:n_neg_test]))
        split_file.write_text(
            json.dumps({"seed": args.seed, "test_size": args.test_size, "test_genes": sorted(test_genes)}, indent=2)
        )

    test_mask = gene_df["gene_symbol"].isin(test_genes).to_numpy()
    train_mask = ~test_mask
    x_train, y_train = x[train_mask], y[train_mask]
    x_test, y_test = x[test_mask], y[test_mask]

    mu = x_train.mean(axis=0)
    sd = x_train.std(axis=0)
    sd[sd == 0] = 1.0
    x_train = (x_train - mu) / sd
    x_test = (x_test - mu) / sd

    print("[5/7] Initializing LoRA adapter + head...")
    d = x_train.shape[1]
    r = int(args.lora_rank)
    scale = float(args.lora_alpha) / max(1, r)
    a = rng.normal(0.0, 0.01, size=(d, r))
    b = rng.normal(0.0, 0.01, size=(r, d))
    w1 = rng.normal(0.0, 0.02, size=(d, args.hidden_dim))
    b1 = np.zeros(args.hidden_dim, dtype=np.float64)
    w2 = rng.normal(0.0, 0.02, size=(args.hidden_dim,))
    b2 = 0.0

    print("[6/7] Training...")
    history: list[dict[str, float]] = []
    n = len(y_train)
    for epoch in range(1, args.epochs + 1):
        z, h_pre, h, p = forward(x_train, a, b, scale, w1, b1, w2, b2)
        train_loss = bce(y_train, p)
        err = p - y_train  # dL/dlogits

        grad_w2 = (h.T @ err) / n + args.l2 * w2
        grad_b2 = float(err.mean())
        dh = np.outer(err, w2)
        dh_pre = dh * (h_pre > 0)
        grad_w1 = (z.T @ dh_pre) / n + args.l2 * w1
        grad_b1 = dh_pre.mean(axis=0)
        dz = dh_pre @ w1.T

        m = x_train @ a
        dn = dz * scale
        grad_b = (m.T @ dn) / n + args.l2 * b
        dm = dn @ b.T
        grad_a = (x_train.T @ dm) / n + args.l2 * a

        a -= args.lr * grad_a
        b -= args.lr * grad_b
        w1 -= args.lr * grad_w1
        b1 -= args.lr * grad_b1
        w2 -= args.lr * grad_w2
        b2 -= args.lr * grad_b2

        _, _, _, pt = forward(x_test, a, b, scale, w1, b1, w2, b2)
        test_loss = bce(y_test, pt)
        history.append({"epoch": epoch, "train_loss": train_loss, "test_loss": test_loss})
        if epoch % 25 == 0 or epoch == 1 or epoch == args.epochs:
            print(f"      epoch {epoch}/{args.epochs} train_loss={train_loss:.4f} test_loss={test_loss:.4f}")

    print("[7/7] Writing outputs...")
    _, _, _, p_train = forward(x_train, a, b, scale, w1, b1, w2, b2)
    _, _, _, p_test = forward(x_test, a, b, scale, w1, b1, w2, b2)
    y_pred_test = (p_test >= 0.5).astype(int)

    metrics = {
        "model": "lora_ffn",
        "n_samples": int(len(y)),
        "n_train": int(len(y_train)),
        "n_test": int(len(y_test)),
        "n_pos": int((y == 1).sum()),
        "n_neg": int((y == 0).sum()),
        "train_accuracy": float((((p_train >= 0.5).astype(int)) == y_train).mean()),
        "test_accuracy": float((y_pred_test == y_test).mean()),
        "test_auroc": roc_auc(y_test, p_test),
        "test_auprc": pr_auc(y_test, p_test),
        "runtime_seconds": float(time.perf_counter() - t0),
    }
    hparams["runtime_seconds"] = metrics["runtime_seconds"]

    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    (out_dir / "hyperparameters.json").write_text(json.dumps(hparams, indent=2, default=str))

    pred_df = gene_df.loc[test_mask, ["gene_symbol", "uniprot_accession", "label", "integrated_score"]].copy()
    pred_df["y_true"] = y_test.astype(int)
    pred_df["y_prob"] = p_test
    pred_df["y_pred"] = y_pred_test
    pred_df.to_csv(out_dir / "test_predictions.csv", index=False)

    hist_df = pd.DataFrame(history)
    hist_df.to_csv(out_dir / "loss_history.csv", index=False)

    plt.figure(figsize=(8, 5))
    plt.plot(hist_df["epoch"], hist_df["train_loss"], label="train_loss")
    plt.plot(hist_df["epoch"], hist_df["test_loss"], label="test_loss")
    plt.xlabel("Epoch")
    plt.ylabel("BCE Loss")
    plt.title("LoRA Predictor Loss Curves")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "loss_curve.png", dpi=200)
    plt.close()

    plt.figure(figsize=(6, 5))
    plt.hist(p_test[y_test == 1], bins=25, alpha=0.6, label="AD")
    plt.hist(p_test[y_test == 0], bins=25, alpha=0.6, label="control")
    plt.xlabel("Predicted Probability")
    plt.ylabel("Count")
    plt.title("Test Prediction Distribution")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "test_prob_hist.png", dpi=200)
    plt.close()

    print(json.dumps(metrics, indent=2))
    print(f"Wrote: {out_dir / 'metrics.json'}")
    print(f"Wrote: {out_dir / 'hyperparameters.json'}")
    print(f"Wrote: {out_dir / 'labels_used.csv'}")
    print(f"Wrote: {out_dir / 'gene_to_uniprot_mapping.csv'}")
    print(f"Wrote: {out_dir / 'test_predictions.csv'}")
    print(f"Wrote: {out_dir / 'loss_history.csv'}")
    print(f"Wrote: {out_dir / 'loss_curve.png'}")
    print(f"Wrote: {out_dir / 'test_prob_hist.png'}")


if __name__ == "__main__":
    main()

