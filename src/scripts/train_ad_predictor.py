#!/usr/bin/env python3
"""Train the LOI baseline AD prediction head on DrugCLIP embeddings.

This script operationalizes the LOI's central claim test:
- construct AD-vs-matched-control labels using expression-derived biology,
- align those labels to DrugCLIP protein embeddings,
- train a lightweight prediction head,
- output metrics and predictions for ablation/control analysis.

It is the minimal reproducible baseline for showing whether pretrained
representations contain AD-relevant signal.
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
    p.add_argument("--data-dir", type=Path, default=Path("data/raw/bulk_rna_seq_human_brain"))
    p.add_argument("--ad-genes-path", type=Path, default=Path("data/processed/ad_genes.csv"))
    p.add_argument(
        "--hgnc-mapping-path",
        type=Path,
        default=Path("data/download/hgnc_complete_set.txt"),
    )
    p.add_argument(
        "--embeddings-npy",
        type=Path,
        default=Path("data/download/dtwg_af_embeddings.npy"),
    )
    p.add_argument(
        "--names-npy",
        type=Path,
        default=Path("data/download/dtwg_af_names_.npy"),
    )
    p.add_argument("--controls-per-ad", type=int, default=2)
    p.add_argument("--min-global-expression", type=float, default=1e-6)
    p.add_argument("--test-size", type=float, default=0.2)
    p.add_argument("--split-file", type=Path, default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--lr", type=float, default=0.001)
    p.add_argument("--l2", type=float, default=1e-4)
    p.add_argument("--hidden-dim", type=int, default=64)
    p.add_argument(
        "--ablation",
        type=str,
        choices=["embedding", "random_embedding", "label_shuffle"],
        default="embedding",
    )
    return p.parse_args()


def zscore(s: pd.Series) -> pd.Series:
    sd = float(s.std(ddof=0))
    if sd == 0.0 or np.isnan(sd):
        return pd.Series(np.zeros(len(s)), index=s.index)
    return (s - float(s.mean())) / sd


def detect_region(row: pd.Series) -> str | None:
    main = str(row.get("main_structure", "")).strip().upper()
    sub = str(row.get("sub_structure", "")).strip().lower()
    acr = str(row.get("ontology_structure_acronym", "")).strip().upper()

    if "HC" in acr or "HIP" in sub:
        return "hippocampus"
    if "PHG" in acr or "PARAHIP" in sub:
        return "entorhinal"
    if main == "TL" or any(tag in acr for tag in ("MTG", "STG", "ITG", "FUG", "TL")):
        return "temporal"
    return None


def load_ad_genes(args: argparse.Namespace) -> set[str]:
    df = pd.read_csv(args.ad_genes_path)
    if "gene_symbol" not in df.columns:
        raise ValueError("data/processed/ad_genes.csv must contain column `gene_symbol`.")
    return {str(g).strip().upper() for g in df["gene_symbol"].dropna().tolist() if str(g).strip()}


def _first_accession(value: object) -> str | None:
    if pd.isna(value):
        return None
    text = str(value).strip()
    if not text:
        return None
    for sep in ["|", ";", ","]:
        text = text.replace(sep, " ")
    token = text.split()[0].strip().upper()
    return token or None


def build_gene_to_uniprot_map(gene_symbols: list[str], hgnc_mapping_path: Path) -> pd.DataFrame:
    if not hgnc_mapping_path.exists():
        fallback_paths = [
            Path("data/download/hgnc_complete_set.txt"),
            Path("data/raw/gene_symbol_to_uniprot_human/hgnc_complete_set.txt"),
        ]
        found = next((p for p in fallback_paths if p.exists()), None)
        if found is None:
            raise FileNotFoundError(
                "HGNC mapping file not found. Expected one of: "
                + ", ".join(str(p) for p in [hgnc_mapping_path, *fallback_paths])
            )
        hgnc_mapping_path = found

    hgnc = pd.read_csv(hgnc_mapping_path, sep="\t", low_memory=False)
    if "symbol" not in hgnc.columns or "uniprot_ids" not in hgnc.columns:
        raise ValueError("HGNC mapping file must contain columns `symbol` and `uniprot_ids`.")
    hgnc_map = hgnc[["symbol", "uniprot_ids"]].copy()
    hgnc_map["gene_symbol"] = hgnc_map["symbol"].astype(str).str.upper().str.strip()
    hgnc_map["uniprot_accession"] = hgnc_map["uniprot_ids"].apply(_first_accession)
    hgnc_map = hgnc_map[["gene_symbol", "uniprot_accession"]].drop_duplicates(subset=["gene_symbol"], keep="first")

    requested = pd.DataFrame({"gene_symbol": sorted(set(gene_symbols))})
    return requested.merge(hgnc_map, on="gene_symbol", how="left")


def load_expression_matrix(data_dir: Path, sample_names: list[str]) -> pd.DataFrame:
    expr = pd.read_csv(data_dir / "RNAseqTPM.csv", header=None, low_memory=False)
    expected_cols = len(sample_names) + 1
    if expr.shape[1] != expected_cols:
        raise ValueError(f"Unexpected RNAseqTPM shape {expr.shape}. Expected {expected_cols} columns.")
    expr.columns = ["gene_symbol", *sample_names]
    return expr


def select_matched_controls(features: pd.DataFrame, ad_symbols: set[str], controls_per_ad: int) -> set[str]:
    candidates = features[~features["gene_symbol"].isin(ad_symbols)].copy()
    ad_df = features[features["gene_symbol"].isin(ad_symbols)].copy()

    selected: set[str] = set()
    for _, row in ad_df.iterrows():
        candidates["dist"] = (candidates["global_mean"] - row["global_mean"]).abs()
        pool = candidates[~candidates["gene_symbol"].isin(selected)].nsmallest(controls_per_ad, "dist")
        selected.update(pool["gene_symbol"].tolist())
    return selected


def build_label_table(args: argparse.Namespace) -> pd.DataFrame:
    data_dir = args.data_dir
    genes = pd.read_csv(data_dir / "Genes.csv", usecols=["gene_symbol", "gene_id"])
    sample_annot = pd.read_csv(data_dir / "SampleAnnot.csv")
    sample_annot["region"] = sample_annot.apply(detect_region, axis=1)
    sample_annot = sample_annot[~sample_annot["region"].isna()].copy()
    if sample_annot.empty:
        raise ValueError("No vulnerable-region samples found after region mapping.")

    sample_names_all = pd.read_csv(data_dir / "SampleAnnot.csv")["RNAseq_sample_name"].tolist()
    expr = load_expression_matrix(data_dir, sample_names_all)

    vulnerable_samples = sample_annot["RNAseq_sample_name"].tolist()
    expr_small = expr[["gene_symbol", *vulnerable_samples]].copy()
    merged = genes.merge(expr_small, on="gene_symbol", how="inner")
    merged["gene_symbol"] = merged["gene_symbol"].astype(str).str.upper().str.strip()

    region_cols = sample_annot.groupby("region")["RNAseq_sample_name"].apply(list).to_dict()
    for region, cols in region_cols.items():
        merged[f"expr_{region}"] = merged[cols].mean(axis=1)

    expr_cols = [f"expr_{r}" for r in region_cols.keys()]
    merged["global_mean"] = merged[expr_cols].mean(axis=1)
    merged = merged[merged["global_mean"] >= args.min_global_expression].copy()

    ad_symbols = load_ad_genes(args)
    ad_present = ad_symbols.intersection(set(merged["gene_symbol"]))
    if not ad_present:
        raise ValueError("None of the AD genes were found in the expression data.")

    control_symbols = select_matched_controls(merged, ad_present, args.controls_per_ad)
    selected = merged[merged["gene_symbol"].isin(ad_present) | merged["gene_symbol"].isin(control_symbols)].copy()
    selected["label"] = np.where(selected["gene_symbol"].isin(ad_present), "AD", "control")
    selected["y"] = np.where(selected["label"] == "AD", 1.0, 0.0)

    integrated_parts = []
    for expr_col in expr_cols:
        other_cols = [c for c in expr_cols if c != expr_col]
        region_values = selected[expr_col]
        specificity = selected[expr_col] - selected[other_cols].mean(axis=1)
        integrated = 0.7 * zscore(region_values) + 0.3 * zscore(specificity)
        integrated_parts.append(integrated)
    selected["integrated_score"] = np.mean(np.vstack(integrated_parts), axis=0)

    return selected[["gene_id", "gene_symbol", "label", "y", "integrated_score"]].drop_duplicates(
        subset=["gene_symbol"]
    )


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -40, 40)))


def relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(x, 0.0)


def ffn_forward(
    x: np.ndarray,
    w1: np.ndarray,
    b1: np.ndarray,
    w2: np.ndarray,
    b2: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    h_pre = x @ w1 + b1
    h = relu(h_pre)
    logits = (h @ w2).reshape(-1) + b2
    probs = sigmoid(logits)
    return h_pre, h, probs


def bce_loss_ffn(
    x: np.ndarray,
    y: np.ndarray,
    w1: np.ndarray,
    b1: np.ndarray,
    w2: np.ndarray,
    b2: float,
    l2: float,
) -> float:
    _, _, probs = ffn_forward(x, w1, b1, w2, b2)
    eps = 1e-12
    ce = -(y * np.log(probs + eps) + (1.0 - y) * np.log(1.0 - probs + eps)).mean()
    reg = 0.5 * l2 * float(np.sum(w1 * w1) + np.sum(w2 * w2))
    return float(ce + reg)


def train_small_ffn(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    hidden_dim: int,
    epochs: int,
    lr: float,
    l2: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, list[dict[str, float]]]:
    n, d = x_train.shape
    rng = np.random.default_rng(42)
    w1 = rng.normal(0.0, 1.0 / np.sqrt(d), size=(d, hidden_dim)).astype(np.float64)
    b1 = np.zeros(hidden_dim, dtype=np.float64)
    w2 = rng.normal(0.0, 1.0 / np.sqrt(hidden_dim), size=(hidden_dim, 1)).astype(np.float64)
    b2 = 0.0
    history: list[dict[str, float]] = []

    for epoch in range(1, epochs + 1):
        h_pre, h, probs = ffn_forward(x_train, w1, b1, w2, b2)
        err = probs - y_train

        # Backpropagation through 1-hidden-layer FFN.
        grad_w2 = (h.T @ err.reshape(-1, 1)) / n + l2 * w2
        grad_b2 = float(err.mean())
        grad_h = err.reshape(-1, 1) @ w2.T
        grad_h_pre = grad_h * (h_pre > 0.0)
        grad_w1 = (x_train.T @ grad_h_pre) / n + l2 * w1
        grad_b1 = grad_h_pre.mean(axis=0)

        w2 -= lr * grad_w2
        b2 -= lr * grad_b2
        w1 -= lr * grad_w1
        b1 -= lr * grad_b1

        history.append(
            {
                "epoch": float(epoch),
                "train_loss": bce_loss_ffn(x_train, y_train, w1, b1, w2, b2, l2),
                "test_loss": bce_loss_ffn(x_test, y_test, w1, b1, w2, b2, l2),
            }
        )

    return w1, b1, w2, b2, history


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


def main() -> None:
    args = parse_args()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(f"results/ad_predictor_{timestamp}_{args.ablation}")
    output_dir.mkdir(parents=True, exist_ok=True)
    hparams = vars(args).copy()
    hparams["output_dir"] = str(output_dir)
    hparams["timestamp"] = timestamp
    with open(output_dir / "hyperparameters.json", "w") as f:
        json.dump(hparams, f, indent=2, default=str)

    print("[1/7] Building AD/control label table from expression data...")
    labels = build_label_table(args)
    labels.to_csv(output_dir / "labels_used.csv", index=False)
    print(f"      labels built: {len(labels)} genes")

    print("[2/7] Loading embeddings and names...")
    names = np.load(args.names_npy, allow_pickle=True)
    embeddings = np.load(args.embeddings_npy)
    if len(names) != embeddings.shape[0]:
        raise ValueError("names and embeddings row counts do not match.")
    print(f"      embeddings shape: {embeddings.shape}")

    print("[3/7] Extracting UniProt accessions from embedding names...")
    # Embedding names are AF paths like ".../AF-Q8NH85-F1-model_v4...".
    acc = pd.Series(names).astype(str).str.extract(r"AF-([A-Z0-9]+)-F1", expand=False).str.upper()
    names_df = pd.DataFrame(
        {
            "uniprot_accession": acc,
            "row_idx": np.arange(len(names)),
        }
    ).dropna(subset=["uniprot_accession"])
    print(f"      extracted accessions: {len(names_df)} rows")

    print("[4/7] Mapping gene symbols to UniProt...")
    mapping_df = build_gene_to_uniprot_map(labels["gene_symbol"].tolist(), args.hgnc_mapping_path)
    mapping_df.to_csv(output_dir / "gene_to_uniprot_mapping.csv", index=False)
    n_mapped = int(mapping_df["uniprot_accession"].notna().sum())
    print(f"Mapped gene symbols to UniProt: {n_mapped}/{len(mapping_df)}")

    labels_with_acc = labels.merge(mapping_df, on="gene_symbol", how="left")
    merged = labels_with_acc.merge(names_df, on="uniprot_accession", how="inner")
    if merged.empty:
        raise ValueError("No overlapping genes between labels and embedding names.")
    print(f"      overlap after mapping/join: {len(merged)} rows ({merged['gene_symbol'].nunique()} unique genes)")

    # Aggregate multiple embedding rows to one vector per gene.
    row_lists = merged.groupby("gene_symbol")["row_idx"].apply(list)
    gene_df = (
        merged.groupby("gene_symbol", as_index=False)
        .agg(
            {
                "label": "first",
                "y": "first",
                "integrated_score": "mean",
                "uniprot_accession": "first",
            }
        )
    )
    gene_df["row_indices"] = gene_df["gene_symbol"].map(row_lists)
    x = np.vstack(
        [
            embeddings[np.asarray(indices, dtype=int)].mean(axis=0)
            for indices in gene_df["row_indices"]
        ]
    )
    y = gene_df["y"].to_numpy(dtype=np.float64)
    print(f"      aggregated to one embedding per gene: {len(gene_df)} genes")

    print("[5/7] Preparing train/test split...")
    print(f"      ablation mode: {args.ablation}")

    rng = np.random.default_rng(args.seed)
    if args.ablation == "random_embedding":
        x = rng.normal(loc=0.0, scale=1.0, size=x.shape).astype(np.float64)
    # Split by gene (already one row per gene after aggregation).
    gene_table = gene_df[["gene_symbol", "y"]].copy()
    split_file = args.split_file or Path(f"data/processed/ad_predictor_split_seed_{args.seed}.json")
    split_file.parent.mkdir(parents=True, exist_ok=True)
    available_genes = set(gene_table["gene_symbol"].tolist())

    if split_file.exists():
        with open(split_file) as f:
            split_data = json.load(f)
        test_genes = set(split_data.get("test_genes", []))
        test_genes = test_genes.intersection(available_genes)
        print(f"      using existing split: {split_file}")
    else:
        pos_genes = gene_table.loc[gene_table["y"] == 1, "gene_symbol"].to_numpy()
        neg_genes = gene_table.loc[gene_table["y"] == 0, "gene_symbol"].to_numpy()
        rng.shuffle(pos_genes)
        rng.shuffle(neg_genes)

        n_pos_test_genes = max(1, int(round(len(pos_genes) * args.test_size)))
        n_neg_test_genes = max(1, int(round(len(neg_genes) * args.test_size)))
        test_genes = set(pos_genes[:n_pos_test_genes]).union(set(neg_genes[:n_neg_test_genes]))
        with open(split_file, "w") as f:
            json.dump(
                {
                    "seed": args.seed,
                    "test_size": args.test_size,
                    "test_genes": sorted(test_genes),
                },
                f,
                indent=2,
            )
        print(f"      wrote new split: {split_file}")

    test_mask = gene_df["gene_symbol"].isin(test_genes).to_numpy()
    train_mask = ~test_mask
    idx = np.arange(len(y))
    train_idx = idx[train_mask]
    test_idx = idx[test_mask]

    print(
        f"      unique genes: total={gene_table.shape[0]} "
        f"train={gene_table.shape[0] - len(test_genes)} test={len(test_genes)}"
    )
    print(f"      samples: total={len(y)} train={len(train_idx)} test={len(test_idx)}")
    print(f"      class counts (genes): pos={int((y==1).sum())} neg={int((y==0).sum())}")

    x_train = x[train_idx]
    y_train = y[train_idx]
    x_test = x[test_idx]
    y_test = y[test_idx]

    if args.ablation == "label_shuffle":
        y_train = rng.permutation(y_train)

    mean = x_train.mean(axis=0)
    std = x_train.std(axis=0)
    std[std == 0] = 1.0
    x_train = (x_train - mean) / std
    x_test = (x_test - mean) / std

    print("[6/7] Training small FFN prediction head...")
    print(f"      architecture: 768 -> {args.hidden_dim} -> 1")
    w1, b1, w2, b2, loss_history = train_small_ffn(
        x_train=x_train,
        y_train=y_train,
        x_test=x_test,
        y_test=y_test,
        hidden_dim=args.hidden_dim,
        epochs=args.epochs,
        lr=args.lr,
        l2=args.l2,
    )

    print("[7/7] Evaluating and writing outputs...")
    _, _, train_probs = ffn_forward(x_train, w1, b1, w2, b2)
    _, _, test_probs = ffn_forward(x_test, w1, b1, w2, b2)
    test_pred = (test_probs >= 0.5).astype(int)

    metrics = {
        "ablation": args.ablation,
        "n_samples": int(len(y)),
        "n_train": int(len(train_idx)),
        "n_test": int(len(test_idx)),
        "n_pos": int((y == 1).sum()),
        "n_neg": int((y == 0).sum()),
        "train_accuracy": float((((train_probs >= 0.5).astype(int)) == y_train).mean()),
        "test_accuracy": float((test_pred == y_test).mean()),
        "test_auroc": roc_auc(y_test, test_probs),
        "test_auprc": pr_auc(y_test, test_probs),
    }

    with open(output_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    pred_df = gene_df.iloc[test_idx][["gene_symbol", "uniprot_accession", "label", "integrated_score"]].copy()
    pred_df["y_true"] = y_test.astype(int)
    pred_df["y_prob"] = test_probs
    pred_df["y_pred"] = test_pred
    pred_df.to_csv(output_dir / "test_predictions.csv", index=False)

    loss_df = pd.DataFrame(loss_history)
    loss_df.to_csv(output_dir / "loss_history.csv", index=False)
    plt.figure(figsize=(8, 5))
    plt.plot(loss_df["epoch"], loss_df["train_loss"], label="train_loss")
    plt.plot(loss_df["epoch"], loss_df["test_loss"], label="test_loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Curves")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "loss_curve.png", dpi=200)
    plt.close()

    print(json.dumps(metrics, indent=2))
    print(f"Wrote: {output_dir / 'hyperparameters.json'}")
    print(f"Wrote: {output_dir / 'labels_used.csv'}")
    print(f"Wrote: {output_dir / 'gene_to_uniprot_mapping.csv'}")
    print(f"Wrote: {output_dir / 'metrics.json'}")
    print(f"Wrote: {output_dir / 'test_predictions.csv'}")
    print(f"Wrote: {output_dir / 'loss_history.csv'}")
    print(f"Wrote: {output_dir / 'loss_curve.png'}")


if __name__ == "__main__":
    main()
