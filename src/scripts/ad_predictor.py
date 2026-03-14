#!/usr/bin/env python3
"""Run the LOI baseline AD predictor on DrugCLIP embeddings.

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
import yaml


PATH_ARG_KEYS = {
    "config",
    "results_dir",
    "output_dir",
    "data_dir",
    "ad_genes_path",
    "hgnc_mapping_path",
    "embeddings_npy",
    "names_npy",
    "split_file",
}


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, default=None)
    p.add_argument("--results-dir", type=Path, default=Path("results/ad_predictor"))
    p.add_argument("--output-dir", type=Path, default=None)
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
    p.add_argument("--num-folds", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--lr", type=float, default=0.001)
    p.add_argument("--l2", type=float, default=1e-4)
    p.add_argument("--hidden-dim", type=int, default=64)
    p.add_argument(
        "--loss-selection",
        type=str,
        choices=["bce", "weighted_bce"],
        default="bce",
    )
    p.add_argument(
        "--ablation",
        type=str,
        choices=["embedding", "random_embedding", "label_shuffle"],
        default="embedding",
    )
    return p


def parse_args() -> argparse.Namespace:
    return build_parser().parse_args()


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
    if w2.size == 0:
        logits = (x @ w1).reshape(-1) + b2
        probs = sigmoid(logits)
        return logits[:, None], x, probs
    h_pre = x @ w1 + b1
    h = relu(h_pre)
    logits = (h @ w2).reshape(-1) + b2
    probs = sigmoid(logits)
    return h_pre, h, probs


def resolve_class_weights(y: np.ndarray, loss_selection: str) -> tuple[float, float]:
    if loss_selection == "bce":
        return 1.0, 1.0
    if loss_selection != "weighted_bce":
        raise ValueError(f"Unsupported loss selection: {loss_selection}")

    n_pos = int((y == 1).sum())
    n_neg = int((y == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return 1.0, 1.0
    return float(n_neg / n_pos), 1.0


def bce_loss_ffn(
    x: np.ndarray,
    y: np.ndarray,
    w1: np.ndarray,
    b1: np.ndarray,
    w2: np.ndarray,
    b2: float,
    l2: float,
    pos_weight: float = 1.0,
    neg_weight: float = 1.0,
) -> float:
    _, _, probs = ffn_forward(x, w1, b1, w2, b2)
    eps = 1e-12
    ce = -(pos_weight * y * np.log(probs + eps) + neg_weight * (1.0 - y) * np.log(1.0 - probs + eps)).mean()
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
    loss_selection: str,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, list[dict[str, float]]]:
    n, d = x_train.shape
    rng = np.random.default_rng(seed)
    if hidden_dim < 0:
        raise ValueError("hidden_dim must be >= 0")
    if hidden_dim == 0:
        w1 = rng.normal(0.0, 1.0 / np.sqrt(d), size=(d, 1)).astype(np.float64)
        b1 = np.zeros(1, dtype=np.float64)
        w2 = np.zeros((0, 1), dtype=np.float64)
    else:
        w1 = rng.normal(0.0, 1.0 / np.sqrt(d), size=(d, hidden_dim)).astype(np.float64)
        b1 = np.zeros(hidden_dim, dtype=np.float64)
        w2 = rng.normal(0.0, 1.0 / np.sqrt(hidden_dim), size=(hidden_dim, 1)).astype(np.float64)
    b2 = 0.0
    history: list[dict[str, float]] = []
    pos_weight, neg_weight = resolve_class_weights(y_train, loss_selection)

    for epoch in range(1, epochs + 1):
        h_pre, h, probs = ffn_forward(x_train, w1, b1, w2, b2)
        sample_weights = np.where(y_train == 1.0, pos_weight, neg_weight)
        err = (probs - y_train) * sample_weights

        if hidden_dim == 0:
            # Linear probe path: x -> sigmoid.
            grad_w1 = (x_train.T @ err.reshape(-1, 1)) / n + l2 * w1
            grad_b1 = np.zeros_like(b1)
            grad_w2 = np.zeros_like(w2)
            grad_b2 = float(err.mean())
        else:
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
                "train_loss": bce_loss_ffn(
                    x_train, y_train, w1, b1, w2, b2, l2, pos_weight=pos_weight, neg_weight=neg_weight
                ),
                "test_loss": bce_loss_ffn(
                    x_test, y_test, w1, b1, w2, b2, l2, pos_weight=pos_weight, neg_weight=neg_weight
                ),
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


def roc_curve_points(y_true: np.ndarray, y_score: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    order = np.argsort(-y_score, kind="mergesort")
    y = y_true[order]
    total_pos = int((y_true == 1).sum())
    total_neg = int((y_true == 0).sum())
    if total_pos == 0 or total_neg == 0:
        return np.asarray([0.0, 1.0]), np.asarray([0.0, 1.0])
    tp = np.cumsum(y == 1)
    fp = np.cumsum(y == 0)
    tpr = tp / total_pos
    fpr = fp / total_neg
    return np.concatenate(([0.0], fpr, [1.0])), np.concatenate(([0.0], tpr, [1.0]))


def pr_curve_points(y_true: np.ndarray, y_score: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    order = np.argsort(-y_score, kind="mergesort")
    y = y_true[order]
    total_pos = int((y_true == 1).sum())
    if total_pos == 0:
        return np.asarray([0.0, 1.0]), np.asarray([1.0, 1.0])
    tp = np.cumsum(y == 1)
    fp = np.cumsum(y == 0)
    recall = tp / total_pos
    precision = tp / np.maximum(tp + fp, 1)
    return np.concatenate(([0.0], recall, [1.0])), np.concatenate(([1.0], precision, [precision[-1]]))


def stratified_kfold_indices(y: np.ndarray, num_folds: int, seed: int) -> list[tuple[np.ndarray, np.ndarray]]:
    if num_folds < 2:
        raise ValueError("num_folds must be >= 2")

    y_int = y.astype(int)
    idx_all = np.arange(len(y_int))
    pos = idx_all[y_int == 1].copy()
    neg = idx_all[y_int == 0].copy()
    if len(pos) < num_folds or len(neg) < num_folds:
        raise ValueError(f"Not enough samples per class for {num_folds}-fold CV: pos={len(pos)} neg={len(neg)}.")

    rng = np.random.default_rng(seed)
    rng.shuffle(pos)
    rng.shuffle(neg)
    pos_folds = np.array_split(pos, num_folds)
    neg_folds = np.array_split(neg, num_folds)

    folds: list[tuple[np.ndarray, np.ndarray]] = []
    for fold_idx in range(num_folds):
        test_idx = np.concatenate([pos_folds[fold_idx], neg_folds[fold_idx]])
        train_idx = np.setdiff1d(idx_all, test_idx)
        folds.append((train_idx, test_idx))
    return folds


def run_single(args: argparse.Namespace, output_dir: Path) -> dict[str, float | int | str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    hparams = vars(args).copy()
    hparams["output_dir"] = str(output_dir)
    hparams["script"] = "src/scripts/ad_predictor.py"
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
    gene_df = merged.groupby("gene_symbol", as_index=False).agg(
        {
            "label": "first",
            "y": "first",
            "integrated_score": "mean",
            "uniprot_accession": "first",
        }
    )
    gene_df["row_indices"] = gene_df["gene_symbol"].map(row_lists)
    x = np.vstack([embeddings[np.asarray(indices, dtype=int)].mean(axis=0) for indices in gene_df["row_indices"]])
    y = gene_df["y"].to_numpy(dtype=np.float64)
    print(f"      aggregated to one embedding per gene: {len(gene_df)} genes")

    print("[5/7] Preparing stratified k-fold CV...")
    print(f"      ablation mode: {args.ablation}")

    rng = np.random.default_rng(args.seed)
    if args.ablation == "random_embedding":
        x = rng.normal(loc=0.0, scale=1.0, size=x.shape).astype(np.float64)
    gene_table = gene_df[["gene_symbol", "y"]].copy()
    folds = stratified_kfold_indices(y=y, num_folds=args.num_folds, seed=args.seed)
    print(f"      unique genes: total={gene_table.shape[0]} folds={args.num_folds}")
    print(f"      class counts (genes): pos={int((y==1).sum())} neg={int((y==0).sum())}")
    print("[6/7] Training small FFN prediction head across folds...")
    if args.hidden_dim == 0:
        print("      architecture: 768 -> 1 (linear probe)")
    else:
        print(f"      architecture: 768 -> {args.hidden_dim} -> 1")

    oof_probs = np.zeros(len(y), dtype=np.float64)
    oof_pred = np.zeros(len(y), dtype=int)
    oof_fold = np.full(len(y), -1, dtype=int)
    fold_rows: list[dict[str, float | int | str]] = []
    all_loss_history: list[pd.DataFrame] = []
    train_acc_values: list[float] = []
    train_sizes: list[int] = []
    test_sizes: list[int] = []
    pos_weights: list[float] = []
    neg_weights: list[float] = []

    for fold_idx, (train_idx, test_idx) in enumerate(folds):
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

        pos_weight, neg_weight = resolve_class_weights(y_train, args.loss_selection)
        print(
            f"      fold {fold_idx + 1}/{args.num_folds}: "
            f"train={len(train_idx)} test={len(test_idx)} "
            f"pos_weight={pos_weight:.4f} neg_weight={neg_weight:.4f}"
        )
        w1, b1, w2, b2, loss_history = train_small_ffn(
            x_train=x_train,
            y_train=y_train,
            x_test=x_test,
            y_test=y_test,
            hidden_dim=args.hidden_dim,
            epochs=args.epochs,
            lr=args.lr,
            l2=args.l2,
            loss_selection=args.loss_selection,
            seed=args.seed * 10_000 + fold_idx,
        )

        _, _, train_probs = ffn_forward(x_train, w1, b1, w2, b2)
        _, _, test_probs = ffn_forward(x_test, w1, b1, w2, b2)
        test_pred = (test_probs >= 0.5).astype(int)
        train_acc = float((((train_probs >= 0.5).astype(int)) == y_train).mean())
        test_acc = float((test_pred == y_test).mean())
        test_auroc = roc_auc(y_test, test_probs)
        test_auprc = pr_auc(y_test, test_probs)

        oof_probs[test_idx] = test_probs
        oof_pred[test_idx] = test_pred
        oof_fold[test_idx] = fold_idx

        fold_rows.append(
            {
                "fold": int(fold_idx),
                "n_train": int(len(train_idx)),
                "n_test": int(len(test_idx)),
                "n_test_pos": int((y_test == 1).sum()),
                "n_test_neg": int((y_test == 0).sum()),
                "train_accuracy": train_acc,
                "test_accuracy": test_acc,
                "test_auroc": test_auroc,
                "test_auprc": test_auprc,
                "pos_weight": pos_weight,
                "neg_weight": neg_weight,
            }
        )
        loss_df_fold = pd.DataFrame(loss_history)
        loss_df_fold["fold"] = int(fold_idx)
        all_loss_history.append(loss_df_fold)
        train_acc_values.append(train_acc)
        train_sizes.append(int(len(train_idx)))
        test_sizes.append(int(len(test_idx)))
        pos_weights.append(pos_weight)
        neg_weights.append(neg_weight)

    print("[7/7] Evaluating pooled out-of-fold predictions and writing outputs...")
    if np.any(oof_fold < 0):
        raise RuntimeError("Some samples were not assigned an out-of-fold prediction.")

    metrics = {
        "ablation": args.ablation,
        "loss_selection": args.loss_selection,
        "num_folds": int(args.num_folds),
        "mean_pos_weight": float(np.mean(pos_weights)),
        "mean_neg_weight": float(np.mean(neg_weights)),
        "n_samples": int(len(y)),
        "n_pos": int((y == 1).sum()),
        "n_neg": int((y == 0).sum()),
        "mean_train_size": float(np.mean(train_sizes)),
        "mean_test_size": float(np.mean(test_sizes)),
        "train_accuracy_mean": float(np.mean(train_acc_values)),
        "train_accuracy_std": float(np.std(train_acc_values, ddof=0)),
        "test_accuracy": float((oof_pred == y.astype(int)).mean()),
        "test_auroc": roc_auc(y, oof_probs),
        "test_auprc": pr_auc(y, oof_probs),
        "test_prevalence": float((y == 1).mean()),
    }

    with open(output_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    fold_metrics_df = pd.DataFrame(fold_rows)
    fold_metrics_df.to_csv(output_dir / "fold_metrics.csv", index=False)

    pred_df = gene_df[["gene_symbol", "uniprot_accession", "label", "integrated_score"]].copy()
    pred_df["fold"] = oof_fold.astype(int)
    pred_df["y_true"] = y.astype(int)
    pred_df["y_prob"] = oof_probs
    pred_df["y_pred"] = oof_pred
    pred_df.to_csv(output_dir / "test_predictions.csv", index=False)

    roc_fpr, roc_tpr = roc_curve_points(y, oof_probs)
    roc_df = pd.DataFrame({"fpr": roc_fpr, "tpr": roc_tpr})
    roc_df.to_csv(output_dir / "roc_curve.csv", index=False)
    plt.figure(figsize=(6, 6))
    plt.plot(roc_fpr, roc_tpr, color="#F58518", label=f"AUROC = {metrics['test_auroc']:.3f}")
    plt.plot([0.0, 1.0], [0.0, 1.0], linestyle="--", color="#9A9A9A", linewidth=1)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(output_dir / "roc_curve.png", dpi=200)
    plt.close()

    pr_recall, pr_precision = pr_curve_points(y, oof_probs)
    pr_df = pd.DataFrame({"recall": pr_recall, "precision": pr_precision})
    pr_df.to_csv(output_dir / "pr_curve.csv", index=False)
    plt.figure(figsize=(6, 6))
    plt.plot(pr_recall, pr_precision, color="#54A24B", label=f"AUPRC = {metrics['test_auprc']:.3f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.legend(loc="lower left")
    plt.tight_layout()
    plt.savefig(output_dir / "pr_curve.png", dpi=200)
    plt.close()

    loss_df = pd.concat(all_loss_history, ignore_index=True)
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
    print(f"Wrote: {output_dir / 'fold_metrics.csv'}")
    print(f"Wrote: {output_dir / 'test_predictions.csv'}")
    print(f"Wrote: {output_dir / 'roc_curve.csv'}")
    print(f"Wrote: {output_dir / 'roc_curve.png'}")
    print(f"Wrote: {output_dir / 'pr_curve.csv'}")
    print(f"Wrote: {output_dir / 'pr_curve.png'}")
    print(f"Wrote: {output_dir / 'loss_history.csv'}")
    print(f"Wrote: {output_dir / 'loss_curve.png'}")
    return metrics


def _namespace_from_overrides(base: dict[str, object], overrides: dict[str, object]) -> argparse.Namespace:
    payload = dict(base)
    payload.update(overrides)
    for key in PATH_ARG_KEYS:
        value = payload.get(key)
        if value is not None and not isinstance(value, Path):
            payload[key] = Path(value)
    return argparse.Namespace(**payload)


def run_config_experiment(args: argparse.Namespace) -> Path:
    cfg = yaml.safe_load(args.config.read_text())
    expected_script = Path("src/scripts/ad_predictor.py")
    configured_script = Path(str(cfg.get("script", "")))
    if configured_script != expected_script:
        raise ValueError(f"Config script mismatch: expected `{expected_script}`, found `{configured_script}`.")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    root_dir = args.results_dir / f"experiment_runs_{timestamp}"
    root_dir.mkdir(parents=True, exist_ok=True)
    (root_dir / "config.snapshot.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))

    defaults = {k: v for k, v in cfg.get("defaults", {}).items()}
    parser_defaults = vars(build_parser().parse_args([]))
    actual_args = vars(args).copy()
    base_args = dict(parser_defaults)
    base_args.update(defaults)
    for key, value in actual_args.items():
        if key not in parser_defaults or value != parser_defaults[key]:
            base_args[key] = value
    base_args["config"] = args.config
    base_args["results_dir"] = args.results_dir
    base_args["output_dir"] = None

    rows: list[dict[str, object]] = []
    ablations = cfg.get("ablations", [])
    seeds = cfg.get("seeds", [])
    if not ablations or not seeds:
        raise ValueError("Experiment config must define non-empty `ablations` and `seeds`.")

    for ablation_cfg in ablations:
        ablation_name = str(ablation_cfg["name"])
        run_overrides = {
            "ablation": ablation_cfg.get("cli_ablation", defaults.get("ablation", "embedding")),
            "hidden_dim": int(ablation_cfg.get("hidden_dim", defaults.get("hidden_dim", 64))),
            "loss_selection": str(ablation_cfg.get("loss_selection", defaults.get("loss_selection", "bce"))),
        }
        for seed in seeds:
            run_args = _namespace_from_overrides(base_args, {**run_overrides, "seed": int(seed)})
            run_dir = root_dir / "runs" / ablation_name / f"seed_{int(seed)}"
            print(f"[run] ablation={ablation_name} seed={int(seed)} -> {run_dir}")
            metrics = run_single(run_args, output_dir=run_dir)
            row = {
                "ablation_name": ablation_name,
                "cli_ablation": run_args.ablation,
                "seed": int(seed),
                "hidden_dim": int(run_args.hidden_dim),
                "loss_selection": str(run_args.loss_selection),
                **metrics,
            }
            rows.append(row)

    summary_df = pd.DataFrame(rows)
    summary_df.to_csv(root_dir / "summary.csv", index=False)
    summary_by_ablation = (
        summary_df.groupby("ablation_name", as_index=False)
        .agg(
            mean_test_accuracy=("test_accuracy", "mean"),
            mean_test_auroc=("test_auroc", "mean"),
            mean_test_auprc=("test_auprc", "mean"),
            mean_train_accuracy=("train_accuracy", "mean"),
        )
        .sort_values("mean_test_auprc", ascending=False)
    )
    summary_by_ablation.to_csv(root_dir / "summary_by_ablation.csv", index=False)
    print(f"Wrote: {root_dir / 'summary.csv'}")
    print(f"Wrote: {root_dir / 'summary_by_ablation.csv'}")
    return root_dir


def main() -> None:
    args = parse_args()
    if args.config is not None:
        out = run_config_experiment(args)
        print(f"Wrote: {out}")
        return

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = args.output_dir or (
        args.results_dir / f"ad_predictor_{timestamp}_{args.ablation}_{args.loss_selection}"
    )
    run_single(args, output_dir=output_dir)


if __name__ == "__main__":
    main()
