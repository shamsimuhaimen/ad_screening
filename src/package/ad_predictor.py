"""Run the LOI baseline AD predictor on DrugCLIP embeddings."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from attrs import asdict, frozen
from sklearn.model_selection import StratifiedKFold


PATH_ARG_KEYS = {
    "data_dir",
    "ad_genes_path",
    "hgnc_mapping_path",
    "embeddings_npy",
    "names_npy",
    "output_dir",
}


@frozen
class ADPredictorConfig:
    data_dir: Path
    ad_genes_path: Path
    hgnc_mapping_path: Path
    embeddings_npy: Path
    names_npy: Path
    output_dir: Path
    controls_per_ad: int
    min_global_expression: float
    num_folds: int
    seed: int
    epochs: int
    lr: float
    l2: float
    hidden_dim: int
    loss_selection: str
    label_shuffle: bool
    random_embeddings: bool
    debug_plots: bool = False


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


def load_ad_genes(args: ADPredictorConfig) -> set[str]:
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


def build_candidate_gene_table(args: ADPredictorConfig) -> pd.DataFrame:
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

    merged["label"] = np.where(merged["gene_symbol"].isin(ad_present), "AD", "control")
    merged["y"] = np.where(merged["label"] == "AD", 1.0, 0.0)

    integrated_parts = []
    for expr_col in expr_cols:
        other_cols = [c for c in expr_cols if c != expr_col]
        region_values = merged[expr_col]
        specificity = merged[expr_col] - merged[other_cols].mean(axis=1)
        integrated = 0.7 * zscore(region_values) + 0.3 * zscore(specificity)
        integrated_parts.append(integrated)
    merged["integrated_score"] = np.mean(np.vstack(integrated_parts), axis=0)

    return merged[["gene_id", "gene_symbol", "label", "y", "integrated_score", "global_mean"]].drop_duplicates(
        subset=["gene_symbol"]
    )


def select_fold_examples(features: pd.DataFrame, controls_per_ad: int) -> pd.DataFrame:
    ad_symbols = set(features.loc[features["y"] == 1.0, "gene_symbol"])
    if not ad_symbols:
        raise ValueError("Fold split does not contain any AD genes.")
    control_symbols = select_matched_controls(features, ad_symbols, controls_per_ad)
    selected_symbols = ad_symbols.union(control_symbols)
    selected = features[features["gene_symbol"].isin(selected_symbols)].copy()
    if selected["y"].nunique() < 2:
        raise ValueError("Fold split does not contain both classes after matched control selection.")
    return selected


def ffn_forward(x: np.ndarray, model: object) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run inference through the trained PyTorch FFN and return hidden activations plus probabilities."""
    try:
        import torch
    except ImportError as exc:
        raise ImportError("ffn_forward now requires PyTorch to be installed.") from exc

    x_tensor = torch.tensor(x, dtype=torch.float64)
    with torch.no_grad():
        if model.hidden is None:
            logits = model.output(x_tensor).reshape(-1)
            probs = torch.sigmoid(logits)
            return logits[:, None].cpu().numpy(), x_tensor.cpu().numpy(), probs.cpu().numpy()
        h_pre = model.hidden(x_tensor)
        h = torch.relu(h_pre)
        logits = model.output(h).reshape(-1)
        probs = torch.sigmoid(logits)
        return h_pre.cpu().numpy(), h.cpu().numpy(), probs.cpu().numpy()


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
) -> tuple[object, list[dict[str, float]]]:
    try:
        import torch
        import torch.nn.functional as F
    except ImportError as exc:
        raise ImportError("train_small_ffn now requires PyTorch to be installed.") from exc

    n, d = x_train.shape
    if hidden_dim < 0:
        raise ValueError("hidden_dim must be >= 0")
    torch.manual_seed(seed)

    x_train_t = torch.tensor(x_train, dtype=torch.float64)
    y_train_t = torch.tensor(y_train, dtype=torch.float64)
    x_test_t = torch.tensor(x_test, dtype=torch.float64)
    y_test_t = torch.tensor(y_test, dtype=torch.float64)

    class SmallFFN(torch.nn.Module):
        def __init__(self, input_dim: int, hidden_size: int) -> None:
            super().__init__()
            self.hidden = None if hidden_size == 0 else torch.nn.Linear(input_dim, hidden_size, dtype=torch.float64)
            out_in = input_dim if hidden_size == 0 else hidden_size
            self.output = torch.nn.Linear(out_in, 1, dtype=torch.float64)

            if self.hidden is None:
                torch.nn.init.normal_(self.output.weight, mean=0.0, std=1.0 / np.sqrt(input_dim))
            else:
                torch.nn.init.normal_(self.hidden.weight, mean=0.0, std=1.0 / np.sqrt(input_dim))
                torch.nn.init.zeros_(self.hidden.bias)
                torch.nn.init.normal_(self.output.weight, mean=0.0, std=1.0 / np.sqrt(hidden_size))
            torch.nn.init.zeros_(self.output.bias)

        def forward(self, x_tensor: torch.Tensor) -> torch.Tensor:
            if self.hidden is None:
                return self.output(x_tensor).reshape(-1)
            return self.output(torch.relu(self.hidden(x_tensor))).reshape(-1)

    model = SmallFFN(d, hidden_dim)

    history: list[dict[str, float]] = []
    pos_weight, neg_weight = resolve_class_weights(y_train, loss_selection)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=l2)

    def weighted_loss(x_tensor: torch.Tensor, y_tensor: torch.Tensor) -> torch.Tensor:
        logits = model(x_tensor)
        base = F.binary_cross_entropy_with_logits(logits, y_tensor, reduction="none")
        weights = torch.where(
            y_tensor == 1.0,
            torch.tensor(pos_weight, dtype=torch.float64),
            torch.tensor(neg_weight, dtype=torch.float64),
        )
        return (base * weights).mean()

    for epoch in range(1, epochs + 1):
        optimizer.zero_grad()
        train_loss = weighted_loss(x_train_t, y_train_t)
        train_loss.backward()
        optimizer.step()

        with torch.no_grad():
            test_loss = weighted_loss(x_test_t, y_test_t)

        history.append(
            {
                "epoch": float(epoch),
                "train_loss": float(train_loss.detach().cpu().item()),
                "test_loss": float(test_loss.detach().cpu().item()),
            }
        )

    return model, history


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
    pos_count = int((y_int == 1).sum())
    neg_count = int((y_int == 0).sum())
    if pos_count < num_folds or neg_count < num_folds:
        raise ValueError(f"Not enough samples per class for {num_folds}-fold CV: pos={pos_count} neg={neg_count}.")

    splitter = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=seed)
    x_dummy = np.zeros((len(y_int), 1), dtype=np.float64)
    return [(train_idx, test_idx) for train_idx, test_idx in splitter.split(x_dummy, y_int)]


def run_single(config: ADPredictorConfig, output_dir: Path) -> dict[str, float | int | str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    if config.label_shuffle and config.random_embeddings:
        raise ValueError("`label_shuffle` and `random_embeddings` cannot both be enabled.")
    config_dict = asdict(config)
    config_dict["output_dir"] = str(output_dir)
    config_dict["implementation"] = "package.ad_predictor.run_ad_predictor"
    with open(output_dir / "config.json", "w") as f:
        json.dump(config_dict, f, indent=2, default=str)

    print("[1/7] Building candidate gene table from expression data...")
    labels = build_candidate_gene_table(config)
    labels.to_csv(output_dir / "labels_used.csv", index=False)
    print(f"      candidate genes built: {len(labels)} genes")

    print("[2/7] Loading embeddings and names...")
    names = np.load(config.names_npy, allow_pickle=True)
    embeddings = np.load(config.embeddings_npy)
    if len(names) != embeddings.shape[0]:
        raise ValueError("names and embeddings row counts do not match.")
    print(f"      embeddings shape: {embeddings.shape}")

    print("[3/7] Extracting UniProt accessions from embedding names...")
    acc = pd.Series(names).astype(str).str.extract(r"AF-([A-Z0-9]+)-F1", expand=False).str.upper()
    names_df = pd.DataFrame({"uniprot_accession": acc, "row_idx": np.arange(len(names))}).dropna(
        subset=["uniprot_accession"]
    )
    print(f"      extracted accessions: {len(names_df)} rows")

    print("[4/7] Mapping gene symbols to UniProt...")
    mapping_df = build_gene_to_uniprot_map(labels["gene_symbol"].tolist(), config.hgnc_mapping_path)
    mapping_df.to_csv(output_dir / "gene_to_uniprot_mapping.csv", index=False)
    n_mapped = int(mapping_df["uniprot_accession"].notna().sum())
    print(f"Mapped gene symbols to UniProt: {n_mapped}/{len(mapping_df)}")

    labels_with_acc = labels.merge(mapping_df, on="gene_symbol", how="left")
    merged = labels_with_acc.merge(names_df, on="uniprot_accession", how="inner")
    if merged.empty:
        raise ValueError("No overlapping genes between labels and embedding names.")
    print(f"      overlap after mapping/join: {len(merged)} rows ({merged['gene_symbol'].nunique()} unique genes)")

    row_lists = merged.groupby("gene_symbol")["row_idx"].apply(list)
    gene_df = merged.groupby("gene_symbol", as_index=False).agg(
        {
            "label": "first",
            "y": "first",
            "integrated_score": "mean",
            "global_mean": "mean",
            "uniprot_accession": "first",
        }
    )
    gene_df["row_indices"] = gene_df["gene_symbol"].map(row_lists)
    x = np.vstack([embeddings[np.asarray(indices, dtype=int)].mean(axis=0) for indices in gene_df["row_indices"]])
    y = gene_df["y"].to_numpy(dtype=np.float64)
    print(f"      aggregated to one embedding per gene: {len(gene_df)} genes")

    print("[5/7] Preparing stratified k-fold CV...")
    print(
        "      training modifiers: "
        f"label_shuffle={config.label_shuffle} random_embeddings={config.random_embeddings}"
    )

    rng = np.random.default_rng(config.seed)
    if config.random_embeddings:
        x = rng.normal(loc=0.0, scale=1.0, size=x.shape).astype(np.float64)
    gene_table = gene_df[["gene_symbol", "y"]].copy()
    folds = stratified_kfold_indices(y=y, num_folds=config.num_folds, seed=config.seed)
    print(f"      unique genes: total={gene_table.shape[0]} folds={config.num_folds}")
    print(f"      class counts (genes): pos={int((y==1).sum())} neg={int((y==0).sum())}")
    print("[6/7] Training small FFN prediction head across folds...")
    if config.hidden_dim == 0:
        print("      architecture: 768 -> 1 (linear probe)")
    else:
        print(f"      architecture: 768 -> {config.hidden_dim} -> 1")

    fold_rows: list[dict[str, float | int | str]] = []
    all_loss_history: list[pd.DataFrame] = []
    train_acc_values: list[float] = []
    train_sizes: list[int] = []
    test_sizes: list[int] = []
    pos_weights: list[float] = []
    neg_weights: list[float] = []
    test_prediction_rows: list[pd.DataFrame] = []

    for fold_idx, (train_idx, test_idx) in enumerate(folds):
        train_genes = select_fold_examples(gene_df.iloc[train_idx].copy(), config.controls_per_ad)
        test_genes = select_fold_examples(gene_df.iloc[test_idx].copy(), config.controls_per_ad)

        x_train = np.vstack([x[int(idx)] for idx in train_genes.index])
        y_train = train_genes["y"].to_numpy(dtype=np.float64)
        x_test = np.vstack([x[int(idx)] for idx in test_genes.index])
        y_test = test_genes["y"].to_numpy(dtype=np.float64)

        if config.label_shuffle:
            y_train = rng.permutation(y_train)

        mean = x_train.mean(axis=0)
        std = x_train.std(axis=0)
        std[std == 0] = 1.0
        x_train = (x_train - mean) / std
        x_test = (x_test - mean) / std

        pos_weight, neg_weight = resolve_class_weights(y_train, config.loss_selection)
        print(
            f"      fold {fold_idx + 1}/{config.num_folds}: "
            f"train={len(train_idx)} test={len(test_idx)} "
            f"pos_weight={pos_weight:.4f} neg_weight={neg_weight:.4f}"
        )
        model, loss_history = train_small_ffn(
            x_train=x_train,
            y_train=y_train,
            x_test=x_test,
            y_test=y_test,
            hidden_dim=config.hidden_dim,
            epochs=config.epochs,
            lr=config.lr,
            l2=config.l2,
            loss_selection=config.loss_selection,
            seed=config.seed * 10_000 + fold_idx,
        )

        _, _, train_probs = ffn_forward(x_train, model)
        _, _, test_probs = ffn_forward(x_test, model)
        test_pred = (test_probs >= 0.5).astype(int)
        train_acc = float((((train_probs >= 0.5).astype(int)) == y_train).mean())
        test_acc = float((test_pred == y_test).mean())
        test_auroc = roc_auc(y_test, test_probs)
        test_auprc = pr_auc(y_test, test_probs)

        fold_rows.append(
            {
                "fold": int(fold_idx),
                "n_train": int(len(train_genes)),
                "n_test": int(len(test_genes)),
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
        train_sizes.append(int(len(train_genes)))
        test_sizes.append(int(len(test_genes)))
        pos_weights.append(pos_weight)
        neg_weights.append(neg_weight)
        fold_pred_df = test_genes[["gene_symbol", "uniprot_accession", "label", "integrated_score"]].copy()
        fold_pred_df["fold"] = int(fold_idx)
        fold_pred_df["y_true"] = y_test.astype(int)
        fold_pred_df["y_prob"] = test_probs
        fold_pred_df["y_pred"] = test_pred
        test_prediction_rows.append(fold_pred_df)

    print("[7/7] Evaluating pooled out-of-fold predictions and writing outputs...")
    pred_df = pd.concat(test_prediction_rows, ignore_index=True)
    y_true = pred_df["y_true"].to_numpy(dtype=int)
    y_prob = pred_df["y_prob"].to_numpy(dtype=np.float64)
    y_pred = pred_df["y_pred"].to_numpy(dtype=int)

    metrics = {
        "label_shuffle": bool(config.label_shuffle),
        "random_embeddings": bool(config.random_embeddings),
        "loss_selection": config.loss_selection,
        "num_folds": int(config.num_folds),
        "mean_pos_weight": float(np.mean(pos_weights)),
        "mean_neg_weight": float(np.mean(neg_weights)),
        "n_samples": int(len(y_true)),
        "n_pos": int((y_true == 1).sum()),
        "n_neg": int((y_true == 0).sum()),
        "mean_train_size": float(np.mean(train_sizes)),
        "mean_test_size": float(np.mean(test_sizes)),
        "train_accuracy_mean": float(np.mean(train_acc_values)),
        "train_accuracy_std": float(np.std(train_acc_values, ddof=0)),
        "test_accuracy": float((y_pred == y_true).mean()),
        "test_auroc": roc_auc(y_true, y_prob),
        "test_auprc": pr_auc(y_true, y_prob),
        "test_prevalence": float((y_true == 1).mean()),
    }

    with open(output_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    pd.DataFrame(fold_rows).to_csv(output_dir / "fold_metrics.csv", index=False)

    pred_df.to_csv(output_dir / "test_predictions.csv", index=False)

    roc_fpr, roc_tpr = roc_curve_points(y_true, y_prob)
    pd.DataFrame({"fpr": roc_fpr, "tpr": roc_tpr}).to_csv(output_dir / "roc_curve.csv", index=False)

    pr_recall, pr_precision = pr_curve_points(y_true, y_prob)
    pd.DataFrame({"recall": pr_recall, "precision": pr_precision}).to_csv(output_dir / "pr_curve.csv", index=False)

    loss_df = pd.concat(all_loss_history, ignore_index=True)
    loss_df.to_csv(output_dir / "loss_history.csv", index=False)

    print(json.dumps(metrics, indent=2))
    print(f"Wrote: {output_dir / 'config.json'}")
    print(f"Wrote: {output_dir / 'labels_used.csv'}")
    print(f"Wrote: {output_dir / 'gene_to_uniprot_mapping.csv'}")
    print(f"Wrote: {output_dir / 'metrics.json'}")
    print(f"Wrote: {output_dir / 'fold_metrics.csv'}")
    print(f"Wrote: {output_dir / 'test_predictions.csv'}")
    print(f"Wrote: {output_dir / 'roc_curve.csv'}")
    print(f"Wrote: {output_dir / 'pr_curve.csv'}")
    print(f"Wrote: {output_dir / 'loss_history.csv'}")
    return metrics


def _config_from_payload(payload: dict[str, object]) -> ADPredictorConfig:
    """Coerce forwarded experiment-config values into the frozen config used by the implementation."""
    converted = dict(payload)
    for key in PATH_ARG_KEYS:
        value = converted.get(key)
        if value is not None and not isinstance(value, Path):
            converted[key] = Path(value)
    return ADPredictorConfig(**converted)


def run_ad_predictor(
    experiment_config: dict[str, object],
    ablation_name: str,
    seed: int,
    output_dir: Path,
) -> dict[str, float | int | str]:
    """Run one concrete AD predictor job selected from an experiment YAML payload."""
    ablations = experiment_config.get("ablations", [])
    ablation_cfg = next((cfg for cfg in ablations if str(cfg.get("name")) == ablation_name), None)
    if ablation_cfg is None:
        raise ValueError(f"Ablation `{ablation_name}` not found in experiment config.")

    payload = dict(experiment_config.get("defaults", {}))
    payload.update({key: value for key, value in ablation_cfg.items() if key not in {"name", "description"}})
    payload["seed"] = seed
    payload["output_dir"] = output_dir
    args = _config_from_payload(payload)
    return run_single(args, output_dir=output_dir)
