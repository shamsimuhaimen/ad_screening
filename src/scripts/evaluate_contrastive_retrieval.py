#!/usr/bin/env python3
"""Evaluate CLIP-style contrastive retrieval on DrugCLIP gene embeddings.

In LOI context, this asks: given a query embedding row, can we retrieve its
correct gene identity from a candidate gene set by cosine similarity?

Outputs:
- summary.json: Recall@K / MRR / query counts
- per_query_ranks.csv: rank of the true gene per query
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
    p.add_argument("--embeddings-npy", type=Path, default=Path("data/download/dtwg_af_embeddings.npy"))
    p.add_argument("--names-npy", type=Path, default=Path("data/download/dtwg_af_names_.npy"))
    p.add_argument("--max-queries", type=int, default=10000)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output-dir", type=Path, default=None)
    return p.parse_args()


def l2_normalize(x: np.ndarray) -> np.ndarray:
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)


def load_query_table(labels_csv: Path, mapping_csv: Path, names_npy: Path) -> pd.DataFrame:
    labels = pd.read_csv(labels_csv)
    labels = labels[["gene_symbol", "label"]].drop_duplicates(subset=["gene_symbol"], keep="first").copy()
    labels["gene_symbol"] = labels["gene_symbol"].astype(str).str.strip().str.upper()
    labels["y"] = (labels["label"].astype(str).str.upper() == "AD").astype(int)

    mapping = pd.read_csv(mapping_csv)
    mapping = mapping[["gene_symbol", "uniprot_accession"]].dropna().copy()
    mapping["gene_symbol"] = mapping["gene_symbol"].astype(str).str.strip().str.upper()
    mapping["uniprot_accession"] = mapping["uniprot_accession"].astype(str).str.strip().str.upper()

    names = np.load(names_npy, allow_pickle=True)
    accession = pd.Series(names).astype(str).str.extract(r"AF-([A-Z0-9]+)-F1", expand=False).str.upper()
    names_df = pd.DataFrame({"uniprot_accession": accession, "row_idx": np.arange(len(accession), dtype=int)}).dropna()

    merged = labels.merge(mapping, on="gene_symbol", how="left").merge(names_df, on="uniprot_accession", how="inner")
    merged = merged[["gene_symbol", "label", "y", "row_idx"]].drop_duplicates(subset=["gene_symbol", "row_idx"])
    if merged.empty:
        raise ValueError("No overlap between labels/mapping and embedding names.")
    return merged


def compute_retrieval(
    query_df: pd.DataFrame,
    embeddings: np.ndarray,
    batch_size: int,
) -> pd.DataFrame:
    # Candidate space is one centroid per gene.
    genes = sorted(query_df["gene_symbol"].unique().tolist())
    gene_to_idx = {g: i for i, g in enumerate(genes)}
    idx_by_gene = query_df.groupby("gene_symbol")["row_idx"].apply(list).to_dict()

    candidate = np.vstack(
        [embeddings[np.asarray(idx_by_gene[g], dtype=int)].mean(axis=0, dtype=np.float64) for g in genes]
    ).astype(np.float32)
    candidate = l2_normalize(candidate)

    query_rows = query_df["row_idx"].to_numpy(dtype=int)
    qvec = l2_normalize(embeddings[query_rows].astype(np.float32))
    true_idx = query_df["gene_symbol"].map(gene_to_idx).to_numpy(dtype=int)

    ranks: list[int] = []
    for start in range(0, len(qvec), batch_size):
        end = min(start + batch_size, len(qvec))
        sim = qvec[start:end] @ candidate.T
        order = np.argsort(-sim, axis=1)
        t = true_idx[start:end]
        pos = np.argmax(order == t[:, None], axis=1) + 1
        ranks.extend(pos.tolist())
        print(f"      retrieval progress: {end}/{len(qvec)} queries")

    out = query_df.reset_index(drop=True).copy()
    out["true_rank"] = np.asarray(ranks, dtype=int)
    return out


def summarize(df: pd.DataFrame) -> dict[str, float]:
    rank = df["true_rank"].to_numpy(dtype=int)
    return {
        "n_queries": int(len(df)),
        "n_candidate_genes": int(df["gene_symbol"].nunique()),
        "recall_at_1": float(np.mean(rank <= 1)),
        "recall_at_5": float(np.mean(rank <= 5)),
        "recall_at_10": float(np.mean(rank <= 10)),
        "mrr": float(np.mean(1.0 / rank)),
        "median_true_rank": float(np.median(rank)),
        "mean_true_rank": float(np.mean(rank)),
    }


def main() -> None:
    args = parse_args()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = args.output_dir or Path(f"results/contrastive_retrieval_{timestamp}")
    out_dir.mkdir(parents=True, exist_ok=True)

    print("[1/4] Building query table...")
    query_df = load_query_table(args.labels_csv, args.mapping_csv, args.names_npy)
    print(f"      query rows before cap: {len(query_df)}")

    if args.max_queries > 0 and len(query_df) > args.max_queries:
        rng = np.random.default_rng(args.seed)
        keep = rng.choice(np.arange(len(query_df)), size=args.max_queries, replace=False)
        query_df = query_df.iloc[np.sort(keep)].reset_index(drop=True)
    print(f"      query rows used: {len(query_df)}")

    print("[2/4] Loading embeddings...")
    embeddings = np.load(args.embeddings_npy)
    print(f"      embeddings shape: {embeddings.shape}")

    print("[3/4] Running retrieval...")
    per_query = compute_retrieval(query_df, embeddings, batch_size=args.batch_size)

    print("[4/4] Writing outputs...")
    summary = summarize(per_query)
    per_query.to_csv(out_dir / "per_query_ranks.csv", index=False)
    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))
    print(f"Wrote: {out_dir / 'summary.json'}")
    print(f"Wrote: {out_dir / 'per_query_ranks.csv'}")


if __name__ == "__main__":
    main()
