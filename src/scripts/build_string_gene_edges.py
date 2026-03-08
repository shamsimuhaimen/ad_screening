#!/usr/bin/env python3
"""Build gene-symbol PPI edges from STRING files for LOI network tests.

The AD PPI analysis script works best when node IDs match our label table
(gene symbols). STRING links are keyed by STRING protein IDs, so this script
maps those IDs to preferred gene symbols and emits a compact edge list:

- gene1
- gene2
- combined_score

This keeps the PPI step reproducible and removes ad-hoc shell/python blocks.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--links-path",
        type=Path,
        default=Path("data/download/9606.protein.links.detailed.v12.0.txt.gz"),
    )
    p.add_argument(
        "--info-path",
        type=Path,
        default=Path("data/download/9606.protein.info.v12.0.txt.gz"),
    )
    p.add_argument(
        "--output-path",
        type=Path,
        default=Path("data/processed/string_gene_edges.tsv"),
    )
    p.add_argument("--chunksize", type=int, default=500_000)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    args.output_path.parent.mkdir(parents=True, exist_ok=True)

    print("[1/3] Loading STRING files...")
    info = pd.read_csv(args.info_path, sep="\t")

    print("[2/3] Mapping STRING protein IDs to preferred gene symbols...")
    mapper = info[["#string_protein_id", "preferred_name"]].dropna().drop_duplicates()
    mapper = mapper.rename(columns={"#string_protein_id": "string_id", "preferred_name": "gene_symbol"})
    mapper["string_id"] = mapper["string_id"].astype(str)
    mapper["gene_symbol"] = mapper["gene_symbol"].astype(str).str.strip().str.upper()
    id_to_gene = dict(zip(mapper["string_id"], mapper["gene_symbol"]))

    total_rows = 0
    kept_rows = 0
    wrote_header = False
    if args.output_path.exists():
        args.output_path.unlink()

    for i, chunk in enumerate(pd.read_csv(args.links_path, sep=r"\s+", chunksize=args.chunksize), start=1):
        total_rows += len(chunk)
        chunk["gene1"] = chunk["protein1"].map(id_to_gene)
        chunk["gene2"] = chunk["protein2"].map(id_to_gene)
        chunk = chunk.dropna(subset=["gene1", "gene2"])
        chunk = chunk[chunk["gene1"] != chunk["gene2"]]
        if chunk.empty:
            print(f"      chunk {i}: processed={total_rows:,} kept={kept_rows:,}")
            continue

        # Canonicalize undirected edge direction per row.
        g1 = chunk["gene1"].to_numpy()
        g2 = chunk["gene2"].to_numpy()
        lo = np.where(g1 <= g2, g1, g2)
        hi = np.where(g1 <= g2, g2, g1)
        out = pd.DataFrame(
            {
                "gene1": lo,
                "gene2": hi,
                "combined_score": chunk["combined_score"].to_numpy(),
            }
        )

        out.to_csv(args.output_path, sep="\t", index=False, mode="a", header=not wrote_header)
        wrote_header = True
        kept_rows += len(out)
        print(f"      chunk {i}: processed={total_rows:,} kept={kept_rows:,}")

    print("[3/3] Writing gene-level edge list...")
    print(f"Wrote {kept_rows:,} edges to {args.output_path}")


if __name__ == "__main__":
    main()
