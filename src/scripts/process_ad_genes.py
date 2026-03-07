#!/usr/bin/env python3
"""Build the LOI disease-gene input from the AD protein compilation paper.

The LOI uses an AD-specific disease gene set as the positive class definition.
This script extracts gene symbols from the selected Askenazi 2023 supplementary
sheet and writes `data/processed/ad_genes.csv` for downstream label creation
and training.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--input-xlsx",
        type=Path,
        default=Path("data/download/41467_2023_40208_MOESM4_ESM.xlsx"),
    )
    p.add_argument(
        "--output-csv",
        type=Path,
        default=Path("data/processed/ad_genes.csv"),
    )
    p.add_argument(
        "--sheet",
        type=str,
        default="Supplementary Data 2",
        help="Sheet to extract AD genes from (default: Supplementary Data 2, 848 proteins).",
    )
    return p.parse_args()


def normalize_gene_token(token: str) -> str | None:
    t = str(token).strip().upper()
    if not t or t == "NAN":
        return None
    return t


def main() -> None:
    args = parse_args()
    xls = pd.ExcelFile(args.input_xlsx)

    if args.sheet not in xls.sheet_names:
        raise ValueError(f"Sheet '{args.sheet}' not found. Available: {xls.sheet_names}")

    genes: set[str] = set()
    df = xls.parse(args.sheet)
    if "Gene" not in df.columns:
        raise ValueError(f"Sheet '{args.sheet}' does not contain a 'Gene' column.")
    col = df["Gene"].dropna().astype(str)
    for raw in col:
        # Some entries include aliases separated by ';', e.g. "C4B; C4B_2".
        for token in raw.replace(",", ";").split(";"):
            gene = normalize_gene_token(token)
            if gene is not None:
                genes.add(gene)

    out = pd.DataFrame({"gene_symbol": sorted(genes)})
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output_csv, index=False)

    print(f"Wrote {len(out):,} genes to {args.output_csv}")


if __name__ == "__main__":
    main()
