#!/usr/bin/env python3
"""Retrieve candidate drugs for AD genes from DrugCLIP ChEMBL target tables.

This script supports a fast LOI triage step: for AD-labeled genes that appear
in DrugCLIP target folders, pull human assay compounds and rank candidates by:
- number of AD targets hit,
- potency evidence (pChEMBL / nM threshold),
- assay count support.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

TARGET_ALIAS_MAP: dict[str, set[str]] = {
    "5HT2A": {"HTR2A"},
    "CB2": {"CNR2"},
    "D2R": {"DRD2"},
    "NET": {"SLC6A2"},
    "ALDH1": {"ALDH1A1", "ALDH1A2", "ALDH1A3"},
    "GABAA_A1B2C2": {"GABRA1", "GABRB2", "GABRG2"},
    "GLUN1_2A": {"GRIN1", "GRIN2A"},
    "GLUN1_2B": {"GRIN1", "GRIN2B"},
    "MTORC1": {"MTOR"},
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--labels-csv",
        type=Path,
        required=True,
        help="labels_used.csv from train_ad_predictor.py (needs gene_symbol + y).",
    )
    p.add_argument(
        "--ad-genes-csv",
        type=Path,
        default=Path("data/processed/ad_genes.csv"),
        help="Optional AD gene list (paper-derived). If present, used as the AD target set.",
    )
    p.add_argument(
        "--targets-root",
        type=Path,
        default=Path("data/raw/drugclip_data/targets"),
        help="Directory containing target subfolders with ChEMBL/human.tsv.",
    )
    p.add_argument("--min-pchembl", type=float, default=6.0, help="Potency threshold (higher is stronger).")
    p.add_argument("--max-standard-value-nm", type=float, default=1000.0, help="Potency threshold in nM.")
    p.add_argument("--top-n", type=int, default=100)
    p.add_argument("--output-dir", type=Path, default=None)
    return p.parse_args()


def clean_float(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def is_potent(df: pd.DataFrame, min_pchembl: float, max_nm: float) -> pd.Series:
    pchembl = clean_float(df.get("pChEMBL Value", pd.Series(index=df.index, dtype=float)))
    value_nm = clean_float(df.get("Standard Value", pd.Series(index=df.index, dtype=float)))
    rel = df.get("Standard Relation", pd.Series(index=df.index, dtype=object)).astype(str)
    has_nm = value_nm.notna() & rel.str.contains("=", regex=False)
    potent_nm = has_nm & (value_nm <= max_nm)
    potent_pc = pchembl.notna() & (pchembl >= min_pchembl)
    return potent_nm | potent_pc


def load_ad_genes(labels_csv: Path) -> set[str]:
    labels = pd.read_csv(labels_csv)
    required = {"gene_symbol", "y"}
    missing = required.difference(labels.columns)
    if missing:
        raise ValueError(f"{labels_csv} missing columns: {sorted(missing)}")
    labels["gene_symbol"] = labels["gene_symbol"].astype(str).str.strip().str.upper()
    labels["y"] = labels["y"].astype(int)
    return set(labels.loc[labels["y"] == 1, "gene_symbol"].tolist())


def load_ad_genes_from_csv(ad_genes_csv: Path) -> set[str]:
    if not ad_genes_csv.exists():
        return set()
    df = pd.read_csv(ad_genes_csv)
    candidates = [c for c in df.columns if "gene" in c.lower() and "symbol" in c.lower()]
    col = candidates[0] if candidates else df.columns[0]
    s = (
        df[col]
        .astype(str)
        .str.strip()
        .str.upper()
    )
    s = s[(s != "") & (s != "NAN")]
    return set(s.tolist())


def target_folder_to_gene_candidates(target_name: str) -> set[str]:
    t = target_name.strip().upper()
    candidates = {t}
    if t in TARGET_ALIAS_MAP:
        candidates.update(TARGET_ALIAS_MAP[t])
    return candidates


def gather_rows(targets_root: Path, ad_genes: set[str], min_pchembl: float, max_nm: float) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    dirs = sorted([d for d in targets_root.iterdir() if d.is_dir()])
    print(f"Scanning {len(dirs)} DrugCLIP target folders...")
    for i, target_dir in enumerate(dirs, start=1):
        target_name = target_dir.name.strip().upper()
        gene_candidates = target_folder_to_gene_candidates(target_name)
        matched_ad = sorted(gene_candidates.intersection(ad_genes))
        if not matched_ad:
            continue

        human_tsv = target_dir / "ChEMBL" / "human.tsv"
        if not human_tsv.exists():
            continue

        df = pd.read_csv(human_tsv, sep="\t", low_memory=False)
        if df.empty:
            continue

        keep_cols = [
            "Molecule ChEMBL ID",
            "Molecule Name",
            "Smiles",
            "Molecule Max Phase",
            "pChEMBL Value",
            "Standard Value",
            "Standard Units",
            "Standard Relation",
            "Assay ChEMBL ID",
            "Target ChEMBL ID",
            "Target Name",
            "Action Type",
        ]
        present = [c for c in keep_cols if c in df.columns]
        df = df[present].copy()
        df["target_folder"] = target_name
        df["target_gene_matches"] = ";".join(matched_ad)
        df["is_potent"] = is_potent(df, min_pchembl=min_pchembl, max_nm=max_nm)
        df = df[df["is_potent"]].copy()
        if df.empty:
            continue

        # For composite folders, count support for each matched AD gene.
        for g in matched_ad:
            x = df.copy()
            x["target_gene"] = g
            rows.append(x)
        if i % 5 == 0 or i == len(dirs):
            print(f"  progress {i}/{len(dirs)} folders, potent rows kept: {sum(len(x) for x in rows):,}")

    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


def aggregate_candidates(evidence: pd.DataFrame) -> pd.DataFrame:
    evidence = evidence.copy()
    evidence["pchembl_num"] = clean_float(evidence.get("pChEMBL Value", pd.Series(index=evidence.index)))
    evidence["std_value_nm_num"] = clean_float(evidence.get("Standard Value", pd.Series(index=evidence.index)))

    grouped = evidence.groupby("Molecule ChEMBL ID", dropna=False)
    out = grouped.agg(
        n_ad_targets_hit=("target_gene", "nunique"),
        n_assays=("Assay ChEMBL ID", "nunique"),
        max_pchembl=("pchembl_num", "max"),
        mean_pchembl=("pchembl_num", "mean"),
        min_standard_value_nm=("std_value_nm_num", "min"),
        molecule_name=("Molecule Name", "first"),
        smiles=("Smiles", "first"),
        max_phase=("Molecule Max Phase", "max"),
    ).reset_index()

    target_list = grouped["target_gene"].apply(lambda x: ";".join(sorted(set(x.astype(str))))).reset_index(name="ad_targets")
    out = out.merge(target_list, on="Molecule ChEMBL ID", how="left")

    # Simple interpretable rank score: prioritize multi-target AD coverage, then potency/support.
    out["rank_score"] = (
        out["n_ad_targets_hit"].fillna(0).astype(float) * 10.0
        + out["max_pchembl"].fillna(0.0)
        + np.log1p(out["n_assays"].fillna(0).astype(float))
    )
    out = out.sort_values(
        by=["n_ad_targets_hit", "max_pchembl", "n_assays", "rank_score"],
        ascending=[False, False, False, False],
    ).reset_index(drop=True)
    return out


def main() -> None:
    args = parse_args()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = args.output_dir or Path(f"results/drug_retrieval_{timestamp}")
    out_dir.mkdir(parents=True, exist_ok=True)

    print("[1/4] Loading AD genes...")
    ad_genes_from_labels = load_ad_genes(args.labels_csv)
    ad_genes_from_paper = load_ad_genes_from_csv(args.ad_genes_csv)
    if ad_genes_from_paper:
        ad_genes = ad_genes_from_paper
        source = str(args.ad_genes_csv)
    else:
        ad_genes = ad_genes_from_labels
        source = str(args.labels_csv)
    print(f"      AD genes loaded: {len(ad_genes)} (source: {source})")

    print("[2/4] Gathering AD-target compound evidence...")
    evidence = gather_rows(
        targets_root=args.targets_root,
        ad_genes=ad_genes,
        min_pchembl=args.min_pchembl,
        max_nm=args.max_standard_value_nm,
    )
    if evidence.empty:
        raise ValueError("No potent compound evidence found for AD genes in available DrugCLIP targets.")
    print(f"      potent evidence rows: {len(evidence):,}")

    print("[3/4] Aggregating candidate drugs...")
    candidates = aggregate_candidates(evidence)
    top = candidates.head(args.top_n).copy()
    print(f"      unique compounds: {len(candidates):,}")

    print("[4/4] Writing outputs...")
    evidence.to_csv(out_dir / "ad_target_compound_evidence.csv", index=False)
    candidates.to_csv(out_dir / "candidate_drugs_all.csv", index=False)
    top.to_csv(out_dir / "candidate_drugs_top.csv", index=False)

    summary = {
        "labels_csv": str(args.labels_csv),
        "ad_genes_source": source,
        "targets_root": str(args.targets_root),
        "min_pchembl": float(args.min_pchembl),
        "max_standard_value_nm": float(args.max_standard_value_nm),
        "n_ad_genes_in_labels": int(len(ad_genes)),
        "n_evidence_rows": int(len(evidence)),
        "n_candidate_compounds": int(len(candidates)),
        "n_top_reported": int(min(args.top_n, len(candidates))),
        "n_unique_ad_targets_covered": int(evidence["target_gene"].nunique()),
    }
    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))
    print(f"Wrote: {out_dir / 'summary.json'}")
    print(f"Wrote: {out_dir / 'candidate_drugs_top.csv'}")


if __name__ == "__main__":
    main()
