#!/usr/bin/env python3
"""Characterize retrieved AD candidate drugs for quick biological triage.

Consumes outputs of retrieve_ad_candidate_drugs.py and writes compact
descriptive summaries of:
- target coverage,
- potency distribution,
- clinical phase distribution,
- top compounds/targets by support.
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
    p.add_argument(
        "--retrieval-dir",
        type=Path,
        required=True,
        help="Directory containing candidate_drugs_all.csv and ad_target_compound_evidence.csv",
    )
    p.add_argument("--top-n", type=int, default=25)
    p.add_argument("--output-dir", type=Path, default=None)
    return p.parse_args()


def clean_phase(x: pd.Series) -> pd.Series:
    s = pd.to_numeric(x, errors="coerce")
    s = s.fillna(-1).astype(int)
    return s


def potency_tier(max_pchembl: float, min_nm: float) -> str:
    if np.isfinite(max_pchembl):
        if max_pchembl >= 8:
            return "very_strong_pchembl>=8"
        if max_pchembl >= 7:
            return "strong_pchembl>=7"
        if max_pchembl >= 6:
            return "moderate_pchembl>=6"
    if np.isfinite(min_nm):
        if min_nm <= 100:
            return "strong_nm<=100"
        if min_nm <= 1000:
            return "moderate_nm<=1000"
    return "weak_or_missing"


def main() -> None:
    args = parse_args()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = args.output_dir or Path(f"results/drug_characterization_{timestamp}")
    out_dir.mkdir(parents=True, exist_ok=True)

    all_csv = args.retrieval_dir / "candidate_drugs_all.csv"
    ev_csv = args.retrieval_dir / "ad_target_compound_evidence.csv"
    if not all_csv.exists() or not ev_csv.exists():
        raise FileNotFoundError("retrieval dir must contain candidate_drugs_all.csv and ad_target_compound_evidence.csv")

    print("[1/4] Loading candidate and evidence tables...")
    cand = pd.read_csv(all_csv)
    ev = pd.read_csv(ev_csv)

    print("[2/4] Computing target/potency/phase summaries...")
    cand["max_phase_clean"] = clean_phase(cand.get("max_phase", pd.Series(index=cand.index, dtype=float)))
    cand["max_pchembl"] = pd.to_numeric(cand.get("max_pchembl"), errors="coerce")
    cand["min_standard_value_nm"] = pd.to_numeric(cand.get("min_standard_value_nm"), errors="coerce")
    cand["potency_tier"] = [
        potency_tier(p, n) for p, n in zip(cand["max_pchembl"].to_numpy(), cand["min_standard_value_nm"].to_numpy())
    ]

    phase_counts = (
        cand["max_phase_clean"]
        .value_counts(dropna=False)
        .rename_axis("max_phase")
        .reset_index(name="n_compounds")
        .sort_values("max_phase")
    )
    potency_counts = (
        cand["potency_tier"]
        .value_counts(dropna=False)
        .rename_axis("potency_tier")
        .reset_index(name="n_compounds")
    )
    target_counts = (
        ev.groupby("target_gene", as_index=False)
        .agg(
            n_evidence_rows=("Molecule ChEMBL ID", "size"),
            n_unique_compounds=("Molecule ChEMBL ID", "nunique"),
            median_pchembl=("pChEMBL Value", lambda s: float(pd.to_numeric(s, errors="coerce").median())),
        )
        .sort_values(["n_unique_compounds", "n_evidence_rows"], ascending=False)
    )

    print("[3/4] Selecting top compounds...")
    top = cand.sort_values(
        by=["n_ad_targets_hit", "max_pchembl", "n_assays", "rank_score"],
        ascending=[False, False, False, False],
    ).head(args.top_n)
    top = top[
        [
            "Molecule ChEMBL ID",
            "molecule_name",
            "n_ad_targets_hit",
            "ad_targets",
            "max_pchembl",
            "min_standard_value_nm",
            "n_assays",
            "max_phase_clean",
            "rank_score",
        ]
    ].copy()

    print("[4/4] Writing outputs...")
    phase_counts.to_csv(out_dir / "phase_distribution.csv", index=False)
    potency_counts.to_csv(out_dir / "potency_tier_distribution.csv", index=False)
    target_counts.to_csv(out_dir / "target_coverage_summary.csv", index=False)
    top.to_csv(out_dir / "top_compounds_profile.csv", index=False)

    summary = {
        "retrieval_dir": str(args.retrieval_dir),
        "n_compounds": int(len(cand)),
        "n_targets_with_evidence": int(ev["target_gene"].nunique()),
        "n_multi_target_compounds": int((cand["n_ad_targets_hit"] >= 2).sum()),
        "fraction_multi_target_compounds": float((cand["n_ad_targets_hit"] >= 2).mean()),
        "fraction_phase2plus": float((cand["max_phase_clean"] >= 2).mean()),
        "fraction_pchembl7plus": float((cand["max_pchembl"] >= 7).mean()),
        "fraction_nm100_or_better": float((cand["min_standard_value_nm"] <= 100).mean()),
        "top_n_profiled": int(min(args.top_n, len(cand))),
    }
    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))
    print(f"Wrote: {out_dir / 'summary.json'}")
    print(f"Wrote: {out_dir / 'top_compounds_profile.csv'}")


if __name__ == "__main__":
    main()
