#!/usr/bin/env python3
"""Run traceable AD experiments with Prefect over a config-defined matrix.

One-click usage:
    python src/scripts/run_prefect_experiments.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd
import yaml
from prefect import flow

from characterize_embedding_space import build_gene_embeddings, train_logistic
from train_ad_predictor import build_gene_to_uniprot_map, build_label_table


DEFAULT_CONFIG = Path("experiments/prefect_experiments.yaml")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    p.add_argument("--results-dir", type=Path, default=Path("results"))
    p.add_argument("--max-runs", type=int, default=None, help="Optional cap for smoke tests.")
    p.add_argument(
        "--bootstrap-data",
        action="store_true",
        help="If core downloaded inputs are missing, run src/scripts/download_data.py before experiments.",
    )
    return p.parse_args()


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _git_commit() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            text=True,
            capture_output=True,
        )
        return out.stdout.strip()
    except Exception:
        return "unknown"


def _stable_hash(payload: dict[str, Any]) -> str:
    text = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]


def _required_download_inputs() -> list[Path]:
    return [
        Path("data/raw/bulk_rna_seq_human_brain/Genes.csv"),
        Path("data/raw/bulk_rna_seq_human_brain/SampleAnnot.csv"),
        Path("data/raw/bulk_rna_seq_human_brain/RNAseqTPM.csv"),
        Path("data/download/dtwg_af_embeddings.npy"),
        Path("data/download/dtwg_af_names_.npy"),
        Path("data/download/hgnc_complete_set.txt"),
    ]


def ensure_bootstrap_data(bootstrap_data: bool) -> None:
    missing = [p for p in _required_download_inputs() if not p.exists()]
    if not missing:
        return
    if not bootstrap_data:
        names = ", ".join(str(p) for p in missing[:4])
        if len(missing) > 4:
            names += ", ..."
        raise FileNotFoundError(
            "Missing downloaded inputs required by Prefect workflow: "
            f"{names}. Re-run with --bootstrap-data."
        )
    cmd = ["python", "src/scripts/download_data.py"]
    proc = subprocess.run(cmd, check=False, text=True, capture_output=True)
    if proc.returncode != 0:
        raise RuntimeError(
            "Bootstrap download failed. Command: python src/scripts/download_data.py\n"
            f"STDOUT:\n{proc.stdout}\n\nSTDERR:\n{proc.stderr}"
        )

    still_missing = [p for p in _required_download_inputs() if not p.exists()]
    if still_missing:
        missing_names = ", ".join(str(p) for p in still_missing)
        raise FileNotFoundError(f"Bootstrap finished but required inputs are still missing: {missing_names}")


def _stratified_split(y: np.ndarray, test_size: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    idx_all = np.arange(len(y))
    pos = idx_all[y == 1].copy()
    neg = idx_all[y == 0].copy()
    rng.shuffle(pos)
    rng.shuffle(neg)
    n_pos_test = max(1, int(round(len(pos) * test_size)))
    n_neg_test = max(1, int(round(len(neg) * test_size)))
    test_idx = np.concatenate([pos[:n_pos_test], neg[:n_neg_test]])
    train_idx = np.setdiff1d(idx_all, test_idx)
    return train_idx, test_idx


def _load_annotation_counts(path: Path | None) -> pd.Series | None:
    if path is None or not path.exists():
        return None
    ann = pd.read_csv(path)
    required = {"gene_symbol", "annotation_count"}
    if not required.issubset(ann.columns):
        return None
    ann = ann[["gene_symbol", "annotation_count"]].copy()
    ann["gene_symbol"] = ann["gene_symbol"].astype(str).str.upper().str.strip()
    ann["annotation_count"] = pd.to_numeric(ann["annotation_count"], errors="coerce")
    ann = ann.dropna(subset=["annotation_count"]).drop_duplicates("gene_symbol", keep="first")
    if ann.empty:
        return None
    return ann.set_index("gene_symbol")["annotation_count"]


def _build_baseline_labels(
    observed_labels: pd.DataFrame,
    baseline_name: str,
    seed: int,
    replicate: int,
    annotation_counts: pd.Series | None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    rng = np.random.default_rng(seed * 100_000 + replicate)
    labels = observed_labels.copy()
    n_ad = int((labels["y"] == 1).sum())

    meta = {
        "baseline": baseline_name,
        "seed": int(seed),
        "replicate": int(replicate),
        "fallback_used": False,
    }

    if baseline_name == "observed_ad":
        return labels, meta

    if baseline_name == "permute_labels":
        perm = rng.permutation(labels["y"].to_numpy(dtype=int))
        labels["y"] = perm
        labels["label"] = np.where(labels["y"] == 1, "AD", "control")
        return labels, meta

    if baseline_name == "random_matched_set":
        picked = set(rng.choice(labels["gene_symbol"].to_numpy(), size=n_ad, replace=False).tolist())
        labels["y"] = labels["gene_symbol"].isin(picked).astype(int)
        labels["label"] = np.where(labels["y"] == 1, "AD", "control")
        return labels, meta

    if baseline_name == "random_matched_set_plus_bias":
        if annotation_counts is None:
            meta["fallback_used"] = True
            picked = set(rng.choice(labels["gene_symbol"].to_numpy(), size=n_ad, replace=False).tolist())
            labels["y"] = labels["gene_symbol"].isin(picked).astype(int)
            labels["label"] = np.where(labels["y"] == 1, "AD", "control")
            return labels, meta

        work = labels[["gene_symbol", "y"]].copy()
        work = work.merge(
            annotation_counts.rename("annotation_count").reset_index(),
            on="gene_symbol",
            how="left",
        )
        work["annotation_count"] = work["annotation_count"].fillna(work["annotation_count"].median())
        q = int(min(10, max(2, work["annotation_count"].nunique())))
        work["bin"] = pd.qcut(work["annotation_count"], q=q, labels=False, duplicates="drop")

        observed_ad_by_bin = work.loc[work["y"] == 1].groupby("bin")["gene_symbol"].nunique().to_dict()
        chosen: list[str] = []
        for b, need in observed_ad_by_bin.items():
            pool = work.loc[work["bin"] == b, "gene_symbol"].to_numpy()
            if len(pool) == 0:
                continue
            take = min(int(need), len(pool))
            chosen.extend(rng.choice(pool, size=take, replace=False).tolist())

        chosen_set = set(chosen)
        if len(chosen_set) < n_ad:
            remaining = [g for g in work["gene_symbol"].tolist() if g not in chosen_set]
            fill = rng.choice(np.asarray(remaining), size=n_ad - len(chosen_set), replace=False).tolist()
            chosen_set.update(fill)

        labels["y"] = labels["gene_symbol"].isin(chosen_set).astype(int)
        labels["label"] = np.where(labels["y"] == 1, "AD", "control")
        return labels, meta

    raise ValueError(f"Unsupported baseline: {baseline_name}")


def prepare_cohort_artifacts(cohort: dict[str, Any], annotation_counts_csv: str | None, root_dir: Path) -> dict[str, str]:
    cohort_name = cohort["name"]
    out_dir = root_dir / "cohorts" / cohort_name
    out_dir.mkdir(parents=True, exist_ok=True)

    args_ns = SimpleNamespace(
        data_dir=Path("data/raw/bulk_rna_seq_human_brain"),
        ad_genes_path=Path("data/processed/ad_genes.csv"),
        hgnc_mapping_path=Path("data/download/hgnc_complete_set.txt"),
        controls_per_ad=int(cohort["controls_per_ad"]),
        min_global_expression=float(cohort["min_global_expression"]),
    )

    labels = build_label_table(args_ns)
    labels["gene_symbol"] = labels["gene_symbol"].astype(str).str.upper().str.strip()
    labels_csv = out_dir / "observed_labels.csv"
    labels.to_csv(labels_csv, index=False)

    mapping = build_gene_to_uniprot_map(labels["gene_symbol"].tolist(), args_ns.hgnc_mapping_path)
    mapping["gene_symbol"] = mapping["gene_symbol"].astype(str).str.upper().str.strip()
    mapping_csv = out_dir / "gene_to_uniprot_mapping.csv"
    mapping.to_csv(mapping_csv, index=False)

    annotation_counts = _load_annotation_counts(Path(annotation_counts_csv) if annotation_counts_csv else None)
    ann_csv = out_dir / "annotation_counts_resolved.csv"
    if annotation_counts is not None:
        annotation_counts.rename("annotation_count").reset_index().to_csv(ann_csv, index=False)
    else:
        pd.DataFrame(columns=["gene_symbol", "annotation_count"]).to_csv(ann_csv, index=False)

    return {
        "cohort_name": cohort_name,
        "labels_csv": str(labels_csv),
        "mapping_csv": str(mapping_csv),
        "annotation_counts_csv": str(ann_csv),
    }


def run_classifier_probe(
    labels_csv: str,
    mapping_csv: str,
    ablation: str,
    seed: int,
    analysis_cfg: dict[str, Any],
    out_dir: Path,
) -> dict[str, Any]:
    gene_df, x = build_gene_embeddings(
        labels_csv=Path(labels_csv),
        mapping_csv=Path(mapping_csv),
        embeddings_npy=Path("data/download/dtwg_af_embeddings.npy"),
        names_npy=Path("data/download/dtwg_af_names_.npy"),
    )
    y = gene_df["y"].to_numpy(dtype=int)
    rng = np.random.default_rng(seed)

    if ablation == "random_embedding":
        x = rng.normal(0.0, 1.0, size=x.shape)

    train_idx, test_idx = _stratified_split(y=y, test_size=float(analysis_cfg["probe_test_size"]), seed=seed)
    x_train = x[train_idx]
    y_train = y[train_idx].astype(float)
    x_test = x[test_idx]
    y_test = y[test_idx].astype(int)

    if ablation == "label_shuffle":
        y_train = rng.permutation(y_train)

    mu = x_train.mean(axis=0)
    sd = x_train.std(axis=0)
    sd[sd == 0] = 1.0
    x_train = (x_train - mu) / sd
    x_test = (x_test - mu) / sd

    m = train_logistic(
        x_train=x_train,
        y_train=y_train,
        x_test=x_test,
        y_test=y_test,
        epochs=int(analysis_cfg["probe_epochs"]),
        lr=float(analysis_cfg["probe_lr"]),
        l2=float(analysis_cfg["probe_l2"]),
    )

    metrics = {
        "n_genes": int(len(gene_df)),
        "n_ad": int((y == 1).sum()),
        "n_control": int((y == 0).sum()),
        "seed": int(seed),
        "ablation": ablation,
        "test_accuracy": float(m["test_accuracy"]),
        "test_auroc": float(m["test_auroc"]),
        "test_auprc": float(m["test_auprc"]),
        "train_accuracy": float(m["train_accuracy"]),
    }

    out_path = out_dir / "classifier_metrics.json"
    out_path.write_text(json.dumps(metrics, indent=2))
    return metrics


def run_script(cmd: list[str], cwd: str, out_log: Path) -> int:
    proc = subprocess.run(cmd, cwd=cwd, check=False, text=True, capture_output=True)
    out_log.write_text(proc.stdout + "\n\nSTDERR:\n" + proc.stderr)
    if proc.returncode != 0:
        raise RuntimeError(f"Command failed ({proc.returncode}): {' '.join(cmd)}")
    return proc.returncode


def _build_matrix(cfg: dict[str, Any]) -> list[dict[str, Any]]:
    runs: list[dict[str, Any]] = []
    for cohort in cfg["cohorts"]:
        for ablation in cfg["ablations"]:
            for baseline in cfg["baselines"]:
                baseline_name = baseline["name"]
                reps = int(baseline.get("replicates", 1))
                for rep in range(reps):
                    for seed in cfg["seeds"]:
                        if baseline_name == "observed_ad" and rep > 0:
                            continue
                        runs.append(
                            {
                                "cohort": cohort["name"],
                                "ablation": ablation,
                                "baseline": baseline_name,
                                "replicate": int(rep),
                                "seed": int(seed),
                            }
                        )
    return runs


@flow(name="ad-screening-experiments")
def run_flow(config_path: Path, results_dir: Path, max_runs: int | None = None) -> Path:
    cfg = yaml.safe_load(config_path.read_text())
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    root_dir = results_dir / f"prefect_experiments_{ts}"
    root_dir.mkdir(parents=True, exist_ok=True)

    (root_dir / "config.snapshot.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
    git_sha = _git_commit()

    cohort_artifacts: dict[str, dict[str, str]] = {}
    for cohort in cfg["cohorts"]:
        art = prepare_cohort_artifacts(
            cohort=cohort,
            annotation_counts_csv=cfg.get("annotation_counts_csv"),
            root_dir=root_dir,
        )
        cohort_artifacts[cohort["name"]] = art

    matrix = _build_matrix(cfg)
    if max_runs is not None:
        matrix = matrix[:max_runs]

    rows: list[dict[str, Any]] = []
    for entry in matrix:
        cohort_name = entry["cohort"]
        art = cohort_artifacts[cohort_name]
        observed_labels = pd.read_csv(art["labels_csv"])
        ann_counts = _load_annotation_counts(Path(art["annotation_counts_csv"]))
        labels_df, baseline_meta = _build_baseline_labels(
            observed_labels=observed_labels,
            baseline_name=entry["baseline"],
            seed=int(entry["seed"]),
            replicate=int(entry["replicate"]),
            annotation_counts=ann_counts,
        )

        run_signature = {
            "git_sha": git_sha,
            **entry,
            "analysis": cfg["analysis"],
        }
        run_id = _stable_hash(run_signature)
        run_dir = root_dir / "matrix_runs" / run_id
        run_dir.mkdir(parents=True, exist_ok=True)

        labels_path = run_dir / "labels.csv"
        labels_df.to_csv(labels_path, index=False)

        cls_metrics = run_classifier_probe(
            labels_csv=str(labels_path),
            mapping_csv=art["mapping_csv"],
            ablation=entry["ablation"],
            seed=int(entry["seed"]),
            analysis_cfg=cfg["analysis"],
            out_dir=run_dir,
        )

        ppi_dir = run_dir / "ppi_signal"
        ppi_dir.mkdir(parents=True, exist_ok=True)
        ppi_cmd = [
            "python",
            "src/scripts/analyze_ppi_signal.py",
            "--labels-csv",
            str(labels_path),
            "--ppi-path",
            str(cfg["analysis"]["ppi_path"]),
            "--source-col",
            str(cfg["analysis"]["ppi_source_col"]),
            "--target-col",
            str(cfg["analysis"]["ppi_target_col"]),
            "--score-col",
            str(cfg["analysis"]["ppi_score_col"]),
            "--min-score",
            str(cfg["analysis"]["ppi_min_score"]),
            "--n-permutations",
            str(cfg["analysis"]["n_permutations"]),
            "--permute-mode",
            str(cfg["analysis"]["permute_mode"]),
            "--degree-bins",
            str(cfg["analysis"]["degree_bins"]),
            "--seed",
            str(entry["seed"]),
            "--output-dir",
            str(ppi_dir),
        ]
        run_script(ppi_cmd, cwd=".", out_log=run_dir / "ppi_signal.log")

        hops_dir = run_dir / "cosine_hops"
        hops_dir.mkdir(parents=True, exist_ok=True)
        hops_cmd = [
            "python",
            "src/scripts/analyze_cosine_ppi_hops.py",
            "--labels-csv",
            str(labels_path),
            "--mapping-csv",
            str(art["mapping_csv"]),
            "--ppi-path",
            str(cfg["analysis"]["ppi_path"]),
            "--source-col",
            str(cfg["analysis"]["ppi_source_col"]),
            "--target-col",
            str(cfg["analysis"]["ppi_target_col"]),
            "--score-col",
            str(cfg["analysis"]["ppi_score_col"]),
            "--min-score",
            str(cfg["analysis"]["ppi_min_score"]),
            "--hops",
            str(cfg["analysis"]["hops"]),
            "--n-permutations",
            str(cfg["analysis"]["n_permutations"]),
            "--permute-mode",
            str(cfg["analysis"]["cosine_permute_mode"]),
            "--degree-bins",
            str(cfg["analysis"]["degree_bins"]),
            "--seed",
            str(entry["seed"]),
            "--output-dir",
            str(hops_dir),
        ]
        run_script(hops_cmd, cwd=".", out_log=run_dir / "cosine_hops.log")

        manifest = {
            "run_id": run_id,
            "git_sha": git_sha,
            **entry,
            "baseline_meta": baseline_meta,
            "analysis": cfg["analysis"],
            "inputs": {
                "labels_sha256": _sha256(labels_path),
                "mapping_sha256": _sha256(Path(art["mapping_csv"])),
                "ppi_sha256": _sha256(Path(cfg["analysis"]["ppi_path"])),
            },
            "outputs": {
                "classifier_metrics": str(run_dir / "classifier_metrics.json"),
                "ppi_summary": str(ppi_dir / "summary.json"),
                "hops_significance": str(hops_dir / "summary_significance.csv"),
            },
        }
        (run_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2))

        row = {
            "run_id": run_id,
            **entry,
            "fallback_used": baseline_meta["fallback_used"],
            "probe_test_accuracy": cls_metrics["test_accuracy"],
            "probe_test_auroc": cls_metrics["test_auroc"],
            "probe_test_auprc": cls_metrics["test_auprc"],
        }
        rows.append(row)

    summary_df = pd.DataFrame(rows)
    summary_df.to_csv(root_dir / "run_matrix_summary.csv", index=False)
    return root_dir


def main() -> None:
    args = parse_args()
    ensure_bootstrap_data(bootstrap_data=args.bootstrap_data)
    # Use the underlying function to avoid requiring a local Prefect API server.
    runner = getattr(run_flow, "fn", run_flow)
    out = runner(config_path=args.config, results_dir=args.results_dir, max_runs=args.max_runs)
    print(f"Wrote: {out}")
    print(f"Wrote: {out / 'run_matrix_summary.csv'}")


if __name__ == "__main__":
    main()
