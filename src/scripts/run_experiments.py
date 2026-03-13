#!/usr/bin/env python3
"""Run traceable AD experiments over a config-defined matrix.

One-click usage:
    python src/scripts/run_experiments.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from urllib.request import Request, urlopen

import matplotlib
import numpy as np
import pandas as pd
import yaml

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from characterize_embedding_space import build_gene_embeddings, train_logistic
from train_ad_predictor import build_gene_to_uniprot_map, build_label_table


DEFAULT_CONFIG = Path("experiments/exp_colab_test.yaml")
POSTGRES_ENV_FILE = Path("docker/.env.postgres")
DEFAULT_LOCAL_PREFECT_API_URL = "http://127.0.0.1:4200/api"


def _parse_env_file(path: Path) -> dict[str, str]:
    """Parse a simple KEY=VALUE env file into a dictionary."""
    values: dict[str, str] = {}
    if not path.exists():
        return values
    for raw_line in path.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip("'\"")
        if key:
            values[key] = value
    return values


def _bootstrap_prefect_database_env() -> None:
    """Populate the local workflow database URL from the Postgres env file when possible."""
    if os.environ.get("PREFECT_API_DATABASE_CONNECTION_URL"):
        return
    pg_env = _parse_env_file(POSTGRES_ENV_FILE)
    if not pg_env:
        return

    db = pg_env.get("POSTGRES_DB")
    user = pg_env.get("POSTGRES_USER")
    password = pg_env.get("POSTGRES_PASSWORD")
    if not all([db, user, password]):
        return

    port = pg_env.get("POSTGRES_PORT", "5432")
    os.environ["PREFECT_API_DATABASE_CONNECTION_URL"] = f"postgresql+asyncpg://{user}:{password}@127.0.0.1:{port}/{db}"


_bootstrap_prefect_database_env()


def _bootstrap_prefect_api_url() -> None:
    """Default workflow API URL to the local server when the Postgres env file exists."""
    if os.environ.get("PREFECT_API_URL"):
        return
    if POSTGRES_ENV_FILE.exists():
        os.environ["PREFECT_API_URL"] = DEFAULT_LOCAL_PREFECT_API_URL


_bootstrap_prefect_api_url()

from prefect import flow, task


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the experiment runner."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    p.add_argument("--results-dir", type=Path, default=Path("results"))
    p.add_argument(
        "--num-workers",
        type=int,
        default=2,
        help="Set workflow thread-pool worker count via PREFECT_TASK_RUNNER_THREAD_POOL_MAX_WORKERS.",
    )
    p.add_argument(
        "--bootstrap-data",
        action="store_true",
        help="If core downloaded inputs are missing, run src/scripts/download_data.py before experiments.",
    )
    p.add_argument(
        "--local-server",
        action="store_true",
        help="Run with the local workflow server (defaults to http://127.0.0.1:4200/api).",
    )
    return p.parse_args()


def _sha256(path: Path) -> str:
    """Return the SHA-256 digest for a file."""
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _safe_run_capture(cmd: list[str]) -> tuple[int, str, str]:
    """Run a command and return its exit code, stdout, and stderr without raising."""
    try:
        proc = subprocess.run(cmd, check=False, text=True, capture_output=True)
        return proc.returncode, proc.stdout, proc.stderr
    except Exception as exc:
        return 1, "", str(exc)


def _detect_machine_type() -> dict[str, Any]:
    """Return best-effort GCP machine metadata for reproducibility."""
    try:
        req = Request(
            "http://metadata.google.internal/computeMetadata/v1/instance/machine-type",
            headers={"Metadata-Flavor": "Google"},
        )
        with urlopen(req, timeout=1.0) as resp:
            full_name = resp.read().decode("utf-8").strip()
        machine_type = full_name.rsplit("/", 1)[-1]
        return {
            "machine_type": machine_type,
            "gcp_machine_type_path": full_name,
            "source": "gcp-metadata",
        }
    except Exception:
        return {
            "machine_type": None,
            "source": "unavailable",
        }


def _file_manifest(path: Path) -> dict[str, Any]:
    """Build a reproducibility manifest record for a single file path."""
    rec: dict[str, Any] = {"path": str(path)}
    if not path.exists():
        rec["exists"] = False
        return rec
    stat = path.stat()
    rec.update(
        {
            "exists": True,
            "size_bytes": int(stat.st_size),
            "mtime_utc": datetime.utcfromtimestamp(stat.st_mtime).isoformat() + "Z",
            "sha256": _sha256(path),
        }
    )
    return rec


def write_dependency_manifest(root_dir: Path) -> Path:
    """Write environment and dependency metadata for the current run."""
    manifest_path = root_dir / "dependency_manifest.json"
    dep_files = [Path("environment.yml"), Path("pyproject.toml"), Path("requirements.txt")]

    pip_rc, pip_out, pip_err = _safe_run_capture([sys.executable, "-m", "pip", "freeze"])
    conda_rc, conda_out, conda_err = _safe_run_capture(["conda", "list", "--explicit"])

    manifest = {
        "python_executable": sys.executable,
        "python_version": sys.version,
        "platform": platform.platform(),
        "runtime": {
            "hostname": platform.node(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "cpu_count": os.cpu_count(),
            **_detect_machine_type(),
        },
        "dependency_files": [_file_manifest(p) for p in dep_files],
        "pip_freeze": {
            "returncode": pip_rc,
            "stdout": pip_out,
            "stderr": pip_err,
        },
        "conda_list_explicit": {
            "returncode": conda_rc,
            "stdout": conda_out,
            "stderr": conda_err,
        },
    }
    manifest_path.write_text(json.dumps(manifest, indent=2))
    return manifest_path


def _git_commit() -> str:
    """Return the current git commit SHA, or 'unknown' if unavailable."""
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
    """Create a short stable hash for a JSON-serializable payload."""
    text = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]


def _required_download_inputs() -> list[Path]:
    """List downloaded inputs that must exist before experiments can run."""
    return [
        Path("data/raw/bulk_rna_seq_human_brain/Genes.csv"),
        Path("data/raw/bulk_rna_seq_human_brain/SampleAnnot.csv"),
        Path("data/raw/bulk_rna_seq_human_brain/RNAseqTPM.csv"),
        Path("data/download/dtwg_af_embeddings.npy"),
        Path("data/download/dtwg_af_names_.npy"),
        Path("data/download/hgnc_complete_set.txt"),
    ]


def ensure_bootstrap_data(bootstrap_data: bool) -> None:
    """Ensure required downloaded inputs exist, optionally fetching them on demand."""
    missing = [p for p in _required_download_inputs() if not p.exists()]
    if not missing:
        return
    if not bootstrap_data:
        names = ", ".join(str(p) for p in missing[:4])
        if len(missing) > 4:
            names += ", ..."
        raise FileNotFoundError(
            "Missing downloaded inputs required by the experiment workflow: " f"{names}. Re-run with --bootstrap-data."
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
    """Create a simple stratified train/test split over binary labels."""
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
    """Load per-gene annotation counts if a valid CSV is available."""
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


def _normalize_label_schema(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize label columns to the schema expected by downstream tasks."""
    out = df.copy()
    # gene_id is metadata-only in this pipeline; use gene_symbol everywhere downstream.
    if "gene_id" in out.columns:
        out = out.drop(columns=["gene_id"])
    if "y" in out.columns:
        out["y"] = pd.to_numeric(out["y"], errors="coerce").astype("Int64")
    return out


def _build_baseline_labels(
    observed_labels: pd.DataFrame,
    baseline_name: str,
    seed: int,
    replicate: int,
    annotation_counts: pd.Series | None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Generate baseline labels and metadata for one matrix entry."""
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


def prepare_cohort_artifacts(
    cohort: dict[str, Any], annotation_counts_csv: str | None, root_dir: Path
) -> dict[str, str]:
    """Materialize cohort-specific inputs used across matrix runs."""
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

    labels = _normalize_label_schema(build_label_table(args_ns))
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
    """Train and evaluate the embedding probe for one run configuration."""
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
    elif ablation == "validation_oracle":
        # Deliberately leak the true label into the probe input to validate
        # that the metrics pipeline can reach its expected upper bound.
        x_train = np.concatenate([x_train, y_train[:, None]], axis=1)
        x_test = np.concatenate([x_test, y_test[:, None].astype(float)], axis=1)

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
    """Run a subprocess, log stdout/stderr, and raise on failure."""
    proc = subprocess.run(cmd, cwd=cwd, check=False, text=True, capture_output=True)
    out_log.write_text(proc.stdout + "\n\nSTDERR:\n" + proc.stderr)
    if proc.returncode != 0:
        raise RuntimeError(f"Command failed ({proc.returncode}): {' '.join(cmd)}")
    return proc.returncode


def _build_matrix(cfg: dict[str, Any]) -> list[dict[str, Any]]:
    """Expand the experiment config into a concrete run matrix."""
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


def _matrix_submission_batch_size() -> int:
    """Derive the parent-flow submission batch size from the configured worker cap."""
    raw = os.environ.get("PREFECT_TASK_RUNNER_THREAD_POOL_MAX_WORKERS")
    if raw is None:
        return 1
    try:
        return max(1, int(raw))
    except ValueError:
        return 1


def write_summary_graph(root_dir: Path, summary_df: pd.DataFrame) -> Path:
    """Write a summary plot and baseline rollup for completed matrix runs."""
    plot_path = root_dir / "run_matrix_summary.png"

    if summary_df.empty:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.text(0.5, 0.5, "No runs completed", ha="center", va="center")
        ax.axis("off")
        fig.tight_layout()
        fig.savefig(plot_path, dpi=180)
        plt.close(fig)
        return plot_path

    count_df = (
        summary_df.groupby("baseline", as_index=False)
        .size()
        .rename(columns={"size": "n_runs"})
        .sort_values("n_runs", ascending=False)
    )
    metric_df = (
        summary_df.groupby("baseline", as_index=False)
        .agg(
            mean_auroc=("probe_test_auroc", "mean"),
            mean_auprc=("probe_test_auprc", "mean"),
            mean_acc=("probe_test_accuracy", "mean"),
        )
        .sort_values("mean_auroc", ascending=False)
    )
    metric_df.to_csv(root_dir / "run_matrix_summary_by_baseline.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].bar(count_df["baseline"], count_df["n_runs"], color="#4C78A8")
    axes[0].set_title("Runs by Baseline")
    axes[0].set_ylabel("Run Count")
    axes[0].tick_params(axis="x", rotation=25)

    x = np.arange(len(metric_df))
    width = 0.25
    axes[1].bar(x - width, metric_df["mean_auroc"], width=width, label="AUROC", color="#F58518")
    axes[1].bar(x, metric_df["mean_auprc"], width=width, label="AUPRC", color="#54A24B")
    axes[1].bar(x + width, metric_df["mean_acc"], width=width, label="Accuracy", color="#B279A2")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(metric_df["baseline"], rotation=25)
    axes[1].set_ylim(0.0, 1.0)
    axes[1].set_title("Mean Probe Metrics by Baseline")
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(plot_path, dpi=180)
    plt.close(fig)
    return plot_path


@task(name="prepare-cohort-artifacts")
def prepare_cohort_artifacts_task(
    cohort: dict[str, Any], annotation_counts_csv: str | None, root_dir: Path
) -> dict[str, str]:
    """Task wrapper for cohort artifact preparation."""
    return prepare_cohort_artifacts(cohort=cohort, annotation_counts_csv=annotation_counts_csv, root_dir=root_dir)


@task(name="write-dependency-manifest")
def write_dependency_manifest_task(root_dir: Path) -> str:
    """Task wrapper for dependency manifest generation."""
    return str(write_dependency_manifest(root_dir=root_dir))


@task(name="ensure-bootstrap-data")
def ensure_bootstrap_data_task(bootstrap_data: bool) -> None:
    """Task wrapper for input bootstrap checks."""
    ensure_bootstrap_data(bootstrap_data=bootstrap_data)


@task(name="materialize-run-labels")
def materialize_run_labels_task(entry: dict[str, Any], art: dict[str, str], run_dir: Path) -> dict[str, Any]:
    """Materialize the labels CSV and baseline metadata for one run."""
    observed_labels = pd.read_csv(art["labels_csv"])
    ann_counts = _load_annotation_counts(Path(art["annotation_counts_csv"]))
    labels_df, baseline_meta = _build_baseline_labels(
        observed_labels=observed_labels,
        baseline_name=entry["baseline"],
        seed=int(entry["seed"]),
        replicate=int(entry["replicate"]),
        annotation_counts=ann_counts,
    )
    labels_df = _normalize_label_schema(labels_df)
    labels_path = run_dir / "labels.csv"
    labels_df.to_csv(labels_path, index=False)
    return {"labels_path": str(labels_path), "baseline_meta": baseline_meta}


@task(name="run-classifier-probe")
def run_classifier_probe_task(
    labels_artifact: dict[str, Any],
    art: dict[str, str],
    entry: dict[str, Any],
    analysis_cfg: dict[str, Any],
    run_dir: Path,
) -> dict[str, Any]:
    """Task wrapper for the classifier probe analysis."""
    return run_classifier_probe(
        labels_csv=str(labels_artifact["labels_path"]),
        mapping_csv=art["mapping_csv"],
        ablation=entry["ablation"],
        seed=int(entry["seed"]),
        analysis_cfg=analysis_cfg,
        out_dir=run_dir,
    )


@task(name="run-ppi-signal")
def run_ppi_signal_task(
    labels_artifact: dict[str, Any],
    analysis_cfg: dict[str, Any],
    entry: dict[str, Any],
    run_dir: Path,
) -> str:
    """Run the PPI enrichment analysis script for one matrix entry."""
    ppi_dir = run_dir / "ppi_signal"
    ppi_dir.mkdir(parents=True, exist_ok=True)
    ppi_cmd = [
        "python",
        "src/scripts/analyze_ppi_signal.py",
        "--labels-csv",
        str(labels_artifact["labels_path"]),
        "--ppi-path",
        str(analysis_cfg["ppi_path"]),
        "--source-col",
        str(analysis_cfg["ppi_source_col"]),
        "--target-col",
        str(analysis_cfg["ppi_target_col"]),
        "--score-col",
        str(analysis_cfg["ppi_score_col"]),
        "--min-score",
        str(analysis_cfg["ppi_min_score"]),
        "--n-permutations",
        str(analysis_cfg["n_permutations"]),
        "--permute-mode",
        str(analysis_cfg["permute_mode"]),
        "--degree-bins",
        str(analysis_cfg["degree_bins"]),
        "--seed",
        str(entry["seed"]),
        "--output-dir",
        str(ppi_dir),
    ]
    run_script(ppi_cmd, cwd=".", out_log=run_dir / "ppi_signal.log")
    return str(ppi_dir / "summary.json")


@task(name="run-cosine-hops")
def run_cosine_hops_task(
    labels_artifact: dict[str, Any],
    art: dict[str, str],
    analysis_cfg: dict[str, Any],
    entry: dict[str, Any],
    run_dir: Path,
) -> str:
    """Run the cosine-over-PPI-hops analysis script for one matrix entry."""
    hops_dir = run_dir / "cosine_hops"
    hops_dir.mkdir(parents=True, exist_ok=True)
    hops_cmd = [
        "python",
        "src/scripts/analyze_cosine_ppi_hops.py",
        "--labels-csv",
        str(labels_artifact["labels_path"]),
        "--mapping-csv",
        str(art["mapping_csv"]),
        "--ppi-path",
        str(analysis_cfg["ppi_path"]),
        "--source-col",
        str(analysis_cfg["ppi_source_col"]),
        "--target-col",
        str(analysis_cfg["ppi_target_col"]),
        "--score-col",
        str(analysis_cfg["ppi_score_col"]),
        "--min-score",
        str(analysis_cfg["ppi_min_score"]),
        "--hops",
        str(analysis_cfg["hops"]),
        "--n-permutations",
        str(analysis_cfg["n_permutations"]),
        "--permute-mode",
        str(analysis_cfg["cosine_permute_mode"]),
        "--degree-bins",
        str(analysis_cfg["degree_bins"]),
        "--seed",
        str(entry["seed"]),
        "--output-dir",
        str(hops_dir),
    ]
    run_script(hops_cmd, cwd=".", out_log=run_dir / "cosine_hops.log")
    return str(hops_dir / "summary_significance.csv")


@task(name="write-run-manifest")
def write_run_manifest_task(
    run_id: str,
    git_sha: str,
    entry: dict[str, Any],
    baseline_meta: dict[str, Any],
    analysis_cfg: dict[str, Any],
    dependency_manifest_path: str,
    labels_artifact: dict[str, Any],
    art: dict[str, str],
    run_dir: Path,
) -> str:
    """Write a reproducibility manifest for one matrix run."""
    labels_path = Path(labels_artifact["labels_path"])
    manifest = {
        "run_id": run_id,
        "git_sha": git_sha,
        **entry,
        "baseline_meta": baseline_meta,
        "analysis": analysis_cfg,
        "manifests": {
            "dependency_manifest": dependency_manifest_path,
        },
        "inputs": {
            "labels_sha256": _sha256(labels_path),
            "mapping_sha256": _sha256(Path(art["mapping_csv"])),
            "ppi_sha256": _sha256(Path(analysis_cfg["ppi_path"])),
        },
        "outputs": {
            "classifier_metrics": str(run_dir / "classifier_metrics.json"),
            "ppi_summary": str(run_dir / "ppi_signal" / "summary.json"),
            "hops_significance": str(run_dir / "cosine_hops" / "summary_significance.csv"),
        },
    }
    manifest_path = run_dir / "run_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    return str(manifest_path)


@task(name="build-summary-row")
def build_summary_row_task(
    run_id: str,
    entry: dict[str, Any],
    baseline_meta: dict[str, Any],
    cls_metrics: dict[str, Any],
    ppi_summary_path: str,
    hops_summary_path: str,
) -> dict[str, Any]:
    """Combine run outputs into one summary row for the matrix report."""
    row = {
        "run_id": run_id,
        **entry,
        "fallback_used": baseline_meta["fallback_used"],
        "probe_test_accuracy": cls_metrics["test_accuracy"],
        "probe_test_auroc": cls_metrics["test_auroc"],
        "probe_test_auprc": cls_metrics["test_auprc"],
    }
    ppi = json.loads(Path(ppi_summary_path).read_text())
    row.update(
        {
            "ppi_ad_ad_edge_enrichment": float(ppi.get("ad_ad_edge_enrichment", np.nan)),
            "ppi_mean_ad_neighbor_fraction": float(ppi.get("mean_ad_neighbor_fraction", np.nan)),
            "ppi_ad_lcc_size": float(ppi.get("ad_lcc_size", np.nan)),
            "ppi_mean_ad_shortest_path": float(ppi.get("mean_ad_shortest_path", np.nan)),
        }
    )

    hops = pd.read_csv(hops_summary_path)
    for _, r in hops.iterrows():
        k = int(r["k"])
        m = str(r["metric"])
        row[f"hops_{m}_k{k}_delta"] = float(r["delta_ad_minus_control"])
        row[f"hops_{m}_k{k}_pvalue_right"] = float(r["pvalue_right"])
    return row


@task(name="write-summary-artifacts")
def write_summary_artifacts_task(root_dir: Path, rows: list[dict[str, Any]]) -> None:
    """Write CSV and plot summaries for all completed matrix runs."""
    summary_df = pd.DataFrame(rows)
    summary_df.to_csv(root_dir / "run_matrix_summary.csv", index=False)
    write_summary_graph(root_dir=root_dir, summary_df=summary_df)


@flow(name="ad-screening-experiment-run")
def run_matrix_entry_flow(
    entry: dict[str, Any],
    art: dict[str, str],
    analysis_cfg: dict[str, Any],
    git_sha: str,
    root_dir: Path,
    dependency_manifest_path: str,
) -> dict[str, Any]:
    """Execute one matrix entry end to end and return its summary row."""
    run_signature = {
        "git_sha": git_sha,
        **entry,
        "analysis": analysis_cfg,
    }
    run_id = _stable_hash(run_signature)
    run_dir = root_dir / "matrix_runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    labels_future = materialize_run_labels_task.submit(entry=entry, art=art, run_dir=run_dir)
    cls_future = run_classifier_probe_task.submit(
        labels_artifact=labels_future,
        art=art,
        entry=entry,
        analysis_cfg=analysis_cfg,
        run_dir=run_dir,
    )
    ppi_future = run_ppi_signal_task.submit(
        labels_artifact=labels_future,
        analysis_cfg=analysis_cfg,
        entry=entry,
        run_dir=run_dir,
    )
    hops_future = run_cosine_hops_task.submit(
        labels_artifact=labels_future,
        art=art,
        analysis_cfg=analysis_cfg,
        entry=entry,
        run_dir=run_dir,
    )

    labels_artifact = labels_future.result()
    baseline_meta = labels_artifact["baseline_meta"]
    cls_metrics = cls_future.result()
    ppi_summary_path = ppi_future.result()
    hops_summary_path = hops_future.result()

    write_run_manifest_task.submit(
        run_id=run_id,
        git_sha=git_sha,
        entry=entry,
        baseline_meta=baseline_meta,
        analysis_cfg=analysis_cfg,
        dependency_manifest_path=dependency_manifest_path,
        labels_artifact=labels_artifact,
        art=art,
        run_dir=run_dir,
    ).result()

    return build_summary_row_task.submit(
        run_id=run_id,
        entry=entry,
        baseline_meta=baseline_meta,
        cls_metrics=cls_metrics,
        ppi_summary_path=ppi_summary_path,
        hops_summary_path=hops_summary_path,
    ).result()


@task(name="run-matrix-entry")
def run_matrix_entry_task(
    entry: dict[str, Any],
    art: dict[str, str],
    analysis_cfg: dict[str, Any],
    git_sha: str,
    root_dir: Path,
    dependency_manifest_path: str,
) -> dict[str, Any]:
    """Task wrapper that launches a single matrix-entry subflow."""
    return run_matrix_entry_flow(
        entry=entry,
        art=art,
        analysis_cfg=analysis_cfg,
        git_sha=git_sha,
        root_dir=root_dir,
        dependency_manifest_path=dependency_manifest_path,
    )


@flow(name="ad-screening-experiment-batch")
def run_flow(
    config_path: Path,
    results_dir: Path,
    bootstrap_data: bool = False,
) -> Path:
    """Run the full experiment matrix and write summary artifacts."""
    ensure_bootstrap_data_task.submit(bootstrap_data=bootstrap_data).result()
    cfg = yaml.safe_load(config_path.read_text())
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    root_dir = results_dir / f"experiment_runs_{ts}"
    root_dir.mkdir(parents=True, exist_ok=True)

    (root_dir / "config.snapshot.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
    git_sha = _git_commit()

    cohort_futures: dict[str, Any] = {}
    for cohort in cfg["cohorts"]:
        future = prepare_cohort_artifacts_task.submit(
            cohort=cohort,
            annotation_counts_csv=cfg.get("annotation_counts_csv"),
            root_dir=root_dir,
        )
        cohort_futures[cohort["name"]] = future

    cohort_artifacts = {name: fut.result() for name, fut in cohort_futures.items()}
    dep_manifest_path = write_dependency_manifest_task.submit(root_dir=root_dir).result()

    matrix = _build_matrix(cfg)

    rows = []
    batch_size = _matrix_submission_batch_size()
    for start in range(0, len(matrix), batch_size):
        batch_futures = []
        for entry in matrix[start : start + batch_size]:
            batch_futures.append(
                run_matrix_entry_task.submit(
                    entry=entry,
                    art=cohort_artifacts[entry["cohort"]],
                    analysis_cfg=cfg["analysis"],
                    git_sha=git_sha,
                    root_dir=root_dir,
                    dependency_manifest_path=dep_manifest_path,
                )
            )
        rows.extend(f.result() for f in batch_futures)

    write_summary_artifacts_task.submit(root_dir=root_dir, rows=rows).result()
    return root_dir


def main() -> None:
    """CLI entry point for running the experiment workflow."""
    args = parse_args()
    if args.num_workers is not None:
        if args.num_workers < 1:
            raise ValueError("--num-workers must be >= 1")
        os.environ["PREFECT_TASK_RUNNER_THREAD_POOL_MAX_WORKERS"] = str(args.num_workers)
    if args.local_server:
        os.environ.setdefault("PREFECT_API_URL", DEFAULT_LOCAL_PREFECT_API_URL)
    out = run_flow(
        config_path=args.config,
        results_dir=args.results_dir,
        bootstrap_data=args.bootstrap_data,
    )
    print(f"Wrote: {out}")
    print(f"Wrote: {out / 'run_matrix_summary.csv'}")
    print(f"Wrote: {out / 'run_matrix_summary.png'}")


if __name__ == "__main__":
    main()
