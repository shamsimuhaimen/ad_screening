#!/usr/bin/env python3
"""Launch a named experiment script.

One-click usage:
    python src/scripts/run_experiment.py --experiment-name ad_predictor
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


DEFAULT_EXPERIMENT_NAME = "ad_predictor"
POSTGRES_ENV_FILE = Path("docker/.env.postgres")
DEFAULT_LOCAL_PREFECT_API_URL = "http://127.0.0.1:4200/api"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--experiment-name", "--experiment", dest="experiment_name", type=str, default=DEFAULT_EXPERIMENT_NAME
    )
    p.add_argument("--config", type=Path, default=None)
    p.add_argument("--results-dir", type=Path, default=Path("results"))
    p.add_argument("--bootstrap-data", action="store_true")
    p.add_argument("--local-server", action="store_true")
    p.add_argument(
        "script_args",
        nargs=argparse.REMAINDER,
        help="Additional arguments passed through to the experiment script after `--`.",
    )
    return p.parse_args()


def _parse_env_file(path: Path) -> dict[str, str]:
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


def _bootstrap_prefect_api_url() -> None:
    if os.environ.get("PREFECT_API_URL"):
        return
    if POSTGRES_ENV_FILE.exists():
        os.environ["PREFECT_API_URL"] = DEFAULT_LOCAL_PREFECT_API_URL


def resolve_experiment_paths(experiment_name: str, config_override: Path | None) -> tuple[Path, Path, Path]:
    config_path = config_override or Path("experiments") / f"{experiment_name}.yaml"
    script_path = Path("src/scripts") / f"{experiment_name}.py"
    results_path = Path("results") / experiment_name
    return config_path, script_path, results_path


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
            "Missing downloaded inputs required by the experiment workflow: " f"{names}. Re-run with --bootstrap-data."
        )
    proc = subprocess.run(
        [sys.executable, "src/scripts/download_data.py"],
        check=False,
        text=True,
    )
    if proc.returncode != 0:
        raise RuntimeError("Bootstrap download failed.")


def main() -> None:
    args = parse_args()
    _bootstrap_prefect_database_env()
    if args.local_server:
        _bootstrap_prefect_api_url()

    ensure_bootstrap_data(bootstrap_data=args.bootstrap_data)

    config_path, script_path, default_results_path = resolve_experiment_paths(args.experiment_name, args.config)
    results_path = args.results_dir / args.experiment_name if args.results_dir == Path("results") else args.results_dir

    if not config_path.exists():
        raise FileNotFoundError(f"Experiment config not found: {config_path}")
    if not script_path.exists():
        raise FileNotFoundError(f"Experiment script not found: {script_path}")

    cmd = [
        sys.executable,
        str(script_path),
        "--config",
        str(config_path),
        "--results-dir",
        str(results_path if args.results_dir != Path("results") else default_results_path),
    ]
    passthrough = args.script_args
    if passthrough and passthrough[0] == "--":
        passthrough = passthrough[1:]
    cmd.extend(passthrough)

    proc = subprocess.run(cmd, check=False)
    if proc.returncode != 0:
        raise SystemExit(proc.returncode)


if __name__ == "__main__":
    main()
