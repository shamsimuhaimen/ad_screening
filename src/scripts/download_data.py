#!/usr/bin/env python3
"""Download and extract all data used in the project."""

from __future__ import annotations

import zipfile
from pathlib import Path
from urllib.request import urlopen
from urllib.parse import urlparse, unquote
from collections import namedtuple


DOWNLOAD_DIR = Path("data/download")
RAW_DIR = Path("data/raw")

Dataset = namedtuple("Dataset", ["name","url"])
DATASETS: list[Dataset] = [
    Dataset("bulk_rna_seq_human_brain", "https://human.brain-map.org/api/v2/well_known_file_download/278447594"),
    Dataset("ontology_human_brain", "https://raw.githubusercontent.com/brain-bican/human_brain_atlas_ontology/main/hbao.owl"),
]

def download(url: str, dest_dir: Path) -> str:
    dest_dir.mkdir(parents=True, exist_ok=True)

    with urlopen(url) as response:
        response_filename = response.info().get_filename()
        url_filename = Path(unquote(urlparse(url).path)).name
        filename =response_filename or url_filename

        if Path(dest_dir/filename).is_file():
            return filename

        with open(dest_dir/filename, "wb") as f:
            chunk = response.read(1024 * 1024)
            while chunk:
                f.write(chunk)
                chunk = response.read(1024 * 1024)

    return filename


def extract(zip_path: Path, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(out_dir)


def symlink(dataset_name: str, filename: str) -> None:
    """Symlinks the downloaded filename from the raw directory."""

    target_dir = RAW_DIR / dataset_name
    target_dir.mkdir(parents=True, exist_ok=True)
    target_path = target_dir / filename
    source_path = DOWNLOAD_DIR / filename
    if target_path.exists() or target_path.is_symlink():
        target_path.unlink()
    target_path.symlink_to(source_path)


def main() -> None:
    for dataset_name, url in DATASETS:
        print(f"Downloading {dataset_name}")

        filename = download(url, DOWNLOAD_DIR)

        if filename.endswith("zip"):
            extract(DOWNLOAD_DIR / filename, RAW_DIR / dataset_name)
        else:
            symlink(dataset_name, filename)
    

if __name__ == "__main__":
    main()
