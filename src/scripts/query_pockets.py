"""Iterate over known AD drugs, load their DrugCLIP small-molecule embeddings, and print how many were loaded."""

from __future__ import annotations

from dataclasses import asdict
import json

from package.drug import FDA_APPROVED_AD_SMALL_MOLECULES, Drug


def get_fda_approved_ad_small_molecules() -> list[Drug]:
    """Return the hard-coded FDA-approved Alzheimer's small-molecule seeds."""
    return list(FDA_APPROVED_AD_SMALL_MOLECULES)


def run_drug_embedding() -> list[dict[str, str]]:
    """Return the seed table as serializable records."""
    return [asdict(seed) for seed in FDA_APPROVED_AD_SMALL_MOLECULES]


def main() -> None:
    """Print the seed table as JSON."""
    print(json.dumps(run_drug_embedding(), indent=2))


if __name__ == "__main__":
    main()
