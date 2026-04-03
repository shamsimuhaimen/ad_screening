"""Generate one per-drug SDF under data/processed/drug_sdf for DrugCLIP upload."""

from __future__ import annotations

from pathlib import Path
from package.drug import FDA_APPROVED_AD_DRUGS, write_drug_sdf


OUTPUT_DIR = Path("data/processed/drug_sdf")


def main() -> None:
    """Write one SDF file per FDA-approved AD drug and print each output path.

    Why it's there: DrugCLIP upload expects structure files, and one SDF per drug
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for drug in FDA_APPROVED_AD_DRUGS:
        output_path = OUTPUT_DIR / f"{drug.drug_name}.sdf"
        write_drug_sdf(output_path, drugs=[drug])
        print(output_path)


if __name__ == "__main__":
    main()
