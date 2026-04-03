"""Scan DrugCLIP molecule embeddings and report matches for the FDA-approved AD drugs."""

from __future__ import annotations

from package.drug import FDA_APPROVED_AD_DRUGS, collect_candidate_smiles_by_chembl, iter_drugclip_embedded_drugs


def main() -> None:
    """Scan the raw DrugCLIP molecule archive and print FDA-AD-drug matches.

    Why it's there: this provides the no-argument script entrypoint requested
    for quick manual checks. It streams through the raw embedded-molecule
    records, matches them against the FDA-approved AD drug SMILES set, and
    reports both the matched records and the drugs that remain missing.
    """
    matched_record_count = 0
    matched_drug_names: set[str] = set()
    records_scanned = 0
    candidate_smiles = collect_candidate_smiles_by_chembl(FDA_APPROVED_AD_DRUGS)
    drug_by_name = {drug.drug_name: drug for drug in FDA_APPROVED_AD_DRUGS}
    remaining_smiles_by_drug = {drug.drug_name: set(candidate_smiles[drug.drug_name]) for drug in FDA_APPROVED_AD_DRUGS}

    for record in iter_drugclip_embedded_drugs():
        records_scanned += 1
        if records_scanned % 100000 == 0:
            print(f"Scanned {records_scanned} drug embeddings")

        matched_drug_name = None
        for drug_name, remaining_smiles in remaining_smiles_by_drug.items():
            if record["smiles"] in remaining_smiles:
                matched_drug_name = drug_name
                break

        if matched_drug_name is None:
            continue

        remaining_smiles_by_drug[matched_drug_name].remove(record["smiles"])
        drug = drug_by_name[matched_drug_name]
        matched_record_count += 1
        matched_drug_names.add(matched_drug_name)
        print(f"- {drug.drug_name}: {drug.chembl_id} " f"[{record['fold_path']} row {record['row_idx']}]")

    print("Done")
    all_drugs = list(FDA_APPROVED_AD_DRUGS)
    missing = [drug.drug_name for drug in all_drugs if drug.drug_name not in matched_drug_names]

    print(f"Loaded {matched_record_count} DrugCLIP small-molecule embeddings.")

    if missing:
        print(f"Missing embeddings for: {', '.join(missing)}")


if __name__ == "__main__":
    main()
