"""Drug definitions and shared helpers for AD small-molecule DrugCLIP workflows."""

from __future__ import annotations

from dataclasses import asdict
from dataclasses import dataclass
import csv
from collections.abc import Iterator
from pathlib import Path
import pickle

import numpy as np

from rdkit import Chem
from rdkit.Chem import AllChem

ENCODED_MOL_EMBS_DIR = Path("data/raw/drugclip_data/encoded_mol_embs")
TARGETS_DIR = Path("data/raw/drugclip_data/targets")


@dataclass(frozen=True)
class Drug:
    drug_name: str
    chembl_id: str
    smiles: str

    def to_rdkit_mol(self) -> Chem.Mol:
        """Build an RDKit molecule with hydrogens, 3D coordinates, and metadata.

        Why it's there: a valid SDF record needs an actual molecule object, not
        just a SMILES string. Keeping the SMILES-to-molecule conversion on the
        model makes SDF generation reusable and ensures the same identifying
        properties are attached to every exported ligand.
        """
        mol = Chem.MolFromSmiles(self.smiles)
        if mol is None:
            raise ValueError(f"Unable to parse SMILES for {self.drug_name}: {self.smiles}")

        mol = Chem.AddHs(mol)
        embed_status = AllChem.EmbedMolecule(mol, AllChem.ETKDGv3())
        if embed_status != 0:
            raise ValueError(f"Unable to generate 3D coordinates for {self.drug_name}")

        AllChem.UFFOptimizeMolecule(mol)
        mol.SetProp("_Name", self.drug_name)
        mol.SetProp("chembl_id", self.chembl_id)
        mol.SetProp("smiles", self.smiles)
        return mol


DONEPEZIL = Drug(
    drug_name="donepezil",
    chembl_id="CHEMBL502",
    smiles="COc1cc2c(cc1OC)C(=O)C(CC1CCN(Cc3ccccc3)CC1)C2",
)
RIVASTIGMINE = Drug(
    drug_name="rivastigmine",
    chembl_id="CHEMBL636",
    smiles="CCN(C)C(=O)Oc1cccc([C@H](C)N(C)C)c1",
)
GALANTAMINE = Drug(
    drug_name="galantamine",
    chembl_id="CHEMBL659",
    smiles="COc1ccc2c3c1O[C@H]1C[C@@H](O)C=C[C@@]31CCN(C)C2",
)
MEMANTINE = Drug(
    drug_name="memantine",
    chembl_id="CHEMBL807",
    smiles="CC12CC3CC(C)(C1)CC(N)(C3)C2",
)

FDA_APPROVED_AD_DRUGS = [
    DONEPEZIL,
    RIVASTIGMINE,
    GALANTAMINE,
    MEMANTINE,
]


def collect_candidate_smiles_by_chembl(drugs: list[Drug]) -> dict[str, set[str]]:
    """Return candidate SMILES strings for each drug using local DrugCLIP ChEMBL tables.

    Why it's there: exact canonical SMILES strings often differ across sources
    because of salts, stereochemistry formatting, or alternate stored forms.
    Looking up all local SMILES observed for the same ChEMBL ID improves the
    chance of finding the corresponding encoded DrugCLIP molecule embedding.
    """
    candidate_smiles = {drug.drug_name: {drug.smiles} for drug in drugs}
    target_tables = sorted(TARGETS_DIR.glob("*/ChEMBL/*.tsv"))

    for table_path in target_tables:
        with table_path.open("r", newline="") as f:
            reader = csv.DictReader(f, delimiter="\t")
            for row in reader:
                chembl_id = (row.get("Molecule ChEMBL ID") or "").strip()
                smiles = (row.get("Smiles") or "").strip()
                if not chembl_id or not smiles:
                    continue

                for drug in drugs:
                    if chembl_id == drug.chembl_id:
                        candidate_smiles[drug.drug_name].add(smiles)

    return candidate_smiles


def iter_drugclip_embedded_drugs() -> Iterator[dict[str, object]]:
    """Yield raw encoded DrugCLIP molecule records across every stored fold.

    Why it's there: the encoded molecule archive is sharded across many fold
    pickle files. Scripts should not each reimplement the same fold traversal.
    This shared generator streams the raw embedded-molecule records without
    imposing any drug-specific matching policy.
    """

    if not ENCODED_MOL_EMBS_DIR.exists():
        raise FileNotFoundError(f"Missing encoded molecule embeddings directory: {ENCODED_MOL_EMBS_DIR}")

    for fold_path in sorted(ENCODED_MOL_EMBS_DIR.rglob("fold*.pkl")):
        with fold_path.open("rb") as f:
            payload = pickle.load(f)

        if not isinstance(payload, list) or len(payload) != 2:
            raise ValueError(f"Unexpected fold payload format in {fold_path}")

        embeddings, labels = payload
        if not isinstance(embeddings, np.ndarray):
            raise ValueError(f"Expected numpy embeddings array in {fold_path}")

        for row_idx, label in enumerate(labels):
            if not isinstance(label, str) or "," not in label:
                continue

            hit_id, smiles = label.split(",", 1)
            yield {
                "hit_id": hit_id,
                "fold_path": str(fold_path),
                "row_idx": row_idx,
                "smiles": smiles,
                "embedding_dim": int(embeddings.shape[1]),
                "embedding": embeddings[row_idx].tolist(),
            }


def write_drug_sdf(output_path: Path, drugs: list[Drug]) -> Path:
    """Write the requested drugs as a multi-record SDF file.

    Why it's there: DrugCLIP upload workflows may require a structure file
    rather than a CSV of SMILES. This function materializes the hard-coded AD
    drug seeds as 3D RDKit molecules in one SDF artifact.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = Chem.SDWriter(str(output_path))
    try:
        for drug in drugs:
            writer.write(drug.to_rdkit_mol())
    finally:
        writer.close()

    return output_path
