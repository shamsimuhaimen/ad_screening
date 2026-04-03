"""Drug seed definitions for Alzheimer's small-molecule retrieval."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Drug:
    drug_name: str
    chembl_id: str
    smiles: str


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

FDA_APPROVED_AD_SMALL_MOLECULES = (
    DONEPEZIL,
    RIVASTIGMINE,
    GALANTAMINE,
    MEMANTINE,
)

