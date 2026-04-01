import numpy as np
import pandas as pd
import re

# 1. Load the raw DrugCLIP data
print("Loading data...")
names = np.load("data/download/dtwg_af_names_.npy", allow_pickle=True)
embeddings = np.load("data/download/dtwg_af_embeddings.npy")

# 2. Extract UniProt IDs from pocket names (e.g., 'AF-P05067-F1')
print("Mapping pockets to proteins...")
uniprot_ids = [re.search(r"AF-([A-Z0-9]+)-F1", str(name)).group(1) for name in names]

# Create a DataFrame for easy grouping
df = pd.DataFrame({"uniprot_id": uniprot_ids, "row_idx": np.arange(len(uniprot_ids))})

# 3. Calculate Pockets per Protein
pockets_per_protein = df.groupby("uniprot_id").size()
avg_pockets = pockets_per_protein.mean()

print(f"\n--- Statistics ---")
print(f"Total Pockets: {len(names)}")
print(f"Total Unique Proteins: {len(pockets_per_protein)}")
print(f"Average Pockets per Protein: {avg_pockets:.2f}")

# 4. Calculate Internal Similarity (Coherence)
print("\nAnalyzing pocket similarity (this may take a moment)...")


def get_internal_sim(indices):
    if len(indices) < 2:
        return np.nan

    # Extract embeddings for this protein's pockets
    vecs = embeddings[indices]

    # Unit normalize vectors for fast cosine similarity via dot product
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    norm_vecs = vecs / np.maximum(norms, 1e-12)

    # Matrix of pairwise similarities
    sim_matrix = norm_vecs @ norm_vecs.T

    # Average of the off-diagonal elements (similarity to OTHER pockets)
    mask = ~np.eye(sim_matrix.shape[0], dtype=bool)
    return sim_matrix[mask].mean()


# Apply to all proteins and drop those with only 1 pocket
coherence_scores = df.groupby("uniprot_id")["row_idx"].apply(list).apply(get_internal_sim).dropna()

print(f"Mean Internal Similarity (Coherence): {coherence_scores.mean():.4f}")
print(f"Median Internal Similarity: {coherence_scores.median():.4f}")
print(f"Min Similarity (Most diverse protein): {coherence_scores.min():.4f}")
