#!/usr/bin/env python3
"""Quantify AD biological signal in a PPI graph for the LOI.

This script tests whether AD-labeled genes are topologically concentrated in a
protein-protein interaction (PPI) network relative to random label assignments.
It complements embedding-space diagnostics with explicit biological-network
evidence.
"""

from __future__ import annotations

import argparse
import json
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Graph:
    adjacency: dict[str, set[str]]
    edge_count: int


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--labels-csv", type=Path, required=True)
    p.add_argument(
        "--ppi-path",
        type=Path,
        required=True,
        help="Path to PPI edge list table (csv/tsv).",
    )
    p.add_argument(
        "--ppi-id-type",
        choices=["gene_symbol", "uniprot"],
        default="gene_symbol",
        help="Node ID namespace used by the PPI file.",
    )
    p.add_argument(
        "--mapping-csv",
        type=Path,
        default=Path("results/ad_predictor_20260307_210253_embedding/gene_to_uniprot_mapping.csv"),
        help="Required if --ppi-id-type uniprot. Must include gene_symbol + uniprot_accession.",
    )
    p.add_argument("--source-col", type=str, default=None)
    p.add_argument("--target-col", type=str, default=None)
    p.add_argument("--score-col", type=str, default=None)
    p.add_argument("--min-score", type=float, default=0.0)
    p.add_argument("--n-permutations", type=int, default=1000)
    p.add_argument(
        "--permute-mode",
        choices=["label_shuffle", "degree_matched"],
        default="label_shuffle",
        help="Permutation null: global label shuffle or shuffle within degree bins.",
    )
    p.add_argument(
        "--degree-bins",
        type=int,
        default=10,
        help="Number of degree quantile bins for degree-matched permutations.",
    )
    p.add_argument(
        "--permute-shortest-path",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Include mean AD shortest-path in each permutation (much slower).",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output-dir", type=Path, default=None)
    return p.parse_args()


def normalize_id(x: object) -> str:
    return str(x).strip().upper()


def load_labels(labels_csv: Path, ppi_id_type: str, mapping_csv: Path) -> pd.DataFrame:
    labels = pd.read_csv(labels_csv)
    required = {"gene_symbol", "y"}
    missing = required.difference(labels.columns)
    if missing:
        raise ValueError(f"labels csv missing columns: {sorted(missing)}")

    labels = labels.copy()
    labels["gene_symbol"] = labels["gene_symbol"].astype(str).str.strip().str.upper()
    labels["y"] = labels["y"].astype(int)

    if ppi_id_type == "gene_symbol":
        labels["node_id"] = labels["gene_symbol"]
        return labels[["gene_symbol", "y", "node_id"]].drop_duplicates("node_id")

    mapping = pd.read_csv(mapping_csv)
    required_map = {"gene_symbol", "uniprot_accession"}
    missing_map = required_map.difference(mapping.columns)
    if missing_map:
        raise ValueError(f"mapping csv missing columns: {sorted(missing_map)}")

    mapping = mapping.copy()
    mapping["gene_symbol"] = mapping["gene_symbol"].astype(str).str.strip().str.upper()
    mapping["uniprot_accession"] = mapping["uniprot_accession"].astype(str).str.strip().str.upper()
    mapping = mapping[mapping["uniprot_accession"] != ""]

    merged = labels.merge(mapping, on="gene_symbol", how="inner")
    merged["node_id"] = merged["uniprot_accession"]
    return merged[["gene_symbol", "y", "node_id"]].drop_duplicates("node_id")


def read_ppi_table(path: Path) -> pd.DataFrame:
    p = str(path).lower()
    if p.endswith(".tsv") or p.endswith(".tsv.gz") or p.endswith(".txt") or p.endswith(".txt.gz"):
        return pd.read_csv(path, sep="\t", low_memory=False)
    return pd.read_csv(path, sep=None, engine="python")


def build_graph(
    ppi_df: pd.DataFrame,
    source_col: str | None,
    target_col: str | None,
    score_col: str | None,
    min_score: float,
) -> Graph:
    if source_col is None or target_col is None:
        if len(ppi_df.columns) < 2:
            raise ValueError("PPI table must have at least 2 columns for source/target.")
        source_col = source_col or ppi_df.columns[0]
        target_col = target_col or ppi_df.columns[1]

    df = ppi_df[[source_col, target_col] + ([score_col] if score_col else [])].copy()
    df = df.rename(columns={source_col: "src", target_col: "dst"})

    if score_col is not None:
        df = df[df[score_col].astype(float) >= float(min_score)]
    elif min_score > 0:
        raise ValueError("--min-score > 0 requires --score-col.")

    df["src"] = df["src"].map(normalize_id)
    df["dst"] = df["dst"].map(normalize_id)
    df = df[(df["src"] != "") & (df["dst"] != "") & (df["src"] != df["dst"])]

    adjacency: dict[str, set[str]] = {}
    seen: set[tuple[str, str]] = set()
    for s, t in df[["src", "dst"]].itertuples(index=False):
        a, b = (s, t) if s < t else (t, s)
        if (a, b) in seen:
            continue
        seen.add((a, b))
        adjacency.setdefault(a, set()).add(b)
        adjacency.setdefault(b, set()).add(a)

    return Graph(adjacency=adjacency, edge_count=len(seen))


def count_edges_labeled(graph: Graph, label_by_node: dict[str, int]) -> tuple[int, int, int]:
    ad_ad = 0
    ad_ctrl = 0
    ctrl_ctrl = 0
    for u, neigh in graph.adjacency.items():
        if u not in label_by_node:
            continue
        yu = label_by_node[u]
        for v in neigh:
            if u >= v or v not in label_by_node:
                continue
            yv = label_by_node[v]
            if yu == 1 and yv == 1:
                ad_ad += 1
            elif yu == 0 and yv == 0:
                ctrl_ctrl += 1
            else:
                ad_ctrl += 1
    return ad_ad, ad_ctrl, ctrl_ctrl


def lcc_size(nodes: set[str], adjacency: dict[str, set[str]]) -> int:
    remaining = set(nodes)
    best = 0
    while remaining:
        start = remaining.pop()
        q = deque([start])
        comp = {start}
        while q:
            u = q.popleft()
            for v in adjacency.get(u, set()):
                if v in remaining:
                    remaining.remove(v)
                    comp.add(v)
                    q.append(v)
        best = max(best, len(comp))
    return best


def bfs_distances(source: str, allowed: set[str], adjacency: dict[str, set[str]]) -> dict[str, int]:
    dist = {source: 0}
    q = deque([source])
    while q:
        u = q.popleft()
        for v in adjacency.get(u, set()):
            if v not in allowed or v in dist:
                continue
            dist[v] = dist[u] + 1
            q.append(v)
    return dist


def mean_ad_shortest_path(ad_nodes: list[str], labeled_nodes: set[str], adjacency: dict[str, set[str]]) -> float:
    if len(ad_nodes) < 2:
        return float("nan")
    total = 0.0
    count = 0
    for i, u in enumerate(ad_nodes):
        dist = bfs_distances(u, labeled_nodes, adjacency)
        for v in ad_nodes[i + 1 :]:
            if v in dist:
                total += float(dist[v])
                count += 1
    if count == 0:
        return float("inf")
    return total / count


def mean_ad_neighbor_fraction(
    nodes: list[str],
    label_by_node: dict[str, int],
    adjacency: dict[str, set[str]],
) -> tuple[float, pd.DataFrame]:
    rows = []
    for u in nodes:
        neigh = [v for v in adjacency.get(u, set()) if v in label_by_node]
        ad_frac = float("nan")
        if neigh:
            ad_frac = float(np.mean([label_by_node[v] for v in neigh]))
        rows.append(
            {
                "node_id": u,
                "label": int(label_by_node[u]),
                "degree_labeled": len(neigh),
                "ad_neighbor_fraction": ad_frac,
            }
        )
    df = pd.DataFrame(rows)
    return float(df["ad_neighbor_fraction"].dropna().mean()), df


def summarize_metrics(
    label_by_node: dict[str, int],
    graph: Graph,
    include_shortest_path: bool,
    return_node_metrics: bool,
) -> tuple[dict[str, float], pd.DataFrame | None]:
    labeled_nodes = set(label_by_node.keys())
    ad_nodes = sorted([n for n, y in label_by_node.items() if y == 1])
    ctrl_nodes = sorted([n for n, y in label_by_node.items() if y == 0])

    ad_ad, ad_ctrl, ctrl_ctrl = count_edges_labeled(graph, label_by_node)
    labeled_edges = ad_ad + ad_ctrl + ctrl_ctrl
    n_labeled = len(labeled_nodes)
    n_ad = len(ad_nodes)

    expected_ad_ad_edge_frac = 0.0
    if n_labeled >= 2:
        expected_ad_ad_edge_frac = (n_ad * (n_ad - 1)) / max(n_labeled * (n_labeled - 1), 1)
    observed_ad_ad_edge_frac = ad_ad / labeled_edges if labeled_edges else 0.0
    ad_ad_enrichment = observed_ad_ad_edge_frac / max(expected_ad_ad_edge_frac, 1e-12)

    ad_sub_lcc = lcc_size(set(ad_nodes), graph.adjacency) if ad_nodes else 0
    mean_ad_path = (
        mean_ad_shortest_path(ad_nodes, labeled_nodes, graph.adjacency)
        if include_shortest_path
        else float("nan")
    )
    mean_ad_neigh_frac, node_metrics = mean_ad_neighbor_fraction(
        nodes=sorted(labeled_nodes),
        label_by_node=label_by_node,
        adjacency=graph.adjacency,
    )

    return {
        "n_labeled_nodes_in_ppi": float(n_labeled),
        "n_ad_nodes_in_ppi": float(n_ad),
        "n_control_nodes_in_ppi": float(len(ctrl_nodes)),
        "n_labeled_edges": float(labeled_edges),
        "ad_ad_edges": float(ad_ad),
        "ad_control_edges": float(ad_ctrl),
        "control_control_edges": float(ctrl_ctrl),
        "expected_ad_ad_edge_fraction": float(expected_ad_ad_edge_frac),
        "observed_ad_ad_edge_fraction": float(observed_ad_ad_edge_frac),
        "ad_ad_edge_enrichment": float(ad_ad_enrichment),
        "ad_lcc_size": float(ad_sub_lcc),
        "mean_ad_shortest_path": float(mean_ad_path),
        "mean_ad_neighbor_fraction": float(mean_ad_neigh_frac),
    }, (node_metrics if return_node_metrics else None)


def run_permutations(
    nodes: list[str],
    labels: np.ndarray,
    graph: Graph,
    n_perm: int,
    seed: int,
    include_shortest_path: bool,
    permute_mode: str,
    degree_bin_by_node: dict[str, int] | None,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows: list[dict[str, float]] = []
    progress_every = max(1, n_perm // 10)
    base = labels.copy()
    bins = None
    if permute_mode == "degree_matched":
        if degree_bin_by_node is None:
            raise ValueError("degree_bin_by_node is required for degree_matched mode.")
        bins = np.asarray([degree_bin_by_node[n] for n in nodes], dtype=int)
        unique_bins = np.unique(bins)
    for i in range(n_perm):
        if permute_mode == "label_shuffle":
            y_perm = rng.permutation(base)
        else:
            y_perm = base.copy()
            for b in unique_bins:
                idx = np.where(bins == b)[0]
                if len(idx) > 1:
                    y_perm[idx] = rng.permutation(y_perm[idx])
        label_by_node = dict(zip(nodes, y_perm.tolist()))
        obs, _ = summarize_metrics(
            label_by_node=label_by_node,
            graph=graph,
            include_shortest_path=include_shortest_path,
            return_node_metrics=False,
        )
        obs["perm_idx"] = i
        rows.append(obs)
        if (i + 1) % progress_every == 0 or (i + 1) == n_perm:
            print(f"      permutation progress: {i + 1}/{n_perm}")
    return pd.DataFrame(rows)


def pvalue_right(perm: np.ndarray, observed: float) -> float:
    return float((np.sum(perm >= observed) + 1) / (len(perm) + 1))


def pvalue_left(perm: np.ndarray, observed: float) -> float:
    return float((np.sum(perm <= observed) + 1) / (len(perm) + 1))


def compute_degree_bins(
    nodes: list[str],
    graph: Graph,
    degree_bins: int,
) -> dict[str, int]:
    deg = pd.Series({n: len(graph.adjacency.get(n, set())) for n in nodes})
    q = min(max(2, degree_bins), int(deg.nunique()))
    if q <= 1:
        return {n: 0 for n in nodes}
    try:
        bins = pd.qcut(deg, q=q, labels=False, duplicates="drop")
    except ValueError:
        return {n: 0 for n in nodes}
    return {n: int(bins.loc[n]) for n in nodes}


def main() -> None:
    args = parse_args()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = args.output_dir or Path(f"results/ppi_signal_{timestamp}")
    out_dir.mkdir(parents=True, exist_ok=True)

    print("[1/5] Loading labels...")
    labels = load_labels(
        labels_csv=args.labels_csv,
        ppi_id_type=args.ppi_id_type,
        mapping_csv=args.mapping_csv,
    )
    print(f"      labels loaded: {len(labels)}")

    print("[2/5] Loading and filtering PPI edges...")
    ppi_df = read_ppi_table(args.ppi_path)
    graph = build_graph(
        ppi_df=ppi_df,
        source_col=args.source_col,
        target_col=args.target_col,
        score_col=args.score_col,
        min_score=args.min_score,
    )
    print(f"      ppi edges: {graph.edge_count}, ppi nodes: {len(graph.adjacency)}")

    print("[3/5] Intersecting labels with PPI nodes...")
    labels = labels[labels["node_id"].isin(graph.adjacency)].copy()
    labels = labels.drop_duplicates("node_id")
    if len(labels) < 20:
        raise ValueError("Too few labeled nodes overlap with PPI (<20). Check ID type and mapping.")
    print(f"      overlapping labeled nodes: {len(labels)}")

    nodes = labels["node_id"].tolist()
    y = labels["y"].astype(int).to_numpy()
    label_by_node = dict(zip(nodes, y.tolist()))

    print("[4/5] Computing observed topology metrics...")
    observed, node_metrics = summarize_metrics(
        label_by_node=label_by_node,
        graph=graph,
        include_shortest_path=True,
        return_node_metrics=True,
    )

    print("[5/5] Running permutation test...")
    degree_bin_by_node = None
    if args.permute_mode == "degree_matched":
        degree_bin_by_node = compute_degree_bins(nodes, graph, args.degree_bins)
    perm_df = run_permutations(
        nodes=nodes,
        labels=y,
        graph=graph,
        n_perm=args.n_permutations,
        seed=args.seed,
        include_shortest_path=args.permute_shortest_path,
        permute_mode=args.permute_mode,
        degree_bin_by_node=degree_bin_by_node,
    )

    summary = {
        "ppi_path": str(args.ppi_path),
        "ppi_id_type": args.ppi_id_type,
        "n_permutations": int(args.n_permutations),
        "permute_mode": args.permute_mode,
        "degree_bins": int(args.degree_bins),
        "seed": int(args.seed),
        **observed,
    }

    for metric in ["ad_ad_edges", "ad_ad_edge_enrichment", "ad_lcc_size", "mean_ad_neighbor_fraction"]:
        perm = perm_df[metric].to_numpy(dtype=float)
        summary[f"{metric}_perm_mean"] = float(np.mean(perm))
        summary[f"{metric}_perm_std"] = float(np.std(perm))
        summary[f"{metric}_pvalue_right"] = pvalue_right(perm, float(summary[metric]))

    metric = "mean_ad_shortest_path"
    perm = perm_df[metric].to_numpy(dtype=float)
    perm = perm[np.isfinite(perm)]
    if np.isfinite(summary[metric]) and len(perm):
        summary[f"{metric}_perm_mean"] = float(np.mean(perm))
        summary[f"{metric}_perm_std"] = float(np.std(perm))
        summary[f"{metric}_pvalue_left"] = pvalue_left(perm, float(summary[metric]))
    else:
        summary[f"{metric}_perm_mean"] = float("nan")
        summary[f"{metric}_perm_std"] = float("nan")
        summary[f"{metric}_pvalue_left"] = float("nan")

    labels.to_csv(out_dir / "labels_in_ppi.csv", index=False)
    node_metrics.to_csv(out_dir / "node_metrics.csv", index=False)
    perm_df.to_csv(out_dir / "permutation_metrics.csv", index=False)
    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))
    print(f"Wrote: {out_dir / 'summary.json'}")
    print(f"Wrote: {out_dir / 'labels_in_ppi.csv'}")
    print(f"Wrote: {out_dir / 'node_metrics.csv'}")
    print(f"Wrote: {out_dir / 'permutation_metrics.csv'}")


if __name__ == "__main__":
    main()
