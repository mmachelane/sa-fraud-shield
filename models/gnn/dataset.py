"""
Load the HeteroData graph and produce train/val/test masks for account nodes.

Split strategy (temporal is not available at node level, so random stratified):
    train : 70%
    val   : 15%
    test  : 15%

Usage:
    from models.gnn.dataset import load_graph

    data, split = load_graph("data/graphs/hetero_graph.pt")
    # data  — HeteroData on device
    # split — dict with keys "train", "val", "test" each a bool mask tensor
"""

from __future__ import annotations

from pathlib import Path

import torch
from torch_geometric.data import HeteroData


def load_graph(
    graph_path: Path | str,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    seed: int = 42,
    device: str | torch.device = "cpu",
) -> tuple[HeteroData, dict[str, torch.Tensor]]:
    """
    Load graph and return (data, split_masks).

    split_masks keys: "train", "val", "test"
    Each value is a boolean tensor of shape [n_accounts].
    """
    graph_path = Path(graph_path)
    if not graph_path.exists():
        raise FileNotFoundError(
            f"Graph not found: {graph_path}\nRun: python -m data_generation.graph_builder"
        )

    data: HeteroData = torch.load(graph_path, weights_only=False)
    data = data.to(device)

    n = data["account"].num_nodes
    y = data["account"].y

    # Stratified split — preserve fraud ratio in each split
    fraud_idx = (y == 1).nonzero(as_tuple=True)[0]
    clean_idx = (y == 0).nonzero(as_tuple=True)[0]

    rng = torch.Generator().manual_seed(seed)

    def _split_indices(idx: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        perm = idx[torch.randperm(len(idx), generator=rng)]
        n_train = int(len(perm) * train_ratio)
        n_val = int(len(perm) * val_ratio)
        return perm[:n_train], perm[n_train : n_train + n_val], perm[n_train + n_val :]

    fraud_train, fraud_val, fraud_test = _split_indices(fraud_idx)
    clean_train, clean_val, clean_test = _split_indices(clean_idx)

    def _mask(pos: torch.Tensor, neg: torch.Tensor) -> torch.Tensor:
        mask = torch.zeros(n, dtype=torch.bool, device=device)
        mask[torch.cat([pos, neg])] = True
        return mask

    split = {
        "train": _mask(fraud_train, clean_train),
        "val": _mask(fraud_val, clean_val),
        "test": _mask(fraud_test, clean_test),
    }

    return data, split


def summarise(data: HeteroData, split: dict[str, torch.Tensor]) -> None:
    """Print a quick summary of the loaded graph and split."""
    y = data["account"].y
    print("Graph summary")
    print("─" * 40)
    for ntype in data.node_types:
        print(f"  {ntype:<12} {data[ntype].num_nodes:>8,} nodes")
    print()
    for etype in data.edge_types:
        rel = etype[1]
        n_edges = data[etype].edge_index.shape[1]
        print(f"  {rel:<25} {n_edges:>10,} edges")
    print()
    for split_name, mask in split.items():
        n_total = int(mask.sum())
        n_fraud = int(y[mask].sum())
        print(
            f"  {split_name:<6} {n_total:>7,} accounts  "
            f"{n_fraud:>5,} fraud ({100 * n_fraud / n_total:.2f}%)"
        )


if __name__ == "__main__":
    import logging

    logging.basicConfig(level=logging.INFO)
    data, split = load_graph("data/graphs/hetero_graph.pt")
    summarise(data, split)
