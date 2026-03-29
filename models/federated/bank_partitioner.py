"""
Split the heterogeneous fraud graph into 5 non-IID SA bank shards.

Each shard contains:
  - The bank's account nodes (filtered from the full 50k)
  - All device and merchant nodes (full retention — edges filtered automatically)
  - Stratified train/val/test masks on the shard's account nodes

Usage:
    from models.federated.bank_partitioner import partition_graph
    shards = partition_graph("data/graphs/hetero_graph.pt", "data/raw/accounts.parquet")
    capitec_shard = shards["capitec"]
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
import torch
from torch_geometric.data import HeteroData

logger = logging.getLogger(__name__)

# Smaller banks are merged into the 5 main shards
_BANK_MERGE: dict[str, str] = {
    "tymebank": "capitec",
    "discovery": "fnb",
    "african_bank": "nedbank",
}

_MAIN_BANKS = ["capitec", "fnb", "absa", "standard_bank", "nedbank"]


def _stratified_masks(
    y: torch.Tensor,
    train_ratio: float,
    val_ratio: float,
    rng: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return bool train/val/test masks for a set of account labels."""
    n = len(y)
    fraud_idx = (y == 1).nonzero(as_tuple=True)[0]
    clean_idx = (y == 0).nonzero(as_tuple=True)[0]

    def _split(idx: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        perm = idx[torch.randperm(len(idx), generator=rng)]
        n_train = int(len(perm) * train_ratio)
        n_val = int(len(perm) * val_ratio)
        return perm[:n_train], perm[n_train : n_train + n_val], perm[n_train + n_val :]

    f_tr, f_val, f_te = _split(fraud_idx)
    c_tr, c_val, c_te = _split(clean_idx)

    train_mask = torch.zeros(n, dtype=torch.bool)
    val_mask = torch.zeros(n, dtype=torch.bool)
    test_mask = torch.zeros(n, dtype=torch.bool)

    train_mask[torch.cat([f_tr, c_tr])] = True
    val_mask[torch.cat([f_val, c_val])] = True
    test_mask[torch.cat([f_te, c_te])] = True

    return train_mask, val_mask, test_mask


def partition_graph(
    graph_path: Path | str,
    accounts_path: Path | str,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    seed: int = 42,
    device: str | torch.device = "cpu",
) -> dict[str, HeteroData]:
    """
    Load the full fraud graph and split into 5 bank shards.

    Args:
        graph_path:    Path to hetero_graph.pt
        accounts_path: Path to accounts.parquet (for bank labels)
        train_ratio:   Fraction of each shard's accounts used for training
        val_ratio:     Fraction used for validation (remainder = test)
        seed:          RNG seed for reproducible splits
        device:        torch device

    Returns:
        dict mapping bank name → HeteroData shard with train/val/test masks
        on the account nodes.
    """
    graph_path = Path(graph_path)
    accounts_path = Path(accounts_path)

    data: HeteroData = torch.load(graph_path, map_location="cpu", weights_only=False)

    # accounts.parquet rows align 1-to-1 with graph account node indices
    accounts_df = pd.read_parquet(accounts_path, columns=["bank"])
    assert len(accounts_df) == data["account"].num_nodes, (
        f"accounts.parquet rows ({len(accounts_df)}) != graph account nodes "
        f"({data['account'].num_nodes})"
    )

    # Merge small banks into main 5
    bank_series = accounts_df["bank"].replace(_BANK_MERGE)

    rng = torch.Generator().manual_seed(seed)
    shards: dict[str, HeteroData] = {}

    for bank in _MAIN_BANKS:
        account_indices = torch.tensor(
            (bank_series == bank).nonzero()[0].tolist(), dtype=torch.long
        )

        shard: HeteroData = data.subgraph({"account": account_indices})
        shard = shard.to(device)

        n_accounts = shard["account"].num_nodes
        y = shard["account"].y
        n_fraud = int(y.sum().item())

        train_mask, val_mask, test_mask = _stratified_masks(y, train_ratio, val_ratio, rng)
        shard["account"].train_mask = train_mask
        shard["account"].val_mask = val_mask
        shard["account"].test_mask = test_mask

        shards[bank] = shard

        logger.info(
            f"{bank:<15} accounts={n_accounts:>6,}  fraud={n_fraud:>4,} "
            f"({100 * n_fraud / n_accounts:.2f}%)  "
            f"train={int(train_mask.sum()):>5,}  val={int(val_mask.sum()):>4,}  "
            f"test={int(test_mask.sum()):>4,}"
        )

    return shards
