"""
Heterogeneous Graph Neural Network for fraud ring detection.

Architecture:
    Three HeteroConv layers, each using SAGEConv per edge type, each followed
    by BatchNorm to stabilise training.
    Account node embeddings are passed through a two-layer MLP classifier.

    Why 3 layers?
        Notebook analysis showed ~45% of fraud ring members were "peripheral" —
        they scored like clean accounts. With 2 layers, a node only aggregates
        information from 2 hops away. Adding a 3rd layer extends the receptive
        field so peripheral ring members receive signal from hub members
        3 edges away.

    Why BatchNorm?
        Training plateaued at epoch 6 in the 2-layer version. Embedding
        magnitudes grew unevenly across node types, saturating ReLU and
        killing gradients early. BatchNorm normalises each layer's output,
        keeping gradients healthy for longer.

    Input dims:
        account  — 2 features (log_income, log_age)
        device   — 1 feature  (constant)
        merchant — 1 feature  (constant)

    Output:
        Logit per account node (binary: fraud ring member or not)
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor
from torch_geometric.data import HeteroData
from torch_geometric.nn import HeteroConv, SAGEConv


def migrate_state_dict(state_dict: dict) -> dict:
    """
    Remap legacy checkpoint keys to the current ModuleList layout.

    Old layout (hardcoded layers):  conv1.*, conv2.*, conv3.*, bn1.*, bn2.*, bn3.*
    New layout (ModuleList):        convs.0.*, convs.1.*, convs.2.*, bns.0.*, bns.1.*, bns.2.*
    """
    mapping = {
        "conv1.": "convs.0.",
        "conv2.": "convs.1.",
        "conv3.": "convs.2.",
        "bn1.": "bns.0.",
        "bn2.": "bns.1.",
        "bn3.": "bns.2.",
    }
    new_sd = {}
    for k, v in state_dict.items():
        for old, new in mapping.items():
            if k.startswith(old):
                k = new + k[len(old) :]
                break
        new_sd[k] = v
    return new_sd


def _hetero_conv(hidden_dim: int) -> HeteroConv:
    """Build one HeteroConv layer covering all 5 edge types."""
    return HeteroConv(
        {
            ("account", "transacts_with", "merchant"): SAGEConv(
                (hidden_dim, hidden_dim), hidden_dim
            ),
            ("account", "uses_device", "device"): SAGEConv((hidden_dim, hidden_dim), hidden_dim),
            ("device", "shared_by", "account"): SAGEConv((hidden_dim, hidden_dim), hidden_dim),
            ("account", "payshap_transfer", "account"): SAGEConv(
                (hidden_dim, hidden_dim), hidden_dim
            ),
            ("account", "eft_transfer", "account"): SAGEConv((hidden_dim, hidden_dim), hidden_dim),
        },
        aggr="sum",
    )


class FraudRingGNN(nn.Module):
    """
    Heterogeneous GraphSAGE with optional BatchNorm + MLP classifier.

    Args:
        hidden_dim:     Size of node embeddings after each conv layer.
        dropout:        Dropout probability on the classifier head.
        num_layers:     Number of HeteroConv layers (2 = v1 baseline, 3 = v2 default).
        use_batchnorm:  Apply BatchNorm after each conv layer (False for v1, True for v2+).
    """

    def __init__(
        self,
        hidden_dim: int = 64,
        dropout: float = 0.3,
        num_layers: int = 3,
        use_batchnorm: bool = True,
    ) -> None:
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be >= 1")
        self.num_layers = num_layers
        self.use_batchnorm = use_batchnorm

        # ── Input projections (align all node types to hidden_dim) ────────────
        self.proj = nn.ModuleDict(
            {
                "account": nn.Linear(2, hidden_dim),
                "device": nn.Linear(1, hidden_dim),
                "merchant": nn.Linear(1, hidden_dim),
            }
        )

        # ── Conv layers ────────────────────────────────────────────────────────
        self.convs = nn.ModuleList([_hetero_conv(hidden_dim) for _ in range(num_layers)])

        # ── Optional BatchNorm per node type per layer ─────────────────────────
        node_types = ["account", "device", "merchant"]
        self.bns: nn.ModuleList | None
        if use_batchnorm:
            self.bns = nn.ModuleList(
                [
                    nn.ModuleDict({n: nn.BatchNorm1d(hidden_dim) for n in node_types})
                    for _ in range(num_layers)
                ]
            )
        else:
            self.bns = None

        # ── Classifier head (account nodes only) ──────────────────────────────
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    def _apply_conv(
        self,
        conv: HeteroConv,
        bn: nn.ModuleDict | None,
        x_dict: dict[str, Tensor],
        edge_index_dict: dict,
    ) -> dict[str, Tensor]:
        """One conv + optional BatchNorm + ReLU step for all node types."""
        x_dict = conv(x_dict, edge_index_dict)
        if bn is not None:
            return {k: torch.relu(bn[k](v)) for k, v in x_dict.items()}
        return {k: torch.relu(v) for k, v in x_dict.items()}

    def forward(self, data: HeteroData) -> Tensor:
        """
        Forward pass.

        Returns:
            logits — shape [n_accounts], one raw score per account node.
        """
        # Project each node type to hidden_dim
        x_dict: dict[str, Tensor] = {ntype: self.proj[ntype](data[ntype].x) for ntype in self.proj}

        # Message passing with optional BatchNorm
        bns = self.bns if self.bns is not None else [None] * self.num_layers
        for conv, bn in zip(self.convs, bns):
            x_dict = self._apply_conv(conv, bn, x_dict, data.edge_index_dict)

        # Classify account nodes
        return self.classifier(x_dict["account"]).squeeze(-1)
