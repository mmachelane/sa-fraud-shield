"""
Flower NumPyClient wrapping the GNN trainer for one SA bank shard.

Each client:
  - Trains the shared FraudRingGNN on its local subgraph
  - Clips its parameter update to clip_norm before returning weights
  - Returns num_examples = training account count for FedAvg weighting
"""

from __future__ import annotations

import logging

import numpy as np
import torch
import torch.nn as nn
from flwr.client import NumPyClient
from sklearn.metrics import average_precision_score, roc_auc_score
from torch_geometric.data import HeteroData

from models.gnn.model import FraudRingGNN

logger = logging.getLogger(__name__)


def _get_parameters(model: FraudRingGNN) -> list[np.ndarray]:
    return [p.cpu().numpy() for p in model.state_dict().values()]


def _set_parameters(model: FraudRingGNN, parameters: list[np.ndarray]) -> None:
    state_dict = dict(
        zip(
            model.state_dict().keys(),
            [torch.from_numpy(p.copy()) for p in parameters],
        )
    )
    model.load_state_dict(state_dict, strict=True)


class FraudShieldClient(NumPyClient):
    """
    Federated client for one SA bank shard.

    Args:
        bank_name:     Name of the bank (for logging)
        shard:         HeteroData subgraph with train/val/test masks on account nodes
        hidden_dim:    GNN hidden dimension (must match server model)
        dropout:       Dropout probability
        local_epochs:  Number of local training epochs per federation round
        lr:            Local learning rate
        clip_norm:     L2 norm threshold for update clipping
        device:        Torch device
    """

    def __init__(
        self,
        bank_name: str,
        shard: HeteroData,
        hidden_dim: int = 64,
        dropout: float = 0.2,
        local_epochs: int = 5,
        lr: float = 1e-4,
        clip_norm: float = 1.0,
        device: torch.device = torch.device("cpu"),
    ) -> None:
        self.bank_name = bank_name
        self.shard = shard
        self.local_epochs = local_epochs
        self.lr = lr
        self.clip_norm = clip_norm
        self.device = device

        self.model = FraudRingGNN(
            hidden_dim=hidden_dim, dropout=dropout, num_layers=3, use_batchnorm=True
        ).to(device)

        # Per-shard pos_weight — each bank has a different fraud rate
        train_mask = shard["account"].train_mask
        y_train = shard["account"].y[train_mask].float()
        n_fraud = int(y_train.sum().item())
        n_clean = int((~y_train.bool()).sum().item())
        pos_weight = torch.tensor(n_clean / max(n_fraud, 1), device=device)
        self.criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        logger.debug(
            f"{bank_name}: n_train={int(train_mask.sum())}  "
            f"n_fraud={n_fraud}  pos_weight={pos_weight.item():.1f}"
        )

    # ── Flower interface ───────────────────────────────────────────────────────

    def get_parameters(self, config: dict) -> list[np.ndarray]:
        return _get_parameters(self.model)

    def fit(self, parameters: list[np.ndarray], config: dict) -> tuple[list[np.ndarray], int, dict]:
        _set_parameters(self.model, parameters)
        old_params = [p.clone() for p in self.model.parameters()]

        optimiser = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        train_mask = self.shard["account"].train_mask
        y = self.shard["account"].y.float().to(self.device)

        self.model.train()
        final_loss = 0.0
        for _ in range(self.local_epochs):
            optimiser.zero_grad()
            logits = self.model(self.shard)
            loss = self.criterion(logits[train_mask], y[train_mask])
            loss.backward()
            optimiser.step()
            final_loss = loss.item()

        # Clip the update (central DP — server adds noise after aggregation)
        with torch.no_grad():
            update_norm = torch.sqrt(
                sum(
                    (p - p_old).pow(2).sum()
                    for p, p_old in zip(self.model.parameters(), old_params)
                )
            ).item()
            if update_norm > self.clip_norm:
                scale = self.clip_norm / update_norm
                for p, p_old in zip(self.model.parameters(), old_params):
                    p.copy_(p_old + (p - p_old) * scale)

        num_examples = int(train_mask.sum().item())
        return _get_parameters(self.model), num_examples, {"train_loss": final_loss}

    def evaluate(self, parameters: list[np.ndarray], config: dict) -> tuple[float, int, dict]:
        _set_parameters(self.model, parameters)
        self.model.eval()

        val_mask = self.shard["account"].val_mask
        y = self.shard["account"].y.float().to(self.device)

        with torch.no_grad():
            logits = self.model(self.shard)
            loss = self.criterion(logits[val_mask], y[val_mask]).item()
            proba = torch.sigmoid(logits[val_mask]).cpu().numpy()

        val_y = y[val_mask].cpu().numpy()
        try:
            auc = float(roc_auc_score(val_y, proba))
            ap = float(average_precision_score(val_y, proba))
        except ValueError:
            auc, ap = 0.0, 0.0

        num_examples = int(val_mask.sum().item())
        return loss, num_examples, {"local_val_auc": auc, "local_val_ap": ap}
